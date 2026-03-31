"""
Label news articles with hierarchical industry categories and sentiment.

Pure business logic — no infrastructure setup.
Logging sinks, checkpoint handlers, and stores are injected by the caller.

Pipeline:
  1. Keyword matching on title → high-confidence hits (~60-70%)
  2. Remaining ambiguous titles → batch LLM via GLM-4-Flash
  3. Results written to ClickHouse news_classified table
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import sample as random_sample
from typing import Any, Literal, TypeVar

import polars as pl
import yaml
from json_repair import loads
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from src.common import CONFIGS_DIR, get_config
from src.db.clickhouse import get_store
from src.utils.progress_manager import ProgressManager
from src.utils.s3_client import upload_json as _s3_upload_json

# Type alias: (run_id, stage, batch_idx, saved_count) -> None
CheckpointFn = Callable[[str | None, str, int, int], None]


def _labeling_cfg():
    return get_config().labeling


# ──────────────────────────────────────────────
# LLM parse-error diagnostics
# ──────────────────────────────────────────────


class InsufficientBalanceError(RuntimeError):
    """Raised when the API rejects a request due to insufficient balance (error code 1113).
    Further requests will fail identically, so the pipeline terminates immediately."""


class LLMParseError(Exception):
    """Raised when LLM response cannot be parsed; carries raw input/output for diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        titles: list[str],
        raw_content: str,
        contents: list[str | None] | None = None,
        raw_request: str = "",
    ):
        super().__init__(message)
        self.titles = titles
        self.raw_content = raw_content
        self.contents = contents
        self.raw_request = raw_request


def _dump_llm_error(
    err: LLMParseError,
    *,
    level: int,
    task_id: str,
    run_id: str | None,
    major: str | None,
    batch_idx: int,
    model: str,
) -> None:
    """Upload failed LLM input/output to SeaweedFS for post-mortem analysis."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    major_part = major.replace("/", "_") if major else "all"
    run_part = run_id if run_id else "none"
    key = f"level{level}/{task_id}/{run_part}/{major_part}_batch{batch_idx}_{ts}.json"
    if level == 2 and err.contents:
        input_items = [{"title": t, "content": c} for t, c in zip(err.titles, err.contents)]
    else:
        input_items = [{"title": t} for t in err.titles]
    payload = {
        "timestamp": ts,
        "level": level,
        "task_id": task_id,
        "major_category": major,
        "batch_idx": batch_idx,
        "model": model,
        "error": str(err),
        "input_items": input_items,
        "raw_request": err.raw_request,
        "raw_response": err.raw_content,
    }
    logger.error(
        f"LLM parse error (level={level}, major={major}, batch={batch_idx}) — dumping to S3: s3://{_labeling_cfg().s3_bucket}/{key}"
    )
    _s3_upload_json(_labeling_cfg().s3_bucket, key, payload)


# ──────────────────────────────────────────────
# Config loaders
# ──────────────────────────────────────────────


def _load_hierarchy() -> dict[str, list[str]]:
    return yaml.safe_load((CONFIGS_DIR / "hierarchy.yaml").read_text(encoding="utf-8"))


def _load_major_keywords() -> dict[str, list[str]]:
    return yaml.safe_load((CONFIGS_DIR / "major_keywords.yaml").read_text(encoding="utf-8"))


def _load_sub_keywords() -> dict[str, list[str]]:
    return yaml.safe_load((CONFIGS_DIR / "sub_keywords.yaml").read_text(encoding="utf-8"))


def _load_level1_prompt(major_categories: list[str]) -> str:
    template = (CONFIGS_DIR / "prompts" / "label_level1_system.md").read_text(encoding="utf-8").strip()
    categories_yaml = yaml.dump(major_categories, allow_unicode=True, default_flow_style=False).strip()
    return template.replace("{major_categories}", categories_yaml)


def _load_level2_prompt(major: str, subs: list[str]) -> str:
    template = (CONFIGS_DIR / "prompts" / "label_level2_system.md").read_text(encoding="utf-8").strip()
    subs_yaml = yaml.dump(subs, allow_unicode=True, default_flow_style=False).strip()
    return template.replace("{major}", major).replace("{subs}", subs_yaml)


def _load_json_fixer_prompt() -> str:
    return (CONFIGS_DIR / "prompts" / "label_json_fixer_system.md").read_text(encoding="utf-8").strip()


# ──────────────────────────────────────────────
# Hierarchy (loaded once per process)
# ──────────────────────────────────────────────


hierarchy: dict[str, list[str]] = _load_hierarchy()
sub_to_major: dict[str, str] = {sub: major for major, subs in hierarchy.items() for sub in subs}
all_major_categories = list(hierarchy.keys())

major_keywords: dict[str, list[str]] = _load_major_keywords()
sub_keywords: dict[str, list[str]] = _load_sub_keywords()

# Build set of major categories with only one sub-category
_single_sub_majors: dict[str, str] = {major: subs[0] for major, subs in hierarchy.items() if len(subs) == 1}

# ──────────────────────────────────────────────
# JSON Schema (loaded once from config files)
# ──────────────────────────────────────────────


class Level1Item(BaseModel):
    title: str = Field(description="原始新闻标题")
    major: str = Field(description="一级行业标签（用于训练新闻分类模型）")
    confidence: float = Field(ge=0.0, le=1.0, description="模型对该分类的置信度")


class Level1AnalysisResult(BaseModel):
    items: list[Level1Item] = Field(min_length=1, description="新闻分类结果数组")


class Level2Analysis(BaseModel):
    logic: str = Field(description="为何不是中性？对 EPS 或估值的影响路径是什么？")
    key_evidence: str = Field(description="提取正文/标题中的核心数据或事实点")
    expectation: Literal["超预期", "符合预期", "低于预期"] = Field(description="市场预期")


class Level2Item(BaseModel):
    title: str = Field(description="原始新闻标题")
    sub: str = Field(description="细分行业标签（必须来自 YAML 列表）")
    sentiment: Literal["利好", "中性", "利空"] = Field(description="情感极性")
    impact_score: float = Field(ge=0.0, le=1.0, description="影响程度分数，0.0-1.0，分数越高影响越大")
    analysis: Level2Analysis = Field(description="分析逻辑")
    confidence: float = Field(ge=0, le=1, description="模型对该分类的置信度")


class Level2AnalysisResult(BaseModel):
    items: list[Level2Item] = Field(min_length=1, description="金融新闻结构化标注结果列表")


# ──────────────────────────────────────────────
# Result models for structured output
# ──────────────────────────────────────────────


class Level1LabelResult(BaseModel):
    """Structured result for a single level-1 labeled news item."""

    news_id: str
    title: str
    major_category: str
    confidence: float
    label_source: str
    task_id: str


class Level2LabelResult(BaseModel):
    """Structured result for a single level-2 labeled news item."""

    news_id: str
    title: str
    datetime: str
    major_category: str
    sub_category: str
    sentiment: str
    impact_score: float
    analysis_logic: str
    key_evidence: str
    expectation: str
    confidence: float
    label_source: str
    level1_task_id: str
    level2_task_id: str
    run_id: str | None


T = TypeVar("T", bound=BaseModel)


def _smart_loads(raw: str, repair: bool = True, **kwargs) -> Any:
    """
    Unified JSON loader that passes extra arguments to the underlying function.
    """
    if not raw:
        return {}

    try:
        if repair:
            return loads(raw, **kwargs)
        else:
            return json.loads(raw)
    except Exception:
        # Fallback to prevent breaking the stream
        return {}


def _try_parse(raw: str, model_cls: type[T], repair: bool) -> T | None:
    """Full parsing chain: direct → list fallback → markdown extraction."""
    source = "repair_json" if repair else "raw"

    try:
        data = _smart_loads(raw, repair=repair, schema=model_cls)
        return model_cls.model_validate(data)
    except Exception:
        pass

    try:
        lst = _smart_loads(raw, repair=repair, schema=model_cls)
        if isinstance(lst, list) and lst:
            logger.info(f"  JSON parse: {source} list fallback succeeded")
            return model_cls.model_validate(lst[0])
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            data = _smart_loads(json_str, repair=repair, schema=model_cls)
            logger.info(f"  JSON parse: {source} markdown extraction succeeded")
            return model_cls.model_validate(data)
        except Exception:
            pass

        try:
            lst = _smart_loads(json_str, repair=repair, schema=model_cls)
            if isinstance(lst, list) and lst:
                logger.info(f"  JSON parse: {source} markdown + list fallback succeeded")
                return model_cls.model_validate(lst[0])
        except Exception:
            pass

    return None


def _extract_json_from_response(raw: str, model_cls: type[T]) -> T | None:
    """Try parsing with raw content first, then with repair_json if needed."""
    result = _try_parse(raw, model_cls, repair=False)
    if result is not None:
        return result

    logger.warning("  Initial JSON parse failed, attempting repair...")
    return _try_parse(raw, model_cls, repair=True)


def _flush_checkpoint(
    results: list[dict],
    run_id: str | None,
    stage: str,
    batch_idx: int,
    *,
    checkpoint_fn: CheckpointFn | None = None,
) -> int:
    """Flush accumulated results to store and optionally record checkpoint."""
    if not results:
        return 0
    store = get_store()
    saved = store.save_major_labels(results, run_id=run_id)
    if checkpoint_fn:
        checkpoint_fn(run_id, stage, batch_idx, saved)
    logger.info(f"Checkpoint [{stage}] batch={batch_idx} saved={saved}")
    return saved


# ──────────────────────────────────────────────
# Labeling pipeline configs
# ──────────────────────────────────────────────


class LabelingConfig(BaseModel):
    """Shared LLM parameters for both labeling levels."""

    model: str
    temperature: float
    max_tokens: int | None = None
    checkpoint_every: int
    llm_retry: int
    seed: int | None = None
    batch_size: int
    sample_size: int | None = None
    concurrency: int = 1  # number of parallel workers for LLM batches


class Level1Config(LabelingConfig):
    """Config for level-1 major-category labeling."""

    start: int = 0


class Level2Config(LabelingConfig):
    """Config for level-2 sub-category + sentiment labeling."""

    level1_task_id: str
    major_categories: list[str] | None = None  # if None, use all major categories


# ──────────────────────────────────────────────
# Keyword classifiers
# ──────────────────────────────────────────────


def _keyword_classify_major(title: str) -> tuple[str, float] | None:
    scores: dict[str, int] = {}
    for major, keywords in major_keywords.items():
        count = sum(1 for kw in keywords if kw in title)
        if count > 0:
            scores[major] = count
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = ranked[0]
    total_hits = sum(scores.values())
    confidence = best_score / total_hits if total_hits > 0 else 0

    # B: conflict detection — if top-2 scores are close, defer to LLM
    if len(ranked) >= 2:
        second_score = ranked[1][1]
        if best_score > 0 and second_score / best_score >= 0.7:
            return None  # too ambiguous, defer to LLM

    if confidence >= 0.6 and best_score >= 1:
        return best_name, min(confidence, 0.95)
    return None


def _keyword_classify_sub(title: str, major: str) -> tuple[str, float] | None:
    scores: dict[str, int] = {}
    for sub in hierarchy.get(major, []):
        count = sum(1 for kw in sub_keywords.get(sub, []) if kw in title)
        if count > 0:
            scores[sub] = count
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = ranked[0]
    total_hits = sum(scores.values())
    confidence = best_score / total_hits if total_hits > 0 else 0

    # B: conflict detection
    if len(ranked) >= 2:
        second_score = ranked[1][1]
        if best_score > 0 and second_score / best_score >= 0.7:
            return None

    if confidence >= 0.5 and best_score >= 1:
        return best_name, min(confidence, 0.90)
    return None


# ──────────────────────────────────────────────
# LLM classifiers
# ──────────────────────────────────────────────


class LLMUsage(BaseModel):
    """Accumulated token usage statistics across LLM batches."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    batches: int = 0
    total_time: float = 0.0

    @classmethod
    def from_response(cls, response: object, elapsed: float) -> LLMUsage:
        inst = cls(total_time=elapsed, batches=1)
        resp_usage = getattr(response, "usage", None)
        if resp_usage is not None:
            inst.total_tokens = getattr(resp_usage, "total_tokens", 0) or 0
            inst.prompt_tokens = getattr(resp_usage, "prompt_tokens", 0) or 0
            inst.completion_tokens = getattr(resp_usage, "completion_tokens", 0) or 0
            details = getattr(resp_usage, "prompt_tokens_details", None)
            if details is not None:
                inst.cached_tokens = getattr(details, "cached_tokens", 0) or 0
        return inst

    def add(self, other: LLMUsage) -> None:
        self.total_tokens += other.total_tokens
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.batches += other.batches
        self.total_time += other.total_time

    def print_summary(self, title: str) -> None:
        if self.batches == 0:
            return
        logger.info(
            f"{title} Token Statistics: "
            f"batches={self.batches} total_time={self.total_time:.1f}s "
            f"prompt_tokens={self.prompt_tokens:,} completion_tokens={self.completion_tokens:,} "
            f"total_tokens={self.total_tokens:,} cached_tokens={self.cached_tokens:,} "
            f"cache_rate={self.cached_tokens / max(self.prompt_tokens, 1) * 100:.1f}%"
        )


def _llm_classify_level1(
    client: OpenAI,
    titles: list[str],
    config: LabelingConfig,
    batch_idx: int = 0,
) -> tuple[list[dict], LLMUsage]:
    t0 = time.time()
    _retry = config.llm_retry
    logger.info(f"LLM level-1 request batch={batch_idx + 1}: {len(titles)} titles")

    try:
        system_prompt = _load_level1_prompt(all_major_categories)
        titles_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
        user_content = f"请分类以下新闻标题：\n{titles_str}"

        last_err: Exception | None = None
        for attempt in range(1 + _retry):
            if attempt > 0:
                logger.warning(f"  level-1 retry {attempt}/{_retry} after empty/parse failure")
                time.sleep(1.0 * attempt)
            try:
                response = client.beta.chat.completions.parse(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=config.max_tokens,
                    response_format=Level1AnalysisResult,
                    temperature=config.temperature,
                    seed=config.seed,
                    extra_body={"reasoning_split": True},
                )

                elapsed = time.time() - t0
                usage = LLMUsage.from_response(response, elapsed)
                parsed_result = response.choices[0].message.parsed
                logger.info(f"  level-1 response attempt={attempt + 1} parsed={parsed_result is not None}")

                if parsed_result is None:
                    last_err = LLMParseError(
                        "LLM returned empty response",
                        titles=titles,
                        raw_content=response.choices[0].message.content or "",
                        raw_request=user_content,
                    )
                    logger.warning(f"  level-1 empty response attempt={attempt + 1}")
                    continue
                logger.info(
                    f"  level-1 done — {elapsed:.2f}s  tokens={usage.total_tokens} cached={usage.cached_tokens}"
                )

                return [item.model_dump() for item in parsed_result.items], usage
            except LLMParseError:
                raise
            except Exception as e:
                err_str = str(e)
                if "1113" in err_str:
                    logger.error(f"  level-1 insufficient balance — terminating pipeline: {e}")
                    raise InsufficientBalanceError(err_str) from e
                last_err = e
                logger.warning(f"  level-1 API error attempt={attempt + 1}: {e}")
                continue

        # All retries exhausted
        if isinstance(last_err, LLMParseError):
            raise last_err
        raise LLMParseError(
            str(last_err),
            titles=titles,
            raw_content="",
            raw_request=user_content,
        )
    except (LLMParseError, InsufficientBalanceError):
        raise
    except Exception as e:
        logger.error(f"LLM level-1 request failed: {type(e).__name__}: {e}")
        raise


def _llm_classify_level2(
    client: OpenAI,
    titles: list[str],
    major: str,
    config: LabelingConfig,
    contents: list[str | None] | None = None,
    batch_idx: int = 0,
) -> tuple[list[dict], LLMUsage]:
    t0 = time.time()
    item_count = len(titles)
    content_count = sum(1 for c in contents if c) if contents else 0
    logger.info(
        f"LLM level-2 request batch={batch_idx + 1} [{major}]: {item_count} items ({content_count} with content)"
    )

    try:
        system_prompt = _load_level2_prompt(major, hierarchy[major])
        json_fixer_prompt = _load_json_fixer_prompt()
        # Build user message: include content (truncated) when available
        parts: list[str] = []
        for i, t in enumerate(titles):
            body = ""
            if contents and i < len(contents) and contents[i]:
                # Truncate to ~500 chars to keep token usage reasonable
                raw = str(contents[i]).strip()
                body = raw[:500] + ("..." if len(raw) > 500 else "")
            if body:
                parts.append(f"{i + 1}. 标题：{t}\n   正文：{body}")
            else:
                parts.append(f"{i + 1}. 标题：{t}")
        user_content = "请分类以下新闻并判断情绪：\n" + "\n".join(parts)

        # First attempt: original classification request
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            seed=config.seed,
            extra_body={"reasoning_split": True},
        )

        elapsed = time.time() - t0
        usage = LLMUsage.from_response(response, elapsed)
        raw_content = response.choices[0].message.content or ""
        parsed_result = _extract_json_from_response(raw_content, Level2AnalysisResult)
        logger.info(
            f"  level-2 [{major}] batch={batch_idx + 1} [情感分析] attempt=1 — "
            f"{elapsed:.2f}s  tokens={usage.total_tokens} cached={usage.cached_tokens} "
            f"parsed={parsed_result is not None}"
        )

        if parsed_result is not None:
            return [item.model_dump() for item in parsed_result.items], usage

        # Second attempt: JSON fixer (no retry)
        t0_fixer = time.time()
        logger.warning(f"  level-2 [{major}] batch={batch_idx + 1} parse failed, trying JSON fixer...")

        fixer_response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": json_fixer_prompt},
                {"role": "user", "content": raw_content},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            seed=config.seed,
            extra_body={"reasoning_split": True},
        )

        fixer_elapsed = time.time() - t0_fixer
        fixer_usage = LLMUsage.from_response(fixer_response, fixer_elapsed)
        fixer_raw = fixer_response.choices[0].message.content or ""
        parsed_result = _extract_json_from_response(fixer_raw, Level2AnalysisResult)
        logger.info(
            f"  level-2 [{major}] batch={batch_idx + 1} [JSON修复] attempt=2 — "
            f"{fixer_elapsed:.2f}s  tokens={fixer_usage.total_tokens} cached={fixer_usage.cached_tokens} "
            f"parsed={parsed_result is not None}"
        )

        if parsed_result is not None:
            total_usage = LLMUsage.from_response(response, elapsed)
            total_usage.add(fixer_usage)
            logger.info(f"  level-2 [{major}] batch={batch_idx + 1} fixer succeeded")
            return [item.model_dump() for item in parsed_result.items], total_usage

        logger.error(f"  level-2 [{major}] batch={batch_idx + 1} JSON fixer also failed to parse response")

        # Both attempts failed
        raise LLMParseError(
            "LLM returned unparseable response after fixer",
            titles=titles,
            raw_content=fixer_raw,
            contents=contents,
            raw_request=user_content,
        )
    except LLMParseError:
        raise
    except Exception as e:
        err_str = str(e)
        if "1113" in err_str:
            logger.error(f"  level-2 [{major}] insufficient balance — terminating pipeline: {e}")
            raise InsufficientBalanceError(err_str) from e
        logger.error(f"LLM level-2 [{major}] request failed: {type(e).__name__}: {e}")
        raise


def _group_by_major(items: list[dict]) -> dict[str, list[dict]]:
    """Group records by major_category."""
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["major_category"], []).append(item)
    return groups


# ──────────────────────────────────────────────
# Public pipeline functions
# ──────────────────────────────────────────────


class Level1Pipeline:
    """Level-1 major-category labeling pipeline."""

    def __init__(
        self,
        client: OpenAI,
        config: Level1Config,
        run_id: str,
        task_id: str,
        checkpoint_fn: CheckpointFn | None = None,
    ):
        self.client = client
        self.config = config
        self.run_id = run_id
        self.task_id = task_id or "unknown"
        self.checkpoint_fn = checkpoint_fn

    def run(self) -> LLMUsage:
        """Run level-1 major-category labeling pipeline."""
        sample_size = self.config.sample_size
        assert sample_size is not None and sample_size > 0, (
            f"sample_size must be a positive integer for level-1, got {sample_size!r}"
        )
        store = get_store()
        df = store.sample_news(sample_size, seed=self.config.seed, offset=self.config.start)

        keyword_results: list[dict] = []
        llm_pending: list[dict] = []

        for row in df.iter_rows(named=True):
            result = _keyword_classify_major(row["title"])
            if result:
                major, conf = result
                keyword_results.append(
                    {
                        "news_id": row["news_id"],
                        "title": row["title"],
                        "major_category": major,
                        "confidence": conf,
                        "label_source": "keyword",
                        "task_id": self.task_id,
                    }
                )
            else:
                llm_pending.append(row)

        total_saved = 0
        if keyword_results:
            total_saved += _flush_checkpoint(
                keyword_results,
                self.run_id,
                "level1-keyword",
                0,
                checkpoint_fn=self.checkpoint_fn,
            )

        # Phase 2: LLM fallback
        llm_batch_results: list[Level1LabelResult] = []
        total_batches = -(-len(llm_pending) // self.config.batch_size)
        logger.info(
            f"level-1 LLM phase: {len(llm_pending)} items -> "
            f"{total_batches} batches (batch_size={self.config.batch_size})"
        )
        llm_usage = LLMUsage()
        phase_start = time.time()
        self.manager = ProgressManager(self.run_id)
        self.manager.init_major("", total_batches=total_batches)

        try:
            max_concurrency = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            max_concurrency = os.cpu_count() or 4
        effective_concurrency = min(self.config.concurrency, max_concurrency)

        batches = [
            (batch_idx, llm_pending[i : i + self.config.batch_size])
            for batch_idx, i in enumerate(range(0, len(llm_pending), self.config.batch_size))
        ]
        pending_counter = {"total": len(batches)}
        pending_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = {
                executor.submit(
                    self._run_batch,
                    batch_items,
                    batch_idx,
                    total_batches,
                    pending_counter,
                    pending_lock,
                ): batch_idx
                for batch_idx, batch_items in batches
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    saved, usage, is_last, results = future.result()
                    llm_usage.add(usage)
                    llm_batch_results.extend(results)
                    total_saved += saved
                    if is_last:
                        self.manager.finalize(
                            "",
                            total_saved=total_saved,
                            total_tokens=llm_usage.total_tokens,
                            total_time=time.time() - phase_start,
                        )
                except Exception as e:
                    logger.error(
                        f"level-1 batch {batch_idx + 1} failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )

        # Reverse validation
        if keyword_results:
            verify_count = max(1, len(keyword_results) // 20)
            verify_sample = random_sample(keyword_results, min(verify_count, len(keyword_results)))
            mismatches = 0
            try:
                llm_results_verify, _verify_usage = _llm_classify_level1(
                    self.client, [r["title"] for r in verify_sample], self.config
                )
                llm_usage.add(_verify_usage)
                for i, r in enumerate(llm_results_verify):
                    if i < len(verify_sample) and r.get("major", "") != verify_sample[i]["major_category"]:
                        mismatches += 1
            except Exception as e:
                logger.warning(f"level-1 validation batch failed: {e}")
            accuracy = (len(verify_sample) - mismatches) / len(verify_sample) * 100
            logger.info(
                f"Level-1 reverse validation: accuracy={accuracy:.1f}% ({mismatches}/{len(verify_sample)} mismatches)"
            )

        return llm_usage

    def _run_batch(
        self,
        batch: list[dict],
        batch_idx: int,
        total_batches: int,
        pending_counter: dict[str, int],
        pending_lock: threading.Lock,
    ) -> tuple[int, LLMUsage, bool, list[Level1LabelResult]]:
        """Process a single level-1 LLM batch. Thread-safe."""
        batch_error: str | None = None
        batch_usage = LLMUsage()
        llm_batch_results: list[Level1LabelResult] = []
        saved = 0

        try:
            client = self.client
            config = self.config
            task_id = self.task_id
            titles = [r["title"] for r in batch]
            logger.info(f"level-1 batch [{batch_idx + 1}/{total_batches}] — {len(batch)} items")
            results, usage = _llm_classify_level1(client, titles, config, batch_idx=batch_idx)
            batch_usage = usage
            logger.info(
                f"  level-1 batch={batch_idx + 1}/{total_batches} — "
                f"prompt={usage.prompt_tokens:,} completion={usage.completion_tokens:,} "
                f"total={usage.total_tokens:,} time={usage.total_time:.2f}s"
            )
            for j, r in enumerate(results):
                if j < len(batch):
                    major = r.get("major", "")
                    if major in all_major_categories:
                        llm_batch_results.append(
                            Level1LabelResult(
                                news_id=batch[j]["news_id"],
                                title=batch[j]["title"],
                                major_category=major,
                                confidence=r.get("confidence", 0.5),
                                label_source="llm",
                                task_id=task_id,
                            )
                        )
        except LLMParseError as e:
            _dump_llm_error(
                e,
                level=1,
                task_id=self.task_id,
                run_id=self.run_id,
                major=None,
                batch_idx=batch_idx,
                model=self.config.model,
            )
            batch_error = f"parse_failed: {e}"
            logger.warning(f"level-1 [] batch {batch_idx + 1} parse failed (dumped to S3): {e}")
        except InsufficientBalanceError:
            raise
        except Exception as e:
            batch_error = f"error: {e}"
            logger.warning(f"level-1 [] batch {batch_idx + 1} failed: {e}")

        self.manager.update_progress(
            "",
            batch_idx=batch_idx + 1,
            saved_count=0,
            tokens=batch_usage.total_tokens,
            elapsed=batch_usage.total_time,
            error=batch_error,
        )

        is_last = False
        with pending_lock:
            pending_counter["total"] -= 1
            if pending_counter["total"] == 0:
                is_last = True

        if llm_batch_results:
            saved = _flush_checkpoint(
                [r.model_dump() for r in llm_batch_results],
                self.run_id,
                "level1-llm",
                batch_idx + 1,
                checkpoint_fn=None,
            )
        return saved, batch_usage, is_last, llm_batch_results


class Level2Pipeline:
    """Level-2 sub-category + sentiment labeling pipeline."""

    def __init__(
        self,
        client: OpenAI,
        config: Level2Config,
        run_id: str,
        task_id: str,
        checkpoint_fn: CheckpointFn | None = None,
    ):
        self.client = client
        self.config = config
        self.run_id = run_id
        self.task_id = task_id or "unknown"
        self.checkpoint_fn = checkpoint_fn

    def run(self) -> LLMUsage:
        """Run level-2 sub-category + sentiment labeling pipeline."""
        assert self.config.level1_task_id, "level1_task_id must be specified in config for level-2"

        try:
            max_concurrency = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            max_concurrency = os.cpu_count() or 4
        effective_concurrency = min(self.config.concurrency, max_concurrency)

        major_filter = ""
        if self.config.major_categories:
            cats = ", ".join(f"'{c}'" for c in self.config.major_categories)
            major_filter = f"AND c.major_category IN ({cats})"

        store = get_store()
        rows = store.execute(
            f"""
            SELECT c.news_id, c.title, c.major_category, c.task_id, r.datetime, r.content
            FROM news_classified c JOIN news_raw r ON c.news_id = r.news_id
            WHERE c.task_id = %(source_id)s
              AND c.major_category IS NOT NULL
              {major_filter}
              AND NOT EXISTS (
                  SELECT 1 FROM news_sub_classified sc
                  WHERE sc.news_id = c.news_id
                    AND sc.level1_task_id = c.task_id
                    AND sc.level2_task_id = %(task_id)s
              )
            ORDER BY c.news_id
            """,
            {"source_id": self.config.level1_task_id, "task_id": self.task_id},
        )
        df = (
            pl.DataFrame(
                rows, schema=["news_id", "title", "major_category", "task_id", "datetime", "content"], orient="row"
            )
            if rows
            else pl.DataFrame(
                schema=["news_id", "title", "major_category", "task_id", "datetime", "content"], orient="row"
            )
        )

        if self.config.sample_size is not None and self.config.sample_size < len(df):
            df = df.sample(n=self.config.sample_size, seed=self.config.seed)

        keyword_results: list[dict] = []
        keyword_sub_map: dict[str, tuple[str, float]] = {}
        llm_pending_by_major: dict[str, list[dict]] = {}

        for row in df.iter_rows(named=True):
            major = row["major_category"]
            result = _keyword_classify_sub(row["title"], major)
            if result and major not in _single_sub_majors:
                sub, conf = result
                keyword_sub_map[str(row["news_id"])] = (sub, conf)
                keyword_results.append({**row, "sub_category": sub})
            llm_pending_by_major.setdefault(major, []).append(row)

        self.keyword_sub_map = keyword_sub_map

        total_batches = sum(-(-len(p) // self.config.batch_size) for p in llm_pending_by_major.values())
        self.manager = ProgressManager(self.run_id)
        self.manager.init_overall(total_batches)

        total_saved = 0
        llm_usage = LLMUsage()

        majors_to_batches: dict[str, list[tuple[int, list[dict]]]] = {}
        for major, pending in llm_pending_by_major.items():
            batches = []
            for i in range(0, len(pending), self.config.batch_size):
                batch_items = pending[i : i + self.config.batch_size]
                batches.append((i // self.config.batch_size, batch_items))
            majors_to_batches[major] = batches

        pending_counter: dict[str, int] = {}
        pending_lock = threading.Lock()
        major_total_saved: dict[str, int] = {}
        major_total_usage: dict[str, LLMUsage] = {}

        for major, batches in majors_to_batches.items():
            pending_counter[major] = len(batches)
            major_total_saved[major] = 0
            major_total_usage[major] = LLMUsage()
            self.manager.init_major(major, total_batches=len(batches))

        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = {}
            for major, batches in majors_to_batches.items():
                for batch_idx, batch_items in batches:
                    future = executor.submit(
                        self._run_batch,
                        major,
                        batch_items,
                        batch_idx,
                        len(batches),
                        pending_counter,
                        pending_lock,
                    )
                    futures[future] = (major, batch_idx)

            all_batch_results: list[Level2LabelResult] = []
            for future in as_completed(futures):
                major, batch_idx = futures[future]
                try:
                    saved, batch_usage, is_last, batch_results = future.result()
                    batch_saved = len(batch_results)
                    total_saved += batch_saved
                    all_batch_results.extend(batch_results)
                    with pending_lock:
                        major_total_saved[major] += batch_saved
                        major_total_usage[major].add(batch_usage)
                    if is_last:
                        self.manager.finalize(
                            major,
                            total_saved=major_total_saved[major],
                            total_tokens=major_total_usage[major].total_tokens,
                            total_time=major_total_usage[major].total_time,
                        )
                        logger.info(f"Completed major [{major}]: saved={major_total_saved[major]}")
                except Exception as e:
                    logger.error(
                        f"Batch [{major}][{batch_idx + 1}] failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                    )

            # Flush all Level2 results after collection
            if all_batch_results:
                total_saved = store.save_sub_labels(
                    [r.model_dump() for r in all_batch_results],
                    run_id=self.run_id,
                )

            llm_usage = LLMUsage()
            for usage in major_total_usage.values():
                llm_usage.add(usage)

        # Reverse validation
        if keyword_results:
            verify_count = max(1, len(keyword_results) // 20)
            verify_sample = random_sample(keyword_results, min(verify_count, len(keyword_results)))
            mismatches = 0
            for major, group in _group_by_major(verify_sample).items():
                try:
                    titles = [r["title"] for r in group]
                    llm_res, _v_usage = _llm_classify_level2(self.client, titles, major, self.config)
                    llm_usage.add(_v_usage)
                    for i, r in enumerate(llm_res):
                        if i < len(group) and r.get("sub", "") != group[i]["sub_category"]:
                            mismatches += 1
                except Exception as e:
                    logger.warning(f"level-2 validation [{major}] failed: {e}")
            accuracy = (len(verify_sample) - mismatches) / len(verify_sample) * 100
            logger.info(
                f"Level-2 reverse validation: accuracy={accuracy:.1f}% ({mismatches}/{len(verify_sample)} mismatches)"
            )

        return llm_usage

    def _run_batch(
        self,
        major: str,
        batch: list[dict],
        batch_idx: int,
        total_batches: int,
        pending_counter: dict[str, int],
        pending_lock: threading.Lock,
    ) -> tuple[int, LLMUsage, bool, list[Level2LabelResult]]:
        """Process a single batch for level-2. Thread-safe."""
        batch_error: str | None = None
        batch_usage = LLMUsage()
        llm_batch_results: list[Level2LabelResult] = []

        client = self.client
        config = self.config
        current_task_id = self.task_id
        run_id = self.run_id
        keyword_sub_map = self.keyword_sub_map

        try:
            titles = [r["title"] for r in batch]
            batch_contents = [r.get("content") for r in batch]
            logger.info(
                f"Processing batch [{batch_idx + 1}/{total_batches}] for major [{major}] with {len(batch)} items"
            )
            results, usage = _llm_classify_level2(
                client, titles, major, config, contents=batch_contents, batch_idx=batch_idx
            )
            batch_usage = usage
            logger.info(
                f"  level-2 [{major}] batch={batch_idx + 1}/{total_batches} — "
                f"prompt={usage.prompt_tokens:,} completion={usage.completion_tokens:,} "
                f"total={usage.total_tokens:,} time={usage.total_time:.2f}s"
            )
            for j, r in enumerate(results):
                if j >= len(batch):
                    continue
                news_id_str = str(batch[j]["news_id"])
                if major in _single_sub_majors:
                    sub, conf = _single_sub_majors[major], r.get("confidence", 0.5)
                    label_source = "auto+llm"
                elif news_id_str in keyword_sub_map:
                    sub, conf = keyword_sub_map[news_id_str]
                    label_source = "keyword+llm"
                else:
                    sub, conf = r.get("sub", ""), r.get("confidence", 0.5)
                    label_source = "llm"
                if sub in sub_to_major:
                    llm_batch_results.append(
                        Level2LabelResult(
                            news_id=batch[j]["news_id"],
                            title=batch[j]["title"],
                            datetime=batch[j]["datetime"],
                            major_category=major,
                            sub_category=sub,
                            sentiment=r.get("sentiment", "中性"),
                            impact_score=r.get("impact_score", 0.5),
                            analysis_logic=r.get("analysis", {}).get("logic", ""),
                            key_evidence=r.get("analysis", {}).get("key_evidence", ""),
                            expectation=r.get("analysis", {}).get("expectation", "符合预期"),
                            confidence=conf,
                            label_source=label_source,
                            level1_task_id=batch[j]["task_id"],
                            level2_task_id=current_task_id,
                            run_id=run_id,
                        )
                    )
        except LLMParseError as e:
            _dump_llm_error(
                e, level=2, task_id=current_task_id, run_id=run_id, major=major, batch_idx=batch_idx, model=config.model
            )
            batch_error = f"parse_failed: {e}"
            logger.warning(f"level-2 [{major}] batch {batch_idx + 1} parse failed (dumped to S3): {e}")
        except InsufficientBalanceError:
            raise
        except Exception as e:
            batch_error = f"error: {e}"
            logger.warning(f"level-2 [{major}] batch {batch_idx + 1} failed: {e}")

        self.manager.update_progress(
            major,
            batch_idx=batch_idx + 1,
            saved_count=0,
            tokens=batch_usage.total_tokens,
            elapsed=batch_usage.total_time,
            error=batch_error,
        )

        is_last = False
        with pending_lock:
            pending_counter[major] -= 1
            if pending_counter[major] == 0:
                is_last = True

        # Level2 results are saved by run() after collecting all batches
        return 0, batch_usage, is_last, llm_batch_results
