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

import re
import time
import traceback
from collections.abc import Callable
from random import sample as random_sample
from typing import TYPE_CHECKING, Literal, TypeVar

import polars as pl
import yaml
from json_repair import repair_json
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.common import CONFIGS_DIR, console, get_config
from src.utils.s3_client import upload_json as _s3_upload_json

if TYPE_CHECKING:
    from src.db.store import ClickHouseStore

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


T = TypeVar("T", bound=BaseModel)


def _extract_json_from_response(raw: str, model_cls: type[T]) -> T | None:
    """Multi-layer fallback JSON parsing:
    1. Direct parse
    2. List fallback (take first element)
    3. Markdown code block extraction, then retry
    """
    # 1. Direct parse
    try:
        data = repair_json(raw)
        return model_cls.model_validate(data)
    except Exception:
        pass

    # 2. List fallback
    try:
        lst = repair_json(raw)
        if isinstance(lst, list) and lst:
            logger.info("  JSON parse: list fallback succeeded")
            return model_cls.model_validate(lst[0])
    except Exception:
        pass

    # 3. Markdown code block extraction
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            data = repair_json(json_str)
            logger.info("  JSON parse: markdown extraction succeeded")
            return model_cls.model_validate(data)
        except Exception:
            pass

        # List fallback (if markdown contains a list)
        try:
            lst = repair_json(json_str)
            if isinstance(lst, list) and lst:
                logger.info("  JSON parse: markdown + list fallback succeeded")
                return model_cls.model_validate(lst[0])
        except Exception:
            pass

    return None


def _flush_checkpoint(
    results: list[dict],
    run_id: str | None,
    stage: str,
    batch_idx: int,
    *,
    store: ClickHouseStore,
    checkpoint_fn: CheckpointFn | None = None,
) -> int:
    """Flush accumulated results to store and optionally record checkpoint."""
    if not results:
        return 0
    saved = store.save_labels(results, run_id=run_id)
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


class Level1Config(LabelingConfig):
    """Config for level-1 major-category labeling."""

    start: int = 0


class Level2Config(LabelingConfig):
    """Config for level-2 sub-category + sentiment labeling."""

    level1_task_id: str
    major_categories: list[str] | None = None  # if None, use all major categories
    concurrency: int = 1  # number of parallel workers for major categories


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
        console.print(
            f"\n[bold]📊 {title} Token Statistics[/bold]\n"
            f"  Batches: [cyan]{self.batches}[/cyan]  "
            f"Total time: [cyan]{self.total_time:.1f}s[/cyan]  "
            f"Avg: [cyan]{self.total_time / self.batches:.2f}s[/cyan]/batch\n"
            f"  Prompt tokens:     Total=[blue]{self.prompt_tokens:,}[/blue]  "
            f"Avg=[blue]{self.prompt_tokens / self.batches:,.0f}[/blue]/batch\n"
            f"  Completion tokens: Total=[blue]{self.completion_tokens:,}[/blue]  "
            f"Avg=[blue]{self.completion_tokens / self.batches:,.0f}[/blue]/batch\n"
            f"  Total tokens:      Total=[bold]{self.total_tokens:,}[/bold]  "
            f"Avg=[bold]{self.total_tokens / self.batches:,.0f}[/bold]/batch\n"
            f"  Cached tokens: [green]{self.cached_tokens:,}[/green]  "
            f"Cache rate: [green]"
            f"{self.cached_tokens / max(self.prompt_tokens, 1) * 100:.1f}%[/green]"
        )


def _llm_classify_level1(
    client: OpenAI,
    titles: list[str],
    config: LabelingConfig,
    batch_idx: int = 0,
) -> tuple[list[dict], LLMUsage]:
    t0 = time.time()
    _retry = config.llm_retry
    logger.info(f"LLM level-1 request batch={batch_idx}: {len(titles)} titles")

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
    logger.info(f"LLM level-2 request batch={batch_idx} [{major}]: {item_count} items ({content_count} with content)")

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
            f"  level-2 [{major}] batch={batch_idx} [情感分析] attempt=1 — "
            f"{elapsed:.2f}s  tokens={usage.total_tokens} cached={usage.cached_tokens} "
            f"parsed={parsed_result is not None}"
        )

        if parsed_result is not None:
            return [item.model_dump() for item in parsed_result.items], usage

        # Second attempt: JSON fixer (no retry)
        t0_fixer = time.time()
        logger.warning(f"  level-2 [{major}] batch={batch_idx} parse failed, trying JSON fixer...")

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
            f"  level-2 [{major}] batch={batch_idx} [JSON修复] attempt=2 — "
            f"{fixer_elapsed:.2f}s  tokens={fixer_usage.total_tokens} cached={fixer_usage.cached_tokens} "
            f"parsed={parsed_result is not None}"
        )

        if parsed_result is not None:
            total_usage = LLMUsage.from_response(response, elapsed)
            total_usage.add(fixer_usage)
            logger.info(f"  level-2 [{major}] batch={batch_idx} fixer succeeded")
            return [item.model_dump() for item in parsed_result.items], total_usage

        logger.error(f"  level-2 [{major}] batch={batch_idx} JSON fixer also failed to parse response")

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


def run_level1(
    client: OpenAI,
    *,
    config: Level1Config,
    store: ClickHouseStore,
    run_id: str | None = None,
    task_id: str | None = None,
    checkpoint_fn: CheckpointFn | None = None,
) -> None:
    assert config.sample_size is not None, "sample_size must be specified in config for level-1"
    console.print(f"\n[bold]Level-1 major-category labeling[/bold] (sample [cyan]{config.sample_size}[/cyan])")

    current_task_id = task_id or "unknown"
    console.print(f"  Task ID: [cyan]{current_task_id[:12]}...[/cyan]")

    # Sample from news_raw; seed makes the sample deterministic across ablation runs.
    # start (offset) allows continuation tasks to resume after a partial run.
    start = config.start
    df = store.sample_news(config.sample_size, seed=config.seed, offset=start)
    seed_note = f" seed={config.seed}" if config.seed is not None else ""
    start_note = f" start={start}" if start else ""
    console.print(f"Fetched [bold]{len(df)}[/bold] records from news_raw{seed_note}{start_note}")

    keyword_results: list[dict] = []
    llm_pending: list[dict] = []

    # Phase 1: keyword
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Keyword pre-classification...", total=len(df))
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
                        "task_id": current_task_id,
                    }
                )
            else:
                llm_pending.append(row)
            progress.advance(task)

    hit_pct = len(keyword_results) / max(len(df), 1) * 100
    console.print(
        f"  Keyword hits: [green]{len(keyword_results)}[/green] ({hit_pct:.1f}%)  "
        f"Pending LLM: [yellow]{len(llm_pending)}[/yellow]"
    )

    # Save keyword results immediately as checkpoint
    total_saved = 0
    if keyword_results:
        total_saved += _flush_checkpoint(
            keyword_results,
            run_id,
            "level1-keyword",
            0,
            store=store,
            checkpoint_fn=checkpoint_fn,
        )
        console.print(f"  [dim]Checkpoint: saved {total_saved} keyword results[/dim]")

    # Phase 2: LLM fallback
    llm_batch_results: list[dict] = []
    total_batches = -(-len(llm_pending) // config.batch_size)
    batch_count = 0
    llm_usage = LLMUsage()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("LLM fallback...", total=total_batches)
        for i in range(0, len(llm_pending), config.batch_size):
            batch = llm_pending[i : i + config.batch_size]
            try:
                titles = [r["title"] for r in batch]
                results, usage = _llm_classify_level1(client, titles, config, batch_idx=batch_count)
                llm_usage.add(usage)
                for j, r in enumerate(results):
                    if j < len(batch):
                        major = r.get("major", "")
                        if major in all_major_categories:
                            llm_batch_results.append(
                                {
                                    "news_id": batch[j]["news_id"],
                                    "title": batch[j]["title"],
                                    "major_category": major,
                                    "confidence": r.get("confidence", 0.5),
                                    "label_source": "llm",
                                    "task_id": current_task_id,
                                }
                            )
            except LLMParseError as e:
                _dump_llm_error(
                    e,
                    level=1,
                    task_id=current_task_id,
                    run_id=run_id,
                    major=None,
                    batch_idx=batch_count,
                    model=config.model,
                )
                progress.console.print(f"  [red]✗ batch {batch_count} parse failed (dumped to S3): {e}[/red]")
            except InsufficientBalanceError:
                raise
            except Exception as e:
                progress.console.print(f"  [red]✗ batch failed: {e}[/red]")
            batch_count += 1
            progress.advance(task)

            # Checkpoint every N batches
            if batch_count % config.checkpoint_every == 0 and llm_batch_results:
                total_saved += _flush_checkpoint(
                    llm_batch_results,
                    run_id,
                    "level1-llm",
                    batch_count,
                    store=store,
                    checkpoint_fn=checkpoint_fn,
                )
                llm_batch_results = []

            if i + config.batch_size < len(llm_pending):
                time.sleep(0.5)

    # Flush remaining
    if llm_batch_results:
        total_saved += _flush_checkpoint(
            llm_batch_results,
            run_id,
            "level1-llm",
            batch_count,
            store=store,
            checkpoint_fn=checkpoint_fn,
        )

    console.print(f"[bold green]✓ Level-1 done[/bold green] — saved [bold]{total_saved}[/bold] labels")
    llm_usage.print_summary("Level-1")

    # D: reverse validation — sample 5% of keyword results and verify with LLM
    if keyword_results:
        verify_count = max(1, len(keyword_results) // 20)  # 5%
        verify_sample = random_sample(keyword_results, min(verify_count, len(keyword_results)))
        console.print(
            f"\n[bold]Reverse validation[/bold]: sampling "
            f"[cyan]{len(verify_sample)}[/cyan] keyword results for LLM verification..."
        )

        verify_titles = [r["title"] for r in verify_sample]
        mismatches = 0
        try:
            llm_results_verify, _verify_usage = _llm_classify_level1(client, verify_titles, config)
            llm_usage.add(_verify_usage)
            for i, r in enumerate(llm_results_verify):
                if i < len(verify_sample):
                    llm_major = r.get("major", "")
                    kw_major = verify_sample[i]["major_category"]
                    if llm_major != kw_major:
                        mismatches += 1
        except Exception as e:
            console.print(f"  [red]✗ validation batch failed: {e}[/red]")
            return

        accuracy = (len(verify_sample) - mismatches) / len(verify_sample) * 100
        if accuracy >= 90:
            console.print(
                f"  Keyword accuracy: [bold green]{accuracy:.1f}%[/bold green]"
                f" ({mismatches}/{len(verify_sample)} mismatches)"
            )
        else:
            console.print(
                f"  [bold red]⚠ Keyword accuracy only {accuracy:.1f}%"
                f" ({mismatches}/{len(verify_sample)} mismatches)[/bold red]\n"
                f"  [yellow]Consider tuning configs/major_keywords.yaml and rerun.[/yellow]"
            )


def _run_level2_major(
    major: str,
    pending: list[dict],
    client: OpenAI,
    config: Level2Config,
    keyword_sub_map: dict[str, tuple[str, float]],
    current_task_id: str,
    run_id: str | None,
    store: ClickHouseStore,
    checkpoint_fn: CheckpointFn | None,
) -> tuple[int, LLMUsage]:
    """Process all batches for a single major category. Used by both serial and parallel paths.

    Returns (total_saved, llm_usage).
    """
    batch_count = 0
    total_batches = -(-len(pending) // config.batch_size)
    llm_usage = LLMUsage()
    llm_batch_results: list[dict] = []
    total_saved = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"LLM [{major}]", total=total_batches)
        for i in range(0, len(pending), config.batch_size):
            batch = pending[i : i + config.batch_size]
            try:
                titles = [r["title"] for r in batch]
                batch_contents = [r.get("content") for r in batch]
                logger.info(
                    f"Processing batch [{batch_count}/{total_batches}] for major [{major}] with {len(batch)} items"
                )

                results, usage = _llm_classify_level2(
                    client,
                    titles,
                    major,
                    config,
                    contents=batch_contents,
                    batch_idx=batch_count,
                )
                llm_usage.add(usage)
                logger.info(
                    f"  level-2 [{major}] batch={batch_count}/{total_batches} — "
                    f"prompt={usage.prompt_tokens:,} completion={usage.completion_tokens:,} "
                    f"total={usage.total_tokens:,} time={usage.total_time:.2f}s"
                )
                for j, r in enumerate(results):
                    if j < len(batch):
                        news_id_str = str(batch[j]["news_id"])
                        # Determine sub_category and label_source
                        if major in _single_sub_majors:
                            sub = _single_sub_majors[major]
                            conf = r.get("confidence", 0.5)
                            label_source = "auto+llm"
                        elif news_id_str in keyword_sub_map:
                            sub, conf = keyword_sub_map[news_id_str]
                            label_source = "keyword+llm"
                        else:
                            sub = r.get("sub", "")
                            conf = r.get("confidence", 0.5)
                            label_source = "llm"
                        if sub in sub_to_major:
                            llm_batch_results.append(
                                {
                                    "news_id": batch[j]["news_id"],
                                    "title": batch[j]["title"],
                                    "datetime": batch[j]["datetime"],
                                    "major_category": major,
                                    "sub_category": sub,
                                    "sentiment": r.get("sentiment", "中性"),
                                    "impact_score": r.get("impact_score", 0.5),
                                    "analysis_logic": r.get("analysis", {}).get("logic", ""),
                                    "key_evidence": r.get("analysis", {}).get("key_evidence", ""),
                                    "expectation": r.get("analysis", {}).get("expectation", "符合预期"),
                                    "confidence": conf,
                                    "label_source": label_source,
                                    "level1_task_id": batch[j]["task_id"],
                                    "level2_task_id": current_task_id,
                                    "run_id": run_id,
                                }
                            )
            except LLMParseError as e:
                _dump_llm_error(
                    e,
                    level=2,
                    task_id=current_task_id,
                    run_id=run_id,
                    major=major,
                    batch_idx=batch_count,
                    model=config.model,
                )
                progress.console.print(f"  [red]✗ batch {batch_count} [{major}] parse failed (dumped to S3): {e}[/red]")
            except InsufficientBalanceError:
                raise
            except Exception as e:
                progress.console.print(f"  [red]✗ batch failed: {e}[/red]")
            batch_count += 1
            progress.advance(task)

            # Checkpoint every N batches
            if batch_count % config.checkpoint_every == 0 and llm_batch_results:
                saved = store.save_sub_labels(llm_batch_results, run_id)
                if checkpoint_fn:
                    checkpoint_fn(run_id, "level2-llm", batch_count, saved)
                total_saved += saved
                llm_batch_results = []

    # Flush remaining LLM results
    if llm_batch_results:
        saved = store.save_sub_labels(llm_batch_results, run_id)
        if checkpoint_fn:
            checkpoint_fn(run_id, "level2-llm", batch_count, saved)
        total_saved += saved

    return total_saved, llm_usage


def run_level2(
    client: OpenAI,
    *,
    config: Level2Config,
    store: ClickHouseStore,
    run_id: str | None = None,
    task_id: str | None = None,
    checkpoint_fn: CheckpointFn | None = None,
) -> None:
    sample_note = f"sample [cyan]{config.sample_size}[/cyan]" if config.sample_size else "all records"
    console.print(f"\n[bold]Level-2 sub-category + sentiment labeling[/bold] ({sample_note})")

    current_task_id = task_id or "unknown"
    console.print(f"  Task ID: [cyan]{current_task_id[:12]}...[/cyan]")

    assert config.level1_task_id, "level1_task_id must be specified in config for level-2"
    source_task_id = config.level1_task_id
    console.print(f"  Level-1 Task ID: [cyan]{source_task_id[:12]}...[/cyan]")

    # Determine effective concurrency
    majors_count = len(config.major_categories) if config.major_categories else len(all_major_categories)
    effective_concurrency = config.concurrency
    if effective_concurrency > majors_count:
        logger.warning(
            f"concurrency={effective_concurrency} > number of majors ({majors_count}), downgrading to {majors_count}"
        )
        effective_concurrency = majors_count

    console.print(f"  Concurrency: [cyan]{effective_concurrency}[/cyan] (majors: {majors_count})")

    # Build WHERE clause for major_categories filter
    major_filter = ""
    if config.major_categories:
        cats = ", ".join(f"'{c}'" for c in config.major_categories)
        major_filter = f"AND c.major_category IN ({cats})"

    # Fetch ALL eligible level-1 results (no LIMIT yet)
    rows = store.execute(
        f"""
        SELECT c.news_id, c.title, c.major_category, c.task_id, r.datetime, r.content
        FROM news_classified c
        JOIN news_raw r ON c.news_id = r.news_id
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
        {"source_id": source_task_id, "task_id": current_task_id},
    )
    if rows:
        df = pl.DataFrame(
            rows, schema=["news_id", "title", "major_category", "task_id", "datetime", "content"], orient="row"
        )
    else:
        df = pl.DataFrame(schema=["news_id", "title", "major_category", "task_id", "datetime", "content"])

    total_eligible = len(df)
    console.print(f"Fetched [bold]{total_eligible}[/bold] eligible records from news_classified")

    # Apply random sampling if sample_size is specified and smaller than total
    if config.sample_size is not None and config.sample_size < total_eligible:
        df = df.sample(n=config.sample_size, seed=config.seed)
        console.print(f"Sampled [bold]{len(df)}[/bold] records (seed={config.seed})")

    keyword_results: list[dict] = []  # rows with keyword sub hit, for stats/validation
    keyword_sub_map: dict[str, tuple[str, float]] = {}  # news_id -> (sub, conf) from keyword
    llm_pending_by_major: dict[str, list[dict]] = {}

    # Phase 1: keyword pre-classification
    # Note: single-sub majors still go to LLM for sentiment; sub_category is forced in Phase 2.
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Keyword pre-classification...", total=len(df))
        for row in df.iter_rows(named=True):
            major = row["major_category"]

            result = _keyword_classify_sub(row["title"], major)
            if result and major not in _single_sub_majors:
                # Keyword hit: record predicted sub but still send to LLM for sentiment
                sub, conf = result
                keyword_sub_map[str(row["news_id"])] = (sub, conf)
                keyword_results.append(row)  # keep for reverse-validation stats
            # All rows go to LLM (sentiment needed; single-sub sub forced in Phase 2)
            llm_pending_by_major.setdefault(major, []).append(row)
            progress.advance(task)

    llm_pending_total = sum(len(v) for v in llm_pending_by_major.values())
    kw_pct = len(keyword_results) / max(len(df), 1) * 100
    console.print(
        f"  Keyword pre-classified: [green]{len(keyword_results)}[/green]"
        f" ({kw_pct:.1f}%, sub fixed, LLM for sentiment)  "
        f"LLM total: [yellow]{llm_pending_total}[/yellow]"
    )

    # Phase 2: LLM for all items (sentiment + sub for non-keyword)
    total_saved = 0
    llm_usage = LLMUsage()

    if effective_concurrency <= 1:
        # Serial path (original behavior)
        for major, pending in llm_pending_by_major.items():
            saved, usage = _run_level2_major(
                major=major,
                pending=pending,
                client=client,
                config=config,
                keyword_sub_map=keyword_sub_map,
                current_task_id=current_task_id,
                run_id=run_id,
                store=store,
                checkpoint_fn=checkpoint_fn,
            )
            total_saved += saved
            llm_usage.add(usage)
            logger.info(f"Completed major [{major}]: saved={saved}")
            time.sleep(0.5)
    else:
        # Parallel path using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed

        majors = list(llm_pending_by_major.keys())
        console.print(f"  Running [bold]{effective_concurrency}[/bold] parallel workers...")

        with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
            futures = {
                executor.submit(
                    _run_level2_major,
                    major,
                    llm_pending_by_major[major],
                    client,
                    config,
                    keyword_sub_map,
                    current_task_id,
                    run_id,
                    store,
                    checkpoint_fn,
                ): major
                for major in majors
            }

            for future in as_completed(futures):
                major = futures[future]
                try:
                    saved, usage = future.result()
                    total_saved += saved
                    llm_usage.add(usage)
                    logger.info(f"Completed major [{major}]: saved={saved}")
                except Exception as e:
                    logger.error(f"Major [{major}] failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    console.print(
        f"[bold green]✓ Level-2 done[/bold green] — updated [bold]{len(keyword_results) + total_saved}[/bold] labels"
    )
    llm_usage.print_summary("Level-2")

    # D: reverse validation — sample 5% of keyword results and verify with LLM
    if keyword_results:
        verify_count = max(1, len(keyword_results) // 20)
        verify_sample = random_sample(keyword_results, min(verify_count, len(keyword_results)))
        console.print(
            f"\n[bold]Reverse validation[/bold]: sampling "
            f"[cyan]{len(verify_sample)}[/cyan] keyword results for LLM verification..."
        )

        mismatches = 0
        for major, group in _group_by_major(verify_sample).items():
            try:
                titles = [r["title"] for r in group]
                llm_res, _v_usage = _llm_classify_level2(client, titles, major, config)
                llm_usage.add(_v_usage)
                for i, r in enumerate(llm_res):
                    if i < len(group):
                        if r.get("sub", "") != group[i]["sub_category"]:
                            mismatches += 1
            except Exception as e:
                console.print(f"  [red]✗ validation [{major}] failed: {e}[/red]")

        accuracy = (len(verify_sample) - mismatches) / len(verify_sample) * 100
        if accuracy >= 90:
            console.print(
                f"  Keyword accuracy: [bold green]{accuracy:.1f}%[/bold green]"
                f" ({mismatches}/{len(verify_sample)} mismatches)"
            )
        else:
            console.print(
                f"  [bold red]⚠ Keyword accuracy only {accuracy:.1f}%"
                f" ({mismatches}/{len(verify_sample)} mismatches)[/bold red]\n"
                f"  [yellow]Consider tuning configs/sub_keywords.yaml and rerun.[/yellow]"
            )


def print_label_stats(store: ClickHouseStore) -> None:
    stats = store.execute("""
        SELECT major_category, label_source, COUNT(*) AS cnt,
               round(AVG(confidence), 3) AS avg_conf
        FROM news_classified
        WHERE major_category IS NOT NULL
        GROUP BY major_category, label_source
        ORDER BY major_category, label_source
    """)

    table = Table(title="Label Statistics", show_lines=True)
    table.add_column("Major Category", style="cyan")
    table.add_column("Source", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Avg Confidence", justify="right", style="magenta")
    for row in stats:
        major, source, cnt, avg_conf = row
        src_style = "green" if source == "keyword" else "yellow"
        table.add_row(
            major or "-",
            f"[{src_style}]{source}[/]",
            str(cnt),
            str(avg_conf),
        )
    console.print(table)

    total = store.execute("SELECT COUNT(*) FROM news_classified")[0][0]
    kw_count = store.execute("SELECT COUNT(*) FROM news_classified WHERE label_source = 'keyword'")[0][0]
    console.print(
        f"Total: [bold]{total}[/bold] | Keyword: [green]{kw_count}[/green] | LLM: [yellow]{total - kw_count}[/yellow]"
    )
