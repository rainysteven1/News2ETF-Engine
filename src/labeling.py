"""
Label news articles with hierarchical industry categories and sentiment.

Pipeline:
  1. Keyword matching on title → high-confidence hits (~60-70%)
  2. Remaining ambiguous titles → batch LLM via GLM-4-Flash
  3. Results written to DuckDB news_classified table
"""

import json
import re
import time
from dataclasses import dataclass
from random import sample as random_sample
from typing import Any

import duckdb
import jsonschema
import yaml
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from zhipuai import ZhipuAI

from src.common import CONFIGS_DIR, console, get_labeling_config, get_loki_url
from src.common.tracking import save_checkpoint
from src.store import batch_update_labels, sample_unlabeled, save_labels
from src.utils.loki_sink import LokiSink

LLM_BATCH_SIZE = 20
CHECKPOINT_EVERY = 5  # save checkpoint every N batches

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
    template = (CONFIGS_DIR / "prompts" / "label_level1_system.txt").read_text(encoding="utf-8").strip()
    schema_str = json.dumps(_LEVEL1_SCHEMA, indent=2, ensure_ascii=False)
    return template.replace("{major_categories}", json.dumps(major_categories, ensure_ascii=False)).replace(
        "{schema}", schema_str
    )


def _load_level2_prompt(major: str, subs: list[str]) -> str:
    template = (CONFIGS_DIR / "prompts" / "label_level2_system.txt").read_text(encoding="utf-8").strip()
    schema_str = json.dumps(_LEVEL2_SCHEMA, indent=2, ensure_ascii=False)
    return (
        template.replace("{major}", major)
        .replace("{subs}", json.dumps(subs, ensure_ascii=False))
        .replace("{schema}", schema_str)
    )


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

_LEVEL1_SCHEMA: dict = json.loads((CONFIGS_DIR / "prompts" / "label_level1_schema.json").read_text(encoding="utf-8"))
_LEVEL2_SCHEMA: dict = json.loads((CONFIGS_DIR / "prompts" / "label_level2_schema.json").read_text(encoding="utf-8"))
# Extract per-item schema for response validation
_LEVEL1_ITEM_SCHEMA: dict = _LEVEL1_SCHEMA["items"]
_LEVEL2_ITEM_SCHEMA: dict = _LEVEL2_SCHEMA["items"]


# ──────────────────────────────────────────────
# Loki logging setup
# ──────────────────────────────────────────────

logger.remove()  # suppress default stderr output so Rich progress bars stay clean

_loki_sink: LokiSink | None = None


def setup_loki_logging(run_id: str | None, level: str = "labeling") -> int | None:
    """Add a Loki sink with run_id label. Returns handler ID for cleanup."""
    global _loki_sink
    loki_url = get_loki_url()
    if not loki_url:
        logger.info("Loki URL not configured, skipping Loki logging")
        return None

    rid = run_id or "unknown"
    _loki_sink = LokiSink(
        url=loki_url,
        labels={
            "app": "news2etf",
            "component": "labeling",
            "level_stage": level,
            "run_id": rid,
        },
    )
    handler_id = logger.add(
        _loki_sink.write,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )
    logger.info(f"Loki logging enabled → {loki_url}  run_id={rid}")
    return handler_id


def teardown_loki_logging(handler_id: int | None) -> None:
    """Flush and remove the Loki sink."""
    global _loki_sink
    if _loki_sink is not None:
        _loki_sink.stop()
        _loki_sink = None
    if handler_id is not None:
        logger.remove(handler_id)


def _flush_checkpoint(
    con: duckdb.DuckDBPyConnection,
    results: list[dict],
    run_id: str | None,
    stage: str,
    batch_idx: int,
) -> int:
    """Flush accumulated results to DuckDB and record checkpoint in SQLite."""
    if not results:
        return 0
    saved = save_labels(con, results)
    save_checkpoint(run_id, stage, batch_idx, saved)
    logger.info(f"Checkpoint [{stage}] batch={batch_idx} saved={saved}")
    return saved


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


@dataclass
class LLMUsage:
    """Accumulated token usage statistics across LLM batches."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    batches: int = 0
    total_time: float = 0.0

    def add(self, usage: dict, elapsed: float) -> None:
        self.total_tokens += usage.get("total_tokens", 0)
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.cached_tokens += usage.get("cached_tokens", 0)
        self.batches += 1
        self.total_time += elapsed

    def print_summary(self, title: str) -> None:
        if self.batches == 0:
            return
        console.print(
            f"\n[bold]📊 {title} Token Statistics[/bold]\n"
            f"  Batches: [cyan]{self.batches}[/cyan]  "
            f"Total time: [cyan]{self.total_time:.1f}s[/cyan]  "
            f"Avg: [cyan]{self.total_time / self.batches:.2f}s[/cyan]/batch\n"
            f"  Prompt tokens: [blue]{self.prompt_tokens:,}[/blue]  "
            f"Completion tokens: [blue]{self.completion_tokens:,}[/blue]  "
            f"Total: [bold]{self.total_tokens:,}[/bold]\n"
            f"  Cached tokens: [green]{self.cached_tokens:,}[/green]  "
            f"Cache rate: [green]"
            f"{self.cached_tokens / max(self.prompt_tokens, 1) * 100:.1f}%[/green]"
        )


def _extract_usage(response: object) -> dict[str, Any]:
    """Safely extract token usage from a ZhipuAI response."""
    usage: dict = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    }
    resp_usage = getattr(response, "usage", None)
    if resp_usage is None:
        return usage
    usage["total_tokens"] = getattr(resp_usage, "total_tokens", 0) or 0
    usage["prompt_tokens"] = getattr(resp_usage, "prompt_tokens", 0) or 0
    usage["completion_tokens"] = getattr(resp_usage, "completion_tokens", 0) or 0
    # cached_tokens may live under prompt_tokens_details
    details = getattr(resp_usage, "prompt_tokens_details", None)
    if details is not None:
        usage["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0
    return usage


def _parse_llm_json(content: str, item_schema: dict | None = None) -> list[dict]:
    """Parse LLM JSON response with tolerance for common formatting errors.

    If *item_schema* is provided, each element is validated against it;
    items that fail validation are dropped with a warning instead of
    raising, so a single malformed item doesn't discard the whole batch.
    """
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    def _try_parse(raw: str) -> list[dict]:
        parsed = json.loads(raw)
        # LLM occasionally wraps the array in an object key
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        return parsed  # type: ignore[return-value]

    parsed: list[dict] | None = None
    try:
        parsed = _try_parse(content)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        # C: common LLM format error fixes
        # 1. trailing comma: [...,]
        content = re.sub(r",\s*([}\]])", r"\1", content)
        # 2. single quotes → double quotes
        content = content.replace("'", '"')
        # 3. try to extract the first [...] block
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = _try_parse(match.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        raise json.JSONDecodeError("LLM response cannot be parsed as JSON", content, 0)

    if item_schema is None:
        return parsed

    valid: list[dict] = []
    for i, item in enumerate(parsed):
        try:
            jsonschema.validate(instance=item, schema=item_schema)
            valid.append(item)
        except jsonschema.ValidationError as exc:
            logger.warning(f"  item[{i}] schema validation failed — skipped: {exc.message}")
    return valid


def _llm_classify_level1(client: ZhipuAI, titles: list[str]) -> tuple[list[dict], dict]:
    t0 = time.time()
    logger.info(f"LLM level-1 request: {len(titles)} titles")
    for i, t in enumerate(titles):
        logger.debug(f"  sample {i + 1}: {t}")
    try:
        cfg = get_labeling_config(1)
        system_prompt = _load_level1_prompt(all_major_categories)
        titles_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分类以下新闻标题：\n{titles_str}"},
            ],
            temperature=cfg["temperature"],
            response_format={"type": "json_object"},
            max_tokens=cfg["max_tokens"],
        )
        elapsed = time.time() - t0

        usage = _extract_usage(response)
        usage["_elapsed"] = elapsed
        content = response.choices[0].message.content.strip()  # type: ignore

        logger.debug(f"  response: {content}")
        logger.info(f"  level-1 done — {elapsed:.2f}s  tokens={usage['total_tokens']} cached={usage['cached_tokens']}")
        return _parse_llm_json(content, item_schema=_LEVEL1_ITEM_SCHEMA), usage
    except Exception as e:
        logger.error(f"LLM level-1 request failed: {e}")
        raise


def _llm_classify_level2(client: ZhipuAI, titles: list[str], major: str) -> tuple[list[dict], dict]:
    t0 = time.time()
    logger.info(f"LLM level-2 request [{major}]: {len(titles)} titles")
    for i, t in enumerate(titles):
        logger.debug(f"  sample {i + 1}: {t}")
    try:
        cfg = get_labeling_config(2)
        system_prompt = _load_level2_prompt(major, hierarchy[major])
        titles_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请分类以下新闻标题：\n{titles_str}"},
            ],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        elapsed = time.time() - t0
        usage = _extract_usage(response)
        usage["_elapsed"] = elapsed
        content = response.choices[0].message.content.strip()  # type: ignore
        logger.debug(f"  response: {content}")
        logger.info(
            f"  level-2 [{major}] done — {elapsed:.2f}s  tokens={usage['total_tokens']} cached={usage['cached_tokens']}"
        )
        return _parse_llm_json(content, item_schema=_LEVEL2_ITEM_SCHEMA), usage
    except Exception as e:
        logger.error(f"LLM level-2 [{major}] request failed: {e}")
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
    con: duckdb.DuckDBPyConnection,
    client: ZhipuAI,
    sample_size: int,
    *,
    run_id: str | None = None,
) -> None:
    loki_handler = setup_loki_logging(run_id, level="level1")
    console.print(f"\n[bold]Level-1 major-category labeling[/bold] (sample [cyan]{sample_size}[/cyan])")

    df = sample_unlabeled(con, sample_size)
    console.print(f"Fetched [bold]{len(df)}[/bold] unlabeled records (ordered by datetime)")

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
                        "sub_category": None,
                        "sentiment": None,
                        "confidence": conf,
                        "label_source": "keyword",
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
        total_saved += _flush_checkpoint(con, keyword_results, run_id, "level1-keyword", 0)
        console.print(f"  [dim]Checkpoint: saved {total_saved} keyword results[/dim]")

    # Phase 2: LLM fallback
    llm_batch_results: list[dict] = []
    total_batches = -(-len(llm_pending) // LLM_BATCH_SIZE)
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
        for i in range(0, len(llm_pending), LLM_BATCH_SIZE):
            batch = llm_pending[i : i + LLM_BATCH_SIZE]
            try:
                titles = [r["title"] for r in batch]
                results, usage = _llm_classify_level1(client, titles)
                llm_usage.add(usage, usage.get("_elapsed", 0))
                for j, r in enumerate(results):
                    if j < len(batch):
                        major = r.get("major", "")
                        if major in all_major_categories:
                            llm_batch_results.append(
                                {
                                    "news_id": batch[j]["news_id"],
                                    "title": batch[j]["title"],
                                    "major_category": major,
                                    "sub_category": None,
                                    "sentiment": None,
                                    "confidence": r.get("confidence", 0.5),
                                    "label_source": "llm",
                                }
                            )
            except Exception as e:
                progress.console.print(f"  [red]✗ batch failed: {e}[/red]")
            batch_count += 1
            progress.advance(task)

            # Checkpoint every N batches
            if batch_count % CHECKPOINT_EVERY == 0 and llm_batch_results:
                total_saved += _flush_checkpoint(con, llm_batch_results, run_id, "level1-llm", batch_count)
                llm_batch_results = []

            if i + LLM_BATCH_SIZE < len(llm_pending):
                time.sleep(0.5)

    # Flush remaining
    if llm_batch_results:
        total_saved += _flush_checkpoint(con, llm_batch_results, run_id, "level1-llm", batch_count)

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
            llm_results_verify, _verify_usage = _llm_classify_level1(client, verify_titles)
            llm_usage.add(_verify_usage, _verify_usage.get("_elapsed", 0))
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

    teardown_loki_logging(loki_handler)


def run_level2(
    con: duckdb.DuckDBPyConnection,
    client: ZhipuAI,
    sample_size: int,
    *,
    run_id: str | None = None,
) -> None:
    loki_handler = setup_loki_logging(run_id, level="level2")
    console.print(f"\n[bold]Level-2 sub-category + sentiment labeling[/bold] (sample [cyan]{sample_size}[/cyan])")

    df = con.execute(f"""
        SELECT c.news_id, c.title, c.major_category
        FROM news_classified c
        JOIN news_raw r ON c.news_id = r.news_id
        WHERE c.major_category IS NOT NULL AND c.sub_category IS NULL
        ORDER BY r.datetime, c.news_id
        LIMIT {sample_size}
    """).pl()
    console.print(f"Fetched [bold]{len(df)}[/bold] records pending sub-category labeling")

    keyword_results: list[dict] = []
    single_sub_results: list[dict] = []
    llm_pending_by_major: dict[str, list[dict]] = {}

    # Phase 1: keyword + single-sub auto-assign
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Keyword pre-classification...", total=len(df))
        for row in df.iter_rows(named=True):
            major = row["major_category"]

            # If major has only one sub-category, auto-assign without keyword/LLM
            if major in _single_sub_majors:
                single_sub_results.append(
                    {
                        "news_id": row["news_id"],
                        "title": row["title"],
                        "major_category": major,
                        "sub_category": _single_sub_majors[major],
                        "sentiment": None,
                        "confidence": 1.0,
                        "label_source": "auto",
                    }
                )
                progress.advance(task)
                continue

            result = _keyword_classify_sub(row["title"], major)
            if result:
                sub, conf = result
                keyword_results.append(
                    {
                        "news_id": row["news_id"],
                        "title": row["title"],
                        "major_category": major,
                        "sub_category": sub,
                        "sentiment": None,
                        "confidence": conf,
                        "label_source": "keyword",
                    }
                )
            else:
                llm_pending_by_major.setdefault(major, []).append(row)
            progress.advance(task)

    llm_pending_total = sum(len(v) for v in llm_pending_by_major.values())
    hit_pct = len(keyword_results) / max(len(df), 1) * 100
    auto_pct = len(single_sub_results) / max(len(df), 1) * 100
    console.print(
        f"  Auto (single-sub): [blue]{len(single_sub_results)}[/blue] ({auto_pct:.1f}%)  "
        f"Keyword hits: [green]{len(keyword_results)}[/green] ({hit_pct:.1f}%)  "
        f"Pending LLM: [yellow]{llm_pending_total}[/yellow]"
    )

    # Save keyword + auto results immediately as checkpoint
    total_saved = 0
    early_results = single_sub_results + keyword_results
    if early_results:
        total_saved += _flush_checkpoint(
            con,
            [{**r, "sub_category": r["sub_category"], "sentiment": r["sentiment"]} for r in early_results],
            run_id,
            "level2-keyword",
            0,
        )
        # For keyword/auto results we need batch_update since rows exist
        batch_update_labels(con, early_results)
        console.print(f"  [dim]Checkpoint: saved {len(early_results)} keyword/auto results[/dim]")

    # Phase 2: LLM fallback per major
    llm_batch_results: list[dict] = []
    batch_count = 0
    llm_usage = LLMUsage()
    for major, pending in llm_pending_by_major.items():
        total_batches = -(-len(pending) // LLM_BATCH_SIZE)
        with Progress(
            TextColumn("[progress.description]{{task.description}}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"LLM [{major}]", total=total_batches)
            for i in range(0, len(pending), LLM_BATCH_SIZE):
                batch = pending[i : i + LLM_BATCH_SIZE]
                try:
                    titles = [r["title"] for r in batch]
                    results, usage = _llm_classify_level2(client, titles, major)
                    llm_usage.add(usage, usage.get("_elapsed", 0))
                    for j, r in enumerate(results):
                        if j < len(batch):
                            sub = r.get("sub", "")
                            if sub in sub_to_major:
                                llm_batch_results.append(
                                    {
                                        "news_id": batch[j]["news_id"],
                                        "title": batch[j]["title"],
                                        "major_category": major,
                                        "sub_category": sub,
                                        "sentiment": r.get("sentiment", "中性"),
                                        "confidence": r.get("confidence", 0.5),
                                        "label_source": "llm",
                                    }
                                )
                except Exception as e:
                    progress.console.print(f"  [red]✗ batch failed: {e}[/red]")
                batch_count += 1
                progress.advance(task)

                # Checkpoint every N batches
                if batch_count % CHECKPOINT_EVERY == 0 and llm_batch_results:
                    batch_update_labels(con, llm_batch_results)
                    save_checkpoint(run_id, "level2-llm", batch_count, len(llm_batch_results))
                    total_saved += len(llm_batch_results)
                    llm_batch_results = []

                if i + LLM_BATCH_SIZE < len(pending):
                    time.sleep(0.5)

    # Flush remaining LLM results
    if llm_batch_results:
        batch_update_labels(con, llm_batch_results)
        save_checkpoint(run_id, "level2-llm", batch_count, len(llm_batch_results))
        total_saved += len(llm_batch_results)

    console.print(
        f"[bold green]✓ Level-2 done[/bold green] — updated [bold]{len(early_results) + total_saved}[/bold] labels"
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
                llm_res, _v_usage = _llm_classify_level2(client, titles, major)
                llm_usage.add(_v_usage, _v_usage.get("_elapsed", 0))
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

    teardown_loki_logging(loki_handler)


def print_label_stats(con: duckdb.DuckDBPyConnection, console: Console) -> None:
    stats = con.execute("""
        SELECT major_category, label_source, COUNT(*) AS cnt,
               ROUND(AVG(confidence), 3) AS avg_conf
        FROM news_classified
        WHERE major_category IS NOT NULL
        GROUP BY major_category, label_source
        ORDER BY major_category, label_source
    """).pl()

    table = Table(title="Label Statistics", show_lines=True)
    table.add_column("Major Category", style="cyan")
    table.add_column("Source", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Avg Confidence", justify="right", style="magenta")
    for row in stats.iter_rows(named=True):
        src_style = "green" if row["label_source"] == "keyword" else "yellow"
        table.add_row(
            row["major_category"] or "-",
            f"[{src_style}]{row['label_source']}[/]",
            str(row["cnt"]),
            str(row["avg_conf"]),
        )
    console.print(table)

    total = con.execute("SELECT COUNT(*) FROM news_classified").fetchone()[0]  # type: ignore
    kw_count = con.execute("SELECT COUNT(*) FROM news_classified WHERE label_source = 'keyword'").fetchone()[0]  # type: ignore
    console.print(
        f"Total: [bold]{total}[/bold] | Keyword: [green]{kw_count}[/green] | LLM: [yellow]{total - kw_count}[/yellow]"
    )
