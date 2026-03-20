"""
Labeling task executor for the experiment system.

Orchestrates the labeling pipeline: owns DuckDBStore, Loki logging,
and checkpoint management. Passes all dependencies into the pure
business-logic functions in src.labeling.
"""

import uuid
from typing import Any

import duckdb
from loguru import logger

from src.common import get_config
from src.common.registry import TaskExecutor, TaskMetadata, TaskRegistry
from src.db import Task
from src.db.models import TaskCheckpoint
from src.db.store import DuckDBStore, duckdb_store
from src.experiment.manager import ExperimentManager
from src.labeling import Level1Config, Level2Config, run_level1, run_level2
from src.utils.llm_client import get_llm_client
from src.utils.loki_sink import LokiSink


def _setup_loki(run_id: str | None, level_stage: str) -> tuple[LokiSink | None, int | None]:
    """Create a Loki sink and add it to loguru. Returns (sink, handler_id).

    Also raises the default stderr handler to INFO so that DEBUG messages
    (e.g. per-sample titles) only flow to Loki, not the console.
    """
    # Remove all existing handlers, then re-add stderr at INFO level so that
    # DEBUG messages (e.g. per-sample titles) only flow to Loki, not the console.
    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<5}</level> | {message}",
    )

    loki_url = get_config().loki.url
    if not loki_url:
        return None, None

    sink = LokiSink(
        url=loki_url,
        labels={
            "app": "news2etf",
            "component": "labeling",
            "level_stage": level_stage,
            "run_id": run_id or "unknown",
        },
    )
    handler_id = logger.add(
        sink.write,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )
    logger.info(f"Loki logging enabled → {loki_url}  run_id={run_id}")
    return sink, handler_id


def _teardown_loki(sink: LokiSink | None, handler_id: int | None) -> None:
    """Flush and remove the Loki sink."""
    if sink is not None:
        sink.stop()
    if handler_id is not None:
        logger.remove(handler_id)


def _build_summary(con: duckdb.DuckDBPyConnection, task_uuid_str: str, level: int) -> dict:
    """Query DuckDB and build a structured result summary for the task record."""
    if level == 1:
        rows = con.execute(
            """
            SELECT major_category, label_source, COUNT(*) AS cnt,
                   ROUND(AVG(confidence), 3) AS avg_conf
            FROM news_classified
            WHERE task_id = ?
            GROUP BY major_category, label_source
            ORDER BY major_category, label_source
            """,
            [task_uuid_str],
        ).fetchall()
        by_source = {}
        by_major = {}
        for major, source, cnt, avg_conf in rows:
            by_source[source] = by_source.get(source, 0) + cnt
            by_major.setdefault(major or "unknown", {})[source] = {"count": cnt, "avg_confidence": avg_conf}
        return {
            "by_label_source": by_source,
            "by_major_category": by_major,
        }
    else:
        rows = con.execute(
            """
            SELECT major_category, sub_category, label_source,
                   sentiment, COUNT(*) AS cnt,
                   ROUND(AVG(confidence), 3) AS avg_conf
            FROM news_sub_classified
            WHERE level2_task_id = ?
            GROUP BY major_category, sub_category, label_source, sentiment
            ORDER BY major_category, sub_category
            """,
            [task_uuid_str],
        ).fetchall()
        by_source: dict = {}
        by_sub: dict = {}
        by_sentiment: dict = {}
        for major, sub, source, sentiment, cnt, avg_conf in rows:
            by_source[source] = by_source.get(source, 0) + cnt
            key = f"{major} / {sub}"
            by_sub.setdefault(key, {})[source] = {"count": cnt, "avg_confidence": avg_conf}
            if sentiment:
                by_sentiment[sentiment] = by_sentiment.get(sentiment, 0) + cnt
        return {
            "by_label_source": by_source,
            "by_sub_category": by_sub,
            "by_sentiment": by_sentiment,
        }


@TaskRegistry.register
class LabelingTaskExecutor(TaskExecutor):
    """Labeling task executor for news classification.

    Owns:
      - DuckDBStore for data persistence
      - Loki logging lifecycle
      - Checkpoint handler (via ExperimentManager / PostgreSQL)
    """

    def __init__(self) -> None:
        self.store: DuckDBStore = duckdb_store

    @property
    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            name="labeling",
            description="News labeling task with hierarchical classification",
            required_params=["model", "temperature"],
            optional_params=[
                "max_tokens",
                "level",
                "sample_size",
                "force_relabel",
                "level1_task_id",
                "seed",
                "start",
                "batch_size_l1",
                "batch_size_l2",
                "checkpoint_every",
                "llm_retry",
                "merge_source_task_ids",
            ],
        )

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        # Merge task: combines results of multiple level-1 tasks, no LLM needed
        if "merge_source_task_ids" in params:
            source_ids = params["merge_source_task_ids"]
            if not isinstance(source_ids, list) or len(source_ids) < 1:
                return False, "merge_source_task_ids must be a non-empty list of task IDs"
            root_cfg: dict | None = None
            with ExperimentManager() as mgr:
                for sid in source_ids:
                    try:
                        src_uuid = uuid.UUID(str(sid))
                    except ValueError:
                        return False, f"Invalid UUID in merge_source_task_ids: {sid}"
                    src_task = mgr.get_task(src_uuid)
                    if src_task is None:
                        return False, f"Source task not found: {sid}"
                    if src_task.task_type != "labeling":
                        return False, f"Source task {sid} is not a labeling task"
                    src_cfg = src_task.config or {}
                    if src_cfg.get("level", 1) != 1:
                        return False, f"Source task {sid} is not a level-1 task"
                    compat_keys = ["model", "temperature"]
                    if root_cfg is None:
                        root_cfg = {k: src_cfg.get(k) for k in compat_keys}
                    else:
                        for k in compat_keys:
                            if src_cfg.get(k) != root_cfg[k]:
                                return False, (
                                    f"Source task {sid} has incompatible config: "
                                    f"{k}={src_cfg.get(k)!r} vs expected {root_cfg[k]!r}"
                                )
            return True, None

        if "model" not in params:
            return False, "Missing required param: model"
        if "temperature" not in params:
            return False, "Missing required param: temperature"

        # Verify the model is in the registry and its API key is configured
        from src.utils.llm_client import resolve_provider

        model = params["model"]
        try:
            provider = resolve_provider(model)
        except ValueError as e:
            return False, str(e)

        import os

        if not os.environ.get(provider.key_env):
            return False, (f"Model '{model}' requires env var {provider.key_env} to be set")

        level = params.get("level", 1)
        if level not in [1, 2]:
            return False, f"Invalid level: {level}, must be 1 or 2"

        if level == 2:
            level1_task_id_str = params.get("level1_task_id")
            if not level1_task_id_str:
                return False, "level=2 requires 'level1_task_id'"
            try:
                level1_uuid = uuid.UUID(str(level1_task_id_str))
            except ValueError:
                return False, f"Invalid level1_task_id (not a valid UUID): {level1_task_id_str}"

            with ExperimentManager() as mgr:
                level1_task = mgr.get_task(level1_uuid)
            if level1_task is None:
                return False, f"level1_task_id not found: {level1_task_id_str}"
            if level1_task.task_type != "labeling":
                return False, f"Referenced task is not a labeling task: {level1_task_id_str}"
            if (level1_task.config or {}).get("level", 1) != 1:
                return False, f"Referenced task is not a level-1 task: {level1_task_id_str}"

        return True, None

    # ── checkpoint callback ─────────────────────

    def _make_checkpoint_fn(self, task_id: uuid.UUID):
        """Return a checkpoint callback that persists via ExperimentManager."""
        from src.experiment.manager import ExperimentManager

        def _save(run_id: str | None, stage: str, batch_idx: int, saved_count: int) -> None:
            with ExperimentManager() as mgr:
                mgr.save_checkpoint(
                    task_id=task_id,
                    stage=stage,
                    batch_idx=batch_idx,
                    processed_count=saved_count,
                )

        return _save

    # ── execute ─────────────────────────────────

    def execute(self, task: Task, run_id: str | None = None) -> dict[str, Any]:
        # Config is stored directly on the task
        cfg = task.config or {}

        level = cfg.get("level", 1)
        sample_size = cfg.get("sample_size", 10000)
        labeling_defaults = get_config().labeling
        _common = dict(
            model=cfg.get("model", "glm-4-flash"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens"),
            checkpoint_every=cfg.get("checkpoint_every", labeling_defaults.checkpoint_every),
            llm_retry=cfg.get("llm_retry", labeling_defaults.llm_retry),
            seed=cfg.get("seed"),
        )
        if level == 1:
            config = Level1Config(
                **_common,
                start=cfg.get("start", 0),
                batch_size=cfg.get("batch_size_l1", labeling_defaults.batch_size_l1),
            )
        else:
            config = Level2Config(
                **_common,
                batch_size=cfg.get("batch_size_l2", labeling_defaults.batch_size_l2),
            )

        try:
            client = get_llm_client(config.model)
        except ValueError as e:
            return {"status": "error", "message": str(e), "total_labeled": 0}

        task_uuid_str = str(task.task_id)

        # Set up infrastructure
        self.store.init_db()
        loki_sink, loki_handler = _setup_loki(run_id, level_stage=f"level{level}")
        checkpoint_fn = self._make_checkpoint_fn(task.task_id)

        seed = cfg.get("seed", None)

        con = duckdb.connect(str(self.store.db_path))
        try:
            # ── Merge task path: combine multiple level-1 tasks, no LLM ──────
            merge_source_ids = cfg.get("merge_source_task_ids")
            if merge_source_ids:
                from src.common import console

                source_ids = [str(sid) for sid in merge_source_ids]
                console.print(
                    f"\n[bold]Merging {len(source_ids)} level-1 tasks → Task ID {task_uuid_str[:12]}...[/bold]"
                )
                for sid in source_ids:
                    console.print(f"  • [cyan]{sid}[/cyan]")
                merged_count = self.store.merge_classified(con, source_ids, task_uuid_str)
                console.print(
                    f"[bold green]✓ Merge done[/bold green] — [bold]{merged_count}[/bold] labels in merged task"
                )
                summary = _build_summary(con, task_uuid_str, 1)
                return {
                    "status": "success",
                    "message": f"Merged {merged_count} labels from {len(source_ids)} tasks",
                    "total_labeled": merged_count,
                    "level": 1,
                    "merge_source_task_ids": source_ids,
                    **summary,
                }

            if level == 1:
                assert isinstance(config, Level1Config)
                run_level1(
                    con,
                    client,
                    sample_size,
                    config=config,
                    store=self.store,
                    run_id=run_id,
                    task_id=task_uuid_str,
                    seed=seed,
                    checkpoint_fn=checkpoint_fn,
                )
            else:
                assert isinstance(config, Level2Config)
                level1_task_id_str = cfg.get("level1_task_id", "")
                with ExperimentManager() as mgr:
                    level1_task = mgr.get_task(uuid.UUID(level1_task_id_str))
                if level1_task is None:
                    return {
                        "status": "error",
                        "message": f"level1_task_id not found: {level1_task_id_str}",
                        "total_labeled": 0,
                    }
                if level1_task.experiment_id != task.experiment_id:
                    return {
                        "status": "error",
                        "message": (f"level1_task_id {level1_task_id_str} does not belong to the same experiment"),
                        "total_labeled": 0,
                    }
                run_level2(
                    con,
                    client,
                    sample_size,
                    config=config,
                    store=self.store,
                    run_id=run_id,
                    task_id=task_uuid_str,
                    level1_task_id=level1_task_id_str,
                    seed=seed,
                    checkpoint_fn=checkpoint_fn,
                )

            total = con.execute("SELECT COUNT(*) FROM news_classified WHERE task_id = ?", [task_uuid_str]).fetchone()[0]  # type: ignore
            if level == 2:
                total = con.execute(
                    "SELECT COUNT(*) FROM news_sub_classified WHERE level2_task_id = ?", [task_uuid_str]
                ).fetchone()[0]  # type: ignore

            summary = _build_summary(con, task_uuid_str, level)
            return {
                "status": "success",
                "message": f"Task completed: {total} labels created",
                "total_labeled": total,
                "level": level,
                "run_params": {
                    **config.model_dump(),
                    "sample_size": sample_size,
                    "s3_bucket": get_config().labeling.s3_bucket,
                },
                **summary,
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "total_labeled": 0}
        finally:
            con.close()
            _teardown_loki(loki_sink, loki_handler)

    def get_checkpoint_handler(self):
        def get_checkpoints(task_id: str | uuid.UUID) -> list[TaskCheckpoint]:
            with ExperimentManager() as mgr:
                return mgr.get_checkpoints(uuid.UUID(task_id) if isinstance(task_id, str) else task_id)

        return get_checkpoints
