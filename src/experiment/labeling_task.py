"""
Labeling task executor for the experiment system.

Orchestrates the labeling pipeline: owns DuckDBStore, Loki logging,
and checkpoint management. Passes all dependencies into the pure
business-logic functions in src.labeling.
"""

import os
import uuid
from typing import Any

import duckdb
from loguru import logger
from zhipuai import ZhipuAI

from src.common import get_config
from src.common.registry import TaskExecutor, TaskMetadata, TaskRegistry
from src.db import Task
from src.db.models import TaskCheckpoint
from src.db.store import DuckDBStore, duckdb_store
from src.experiment.manager import ExperimentManager
from src.labeling import run_level1, run_level2
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
            required_params=["model", "temperature", "max_tokens"],
            optional_params=["level", "sample_size", "force_relabel"],
        )

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        if "model" not in params:
            return False, "Missing required param: model"
        if "temperature" not in params:
            return False, "Missing required param: temperature"
        if "max_tokens" not in params:
            return False, "Missing required param: max_tokens"

        level = params.get("level", 1)
        if level not in [1, 2]:
            return False, f"Invalid level: {level}, must be 1 or 2"

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
        config = {
            "model": cfg.get("model", "glm-4.5-airx" if level == 1 else "glm-4.7-flashx"),
            "temperature": cfg.get("temperature", 0.1),
            "max_tokens": cfg.get("max_tokens", 8192),
        }

        api_key = os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            return {"status": "error", "message": "ZHIPU_API_KEY not set", "total_labeled": 0}

        client = ZhipuAI(api_key=api_key)
        task_uuid_str = str(task.task_id)

        # Set up infrastructure
        self.store.init_db()
        loki_sink, loki_handler = _setup_loki(run_id, level_stage=f"level{level}")
        checkpoint_fn = self._make_checkpoint_fn(task.task_id)

        seed = cfg.get("seed", None)

        con = duckdb.connect(str(self.store.db_path))
        try:
            fn = run_level1 if level == 1 else run_level2
            fn(
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

            total = con.execute("SELECT COUNT(*) FROM news_classified WHERE task_id = ?", [task_uuid_str]).fetchone()[0]  # type: ignore

            return {
                "status": "success",
                "message": f"Task completed: {total} labels created",
                "total_labeled": total,
                "level": level,
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
