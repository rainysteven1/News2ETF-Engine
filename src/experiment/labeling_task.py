"""
Labeling task executor for the experiment system.

Orchestrates the labeling pipeline: owns ClickHouse store, Loki logging,
and checkpoint management. Passes all dependencies into the pure
business-logic functions in src.labeling.
"""

from functools import cached_property
from typing import Any

from loguru import logger

from src.common import get_config
from src.common.param_metadata import TaskParamSchema
from src.common.registry import TaskExecutor, TaskMetadata, TaskRegistry
from src.db import Task
from src.db.models import TaskCheckpoint
from src.db.store import store as clickhouse_store
from src.experiment.manager import ExperimentManager
from src.labeling import Level1Config, Level2Config, run_level1, run_level2
from src.utils.llm_client import get_llm_client
from src.utils.loki_sink import LokiSink


def _setup_loki(run_id: str | None, level_stage: str) -> tuple[LokiSink | None, int | None]:
    """Create a Loki sink and add it to loguru. Returns (sink, handler_id).

    Also raises the default stderr handler to INFO so that DEBUG messages
    (e.g. per-sample titles) only flow to Loki, not the console.
    """
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


def _build_summary(store, task_uuid_str: str, level: int) -> dict:
    """Query store and build a structured result summary for the task record."""
    if level == 1:
        rows = store.execute(
            """
            SELECT major_category, label_source, COUNT(*) AS cnt,
                   round(AVG(confidence), 3) AS avg_conf
            FROM news_classified
            WHERE task_id = %(task_id)s
            GROUP BY major_category, label_source
            ORDER BY major_category, label_source
            """,
            {"task_id": task_uuid_str},
        )
        by_source = {}
        by_major = {}
        for row in rows:
            major, source, cnt, avg_conf = row
            by_source[source] = by_source.get(source, 0) + cnt
            by_major.setdefault(major or "unknown", {})[source] = {"count": cnt, "avg_confidence": avg_conf}
        return {
            "by_label_source": by_source,
            "by_major_category": by_major,
        }
    else:
        rows = store.execute(
            """
            SELECT major_category, sub_category, label_source,
                   sentiment, COUNT(*) AS cnt,
                   round(AVG(confidence), 3) AS avg_conf
            FROM news_sub_classified
            WHERE level2_task_id = %(task_id)s
            GROUP BY major_category, sub_category, label_source, sentiment
            ORDER BY major_category, sub_category
            """,
            {"task_id": task_uuid_str},
        )
        by_source: dict = {}
        by_sub: dict = {}
        by_sentiment: dict = {}
        for row in rows:
            major, sub, source, sentiment, cnt, avg_conf = row
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
      - ClickHouseStore for data persistence
      - Loki logging lifecycle
      - Checkpoint handler (via ExperimentManager / PostgreSQL)
    """

    task_type_name = "labeling"

    def __init__(self) -> None:
        self.store = clickhouse_store

    @cached_property
    def metadata(self) -> TaskMetadata:
        schema = TaskParamSchema.from_db("labeling")
        return TaskMetadata(
            name="labeling",
            description="News labeling task with hierarchical classification",
            required_params=schema.required_params,
            optional_params=schema.optional_params,
            param_schema=schema,
        )

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        validator = self.metadata.param_validator
        assert validator is not None, "Param schema should always have a validator function"
        return validator.validate(params)

    # ── checkpoint callback ─────────────────────

    def _make_checkpoint_fn(self, run_id: str):
        """Return a checkpoint callback that persists via ExperimentManager."""

        def _save(loki_run_id: str | None, stage: str, batch_idx: int, saved_count: int) -> None:
            with ExperimentManager() as mgr:
                mgr.save_checkpoint(
                    run_id=run_id,
                    stage=stage,
                    batch_idx=batch_idx,
                    processed_count=saved_count,
                )

        return _save

    # ── execute ─────────────────────────────────

    def execute(self, task: Task, run_id: str | None = None) -> dict[str, Any]:
        cfg = task.config or {}

        level = cfg.get("level", 1)
        labeling_defaults = get_config().labeling
        _common = dict(
            model=cfg.get("model", "glm-4-flash"),
            temperature=cfg.get("temperature", 0.1),
            max_tokens=cfg.get("max_tokens"),
            checkpoint_every=cfg.get("checkpoint_every", labeling_defaults.checkpoint_every),
            llm_retry=cfg.get("llm_retry", labeling_defaults.llm_retry),
            seed=cfg.get("seed", None),
            batch_size=cfg.get("batch_size", labeling_defaults.batch_size),
            sample_size=cfg.get("sample_size", None),
        )

        if level == 1:
            config = Level1Config(
                **_common,
                start=cfg.get("start", 0),
            )
        else:
            config = Level2Config(
                **_common,
                level1_task_id=cfg.get("level1_task_id", ""),
                major_categories=cfg.get("major_categories"),
                concurrency=cfg.get("concurrency", 1),
            )

        try:
            client = get_llm_client(config.model)
        except ValueError as e:
            return {"status": "error", "message": str(e), "total_labeled": 0}

        task_uuid_str = str(task.task_id)

        # Set up infrastructure
        self.store.init_db()
        loki_sink, loki_handler = _setup_loki(run_id, level_stage=f"level{level}")

        assert run_id is not None, "run_id must be provided for labeling tasks"
        checkpoint_fn = self._make_checkpoint_fn(run_id)

        try:
            if level == 1:
                assert isinstance(config, Level1Config)
                run_level1(
                    client,
                    config=config,
                    store=self.store,
                    run_id=run_id,
                    task_id=task_uuid_str,
                    checkpoint_fn=checkpoint_fn,
                )
            else:
                assert isinstance(config, Level2Config)
                run_level2(
                    client,
                    config=config,
                    store=self.store,
                    run_id=run_id,
                    task_id=task_uuid_str,
                    checkpoint_fn=checkpoint_fn,
                )

            # Get total count
            total = self.store.execute(
                "SELECT COUNT(*) FROM news_classified WHERE task_id = %(task_id)s",
                {"task_id": task_uuid_str},
            )[0][0]
            if level == 2:
                total = self.store.execute(
                    "SELECT COUNT(*) FROM news_sub_classified WHERE level2_task_id = %(task_id)s",
                    {"task_id": task_uuid_str},
                )[0][0]

            summary = _build_summary(self.store, task_uuid_str, level)
            return {
                "status": "success",
                "message": f"Task completed: {total} labels created",
                "total_labeled": total,
                "level": level,
                "run_id": run_id,
                "run_params": {
                    **config.model_dump(),
                    "s3_bucket": get_config().labeling.s3_bucket,
                },
                **summary,
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "total_labeled": 0, "run_id": run_id}
        finally:
            _teardown_loki(loki_sink, loki_handler)

    def get_checkpoint_handler(self):
        def get_checkpoints(run_id: str) -> list[TaskCheckpoint]:
            with ExperimentManager() as mgr:
                return mgr.get_checkpoints(run_id)

        return get_checkpoints
