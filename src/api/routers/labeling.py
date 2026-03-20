"""Labeling-specific endpoints.

Provides a dedicated merge endpoint to combine multiple level-1 labeling
tasks into a single unified task without re-running LLM inference.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, field_validator

from src.experiment.manager import ExperimentManager
from src.experiment.schema import TaskStatus

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────


class MergeTaskCreate(BaseModel):
    """Request body for merging multiple level-1 tasks.

    source_task_ids: list of completed level-1 labeling task UUIDs to merge.
    experiment_name: optional — the merged task will be associated with this
                     experiment (must already exist).  If omitted, the merged
                     task has no experiment.
    """

    source_task_ids: list[str]
    experiment_name: str | None = None

    @field_validator("source_task_ids")
    @classmethod
    def at_least_one(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("source_task_ids must not be empty")
        for sid in v:
            try:
                uuid.UUID(sid)
            except ValueError:
                raise ValueError(f"Invalid UUID: {sid}")
        return v


class MergeTaskResponse(BaseModel):
    task_id: str
    status: str
    source_task_ids: list[str]
    total_labeled: int | None = None
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_sources(source_task_ids: list[str]) -> tuple[list[str], str | None]:
    """Validate that all source tasks exist, are level-1 labeling tasks, and share
    the same model/temperature config.  Returns (source_ids, error_message)."""
    root_cfg: dict | None = None
    with ExperimentManager() as mgr:
        for sid in source_task_ids:
            task = mgr.get_task(uuid.UUID(sid))
            if task is None:
                return [], f"Source task not found: {sid}"
            if task.task_type != "labeling":
                return [], f"Source task {sid} is not a labeling task"
            src_cfg = task.config or {}
            if src_cfg.get("level", 1) != 1:
                return [], f"Source task {sid} is not a level-1 task"
            compat_keys = ["model", "temperature"]
            if root_cfg is None:
                root_cfg = {k: src_cfg.get(k) for k in compat_keys}
            else:
                for k in compat_keys:
                    if src_cfg.get(k) != root_cfg[k]:
                        return [], (
                            f"Source task {sid} has incompatible config: "
                            f"{k}={src_cfg.get(k)!r} vs expected {root_cfg[k]!r}"
                        )
    return source_task_ids, None


def _resolve_experiment_id(experiment_name: str | None) -> uuid.UUID | None:
    if experiment_name is None:
        return None
    with ExperimentManager() as mgr:
        exp = mgr.get_experiment(experiment_name)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_name}")
    return exp.experiment_id


def _run_merge(task_id: uuid.UUID, source_task_ids: list[str]) -> None:
    """Background worker: executes the merge and updates task status."""
    import duckdb

    from src.db.store import duckdb_store
    from src.experiment.manager import ExperimentManager

    with ExperimentManager() as mgr:
        mgr.update_task_status(task_id, TaskStatus.RUNNING)

    try:
        duckdb_store.init_db()
        task_uuid_str = str(task_id)
        con = duckdb.connect(str(duckdb_store.db_path))
        try:
            merged_count = duckdb_store.merge_classified(con, source_task_ids, task_uuid_str)
        finally:
            con.close()

        result: dict[str, Any] = {
            "status": "success",
            "message": f"Merged {merged_count} labels from {len(source_task_ids)} tasks",
            "total_labeled": merged_count,
            "level": 1,
            "merge_source_task_ids": source_task_ids,
        }
        with ExperimentManager() as mgr:
            mgr.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
    except Exception as exc:
        with ExperimentManager() as mgr:
            mgr.update_task_status(task_id, TaskStatus.FAILED, error_msg=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/merge",
    response_model=MergeTaskResponse,
    status_code=202,
    summary="Merge multiple level-1 labeling tasks",
    description=(
        "Copies all level-1 labels from the given source tasks into a new unified task. "
        "Duplicate news_ids are silently skipped. All source tasks must share the same "
        "model and temperature. The merge runs in the background; poll the returned "
        "task_id via GET /tasks/{task_id} for completion status."
    ),
)
async def merge_labeling_tasks(
    body: MergeTaskCreate,
    background_tasks: BackgroundTasks,
) -> MergeTaskResponse:
    source_ids, err = _validate_sources(body.source_task_ids)
    if err:
        raise HTTPException(status_code=422, detail=err)

    experiment_id = _resolve_experiment_id(body.experiment_name)

    merge_cfg: dict[str, Any] = {
        "task_type": "labeling",
        "level": 1,
        "merge_source_task_ids": source_ids,
    }

    with ExperimentManager() as mgr:
        task = mgr.create_task(
            task_type="labeling",
            config=merge_cfg,
            experiment_id=experiment_id,
        )

    background_tasks.add_task(_run_merge, task.task_id, source_ids)

    return MergeTaskResponse(
        task_id=str(task.task_id),
        status="accepted",
        source_task_ids=source_ids,
        message=f"Merge of {len(source_ids)} tasks enqueued — poll /tasks/{task.task_id} for status",
    )
