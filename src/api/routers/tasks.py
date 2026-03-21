"""Task management endpoints.

All task creation is batch. Each item in the configs list is:
  1. validated: task type is registered, then executor.validate_params() is called
  2. deduped: skipped if any completed task in the database has the same config_hash
  3. created as a pending task if it passes both checks

Long-running executions (POST /{task_id}/run) are dispatched as
BackgroundTasks so the endpoint returns immediately with 202 Accepted.
"""

from __future__ import annotations

import json
import tempfile
import tomllib
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    CheckpointResponse,
    SkippedTaskInfo,
    TaskBatchCreate,
    TaskBatchResponse,
    TaskDetailResponse,
    TaskResponse,
    TaskRunAccepted,
    TaskTypeInfo,
)
from src.common.registry import TaskRegistry
from src.experiment.manager import ExperimentManager
from src.experiment.schema import TaskStatus

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_experiment_id(mgr: ExperimentManager, experiment_name: str | None) -> uuid.UUID | None:
    if experiment_name is None:
        return None
    exp = mgr.get_experiment(experiment_name)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment not found: {experiment_name}")
    return exp.experiment_id


def _find_task(mgr: ExperimentManager, task_id: str):
    all_tasks = mgr.list_tasks()
    for t in all_tasks:
        if str(t.task_id) == task_id or str(t.task_id).startswith(task_id):
            return t
    raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")


def _execute_task(task_id: uuid.UUID, executor: Any) -> None:
    with ExperimentManager() as mgr:
        mgr.update_task_status(task_id, TaskStatus.RUNNING)
    with ExperimentManager() as mgr:
        task = mgr.get_task(task_id)
        if task is None:
            return
    run_id = str(uuid.uuid4())[:8]
    try:
        result = executor.execute(task, run_id=run_id)
        final = TaskStatus.COMPLETED if result.get("status") == "success" else TaskStatus.FAILED
        err = None if final == TaskStatus.COMPLETED else result.get("message", "Unknown error")
        with ExperimentManager() as mgr:
            mgr.update_task_status(task_id, final, result=result, error_msg=err)
    except Exception as exc:
        with ExperimentManager() as mgr:
            mgr.update_task_status(task_id, TaskStatus.FAILED, error_msg=str(exc))


def _validate_config(task_type: str, cfg: dict[str, Any]) -> tuple[bool, str | None]:
    """Two-step validation:
    1. Check task type is registered.
    2. Delegate to the executor's own validate_params.
    """
    executor = TaskRegistry.get_executor(task_type)
    if executor is None:
        return False, f"Unknown task type: '{task_type}'"
    return executor.validate_params(cfg)


def _process_batch(body: TaskBatchCreate) -> TaskBatchResponse:
    """Core batch-create logic. Shared by JSON and file-upload endpoints."""
    # Pre-check: task type must be registered before processing any config
    if not TaskRegistry.is_registered(body.task_type):
        raise HTTPException(status_code=400, detail=f"Unknown task type: {body.task_type}")
    if not body.configs:
        raise HTTPException(status_code=400, detail="configs list must not be empty")

    created: list[TaskResponse] = []
    warnings: list[SkippedTaskInfo] = []

    with ExperimentManager() as mgr:
        experiment_id = _resolve_experiment_id(mgr, body.experiment_name)

        for cfg in body.configs:
            config_hash = ExperimentManager.compute_config_hash(body.task_type, cfg)

            # Step 1 — validate: first check task type, then call executor.validate_params
            ok, err_msg = _validate_config(body.task_type, cfg)
            if not ok:
                warnings.append(
                    SkippedTaskInfo(
                        config=cfg,
                        config_hash=config_hash,
                        reason=f"Validation failed: {err_msg}",
                    )
                )
                continue

            # Step 2 — global dedup: skip if ANY completed task in the DB shares this config_hash
            existing = mgr.find_completed_by_hash(config_hash)
            if existing is not None:
                warnings.append(
                    SkippedTaskInfo(
                        config=cfg,
                        config_hash=config_hash,
                        reason="Duplicate of a completed task",
                        existing_task_id=str(existing.task_id),
                    )
                )
                continue

            # Step 3 — create
            try:
                task = mgr.create_task(
                    task_type=body.task_type,
                    config=cfg,
                    experiment_id=experiment_id,
                )
                created.append(TaskResponse.model_validate(task))
            except ValueError as e:
                warnings.append(
                    SkippedTaskInfo(
                        config=cfg,
                        config_hash=config_hash,
                        reason=str(e),
                    )
                )

    return TaskBatchResponse(
        created=len(created),
        skipped=len(warnings),
        tasks=created,
        warnings=warnings,
    )


# ── Batch create — JSON body ──────────────────────────────────────────────────


@router.post("", response_model=TaskBatchResponse, status_code=201)
def create_tasks(body: TaskBatchCreate) -> TaskBatchResponse:
    """Create tasks from an inline JSON config list."""
    return _process_batch(body)


# ── Batch create — file upload ────────────────────────────────────────────────


@router.post("/upload", response_model=TaskBatchResponse, status_code=201)
async def create_tasks_from_file(
    task_type: str = Form("labeling"),
    experiment_name: str | None = Form(None),
    file: UploadFile = File(..., description="JSON or TOML file containing a list of task configs"),
) -> TaskBatchResponse:
    """Create tasks from an uploaded JSON or TOML config file.

    File format (JSON):
        [{"model": "glm-4-plus", "level": 1, ...}, ...]

    File format (TOML):
        [[task]]
        model = "glm-4-plus"
        level = 1
    """
    content = await file.read()

    filename = file.filename or ""
    try:
        if filename.endswith(".toml"):
            raw = tomllib.loads(content.decode())
            configs: list[dict[str, Any]] = raw.get("task", raw.get("tasks", []))
            if not isinstance(configs, list):
                configs = [configs]
        else:
            parsed = json.loads(content)
            configs = parsed if isinstance(parsed, list) else [parsed]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse config file: {e}")

    return _process_batch(
        TaskBatchCreate(
            experiment_name=experiment_name,
            task_type=task_type,
            configs=configs,
        )
    )


# ── List ──────────────────────────────────────────────────────────────────────


@router.get("", response_model=list[TaskResponse])
def list_tasks(experiment: str | None = None, status: str | None = None) -> list[TaskResponse]:
    with ExperimentManager() as mgr:
        experiment_id = _resolve_experiment_id(mgr, experiment) if experiment else None
        try:
            status_filter = TaskStatus(status) if status else None
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        tasks = mgr.list_tasks(experiment_id=experiment_id, status=status_filter)
        return [TaskResponse.model_validate(t) for t in tasks]


@router.get("/run/{run_id}", response_model=TaskDetailResponse)
def get_task_by_run_id(run_id: str) -> TaskDetailResponse:
    """Look up a task by its run_id (returned in the result after execution)."""
    with ExperimentManager() as mgr:
        task = mgr.find_task_by_run_id(run_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No task found with run_id: {run_id}")
    checkpoints = mgr.get_checkpoints(task.task_id)
    resp = TaskDetailResponse.model_validate(task)
    resp.checkpoints = [CheckpointResponse.model_validate(cp) for cp in checkpoints]
    return resp


# ── Task types ────────────────────────────────────────────────────────────────


@router.get("/types", response_model=list[TaskTypeInfo])
def list_task_types() -> list[TaskTypeInfo]:
    return [
        TaskTypeInfo(
            name=t.name,
            description=t.description,
            required_params=t.required_params or [],
            optional_params=t.optional_params or [],
        )
        for t in TaskRegistry.list_all()
    ]


# ── Detail ────────────────────────────────────────────────────────────────────


@router.get("/{task_id}", response_model=TaskDetailResponse)
def get_task(task_id: str) -> TaskDetailResponse:
    with ExperimentManager() as mgr:
        task = _find_task(mgr, task_id)
        checkpoints = mgr.get_checkpoints(task.task_id)
        resp = TaskDetailResponse.model_validate(task)
        resp.checkpoints = [CheckpointResponse.model_validate(cp) for cp in checkpoints]
        return resp


# ── Run ───────────────────────────────────────────────────────────────────────


@router.post("/{task_id}/run", response_model=TaskRunAccepted, status_code=202)
def run_task(task_id: str, background_tasks: BackgroundTasks) -> TaskRunAccepted:
    with ExperimentManager() as mgr:
        task = _find_task(mgr, task_id)
        tid, ttype, status = task.task_id, task.task_type, task.status

    if status != TaskStatus.PENDING:
        raise HTTPException(status_code=409, detail=f"Task status is '{status}', expected 'pending'")

    executor = TaskRegistry.get_executor(ttype)
    if executor is None:
        raise HTTPException(status_code=400, detail=f"No executor for task type: {ttype}")

    background_tasks.add_task(_execute_task, tid, executor)
    return TaskRunAccepted(task_id=str(tid))


# ── Cancel ────────────────────────────────────────────────────────────────────


@router.post("/{task_id}/cancel", response_model=TaskResponse)
def cancel_task(task_id: str) -> TaskResponse:
    with ExperimentManager() as mgr:
        task = _find_task(mgr, task_id)
        if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            raise HTTPException(status_code=409, detail=f"Cannot cancel task with status '{task.status}'")
        updated = mgr.update_task_status(task.task_id, TaskStatus.CANCELLED)
        return TaskResponse.model_validate(updated)


# ── Export ─────────────────────────────────────────────────────────────────────


@router.get("/{task_id}/export")
def export_task_labels(task_id: str) -> StreamingResponse:
    """Check task exists, then stream its classified records as a parquet file.

    - level=1: exports from news_classified WHERE task_id = ?
    - level=2: exports from news_sub_classified WHERE level2_task_id = ?
    """
    from src.db.store import duckdb_store

    if not duckdb_store.db_path.exists():
        raise HTTPException(status_code=503, detail="DuckDB not initialised.")

    # 1. Verify task exists
    with ExperimentManager() as mgr:
        task = _find_task(mgr, task_id)

    task_type = task.task_type
    level = (task.config or {}).get("level", 1)

    con = duckdb_store.connect(read_only=True)
    try:
        if task_type == "labeling" and level == 1:
            df = con.execute(
                """
                SELECT news_id, title, major_category, confidence, label_source, created_at
                FROM news_classified
                WHERE task_id = ?
                ORDER BY created_at
                """,
                [task_id],
            ).pl()
            filename = f"level1_{task_id}.parquet"
        elif task_type == "labeling" and level == 2:
            df = con.execute(
                """
                SELECT news_id, title, datetime, major_category, sub_category,
                       sentiment, impact_score, confidence, label_source,
                       analysis_logic, key_evidence, expectation,
                       level1_task_id, created_at
                FROM news_sub_classified
                WHERE level2_task_id = ?
                ORDER BY created_at
                """,
                [task_id],
            ).pl()
            filename = f"level2_{task_id}.parquet"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported task_type='{task_type}' level={level} for export.",
            )
    finally:
        con.close()

    if df.is_empty():
        raise HTTPException(status_code=404, detail=f"No records found for task_id: {task_id}")

    tmp = Path(tempfile.gettempdir()) / filename
    df.write_parquet(tmp)

    def iter_file(path: Path, chunk_size: int = 8192):
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk
        Path(path).unlink(missing_ok=True)

    return StreamingResponse(
        iter_file(tmp),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
