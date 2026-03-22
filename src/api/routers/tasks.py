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
    RunResponse,
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


def _execute_task(run_id: str, executor: Any) -> None:
    """Execute a task run. Updates Run status, not Task status."""
    try:
        with ExperimentManager() as mgr:
            run = mgr.get_run(run_id)
            if run is None:
                return
            task = mgr.get_task(run.task_id)
            if task is None:
                return

        result = executor.execute(task, run_id=run.run_id)
        final = TaskStatus.COMPLETED if result.get("status") == "success" else TaskStatus.FAILED
        err = None if final == TaskStatus.COMPLETED else result.get("message", "Unknown error")
        # Extract result (distribution data) and summary (LLMUsage) separately
        result_data = result.get("result")
        summary = result.get("summary")
        with ExperimentManager() as mgr:
            mgr.update_run(run_id, final, result=result_data, summary=summary, error_msg=err)
    except Exception as exc:
        with ExperimentManager() as mgr:
            mgr.update_run(run_id, TaskStatus.FAILED, error_msg=str(exc))


def _validate_config(task_type: str, cfg: dict[str, Any]) -> tuple[bool, str | None]:
    """Two-step validation:
    1. Check task type is registered.
    2. Delegate to the executor's own validate_params.

    Raises HTTPException 400 on validation failure.
    """
    executor = TaskRegistry.get_executor(task_type)
    if executor is None:
        raise HTTPException(status_code=400, detail=f"Unknown task type: '{task_type}'")
    ok, err_msg = executor.validate_params(cfg)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Validation failed: {err_msg}")
    return True, None


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

            # Step 1 — validate: raises HTTPException 400 on failure
            _validate_config(body.task_type, cfg)

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
def list_tasks(experiment: str | None = None) -> list[TaskResponse]:
    with ExperimentManager() as mgr:
        experiment_id = _resolve_experiment_id(mgr, experiment) if experiment else None
        tasks = mgr.list_tasks(experiment_id=experiment_id)
        return [TaskResponse.model_validate(t) for t in tasks]


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
        runs = mgr.get_task_runs(task.task_id)
        resp = TaskResponse.model_validate(task)
        resp.runs = [RunResponse.model_validate(r) for r in runs]
        # Build detail response with checkpoints from all runs
        all_checkpoints = []
        for run in runs:
            checkpoints = mgr.get_checkpoints(run.run_id)
            for cp in checkpoints:
                cp_resp = CheckpointResponse.model_validate(cp)
                all_checkpoints.append(cp_resp)
        return TaskDetailResponse(**resp.model_dump(), checkpoints=all_checkpoints)


# ── Run ───────────────────────────────────────────────────────────────────────


@router.post("/{task_id}/run", response_model=TaskRunAccepted, status_code=202)
def run_task(task_id: str, background_tasks: BackgroundTasks) -> TaskRunAccepted:
    with ExperimentManager() as mgr:
        task = _find_task(mgr, task_id)

    executor = TaskRegistry.get_executor(task.task_type)
    if executor is None:
        raise HTTPException(status_code=400, detail=f"No executor for task type: {task.task_type}")

    with ExperimentManager() as mgr:
        run = mgr.create_run(task.task_id)

    background_tasks.add_task(_execute_task, run.run_id, executor)
    return TaskRunAccepted(task_id=str(task.task_id), run_id=run.run_id)


# ── Run endpoints ─────────────────────────────────────────────────────────────


@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str) -> RunResponse:
    """Get a run by its 8-char run_id."""
    with ExperimentManager() as mgr:
        run = mgr.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return RunResponse.model_validate(run)


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
def cancel_run(run_id: str) -> RunResponse:
    """Cancel a running run."""
    with ExperimentManager() as mgr:
        run = mgr.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        if run.status not in (TaskStatus.PENDING.value, TaskStatus.RUNNING.value):
            raise HTTPException(status_code=409, detail=f"Cannot cancel run with status '{run.status}'")
        updated = mgr.update_run(run_id, TaskStatus.CANCELLED)
    return RunResponse.model_validate(updated)


@router.post("/runs/{run_id}/restart", response_model=TaskRunAccepted, status_code=202)
def restart_run(run_id: str, background_tasks: BackgroundTasks) -> TaskRunAccepted:
    """Restart a run by creating a new run with the same task configuration."""
    with ExperimentManager() as mgr:
        old_run = mgr.get_run(run_id)
        if old_run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        task = mgr.get_task(old_run.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found for run: {run_id}")

    executor = TaskRegistry.get_executor(task.task_type)
    if executor is None:
        raise HTTPException(status_code=400, detail=f"No executor for task type: {task.task_type}")

    with ExperimentManager() as mgr:
        new_run = mgr.restart_run(run_id)

    background_tasks.add_task(_execute_task, new_run.run_id, executor)
    return TaskRunAccepted(task_id=str(task.task_id), run_id=new_run.run_id)


# ── Run result distribution ────────────────────────────────────────────────────


@router.get("/runs/{run_id}/distribution")
def get_distribution(run_id: str) -> dict:
    """Get classification distribution for a completed run.

    Returns the distribution of labels from news_classified (level-1) or
    news_sub_classified (level-2) depending on the task configuration.
    Only returns results when run status is COMPLETED.
    """
    from src.db.store import store

    # 1. Get run and task info
    with ExperimentManager() as mgr:
        run = mgr.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        if run.status != TaskStatus.COMPLETED.value:
            raise HTTPException(
                status_code=409,
                detail=f"Run is not completed. Current status: {run.status}",
            )
        task = mgr.get_task(run.task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found for run: {run_id}")

    task_type = task.task_type
    cfg = task.config or {}
    level = cfg.get("level", 1)
    task_uuid_str = str(task.task_id)

    if task_type == "labeling" and level == 1:
        rows = store.execute(
            """
            SELECT major_category, label_source, COUNT(*) AS cnt,
                   round(AVG(confidence), 3) AS avg_conf
            FROM news_classified
            WHERE task_id = %(task_id)s AND run_id = %(run_id)s
            GROUP BY major_category, label_source
            ORDER BY major_category, label_source
            """,
            {"task_id": task_uuid_str, "run_id": run_id},
        )

        if not rows:
            raise HTTPException(status_code=404, detail=f"No level-1 results found for run_id: {run_id}")

        by_source: dict = {}
        by_major: dict = {}
        total = 0
        for major, source, cnt, avg_conf in rows:
            total += cnt
            by_source[source] = by_source.get(source, 0) + cnt
            by_major.setdefault(major or "unknown", {})[source] = {
                "count": cnt,
                "avg_confidence": avg_conf,
            }

        return {
            "run_id": run_id,
            "task_id": task_uuid_str,
            "level": 1,
            "total": total,
            "by_label_source": by_source,
            "by_major_category": by_major,
        }

    elif task_type == "labeling" and level == 2:
        rows = store.execute(
            """
            SELECT major_category, sub_category, label_source,
                   sentiment, COUNT(*) AS cnt,
                   round(AVG(confidence), 3) AS avg_conf
            FROM news_sub_classified
            WHERE level2_task_id = %(task_id)s AND run_id = %(run_id)s
            GROUP BY major_category, sub_category, label_source, sentiment
            ORDER BY major_category, sub_category
            """,
            {"task_id": task_uuid_str, "run_id": run_id},
        )

        if not rows:
            raise HTTPException(status_code=404, detail=f"No level-2 results found for run_id: {run_id}")

        by_source: dict = {}
        by_sub: dict = {}
        by_sentiment: dict = {}
        total = 0
        for major, sub, source, sentiment, cnt, avg_conf in rows:
            total += cnt
            by_source[source] = by_source.get(source, 0) + cnt
            key = f"{major} / {sub}"
            by_sub.setdefault(key, {})[source] = {"count": cnt, "avg_confidence": avg_conf}
            if sentiment:
                by_sentiment[sentiment] = by_sentiment.get(sentiment, 0) + cnt

        return {
            "run_id": run_id,
            "task_id": task_uuid_str,
            "level": 2,
            "total": total,
            "by_label_source": by_source,
            "by_sub_category": by_sub,
            "by_sentiment": by_sentiment,
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task_type='{task_type}' level={level} for result.",
        )


# ── Export ─────────────────────────────────────────────────────────────────────


@router.get("/{task_id}/export")
def export_task_labels(task_id: str) -> StreamingResponse:
    """Check task exists, then stream its classified records as a parquet file.

    - level=1: exports from news_classified WHERE task_id = ?
    - level=2: exports from news_sub_classified WHERE level2_task_id = ?
    """
    import polars as pl

    from src.db.store import store

    # 1. Verify task exists
    with ExperimentManager() as mgr:
        task = _find_task(mgr, task_id)

    task_type = task.task_type
    level = (task.config or {}).get("level", 1)

    if task_type == "labeling" and level == 1:
        rows = store.execute(
            """
            SELECT news_id, title, major_category, confidence, label_source, created_at
            FROM news_classified
            WHERE task_id = %(task_id)s
            ORDER BY created_at
            """,
            {"task_id": task_id},
        )
        if rows:
            df = pl.DataFrame(
                rows,
                schema=["news_id", "title", "major_category", "confidence", "label_source", "created_at"],
            )
        else:
            df = pl.DataFrame(schema=["news_id", "title", "major_category", "confidence", "label_source", "created_at"])
        filename = f"level1_{task_id}.parquet"
    elif task_type == "labeling" and level == 2:
        rows = store.execute(
            """
            SELECT news_id, title, datetime, major_category, sub_category,
                   sentiment, impact_score, confidence, label_source,
                   analysis_logic, key_evidence, expectation,
                   level1_task_id, created_at
            FROM news_sub_classified
            WHERE level2_task_id = %(task_id)s
            ORDER BY created_at
            """,
            {"task_id": task_id},
        )
        if rows:
            df = pl.DataFrame(
                rows,
                schema=[
                    "news_id",
                    "title",
                    "datetime",
                    "major_category",
                    "sub_category",
                    "sentiment",
                    "impact_score",
                    "confidence",
                    "label_source",
                    "analysis_logic",
                    "key_evidence",
                    "expectation",
                    "level1_task_id",
                    "created_at",
                ],
            )
        else:
            df = pl.DataFrame(
                schema=[
                    "news_id",
                    "title",
                    "datetime",
                    "major_category",
                    "sub_category",
                    "sentiment",
                    "impact_score",
                    "confidence",
                    "label_source",
                    "analysis_logic",
                    "key_evidence",
                    "expectation",
                    "level1_task_id",
                    "created_at",
                ],
            )
        filename = f"level2_{task_id}.parquet"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task_type='{task_type}' level={level} for export.",
        )

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
