"""Pydantic request/response schemas for the News2ETF-Engine API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

# ── Experiments ───────────────────────────────────────────────────────────────


class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None
    task_type: str = "labeling"


class ExperimentResponse(BaseModel):
    experiment_id: UUID
    name: str
    description: str | None
    task_type: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Tasks ─────────────────────────────────────────────────────────────────────


class RunResponse(BaseModel):
    run_id: str
    task_id: UUID
    run_number: int
    status: str
    result: dict[str, Any] | None
    error_msg: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TaskResponse(BaseModel):
    task_id: UUID
    experiment_id: UUID | None
    task_type: str
    config: dict[str, Any]
    config_hash: str
    created_at: datetime
    runs: list[RunResponse] = []

    model_config = {"from_attributes": True}


class CheckpointResponse(BaseModel):
    id: int
    run_id: str
    stage: str
    batch_idx: int
    processed_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class TaskDetailResponse(TaskResponse):
    checkpoints: list[CheckpointResponse] = []


# ── Batch create ──────────────────────────────────────────────────────────────


class TaskBatchCreate(BaseModel):
    """Batch task creation request.

    Each item in `configs` is a full task configuration dict.
    Config resolution: validate → dedup against completed tasks → create.

    Example for labeling tasks:
        {
          "experiment_name": "exp-v1",
          "task_type": "labeling",
          "configs": [
            {"model": "glm-4-plus", "temperature": 0.1, "max_tokens": 8192,
             "level": 1, "sample_size": 5000},
            {"model": "glm-4-plus", "temperature": 0.1, "max_tokens": 8192,
             "level": 2, "sample_size": 5000}
          ]
        }
    """

    experiment_name: str | None = None
    task_type: str = "labeling"
    configs: list[dict[str, Any]]


class SkippedTaskInfo(BaseModel):
    config: dict[str, Any]
    config_hash: str
    reason: str
    existing_task_id: str | None = None


class TaskBatchResponse(BaseModel):
    created: int
    skipped: int
    tasks: list[TaskResponse]
    warnings: list[SkippedTaskInfo] = []


# ── Run / cancel ──────────────────────────────────────────────────────────────


class TaskRunAccepted(BaseModel):
    status: str = "accepted"
    task_id: str
    run_id: str


# ── Task types ────────────────────────────────────────────────────────────────


class TaskTypeInfo(BaseModel):
    name: str
    description: str
    required_params: list[str]
    optional_params: list[str]


# ── Data ──────────────────────────────────────────────────────────────────────


class DataConvertResponse(BaseModel):
    files_converted: int
    message: str


class DataLabelsResponse(BaseModel):
    total: int
    records: list[dict[str, Any]]


# ── Industry ──────────────────────────────────────────────────────────────────


class IndustryBuildRequest(BaseModel):
    model: str = "glm-4-flash"


class IndustryBuildResponse(BaseModel):
    status: str
    message: str
