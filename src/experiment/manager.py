"""
Experiment manager for handling experiments and tasks.

Uses SQLAlchemy ORM for database operations.
"""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db import Experiment, Task, TaskCheckpoint, TaskHistory, get_session
from src.db.session import get_session_sync
from src.experiment.schema import TaskStatus


class ExperimentManager:
    """Manager for experiment and task operations (singleton pattern)."""

    _instance: "ExperimentManager" = None
    _session: Session | None = None
    _own_session: bool = False

    def __new__(cls, session: Session | None = None) -> "ExperimentManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._session = session
            cls._instance._own_session = session is None
        return cls._instance

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = get_session_sync()
        return self._session

    def close(self):
        if self._own_session and self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== Experiment Operations ====================

    def create_experiment(self, name: str, description: str | None = None, task_type: str = "labeling") -> Experiment:
        experiment = Experiment(
            name=name,
            description=description,
            task_type=task_type,
        )
        self.session.add(experiment)
        self.session.commit()
        self.session.refresh(experiment)
        return experiment

    def get_experiment(self, name: str | uuid.UUID) -> Experiment | None:
        if isinstance(name, uuid.UUID):
            stmt = select(Experiment).where(Experiment.experiment_id == name)
        else:
            stmt = select(Experiment).where(Experiment.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_experiments(self) -> list[Experiment]:
        stmt = select(Experiment).order_by(Experiment.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    # ==================== Config Hash ====================

    @staticmethod
    def compute_config_hash(task_type: str, config: dict[str, Any]) -> str:
        """Compute a deterministic MD5 hash for deduplication."""
        payload = json.dumps({"task_type": task_type, **config}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def find_completed_by_hash(self, config_hash: str) -> Task | None:
        """Return any completed task globally matching this config_hash."""
        stmt = select(Task).where(
            Task.config_hash == config_hash,
            Task.status == TaskStatus.COMPLETED,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    # ==================== Task Operations ====================

    def create_task(
        self,
        task_type: str,
        config: dict[str, Any],
        experiment_id: uuid.UUID | None = None,
    ) -> Task:
        """Create a new pending task.

        Raises ValueError if the experiment enforces a different task_type.
        Does NOT check for duplicates — callers should call find_completed_by_hash first.
        """
        if experiment_id is not None:
            exp = self.session.get(Experiment, experiment_id)
            if exp is not None and exp.task_type != task_type:
                raise ValueError(f"Experiment '{exp.name}' only accepts '{exp.task_type}' tasks, got '{task_type}'")

        config_hash = self.compute_config_hash(task_type, config)

        task = Task(
            experiment_id=experiment_id,
            task_type=task_type,
            config=config,
            config_hash=config_hash,
            status=TaskStatus.PENDING,
        )
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task

    def get_task(self, task_id: uuid.UUID | str) -> Task | None:
        if isinstance(task_id, str):
            task_id = uuid.UUID(task_id)
        return self.session.get(Task, task_id)

    def list_tasks(
        self,
        experiment_id: uuid.UUID | None = None,
        status: TaskStatus | None = None,
    ) -> list[Task]:
        stmt = select(Task)
        if experiment_id:
            stmt = stmt.where(Task.experiment_id == experiment_id)
        if status:
            stmt = stmt.where(Task.status == (status.value if isinstance(status, TaskStatus) else status))
        stmt = stmt.order_by(Task.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def update_task_status(
        self,
        task_id: uuid.UUID,
        status: TaskStatus,
        result: dict[str, Any] | None = None,
        error_msg: str | None = None,
    ) -> Task:
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Task not found: {task_id}")

        task.status = status.value if isinstance(status, TaskStatus) else status
        if result is not None:
            task.result = result
        if error_msg is not None:
            task.error_msg = error_msg

        from datetime import datetime

        if status == TaskStatus.RUNNING and task.started_at is None:
            task.started_at = datetime.now()
        elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.finished_at = datetime.now()

        self.session.commit()
        self.session.refresh(task)
        return task

    # ==================== Checkpoint Operations ====================

    def save_checkpoint(
        self,
        task_id: uuid.UUID,
        stage: str,
        batch_idx: int,
        processed_count: int,
        checkpoint_data: dict[str, Any] | None = None,
    ) -> TaskCheckpoint:
        checkpoint = TaskCheckpoint(
            task_id=task_id,
            stage=stage,
            batch_idx=batch_idx,
            processed_count=processed_count,
            checkpoint_data=checkpoint_data,
        )
        self.session.add(checkpoint)
        self.session.commit()
        self.session.refresh(checkpoint)
        return checkpoint

    def get_checkpoints(self, task_id: uuid.UUID) -> list[TaskCheckpoint]:
        stmt = select(TaskCheckpoint).where(TaskCheckpoint.task_id == task_id).order_by(TaskCheckpoint.created_at)
        return list(self.session.execute(stmt).scalars().all())

    # ==================== History Operations ====================

    def record_task_history(self, task_id: uuid.UUID, action: str, detail: dict[str, Any] | None = None) -> TaskHistory:
        history = TaskHistory(task_id=task_id, action=action, detail=detail)
        self.session.add(history)
        self.session.commit()
        return history

    def get_task_history(self, task_id: uuid.UUID) -> list[TaskHistory]:
        stmt = select(TaskHistory).where(TaskHistory.task_id == task_id).order_by(TaskHistory.created_at)
        return list(self.session.execute(stmt).scalars().all())
