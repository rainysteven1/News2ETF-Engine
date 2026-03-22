"""
SQLAlchemy ORM models for the experiment management system.

Uses PostgreSQL with JSONB support for flexible configuration storage.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column, relationship

Base = declarative_base()


class Experiment(Base):
    """Experiments table - groups multiple tasks of the same type."""

    __tablename__ = "experiments"

    experiment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, default="labeling", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(), onupdate=lambda: datetime.now()
    )

    # Relationships
    tasks = relationship("Task", back_populates="experiment", cascade="all, delete-orphan")


class Run(Base):
    """Runs table - stores execution instances of a task."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(8), primary_key=True)  # 8-char, used for Loki labels
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now())

    # Relationships
    task: Mapped["Task"] = relationship("Task", back_populates="runs")
    checkpoints = relationship("TaskCheckpoint", back_populates="run", cascade="all, delete-orphan")
    history = relationship("TaskHistory", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Run(id={self.run_id}, task={str(self.task_id)[:8]}..., status={self.status})>"


class Task(Base):
    """Tasks table - stores immutable task configuration templates."""

    __tablename__ = "tasks"

    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.experiment_id", ondelete="SET NULL"), nullable=True
    )
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now())

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="tasks")
    runs = relationship("Run", back_populates="task", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Task(id={str(self.task_id)[:8]}..., type={self.task_type})>"


class TaskCheckpoint(Base):
    """Task checkpoints table - stores progress for resume capability."""

    __tablename__ = "task_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(8), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'keyword', 'llm', 'validation'
    batch_idx: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checkpoint_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now())

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="checkpoints")

    def __repr__(self) -> str:
        return f"<TaskCheckpoint(id={self.id}, run={self.run_id}, stage={self.stage})>"


class TaskHistory(Base):
    """Task history table - tracks task lifecycle events."""

    __tablename__ = "task_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(8), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'started', 'completed', 'failed'
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now())

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="history")

    def __repr__(self) -> str:
        return f"<TaskHistory(id={self.id}, run={self.run_id}, action={self.action})>"


class ParamSchema(Base):
    """Parameter schema definitions — stored per task_type."""

    __tablename__ = "param_schemas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    param_name: Mapped[str] = mapped_column(String(64), nullable=False)
    param_type: Mapped[str] = mapped_column(String(16), nullable=False)  # string | int | float | bool | list
    required: Mapped[bool] = mapped_column(default=False, nullable=False)
    default_val: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    options: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list
    min_val: Mapped[float | None] = mapped_column(nullable=True)
    max_val: Mapped[float | None] = mapped_column(nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    conditions_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(), onupdate=lambda: datetime.now()
    )

    __table_args__ = (Index("ix_param_schemas_task_type_param_name", "task_type", "param_name", unique=True),)

    def __repr__(self) -> str:
        return f"<ParamSchema(task_type={self.task_type}, param={self.param_name})>"


class ParamValidationRule(Base):
    """Cross-parameter validation rules — stored per task_type."""

    __tablename__ = "param_validation_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    rule_name: Mapped[str] = mapped_column(String(64), nullable=False)
    hook: Mapped[str] = mapped_column(String(64), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now())

    __table_args__ = (Index("ix_param_validation_rules_task_type_rule_name", "task_type", "rule_name", unique=True),)

    def __repr__(self) -> str:
        return f"<ParamValidationRule(task_type={self.task_type}, rule={self.rule_name})>"
