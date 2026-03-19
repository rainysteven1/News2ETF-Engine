"""Schema definitions for the experiment management system."""

from enum import StrEnum


class TaskStatus(StrEnum):
    """Status of a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(StrEnum):
    """Type of task."""

    LABELING = "labeling"
    # Future extensions
    # BACKTEST = "backtest"
    # EVALUATION = "evaluation"
