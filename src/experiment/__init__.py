"""Experiment management module for News2ETF-Engine."""

# Import task executors to register them
# Remove the import if you don't want a task to be available
from src.experiment import labeling_task  # noqa: F401
from src.experiment.manager import ExperimentManager
from src.experiment.schema import TaskStatus, TaskType

__all__ = ["TaskStatus", "TaskType", "ExperimentManager"]
