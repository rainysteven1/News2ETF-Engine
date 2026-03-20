"""
Task registry for managing different task types in the experiment system.

Provides a decorator-based registration mechanism for task executors,
allowing new task types to be added without modifying core code.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class TaskMetadata(BaseModel):
    """Metadata for a task type."""

    name: str
    description: str
    required_params: list[str] | None = None
    optional_params: list[str] | None = None


class TaskExecutor(ABC):
    """Base class for task executors.

    All task types must implement this interface and register themselves
    using the @TaskRegistry.register decorator.
    """

    @property
    @abstractmethod
    def metadata(self) -> TaskMetadata:
        """Return task metadata."""
        pass

    @abstractmethod
    def validate_params(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate task parameters.

        Returns:
            (is_valid, error_message)
        """
        pass

    @abstractmethod
    def execute(self, task: "Task", run_id: str | None = None) -> dict[str, Any]:
        """Execute the task.

        Args:
            task: The task to execute
            run_id: Optional run ID for tracking

        Returns:
            Result dictionary with execution statistics
        """
        pass

    @abstractmethod
    def get_checkpoint_handler(self) -> Any:
        """Return checkpoint handler for resuming execution."""
        pass


class TaskRegistry:
    """Registry for task executors (singleton pattern).

    Manages registration and retrieval of task executors.
    """

    _executors: dict[str, type[TaskExecutor]] = {}
    _metadata: dict[str, TaskMetadata] = {}

    @classmethod
    def register(cls, executor_cls: type[TaskExecutor]) -> type[TaskExecutor]:
        """Decorator to register a task executor.

        Usage:
            @TaskRegistry.register
            class MyTaskExecutor(TaskExecutor):
                ...
        """
        executor = executor_cls()
        metadata = executor.metadata
        cls._executors[metadata.name] = executor_cls
        cls._metadata[metadata.name] = metadata
        return executor_cls

    @classmethod
    def get_executor(cls, name: str) -> TaskExecutor | None:
        """Get an executor instance by task type name."""
        executor_cls = cls._executors.get(name)
        return executor_cls() if executor_cls else None

    @classmethod
    def get_metadata(cls, name: str) -> TaskMetadata | None:
        """Get task metadata by name."""
        return cls._metadata.get(name)

    @classmethod
    def list_all(cls) -> list[TaskMetadata]:
        """List all registered task types."""
        return list(cls._metadata.values())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a task type is registered."""
        return name in cls._metadata

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tasks (useful for testing)."""
        cls._executors.clear()
        cls._metadata.clear()
