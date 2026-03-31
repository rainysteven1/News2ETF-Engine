"""
Task registry for managing different task types in the experiment system.

Provides a decorator-based registration mechanism for task executors,
allowing new task types to be added without modifying core code.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Any

from pydantic import BaseModel

from src.common.param_metadata import TaskParamSchema
from src.common.param_validator import ParamValidator
from src.db.models import Task


class TaskMetadata(BaseModel):
    """Metadata for a task type."""

    name: str
    description: str
    required_params: list[str] | None = None
    optional_params: list[str] | None = None
    param_schema: TaskParamSchema | None = None

    @property
    def param_validator(self) -> ParamValidator | None:
        """Build a ParamValidator from the embedded schema, if available."""
        if self.param_schema is None:
            return None
        # Import here to avoid circular dependency at module load time
        from src.common.param_validator import ParamValidator

        return ParamValidator(self.param_schema)


class TaskExecutor(ABC):
    """Base class for task executors.

    All task types must implement this interface and register themselves
    using the @TaskRegistry.register decorator.
    """

    # Set this on each subclass so TaskRegistry.register can defer metadata access
    task_type_name: str

    @cached_property
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
    def execute(self, task: Task, run_id: str) -> dict[str, Any]:
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
    _metadata_factories: dict[str, Callable[[], TaskMetadata]] = {}

    @classmethod
    def register(cls, executor_cls: type[TaskExecutor]) -> type[TaskExecutor]:
        """Decorator to register a task executor.

        Usage:
            @TaskRegistry.register
            class MyTaskExecutor(TaskExecutor):
                ...
        """
        executor = executor_cls()
        # Defer all executor.metadata access — it hits the DB and is not needed at import time.
        # We need the name for registration; assume the executor declares it statically.
        task_name = getattr(executor, "task_type_name", None)
        if task_name is None:
            raise AttributeError(
                f"{executor_cls.__name__} must define `task_type_name: str` class attribute "
                "for lazy registry registration"
            )

        def factory():
            return executor.metadata

        cls._executors[task_name] = executor_cls
        cls._metadata_factories[task_name] = factory
        return executor_cls

    @classmethod
    def get_executor(cls, name: str) -> TaskExecutor | None:
        """Get an executor instance by task type name."""
        executor_cls = cls._executors.get(name)
        return executor_cls() if executor_cls else None

    @classmethod
    def get_metadata(cls, name: str) -> TaskMetadata | None:
        """Get task metadata by name (computed lazily on first access)."""
        factory = cls._metadata_factories.get(name)
        return factory() if factory else None

    @classmethod
    def list_all(cls) -> list[TaskMetadata]:
        """List all registered task types."""
        return [factory() for factory in cls._metadata_factories.values()]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a task type is registered."""
        return name in cls._metadata_factories

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tasks (useful for testing)."""
        cls._executors.clear()
        cls._metadata_factories.clear()
