"""Auto-generated parameter validation backed by TaskParamSchema."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from src.common.param_metadata import ParamMetadata, TaskParamSchema, ValidationRule

# ── Hook registry ─────────────────────────────────────────────────────────────

HookFunc = Callable[[dict[str, Any], ValidationRule], tuple[bool, str | None]]


class HookRegistry:
    """Task-type-scoped hook registry. Supports same hook name with different
    implementations per task_type, with fallback to a global (None) key."""

    def __init__(self) -> None:
        self._hooks: dict[tuple[str | None, str], HookFunc] = {}

    def register(self, task_type: str, name: str, func: HookFunc) -> None:
        """Register a hook for a specific task_type."""
        self._hooks[(task_type, name)] = func

    def register_global(self, name: str, func: HookFunc) -> None:
        """Register a hook available to all task_types."""
        self._hooks[(None, name)] = func

    def get(self, task_type: str | None, name: str) -> HookFunc | None:
        """Get a hook for (task_type, name), falling back to (None, name)."""
        func = self._hooks.get((task_type, name))
        if func is not None:
            return func
        return self._hooks.get((None, name))

    def _call(self, rule: ValidationRule, params: dict[str, Any], task_type: str | None) -> tuple[bool, str | None]:
        """Dispatch to a named hook function for the given task_type."""
        hook_name = rule.hook
        if hook_name is None:
            return True, None
        func = self.get(task_type, hook_name)
        if func is None:
            logger.warning(
                f"Validation hook {hook_name!r} not found for task_type={task_type!r} — skipping rule {rule.name!r}"
            )
            return True, None
        return func(params, rule)


_hook_registry = HookRegistry()


def register_hook(task_type: str, name: str) -> Callable[[HookFunc], HookFunc]:
    """Decorator: register a named validation hook for a specific task_type.

    Usage:
        @register_hook("labeling", "validate_model_env_var")
        def _validate_model_env_var(params, rule):
            ...
    """

    def decorator(func: HookFunc) -> HookFunc:
        _hook_registry.register(task_type, name, func)
        return func

    return decorator


def register_global_hook(name: str) -> Callable[[HookFunc], HookFunc]:
    """Decorator: register a hook available to all task_types.

    Usage:
        @register_global_hook("my_hook")
        def _my_hook(params, rule):
            ...
    """

    def decorator(func: HookFunc) -> HookFunc:
        _hook_registry.register_global(name, func)
        return func

    return decorator


def _call_hook(rule: ValidationRule, params: dict[str, Any], task_type: str | None = None) -> tuple[bool, str | None]:
    """Dispatch to a named hook function."""
    return _hook_registry._call(rule, params, task_type)


# ── Built-in validation helpers ───────────────────────────────────────────────


def _check_type(value: Any, meta: ParamMetadata) -> tuple[bool, str | None]:
    """Check value matches the declared param type."""
    if value is None:
        return True, None

    type_map: dict[str, Callable[[Any], bool]] = {
        "string": lambda v: isinstance(v, str),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "bool": lambda v: isinstance(v, bool),
        "list": lambda v: isinstance(v, list),
    }
    checker = type_map.get(meta.type)
    if checker is None:
        logger.warning(f"Unknown param type {meta.type!r} for {meta.name}")
        return True, None
    if not checker(value):
        return False, f"expected {meta.type}, got {type(value).__name__}"
    return True, None


def _check_options(value: Any, meta: ParamMetadata) -> tuple[bool, str | None]:
    """Check value is in the allowed options list."""
    if meta.options is None or value is None:
        return True, None
    if value not in meta.options:
        return False, f"must be one of {meta.options}, got {value!r}"
    return True, None


def _check_bounds(value: Any, meta: ParamMetadata) -> tuple[bool, str | None]:
    """Check numeric value is within min/max bounds."""
    if not meta.is_numeric() or value is None:
        return True, None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"cannot convert {value!r} to number"
    if meta.min is not None and v < meta.min:
        return False, f"{v} is below minimum {meta.min}"
    if meta.max is not None and v > meta.max:
        return False, f"{v} is above maximum {meta.max}"
    return True, None


def _check_conditions(params: dict[str, Any], meta: ParamMetadata, param_name: str) -> tuple[bool, str | None]:
    """Check conditional requirement (parent-child activation)."""
    if meta.conditions is None or meta.conditions.required_when is None:
        return True, None

    for parent_name, parent_value in meta.conditions.required_when.items():
        actual = params.get(parent_name)
        if actual != parent_value:
            return True, None  # parent condition not met; param is inactive

    # Parent condition met — param is now effectively required
    if params.get(param_name) is None:
        return False, f"{param_name!r} is required when {meta.conditions.required_when}"
    return True, None


# ── Built-in hooks for labeling task ────────────────────────────────────────
@register_hook("labeling", "validate_model_env_var")
def _validate_model_env_var(params: dict[str, Any], rule: ValidationRule) -> tuple[bool, str | None]:
    """Hook: check model is registered and its API key env var is set."""
    from src.utils.llm_client import MODEL_REGISTRY, resolve_provider

    model = params.get("model")
    if model is None:
        return True, None  # let required check handle this

    if model not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        return False, f"Unknown model {model!r}. Known models: {known}"

    try:
        provider = resolve_provider(model)
    except ValueError as e:
        return False, str(e)

    env_var = provider.key_env
    if not os.environ.get(env_var):
        return False, f"Model {model!r} requires env var {env_var} to be set"
    return True, None


def _get_industry_major_categories() -> list[str]:
    """Load major categories from industry_dict.json."""
    import json

    path = Path(__file__).parent.parent.parent / "data" / "industry_dict.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


@register_hook("labeling", "validate_level1_task_id")
def _validate_level1_task_id(params: dict[str, Any], rule: ValidationRule) -> tuple[bool, str | None]:
    """Hook: level=2 requires a valid level1_task_id that points to a level-1 labeling task.

    If the task is not in PostgreSQL but exists in DuckDB (orphan), allow it with a warning.
    """
    import duckdb

    from src.db import Task
    from src.db.session import get_session
    from src.db.store import duckdb_store

    level = params.get("level", 1)
    level1_task_id = params.get("level1_task_id")

    if level != 2:
        return True, None

    if not level1_task_id:
        return False, rule.error_message or "level=2 requires 'level1_task_id'"

    try:
        uuid.UUID(str(level1_task_id))
    except ValueError:
        return False, f"Invalid level1_task_id (not a valid UUID): {level1_task_id}"

    with get_session() as session:
        task = session.query(Task).filter(Task.task_id == uuid.UUID(str(level1_task_id))).first()

    if task is not None:
        # Found in PostgreSQL — verify it's a level-1 labeling task
        if task.task_type != "labeling":
            return False, f"Referenced task is not a labeling task: {level1_task_id}"
        if (task.config or {}).get("level", 1) != 1:
            return False, f"Referenced task is not a level-1 task: {level1_task_id}"
        return True, None

    # Not in PostgreSQL — check DuckDB
    if duckdb_store.db_path.exists():
        con = duckdb.connect(str(duckdb_store.db_path), read_only=True)
        try:
            row = con.execute(
                "SELECT 1 FROM news_classified WHERE task_id = ? LIMIT 1",
                [str(level1_task_id)],
            ).fetchone()
        finally:
            con.close()
        if row is not None:
            logger.warning(
                f"level1_task_id={level1_task_id} not found in PostgreSQL "
                f"but found in DuckDB — allowing orphan level-1 reference"
            )
            return True, None

    return False, f"level1_task_id not found: {level1_task_id}"


@register_hook("labeling", "validate_major_categories")
def _validate_major_categories(params: dict[str, Any], rule: ValidationRule) -> tuple[bool, str | None]:
    """Hook: major_categories (if provided) must be a list of values from industry_dict.json."""
    major_categories = params.get("major_categories")
    if major_categories is None:
        return True, None  # optional field

    if not isinstance(major_categories, list):
        return False, f"major_categories must be a list, got {type(major_categories).__name__}"

    valid_majors = set(_get_industry_major_categories())
    invalid = [c for c in major_categories if c not in valid_majors]
    if invalid:
        return False, f"Unknown major_categories: {invalid}. Valid ones: {sorted(valid_majors)}"

    return True, None


# ── Validator class ────────────────────────────────────────────────────────────


class ParamValidator:
    """Auto-generated validator backed by TaskParamSchema.

    Usage:
        schema = TaskParamSchema.from_db("labeling")
        validator = ParamValidator(schema)
        is_valid, error = validator.validate(user_params)
    """

    def __init__(self, schema: TaskParamSchema) -> None:
        self.schema = schema

    def validate(self, params: dict[str, Any]) -> tuple[bool, str | None]:
        """Run all validations on params. Returns (is_valid, error_message)."""
        # 1. Apply defaults so subsequent checks see the full resolved params
        params = self.schema.apply_defaults(params)

        # 2. Check always-required params
        for name in self.schema.required_params:
            if params.get(name) is None:
                return False, f"Missing required parameter: {name!r}"

        # 3. Per-param validation: type, options, bounds
        for name, value in params.items():
            meta = self.schema.get_param(name)
            if meta is None:
                logger.warning(f"Unknown parameter {name!r} not in schema — skipping")
                continue

            ok, err = _check_type(value, meta)
            if not ok:
                return False, f"Invalid type for {name!r}: {err}"

            ok, err = _check_options(value, meta)
            if not ok:
                return False, f"Invalid value for {name!r}: {err}"

            ok, err = _check_bounds(value, meta)
            if not ok:
                return False, f"Out of bounds for {name!r}: {err}"

        # 4. Conditional requirements (parent-child)
        for name, meta in self.schema.parameters.items():
            if meta.conditions is not None:
                ok, err = _check_conditions(params, meta, name)
                if not ok:
                    return False, err

        # 5. Cross-param validation rules (hooks)
        for rule in self.schema.validation_rules:
            ok, err = _call_hook(rule, params, self.schema.task_type)
            if not ok:
                msg = rule.error_message or err or f"Validation failed: {rule.name}"
                return False, msg

        return True, None
