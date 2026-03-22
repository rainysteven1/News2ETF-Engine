"""Parameter metadata system — Pydantic models and DB-backed schema loading."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class ParamConditions(BaseModel):
    """Conditions under which a parameter becomes required or active."""

    required_when: dict[str, Any] | None = None


class ParamMetadata(BaseModel):
    """Metadata for a single task parameter."""

    name: str
    type: str  # string | int | float | bool | list
    required: bool = False
    default: Any | None = None
    options: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    description: str | None = None
    conditions: ParamConditions | None = None

    def is_numeric(self) -> bool:
        return self.type in ("int", "float")

    def is_list(self) -> bool:
        return self.type == "list"


class ValidationRule(BaseModel):
    """A cross-parameter validation rule, typically dispatched via a named hook."""

    name: str
    hook: str
    task_type: str | None = None
    error_message: str | None = None
    params: dict[str, Any] | None = None


class TaskParamSchema(BaseModel):
    """Full parameter schema for a task type, loaded from the database."""

    task_type: str
    parameters: dict[str, ParamMetadata]  # keyed by param_name
    validation_rules: list[ValidationRule] = []

    @property
    def required_params(self) -> list[str]:
        """Names of always-required parameters (backwards-compatible with TaskMetadata)."""
        return [name for name, p in self.parameters.items() if p.required]

    @property
    def optional_params(self) -> list[str]:
        """Names of optional parameters (backwards-compatible with TaskMetadata)."""
        return [name for name, p in self.parameters.items() if not p.required]

    def get_param(self, name: str) -> ParamMetadata | None:
        return self.parameters.get(name)

    def apply_defaults(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a new dict with defaults filled in for missing optional params."""
        result = dict(params)
        for name, meta in self.parameters.items():
            if name not in result and meta.default is not None:
                result[name] = meta.default
        return result

    @classmethod
    def from_db(cls, task_type: str) -> TaskParamSchema:
        """Load the schema for a task type from the database.

        Raises:
            ValueError: If no schema is found for the given task_type.
        """
        from src.db.models import ParamSchema, ParamValidationRule
        from src.db.session import get_session

        with get_session() as session:
            rows = session.query(ParamSchema).filter_by(task_type=task_type).all()
            rule_rows = session.query(ParamValidationRule).filter_by(task_type=task_type).all()

        if not rows and not rule_rows:
            raise ValueError(f"No param schema found for task_type={task_type!r}")

        parameters: dict[str, ParamMetadata] = {}
        for r in rows:
            opts: list[Any] | None = None
            default_val: Any | None = None
            conditions: ParamConditions | None = None

            if r.options:
                opts = json.loads(r.options)

            if r.default_val:
                default_val = json.loads(r.default_val)

            if r.conditions_json:
                conditions = ParamConditions.model_validate(json.loads(r.conditions_json))

            parameters[r.param_name] = ParamMetadata(
                name=r.param_name,
                type=r.param_type,
                required=r.required,
                default=default_val,
                options=opts,
                min=r.min_val,
                max=r.max_val,
                description=r.description,
                conditions=conditions,
            )

        validation_rules: list[ValidationRule] = []
        for r in rule_rows:
            validation_rules.append(
                ValidationRule(
                    name=r.rule_name,
                    hook=r.hook,
                    task_type=r.task_type,
                    error_message=r.error_message,
                    params=json.loads(r.params_json) if r.params_json else None,
                )
            )

        return cls(
            task_type=task_type,
            parameters=parameters,
            validation_rules=validation_rules,
        )
