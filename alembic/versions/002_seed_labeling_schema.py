"""seed labeling param schema

Revision ID: 002
Revises: 001
Create Date: 2026-03-21

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # param_schemas rows for task_type = "labeling"
    param_schemas_rows = [
        # Shared (LabelingConfig)
        {
            "task_type": "labeling",
            "param_name": "model",
            "param_type": "string",
            "required": True,
            "default_val": None,
            "options": None,
            "min_val": None,
            "max_val": None,
            "description": "LLM model identifier",
            "conditions_json": None,
        },
        {
            "task_type": "labeling",
            "param_name": "temperature",
            "param_type": "float",
            "required": True,
            "default_val": None,
            "options": None,
            "min_val": 0.0,
            "max_val": 2.0,
            "description": "Sampling temperature",
            "conditions_json": None,
        },
        {
            "task_type": "labeling",
            "param_name": "max_tokens",
            "param_type": "int",
            "required": False,
            "default_val": "8192",
            "options": None,
            "min_val": 1.0,
            "max_val": 32768.0,
            "description": "Max tokens per response",
            "conditions_json": None,
        },
        {
            "task_type": "labeling",
            "param_name": "checkpoint_every",
            "param_type": "int",
            "required": True,
            "default_val": "10",
            "options": None,
            "min_val": 1.0,
            "max_val": None,
            "description": "Checkpoint interval in batches",
            "conditions_json": None,
        },
        {
            "task_type": "labeling",
            "param_name": "llm_retry",
            "param_type": "int",
            "required": True,
            "default_val": "2",
            "options": None,
            "min_val": 0.0,
            "max_val": 10.0,
            "description": "Number of LLM retry attempts",
            "conditions_json": None,
        },
        {
            "task_type": "labeling",
            "param_name": "seed",
            "param_type": "int",
            "required": False,
            "default_val": "42",
            "options": None,
            "min_val": None,
            "max_val": None,
            "description": "Random seed for reproducibility",
            "conditions_json": None,
        },
        # Level selection
        {
            "task_type": "labeling",
            "param_name": "level",
            "param_type": "int",
            "required": True,
            "default_val": "1",
            "options": "[1, 2]",
            "min_val": 1.0,
            "max_val": 2.0,
            "description": "Classification level",
            "conditions_json": None,
        },
        # Level1Config adds
        {
            "task_type": "labeling",
            "param_name": "start",
            "param_type": "int",
            "required": False,
            "default_val": "0",
            "options": None,
            "min_val": 0.0,
            "max_val": None,
            "description": "Starting offset for level 1",
            "conditions_json": '{"required_when": {"level": 1}}',
        },
        {
            "task_type": "labeling",
            "param_name": "batch_size",
            "param_type": "int",
            "required": False,
            "default_val": "20",
            "options": None,
            "min_val": 1.0,
            "max_val": None,
            "description": "Batch size for LLM calls",
            "conditions_json": None,
        },
        # Level2Config
        {
            "task_type": "labeling",
            "param_name": "level1_task_id",
            "param_type": "string",
            "required": False,
            "default_val": None,
            "options": None,
            "min_val": None,
            "max_val": None,
            "description": "Parent level-1 task ID (required when level=2)",
            "conditions_json": '{"required_when": {"level": 2}}',
        },
        # sample_size (from LabelingConfig.sample_size)
        {
            "task_type": "labeling",
            "param_name": "sample_size",
            "param_type": "int",
            "required": True,
            "default_val": None,
            "options": None,
            "min_val": 1.0,
            "max_val": None,
            "description": "Number of news records to process",
            "conditions_json": None,
        },
    ]

    now = sa.func.now()
    for row in param_schemas_rows:
        op.execute(
            f"""
            INSERT INTO param_schemas
                (task_type, param_name, param_type, required, default_val, options,
                 min_val, max_val, description, conditions_json, created_at, updated_at)
            VALUES
                ('{row["task_type"]}', '{row["param_name"]}', '{row["param_type"]}',
                 {row["required"]}, {sa.text("null" if row["default_val"] is None else f"'{row['default_val']}'")},
                 {sa.text("null" if row["options"] is None else f"'{row['options']}'")},
                 {sa.text("null" if row["min_val"] is None else str(row["min_val"]))},
                 {sa.text("null" if row["max_val"] is None else str(row["max_val"]))},
                 {sa.text("null" if row["description"] is None else f"'{row['description']}'")},
                 {sa.text("null" if row["conditions_json"] is None else f"'{row['conditions_json']}'")},
                 {now}, {now})
            """
        )

    # param_validation_rules rows for task_type = "labeling"
    op.execute(
        f"""
        INSERT INTO param_validation_rules
            (task_type, rule_name, hook, error_message, params_json, created_at)
        VALUES
            ('labeling', 'validate_model', 'validate_model_env_var',
             null, null, {now})
        """
    )
    op.execute(
        f"""
        INSERT INTO param_validation_rules
            (task_type, rule_name, hook, error_message, params_json, created_at)
        VALUES
            ('labeling', 'validate_level1_task_id', 'validate_level1_task_id',
             null, null, {now})
        """
    )


def downgrade() -> None:
    op.execute("DELETE FROM param_validation_rules WHERE task_type = 'labeling'")
    op.execute("DELETE FROM param_schemas WHERE task_type = 'labeling'")
