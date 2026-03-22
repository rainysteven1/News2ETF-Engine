"""seed llm provider config

Revision ID: 003
Revises: 002
Create Date: 2026-03-22

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: str | Sequence[str] | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create provider_credentials table
    op.create_table(
        "provider_credentials",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("provider_key", sa.String(length=64), nullable=False),
        sa.Column("yaml_key", sa.String(length=128), nullable=False),
        sa.Column("api_key", sa.Text(), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )
    op.create_index("ix_provider_credentials_provider_key", "provider_credentials", ["provider_key"])
    op.create_index("ix_provider_credentials_is_active", "provider_credentials", ["is_active"])
    op.create_index(
        "ix_provider_credentials_provider_key_active", "provider_credentials", ["provider_key", "is_active"]
    )

    # Create model_meta_rules table
    op.create_table(
        "model_meta_rules",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("provider_key", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )
    op.create_index("ix_model_meta_rules_model_name", "model_meta_rules", ["model_name"], unique=True)
    op.create_index("ix_model_meta_rules_provider_key", "model_meta_rules", ["provider_key"])

    # Seed model_meta_rules
    model_rows = [
        # Zhipu GLM
        ("glm-4-flash", "zhipu"),
        ("glm-4.7", "zhipu"),
        ("glm-4.5-airx", "zhipu"),
        ("glm-4.7-flashx", "zhipu"),
        # MiniMax
        ("MiniMax-M2.7", "minimax"),
        ("MiniMax-M2.7-highspeed", "minimax"),
        ("MiniMax-M2.5", "minimax"),
        ("MiniMax-M2.5-highspeed", "minimax"),
        ("MiniMax-M2.1", "minimax"),
        ("MiniMax-M2.1-highspeed", "minimax"),
    ]

    now = sa.func.now()
    for model_name, provider_key in model_rows:
        op.execute(
            f"""
            INSERT INTO model_meta_rules (id, model_name, provider_key, created_at, updated_at)
            VALUES (gen_random_uuid(), '{model_name}', '{provider_key}', {now}, {now})
            """
        )


def downgrade() -> None:
    op.drop_table("model_meta_rules")
    op.drop_table("provider_credentials")
