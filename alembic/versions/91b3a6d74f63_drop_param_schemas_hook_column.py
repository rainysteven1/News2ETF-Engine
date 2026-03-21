"""drop param_schemas hook column

Revision ID: 91b3a6d74f63
Revises: 93c3269b944d
Create Date: 2026-03-21 16:12:25.620501

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '91b3a6d74f63'
down_revision: Union[str, Sequence[str], None] = '93c3269b944d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop hook column from param_schemas (hooks now live in param_validation_rules only)."""
    op.drop_column("param_schemas", "hook")


def downgrade() -> None:
    """Restore hook column."""
    op.add_column("param_schemas", sa.Column("hook", sa.String(64), nullable=True))
