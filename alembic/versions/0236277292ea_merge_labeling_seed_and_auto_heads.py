"""merge labeling seed and auto heads

Revision ID: 0236277292ea
Revises: 002, 3e3c7b806fac
Create Date: 2026-03-21 15:36:10.234629

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0236277292ea'
down_revision: Union[str, Sequence[str], None] = ('002', '3e3c7b806fac')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
