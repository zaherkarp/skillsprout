"""Add user_profile.owner_token_hash for per-profile ownership.

Revision ID: a7c2e9f4b1d3
Revises: d1316f64dcb1
Create Date: 2026-11-28 03:00:00

Why: profiles were identified only by a sequential id, so anyone who could
reach the API could read or delete any profile. Each profile now has a random
owner token, returned once at creation; this column stores its SHA-256 hash
(see app/core/profile_ownership.py).

Existing rows get NULL. That is deliberate: there is no way to tell a legacy
profile's real owner from anyone else, so those profiles stay locked until an
operator runs `python -m app.core.profile_ownership issue <user_id>` and hands
the printed link to the owner. Nullable, no default, no backfill, so the
migration is instant and safe to run on a live table.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a7c2e9f4b1d3"
down_revision: str | None = "d1316f64dcb1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_profile",
        sa.Column("owner_token_hash", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    # Dropping the column throws away every issued token; profiles become
    # reachable by id again if the code that checks tokens is also rolled back.
    op.drop_column("user_profile", "owner_token_hash")
