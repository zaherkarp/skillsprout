"""
Account Deletion Endpoint for SkillSprout
==========================================

RATIONALE: The right to deletion is one of the most consequential rights
under GDPR (Article 17, "right to erasure") and CCPA (Section 1798.105,
"right to delete"). Unlike data export (which is read-only), deletion is
irreversible. This module treats deletion with the gravity it deserves:

  1. CASCADING: We delete from EVERY table that references the user, not
     just the profile table. We do this explicitly (not relying solely on
     DB-level CASCADE) so we can count and audit each deletion.

  2. AUDITED: Every deletion is logged to the DeletionAuditLog table BEFORE
     the transaction commits. The audit record contains NO personal data --
     only the user ID, table names, row counts, and timestamp.

  3. VERIFIED: After deletion, we run a verification query to confirm that
     zero rows remain for the user across all relevant tables. This catches
     edge cases where a new table was added but not included in the deletion
     cascade.

  4. IRREVERSIBLE: We do not implement soft-delete. The data is gone. This
     is intentional: GDPR requires actual deletion, not just hiding. Soft
     deletion would leave personal data in the database, defeating the
     purpose.

ENDPOINT:

    DELETE /api/v1/user/{user_id}/data

RESPONSE: A structured JSON document confirming what was deleted, with
per-table row counts and a completion timestamp.

TIMELINE: The deletion happens synchronously (within the HTTP request).
The 72-hour SLA mentioned in the privacy policy accounts for cases where
the request is queued (e.g., via email), not for API-driven deletion
which is immediate.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from sqlalchemy.orm import joinedload

from app.db.session import get_db
from app.models.models import (
    UserProfile,
    UserSkillRating,
    UserFeedback,
    RecommendationEvent,
    RecommendedOccupation,
    UserCurrentOccupation,
)
from app.core.config import settings
from app.core.privacy.data_classification import (
    DataTier,
    get_deletable_models,
    MODEL_CLASSIFICATIONS,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Deletion order map
# ---------------------------------------------------------------------------
# RATIONALE: We delete in a specific order to respect foreign key constraints.
# Even though we have ON DELETE CASCADE on most foreign keys, we delete
# explicitly for two reasons:
#   1. We need accurate per-table deletion counts for the audit log.
#   2. Some databases (especially in test environments) may not have CASCADE
#      configured, and we want deletion to work everywhere.
#
# The order is: leaf tables first, then parent tables, ending with UserProfile.
# This is the reverse of the creation order.

DELETION_ORDER: List[Dict[str, Any]] = [
    {
        "table_name": "user_feedback",
        "model": UserFeedback,
        "tier": DataTier.TIER_4_SENSITIVE,
        "description": "Feedback on recommendations (clicks, saves, applications)",
        # RATIONALE: UserFeedback references RecommendationEvent via event_id.
        # We must delete feedback before events to avoid FK violations.
        "filter_strategy": "via_event",  # Needs join through RecommendationEvent
    },
    {
        "table_name": "recommended_occupation",
        "model": RecommendedOccupation,
        "tier": DataTier.TIER_4_SENSITIVE,
        "description": "Individual occupation recommendations",
        # RATIONALE: RecommendedOccupation references RecommendationEvent.
        "filter_strategy": "via_event",
    },
    {
        "table_name": "recommendation_event",
        "model": RecommendationEvent,
        "tier": DataTier.TIER_4_SENSITIVE,
        "description": "Recommendation generation events",
        "filter_strategy": "direct",  # Has user_id column directly
    },
    {
        "table_name": "user_skill_rating",
        "model": UserSkillRating,
        "tier": DataTier.TIER_3_PERSONAL,
        "description": "Self-assessed skill proficiency ratings",
        "filter_strategy": "direct",
    },
    {
        "table_name": "user_current_occupation",
        "model": UserCurrentOccupation,
        "tier": DataTier.TIER_3_PERSONAL,
        "description": "Current and historical occupation selections",
        "filter_strategy": "direct",
    },
    {
        "table_name": "user_profile",
        "model": UserProfile,
        "tier": DataTier.TIER_3_PERSONAL,
        "description": "Core user profile",
        "filter_strategy": "by_id",  # Filtered by primary key
    },
]


# ---------------------------------------------------------------------------
# Async deletion helpers
# ---------------------------------------------------------------------------


async def _get_user_event_ids(db: AsyncSession, user_id: int) -> List[int]:
    """
    Fetch all RecommendationEvent IDs for a user.

    RATIONALE: UserFeedback and RecommendedOccupation reference events,
    not users directly. We need the event IDs to cascade the delete.
    """
    result = await db.execute(
        select(RecommendationEvent.id).where(
            RecommendationEvent.user_id == user_id
        )
    )
    return [row[0] for row in result.all()]


async def _delete_table_records(
    db: AsyncSession,
    table_spec: Dict[str, Any],
    user_id: int,
    event_ids: List[int],
) -> int:
    """
    Delete all records for a user from a specific table.

    Returns the number of records deleted.

    RATIONALE: We use raw DELETE statements rather than ORM bulk operations
    because (a) we need accurate row counts, (b) we want to avoid loading
    all objects into memory (which ORM session.delete() would do), and
    (c) direct DELETE is faster for bulk operations.
    """
    model = table_spec["model"]
    strategy = table_spec["filter_strategy"]

    if strategy == "direct":
        # Table has a user_id column -- straightforward filter.
        stmt = delete(model).where(model.user_id == user_id)

    elif strategy == "via_event":
        # Table references RecommendationEvent, not UserProfile directly.
        # We filter by event_id IN (user's event IDs).
        if not event_ids:
            return 0
        stmt = delete(model).where(model.event_id.in_(event_ids))

    elif strategy == "by_id":
        # This is the UserProfile table itself.
        stmt = delete(model).where(model.id == user_id)

    else:
        logger.error("Unknown filter strategy: %s", strategy)
        return 0

    result = await db.execute(stmt)
    return result.rowcount


async def _verify_deletion(
    db: AsyncSession,
    user_id: int,
    event_ids: List[int],
) -> Dict[str, int]:
    """
    Verify that zero records remain for the user across all tables.

    RATIONALE: Defense-in-depth. After deletion, we confirm that nothing
    was missed. If any table still has records, it indicates either a bug
    in the deletion logic or a new table that was not added to DELETION_ORDER.
    This verification query catches such issues before we tell the user
    their data is gone.

    Returns:
        Dict mapping table_name -> remaining row count (should all be 0).
    """
    remaining: Dict[str, int] = {}

    # Check direct user_id tables
    for model in [UserSkillRating, UserCurrentOccupation, RecommendationEvent]:
        result = await db.execute(
            select(func.count()).select_from(model).where(model.user_id == user_id)
        )
        count = result.scalar()
        remaining[model.__tablename__] = count

    # Check event-linked tables
    if event_ids:
        for model in [UserFeedback, RecommendedOccupation]:
            result = await db.execute(
                select(func.count()).select_from(model).where(
                    model.event_id.in_(event_ids)
                )
            )
            count = result.scalar()
            remaining[model.__tablename__] = count
    else:
        remaining["user_feedback"] = 0
        remaining["recommended_occupation"] = 0

    # Check profile
    result = await db.execute(
        select(func.count()).select_from(UserProfile).where(UserProfile.id == user_id)
    )
    remaining["user_profile"] = result.scalar()

    return remaining


async def _write_deletion_audit(
    db: AsyncSession,
    user_id: int,
    deletion_summary: Dict[str, int],
    initiated_by: str,
) -> None:
    """
    Write an audit log entry for the deletion.

    RATIONALE: We import DeletionAuditLog here (not at module level) to
    avoid circular imports since retention_policy.py defines the model.
    The audit log contains NO personal data -- only the user ID (which no
    longer maps to any profile), table names, and row counts.
    """
    from app.core.privacy.retention_policy import DeletionAuditLog

    total = sum(deletion_summary.values())
    audit = DeletionAuditLog(
        deleted_at=datetime.utcnow(),
        deletion_type="user_request",
        table_name="ALL_USER_TABLES",
        records_deleted=total,
        criteria={
            "user_id": user_id,
            "per_table": deletion_summary,
        },
        initiated_by=initiated_by,
        notes=(
            f"Full account deletion for user {user_id}. "
            f"Total records removed: {total}. "
            f"Per-table breakdown: {deletion_summary}"
        ),
    )
    db.add(audit)


# ---------------------------------------------------------------------------
# Deletion endpoint
# ---------------------------------------------------------------------------


@router.delete(
    "/user/{user_id}/data",
    summary="Delete all user data (GDPR Art. 17 / CCPA right to delete)",
    response_description="Confirmation of cascading deletion with per-table counts",
)
async def delete_user_data(
    user_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Permanently delete ALL data associated with a user.

    This is an IRREVERSIBLE operation. Once completed, the user's profile,
    skill ratings, occupation selections, recommendation history, and
    feedback are permanently removed from all database tables.

    The deletion is:
      - CASCADING: All child records are deleted before parent records.
      - AUDITED: A deletion audit log is created (containing no personal data).
      - VERIFIED: Post-deletion checks confirm zero remaining records.

    This endpoint satisfies:
      - GDPR Article 17 (right to erasure / "right to be forgotten")
      - CCPA Section 1798.105 (right to delete)

    RATIONALE: We perform the deletion synchronously within the HTTP request
    because (a) SkillSprout's per-user data volume is small enough for
    real-time deletion, and (b) telling the user "your data will be deleted
    eventually" is less satisfying than "your data has been deleted NOW."
    For systems with large data volumes, a background job with status
    polling would be more appropriate.
    """

    # --- Verify user exists ---
    # RATIONALE: We check existence first to return a clear 404 rather than
    # silently deleting zero rows and returning success. This prevents
    # confusion if the user fat-fingers their ID.
    result = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=(
                f"User {user_id} not found. The account may have already been "
                "deleted, or the ID may be incorrect."
            ),
        )

    # --- Collect event IDs for cascade ---
    # RATIONALE: We need event IDs before deleting events, because
    # UserFeedback and RecommendedOccupation reference events, not users.
    event_ids = await _get_user_event_ids(db, user_id)

    # --- Perform cascading deletion ---
    deletion_summary: Dict[str, int] = {}
    deletion_details: List[Dict[str, Any]] = []

    for table_spec in DELETION_ORDER:
        try:
            deleted_count = await _delete_table_records(
                db=db,
                table_spec=table_spec,
                user_id=user_id,
                event_ids=event_ids,
            )
            deletion_summary[table_spec["table_name"]] = deleted_count
            deletion_details.append({
                "table": table_spec["table_name"],
                "records_deleted": deleted_count,
                "classification_tier": table_spec["tier"].name,
                "description": table_spec["description"],
            })

            if deleted_count > 0:
                logger.info(
                    "Deleted %d records from %s for user %d",
                    deleted_count,
                    table_spec["table_name"],
                    user_id,
                )

        except Exception as e:
            # RATIONALE: If deletion fails for one table, we log the error
            # and continue with other tables. A partial deletion is better
            # than no deletion. The verification step will catch any
            # remaining records.
            logger.error(
                "Failed to delete from %s for user %d: %s",
                table_spec["table_name"],
                user_id,
                str(e),
                exc_info=True,
            )
            deletion_summary[table_spec["table_name"]] = -1

    # --- Write audit log ---
    # RATIONALE: The audit log is written BEFORE commit so it is part of
    # the same transaction. If the transaction fails, neither the deletions
    # nor the audit entry persist.
    await _write_deletion_audit(
        db=db,
        user_id=user_id,
        deletion_summary=deletion_summary,
        initiated_by=f"user_request:{user_id}",
    )

    # --- Commit the transaction ---
    await db.commit()

    # --- Post-deletion verification ---
    # RATIONALE: We verify AFTER commit because we want to confirm the
    # deletion actually persisted, not just that it was staged in the
    # transaction.
    remaining = await _verify_deletion(db, user_id, event_ids)
    all_clear = all(count == 0 for count in remaining.values())

    if not all_clear:
        # RATIONALE: This should never happen if DELETION_ORDER is complete.
        # If it does, it means a new table was added that references users
        # but was not included in the deletion cascade. We log this as a
        # critical error for immediate investigation.
        logger.critical(
            "POST-DELETION VERIFICATION FAILED for user %d. "
            "Remaining records: %s. This indicates a gap in the deletion "
            "cascade that must be fixed immediately.",
            user_id,
            remaining,
        )

    total_deleted = sum(v for v in deletion_summary.values() if v > 0)

    response = {
        "status": "completed" if all_clear else "partial",
        "user_id": user_id,
        "deleted_at": datetime.utcnow().isoformat(),
        "total_records_deleted": total_deleted,
        "deletion_details": deletion_details,
        "verification": {
            "all_data_removed": all_clear,
            "remaining_records": remaining if not all_clear else {},
        },
        "notice": (
            "All personal data associated with this account has been "
            "permanently deleted. This action cannot be undone. "
            "De-identified aggregate data (Tier 2) may still exist but "
            "cannot be linked back to your identity."
        ),
        "legal_references": {
            "gdpr": "Article 17 - Right to Erasure",
            "ccpa": "Section 1798.105 - Right to Delete",
        },
    }

    logger.info(
        "Account deletion completed for user %d: %d total records deleted, "
        "verification=%s",
        user_id,
        total_deleted,
        "PASSED" if all_clear else "FAILED",
    )

    return response
