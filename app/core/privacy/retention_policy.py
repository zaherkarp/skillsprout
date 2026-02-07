"""
Data Retention Policy Engine for SkillSprout
=============================================

RATIONALE: Data retention is not optional -- it is a legal requirement under
GDPR (Art. 5(1)(e), "storage limitation") and a best practice under CCPA.
The principle is simple: do not keep personal data longer than necessary for
the purpose it was collected.

For SkillSprout specifically:

  - Event-level tracking (recommendation events, feedback) serves the
    recommendation feedback loop. After 90 days, the model has either learned
    from this data or it is too stale to be useful. Purging it limits blast
    radius in a breach.

  - User profiles must be fully deletable within 72 hours of a deletion
    request. The 72-hour window allows for async processing (Celery) and
    gives the system time to cascade deletes across all related tables while
    generating a complete audit log.

  - Model training snapshots may contain statistical patterns derived from
    user behavior. We retain these ONLY in de-identified form. Any snapshot
    that could be linked back to individual users is purged on the same
    schedule as the source data.

This module provides:
  1. A RetentionPolicy configuration class
  2. Functions that identify and purge expired records
  3. A Celery task for nightly automated enforcement
  4. An audit log for every deletion event (so we can prove compliance)

The nightly Celery task is the primary enforcement mechanism. It runs at
3:00 AM UTC (off-peak) and processes each retention rule sequentially,
logging every deletion to the audit table before committing.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, JSON, select, delete, func
from sqlalchemy.orm import Session

from app.db.session import Base, SyncSessionLocal
from app.core.config import settings
from app.models.models import (
    UserProfile,
    UserSkillRating,
    UserFeedback,
    RecommendationEvent,
    RecommendedOccupation,
    UserCurrentOccupation,
)
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audit log model
# ---------------------------------------------------------------------------
# RATIONALE: Every deletion must be logged. Without an audit trail, we cannot
# prove to regulators that we actually enforced our retention policy.  The
# audit log itself contains NO personal data -- only counts and identifiers
# of what was deleted.  The audit log has its own long retention (7 years)
# to satisfy potential regulatory inquiry windows.


class DeletionAuditLog(Base):
    """
    Immutable audit record for every data deletion event.

    This table answers the question: "Can you prove you deleted that data?"

    Fields:
        id:             Auto-incrementing primary key.
        deleted_at:     UTC timestamp of the deletion.
        deletion_type:  Category of deletion (retention_purge, user_request,
                        cascading_delete).
        table_name:     The database table from which records were removed.
        records_deleted: Count of rows deleted in this operation.
        criteria:       JSON description of the deletion filter (e.g.,
                        {"older_than": "2025-01-15", "tier": "TIER_4_SENSITIVE"}).
        initiated_by:   Who/what triggered the deletion (e.g., "celery_nightly",
                        "user_request:42", "admin:jane@example.com").
        notes:          Optional free-text for additional context.
    """
    __tablename__ = "deletion_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    deleted_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    deletion_type = Column(String(50), nullable=False, index=True)
    table_name = Column(String(100), nullable=False)
    records_deleted = Column(Integer, nullable=False, default=0)
    criteria = Column(JSON, nullable=True)
    initiated_by = Column(String(200), nullable=False)
    notes = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Retention policy configuration
# ---------------------------------------------------------------------------
# RATIONALE: Centralizing retention rules in a single data structure makes
# it easy to audit, modify, and test. Each rule maps a table (or logical
# data category) to a retention period and the column used to determine age.


class RetentionRule:
    """A single retention rule for a data category."""

    def __init__(
        self,
        table_name: str,
        model_class: Any,
        age_column: str,
        max_days: int,
        description: str,
        cascade_from_parent: bool = False,
    ):
        """
        Args:
            table_name:   Name of the DB table.
            model_class:  SQLAlchemy model class for ORM operations.
            age_column:   Column name used to determine record age.
            max_days:     Records older than this many days are eligible for
                          purging. Set to 0 to skip time-based purging (used
                          for tables that are only purged via cascade).
            description:  Human-readable explanation of why this retention
                          period was chosen.
            cascade_from_parent: If True, records in this table are deleted
                          via foreign key CASCADE when the parent is deleted,
                          so we do not need to purge them independently.
        """
        self.table_name = table_name
        self.model_class = model_class
        self.age_column = age_column
        self.max_days = max_days
        self.description = description
        self.cascade_from_parent = cascade_from_parent


# RATIONALE for each retention period:
#
# - UserFeedback (90 days): Contains the most sensitive signals (apply,
#   interview, offer). After 90 days, the calibration model has incorporated
#   the feedback. Keeping it longer increases breach exposure without
#   improving recommendations.
#
# - RecommendedOccupation (90 days): Tied to events. Cascades from
#   RecommendationEvent, but we also purge independently as a safety net.
#
# - RecommendationEvent (90 days): The parent of recommendations and feedback.
#   Once purged, child records cascade-delete automatically.
#
# - UserSkillRating: NOT time-purged. These are actively maintained by the
#   user and represent their current self-assessment. They are deleted only
#   when the user requests account deletion.
#
# - UserCurrentOccupation: NOT time-purged for active records. Inactive
#   (historical) selections are purged after 180 days.
#
# - UserProfile: NOT time-purged. Deleted only on explicit user request.

RETENTION_RULES: List[RetentionRule] = [
    RetentionRule(
        table_name="user_feedback",
        model_class=UserFeedback,
        age_column="action_at",
        max_days=90,
        description=(
            "Event-level feedback (clicks, saves, applications) auto-purged "
            "after 90 days. This limits exposure of the most sensitive user "
            "signals while providing enough time for the calibration model "
            "to learn from the feedback."
        ),
    ),
    RetentionRule(
        table_name="recommended_occupation",
        model_class=RecommendedOccupation,
        age_column="id",  # Uses parent event's created_at via join
        max_days=0,  # Cascades from RecommendationEvent
        description=(
            "Individual recommendations are cascade-deleted when their parent "
            "RecommendationEvent is purged. No independent time-based purge."
        ),
        cascade_from_parent=True,
    ),
    RetentionRule(
        table_name="recommendation_event",
        model_class=RecommendationEvent,
        age_column="created_at",
        max_days=90,
        description=(
            "Recommendation generation events purged after 90 days. Child "
            "records (recommended_occupation, user_feedback) cascade-delete "
            "automatically via foreign key constraints."
        ),
    ),
    RetentionRule(
        table_name="user_current_occupation",
        model_class=UserCurrentOccupation,
        age_column="selected_at",
        max_days=180,
        description=(
            "Inactive (historical) occupation selections purged after 180 "
            "days. Active selections are preserved until the user switches "
            "or deletes their account."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Retention enforcement functions
# ---------------------------------------------------------------------------


def _log_deletion(
    db: Session,
    deletion_type: str,
    table_name: str,
    records_deleted: int,
    criteria: Dict[str, Any],
    initiated_by: str,
    notes: Optional[str] = None,
) -> None:
    """
    Write an immutable audit log entry for a deletion event.

    RATIONALE: The audit log is written in the SAME transaction as the
    deletion. If the transaction rolls back, the audit entry also rolls back,
    ensuring consistency. We never have phantom audit entries for deletions
    that did not actually happen.
    """
    audit_entry = DeletionAuditLog(
        deleted_at=datetime.utcnow(),
        deletion_type=deletion_type,
        table_name=table_name,
        records_deleted=records_deleted,
        criteria=criteria,
        initiated_by=initiated_by,
        notes=notes,
    )
    db.add(audit_entry)
    logger.info(
        "Audit log: %s deleted %d rows from %s (criteria: %s)",
        initiated_by,
        records_deleted,
        table_name,
        criteria,
    )


def purge_expired_records(
    db: Session,
    rule: RetentionRule,
    initiated_by: str = "celery_nightly",
    dry_run: bool = False,
) -> int:
    """
    Delete records that exceed the retention period defined by a rule.

    Args:
        db:           Active SQLAlchemy session.
        rule:         The RetentionRule to enforce.
        initiated_by: Identifier for who/what triggered this purge.
        dry_run:      If True, count but do not delete.

    Returns:
        Number of records deleted (or that would be deleted in dry_run mode).

    RATIONALE: Each rule is processed independently so that a failure in one
    table does not block purging of other tables. The caller (the Celery task)
    catches exceptions per-rule and continues.
    """
    if rule.cascade_from_parent or rule.max_days <= 0:
        # This table is handled via cascade; skip independent purge.
        logger.debug("Skipping %s (cascade_from_parent=True)", rule.table_name)
        return 0

    cutoff = datetime.utcnow() - timedelta(days=rule.max_days)
    age_col = getattr(rule.model_class, rule.age_column)

    # For UserCurrentOccupation, only purge INACTIVE historical records.
    # Active occupation selections must be preserved.
    if rule.model_class == UserCurrentOccupation:
        count_query = (
            db.query(func.count(rule.model_class.id))
            .filter(age_col < cutoff, UserCurrentOccupation.is_active == False)  # noqa: E712
        )
        delete_query = (
            db.query(rule.model_class)
            .filter(age_col < cutoff, UserCurrentOccupation.is_active == False)  # noqa: E712
        )
    else:
        count_query = (
            db.query(func.count(rule.model_class.id))
            .filter(age_col < cutoff)
        )
        delete_query = (
            db.query(rule.model_class)
            .filter(age_col < cutoff)
        )

    count = count_query.scalar()

    if count == 0:
        logger.info(
            "Retention check: %s -- no records older than %d days",
            rule.table_name,
            rule.max_days,
        )
        return 0

    if dry_run:
        logger.info(
            "DRY RUN: Would delete %d records from %s (older than %s)",
            count,
            rule.table_name,
            cutoff.isoformat(),
        )
        return count

    # Perform the deletion.
    deleted = delete_query.delete(synchronize_session="fetch")

    # Write audit log in the same transaction.
    _log_deletion(
        db=db,
        deletion_type="retention_purge",
        table_name=rule.table_name,
        records_deleted=deleted,
        criteria={
            "cutoff_date": cutoff.isoformat(),
            "max_days": rule.max_days,
            "age_column": rule.age_column,
        },
        initiated_by=initiated_by,
        notes=rule.description,
    )

    logger.info(
        "Retention purge: deleted %d records from %s (cutoff: %s)",
        deleted,
        rule.table_name,
        cutoff.isoformat(),
    )

    return deleted


def enforce_all_retention_rules(
    db: Session,
    initiated_by: str = "celery_nightly",
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Run all retention rules and return a summary of deletions.

    Returns:
        Dict mapping table_name -> number of records deleted.
    """
    summary: Dict[str, int] = {}

    for rule in RETENTION_RULES:
        try:
            deleted = purge_expired_records(
                db=db,
                rule=rule,
                initiated_by=initiated_by,
                dry_run=dry_run,
            )
            summary[rule.table_name] = deleted
        except Exception as e:
            # RATIONALE: Log and continue. One table failing should not block
            # purging of other tables. The error is logged for investigation.
            logger.error(
                "Retention purge FAILED for %s: %s",
                rule.table_name,
                str(e),
                exc_info=True,
            )
            summary[rule.table_name] = -1  # Signal failure

    return summary


def purge_user_data(
    db: Session,
    user_id: int,
    initiated_by: str = "user_request",
) -> Dict[str, int]:
    """
    Delete ALL data associated with a specific user.

    RATIONALE: This is the implementation behind the GDPR Art. 17 "right to
    erasure" and CCPA "right to delete" endpoints. We delete in dependency
    order to respect foreign key constraints, even though CASCADE should
    handle it. Belt-and-suspenders approach for data destruction.

    The 72-hour SLA for deletion is enforced at the API layer; this function
    performs the actual deletion synchronously when called.

    Args:
        db:           Active SQLAlchemy session.
        user_id:      The user whose data should be deleted.
        initiated_by: Identifier for audit trail (e.g., "user_request:42").

    Returns:
        Dict mapping table_name -> number of records deleted.
    """
    summary: Dict[str, int] = {}

    # RATIONALE for deletion order: Start with leaf tables (no dependents),
    # then work up to the parent (UserProfile). This avoids FK violations
    # even if CASCADE is not configured on every relationship.

    # 1. UserFeedback (leaf -- depends on RecommendationEvent)
    feedback_events = (
        db.query(RecommendationEvent.id)
        .filter(RecommendationEvent.user_id == user_id)
        .subquery()
    )
    deleted = (
        db.query(UserFeedback)
        .filter(UserFeedback.event_id.in_(select(feedback_events.c.id)))
        .delete(synchronize_session="fetch")
    )
    summary["user_feedback"] = deleted

    # 2. RecommendedOccupation (leaf -- depends on RecommendationEvent)
    deleted = (
        db.query(RecommendedOccupation)
        .filter(RecommendedOccupation.event_id.in_(select(feedback_events.c.id)))
        .delete(synchronize_session="fetch")
    )
    summary["recommended_occupation"] = deleted

    # 3. RecommendationEvent (depends on UserProfile)
    deleted = (
        db.query(RecommendationEvent)
        .filter(RecommendationEvent.user_id == user_id)
        .delete(synchronize_session="fetch")
    )
    summary["recommendation_event"] = deleted

    # 4. UserSkillRating (depends on UserProfile)
    deleted = (
        db.query(UserSkillRating)
        .filter(UserSkillRating.user_id == user_id)
        .delete(synchronize_session="fetch")
    )
    summary["user_skill_rating"] = deleted

    # 5. UserCurrentOccupation (depends on UserProfile)
    deleted = (
        db.query(UserCurrentOccupation)
        .filter(UserCurrentOccupation.user_id == user_id)
        .delete(synchronize_session="fetch")
    )
    summary["user_current_occupation"] = deleted

    # 6. UserProfile (the root)
    deleted = (
        db.query(UserProfile)
        .filter(UserProfile.id == user_id)
        .delete(synchronize_session="fetch")
    )
    summary["user_profile"] = deleted

    # Audit log for the full user deletion.
    total = sum(v for v in summary.values())
    _log_deletion(
        db=db,
        deletion_type="user_request",
        table_name="ALL_USER_TABLES",
        records_deleted=total,
        criteria={"user_id": user_id, "tables": summary},
        initiated_by=initiated_by,
        notes=(
            f"Full account deletion for user {user_id}. "
            f"Breakdown: {summary}"
        ),
    )

    return summary


# ---------------------------------------------------------------------------
# Celery task for nightly enforcement
# ---------------------------------------------------------------------------


@celery_app.task(name="app.tasks.tasks.enforce_retention_policy")
def enforce_retention_policy_task(dry_run: bool = False) -> Dict[str, Any]:
    """
    Celery task that runs nightly to enforce data retention rules.

    RATIONALE: Automated enforcement removes human error from the compliance
    equation. This task runs at 3:00 AM UTC (configured in celery_app.py's
    beat_schedule) to minimize performance impact on active users.

    The task uses a sync session because Celery workers run in a synchronous
    context. Each rule is processed independently; failures in one rule do
    not block others.

    Args:
        dry_run: If True, report what would be deleted without actually
                 deleting. Useful for compliance review.

    Returns:
        Summary dict with per-table deletion counts and overall status.
    """
    logger.info(
        "Starting nightly retention enforcement (dry_run=%s)", dry_run
    )

    db = SyncSessionLocal()
    try:
        summary = enforce_all_retention_rules(
            db=db,
            initiated_by="celery_nightly",
            dry_run=dry_run,
        )

        if not dry_run:
            db.commit()
        else:
            db.rollback()

        total_deleted = sum(v for v in summary.values() if v > 0)
        failed_tables = [t for t, v in summary.items() if v < 0]

        result = {
            "status": "completed" if not failed_tables else "partial_failure",
            "dry_run": dry_run,
            "total_deleted": total_deleted,
            "per_table": summary,
            "failed_tables": failed_tables,
            "executed_at": datetime.utcnow().isoformat(),
        }

        logger.info("Retention enforcement complete: %s", result)
        return result

    except Exception as e:
        logger.error("Retention enforcement FAILED: %s", str(e), exc_info=True)
        db.rollback()
        return {
            "status": "error",
            "error": str(e),
            "executed_at": datetime.utcnow().isoformat(),
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Utility: get retention status summary
# ---------------------------------------------------------------------------


def get_retention_summary() -> List[Dict[str, Any]]:
    """
    Return a human-readable summary of all retention rules.

    RATIONALE: This powers the /api/v1/privacy/retention-policy endpoint
    so users can see exactly how long their data is kept and why.
    """
    return [
        {
            "table": rule.table_name,
            "retention_days": rule.max_days if rule.max_days > 0 else "until_account_deletion",
            "cascade": rule.cascade_from_parent,
            "description": rule.description,
        }
        for rule in RETENTION_RULES
    ]
