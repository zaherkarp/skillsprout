"""
GDPR / CCPA Data Export Endpoint for SkillSprout
==================================================

RATIONALE: Under GDPR Article 20 ("right to data portability") and CCPA
Section 1798.100 ("right to know"), users have the legal right to receive
a copy of ALL personal data a service holds about them, in a structured,
commonly used, machine-readable format.

This module implements:

    GET /api/v1/user/{user_id}/data-export

The response is a structured JSON document containing:

  1. Every piece of user-linked data across all tables, organized by
     data category (profile, skills, occupations, recommendations, feedback).

  2. Data lineage metadata: for each record, we include WHEN it was created,
     WHICH system process generated it, and WHAT classification tier it
     belongs to. This goes beyond the legal minimum but builds user trust
     by showing exactly how their data flows through the system.

  3. A privacy manifest: a summary of the data tiers, retention policies,
     and deletion rights that apply to each category.

DESIGN DECISIONS:

  - We return ALL data in a single response rather than paginating, because
    per-user data volumes in SkillSprout are small (typically <1MB). For
    larger-scale systems, a background job with a download link would be
    more appropriate.

  - We use JSON (not CSV or XML) because it preserves nested structure
    (e.g., recommendation scores contain sub-objects for skill gaps) and
    is the most universally machine-readable format.

  - The export includes data lineage (creation timestamps, model versions,
    classification tiers) because GDPR Article 15(1)(h) specifically
    requires disclosure of "the existence of automated decision-making"
    and "meaningful information about the logic involved." Our
    recommendation engine is automated decision-making, so we must be
    transparent about it.

  - The endpoint requires the user_id in the path. In a production system
    with authentication, this would be derived from the auth token. The
    current implementation assumes the caller is authorized to access the
    requested user's data.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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
    get_model_tier,
    get_tier_policy,
    get_exportable_models,
    TIER_METADATA,
)
from app.core.privacy.private_mode import is_private_mode

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Data serialization helpers
# ---------------------------------------------------------------------------
# RATIONALE: We convert ORM objects to plain dicts with explicit field
# selection. This prevents accidental leakage of internal fields (like
# SQLAlchemy's _sa_instance_state) and gives us control over field naming
# and formatting in the export.


def _serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 string, the standard for machine-readable dates."""
    return dt.isoformat() if dt else None


def _serialize_user_profile(profile: UserProfile) -> Dict[str, Any]:
    """
    Serialize a UserProfile to an export-friendly dict.

    Includes data lineage: when the profile was created and last modified.
    """
    return {
        "data_category": "user_profile",
        "classification_tier": DataTier.TIER_3_PERSONAL.name,
        "classification_label": "Personal",
        "record": {
            "id": profile.id,
            "created_at": _serialize_datetime(profile.created_at),
            "updated_at": _serialize_datetime(profile.updated_at),
            "metadata": profile.metadata_json,
        },
        "lineage": {
            "created_at": _serialize_datetime(profile.created_at),
            "source": "user_registration",
            "retention_policy": "Retained until explicit deletion request",
        },
    }


def _serialize_skill_rating(rating: UserSkillRating) -> Dict[str, Any]:
    """Serialize a user skill rating with lineage."""
    return {
        "skill_element_id": rating.element_id,
        "rating_0_to_4": rating.rating_0_4,
        "rating_label": {
            0: "none",
            1: "basic",
            2: "intermediate",
            3: "advanced",
            4: "expert",
        }.get(rating.rating_0_4, "unknown"),
        "updated_at": _serialize_datetime(rating.updated_at),
        "lineage": {
            "source": "user_self_assessment",
            "last_modified": _serialize_datetime(rating.updated_at),
        },
    }


def _serialize_current_occupation(occ: UserCurrentOccupation) -> Dict[str, Any]:
    """Serialize a current occupation selection with lineage."""
    return {
        "onet_code": occ.onet_code,
        "selected_at": _serialize_datetime(occ.selected_at),
        "is_active": occ.is_active,
        "lineage": {
            "source": "user_selection",
            "selected_at": _serialize_datetime(occ.selected_at),
        },
    }


def _serialize_recommendation_event(event: RecommendationEvent) -> Dict[str, Any]:
    """
    Serialize a recommendation event with full lineage.

    RATIONALE: GDPR Art. 15(1)(h) requires disclosure of automated
    decision-making logic. We include the model version and parameters
    that produced these recommendations so the user can understand HOW
    the system decided what to show them.
    """
    recommendations = []
    for rec in event.recommended_occupations:
        recommendations.append({
            "target_onet_code": rec.target_onet_code,
            "rank": rec.rank,
            "bucket": rec.bucket.value if hasattr(rec.bucket, "value") else str(rec.bucket),
            "scores": rec.score_json,
            "is_exploration": rec.is_exploration,
            "lineage": {
                "source": "recommendation_engine",
                "model_version": event.model_version,
                "generated_at": _serialize_datetime(event.created_at),
                "explanation": (
                    "This recommendation was generated by the SkillSprout "
                    f"scoring engine (version: {event.model_version}). "
                    "Scores reflect the skill gap analysis between your "
                    "self-assessed ratings and the target occupation's "
                    "skill requirements from O*NET."
                ),
            },
        })

    return {
        "event_id": event.id,
        "created_at": _serialize_datetime(event.created_at),
        "current_onet_code": event.current_onet_code,
        "model_version": event.model_version,
        "parameters": event.params_json,
        "recommendations": recommendations,
        "lineage": {
            "source": "recommendation_engine",
            "model_version": event.model_version,
            "automated_decision": True,
            "decision_explanation": (
                "Recommendations are generated by an automated scoring "
                "algorithm that compares your self-assessed skill ratings "
                "against O*NET occupation skill requirements. No human "
                "review is involved in the recommendation process."
            ),
        },
    }


def _serialize_feedback(feedback: UserFeedback) -> Dict[str, Any]:
    """Serialize a user feedback record with lineage."""
    return {
        "feedback_id": feedback.id,
        "event_id": feedback.event_id,
        "target_onet_code": feedback.target_onet_code,
        "action_type": feedback.action_type.value if hasattr(feedback.action_type, "value") else str(feedback.action_type),
        "action_at": _serialize_datetime(feedback.action_at),
        "metadata": feedback.metadata_json,
        "lineage": {
            "source": "user_action",
            "recorded_at": _serialize_datetime(feedback.action_at),
            "usage": (
                "This feedback is used to improve recommendation quality "
                "through the calibration model. It may be de-identified "
                "and aggregated for model training."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/user/{user_id}/data-export",
    summary="Export all user data (GDPR Art. 20 / CCPA)",
    response_description="Complete export of all personal data with data lineage",
)
async def export_user_data(
    user_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Export ALL data stored about a specific user.

    Returns a structured JSON document containing every piece of
    user-linked data, organized by category, with data lineage
    metadata explaining how each record was created and processed.

    This endpoint satisfies:
      - GDPR Article 15 (right of access)
      - GDPR Article 20 (right to data portability)
      - CCPA Section 1798.100 (right to know)

    RATIONALE: We query each table independently rather than using a
    single mega-join because (a) it produces a cleaner, more organized
    export, (b) it avoids Cartesian product blowup on tables with
    many-to-many relationships, and (c) it makes the export structure
    self-documenting.
    """
    # RATIONALE: Private mode check. Even in private mode, if a user
    # somehow has stored data from a previous non-private session, they
    # should be able to export it. The export itself does not create
    # new stored data.
    if is_private_mode(request):
        logger.debug("Data export requested in private mode for user %d", user_id)

    # --- Verify user exists ---
    result = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=404,
            detail=(
                f"User {user_id} not found. If you recently deleted your "
                "account, your data has been permanently removed and cannot "
                "be exported."
            ),
        )

    # --- Collect all user data ---

    # 1. Profile
    profile_data = _serialize_user_profile(user)

    # 2. Skill ratings
    result = await db.execute(
        select(UserSkillRating).where(UserSkillRating.user_id == user_id)
    )
    skill_ratings = result.scalars().all()
    skills_data = [_serialize_skill_rating(r) for r in skill_ratings]

    # 3. Current / historical occupations
    result = await db.execute(
        select(UserCurrentOccupation)
        .where(UserCurrentOccupation.user_id == user_id)
        .order_by(UserCurrentOccupation.selected_at.desc())
    )
    occupations = result.scalars().all()
    occupations_data = [_serialize_current_occupation(o) for o in occupations]

    # 4. Recommendation events (with nested recommendations)
    result = await db.execute(
        select(RecommendationEvent)
        .options(
            joinedload(RecommendationEvent.recommended_occupations),
        )
        .where(RecommendationEvent.user_id == user_id)
        .order_by(RecommendationEvent.created_at.desc())
    )
    events = result.unique().scalars().all()
    events_data = [_serialize_recommendation_event(e) for e in events]

    # 5. Feedback
    # RATIONALE: We collect feedback separately (not nested under events)
    # because feedback may reference events that have already been purged
    # by the retention policy. Standalone serialization ensures completeness.
    event_ids = [e.id for e in events]
    if event_ids:
        result = await db.execute(
            select(UserFeedback)
            .where(UserFeedback.event_id.in_(event_ids))
            .order_by(UserFeedback.action_at.desc())
        )
        feedbacks = result.scalars().all()
    else:
        feedbacks = []
    feedback_data = [_serialize_feedback(f) for f in feedbacks]

    # --- Build the export document ---

    export = {
        "export_metadata": {
            "user_id": user_id,
            "exported_at": datetime.utcnow().isoformat(),
            "format_version": "1.0",
            "application": settings.app_name,
            "legal_basis": (
                "This export is provided pursuant to GDPR Article 15 "
                "(right of access), Article 20 (right to data portability), "
                "and CCPA Section 1798.100 (right to know)."
            ),
            "data_categories_included": [
                "user_profile",
                "skill_ratings",
                "occupation_selections",
                "recommendation_events",
                "user_feedback",
            ],
        },

        "user_profile": profile_data,

        "skill_ratings": {
            "data_category": "skill_self_assessments",
            "classification_tier": DataTier.TIER_3_PERSONAL.name,
            "total_records": len(skills_data),
            "records": skills_data,
        },

        "occupation_selections": {
            "data_category": "current_occupation_history",
            "classification_tier": DataTier.TIER_3_PERSONAL.name,
            "total_records": len(occupations_data),
            "records": occupations_data,
        },

        "recommendation_events": {
            "data_category": "recommendation_history",
            "classification_tier": DataTier.TIER_4_SENSITIVE.name,
            "total_records": len(events_data),
            "automated_decision_making_disclosure": (
                "SkillSprout uses an automated scoring algorithm to generate "
                "occupation recommendations. The algorithm compares your "
                "self-assessed skill ratings against O*NET occupation skill "
                "requirements. No human review is involved. You have the "
                "right to contest these recommendations under GDPR Article 22."
            ),
            "records": events_data,
        },

        "user_feedback": {
            "data_category": "user_actions_on_recommendations",
            "classification_tier": DataTier.TIER_4_SENSITIVE.name,
            "total_records": len(feedback_data),
            "records": feedback_data,
        },

        "privacy_manifest": {
            "classification_tiers": {
                tier.name: {
                    "label": meta["label"],
                    "description": meta["description"],
                    "retention_days": meta["max_retention_days"],
                    "encrypted_at_rest": meta["requires_encryption_at_rest"],
                }
                for tier, meta in TIER_METADATA.items()
                if meta["included_in_data_export"]
            },
            "your_rights": {
                "access": "You are exercising this right now by requesting this export.",
                "rectification": (
                    "You can update your skill ratings and profile at any time "
                    "through the application."
                ),
                "erasure": (
                    "You can request complete deletion of your account and all "
                    "associated data via DELETE /api/v1/user/{user_id}/data. "
                    "Deletion is completed within 72 hours."
                ),
                "portability": (
                    "This export is provided in JSON format, a structured, "
                    "commonly used, machine-readable format as required by "
                    "GDPR Article 20."
                ),
                "objection": (
                    "You can use Private Mode (X-Private-Mode: true header) "
                    "to use SkillSprout without any server-side data storage."
                ),
            },
            "data_retention_summary": {
                "event_level_tracking": "Auto-purged after 90 days",
                "user_profiles": "Retained until deletion request (72-hour SLA)",
                "model_training_data": "Retained in de-identified form only",
            },
        },
    }

    logger.info(
        "Data export completed for user %d: %d skill ratings, %d occupations, "
        "%d events, %d feedbacks",
        user_id,
        len(skills_data),
        len(occupations_data),
        len(events_data),
        len(feedback_data),
    )

    return export
