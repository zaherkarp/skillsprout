"""Return engagement: progress summary for returning users.

Provides a single endpoint that gives a quick overview of a user's
journey so far, designed to re-engage users who come back after a
period away.

API endpoint:
    GET /api/v1/progress/summary/{user_id}

Returns:
    days_active, skills_developed, occupations_tracked,
    bucket_improvements, next_milestone.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models.models import (
    UserProfile as UserProfileRow,
    UserSkillRating,
    RecommendationEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/progress", tags=["progress"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class NextMilestone(BaseModel):
    """Description of the user's next achievable milestone."""

    description: str
    target_metric: str
    current_value: float
    target_value: float
    progress_pct: float = Field(
        ..., ge=0, le=100, description="Progress toward the milestone (0-100)."
    )


class ProgressSummaryResponse(BaseModel):
    """Aggregated progress summary for a returning user."""

    user_id: int
    days_active: int = Field(
        ..., description="Days since profile creation."
    )
    skills_developed: int = Field(
        ..., description="Number of skills the user has rated."
    )
    occupations_tracked: int = Field(
        ..., description="Number of saved occupations."
    )
    bucket_improvements: int = Field(
        ...,
        description="Total bucket upgrades across all saved occupations.",
    )
    recommendation_events: int = Field(
        ..., description="Total recommendation sessions."
    )
    next_milestone: Optional[NextMilestone] = Field(
        None, description="Suggested next milestone."
    )
    summary_text: str = Field(
        ..., description="Human-readable summary paragraph."
    )
    generated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_bucket_improvements(saved_items: List[Dict[str, Any]]) -> int:
    """Count occupations where the current bucket is better than at save.

    Bucket ordering: ready_now(0) < trainable(1) < long_reskill(2).
    An improvement means the number went *down*.
    """
    order = {"ready_now": 0, "trainable": 1, "long_reskill": 2}
    improvements = 0
    for item in saved_items:
        at_save = order.get(item.get("bucket_at_save", "long_reskill"), 2)
        current = order.get(item.get("current_bucket", "long_reskill"), 2)
        if current < at_save:
            improvements += 1
    return improvements


def _determine_next_milestone(
    skills_developed: int,
    occupations_tracked: int,
    bucket_improvements: int,
    saved_items: List[Dict[str, Any]],
) -> Optional[NextMilestone]:
    """Pick the most relevant next milestone for the user.

    The priority order is:
    1. Rate at least 5 skills (onboarding).
    2. Save at least 3 occupations.
    3. Improve a bucket (trainable -> ready_now or long_reskill -> trainable).
    4. Reach 10 skills developed.
    """
    if skills_developed < 5:
        return NextMilestone(
            description="Rate at least 5 skills to unlock personalised recommendations.",
            target_metric="skills_developed",
            current_value=float(skills_developed),
            target_value=5.0,
            progress_pct=min(100.0, (skills_developed / 5.0) * 100),
        )

    if occupations_tracked < 3:
        return NextMilestone(
            description="Save 3 occupations to start tracking your progress.",
            target_metric="occupations_tracked",
            current_value=float(occupations_tracked),
            target_value=3.0,
            progress_pct=min(100.0, (occupations_tracked / 3.0) * 100),
        )

    # Check if there are trainable occupations that could become ready_now.
    trainable_items = [
        i for i in saved_items
        if i.get("current_bucket") == "trainable"
    ]
    if trainable_items and bucket_improvements == 0:
        return NextMilestone(
            description="Develop skills to move a trainable occupation to ready_now.",
            target_metric="bucket_improvements",
            current_value=0.0,
            target_value=1.0,
            progress_pct=0.0,
        )

    if skills_developed < 10:
        return NextMilestone(
            description="Rate 10 skills for a comprehensive career map.",
            target_metric="skills_developed",
            current_value=float(skills_developed),
            target_value=10.0,
            progress_pct=min(100.0, (skills_developed / 10.0) * 100),
        )

    return None


def _build_summary_text(
    days_active: int,
    skills_developed: int,
    occupations_tracked: int,
    bucket_improvements: int,
) -> str:
    """Generate a friendly, plain-language progress summary."""
    parts: List[str] = []

    if days_active <= 1:
        parts.append("Welcome to SkillSprout!")
    else:
        parts.append(f"You have been active for {days_active} day(s).")

    if skills_developed == 0:
        parts.append("Start by rating your skills to get personalised recommendations.")
    else:
        parts.append(f"You have rated {skills_developed} skill(s) so far.")

    if occupations_tracked > 0:
        parts.append(
            f"You are tracking {occupations_tracked} occupation(s)."
        )
        if bucket_improvements > 0:
            parts.append(
                f"Great news: {bucket_improvements} occupation(s) have moved "
                "to a better readiness bucket since you started tracking them."
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@router.get("/summary/{user_id}", response_model=ProgressSummaryResponse)
async def get_progress_summary(
    user_id: int,
    db: AsyncSession = Depends(get_db),
) -> ProgressSummaryResponse:
    """Return a progress summary for a returning user.

    This endpoint aggregates data from the user's profile, skill ratings,
    saved occupations, and recommendation history into a single snapshot
    designed to quickly re-engage users.
    """
    # Fetch user profile.
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    now = datetime.utcnow()

    # Days active.
    days_active = max(1, (now - user.created_at).days)

    # Skills developed (count of rated skills).
    result = await db.execute(
        select(func.count(UserSkillRating.id)).where(
            UserSkillRating.user_id == user_id
        )
    )
    skills_developed = result.scalar() or 0

    # Recommendation events.
    result = await db.execute(
        select(func.count(RecommendationEvent.id)).where(
            RecommendationEvent.user_id == user_id
        )
    )
    recommendation_events = result.scalar() or 0

    # Saved occupations and bucket improvements.
    meta = user.metadata_json or {}
    saved_items: List[Dict[str, Any]] = meta.get("saved_occupations", [])
    occupations_tracked = len(saved_items)
    bucket_improvements = _count_bucket_improvements(saved_items)

    # Next milestone.
    next_milestone = _determine_next_milestone(
        skills_developed=skills_developed,
        occupations_tracked=occupations_tracked,
        bucket_improvements=bucket_improvements,
        saved_items=saved_items,
    )

    summary_text = _build_summary_text(
        days_active=days_active,
        skills_developed=skills_developed,
        occupations_tracked=occupations_tracked,
        bucket_improvements=bucket_improvements,
    )

    logger.info(
        "Progress summary for user %d: %d days, %d skills, %d occupations, %d improvements",
        user_id,
        days_active,
        skills_developed,
        occupations_tracked,
        bucket_improvements,
    )

    return ProgressSummaryResponse(
        user_id=user_id,
        days_active=days_active,
        skills_developed=skills_developed,
        occupations_tracked=occupations_tracked,
        bucket_improvements=bucket_improvements,
        recommendation_events=recommendation_events,
        next_milestone=next_milestone,
        summary_text=summary_text,
        generated_at=now,
    )
