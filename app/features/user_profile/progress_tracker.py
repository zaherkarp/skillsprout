"""Skill development tracking and re-scoring.

When a user updates their skills we re-run scoring against their saved
occupations and record progress metrics such as ``skills_gained_count``,
``bucket_improvements``, and ``estimated_time_to_ready``.

API endpoint:
    POST /api/v1/skills/update -- update skills and re-score
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.db.session import get_db
from app.ml.scoring import get_baseline_scorer
from app.models.models import (
    Occupation,
    OccupationSkill,
    UserProfile as UserProfileRow,
    UserSkillRating,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/skills", tags=["skills"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SkillUpdate(BaseModel):
    """A single skill rating update."""

    element_id: str
    new_rating: int = Field(..., ge=0, le=4, description="New rating (0-4).")


class SkillsUpdateRequest(BaseModel):
    """Request body for updating skills and triggering re-scoring."""

    user_id: int
    updates: List[SkillUpdate] = Field(
        ..., min_length=1, description="One or more skill updates."
    )


class BucketImprovement(BaseModel):
    """Records a bucket change for a saved occupation."""

    onet_code: str
    occupation_title: Optional[str] = None
    old_bucket: str
    new_bucket: str


class ProgressMetrics(BaseModel):
    """Metrics returned after a skill update."""

    skills_gained_count: int = Field(
        ..., description="Number of skills that improved in this update."
    )
    bucket_improvements: List[BucketImprovement] = Field(
        default_factory=list,
        description="Saved occupations whose bucket improved.",
    )
    estimated_time_to_ready: Optional[int] = Field(
        None,
        description=(
            "Rough estimate in months until the best trainable occupation "
            "reaches ready_now, or None if no trainable occupations."
        ),
    )


class SkillsUpdateResponse(BaseModel):
    """Response for the skill update endpoint."""

    user_id: int
    updated_skills: int
    progress: ProgressMetrics
    updated_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUCKET_ORDER = {"ready_now": 0, "trainable": 1, "long_reskill": 2}


def _is_improvement(old_bucket: str, new_bucket: str) -> bool:
    """Return ``True`` if ``new_bucket`` is strictly better than ``old_bucket``."""
    return BUCKET_ORDER.get(new_bucket, 99) < BUCKET_ORDER.get(old_bucket, 99)


def _estimate_time_to_ready(
    match_score: float,
    gap_severity: float,
    bucket: str,
) -> Optional[int]:
    """Rough heuristic for months to ``ready_now``.

    For ``trainable`` occupations: months ~ gap_severity / 5 (capped 1-24).
    For ``ready_now``: 0.
    For ``long_reskill``: None (too uncertain).

    Args:
        match_score: Current match score (0-100).
        gap_severity: Current gap severity (0-100).
        bucket: Current bucket string.

    Returns:
        Estimated months or ``None``.
    """
    if bucket == "ready_now":
        return 0
    if bucket == "trainable":
        months = max(1, int(gap_severity / 5))
        return min(months, 24)
    return None


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@router.post("/update", response_model=SkillsUpdateResponse)
async def update_skills(
    request: SkillsUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> SkillsUpdateResponse:
    """Update user skill ratings and re-score saved occupations.

    This endpoint:
    1. Persists the new skill ratings.
    2. Re-scores every saved occupation against the updated skills.
    3. Returns progress metrics including bucket improvements.
    """
    user_id = request.user_id

    # Verify user exists.
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # 1. Persist skill updates and count improvements.
    skills_gained = 0
    now = datetime.utcnow()

    for upd in request.updates:
        result = await db.execute(
            select(UserSkillRating).where(
                UserSkillRating.user_id == user_id,
                UserSkillRating.element_id == upd.element_id,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            if upd.new_rating > existing.rating_0_4:
                skills_gained += 1
            existing.rating_0_4 = upd.new_rating
            existing.updated_at = now
        else:
            if upd.new_rating > 0:
                skills_gained += 1
            new_rating = UserSkillRating(
                user_id=user_id,
                element_id=upd.element_id,
                rating_0_4=upd.new_rating,
                updated_at=now,
            )
            db.add(new_rating)

    await db.flush()

    # 2. Reload all user ratings for re-scoring.
    result = await db.execute(
        select(UserSkillRating).where(UserSkillRating.user_id == user_id)
    )
    all_ratings = result.scalars().all()
    user_ratings = {r.element_id: r.rating_0_4 for r in all_ratings}

    # 3. Re-score saved occupations.
    meta = user.metadata_json or {}
    saved_items: List[Dict[str, Any]] = meta.get("saved_occupations", [])

    bucket_improvements: List[BucketImprovement] = []
    best_trainable_time: Optional[int] = None
    scorer = get_baseline_scorer()

    for item in saved_items:
        onet_code = item["onet_code"]
        old_bucket = item.get("current_bucket", item.get("bucket_at_save", "long_reskill"))

        # Fetch occupation and skills.
        result = await db.execute(
            select(Occupation)
            .options(
                joinedload(Occupation.occupation_skills).joinedload(OccupationSkill.skill)
            )
            .where(Occupation.onet_code == onet_code)
        )
        occupation = result.unique().scalar_one_or_none()
        if occupation is None:
            continue

        occ_skills = [
            {
                "element_id": os.skill.element_id,
                "skill_name": os.skill.name,
                "importance": os.importance,
                "level": os.level,
            }
            for os in occupation.occupation_skills
        ]
        if not occ_skills:
            continue

        score = scorer.score_occupation(
            onet_code=onet_code,
            occupation_title=occupation.title,
            occupation_skills=occ_skills,
            user_skill_ratings=user_ratings,
        )

        item["current_bucket"] = score.bucket
        item["match_score"] = score.match_score
        item["gap_severity"] = score.gap_severity
        item["last_rescored_at"] = now.isoformat()

        if _is_improvement(old_bucket, score.bucket):
            bucket_improvements.append(
                BucketImprovement(
                    onet_code=onet_code,
                    occupation_title=occupation.title,
                    old_bucket=old_bucket,
                    new_bucket=score.bucket,
                )
            )

        est = _estimate_time_to_ready(
            score.match_score, score.gap_severity, score.bucket
        )
        if est is not None:
            if best_trainable_time is None or est < best_trainable_time:
                best_trainable_time = est

    meta["saved_occupations"] = saved_items
    user.metadata_json = meta
    user.updated_at = now

    await db.commit()
    await db.refresh(user)

    progress = ProgressMetrics(
        skills_gained_count=skills_gained,
        bucket_improvements=bucket_improvements,
        estimated_time_to_ready=best_trainable_time,
    )

    logger.info(
        "User %d updated %d skills (%d gains, %d bucket improvements)",
        user_id,
        len(request.updates),
        skills_gained,
        len(bucket_improvements),
    )

    return SkillsUpdateResponse(
        user_id=user_id,
        updated_skills=len(request.updates),
        progress=progress,
        updated_at=now,
    )
