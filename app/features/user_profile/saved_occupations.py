"""Saved occupations with progress tracking.

Users can save occupations they are interested in and track how their
readiness (bucket assignment) evolves over time as they develop skills.

API endpoints:
    POST   /api/v1/saved-occupations              -- save an occupation
    GET    /api/v1/saved-occupations/{user_id}     -- list saved occupations
    PATCH  /api/v1/saved-occupations/{saved_id}    -- update notes / status
    DELETE /api/v1/saved-occupations/{saved_id}    -- remove saved occupation

A weekly Celery task re-scores all saved occupations to reflect skill
development progress.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db, SyncSessionLocal
from app.models.models import (
    UserProfile as UserProfileRow,
    Occupation,
    OccupationSkill,
    UserSkillRating,
)
from app.ml.scoring import BaselineScorer, get_baseline_scorer
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/saved-occupations", tags=["saved-occupations"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TrainingStatus(str, Enum):
    """Current training status for a saved occupation."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SaveOccupationRequest(BaseModel):
    """Request to save an occupation for tracking."""

    user_id: int = Field(..., description="User who is saving the occupation.")
    onet_code: str = Field(..., max_length=10, description="O*NET occupation code.")
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="User notes about why they saved this occupation.",
    )


class SavedOccupationUpdate(BaseModel):
    """Partial update for a saved occupation."""

    notes: Optional[str] = Field(None, max_length=2000)
    training_status: Optional[TrainingStatus] = None


class SavedOccupationResponse(BaseModel):
    """Response for a single saved occupation."""

    saved_id: str
    user_id: int
    onet_code: str
    occupation_title: Optional[str] = None
    bucket_at_save: str
    current_bucket: str
    match_score: float
    gap_severity: float
    notes: Optional[str] = None
    training_status: TrainingStatus = TrainingStatus.NOT_STARTED
    saved_at: datetime
    last_rescored_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SavedOccupationListResponse(BaseModel):
    """Response for listing all saved occupations."""

    user_id: int
    total: int
    occupations: List[SavedOccupationResponse]


# ---------------------------------------------------------------------------
# In-memory store (production would use a dedicated DB table)
# ---------------------------------------------------------------------------
# The existing DB schema has no ``saved_occupation`` table, so we store
# saved-occupation state in the ``UserProfile.metadata_json`` under a
# ``"saved_occupations"`` key.  This keeps everything in one place without
# requiring a migration in this iteration.

def _get_saved_list(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the saved occupations list from profile metadata."""
    return meta.get("saved_occupations", [])


def _set_saved_list(
    meta: Dict[str, Any], items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Write the saved occupations list back into profile metadata."""
    meta["saved_occupations"] = items
    return meta


def _next_saved_id(items: List[Dict[str, Any]]) -> str:
    """Generate a simple auto-incrementing ID string."""
    if not items:
        return "saved_1"
    max_num = max(int(i["saved_id"].split("_")[1]) for i in items)
    return f"saved_{max_num + 1}"


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

async def _score_occupation_for_user(
    db: AsyncSession,
    user_id: int,
    onet_code: str,
) -> Dict[str, Any]:
    """Score an occupation against the user's current skills.

    Returns a dict with ``match_score``, ``gap_severity``, and ``bucket``.
    """
    from sqlalchemy.orm import joinedload

    # Fetch user skill ratings.
    result = await db.execute(
        select(UserSkillRating).where(UserSkillRating.user_id == user_id)
    )
    ratings = result.scalars().all()
    user_ratings = {r.element_id: r.rating_0_4 for r in ratings}

    # Fetch occupation skills.
    result = await db.execute(
        select(Occupation)
        .options(
            joinedload(Occupation.occupation_skills).joinedload(OccupationSkill.skill)
        )
        .where(Occupation.onet_code == onet_code)
    )
    occupation = result.unique().scalar_one_or_none()

    if occupation is None:
        return {"match_score": 0.0, "gap_severity": 100.0, "bucket": "long_reskill", "title": None}

    occ_skills = [
        {
            "element_id": os.skill.element_id,
            "skill_name": os.skill.name,
            "importance": os.importance,
            "level": os.level,
        }
        for os in occupation.occupation_skills
    ]

    scorer = get_baseline_scorer()
    score = scorer.score_occupation(
        onet_code=onet_code,
        occupation_title=occupation.title,
        occupation_skills=occ_skills,
        user_skill_ratings=user_ratings,
    )

    return {
        "match_score": score.match_score,
        "gap_severity": score.gap_severity,
        "bucket": score.bucket,
        "title": occupation.title,
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=SavedOccupationResponse, status_code=201)
async def save_occupation(
    request: SaveOccupationRequest,
    db: AsyncSession = Depends(get_db),
) -> SavedOccupationResponse:
    """Save an occupation for progress tracking."""
    # Verify user exists.
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == request.user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Score the occupation.
    scoring = await _score_occupation_for_user(db, request.user_id, request.onet_code)

    meta = user.metadata_json or {}
    items = _get_saved_list(meta)

    # Prevent duplicate saves.
    for item in items:
        if item["onet_code"] == request.onet_code:
            raise HTTPException(
                status_code=409, detail="Occupation already saved"
            )

    now = datetime.utcnow()
    saved_id = _next_saved_id(items)

    entry: Dict[str, Any] = {
        "saved_id": saved_id,
        "user_id": request.user_id,
        "onet_code": request.onet_code,
        "occupation_title": scoring["title"],
        "bucket_at_save": scoring["bucket"],
        "current_bucket": scoring["bucket"],
        "match_score": scoring["match_score"],
        "gap_severity": scoring["gap_severity"],
        "notes": request.notes,
        "training_status": TrainingStatus.NOT_STARTED.value,
        "saved_at": now.isoformat(),
        "last_rescored_at": now.isoformat(),
    }
    items.append(entry)
    _set_saved_list(meta, items)
    user.metadata_json = meta
    user.updated_at = now

    await db.commit()
    await db.refresh(user)

    logger.info(
        "User %d saved occupation %s (bucket=%s)",
        request.user_id,
        request.onet_code,
        scoring["bucket"],
    )

    return SavedOccupationResponse(
        saved_id=saved_id,
        user_id=request.user_id,
        onet_code=request.onet_code,
        occupation_title=scoring["title"],
        bucket_at_save=scoring["bucket"],
        current_bucket=scoring["bucket"],
        match_score=scoring["match_score"],
        gap_severity=scoring["gap_severity"],
        notes=request.notes,
        training_status=TrainingStatus.NOT_STARTED,
        saved_at=now,
        last_rescored_at=now,
    )


@router.get("/{user_id}", response_model=SavedOccupationListResponse)
async def list_saved_occupations(
    user_id: int,
    db: AsyncSession = Depends(get_db),
) -> SavedOccupationListResponse:
    """List all saved occupations for a user."""
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    meta = user.metadata_json or {}
    items = _get_saved_list(meta)

    occupations = [
        SavedOccupationResponse(
            saved_id=item["saved_id"],
            user_id=user_id,
            onet_code=item["onet_code"],
            occupation_title=item.get("occupation_title"),
            bucket_at_save=item["bucket_at_save"],
            current_bucket=item["current_bucket"],
            match_score=item.get("match_score", 0.0),
            gap_severity=item.get("gap_severity", 100.0),
            notes=item.get("notes"),
            training_status=TrainingStatus(
                item.get("training_status", TrainingStatus.NOT_STARTED.value)
            ),
            saved_at=datetime.fromisoformat(item["saved_at"]),
            last_rescored_at=(
                datetime.fromisoformat(item["last_rescored_at"])
                if item.get("last_rescored_at")
                else None
            ),
        )
        for item in items
    ]

    return SavedOccupationListResponse(
        user_id=user_id,
        total=len(occupations),
        occupations=occupations,
    )


@router.patch("/{saved_id}", response_model=SavedOccupationResponse)
async def update_saved_occupation(
    saved_id: str,
    request: SavedOccupationUpdate,
    user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
) -> SavedOccupationResponse:
    """Update notes or training status for a saved occupation."""
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    meta = user.metadata_json or {}
    items = _get_saved_list(meta)

    target = None
    for item in items:
        if item["saved_id"] == saved_id:
            target = item
            break

    if target is None:
        raise HTTPException(status_code=404, detail="Saved occupation not found")

    if request.notes is not None:
        target["notes"] = request.notes
    if request.training_status is not None:
        target["training_status"] = request.training_status.value

    _set_saved_list(meta, items)
    user.metadata_json = meta
    user.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(user)

    return SavedOccupationResponse(
        saved_id=target["saved_id"],
        user_id=user_id,
        onet_code=target["onet_code"],
        occupation_title=target.get("occupation_title"),
        bucket_at_save=target["bucket_at_save"],
        current_bucket=target["current_bucket"],
        match_score=target.get("match_score", 0.0),
        gap_severity=target.get("gap_severity", 100.0),
        notes=target.get("notes"),
        training_status=TrainingStatus(
            target.get("training_status", TrainingStatus.NOT_STARTED.value)
        ),
        saved_at=datetime.fromisoformat(target["saved_at"]),
        last_rescored_at=(
            datetime.fromisoformat(target["last_rescored_at"])
            if target.get("last_rescored_at")
            else None
        ),
    )


@router.delete("/{saved_id}", status_code=204)
async def delete_saved_occupation(
    saved_id: str,
    user_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Remove a saved occupation."""
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    meta = user.metadata_json or {}
    items = _get_saved_list(meta)
    original_len = len(items)
    items = [i for i in items if i["saved_id"] != saved_id]

    if len(items) == original_len:
        raise HTTPException(status_code=404, detail="Saved occupation not found")

    _set_saved_list(meta, items)
    user.metadata_json = meta
    user.updated_at = datetime.utcnow()

    await db.commit()
    logger.info("Deleted saved occupation %s for user %d", saved_id, user_id)


# ---------------------------------------------------------------------------
# Celery task: weekly re-scoring
# ---------------------------------------------------------------------------

@celery_app.task(name="app.features.user_profile.saved_occupations.rescore_saved_occupations")
def rescore_saved_occupations() -> Dict[str, Any]:
    """Re-score all saved occupations for all users.

    This Celery task runs weekly to update the ``current_bucket``,
    ``match_score``, and ``gap_severity`` for every saved occupation,
    reflecting any skill development since the occupation was saved.

    Returns:
        Summary dict with counts of users processed and items rescored.
    """
    logger.info("Starting weekly saved-occupation re-scoring")
    db = SyncSessionLocal()

    users_processed = 0
    items_rescored = 0

    try:
        users = db.query(UserProfileRow).all()

        for user in users:
            meta = user.metadata_json or {}
            items = _get_saved_list(meta)
            if not items:
                continue

            # Fetch user skill ratings (sync).
            ratings_rows = (
                db.query(UserSkillRating)
                .filter(UserSkillRating.user_id == user.id)
                .all()
            )
            user_ratings = {r.element_id: r.rating_0_4 for r in ratings_rows}

            scorer = get_baseline_scorer()
            now = datetime.utcnow()

            for item in items:
                onet_code = item["onet_code"]
                occupation = (
                    db.query(Occupation)
                    .filter(Occupation.onet_code == onet_code)
                    .first()
                )
                if occupation is None:
                    continue

                occ_skills_rows = (
                    db.query(OccupationSkill)
                    .filter(OccupationSkill.onet_code == onet_code)
                    .all()
                )
                occ_skills = [
                    {
                        "element_id": os.element_id,
                        "skill_name": (os.skill.name if hasattr(os, 'skill') and os.skill else os.element_id),
                        "importance": os.importance,
                        "level": os.level,
                    }
                    for os in occ_skills_rows
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
                items_rescored += 1

            _set_saved_list(meta, items)
            user.metadata_json = meta
            user.updated_at = now
            users_processed += 1

        db.commit()
        logger.info(
            "Re-scoring complete: %d users, %d items rescored",
            users_processed,
            items_rescored,
        )
        return {
            "status": "completed",
            "users_processed": users_processed,
            "items_rescored": items_rescored,
        }

    except Exception as exc:
        logger.error("Re-scoring failed: %s", exc, exc_info=True)
        db.rollback()
        return {"status": "error", "error": str(exc)}
    finally:
        db.close()
