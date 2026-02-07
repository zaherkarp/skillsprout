"""Offline capability -- static CSV export of recommendations.

Provides a downloadable summary containing the user's skill profile,
top-10 recommended occupations, and suggested training paths so the
user can work offline or share the results.
"""
import csv
import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.db.session import get_db
from app.models.models import (
    Occupation,
    RecommendationEvent,
    RecommendedOccupation,
    UserCurrentOccupation,
    UserProfile,
    UserSkillRating,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["offline"])


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------

def _build_profile_section(
    user_id: int,
    current_occupation_title: str,
    current_onet_code: str,
    skill_ratings: List[Dict[str, Any]],
) -> List[List[str]]:
    """Build the user-profile section of the CSV.

    Args:
        user_id: The user ID.
        current_occupation_title: Title of the user's current occupation.
        current_onet_code: O*NET code of the current occupation.
        skill_ratings: List of dicts with element_id, skill_name, rating.

    Returns:
        A list of CSV rows (each row is a list of strings).
    """
    rows: List[List[str]] = [
        ["=== USER PROFILE ===", "", "", ""],
        ["User ID", str(user_id), "", ""],
        ["Current Occupation", current_occupation_title, current_onet_code, ""],
        ["Export Date", datetime.utcnow().isoformat(), "", ""],
        ["", "", "", ""],
        ["=== SKILL RATINGS ===", "", "", ""],
        ["Element ID", "Skill Name", "Rating (0-4)", ""],
    ]
    for sr in skill_ratings:
        rows.append([
            sr.get("element_id", ""),
            sr.get("skill_name", ""),
            str(sr.get("rating", "")),
            "",
        ])
    rows.append(["", "", "", ""])
    return rows


def _build_recommendations_section(
    recommendations: List[Dict[str, Any]],
) -> List[List[str]]:
    """Build the recommendations section of the CSV.

    Args:
        recommendations: List of recommendation dicts.

    Returns:
        A list of CSV rows.
    """
    rows: List[List[str]] = [
        ["=== TOP RECOMMENDATIONS ===", "", "", "", "", ""],
        ["Rank", "O*NET Code", "Title", "Bucket", "Match Score", "Training Path"],
    ]
    for rec in recommendations:
        rows.append([
            str(rec.get("rank", "")),
            rec.get("onet_code", ""),
            rec.get("title", ""),
            rec.get("bucket", ""),
            str(rec.get("match_score", "")),
            rec.get("training_suggestion", ""),
        ])
    rows.append(["", "", "", "", "", ""])
    return rows


def generate_csv(
    user_id: int,
    current_occupation_title: str,
    current_onet_code: str,
    skill_ratings: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
) -> str:
    """Generate a complete CSV document.

    Args:
        user_id: User ID.
        current_occupation_title: Current occupation title.
        current_onet_code: Current O*NET code.
        skill_ratings: Skill ratings list.
        recommendations: Recommendations list.

    Returns:
        The CSV content as a string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["SkillSprout Career Transition Report", "", "", "", "", ""])
    writer.writerow(["Generated", datetime.utcnow().isoformat(), "", "", "", ""])
    writer.writerow(["", "", "", "", "", ""])

    for row in _build_profile_section(
        user_id, current_occupation_title, current_onet_code, skill_ratings
    ):
        writer.writerow(row)

    for row in _build_recommendations_section(recommendations):
        writer.writerow(row)

    writer.writerow(["=== END OF REPORT ===", "", "", "", "", ""])

    return output.getvalue()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/api/v1/recommendations/export")
async def export_recommendations(
    user_id: int = Query(..., description="User ID to export recommendations for"),
    format: str = Query("csv", description="Export format (currently only csv)"),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export user profile and top-10 recommendations as a downloadable CSV.

    Generates a self-contained summary including:
      - User's current occupation
      - Self-assessed skill ratings
      - Top 10 recommended occupations with match scores
      - Suggested training paths for each recommendation
    """
    if format.lower() != "csv":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported export format: {format}. Use 'csv'.",
        )

    # Fetch user
    result = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch current occupation
    result = await db.execute(
        select(UserCurrentOccupation)
        .options(joinedload(UserCurrentOccupation.occupation))
        .where(
            and_(
                UserCurrentOccupation.user_id == user_id,
                UserCurrentOccupation.is_active == True,
            )
        )
        .order_by(UserCurrentOccupation.selected_at.desc())
    )
    current_occ = result.scalar_one_or_none()
    if not current_occ:
        raise HTTPException(
            status_code=400,
            detail="User has no current occupation set",
        )

    # Fetch skill ratings
    result = await db.execute(
        select(UserSkillRating)
        .where(UserSkillRating.user_id == user_id)
    )
    ratings = result.scalars().all()
    skill_ratings_data: List[Dict[str, Any]] = []
    for r in ratings:
        # Try to get skill name
        skill_name = r.element_id  # fallback
        try:
            from app.models.models import Skill

            sk_result = await db.execute(
                select(Skill).where(Skill.element_id == r.element_id)
            )
            skill = sk_result.scalar_one_or_none()
            if skill:
                skill_name = skill.name
        except Exception:
            pass
        skill_ratings_data.append({
            "element_id": r.element_id,
            "skill_name": skill_name,
            "rating": r.rating_0_4,
        })

    # Fetch most recent recommendation event
    result = await db.execute(
        select(RecommendationEvent)
        .where(RecommendationEvent.user_id == user_id)
        .order_by(RecommendationEvent.created_at.desc())
    )
    event = result.scalar_one_or_none()

    recommendations_data: List[Dict[str, Any]] = []
    if event:
        result = await db.execute(
            select(RecommendedOccupation)
            .options(joinedload(RecommendedOccupation.occupation))
            .where(RecommendedOccupation.event_id == event.id)
            .order_by(RecommendedOccupation.rank)
        )
        recs = result.scalars().all()

        for rec in recs[:10]:
            score_json = rec.score_json or {}
            recommendations_data.append({
                "rank": rec.rank,
                "onet_code": rec.target_onet_code,
                "title": rec.occupation.title if rec.occupation else rec.target_onet_code,
                "bucket": rec.bucket.value if hasattr(rec.bucket, "value") else str(rec.bucket),
                "match_score": round(score_json.get("match_score", 0), 1),
                "training_suggestion": score_json.get("training_suggestion", ""),
            })

    csv_content = generate_csv(
        user_id=user_id,
        current_occupation_title=current_occ.occupation.title,
        current_onet_code=current_occ.onet_code,
        skill_ratings=skill_ratings_data,
        recommendations=recommendations_data,
    )

    filename = f"skillsprout_report_{user_id}_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
