"""FastAPI router for the explainability engine.

Endpoints:
----------
GET  /api/v1/recommendations/{occupation_id}/explain
    Returns a structured BucketExplanation for a single occupation score.
    Requires that the occupation has been previously scored (i.e., it exists
    in the recommendation event history for the requesting user).

PATCH /api/v1/user/preferences
    Updates the user's risk tolerance preference. This affects which
    ThresholdProfile is used for future explanations and comparisons.

GET  /api/v1/recommendations/compare?ids=101,205,312
    Returns a ComparisonResult for up to 3 occupations. The occupation codes
    are passed as a comma-separated query parameter.

Design notes:
  - All endpoints are async and use the same database session pattern as
    the main API router in app.api.endpoints.
  - We re-score occupations on the fly rather than caching explanations,
    because the explanation depends on the user's current risk tolerance
    which can change between requests.
  - The router is mounted at /api/v1 by the main app, so the full paths
    are as shown above.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.db.session import get_db
from app.models.models import (
    Occupation,
    OccupationSkill,
    RecommendationEvent,
    RecommendedOccupation,
    UserCurrentOccupation,
    UserProfile,
    UserSkillRating,
)
from app.ml.scoring import BaselineScorer, OccupationScore, get_baseline_scorer
from app.features.explainability.bucket_explainer import (
    BucketExplainerEngine,
    BucketExplanation,
    explain_score,
)
from app.features.explainability.comparison_view import (
    ComparisonEngine,
    ComparisonError,
    ComparisonResult,
)
from app.features.explainability.threshold_config import (
    RiskTolerance,
    get_threshold_profile,
    get_all_presets,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
# We define lightweight Pydantic models for the API responses rather than
# directly serialising dataclasses. This gives us control over the JSON
# schema and lets FastAPI generate accurate OpenAPI docs.

class SkillToDevelopResponse(BaseModel):
    """API response model for a single skill gap."""
    element_id: str
    skill_name: str
    required_importance: float
    user_capability: float
    gap_weight: float
    gap_category: str
    typical_training_time: str
    why_it_matters: str
    skill_domain: str


class BucketReasoningResponse(BaseModel):
    """API response model for bucket reasoning."""
    assigned_bucket: str
    match_score: float
    gap_severity: float
    thresholds_used: Dict[str, float]
    match_meets_ready_now: bool
    gap_meets_ready_now: bool
    match_in_trainable_range: bool
    gap_in_trainable_range: bool
    reasoning_text: str


class WhatWouldChangeResponse(BaseModel):
    """API response model for hypothetical bucket changes."""
    to_ready_now: Optional[str] = None
    to_trainable: Optional[str] = None
    match_score_needed: Optional[float] = None
    gap_reduction_needed: Optional[float] = None
    gaps_to_close: List[str]


class ExplanationResponse(BaseModel):
    """Full explanation response for a single occupation."""
    onet_code: str
    summary: str
    skills_you_have: List[str]
    skills_to_develop: List[SkillToDevelopResponse]
    bucket_reasoning: BucketReasoningResponse
    what_would_change_bucket: WhatWouldChangeResponse
    risk_tolerance_used: str
    metadata: Dict[str, Any]


class UserPreferencesRequest(BaseModel):
    """Request body for updating user preferences."""
    risk_tolerance: str = Field(
        ...,
        description=(
            "Risk tolerance level: 'relaxed', 'standard', or 'strict'. "
            "Controls how aggressively occupations are classified into "
            "higher-readiness buckets."
        ),
    )


class UserPreferencesResponse(BaseModel):
    """Response after updating user preferences."""
    user_id: int
    risk_tolerance: str
    profile_description: str
    thresholds: Dict[str, float]


class ComparisonOccupationEntry(BaseModel):
    """Single occupation entry within a comparison response."""
    onet_code: str
    summary: str
    bucket: str
    match_score: float
    gap_severity: float
    skills_you_have: List[str]
    skills_to_develop: List[SkillToDevelopResponse]


class SkillOverlapResponse(BaseModel):
    """Skill overlap analysis in comparison response."""
    shared_skills: List[str]
    shared_gaps: List[str]
    unique_gaps: Dict[str, List[str]]


class ReadinessRankingResponse(BaseModel):
    """Readiness ranking in comparison response."""
    ranked_codes: List[str]
    closest_onet_code: str
    rankings: List[Dict[str, Any]]


class ComparisonResponse(BaseModel):
    """Full comparison response for up to 3 occupations."""
    occupation_codes: List[str]
    occupations: List[ComparisonOccupationEntry]
    skill_overlap: SkillOverlapResponse
    readiness_ranking: ReadinessRankingResponse
    comparison_summary: str


# ---------------------------------------------------------------------------
# Helper: resolve user's risk tolerance from DB metadata
# ---------------------------------------------------------------------------

async def _get_user_risk_tolerance(
    user_id: int,
    db: AsyncSession,
) -> RiskTolerance:
    """Retrieve the user's saved risk tolerance preference.

    The preference is stored in UserProfile.metadata_json under the key
    'risk_tolerance'. If not set, defaults to STANDARD.

    Args:
        user_id: The user's ID.
        db: Async database session.

    Returns:
        The user's RiskTolerance, defaulting to STANDARD.
    """
    result = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user or not user.metadata_json:
        return RiskTolerance.STANDARD

    pref = user.metadata_json.get("risk_tolerance", "standard")
    try:
        return RiskTolerance(pref)
    except ValueError:
        return RiskTolerance.STANDARD


# ---------------------------------------------------------------------------
# Helper: score an occupation for a user
# ---------------------------------------------------------------------------

async def _score_occupation_for_user(
    onet_code: str,
    user_id: int,
    db: AsyncSession,
) -> tuple:
    """Fetch occupation data and score it against the user's skills.

    Returns a tuple of (OccupationScore, occupation_skills_list, user_ratings_dict)
    so the caller can pass all three to the explainer.

    Args:
        onet_code: O*NET occupation code.
        user_id: User's ID.
        db: Database session.

    Returns:
        Tuple of (OccupationScore, List[Dict], Dict[str, int]).

    Raises:
        HTTPException: If occupation or user data is missing.
    """
    # Fetch occupation with skills
    result = await db.execute(
        select(Occupation)
        .options(
            joinedload(Occupation.occupation_skills)
            .joinedload(OccupationSkill.skill)
        )
        .where(Occupation.onet_code == onet_code)
    )
    occupation = result.unique().scalar_one_or_none()
    if not occupation:
        raise HTTPException(status_code=404, detail=f"Occupation {onet_code} not found")

    if not occupation.occupation_skills:
        raise HTTPException(
            status_code=404,
            detail=f"No skill data available for occupation {onet_code}",
        )

    # Build skill list
    occupation_skills = [
        {
            "element_id": os.skill.element_id,
            "skill_name": os.skill.name,
            "importance": os.importance,
            "level": os.level,
        }
        for os in occupation.occupation_skills
    ]

    # Fetch user skill ratings
    result = await db.execute(
        select(UserSkillRating).where(UserSkillRating.user_id == user_id)
    )
    skill_ratings = result.scalars().all()
    user_ratings = {sr.element_id: sr.rating_0_4 for sr in skill_ratings}

    if not user_ratings:
        raise HTTPException(
            status_code=400,
            detail="User has no skill ratings. Rate your skills first.",
        )

    # Fetch user's current occupation for job zone context
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
    current_job_zone = (
        current_occ.occupation.job_zone if current_occ else None
    )

    # Score
    scorer = get_baseline_scorer()
    score = scorer.score_occupation(
        onet_code=onet_code,
        occupation_title=occupation.title,
        occupation_skills=occupation_skills,
        user_skill_ratings=user_ratings,
        current_job_zone=current_job_zone,
        target_job_zone=occupation.job_zone,
    )

    return score, occupation_skills, user_ratings


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _explanation_to_response(expl: BucketExplanation) -> ExplanationResponse:
    """Convert a BucketExplanation dataclass to a Pydantic response model.

    Args:
        expl: The BucketExplanation to serialise.

    Returns:
        ExplanationResponse suitable for JSON serialisation.
    """
    return ExplanationResponse(
        onet_code=expl.onet_code,
        summary=expl.summary,
        skills_you_have=expl.skills_you_have,
        skills_to_develop=[
            SkillToDevelopResponse(
                element_id=s.element_id,
                skill_name=s.skill_name,
                required_importance=s.required_importance,
                user_capability=s.user_capability,
                gap_weight=s.gap_weight,
                gap_category=s.gap_category,
                typical_training_time=s.typical_training_time,
                why_it_matters=s.why_it_matters,
                skill_domain=s.skill_domain,
            )
            for s in expl.skills_to_develop
        ],
        bucket_reasoning=BucketReasoningResponse(
            assigned_bucket=expl.bucket_reasoning.assigned_bucket,
            match_score=expl.bucket_reasoning.match_score,
            gap_severity=expl.bucket_reasoning.gap_severity,
            thresholds_used=expl.bucket_reasoning.thresholds_used,
            match_meets_ready_now=expl.bucket_reasoning.match_meets_ready_now,
            gap_meets_ready_now=expl.bucket_reasoning.gap_meets_ready_now,
            match_in_trainable_range=expl.bucket_reasoning.match_in_trainable_range,
            gap_in_trainable_range=expl.bucket_reasoning.gap_in_trainable_range,
            reasoning_text=expl.bucket_reasoning.reasoning_text,
        ),
        what_would_change_bucket=WhatWouldChangeResponse(
            to_ready_now=expl.what_would_change_bucket.to_ready_now,
            to_trainable=expl.what_would_change_bucket.to_trainable,
            match_score_needed=expl.what_would_change_bucket.match_score_needed,
            gap_reduction_needed=expl.what_would_change_bucket.gap_reduction_needed,
            gaps_to_close=expl.what_would_change_bucket.gaps_to_close,
        ),
        risk_tolerance_used=expl.risk_tolerance_used,
        metadata=expl.metadata,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/recommendations/{occupation_id}/explain",
    response_model=ExplanationResponse,
    summary="Explain a recommendation",
    description=(
        "Returns a structured explanation for why an occupation was scored "
        "and bucketed the way it was. Includes matched skills, gaps with "
        "training time estimates, threshold reasoning, and hypothetical "
        "bucket transitions."
    ),
)
async def explain_recommendation(
    occupation_id: str,
    user_id: int = Query(..., description="User ID for skill context"),
    db: AsyncSession = Depends(get_db),
) -> ExplanationResponse:
    """Generate a structured explanation for a single occupation recommendation.

    The occupation is re-scored against the user's current skill ratings and
    the explanation reflects the user's saved risk tolerance preference.

    Args:
        occupation_id: O*NET code of the occupation to explain.
        user_id: The requesting user's ID.
        db: Database session (injected).

    Returns:
        ExplanationResponse with full structured explanation.
    """
    try:
        # Get user's risk tolerance
        risk_tolerance = await _get_user_risk_tolerance(user_id, db)
        profile = get_threshold_profile(risk_tolerance)

        # Score the occupation
        score, occ_skills, user_ratings = await _score_occupation_for_user(
            occupation_id, user_id, db
        )

        # Generate explanation
        engine = BucketExplainerEngine()
        explanation = engine.explain(
            score,
            occupation_skills=occ_skills,
            user_skill_ratings=user_ratings,
            profile=profile,
        )

        return _explanation_to_response(explanation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/user/preferences",
    response_model=UserPreferencesResponse,
    summary="Update user preferences",
    description=(
        "Update the user's risk tolerance preference. This affects how "
        "occupations are classified into buckets in future explanations "
        "and comparisons."
    ),
)
async def update_user_preferences(
    request: UserPreferencesRequest,
    user_id: int = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Update the user's risk tolerance preference.

    The preference is stored in UserProfile.metadata_json and affects
    all subsequent explanation and comparison requests.

    Args:
        request: The preference update payload.
        user_id: The user's ID (query parameter).
        db: Database session (injected).

    Returns:
        UserPreferencesResponse confirming the update.
    """
    # Validate risk tolerance value
    try:
        tolerance = RiskTolerance(request.risk_tolerance)
    except ValueError:
        valid_values = [rt.value for rt in RiskTolerance]
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid risk_tolerance '{request.risk_tolerance}'. "
                f"Valid values: {valid_values}"
            ),
        )

    # Fetch user
    result = await db.execute(
        select(UserProfile).where(UserProfile.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update metadata
    metadata = user.metadata_json or {}
    metadata["risk_tolerance"] = tolerance.value
    user.metadata_json = metadata

    await db.commit()
    await db.refresh(user)

    # Return confirmation with profile details
    profile = get_threshold_profile(tolerance)
    thresholds = profile.bucket_thresholds

    return UserPreferencesResponse(
        user_id=user_id,
        risk_tolerance=tolerance.value,
        profile_description=profile.description,
        thresholds={
            "ready_now_match_min": thresholds.ready_now_match_min,
            "ready_now_gap_max": thresholds.ready_now_gap_max,
            "trainable_match_min": thresholds.trainable_match_min,
            "trainable_match_max": thresholds.trainable_match_max,
            "trainable_gap_min": thresholds.trainable_gap_min,
            "trainable_gap_max": thresholds.trainable_gap_max,
        },
    )


@router.get(
    "/recommendations/compare",
    response_model=ComparisonResponse,
    summary="Compare occupations side by side",
    description=(
        "Compare up to 3 occupations side by side. Shows skill overlap, "
        "unique gaps, and which occupation is closest to READY_NOW."
    ),
)
async def compare_recommendations(
    ids: str = Query(
        ...,
        description=(
            "Comma-separated O*NET occupation codes to compare (1-3). "
            "Example: ids=15-1252.00,29-1141.00,11-9013.00"
        ),
    ),
    user_id: int = Query(..., description="User ID for skill context"),
    db: AsyncSession = Depends(get_db),
) -> ComparisonResponse:
    """Compare up to 3 occupations side by side.

    Each occupation is scored against the user's current skills and the
    results are analysed for skill overlap, unique gaps, and readiness
    ranking.

    Args:
        ids: Comma-separated O*NET codes (1-3 codes).
        user_id: The requesting user's ID.
        db: Database session (injected).

    Returns:
        ComparisonResponse with full analysis.
    """
    # Parse occupation codes
    codes = [c.strip() for c in ids.split(",") if c.strip()]
    if not codes:
        raise HTTPException(
            status_code=400, detail="At least one occupation code is required."
        )
    if len(codes) > 3:
        raise HTTPException(
            status_code=400,
            detail="Maximum 3 occupations can be compared at once.",
        )

    try:
        # Get user's risk tolerance
        risk_tolerance = await _get_user_risk_tolerance(user_id, db)
        profile = get_threshold_profile(risk_tolerance)

        # Score each occupation
        scores: List[OccupationScore] = []
        occ_skills_map: Dict[str, List[Dict]] = {}
        user_ratings: Optional[Dict[str, int]] = None

        for code in codes:
            score, occ_skills, ratings = await _score_occupation_for_user(
                code, user_id, db
            )
            scores.append(score)
            occ_skills_map[code] = occ_skills
            if user_ratings is None:
                user_ratings = ratings

        # Run comparison
        engine = ComparisonEngine()
        comparison = engine.compare(
            scores,
            occupation_skills_map=occ_skills_map,
            user_skill_ratings=user_ratings,
            risk_tolerance=risk_tolerance,
        )

        # Build response
        occupations = []
        for code in comparison.occupation_codes:
            expl = comparison.explanations[code]
            # Find the score for this code
            score_obj = next(s for s in scores if s.onet_code == code)
            occupations.append(ComparisonOccupationEntry(
                onet_code=code,
                summary=expl.summary,
                bucket=expl.bucket_reasoning.assigned_bucket,
                match_score=expl.bucket_reasoning.match_score,
                gap_severity=expl.bucket_reasoning.gap_severity,
                skills_you_have=expl.skills_you_have,
                skills_to_develop=[
                    SkillToDevelopResponse(
                        element_id=s.element_id,
                        skill_name=s.skill_name,
                        required_importance=s.required_importance,
                        user_capability=s.user_capability,
                        gap_weight=s.gap_weight,
                        gap_category=s.gap_category,
                        typical_training_time=s.typical_training_time,
                        why_it_matters=s.why_it_matters,
                        skill_domain=s.skill_domain,
                    )
                    for s in expl.skills_to_develop
                ],
            ))

        return ComparisonResponse(
            occupation_codes=comparison.occupation_codes,
            occupations=occupations,
            skill_overlap=SkillOverlapResponse(
                shared_skills=comparison.skill_overlap.shared_skills,
                shared_gaps=comparison.skill_overlap.shared_gaps,
                unique_gaps=comparison.skill_overlap.unique_gaps,
            ),
            readiness_ranking=ReadinessRankingResponse(
                ranked_codes=comparison.readiness_ranking.ranked_codes,
                closest_onet_code=comparison.readiness_ranking.closest_onet_code,
                rankings=comparison.readiness_ranking.rankings,
            ),
            comparison_summary=comparison.comparison_summary,
        )

    except ComparisonError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
