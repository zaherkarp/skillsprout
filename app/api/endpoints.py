"""FastAPI endpoint handlers."""
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import joinedload

from app.db.session import get_db
from app.models.models import (
    Occupation, Skill, OccupationSkill, UserProfile,
    UserCurrentOccupation, UserSkillRating, RecommendationEvent,
    RecommendedOccupation, UserFeedback, ModelRegistry, ActionType
)
from app.schemas.schemas import (
    OccupationSearchResult, OccupationDetail, OccupationWithSkills,
    UserProfileCreate, UserProfileResponse, UserCurrentOccupationRequest,
    UserCurrentOccupationResponse, UserSkillRatingsRequest, UserSkillRatingsResponse,
    RecommendationRequest, RecommendationResponse, RecommendationBucket,
    RecommendedOccupation as RecommendedOccupationSchema, SkillGapInfo,
    UserFeedbackRequest, UserFeedbackResponse, ModelStatusResponse, ModelMetrics,
    HealthCheckResponse, OccupationSearchRequest,
)
from app.services.onet_client import get_onet_client, ONetClientError
from app.ml.scoring import get_baseline_scorer
from app.ml.calibration import CalibrationModel, ExplorationPolicy
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== Health Check ====================

@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        demo_mode=settings.is_demo_mode,
        database="connected",
        redis="connected",
    )


# ==================== Occupation Endpoints ====================

@router.get("/occupations/search", response_model=List[OccupationSearchResult])
async def search_occupations(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search for occupations by keyword."""
    try:
        client = get_onet_client()
        results = await client.search_occupations(q, limit=limit)

        return [
            OccupationSearchResult(code=r["code"], title=r["title"])
            for r in results
        ]
    except ONetClientError as e:
        logger.error(f"O*NET error: {e}")
        raise HTTPException(status_code=502, detail=f"O*NET service error: {str(e)}")
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/occupations/{onet_code}", response_model=OccupationDetail)
async def get_occupation(
    onet_code: str,
    db: AsyncSession = Depends(get_db),
):
    """Get occupation metadata."""
    try:
        # Check cache first
        result = await db.execute(
            select(Occupation).where(Occupation.onet_code == onet_code)
        )
        occupation = result.scalar_one_or_none()

        if occupation:
            return OccupationDetail(
                code=occupation.onet_code,
                title=occupation.title,
                description=occupation.description,
                job_zone=occupation.job_zone,
                education=occupation.education_level,
            )

        # Fetch from O*NET
        client = get_onet_client()
        data = await client.get_occupation_meta(onet_code)

        # Cache in database
        occupation = Occupation(
            onet_code=data["code"],
            title=data["title"],
            description=data.get("description"),
            job_zone=data.get("job_zone"),
            education_level=data.get("education"),
            last_fetched_at=datetime.utcnow(),
            raw_json=data.get("raw_data"),
        )
        db.add(occupation)
        await db.commit()

        return OccupationDetail(
            code=occupation.onet_code,
            title=occupation.title,
            description=occupation.description,
            job_zone=occupation.job_zone,
            education=occupation.education_level,
        )
    except ONetClientError as e:
        logger.error(f"O*NET error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching occupation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/occupations/{onet_code}/skills", response_model=OccupationWithSkills)
async def get_occupation_skills(
    onet_code: str,
    db: AsyncSession = Depends(get_db),
):
    """Get occupation with skills."""
    try:
        # Check cache
        result = await db.execute(
            select(Occupation)
            .options(joinedload(Occupation.occupation_skills).joinedload(OccupationSkill.skill))
            .where(Occupation.onet_code == onet_code)
        )
        occupation = result.unique().scalar_one_or_none()

        if occupation and occupation.occupation_skills:
            # Return from cache
            from app.schemas.schemas import OccupationSkill as OccupationSkillSchema

            return OccupationWithSkills(
                code=occupation.onet_code,
                title=occupation.title,
                description=occupation.description,
                job_zone=occupation.job_zone,
                skills=[
                    OccupationSkillSchema(
                        element_id=os.skill.element_id,
                        skill_name=os.skill.name,
                        importance=os.importance,
                        level=os.level,
                    )
                    for os in occupation.occupation_skills
                ],
            )

        # Fetch from O*NET
        client = get_onet_client()

        # Get occupation metadata
        occ_data = await client.get_occupation_meta(onet_code)
        skills_data = await client.get_occupation_skills(onet_code)

        # Ensure occupation exists
        if not occupation:
            occupation = Occupation(
                onet_code=occ_data["code"],
                title=occ_data["title"],
                description=occ_data.get("description"),
                job_zone=occ_data.get("job_zone"),
                education_level=occ_data.get("education"),
                last_fetched_at=datetime.utcnow(),
                raw_json=occ_data.get("raw_data"),
            )
            db.add(occupation)

        # Cache skills
        for skill_data in skills_data:
            # Ensure skill exists
            result = await db.execute(
                select(Skill).where(Skill.element_id == skill_data["element_id"])
            )
            skill = result.scalar_one_or_none()

            if not skill:
                skill = Skill(
                    element_id=skill_data["element_id"],
                    name=skill_data["skill_name"],
                )
                db.add(skill)
                await db.flush()

            # Create occupation-skill link
            result = await db.execute(
                select(OccupationSkill).where(
                    and_(
                        OccupationSkill.onet_code == onet_code,
                        OccupationSkill.element_id == skill_data["element_id"],
                    )
                )
            )
            occ_skill = result.scalar_one_or_none()

            if not occ_skill:
                occ_skill = OccupationSkill(
                    onet_code=onet_code,
                    element_id=skill_data["element_id"],
                    importance=skill_data.get("importance"),
                    level=skill_data.get("level"),
                    last_fetched_at=datetime.utcnow(),
                )
                db.add(occ_skill)

        await db.commit()

        # Fetch again with relationships
        result = await db.execute(
            select(Occupation)
            .options(joinedload(Occupation.occupation_skills).joinedload(OccupationSkill.skill))
            .where(Occupation.onet_code == onet_code)
        )
        occupation = result.unique().scalar_one()

        from app.schemas.schemas import OccupationSkill as OccupationSkillSchema

        return OccupationWithSkills(
            code=occupation.onet_code,
            title=occupation.title,
            description=occupation.description,
            job_zone=occupation.job_zone,
            skills=[
                OccupationSkillSchema(
                    element_id=os.skill.element_id,
                    skill_name=os.skill.name,
                    importance=os.importance,
                    level=os.level,
                )
                for os in occupation.occupation_skills
            ],
        )
    except ONetClientError as e:
        logger.error(f"O*NET error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching occupation skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== User Endpoints ====================

@router.post("/user/profile", response_model=UserProfileResponse, status_code=201)
async def create_user_profile(
    request: UserProfileCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new user profile."""
    try:
        user = UserProfile(
            created_at=datetime.utcnow(),
            metadata_json=request.metadata,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        return UserProfileResponse(
            id=user.id,
            created_at=user.created_at,
            metadata=user.metadata_json,
        )
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user/{user_id}/current-occupation", response_model=UserCurrentOccupationResponse)
async def set_current_occupation(
    user_id: int,
    request: UserCurrentOccupationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Set user's current occupation."""
    try:
        # Verify user exists
        result = await db.execute(select(UserProfile).where(UserProfile.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Verify occupation exists
        result = await db.execute(
            select(Occupation).where(Occupation.onet_code == request.onet_code)
        )
        occupation = result.scalar_one_or_none()
        if not occupation:
            # Try to fetch from O*NET
            try:
                client = get_onet_client()
                occ_data = await client.get_occupation_meta(request.onet_code)
                occupation = Occupation(
                    onet_code=occ_data["code"],
                    title=occ_data["title"],
                    description=occ_data.get("description"),
                    job_zone=occ_data.get("job_zone"),
                    education_level=occ_data.get("education"),
                    last_fetched_at=datetime.utcnow(),
                    raw_json=occ_data.get("raw_data"),
                )
                db.add(occupation)
                await db.flush()
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Occupation not found: {e}")

        # Deactivate previous selections
        await db.execute(
            select(UserCurrentOccupation)
            .where(UserCurrentOccupation.user_id == user_id)
        )
        # Update is_active to False for all existing
        from sqlalchemy import update
        await db.execute(
            update(UserCurrentOccupation)
            .where(UserCurrentOccupation.user_id == user_id)
            .values(is_active=False)
        )

        # Create new selection
        current_occ = UserCurrentOccupation(
            user_id=user_id,
            onet_code=request.onet_code,
            selected_at=datetime.utcnow(),
            is_active=True,
        )
        db.add(current_occ)
        await db.commit()
        await db.refresh(current_occ)

        return UserCurrentOccupationResponse(
            user_id=user_id,
            onet_code=request.onet_code,
            occupation_title=occupation.title,
            selected_at=current_occ.selected_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting current occupation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user/{user_id}/skills/ratings", response_model=UserSkillRatingsResponse)
async def update_skill_ratings(
    user_id: int,
    request: UserSkillRatingsRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update user skill ratings (bulk upsert)."""
    try:
        # Verify user exists
        result = await db.execute(select(UserProfile).where(UserProfile.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        updated_count = 0
        for rating in request.ratings:
            # Check if rating exists
            result = await db.execute(
                select(UserSkillRating).where(
                    and_(
                        UserSkillRating.user_id == user_id,
                        UserSkillRating.element_id == rating.element_id,
                    )
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.rating_0_4 = rating.rating_0_4
                existing.updated_at = datetime.utcnow()
            else:
                new_rating = UserSkillRating(
                    user_id=user_id,
                    element_id=rating.element_id,
                    rating_0_4=rating.rating_0_4,
                    updated_at=datetime.utcnow(),
                )
                db.add(new_rating)
            updated_count += 1

        await db.commit()

        return UserSkillRatingsResponse(
            user_id=user_id,
            updated_count=updated_count,
            ratings=request.ratings,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating skill ratings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Recommendation Endpoint ====================

@router.post("/user/{user_id}/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    request: RecommendationRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate job transition recommendations for user."""
    try:
        # Verify user exists
        result = await db.execute(select(UserProfile).where(UserProfile.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Get current occupation
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
            raise HTTPException(status_code=400, detail="User has no current occupation set")

        # Get user skill ratings
        result = await db.execute(
            select(UserSkillRating).where(UserSkillRating.user_id == user_id)
        )
        skill_ratings = result.scalars().all()
        user_ratings_dict = {sr.element_id: sr.rating_0_4 for sr in skill_ratings}

        if not user_ratings_dict:
            raise HTTPException(status_code=400, detail="User has no skill ratings")

        # Get all occupations to score (from cache)
        result = await db.execute(
            select(Occupation)
            .options(joinedload(Occupation.occupation_skills).joinedload(OccupationSkill.skill))
        )
        occupations = result.unique().scalars().all()

        if not occupations:
            raise HTTPException(status_code=503, detail="No occupations in cache. Run cache warming first.")

        # Score all occupations
        scorer = get_baseline_scorer()
        scored_occupations = []

        for occupation in occupations:
            if occupation.onet_code == current_occ.onet_code:
                continue  # Skip current occupation

            occupation_skills = [
                {
                    "element_id": os.skill.element_id,
                    "skill_name": os.skill.name,
                    "importance": os.importance,
                    "level": os.level,
                }
                for os in occupation.occupation_skills
            ]

            if not occupation_skills:
                continue

            score = scorer.score_occupation(
                onet_code=occupation.onet_code,
                occupation_title=occupation.title,
                occupation_skills=occupation_skills,
                user_skill_ratings=user_ratings_dict,
                current_job_zone=current_occ.occupation.job_zone,
                target_job_zone=occupation.job_zone,
            )

            scored_occupations.append((occupation, score))

        # Create recommendation event
        event = RecommendationEvent(
            user_id=user_id,
            created_at=datetime.utcnow(),
            current_onet_code=current_occ.onet_code,
            model_version=settings.model_version,
            params_json={
                "ready_now_match_threshold": settings.ready_now_match_threshold,
                "ready_now_gap_threshold": settings.ready_now_gap_threshold,
                "use_calibration": request.use_calibration,
                "enable_exploration": request.enable_exploration,
            },
        )
        db.add(event)
        await db.flush()

        # Group by bucket and sort
        buckets_dict = {
            "ready_now": [],
            "trainable": [],
            "long_reskill": [],
        }

        for occupation, score in scored_occupations:
            buckets_dict[score.bucket].append((occupation, score))

        # Sort each bucket by match score (or calibrated score if available)
        for bucket_name in buckets_dict:
            buckets_dict[bucket_name].sort(key=lambda x: x[1].match_score, reverse=True)

        # Save recommendations
        rank = 0
        for bucket_name, items in buckets_dict.items():
            for occupation, score in items[:request.limit_per_bucket]:
                rank += 1
                rec = RecommendedOccupation(
                    event_id=event.id,
                    target_onet_code=occupation.onet_code,
                    rank=rank,
                    bucket=bucket_name,
                    score_json={
                        "match_score": score.match_score,
                        "gap_severity": score.gap_severity,
                        "top_gaps": [
                            {
                                "element_id": g.element_id,
                                "skill_name": g.skill_name,
                                "required_importance": g.required_importance,
                                "user_capability": g.user_capability,
                            }
                            for g in score.top_gaps[:5]
                        ],
                        "training_suggestion": score.training_suggestion,
                        "explanation": score.explanation,
                        "metadata": score.metadata,
                    },
                    is_exploration=False,
                )
                db.add(rec)

        await db.commit()

        # Build response
        bucket_responses = []
        bucket_labels = {
            "ready_now": ("Ready Now", "Jobs you can apply to immediately"),
            "trainable": ("Trainable", "Jobs within reach with focused training"),
            "long_reskill": ("Long-term", "Jobs requiring significant reskilling"),
        }

        for bucket_name, (label, description) in bucket_labels.items():
            items = buckets_dict[bucket_name][:request.limit_per_bucket]

            if items:
                bucket_responses.append(
                    RecommendationBucket(
                        bucket_name=bucket_name,
                        bucket_label=label,
                        description=description,
                        occupations=[
                            RecommendedOccupationSchema(
                                onet_code=occ.onet_code,
                                title=occ.title,
                                rank=score.metadata.get("rank", idx + 1),
                                bucket=score.bucket,
                                match_score=score.match_score,
                                gap_severity=score.gap_severity,
                                top_gaps=[
                                    SkillGapInfo(
                                        element_id=g.element_id,
                                        skill_name=g.skill_name,
                                        required_importance=g.required_importance,
                                        required_level=g.required_level,
                                        user_capability=g.user_capability,
                                        gap_weight=g.gap_weight,
                                    )
                                    for g in score.top_gaps[:5]
                                ],
                                training_suggestion=score.training_suggestion,
                                explanation=score.explanation,
                                is_exploration=False,
                                metadata=score.metadata,
                            )
                            for idx, (occ, score) in enumerate(items)
                        ],
                    )
                )

        # Decision guidance
        if buckets_dict["ready_now"]:
            decision_guidance = "You have immediate opportunities! Start applying to 'Ready Now' jobs while exploring trainable options."
        elif buckets_dict["trainable"]:
            decision_guidance = "Focus on training programs for 'Trainable' jobs. Consider part-time learning while staying in current role."
        else:
            decision_guidance = "Plan for long-term reskilling. Consider formal education, bootcamps, or extended self-study programs."

        return RecommendationResponse(
            event_id=event.id,
            user_id=user_id,
            current_occupation=OccupationDetail(
                code=current_occ.occupation.onet_code,
                title=current_occ.occupation.title,
                description=current_occ.occupation.description,
                job_zone=current_occ.occupation.job_zone,
                education=current_occ.occupation.education_level,
            ),
            model_version=settings.model_version,
            created_at=event.created_at,
            buckets=bucket_responses,
            decision_guidance=decision_guidance,
            total_recommendations=rank,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Feedback Endpoint ====================

@router.post("/feedback", response_model=UserFeedbackResponse, status_code=201)
async def submit_feedback(
    request: UserFeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit user feedback on a recommendation."""
    try:
        # Verify event exists
        result = await db.execute(
            select(RecommendationEvent).where(RecommendationEvent.id == request.event_id)
        )
        event = result.scalar_one_or_none()
        if not event:
            raise HTTPException(status_code=404, detail="Recommendation event not found")

        # Find recommended occupation if exists
        result = await db.execute(
            select(RecommendedOccupation).where(
                and_(
                    RecommendedOccupation.event_id == request.event_id,
                    RecommendedOccupation.target_onet_code == request.target_onet_code,
                )
            )
        )
        recommended_occ = result.scalar_one_or_none()

        # Create feedback
        feedback = UserFeedback(
            event_id=request.event_id,
            target_onet_code=request.target_onet_code,
            recommended_occupation_id=recommended_occ.id if recommended_occ else None,
            action_type=ActionType(request.action_type),
            action_at=datetime.utcnow(),
            metadata_json=request.metadata,
        )
        db.add(feedback)
        await db.commit()
        await db.refresh(feedback)

        return UserFeedbackResponse(
            feedback_id=feedback.id,
            event_id=feedback.event_id,
            target_onet_code=feedback.target_onet_code,
            action_type=feedback.action_type.value,
            action_at=feedback.action_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Model Status Endpoint ====================

@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status(db: AsyncSession = Depends(get_db)):
    """Get current model status and metrics."""
    try:
        # Get active model from registry
        result = await db.execute(
            select(ModelRegistry)
            .where(ModelRegistry.is_active == True)
            .order_by(ModelRegistry.trained_at.desc())
        )
        active_model = result.scalar_one_or_none()

        # Count feedback and recommendations
        result = await db.execute(select(func.count(UserFeedback.id)))
        total_feedback = result.scalar()

        result = await db.execute(select(func.count(RecommendedOccupation.id)))
        total_recommendations = result.scalar()

        if active_model:
            metrics = ModelMetrics(
                accuracy=active_model.metrics_json.get("accuracy") if active_model.metrics_json else None,
                roc_auc=active_model.metrics_json.get("roc_auc") if active_model.metrics_json else None,
                train_samples=active_model.metrics_json.get("train_samples") if active_model.metrics_json else None,
                test_samples=active_model.metrics_json.get("test_samples") if active_model.metrics_json else None,
                positive_rate=active_model.metrics_json.get("positive_rate") if active_model.metrics_json else None,
            )

            return ModelStatusResponse(
                current_model_version=active_model.model_version,
                is_calibrated=True,
                last_trained_at=active_model.trained_at,
                training_samples=active_model.training_samples,
                metrics=metrics,
                total_feedback_events=total_feedback,
                total_recommendations=total_recommendations,
            )
        else:
            # Using baseline only
            return ModelStatusResponse(
                current_model_version=settings.model_version,
                is_calibrated=False,
                last_trained_at=None,
                training_samples=None,
                metrics=None,
                total_feedback_events=total_feedback,
                total_recommendations=total_recommendations,
            )

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
