"""Celery background tasks."""
import logging
from typing import List
from datetime import datetime
import asyncio

from sqlalchemy import select, and_, func
from sqlalchemy.orm import joinedload

from app.tasks.celery_app import celery_app
from app.db.session import SyncSessionLocal, get_sync_db
from app.models.models import (
    Occupation, Skill, OccupationSkill, UserFeedback,
    RecommendationEvent, RecommendedOccupation, ModelRegistry,
    ActionType
)
from app.services.onet_client import get_onet_client, ONetClientError
from app.ml.calibration import CalibrationModel, prepare_training_data_from_feedback
from app.core.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.tasks.warm_occupation_cache")
def warm_occupation_cache(onet_codes: List[str] = None):
    """Warm cache by fetching occupation skills.

    Args:
        onet_codes: List of O*NET codes to cache. If None, uses demo codes.
    """
    logger.info(f"Starting cache warming task")

    if onet_codes is None:
        # Default set for demo/development
        onet_codes = [
            "15-1252.00",  # Software Developers
            "15-1299.08",  # Web Developers
            "15-1244.00",  # Network and Computer Systems Administrators
            "15-1299.09",  # Web and Digital Interface Designers
            "15-1212.00",  # Information Security Analysts
            "11-3021.00",  # Computer and Information Systems Managers
            "15-1241.00",  # Computer Network Architects
            "15-1299.02",  # Geographic Information Systems Technologists
        ]

    db = SyncSessionLocal()
    client = get_onet_client()
    cached_count = 0
    failed_count = 0

    try:
        for onet_code in onet_codes:
            try:
                logger.info(f"Caching {onet_code}")

                # Check if already cached
                occupation = db.query(Occupation).filter(
                    Occupation.onet_code == onet_code
                ).first()

                # Fetch or update occupation metadata
                try:
                    occ_data = asyncio.run(client.get_occupation_meta(onet_code))
                except Exception as e:
                    logger.error(f"Failed to fetch metadata for {onet_code}: {e}")
                    failed_count += 1
                    continue

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
                    db.flush()

                # Fetch skills
                try:
                    skills_data = asyncio.run(client.get_occupation_skills(onet_code))
                except Exception as e:
                    logger.error(f"Failed to fetch skills for {onet_code}: {e}")
                    failed_count += 1
                    continue

                # Cache skills
                for skill_data in skills_data:
                    # Ensure skill exists
                    skill = db.query(Skill).filter(
                        Skill.element_id == skill_data["element_id"]
                    ).first()

                    if not skill:
                        skill = Skill(
                            element_id=skill_data["element_id"],
                            name=skill_data["skill_name"],
                        )
                        db.add(skill)
                        db.flush()

                    # Create or update occupation-skill link
                    occ_skill = db.query(OccupationSkill).filter(
                        and_(
                            OccupationSkill.onet_code == onet_code,
                            OccupationSkill.element_id == skill_data["element_id"],
                        )
                    ).first()

                    if not occ_skill:
                        occ_skill = OccupationSkill(
                            onet_code=onet_code,
                            element_id=skill_data["element_id"],
                            importance=skill_data.get("importance"),
                            level=skill_data.get("level"),
                            last_fetched_at=datetime.utcnow(),
                        )
                        db.add(occ_skill)
                    else:
                        occ_skill.importance = skill_data.get("importance")
                        occ_skill.level = skill_data.get("level")
                        occ_skill.last_fetched_at = datetime.utcnow()

                db.commit()
                cached_count += 1
                logger.info(f"Successfully cached {onet_code}")

            except Exception as e:
                logger.error(f"Error caching {onet_code}: {e}")
                db.rollback()
                failed_count += 1
                continue

        logger.info(f"Cache warming complete. Cached: {cached_count}, Failed: {failed_count}")
        return {"cached": cached_count, "failed": failed_count}

    finally:
        db.close()


@celery_app.task(name="app.tasks.tasks.train_calibration_model_task")
def train_calibration_model_task():
    """Train calibration model from user feedback data."""
    logger.info("Starting calibration model training task")

    db = SyncSessionLocal()

    try:
        # Extract training data from feedback
        # Join with recommendation events and recommended occupations to get features
        query = (
            db.query(
                UserFeedback.id.label("feedback_id"),
                UserFeedback.action_type,
                UserFeedback.event_id,
                UserFeedback.target_onet_code,
                RecommendationEvent.user_id,
                RecommendedOccupation.score_json,
            )
            .join(RecommendationEvent, UserFeedback.event_id == RecommendationEvent.id)
            .outerjoin(
                RecommendedOccupation,
                and_(
                    RecommendedOccupation.event_id == UserFeedback.event_id,
                    RecommendedOccupation.target_onet_code == UserFeedback.target_onet_code,
                ),
            )
            .filter(
                UserFeedback.action_type.in_([
                    ActionType.HIDE,
                    ActionType.APPLY,
                    ActionType.INTERVIEW,
                    ActionType.OFFER,
                ])
            )
        )

        feedback_records = query.all()

        if len(feedback_records) < settings.model_training_min_samples:
            logger.warning(
                f"Insufficient training data: {len(feedback_records)} samples, "
                f"need at least {settings.model_training_min_samples}. Skipping training."
            )
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "samples": len(feedback_records),
            }

        logger.info(f"Preparing training data from {len(feedback_records)} feedback records")

        # Prepare training data
        training_data = []
        for record in feedback_records:
            if record.score_json is None:
                continue

            # Determine label
            if record.action_type in [ActionType.INTERVIEW, ActionType.OFFER]:
                label = 1
            elif record.action_type == ActionType.APPLY:
                label = 1
            elif record.action_type == ActionType.HIDE:
                label = 0
            else:
                continue

            # Extract features from score_json
            score_json = record.score_json
            metadata = score_json.get("metadata", {})

            feature_dict = {
                "match_score": score_json.get("match_score", 0.0),
                "gap_severity": score_json.get("gap_severity", 0.0),
                "job_zone_diff": metadata.get("job_zone_diff", 0.0),
                "target_job_zone": float(metadata.get("target_job_zone", 0)),
                "num_missing_skills": metadata.get("skills_with_gaps", 0),
                "sum_missing_weights": sum(
                    gap.get("gap_weight", 0) for gap in score_json.get("top_gaps", [])
                ),
                "mean_rating": 2.0,  # Placeholder - would need to fetch from user data
                "rating_variance": 1.0,  # Placeholder
                "num_rated_skills": 10,  # Placeholder
                "user_id": record.user_id,
                "target_onet_code": record.target_onet_code,
                "event_id": record.event_id,
            }

            from app.ml.calibration import CalibrationFeatures
            features = CalibrationFeatures(**feature_dict)
            training_data.append((features, label))

        if len(training_data) < settings.model_training_min_samples:
            logger.warning(f"After filtering, only {len(training_data)} valid samples. Skipping.")
            return {
                "status": "skipped",
                "reason": "insufficient_valid_data",
                "samples": len(training_data),
            }

        # Train model
        logger.info(f"Training model on {len(training_data)} samples")
        model = CalibrationModel(model_version=f"v2_calibrated_{datetime.utcnow().strftime('%Y%m%d')}")

        try:
            metrics = model.train(training_data)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}

        # Save model
        artifact_path = model.save()

        # Register in database
        model_record = ModelRegistry(
            model_version=model.model_version,
            trained_at=model.trained_at,
            training_samples=len(training_data),
            metrics_json=metrics,
            artifact_path=artifact_path,
            is_active=False,  # Set manually or via API
            notes="Trained via periodic task",
        )
        db.add(model_record)
        db.commit()

        logger.info(f"Model training complete. Version: {model.model_version}, Metrics: {metrics}")

        return {
            "status": "success",
            "model_version": model.model_version,
            "training_samples": len(training_data),
            "metrics": metrics,
            "artifact_path": artifact_path,
        }

    except Exception as e:
        logger.error(f"Error in training task: {e}", exc_info=True)
        db.rollback()
        return {"status": "error", "error": str(e)}

    finally:
        db.close()


@celery_app.task(name="app.tasks.tasks.search_and_cache_occupations")
def search_and_cache_occupations(search_queries: List[str]):
    """Search for occupations and cache them.

    Args:
        search_queries: List of search queries (e.g., ["software", "engineer", "data"])
    """
    logger.info(f"Searching and caching occupations for queries: {search_queries}")

    client = get_onet_client()
    all_codes = set()

    for query in search_queries:
        try:
            results = asyncio.run(client.search_occupations(query, limit=10))
            for result in results:
                all_codes.add(result["code"])
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")

    logger.info(f"Found {len(all_codes)} unique occupations to cache")

    if all_codes:
        return warm_occupation_cache(list(all_codes))
    else:
        return {"cached": 0, "failed": 0}
