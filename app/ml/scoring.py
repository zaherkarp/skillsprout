"""Baseline scoring engine for job transition recommendations.

This module implements the deterministic baseline model (v1) that computes:
- match_score: How well user capabilities align with occupation requirements
- gap_severity: How severe are the skill gaps
- Bucket assignment: READY_NOW, TRAINABLE, or LONG_RESKILL
"""
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math

from app.core.config import settings


# Mapping from user rating (0-4) to capability scalar
RATING_TO_CAPABILITY = {
    0: 0.0,   # None
    1: 0.25,  # Basic
    2: 0.5,   # Intermediate
    3: 0.75,  # Advanced
    4: 1.0,   # Expert
}


@dataclass
class SkillGap:
    """Represents a skill gap for an occupation."""
    element_id: str
    skill_name: str
    required_importance: float
    required_level: Optional[float]
    user_capability: float
    gap_weight: float  # Weighted importance of this gap


@dataclass
class OccupationScore:
    """Scoring result for a single occupation."""
    onet_code: str
    match_score: float  # 0-100
    gap_severity: float  # 0-100
    top_gaps: List[SkillGap]
    bucket: str  # READY_NOW, TRAINABLE, LONG_RESKILL
    training_suggestion: str
    explanation: str
    metadata: Dict[str, Any]  # Additional scoring details


class BaselineScorer:
    """Baseline scoring model for job transition recommendations.

    This is Model v1: deterministic, rule-based scoring that establishes
    a strong baseline for comparison with future learned models.
    """

    def __init__(
        self,
        ready_now_match_threshold: float = None,
        ready_now_gap_threshold: float = None,
        trainable_match_min: float = None,
        trainable_match_max: float = None,
        trainable_gap_min: float = None,
        trainable_gap_max: float = None,
    ):
        """Initialize scorer with thresholds.

        Args:
            ready_now_match_threshold: Min match score for READY_NOW
            ready_now_gap_threshold: Max gap severity for READY_NOW
            trainable_match_min: Min match score for TRAINABLE
            trainable_match_max: Max match score for TRAINABLE
            trainable_gap_min: Min gap severity for TRAINABLE
            trainable_gap_max: Max gap severity for TRAINABLE
        """
        self.ready_now_match_threshold = (
            ready_now_match_threshold or settings.ready_now_match_threshold
        )
        self.ready_now_gap_threshold = (
            ready_now_gap_threshold or settings.ready_now_gap_threshold
        )
        self.trainable_match_min = trainable_match_min or settings.trainable_match_min
        self.trainable_match_max = trainable_match_max or settings.trainable_match_max
        self.trainable_gap_min = trainable_gap_min or settings.trainable_gap_min
        self.trainable_gap_max = trainable_gap_max or settings.trainable_gap_max

    def score_occupation(
        self,
        onet_code: str,
        occupation_title: str,
        occupation_skills: List[Dict[str, Any]],
        user_skill_ratings: Dict[str, int],
        current_job_zone: Optional[int] = None,
        target_job_zone: Optional[int] = None,
    ) -> OccupationScore:
        """Score a single occupation against user's capabilities.

        Args:
            onet_code: O*NET code for the occupation
            occupation_title: Occupation title
            occupation_skills: List of skills with importance/level
                Expected format: [{"element_id": "2.B.1.a", "importance": 75.0, "level": 5.0, "skill_name": "..."}]
            user_skill_ratings: Dict mapping element_id to rating (0-4)
            current_job_zone: User's current job zone
            target_job_zone: Target occupation's job zone

        Returns:
            OccupationScore with match_score, gap_severity, bucket, etc.
        """
        # Calculate match score and gap severity
        match_score, gap_severity, gaps = self._calculate_scores(
            occupation_skills, user_skill_ratings
        )

        # Assign bucket
        bucket = self._assign_bucket(match_score, gap_severity)

        # Generate training suggestion
        training_suggestion = self._generate_training_suggestion(
            bucket, target_job_zone, len(gaps)
        )

        # Generate explanation
        explanation = self._generate_explanation(
            occupation_title, match_score, gap_severity, bucket, len(gaps)
        )

        # Additional metadata
        metadata = {
            "total_skills": len(occupation_skills),
            "skills_with_gaps": len(gaps),
            "current_job_zone": current_job_zone,
            "target_job_zone": target_job_zone,
            "job_zone_diff": (
                target_job_zone - current_job_zone
                if current_job_zone and target_job_zone
                else None
            ),
        }

        return OccupationScore(
            onet_code=onet_code,
            match_score=round(match_score, 2),
            gap_severity=round(gap_severity, 2),
            top_gaps=gaps[:10],  # Top 10 gaps
            bucket=bucket,
            training_suggestion=training_suggestion,
            explanation=explanation,
            metadata=metadata,
        )

    def _calculate_scores(
        self,
        occupation_skills: List[Dict[str, Any]],
        user_skill_ratings: Dict[str, int],
    ) -> Tuple[float, float, List[SkillGap]]:
        """Calculate match_score and gap_severity.

        Args:
            occupation_skills: Skills required by occupation
            user_skill_ratings: User's self-assessed ratings

        Returns:
            Tuple of (match_score, gap_severity, skill_gaps)
        """
        if not occupation_skills:
            return 0.0, 100.0, []

        # Normalize importance values to use as weights
        total_importance = sum(
            skill.get("importance", 0) or 0 for skill in occupation_skills
        )

        if total_importance == 0:
            # If no importance data, use equal weights
            total_importance = len(occupation_skills)
            for skill in occupation_skills:
                skill["importance"] = 1.0

        # Calculate weighted match and identify gaps
        weighted_match = 0.0
        weighted_gap_mass = 0.0
        gaps = []

        for skill in occupation_skills:
            element_id = skill.get("element_id", "")
            skill_name = skill.get("skill_name", "Unknown Skill")
            importance = skill.get("importance", 0) or 0
            level = skill.get("level")

            # Get user capability for this skill
            user_rating = user_skill_ratings.get(element_id, 0)
            user_capability = RATING_TO_CAPABILITY.get(user_rating, 0.0)

            # Weight for this skill (normalized importance)
            weight = importance / total_importance

            # Contribution to match score
            weighted_match += weight * user_capability

            # Check if this is a gap (low capability)
            if user_capability <= 0.25:  # Ratings 0 or 1
                weighted_gap_mass += weight
                gaps.append(
                    SkillGap(
                        element_id=element_id,
                        skill_name=skill_name,
                        required_importance=importance,
                        required_level=level,
                        user_capability=user_capability,
                        gap_weight=weight,
                    )
                )

        # Convert to 0-100 scale
        match_score = weighted_match * 100
        gap_severity = weighted_gap_mass * 100

        # Sort gaps by weight (importance)
        gaps.sort(key=lambda g: g.gap_weight, reverse=True)

        return match_score, gap_severity, gaps

    def _assign_bucket(self, match_score: float, gap_severity: float) -> str:
        """Assign occupation to a recommendation bucket.

        Bucket logic:
        - READY_NOW: match_score >= 75 AND gap_severity <= 25
        - TRAINABLE: match_score 50-74 OR gap_severity 26-55
        - LONG_RESKILL: else

        Args:
            match_score: Match score (0-100)
            gap_severity: Gap severity (0-100)

        Returns:
            Bucket name: "ready_now", "trainable", or "long_reskill"
        """
        # READY_NOW: high match, low gaps
        if (
            match_score >= self.ready_now_match_threshold
            and gap_severity <= self.ready_now_gap_threshold
        ):
            return "ready_now"

        # TRAINABLE: moderate match or moderate gaps
        if (
            self.trainable_match_min <= match_score <= self.trainable_match_max
            or self.trainable_gap_min <= gap_severity <= self.trainable_gap_max
        ):
            return "trainable"

        # LONG_RESKILL: everything else
        return "long_reskill"

    def _generate_training_suggestion(
        self,
        bucket: str,
        job_zone: Optional[int],
        num_gaps: int,
    ) -> str:
        """Generate training path suggestion based on bucket and job zone.

        Heuristics:
        - job_zone 1-2: Certificate/apprenticeship
        - job_zone 3: Bootcamp/certificate + portfolio
        - job_zone 4-5: Longer reskill (possibly degree)

        Args:
            bucket: Recommendation bucket
            job_zone: Target job zone
            num_gaps: Number of skill gaps

        Returns:
            Training suggestion string
        """
        if bucket == "ready_now":
            return "You can apply now! Consider refreshing skills with online courses or practice projects."

        if bucket == "trainable":
            if job_zone is None:
                return "Consider focused training in the identified skill gaps."

            if job_zone <= 2:
                return f"Fill {num_gaps} skill gap(s) through a certificate program, on-the-job training, or apprenticeship (typically 3-12 months)."
            elif job_zone == 3:
                return f"Fill {num_gaps} skill gap(s) through a bootcamp, certificate program, or self-directed learning with portfolio projects (typically 3-18 months)."
            else:
                return f"Fill {num_gaps} skill gap(s) through an associate degree, extended bootcamp, or comprehensive self-study program (typically 1-2 years)."

        # LONG_RESKILL
        if job_zone is None:
            return "This role requires significant reskilling. Consider longer-term training programs."

        if job_zone <= 2:
            return f"Significant reskilling needed ({num_gaps} gaps). Consider vocational training or extended apprenticeship programs (1-2 years)."
        elif job_zone == 3:
            return f"Significant reskilling needed ({num_gaps} gaps). Consider an associate degree, extended bootcamp, or comprehensive self-study (1-3 years)."
        else:
            return f"Significant reskilling needed ({num_gaps} gaps). Consider a bachelor's degree, extended training program, or multi-year self-directed learning path (2-4 years)."

    def _generate_explanation(
        self,
        occupation_title: str,
        match_score: float,
        gap_severity: float,
        bucket: str,
        num_gaps: int,
    ) -> str:
        """Generate plain language explanation of the score.

        Args:
            occupation_title: Title of the occupation
            match_score: Match score
            gap_severity: Gap severity
            bucket: Recommendation bucket
            num_gaps: Number of gaps

        Returns:
            Explanation string
        """
        if bucket == "ready_now":
            return (
                f"Strong match for {occupation_title}! Your skills align well "
                f"({match_score:.0f}% match) with minimal gaps ({gap_severity:.0f}% severity). "
                "You're ready to apply."
            )
        elif bucket == "trainable":
            return (
                f"Good foundation for {occupation_title} ({match_score:.0f}% match), "
                f"but {num_gaps} key skill gap(s) to address ({gap_severity:.0f}% severity). "
                "With focused training, this role is achievable."
            )
        else:
            return (
                f"Lower match for {occupation_title} ({match_score:.0f}% match) with "
                f"{num_gaps} skill gaps ({gap_severity:.0f}% severity). "
                "This transition requires significant reskilling effort."
            )


def get_baseline_scorer() -> BaselineScorer:
    """Factory function to get baseline scorer with default settings."""
    return BaselineScorer()
