"""Generate synthetic interaction data with realistic patterns.

This module creates fake-but-plausible user interaction data for offline
evaluation of SkillSprout recommendation models.  It is designed so that a
baseline scorer *should* produce better-than-random metrics on the generated
data, giving us a sanity-check that the evaluation pipeline is wired correctly.

Realistic Patterns Modeled
--------------------------
1. **Skill profiles**: Users have correlated skill ratings (e.g., a user good
   at "Programming" is likely also good at "Mathematics").
2. **Bucket-dependent action rates**: READY_NOW recommendations get more saves
   and applies than LONG_RESKILL ones.
3. **Temporal patterns**: Events are spread over ~180 days with realistic
   weekday/weekend distribution.
4. **Return visits**: Users who save a recommendation may or may not return
   within 7 days. High-match users are more likely to return.
5. **Noise**: Some users take surprising actions (e.g., applying to a
   LONG_RESKILL job) to test model robustness.

Usage
-----
    from ml.evaluation.generate_synthetic_interactions import (
        SyntheticDataGenerator,
        GeneratorConfig,
    )

    gen = SyntheticDataGenerator(GeneratorConfig(n_users=500, seed=42))
    interactions = gen.generate()
    interactions.to_parquet("data/synthetic_interactions.parquet")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.ml.scoring import BaselineScorer, OccupationScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill catalog (simplified O*NET-like skills)
# ---------------------------------------------------------------------------

# A small skill catalog with element IDs and names.  In production, these come
# from the O*NET database; here we use a representative subset to exercise all
# code paths.
SKILL_CATALOG: List[Dict[str, str]] = [
    {"element_id": "2.A.1.a", "skill_name": "Reading Comprehension"},
    {"element_id": "2.A.1.b", "skill_name": "Active Listening"},
    {"element_id": "2.A.1.c", "skill_name": "Writing"},
    {"element_id": "2.A.1.d", "skill_name": "Speaking"},
    {"element_id": "2.A.1.e", "skill_name": "Mathematics"},
    {"element_id": "2.A.1.f", "skill_name": "Science"},
    {"element_id": "2.B.1.a", "skill_name": "Critical Thinking"},
    {"element_id": "2.B.1.b", "skill_name": "Active Learning"},
    {"element_id": "2.B.2.i", "skill_name": "Complex Problem Solving"},
    {"element_id": "2.B.3.a", "skill_name": "Operations Analysis"},
    {"element_id": "2.B.3.b", "skill_name": "Technology Design"},
    {"element_id": "2.B.3.d", "skill_name": "Programming"},
    {"element_id": "2.B.4.e", "skill_name": "Quality Control Analysis"},
    {"element_id": "2.B.4.g", "skill_name": "Judgment and Decision Making"},
    {"element_id": "2.B.5.a", "skill_name": "Social Perceptiveness"},
]

# Occupation archetypes -- each defines which skills are important and at
# what level, plus a job zone.  These represent different career clusters.
OCCUPATION_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "onet_code": "15-1252.00",
        "title": "Software Developers",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.B.3.d", "importance": 90.0, "level": 6.0, "skill_name": "Programming"},
            {"element_id": "2.B.2.i", "importance": 85.0, "level": 5.5, "skill_name": "Complex Problem Solving"},
            {"element_id": "2.B.1.a", "importance": 80.0, "level": 5.0, "skill_name": "Critical Thinking"},
            {"element_id": "2.A.1.e", "importance": 70.0, "level": 4.5, "skill_name": "Mathematics"},
            {"element_id": "2.B.3.b", "importance": 75.0, "level": 5.0, "skill_name": "Technology Design"},
            {"element_id": "2.B.1.b", "importance": 65.0, "level": 4.0, "skill_name": "Active Learning"},
            {"element_id": "2.A.1.a", "importance": 60.0, "level": 4.0, "skill_name": "Reading Comprehension"},
        ],
    },
    {
        "onet_code": "13-2011.00",
        "title": "Accountants and Auditors",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.A.1.e", "importance": 90.0, "level": 6.0, "skill_name": "Mathematics"},
            {"element_id": "2.B.1.a", "importance": 80.0, "level": 5.0, "skill_name": "Critical Thinking"},
            {"element_id": "2.A.1.a", "importance": 75.0, "level": 5.0, "skill_name": "Reading Comprehension"},
            {"element_id": "2.A.1.c", "importance": 70.0, "level": 4.5, "skill_name": "Writing"},
            {"element_id": "2.B.4.g", "importance": 75.0, "level": 5.0, "skill_name": "Judgment and Decision Making"},
            {"element_id": "2.B.4.e", "importance": 65.0, "level": 4.0, "skill_name": "Quality Control Analysis"},
            {"element_id": "2.A.1.b", "importance": 60.0, "level": 4.0, "skill_name": "Active Listening"},
        ],
    },
    {
        "onet_code": "29-1141.00",
        "title": "Registered Nurses",
        "job_zone": 3,
        "skills": [
            {"element_id": "2.B.1.a", "importance": 85.0, "level": 5.0, "skill_name": "Critical Thinking"},
            {"element_id": "2.A.1.b", "importance": 85.0, "level": 5.0, "skill_name": "Active Listening"},
            {"element_id": "2.B.5.a", "importance": 80.0, "level": 5.0, "skill_name": "Social Perceptiveness"},
            {"element_id": "2.A.1.f", "importance": 75.0, "level": 4.5, "skill_name": "Science"},
            {"element_id": "2.B.4.g", "importance": 75.0, "level": 4.5, "skill_name": "Judgment and Decision Making"},
            {"element_id": "2.A.1.d", "importance": 70.0, "level": 4.0, "skill_name": "Speaking"},
            {"element_id": "2.B.1.b", "importance": 65.0, "level": 4.0, "skill_name": "Active Learning"},
        ],
    },
    {
        "onet_code": "11-1021.00",
        "title": "General and Operations Managers",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.B.4.g", "importance": 90.0, "level": 6.0, "skill_name": "Judgment and Decision Making"},
            {"element_id": "2.B.1.a", "importance": 85.0, "level": 5.5, "skill_name": "Critical Thinking"},
            {"element_id": "2.A.1.d", "importance": 80.0, "level": 5.0, "skill_name": "Speaking"},
            {"element_id": "2.A.1.b", "importance": 80.0, "level": 5.0, "skill_name": "Active Listening"},
            {"element_id": "2.B.5.a", "importance": 75.0, "level": 4.5, "skill_name": "Social Perceptiveness"},
            {"element_id": "2.B.2.i", "importance": 70.0, "level": 4.5, "skill_name": "Complex Problem Solving"},
            {"element_id": "2.A.1.a", "importance": 65.0, "level": 4.0, "skill_name": "Reading Comprehension"},
        ],
    },
    {
        "onet_code": "25-2021.00",
        "title": "Elementary School Teachers",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.A.1.d", "importance": 90.0, "level": 5.5, "skill_name": "Speaking"},
            {"element_id": "2.A.1.b", "importance": 85.0, "level": 5.0, "skill_name": "Active Listening"},
            {"element_id": "2.B.1.b", "importance": 80.0, "level": 5.0, "skill_name": "Active Learning"},
            {"element_id": "2.B.5.a", "importance": 80.0, "level": 5.0, "skill_name": "Social Perceptiveness"},
            {"element_id": "2.A.1.c", "importance": 75.0, "level": 4.5, "skill_name": "Writing"},
            {"element_id": "2.B.4.g", "importance": 70.0, "level": 4.0, "skill_name": "Judgment and Decision Making"},
            {"element_id": "2.A.1.a", "importance": 65.0, "level": 4.0, "skill_name": "Reading Comprehension"},
        ],
    },
    {
        "onet_code": "49-9071.00",
        "title": "Maintenance and Repair Workers",
        "job_zone": 2,
        "skills": [
            {"element_id": "2.B.3.a", "importance": 80.0, "level": 4.5, "skill_name": "Operations Analysis"},
            {"element_id": "2.B.2.i", "importance": 75.0, "level": 4.0, "skill_name": "Complex Problem Solving"},
            {"element_id": "2.B.4.e", "importance": 70.0, "level": 4.0, "skill_name": "Quality Control Analysis"},
            {"element_id": "2.B.1.a", "importance": 65.0, "level": 3.5, "skill_name": "Critical Thinking"},
            {"element_id": "2.A.1.a", "importance": 55.0, "level": 3.0, "skill_name": "Reading Comprehension"},
            {"element_id": "2.A.1.e", "importance": 50.0, "level": 3.0, "skill_name": "Mathematics"},
        ],
    },
]

# Skill clusters -- users within a cluster tend to have similar strengths.
# Each cluster defines a base rating distribution (mean for each skill).
USER_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "name": "tech_worker",
        "weight": 0.25,
        "base_ratings": {
            "2.B.3.d": 3.5, "2.B.2.i": 3.0, "2.B.1.a": 2.5, "2.A.1.e": 2.5,
            "2.B.3.b": 3.0, "2.B.1.b": 2.5, "2.A.1.a": 2.0, "2.A.1.b": 1.5,
            "2.A.1.c": 1.5, "2.A.1.d": 1.5, "2.A.1.f": 1.5, "2.B.3.a": 2.0,
            "2.B.4.e": 1.5, "2.B.4.g": 1.5, "2.B.5.a": 1.0,
        },
    },
    {
        "name": "healthcare_worker",
        "weight": 0.20,
        "base_ratings": {
            "2.B.3.d": 0.5, "2.B.2.i": 2.0, "2.B.1.a": 3.0, "2.A.1.e": 1.5,
            "2.B.3.b": 0.5, "2.B.1.b": 2.5, "2.A.1.a": 2.0, "2.A.1.b": 3.0,
            "2.A.1.c": 2.0, "2.A.1.d": 2.5, "2.A.1.f": 3.0, "2.B.3.a": 1.5,
            "2.B.4.e": 2.0, "2.B.4.g": 3.0, "2.B.5.a": 3.5,
        },
    },
    {
        "name": "business_worker",
        "weight": 0.25,
        "base_ratings": {
            "2.B.3.d": 0.5, "2.B.2.i": 2.0, "2.B.1.a": 2.5, "2.A.1.e": 2.5,
            "2.B.3.b": 1.0, "2.B.1.b": 2.0, "2.A.1.a": 2.5, "2.A.1.b": 2.5,
            "2.A.1.c": 2.5, "2.A.1.d": 3.0, "2.A.1.f": 0.5, "2.B.3.a": 2.0,
            "2.B.4.e": 2.0, "2.B.4.g": 3.0, "2.B.5.a": 2.5,
        },
    },
    {
        "name": "education_worker",
        "weight": 0.15,
        "base_ratings": {
            "2.B.3.d": 0.5, "2.B.2.i": 1.5, "2.B.1.a": 2.0, "2.A.1.e": 1.5,
            "2.B.3.b": 0.5, "2.B.1.b": 3.0, "2.A.1.a": 3.0, "2.A.1.b": 3.0,
            "2.A.1.c": 3.0, "2.A.1.d": 3.5, "2.A.1.f": 1.0, "2.B.3.a": 1.0,
            "2.B.4.e": 1.0, "2.B.4.g": 2.5, "2.B.5.a": 3.0,
        },
    },
    {
        "name": "trades_worker",
        "weight": 0.15,
        "base_ratings": {
            "2.B.3.d": 0.5, "2.B.2.i": 2.5, "2.B.1.a": 2.0, "2.A.1.e": 2.0,
            "2.B.3.b": 1.5, "2.B.1.b": 1.5, "2.A.1.a": 1.5, "2.A.1.b": 1.5,
            "2.A.1.c": 1.0, "2.A.1.d": 1.5, "2.A.1.f": 1.0, "2.B.3.a": 3.0,
            "2.B.4.e": 3.0, "2.B.4.g": 2.0, "2.B.5.a": 1.0,
        },
    },
]


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation.

    Attributes:
        n_users: Number of synthetic users to create.
        recommendations_per_user: How many recommendation events per user.
        occupations_per_event: Occupations shown per recommendation event.
        time_span_days: Duration in days over which events are spread.
        seed: Random seed for reproducibility.
        noise_std: Standard deviation of Gaussian noise added to base skill
            ratings to create user variation.

        Action probabilities by bucket -- these control how likely a user is
        to take each action type for recommendations in each bucket.
    """

    n_users: int = 200
    recommendations_per_user: int = 3
    occupations_per_event: int = 5
    time_span_days: int = 180
    seed: int = 42
    noise_std: float = 0.8

    # Action probability matrices: {bucket: {action_type: probability}}
    # These are designed so that higher-match buckets produce more positive
    # actions, creating a learnable signal for the model.
    action_probs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "ready_now": {
            "apply": 0.25,      # High-match users apply often
            "save": 0.20,
            "click": 0.20,
            "hide": 0.02,
            "no_action": 0.33,
        },
        "trainable": {
            "apply": 0.08,
            "save": 0.18,
            "click": 0.25,
            "hide": 0.05,
            "no_action": 0.44,
        },
        "long_reskill": {
            "apply": 0.02,      # Low-match users rarely apply
            "save": 0.08,
            "click": 0.15,
            "hide": 0.12,
            "no_action": 0.63,
        },
    })


class SyntheticDataGenerator:
    """Generates synthetic interaction data for offline evaluation.

    The generator creates users with realistic skill profiles, simulates
    recommendation events using the actual BaselineScorer, and generates
    user feedback actions with bucket-dependent probabilities.

    This produces data that has a learnable relationship between model scores
    and user outcomes, which is essential for validating that the evaluation
    pipeline can detect model quality differences.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        self.config = config or GeneratorConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.scorer = BaselineScorer()

    def generate(self) -> pd.DataFrame:
        """Generate the full synthetic interaction dataset.

        Returns:
            DataFrame with columns:
                - event_id: int
                - user_id: int
                - target_onet_code: str
                - occupation_title: str
                - event_timestamp: datetime64
                - action_type: str or None
                - action_timestamp: datetime64 or NaT
                - next_visit_timestamp: datetime64 or NaT
                - bucket: str
                - match_score: float
                - gap_severity: float
                - rank: int
                - model_version: str
                - user_archetype: str
                - num_missing_skills: int
                - sum_missing_weights: float
                - mean_rating: float
                - rating_variance: float
                - num_rated_skills: int
                - job_zone_diff: float
                - target_job_zone: int
        """
        users = self._generate_users()
        interactions = self._generate_interactions(users)

        logger.info(
            "Generated %d synthetic interactions for %d users across %d events",
            len(interactions),
            self.config.n_users,
            interactions["event_id"].nunique(),
        )

        return interactions

    def _generate_users(self) -> List[Dict[str, Any]]:
        """Generate synthetic user profiles with skill ratings.

        Each user is assigned to an archetype, then their ratings are sampled
        around the archetype's base ratings with Gaussian noise, clipped to
        [0, 4] and rounded to integers.

        Returns:
            List of user dicts with 'user_id', 'archetype', 'ratings',
            'current_job_zone'.
        """
        archetype_names = [a["name"] for a in USER_ARCHETYPES]
        archetype_weights = [a["weight"] for a in USER_ARCHETYPES]
        archetype_map = {a["name"]: a for a in USER_ARCHETYPES}

        users: List[Dict[str, Any]] = []

        for user_id in range(1, self.config.n_users + 1):
            # Pick archetype
            archetype_name = self.rng.choice(
                archetype_names, p=archetype_weights
            )
            archetype = archetype_map[archetype_name]

            # Generate skill ratings with noise
            ratings: Dict[str, int] = {}
            for skill in SKILL_CATALOG:
                eid = skill["element_id"]
                base = archetype["base_ratings"].get(eid, 1.0)
                noisy = base + self.rng.normal(0, self.config.noise_std)
                ratings[eid] = int(np.clip(round(noisy), 0, 4))

            # Assign a plausible current job zone based on archetype
            job_zone_map = {
                "tech_worker": 4,
                "healthcare_worker": 3,
                "business_worker": 4,
                "education_worker": 4,
                "trades_worker": 2,
            }
            current_job_zone = job_zone_map.get(archetype_name, 3)

            users.append({
                "user_id": user_id,
                "archetype": archetype_name,
                "ratings": ratings,
                "current_job_zone": current_job_zone,
            })

        return users

    def _generate_interactions(
        self, users: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Generate recommendation events and user feedback.

        For each user, we generate multiple recommendation events.  Each event
        scores a random subset of occupations using the real BaselineScorer,
        then simulates a user action based on the bucket of each result.

        Args:
            users: List of user dicts from ``_generate_users()``.

        Returns:
            DataFrame of interactions.
        """
        base_date = pd.Timestamp("2025-06-01")
        rows: List[Dict[str, Any]] = []
        event_counter = 0

        for user in users:
            user_id = user["user_id"]
            ratings = user["ratings"]
            current_jz = user["current_job_zone"]

            for _ in range(self.config.recommendations_per_user):
                event_counter += 1

                # Random event timestamp within the time span
                days_offset = self.rng.randint(0, self.config.time_span_days)
                hours_offset = self.rng.randint(8, 22)  # business-ish hours
                event_ts = base_date + pd.Timedelta(
                    days=int(days_offset), hours=int(hours_offset)
                )

                # Select a random subset of occupations to recommend
                n_occs = min(
                    self.config.occupations_per_event,
                    len(OCCUPATION_ARCHETYPES),
                )
                occ_indices = self.rng.choice(
                    len(OCCUPATION_ARCHETYPES), size=n_occs, replace=False
                )

                scored_occs: List[Tuple[OccupationScore, Dict[str, Any]]] = []
                for idx in occ_indices:
                    occ = OCCUPATION_ARCHETYPES[idx]
                    score = self.scorer.score_occupation(
                        onet_code=occ["onet_code"],
                        occupation_title=occ["title"],
                        occupation_skills=occ["skills"],
                        user_skill_ratings=ratings,
                        current_job_zone=current_jz,
                        target_job_zone=occ["job_zone"],
                    )
                    scored_occs.append((score, occ))

                # Sort by match_score descending for ranking
                scored_occs.sort(key=lambda x: x[0].match_score, reverse=True)

                for rank, (score, occ) in enumerate(scored_occs, start=1):
                    # Determine user action based on bucket
                    action_type, action_ts, next_visit_ts = (
                        self._simulate_action(
                            score.bucket,
                            score.match_score,
                            event_ts,
                        )
                    )

                    # Compute calibration features for downstream use
                    rating_values = list(ratings.values())
                    mean_rating = float(np.mean(rating_values))
                    rating_variance = float(np.var(rating_values))
                    num_rated = len(rating_values)
                    num_missing = score.metadata.get("skills_with_gaps", 0)
                    sum_missing_weights = sum(
                        g.gap_weight for g in score.top_gaps
                    )
                    jz_diff = float(
                        occ["job_zone"] - current_jz
                    )

                    rows.append({
                        "event_id": event_counter,
                        "user_id": user_id,
                        "target_onet_code": occ["onet_code"],
                        "occupation_title": occ["title"],
                        "event_timestamp": event_ts,
                        "action_type": action_type,
                        "action_timestamp": action_ts,
                        "next_visit_timestamp": next_visit_ts,
                        "bucket": score.bucket,
                        "match_score": score.match_score,
                        "gap_severity": score.gap_severity,
                        "rank": rank,
                        "model_version": "v1_baseline",
                        "user_archetype": user["archetype"],
                        "num_missing_skills": num_missing,
                        "sum_missing_weights": round(sum_missing_weights, 4),
                        "mean_rating": round(mean_rating, 4),
                        "rating_variance": round(rating_variance, 4),
                        "num_rated_skills": num_rated,
                        "job_zone_diff": jz_diff,
                        "target_job_zone": occ["job_zone"],
                    })

        df = pd.DataFrame(rows)

        # Ensure correct dtypes
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        df["action_timestamp"] = pd.to_datetime(df["action_timestamp"])
        df["next_visit_timestamp"] = pd.to_datetime(df["next_visit_timestamp"])

        return df.sort_values("event_timestamp").reset_index(drop=True)

    def _simulate_action(
        self,
        bucket: str,
        match_score: float,
        event_ts: pd.Timestamp,
    ) -> Tuple[Optional[str], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Simulate a user action on a recommendation.

        The action depends on the bucket and has some noise.  Higher match
        scores within a bucket slightly increase the probability of positive
        actions.

        Args:
            bucket: Recommendation bucket (ready_now, trainable, long_reskill).
            match_score: The baseline match score (0-100).
            event_ts: When the recommendation was shown.

        Returns:
            Tuple of (action_type, action_timestamp, next_visit_timestamp).
            action_type is None for no-action.
        """
        probs_dict = self.config.action_probs.get(bucket, {})
        if not probs_dict:
            return None, None, None

        # Slight boost for higher match scores within the bucket
        # This creates a continuous signal for the model to learn.
        match_boost = (match_score / 100.0) * 0.05
        actions = list(probs_dict.keys())
        probs = np.array([probs_dict[a] for a in actions], dtype=float)

        # Boost apply probability slightly for high match scores
        apply_idx = actions.index("apply") if "apply" in actions else -1
        if apply_idx >= 0:
            probs[apply_idx] += match_boost

        # Renormalize
        probs = probs / probs.sum()

        action = self.rng.choice(actions, p=probs)

        if action == "no_action":
            return None, None, None

        # Action happens some hours/days after the event
        if action in ("click",):
            delay_hours = self.rng.exponential(2)  # Quick action
        elif action in ("save",):
            delay_hours = self.rng.exponential(12)
        elif action in ("apply",):
            delay_hours = self.rng.exponential(48)
        elif action in ("hide",):
            delay_hours = self.rng.exponential(1)
        else:
            delay_hours = self.rng.exponential(24)

        action_ts = event_ts + pd.Timedelta(hours=float(delay_hours))

        # Simulate next visit (for save-and-return logic)
        next_visit_ts = None
        if action == "save":
            # Higher match score -> more likely to return quickly
            return_prob = 0.3 + (match_score / 100.0) * 0.4
            if self.rng.random() < return_prob:
                # Return within the window
                return_delay_days = self.rng.uniform(0.5, 6.0)
                next_visit_ts = action_ts + pd.Timedelta(
                    days=float(return_delay_days)
                )
            else:
                # Return late or never
                if self.rng.random() < 0.5:
                    # Late return (outside window)
                    return_delay_days = self.rng.uniform(8.0, 30.0)
                    next_visit_ts = action_ts + pd.Timedelta(
                        days=float(return_delay_days)
                    )
                # else: never returned, next_visit_ts stays None

        return action, action_ts, next_visit_ts


def generate_and_save(
    output_path: str = "data/synthetic_interactions.parquet",
    config: Optional[GeneratorConfig] = None,
) -> pd.DataFrame:
    """Convenience function to generate and optionally save synthetic data.

    Args:
        output_path: Path to save parquet file. Set to empty string to skip
            saving.
        config: Generator configuration. Uses defaults if not provided.

    Returns:
        Generated DataFrame.
    """
    gen = SyntheticDataGenerator(config)
    df = gen.generate()

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved %d interactions to %s", len(df), output_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = generate_and_save()
    print(f"\nGenerated {len(df)} interactions")
    print(f"Columns: {list(df.columns)}")
    print(f"\nBucket distribution:")
    print(df["bucket"].value_counts())
    print(f"\nAction distribution:")
    print(df["action_type"].value_counts(dropna=False))
