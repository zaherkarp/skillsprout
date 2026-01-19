"""Calibration model for learning from user feedback.

This module implements Model v2: a learned calibration layer that predicts
P(success) for (user, occupation) pairs and enables online learning.
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

from app.core.config import settings

logger = logging.getLogger(__name__)

# Model artifacts directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class CalibrationFeatures:
    """Feature set for calibration model."""
    # Baseline scores
    match_score: float
    gap_severity: float

    # Job zone features
    job_zone_diff: float  # target - current
    target_job_zone: float

    # Skill gap features
    num_missing_skills: int
    sum_missing_weights: float

    # User confidence features
    mean_rating: float
    rating_variance: float
    num_rated_skills: int

    # Metadata
    user_id: int
    target_onet_code: str
    event_id: int


@dataclass
class CalibrationPrediction:
    """Prediction from calibration model."""
    predicted_probability: float
    calibrated_score: float  # 0-100 scale
    features: CalibrationFeatures
    model_version: str


class CalibrationModel:
    """Learned calibration model for ranking occupations.

    This model learns from user feedback to predict P(success) and
    improve recommendation ranking over the baseline.
    """

    def __init__(self, model_version: str = "v2_calibrated"):
        """Initialize calibration model.

        Args:
            model_version: Model version identifier
        """
        self.model_version = model_version
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.trained_at: Optional[datetime] = None
        self.metrics: Dict[str, float] = {}

    def extract_features(
        self,
        user_id: int,
        target_onet_code: str,
        event_id: int,
        match_score: float,
        gap_severity: float,
        num_missing_skills: int,
        sum_missing_weights: float,
        current_job_zone: Optional[int],
        target_job_zone: Optional[int],
        user_ratings: Dict[str, int],
    ) -> CalibrationFeatures:
        """Extract features for a (user, occupation) pair.

        Args:
            user_id: User ID
            target_onet_code: Target occupation code
            event_id: Recommendation event ID
            match_score: Baseline match score
            gap_severity: Baseline gap severity
            num_missing_skills: Count of missing skills
            sum_missing_weights: Sum of importance weights for missing skills
            current_job_zone: User's current job zone
            target_job_zone: Target occupation job zone
            user_ratings: User's skill ratings

        Returns:
            CalibrationFeatures instance
        """
        # Job zone difference
        job_zone_diff = 0.0
        if current_job_zone is not None and target_job_zone is not None:
            job_zone_diff = float(target_job_zone - current_job_zone)

        # User confidence metrics
        ratings = list(user_ratings.values())
        mean_rating = np.mean(ratings) if ratings else 0.0
        rating_variance = np.var(ratings) if len(ratings) > 1 else 0.0
        num_rated_skills = len(ratings)

        return CalibrationFeatures(
            match_score=match_score,
            gap_severity=gap_severity,
            job_zone_diff=job_zone_diff,
            target_job_zone=float(target_job_zone or 0),
            num_missing_skills=num_missing_skills,
            sum_missing_weights=sum_missing_weights,
            mean_rating=mean_rating,
            rating_variance=rating_variance,
            num_rated_skills=num_rated_skills,
            user_id=user_id,
            target_onet_code=target_onet_code,
            event_id=event_id,
        )

    def _features_to_array(self, features: CalibrationFeatures) -> np.ndarray:
        """Convert features to numpy array for model input.

        Args:
            features: CalibrationFeatures instance

        Returns:
            Numpy array of feature values
        """
        return np.array([
            features.match_score,
            features.gap_severity,
            features.job_zone_diff,
            features.target_job_zone,
            features.num_missing_skills,
            features.sum_missing_weights,
            features.mean_rating,
            features.rating_variance,
            features.num_rated_skills,
        ]).reshape(1, -1)

    def train(
        self,
        training_data: List[Tuple[CalibrationFeatures, int]],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """Train calibration model on feedback data.

        Args:
            training_data: List of (features, label) tuples
                Label: 1 for positive outcome (interview/offer/apply), 0 for negative (hide)
            test_size: Fraction for test set
            random_state: Random seed

        Returns:
            Dict of evaluation metrics
        """
        if len(training_data) < settings.model_training_min_samples:
            raise ValueError(
                f"Insufficient training data: {len(training_data)} samples, "
                f"need at least {settings.model_training_min_samples}"
            )

        logger.info(f"Training calibration model on {len(training_data)} samples")

        # Convert to arrays
        X = np.array([
            self._features_to_array(features).flatten()
            for features, _ in training_data
        ])
        y = np.array([label for _, label in training_data])

        # Feature names
        self.feature_names = [
            "match_score",
            "gap_severity",
            "job_zone_diff",
            "target_job_zone",
            "num_missing_skills",
            "sum_missing_weights",
            "mean_rating",
            "rating_variance",
            "num_rated_skills",
        ]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train logistic regression
        self.model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced",  # Handle class imbalance
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_rate": float(np.mean(y)),
        }

        self.trained_at = datetime.utcnow()

        logger.info(f"Model trained. Metrics: {self.metrics}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")

        # Log feature importance
        coef = self.model.coef_[0]
        for name, c in zip(self.feature_names, coef):
            logger.info(f"  {name}: {c:.4f}")

        return self.metrics

    def predict(self, features: CalibrationFeatures) -> CalibrationPrediction:
        """Predict success probability for a (user, occupation) pair.

        Args:
            features: CalibrationFeatures instance

        Returns:
            CalibrationPrediction with probability and calibrated score
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert features to array and scale
        X = self._features_to_array(features)
        X_scaled = self.scaler.transform(X)

        # Predict probability
        prob = self.model.predict_proba(X_scaled)[0, 1]

        # Convert to 0-100 calibrated score
        calibrated_score = prob * 100

        return CalibrationPrediction(
            predicted_probability=float(prob),
            calibrated_score=float(calibrated_score),
            features=features,
            model_version=self.model_version,
        )

    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk.

        Args:
            path: Optional custom path. If None, uses default naming.

        Returns:
            Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        if path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = str(MODELS_DIR / f"{self.model_version}_{timestamp}.joblib")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "CalibrationModel":
        """Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded CalibrationModel instance
        """
        model_data = joblib.load(path)

        instance = cls(model_version=model_data["model_version"])
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.trained_at = model_data["trained_at"]
        instance.metrics = model_data["metrics"]

        logger.info(f"Model loaded from {path}")
        logger.info(f"Trained at: {instance.trained_at}, Metrics: {instance.metrics}")
        return instance


class ExplorationPolicy:
    """Epsilon-greedy exploration policy for online learning."""

    def __init__(self, epsilon: float = None):
        """Initialize exploration policy.

        Args:
            epsilon: Probability of exploration (0-1)
        """
        self.epsilon = epsilon or settings.exploration_epsilon

    def should_explore(self) -> bool:
        """Decide whether to explore or exploit.

        Returns:
            True if should explore, False otherwise
        """
        return np.random.random() < self.epsilon

    def select_exploration_items(
        self,
        ranked_recommendations: List[Dict[str, Any]],
        n: int = 2,
    ) -> List[int]:
        """Select items near decision boundary for exploration.

        Args:
            ranked_recommendations: List of recommendations sorted by score
            n: Number of exploration items to select

        Returns:
            Indices of items to explore
        """
        if not ranked_recommendations:
            return []

        # Find items near decision boundaries (e.g., around bucket transitions)
        # For simplicity, select items from the middle of the list
        mid_point = len(ranked_recommendations) // 2
        start_idx = max(0, mid_point - n // 2)
        end_idx = min(len(ranked_recommendations), start_idx + n)

        return list(range(start_idx, end_idx))


def prepare_training_data_from_feedback(
    feedback_records: List[Dict[str, Any]]
) -> List[Tuple[CalibrationFeatures, int]]:
    """Prepare training data from user feedback records.

    Args:
        feedback_records: List of feedback dicts with features and action_type

    Returns:
        List of (features, label) tuples
    """
    training_data = []

    for record in feedback_records:
        action_type = record.get("action_type", "")

        # Determine label
        # Positive: interview, offer, apply (with weights)
        # Negative: hide
        # Ignore: click, save (ambiguous)
        if action_type in ["interview", "offer"]:
            label = 1  # Strong positive
        elif action_type == "apply":
            label = 1  # Weak positive
        elif action_type == "hide":
            label = 0  # Negative
        else:
            continue  # Skip ambiguous actions

        # Extract features from record
        features = CalibrationFeatures(
            match_score=record["match_score"],
            gap_severity=record["gap_severity"],
            job_zone_diff=record.get("job_zone_diff", 0.0),
            target_job_zone=record.get("target_job_zone", 0.0),
            num_missing_skills=record["num_missing_skills"],
            sum_missing_weights=record["sum_missing_weights"],
            mean_rating=record["mean_rating"],
            rating_variance=record["rating_variance"],
            num_rated_skills=record["num_rated_skills"],
            user_id=record["user_id"],
            target_onet_code=record["target_onet_code"],
            event_id=record["event_id"],
        )

        training_data.append((features, label))

    logger.info(f"Prepared {len(training_data)} training samples from {len(feedback_records)} feedback records")
    return training_data
