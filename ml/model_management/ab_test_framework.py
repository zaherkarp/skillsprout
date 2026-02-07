"""Simple A/B testing framework for model comparison.

This module provides deterministic, hash-based user assignment to model
variants so that experiments are reproducible and do not require external
state.  Key features:

- Hash-based user assignment to model variants (deterministic per user).
- Percentage-based allocation, e.g. ``{"v2.3": 90, "v2.4_candidate": 10}``.
- Prediction logging that records which model version served each request.
- Analysis function that uses a chi-squared test to assess whether the
  variants differ in conversion rate.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from scipy import stats as scipy_stats

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ABVariant:
    """Describes a single model variant in an A/B test."""

    name: str
    model_version: str
    traffic_percentage: float  # 0-100

    def __post_init__(self) -> None:
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError(
                f"traffic_percentage must be in [0, 100], got {self.traffic_percentage}"
            )


@dataclass
class PredictionLog:
    """Record of a single prediction served during an experiment."""

    user_id: int
    variant_name: str
    model_version: str
    prediction: float
    outcome: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABAnalysisResult:
    """Result of a chi-squared analysis comparing two or more variants."""

    experiment_name: str
    variants: Dict[str, Dict[str, Any]]  # variant_name -> stats
    chi2_statistic: float
    chi2_p_value: float
    is_significant: bool
    significance_level: float
    winner: Optional[str] = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ABExperiment:
    """A complete A/B experiment definition.

    Attributes:
        name: Human-readable experiment name.
        variants: List of variants with traffic allocations.
        salt: Optional salt for the hash function (allows running
            multiple independent experiments on the same user base).
    """

    name: str
    variants: List[ABVariant]
    salt: str = ""

    def __post_init__(self) -> None:
        total = sum(v.traffic_percentage for v in self.variants)
        if abs(total - 100.0) > 0.01:
            raise ValueError(
                f"Variant percentages must sum to 100, got {total:.2f}"
            )


# ---------------------------------------------------------------------------
# A/B test framework
# ---------------------------------------------------------------------------

class ABTestFramework:
    """Manage model A/B experiments.

    Usage::

        framework = ABTestFramework()
        experiment = framework.create_experiment(
            name="v2.4-rollout",
            allocations={"v2.3": 90, "v2.4_candidate": 10},
        )
        variant = framework.assign_user(experiment, user_id=42)
        framework.log_prediction(experiment, user_id=42, prediction=0.75)
        result = framework.analyse(experiment)
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, ABExperiment] = {}
        self._logs: Dict[str, List[PredictionLog]] = {}

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        allocations: Dict[str, float],
        salt: str = "",
    ) -> ABExperiment:
        """Create and register a new A/B experiment.

        Args:
            name: Unique experiment name.
            allocations: Mapping of ``model_version`` to traffic percentage.
                Must sum to 100.  Example: ``{"v2.3": 90, "v2.4_candidate": 10}``.
            salt: Optional salt for independent hashing.

        Returns:
            The created ``ABExperiment``.

        Raises:
            ValueError: If the name is already taken or allocations are invalid.
        """
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")

        variants = [
            ABVariant(name=version, model_version=version, traffic_percentage=pct)
            for version, pct in allocations.items()
        ]

        experiment = ABExperiment(name=name, variants=variants, salt=salt)
        self._experiments[name] = experiment
        self._logs[name] = []

        logger.info(
            "Created experiment '%s' with variants: %s",
            name,
            {v.name: v.traffic_percentage for v in variants},
        )
        return experiment

    def get_experiment(self, name: str) -> Optional[ABExperiment]:
        """Retrieve an experiment by name.

        Args:
            name: Experiment name.

        Returns:
            The ``ABExperiment`` or ``None``.
        """
        return self._experiments.get(name)

    def list_experiments(self) -> List[ABExperiment]:
        """Return all registered experiments."""
        return list(self._experiments.values())

    # ------------------------------------------------------------------
    # User assignment
    # ------------------------------------------------------------------

    def assign_user(
        self,
        experiment: ABExperiment,
        user_id: int,
    ) -> ABVariant:
        """Deterministically assign a user to a variant.

        The assignment is based on a SHA-256 hash of the experiment name,
        salt, and user ID, mapped to the configured traffic percentages.
        Calling this function multiple times with the same inputs always
        returns the same variant.

        Args:
            experiment: The experiment to assign within.
            user_id: The user identifier.

        Returns:
            The ``ABVariant`` the user is assigned to.
        """
        bucket = self._hash_to_bucket(
            experiment.name, experiment.salt, user_id
        )
        return self._bucket_to_variant(bucket, experiment.variants)

    def get_model_version_for_user(
        self,
        experiment_name: str,
        user_id: int,
    ) -> str:
        """Convenience: return the model version string for a user.

        Args:
            experiment_name: Name of the experiment.
            user_id: User identifier.

        Returns:
            Model version string.

        Raises:
            ValueError: If the experiment does not exist.
        """
        experiment = self._experiments.get(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        variant = self.assign_user(experiment, user_id)
        return variant.model_version

    # ------------------------------------------------------------------
    # Prediction logging
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        experiment: ABExperiment,
        user_id: int,
        prediction: float,
        outcome: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PredictionLog:
        """Log a prediction (and optional outcome) for analysis.

        Args:
            experiment: The experiment context.
            user_id: User identifier.
            prediction: The predicted value (e.g. probability).
            outcome: Optional observed outcome (0 or 1).
            metadata: Optional extra data to store.

        Returns:
            The created ``PredictionLog``.
        """
        variant = self.assign_user(experiment, user_id)
        log_entry = PredictionLog(
            user_id=user_id,
            variant_name=variant.name,
            model_version=variant.model_version,
            prediction=prediction,
            outcome=outcome,
            metadata=metadata or {},
        )
        self._logs.setdefault(experiment.name, []).append(log_entry)
        return log_entry

    def record_outcome(
        self,
        experiment: ABExperiment,
        user_id: int,
        outcome: float,
    ) -> int:
        """Back-fill outcome for all logged predictions of a user.

        Args:
            experiment: The experiment context.
            user_id: User identifier.
            outcome: Observed outcome (0 or 1).

        Returns:
            Number of log entries updated.
        """
        updated = 0
        for entry in self._logs.get(experiment.name, []):
            if entry.user_id == user_id and entry.outcome is None:
                entry.outcome = outcome
                updated += 1
        return updated

    def get_logs(self, experiment_name: str) -> List[PredictionLog]:
        """Return all prediction logs for an experiment.

        Args:
            experiment_name: Experiment name.

        Returns:
            List of ``PredictionLog`` entries.
        """
        return list(self._logs.get(experiment_name, []))

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse(
        self,
        experiment: ABExperiment,
        significance_level: float = 0.05,
    ) -> ABAnalysisResult:
        """Run a chi-squared test on the experiment results.

        The test checks whether the conversion rate (outcome == 1) differs
        significantly across variants.

        Args:
            experiment: The experiment to analyse.
            significance_level: p-value threshold for significance.

        Returns:
            An ``ABAnalysisResult`` with test statistics and a winner
            (if statistically significant).

        Raises:
            ValueError: If there are fewer than 2 variants with data or
                no outcomes have been recorded.
        """
        logs = self._logs.get(experiment.name, [])
        logs_with_outcomes = [l for l in logs if l.outcome is not None]

        if not logs_with_outcomes:
            raise ValueError(
                f"No outcomes recorded for experiment '{experiment.name}'"
            )

        # Group by variant.
        variant_stats: Dict[str, Dict[str, Any]] = {}
        for variant in experiment.variants:
            variant_logs = [
                l for l in logs_with_outcomes if l.variant_name == variant.name
            ]
            total = len(variant_logs)
            conversions = sum(1 for l in variant_logs if l.outcome == 1.0)
            conversion_rate = conversions / total if total > 0 else 0.0

            variant_stats[variant.name] = {
                "total": total,
                "conversions": conversions,
                "conversion_rate": conversion_rate,
                "model_version": variant.model_version,
            }

        # Need at least 2 variants with observations.
        populated = {k: v for k, v in variant_stats.items() if v["total"] > 0}
        if len(populated) < 2:
            raise ValueError(
                "Need at least 2 variants with recorded outcomes for analysis"
            )

        # Build contingency table: rows = variants, cols = [converted, not-converted].
        variant_names = list(populated.keys())
        observed = []
        for name in variant_names:
            s = populated[name]
            observed.append([s["conversions"], s["total"] - s["conversions"]])

        chi2, p_value, dof, expected = scipy_stats.chi2_contingency(observed)

        is_significant = p_value < significance_level

        # Determine winner (highest conversion rate) if significant.
        winner: Optional[str] = None
        if is_significant:
            winner = max(
                populated,
                key=lambda k: populated[k]["conversion_rate"],
            )

        summary_parts = [
            f"Experiment '{experiment.name}': chi2={chi2:.4f}, p={p_value:.4f}.",
        ]
        if is_significant:
            summary_parts.append(
                f"Statistically significant at alpha={significance_level}. "
                f"Winner: {winner} "
                f"(conversion rate: {populated[winner]['conversion_rate']:.2%})."
            )
        else:
            summary_parts.append(
                f"Not statistically significant at alpha={significance_level}."
            )

        result = ABAnalysisResult(
            experiment_name=experiment.name,
            variants=variant_stats,
            chi2_statistic=float(chi2),
            chi2_p_value=float(p_value),
            is_significant=is_significant,
            significance_level=significance_level,
            winner=winner,
            summary=" ".join(summary_parts),
        )

        logger.info("A/B analysis: %s", result.summary)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_to_bucket(
        experiment_name: str,
        salt: str,
        user_id: int,
    ) -> float:
        """Hash experiment+salt+user_id to a deterministic value in [0, 100).

        Args:
            experiment_name: Experiment name.
            salt: Experiment salt.
            user_id: User identifier.

        Returns:
            A float in [0, 100).
        """
        raw = f"{experiment_name}:{salt}:{user_id}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        # Use the first 8 hex chars (32 bits) to derive a bucket.
        value = int(digest[:8], 16)
        return (value % 10000) / 100.0  # 0.00 to 99.99

    @staticmethod
    def _bucket_to_variant(
        bucket: float,
        variants: List[ABVariant],
    ) -> ABVariant:
        """Map a bucket value to the appropriate variant.

        Variants are laid out consecutively on the [0, 100) number line
        according to their ``traffic_percentage``.

        Args:
            bucket: A float in [0, 100).
            variants: Ordered list of variants.

        Returns:
            The variant whose range contains ``bucket``.
        """
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_percentage
            if bucket < cumulative:
                return variant
        # Fallback (should not happen with valid percentages).
        return variants[-1]
