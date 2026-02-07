"""Calibration monitoring for model quality assurance.

This module implements weekly calibration checks that detect prediction
drift and flag models whose predicted probabilities no longer align
with observed outcomes.  The core workflow:

1. Pull last 7 days of (prediction, outcome) pairs.
2. Compute reliability diagram (10 equal-width bins), ECE, per-bucket
   accuracy, and a two-sample K-S test for distribution shift.
3. Persist results in a ``calibration_snapshots`` conceptual table
   (stored via the model registry metadata).
4. Flag issues:  ECE > 0.15 WARNING, ECE > 0.25 ALERT, K-S > 0.1 drift.
5. Generate a weekly markdown report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from app.core.config import settings
from app.tasks.celery_app import celery_app
from app.db.session import SyncSessionLocal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
ECE_WARNING_THRESHOLD: float = 0.15
ECE_ALERT_THRESHOLD: float = 0.25
KS_DRIFT_THRESHOLD: float = 0.10
NUM_BINS: int = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBin:
    """A single bin in the reliability diagram."""

    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_observed: float
    count: int
    accuracy: float


@dataclass
class CalibrationSnapshot:
    """Complete calibration snapshot for a single evaluation window."""

    model_version: str
    evaluated_at: datetime
    window_start: datetime
    window_end: datetime
    total_samples: int
    ece: float
    bins: List[CalibrationBin]
    ks_statistic: float
    ks_p_value: float
    severity: str  # "OK", "WARNING", "ALERT"
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

class CalibrationMonitor:
    """Evaluate and track calibration quality over time.

    This monitor pulls recent ``(predicted_probability, binary_outcome)``
    pairs, bins them into a reliability diagram, and computes the Expected
    Calibration Error (ECE) plus a K-S test for distribution shift between
    the predicted probabilities of this window and a reference distribution.

    Attributes:
        num_bins: Number of equal-width bins for the reliability diagram.
        ece_warning: ECE threshold that triggers a WARNING.
        ece_alert: ECE threshold that triggers an ALERT.
        ks_drift: K-S statistic threshold for drift detection.
    """

    def __init__(
        self,
        num_bins: int = NUM_BINS,
        ece_warning: float = ECE_WARNING_THRESHOLD,
        ece_alert: float = ECE_ALERT_THRESHOLD,
        ks_drift: float = KS_DRIFT_THRESHOLD,
    ) -> None:
        self.num_bins = num_bins
        self.ece_warning = ece_warning
        self.ece_alert = ece_alert
        self.ks_drift = ks_drift
        self._snapshots: List[CalibrationSnapshot] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
        model_version: str,
        reference_predictions: Optional[np.ndarray] = None,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
    ) -> CalibrationSnapshot:
        """Run a full calibration evaluation.

        Args:
            predictions: Array of predicted probabilities in [0, 1].
            outcomes: Array of binary outcomes (0 or 1).
            model_version: Identifier for the model being evaluated.
            reference_predictions: Optional baseline prediction distribution
                for the K-S drift test.  When ``None`` a uniform [0, 1]
                reference is used.
            window_start: Start of the evaluation window.
            window_end: End of the evaluation window.

        Returns:
            A ``CalibrationSnapshot`` with all computed metrics.

        Raises:
            ValueError: If inputs are empty or have mismatched lengths.
        """
        predictions = np.asarray(predictions, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)

        if predictions.size == 0 or outcomes.size == 0:
            raise ValueError("predictions and outcomes must be non-empty")
        if predictions.shape != outcomes.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs "
                f"outcomes {outcomes.shape}"
            )

        now = datetime.utcnow()
        window_end = window_end or now
        window_start = window_start or (now - timedelta(days=7))

        bins = self._build_reliability_diagram(predictions, outcomes)
        ece = self._compute_ece(bins, total_samples=len(predictions))
        ks_stat, ks_p = self._ks_test(predictions, reference_predictions)
        flags = self._determine_flags(ece, ks_stat)
        severity = self._severity_from_flags(flags)

        snapshot = CalibrationSnapshot(
            model_version=model_version,
            evaluated_at=now,
            window_start=window_start,
            window_end=window_end,
            total_samples=int(len(predictions)),
            ece=float(ece),
            bins=bins,
            ks_statistic=float(ks_stat),
            ks_p_value=float(ks_p),
            severity=severity,
            flags=flags,
            metadata={
                "mean_predicted": float(np.mean(predictions)),
                "mean_observed": float(np.mean(outcomes)),
                "positive_rate": float(np.mean(outcomes)),
            },
        )

        self._snapshots.append(snapshot)
        logger.info(
            "Calibration evaluation complete for %s: ECE=%.4f, KS=%.4f, severity=%s",
            model_version,
            ece,
            ks_stat,
            severity,
        )
        return snapshot

    def get_snapshots(self) -> List[CalibrationSnapshot]:
        """Return all stored calibration snapshots."""
        return list(self._snapshots)

    def snapshot_to_dict(self, snapshot: CalibrationSnapshot) -> Dict[str, Any]:
        """Serialise a snapshot to a JSON-friendly dictionary.

        This is used when persisting results to the ``calibration_snapshots``
        conceptual table (stored as JSON in model registry metadata).

        Args:
            snapshot: The snapshot to serialise.

        Returns:
            A dictionary suitable for JSON serialisation.
        """
        return {
            "model_version": snapshot.model_version,
            "evaluated_at": snapshot.evaluated_at.isoformat(),
            "window_start": snapshot.window_start.isoformat(),
            "window_end": snapshot.window_end.isoformat(),
            "total_samples": snapshot.total_samples,
            "ece": snapshot.ece,
            "ks_statistic": snapshot.ks_statistic,
            "ks_p_value": snapshot.ks_p_value,
            "severity": snapshot.severity,
            "flags": snapshot.flags,
            "bins": [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "mean_predicted": b.mean_predicted,
                    "mean_observed": b.mean_observed,
                    "count": b.count,
                    "accuracy": b.accuracy,
                }
                for b in snapshot.bins
            ],
            "metadata": snapshot.metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_reliability_diagram(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
    ) -> List[CalibrationBin]:
        """Bin predictions into equal-width buckets and compute per-bin stats.

        Args:
            predictions: Predicted probabilities.
            outcomes: Binary outcomes.

        Returns:
            List of ``CalibrationBin`` objects.
        """
        bin_edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        bins: List[CalibrationBin] = []

        for i in range(self.num_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i < self.num_bins - 1:
                mask = (predictions >= lower) & (predictions < upper)
            else:
                # Last bin is inclusive on the upper bound.
                mask = (predictions >= lower) & (predictions <= upper)

            count = int(mask.sum())
            if count == 0:
                bins.append(
                    CalibrationBin(
                        bin_lower=float(lower),
                        bin_upper=float(upper),
                        mean_predicted=float((lower + upper) / 2),
                        mean_observed=0.0,
                        count=0,
                        accuracy=0.0,
                    )
                )
                continue

            mean_pred = float(np.mean(predictions[mask]))
            mean_obs = float(np.mean(outcomes[mask]))
            accuracy = float(np.mean(
                (predictions[mask] >= 0.5).astype(float) == outcomes[mask]
            ))

            bins.append(
                CalibrationBin(
                    bin_lower=float(lower),
                    bin_upper=float(upper),
                    mean_predicted=mean_pred,
                    mean_observed=mean_obs,
                    count=count,
                    accuracy=accuracy,
                )
            )

        return bins

    @staticmethod
    def _compute_ece(
        bins: List[CalibrationBin],
        total_samples: int,
    ) -> float:
        """Compute Expected Calibration Error.

        ECE = sum_b (|B_b| / N) * |mean_pred_b - mean_obs_b|

        Args:
            bins: Reliability diagram bins.
            total_samples: Total number of samples across all bins.

        Returns:
            ECE value in [0, 1].
        """
        if total_samples == 0:
            return 0.0

        ece = 0.0
        for b in bins:
            if b.count == 0:
                continue
            weight = b.count / total_samples
            ece += weight * abs(b.mean_predicted - b.mean_observed)
        return ece

    @staticmethod
    def _ks_test(
        predictions: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test for distribution shift.

        When no reference distribution is supplied the predictions are
        compared against a uniform [0, 1] distribution (one-sample K-S).

        Args:
            predictions: Current window predictions.
            reference: Optional reference distribution predictions.

        Returns:
            Tuple of (ks_statistic, p_value).
        """
        if reference is not None and len(reference) > 0:
            result = scipy_stats.ks_2samp(predictions, reference)
        else:
            result = scipy_stats.kstest(predictions, "uniform")
        return float(result.statistic), float(result.pvalue)

    def _determine_flags(self, ece: float, ks_stat: float) -> List[str]:
        """Produce human-readable flag strings for the snapshot.

        Args:
            ece: Expected Calibration Error.
            ks_stat: K-S test statistic.

        Returns:
            List of flag description strings (may be empty).
        """
        flags: List[str] = []
        if ece > self.ece_alert:
            flags.append(f"ALERT: ECE ({ece:.4f}) exceeds {self.ece_alert}")
        elif ece > self.ece_warning:
            flags.append(f"WARNING: ECE ({ece:.4f}) exceeds {self.ece_warning}")

        if ks_stat > self.ks_drift:
            flags.append(
                f"DRIFT: K-S statistic ({ks_stat:.4f}) exceeds {self.ks_drift}"
            )

        return flags

    @staticmethod
    def _severity_from_flags(flags: List[str]) -> str:
        """Derive the overall severity level from a list of flags.

        Args:
            flags: List of flag strings.

        Returns:
            One of ``"OK"``, ``"WARNING"``, or ``"ALERT"``.
        """
        if any(f.startswith("ALERT") for f in flags):
            return "ALERT"
        if any(f.startswith(("WARNING", "DRIFT")) for f in flags):
            return "WARNING"
        return "OK"


# ---------------------------------------------------------------------------
# CalibrationReport
# ---------------------------------------------------------------------------

class CalibrationReport:
    """Generate a weekly markdown report from a ``CalibrationSnapshot``."""

    @staticmethod
    def generate(snapshot: CalibrationSnapshot) -> str:
        """Build a human-readable markdown report.

        Args:
            snapshot: The calibration snapshot to report on.

        Returns:
            A markdown-formatted string.
        """
        lines: List[str] = [
            f"# Calibration Report -- {snapshot.model_version}",
            "",
            f"**Evaluated at:** {snapshot.evaluated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Window:** {snapshot.window_start.strftime('%Y-%m-%d')} to "
            f"{snapshot.window_end.strftime('%Y-%m-%d')}",
            f"**Total samples:** {snapshot.total_samples}",
            "",
            "## Summary",
            "",
            f"| Metric | Value | Threshold | Status |",
            f"|--------|-------|-----------|--------|",
            f"| ECE | {snapshot.ece:.4f} | WARNING > {ECE_WARNING_THRESHOLD}, "
            f"ALERT > {ECE_ALERT_THRESHOLD} | "
            f"{'PASS' if snapshot.ece <= ECE_WARNING_THRESHOLD else snapshot.severity} |",
            f"| K-S Statistic | {snapshot.ks_statistic:.4f} | > {KS_DRIFT_THRESHOLD} | "
            f"{'PASS' if snapshot.ks_statistic <= KS_DRIFT_THRESHOLD else 'DRIFT'} |",
            f"| K-S p-value | {snapshot.ks_p_value:.4f} | -- | -- |",
            "",
            "## Reliability Diagram (bins)",
            "",
            "| Bin | Predicted | Observed | Count | Accuracy |",
            "|-----|-----------|----------|-------|----------|",
        ]

        for b in snapshot.bins:
            lines.append(
                f"| [{b.bin_lower:.2f}, {b.bin_upper:.2f}) "
                f"| {b.mean_predicted:.4f} "
                f"| {b.mean_observed:.4f} "
                f"| {b.count} "
                f"| {b.accuracy:.4f} |"
            )

        lines.append("")

        if snapshot.flags:
            lines.append("## Flags")
            lines.append("")
            for flag in snapshot.flags:
                lines.append(f"- {flag}")
            lines.append("")

        if snapshot.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in snapshot.metadata.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.4f}")
                else:
                    lines.append(f"- **{key}:** {value}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Overall severity: **{snapshot.severity}***")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(name="ml.model_management.calibration_monitor.run_weekly_calibration_check")
def run_weekly_calibration_check() -> Dict[str, Any]:
    """Celery task: weekly calibration evaluation.

    Pulls the last 7 days of ``(prediction, outcome)`` pairs from the
    database, runs the calibration monitor, persists the snapshot, and
    returns a summary dictionary.

    Returns:
        A dictionary containing the snapshot summary and any flags.
    """
    logger.info("Starting weekly calibration check")
    db = SyncSessionLocal()

    try:
        from app.models.models import (
            UserFeedback,
            RecommendedOccupation,
            RecommendationEvent,
            ActionType,
            ModelRegistry,
        )
        from sqlalchemy import and_

        window_end = datetime.utcnow()
        window_start = window_end - timedelta(days=7)

        # Pull predictions and outcomes from the last 7 days.
        # Predictions come from score_json.match_score (normalised to 0-1).
        # Outcomes are derived from feedback action type.
        query = (
            db.query(
                RecommendedOccupation.score_json,
                UserFeedback.action_type,
            )
            .join(
                UserFeedback,
                and_(
                    UserFeedback.event_id == RecommendedOccupation.event_id,
                    UserFeedback.target_onet_code == RecommendedOccupation.target_onet_code,
                ),
            )
            .filter(UserFeedback.action_at >= window_start)
            .filter(UserFeedback.action_at <= window_end)
            .filter(
                UserFeedback.action_type.in_([
                    ActionType.APPLY,
                    ActionType.INTERVIEW,
                    ActionType.OFFER,
                    ActionType.HIDE,
                ])
            )
        )

        rows = query.all()

        if len(rows) < 20:
            logger.warning(
                "Insufficient data for calibration check: %d rows (need >= 20)",
                len(rows),
            )
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "samples": len(rows),
            }

        predictions: List[float] = []
        outcomes: List[float] = []

        for score_json, action_type in rows:
            if score_json is None:
                continue
            # Normalise match_score from 0-100 to 0-1 as a probability proxy.
            match_score = score_json.get("match_score", 0.0)
            pred = max(0.0, min(1.0, match_score / 100.0))
            predictions.append(pred)

            outcome = 1.0 if action_type in (
                ActionType.APPLY, ActionType.INTERVIEW, ActionType.OFFER,
            ) else 0.0
            outcomes.append(outcome)

        if len(predictions) < 20:
            return {
                "status": "skipped",
                "reason": "insufficient_valid_data",
                "samples": len(predictions),
            }

        # Determine current active model version.
        active_model = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.is_active.is_(True))
            .order_by(ModelRegistry.trained_at.desc())
            .first()
        )
        model_version = (
            active_model.model_version if active_model else settings.model_version
        )

        # Run evaluation.
        monitor = CalibrationMonitor()
        snapshot = monitor.evaluate(
            predictions=np.array(predictions),
            outcomes=np.array(outcomes),
            model_version=model_version,
            window_start=window_start,
            window_end=window_end,
        )

        # Persist the snapshot as metadata on the active model registry entry.
        if active_model is not None:
            existing_meta = active_model.metrics_json or {}
            calibration_history = existing_meta.get("calibration_snapshots", [])
            calibration_history.append(monitor.snapshot_to_dict(snapshot))
            existing_meta["calibration_snapshots"] = calibration_history
            existing_meta["latest_calibration"] = monitor.snapshot_to_dict(snapshot)
            active_model.metrics_json = existing_meta
            db.commit()
            logger.info("Calibration snapshot persisted to model registry")

        # Generate report.
        report = CalibrationReport.generate(snapshot)
        logger.info("Weekly calibration report:\n%s", report)

        return {
            "status": "completed",
            "model_version": model_version,
            "ece": snapshot.ece,
            "ks_statistic": snapshot.ks_statistic,
            "severity": snapshot.severity,
            "flags": snapshot.flags,
            "total_samples": snapshot.total_samples,
        }

    except Exception as exc:
        logger.error("Calibration check failed: %s", exc, exc_info=True)
        db.rollback()
        return {"status": "error", "error": str(exc)}
    finally:
        db.close()
