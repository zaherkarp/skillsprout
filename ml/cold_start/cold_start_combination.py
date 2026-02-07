"""Cold-start strategy for unseen skill combinations.

Even when both the *user* and the *occupation* have sufficient history, a
particular ``(origin, target)`` skill combination may be novel -- i.e. the
calibration model has never been trained on a feature vector that looks
like this.

This module uses **Mahalanobis distance** from the training distribution to
flag novel combinations and assign an uncertainty level:

* ``LOW``   -- within the training support, predictions reliable
* ``MEDIUM`` -- on the fringe, predictions less certain
* ``HIGH``  -- far outside training support, fall back to v1 scorer

Integration point
-----------------
After the calibration model produces a prediction, call
:meth:`NoveltyDetector.assess` with the feature vector.  If the returned
uncertainty is ``HIGH``, discard the calibration output and use the
``BaselineScorer`` (v1) result instead.
"""

from __future__ import annotations

import enum
import logging
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyLevel(str, enum.Enum):
    """Uncertainty classification for a feature vector."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NoveltyAssessment:
    """Result of novelty detection for a single feature vector.

    Attributes:
        mahalanobis_distance: Mahalanobis distance from the training
            distribution centroid.
        uncertainty: Classified uncertainty level.
        use_fallback: ``True`` when the caller should discard calibrated
            scores and use the v1 baseline scorer instead.
        percentile: Approximate percentile of this distance relative to
            the training set distances (0-100).
    """

    __slots__ = ("mahalanobis_distance", "uncertainty", "use_fallback", "percentile")

    def __init__(
        self,
        mahalanobis_distance: float,
        uncertainty: UncertaintyLevel,
        use_fallback: bool,
        percentile: float,
    ) -> None:
        self.mahalanobis_distance = mahalanobis_distance
        self.uncertainty = uncertainty
        self.use_fallback = use_fallback
        self.percentile = percentile

    def __repr__(self) -> str:
        return (
            f"NoveltyAssessment(distance={self.mahalanobis_distance:.3f}, "
            f"uncertainty={self.uncertainty.value}, "
            f"use_fallback={self.use_fallback}, "
            f"percentile={self.percentile:.1f})"
        )


class NoveltyDetector:
    """Detect novel (unseen) feature combinations via Mahalanobis distance.

    After fitting on the training-set feature matrix the detector computes
    the Mahalanobis distance for new vectors and classifies them into
    uncertainty buckets.

    Parameters
    ----------
    medium_percentile : float
        Distance percentile above which uncertainty is MEDIUM (default 90).
    high_percentile : float
        Distance percentile above which uncertainty is HIGH (default 99).
    regularisation : float
        Small constant added to the covariance diagonal to ensure
        invertibility (default 1e-6).
    """

    def __init__(
        self,
        medium_percentile: float = 90.0,
        high_percentile: float = 99.0,
        regularisation: float = 1e-6,
    ) -> None:
        if not (0.0 < medium_percentile < high_percentile <= 100.0):
            raise ValueError(
                "Must have 0 < medium_percentile < high_percentile <= 100; "
                f"got medium={medium_percentile}, high={high_percentile}"
            )

        self._medium_pct = medium_percentile
        self._high_pct = high_percentile
        self._reg = regularisation

        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._training_distances: Optional[np.ndarray] = None
        self._medium_threshold: float = 0.0
        self._high_threshold: float = 0.0
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "NoveltyDetector":
        """Fit the detector on a training feature matrix.

        Args:
            X: 2-D array of shape ``(n_samples, n_features)``.  ``NaN``
                values are replaced with column means before fitting.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``X`` has fewer than 2 samples or 1 feature.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 samples to fit, got {n_samples}"
            )
        if n_features < 1:
            raise ValueError(
                f"Need at least 1 feature, got {n_features}"
            )

        # Impute NaN with column means
        X_clean = self._impute_nan(X)

        self._mean = np.mean(X_clean, axis=0)

        # Covariance with regularisation
        cov = np.cov(X_clean, rowvar=False)
        if cov.ndim == 0:
            # Single-feature edge case: cov is scalar
            cov = np.array([[float(cov)]])
        cov += np.eye(cov.shape[0]) * self._reg

        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning(
                "Covariance matrix singular even after regularisation; "
                "using pseudo-inverse."
            )
            self._cov_inv = np.linalg.pinv(cov)

        # Compute distances for training samples to set thresholds
        self._training_distances = self._mahalanobis_batch(X_clean)

        self._medium_threshold = float(
            np.percentile(self._training_distances, self._medium_pct)
        )
        self._high_threshold = float(
            np.percentile(self._training_distances, self._high_pct)
        )

        self._fitted = True
        logger.info(
            "NoveltyDetector fitted on %d samples x %d features. "
            "Thresholds: MEDIUM=%.3f (p%d), HIGH=%.3f (p%d)",
            n_samples,
            n_features,
            self._medium_threshold,
            int(self._medium_pct),
            self._high_threshold,
            int(self._high_pct),
        )
        return self

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def assess(self, x: np.ndarray) -> NoveltyAssessment:
        """Assess the novelty of a single feature vector.

        Args:
            x: 1-D array of shape ``(n_features,)`` or 2-D of shape
                ``(1, n_features)``.  ``NaN`` values are replaced with
                the training mean for that feature.

        Returns:
            :class:`NoveltyAssessment` with distance, uncertainty level,
            and fallback flag.

        Raises:
            RuntimeError: If the detector has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling assess()")

        x_flat = np.asarray(x, dtype=np.float64).flatten()

        # Impute NaN with training mean
        nan_mask = np.isnan(x_flat)
        if nan_mask.any():
            x_flat = x_flat.copy()
            x_flat[nan_mask] = self._mean[nan_mask]

        distance = self._mahalanobis_single(x_flat)
        percentile = self._distance_percentile(distance)

        if distance >= self._high_threshold:
            uncertainty = UncertaintyLevel.HIGH
        elif distance >= self._medium_threshold:
            uncertainty = UncertaintyLevel.MEDIUM
        else:
            uncertainty = UncertaintyLevel.LOW

        return NoveltyAssessment(
            mahalanobis_distance=float(distance),
            uncertainty=uncertainty,
            use_fallback=(uncertainty == UncertaintyLevel.HIGH),
            percentile=percentile,
        )

    def assess_batch(self, X: np.ndarray) -> List[NoveltyAssessment]:
        """Assess novelty for multiple feature vectors.

        Args:
            X: 2-D array of shape ``(n_samples, n_features)``.

        Returns:
            List of :class:`NoveltyAssessment`, one per row.
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling assess_batch()")

        results: List[NoveltyAssessment] = []
        for i in range(X.shape[0]):
            results.append(self.assess(X[i]))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mahalanobis_single(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance for a single vector."""
        diff = x - self._mean
        left = diff @ self._cov_inv
        dist_sq = left @ diff
        # Clamp to zero in case of floating-point issues
        return float(np.sqrt(max(dist_sq, 0.0)))

    def _mahalanobis_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances for a batch."""
        diff = X - self._mean
        left = diff @ self._cov_inv
        dist_sq = np.sum(left * diff, axis=1)
        dist_sq = np.clip(dist_sq, 0.0, None)
        return np.sqrt(dist_sq)

    def _distance_percentile(self, distance: float) -> float:
        """Return the approximate percentile of a distance.

        Uses the training distance distribution.
        """
        if self._training_distances is None or len(self._training_distances) == 0:
            return 50.0

        below = np.sum(self._training_distances <= distance)
        return float(below / len(self._training_distances) * 100.0)

    @staticmethod
    def _impute_nan(X: np.ndarray) -> np.ndarray:
        """Replace NaN values with column means."""
        X_out = X.copy()
        col_means = np.nanmean(X_out, axis=0)
        nan_mask = np.isnan(X_out)
        if nan_mask.any():
            # Replace NaN with column mean
            inds = np.where(nan_mask)
            X_out[inds] = np.take(col_means, inds[1])
        return X_out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has been fitted."""
        return self._fitted

    @property
    def medium_threshold(self) -> float:
        """Mahalanobis distance threshold for MEDIUM uncertainty."""
        return self._medium_threshold

    @property
    def high_threshold(self) -> float:
        """Mahalanobis distance threshold for HIGH uncertainty."""
        return self._high_threshold


# ---------------------------------------------------------------------------
# Convenience function for pipeline integration
# ---------------------------------------------------------------------------


def should_use_fallback(
    detector: NoveltyDetector,
    feature_vector: np.ndarray,
) -> bool:
    """Quick check: should the v1 scorer be used instead of calibration?

    Args:
        detector: A fitted :class:`NoveltyDetector`.
        feature_vector: The combined feature vector (base + transition).

    Returns:
        ``True`` if the vector is ``HIGH`` uncertainty and the v1 scorer
        should be used.
    """
    assessment = detector.assess(feature_vector)
    return assessment.use_fallback
