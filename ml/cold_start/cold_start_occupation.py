"""Cold-start strategy for rare occupations with few interactions.

When a *target occupation* has fewer than a configurable threshold of
recorded interactions, the calibration model has very little signal.  This
module clusters occupations by their O*NET skill vectors using k-means and
uses the cluster-level calibration as a stand-in.

As data accumulates for the individual occupation the model linearly blends
from the cluster prior to the occupation-specific calibration.

Companion utility
-----------------
:func:`compute_silhouette_score` (the ``cluster_quality`` function
referenced in the spec) reports the silhouette score for the current
clustering so operators can monitor quality over time.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sk_silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default number of clusters
DEFAULT_K = 50
# Minimum interactions before we fully trust occupation-specific data
DEFAULT_MIN_INTERACTIONS = 30


class OccupationClusterModel:
    """Cluster occupations by O*NET skill vectors for cold-start fallback.

    Parameters
    ----------
    k : int
        Number of k-means clusters.
    min_interactions : int
        Below this threshold the cluster calibration is used.
    random_state : int
        Seed for k-means reproducibility.
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        min_interactions: int = DEFAULT_MIN_INTERACTIONS,
        random_state: int = 42,
    ) -> None:
        self.k = k
        self.min_interactions = min_interactions
        self.random_state = random_state

        self._kmeans: Optional[KMeans] = None
        self._scaler: Optional[StandardScaler] = None
        self._onet_codes: List[str] = []
        self._skill_ids: List[str] = []
        self._cluster_labels: Optional[np.ndarray] = None
        self._X_scaled: Optional[np.ndarray] = None  # stored for silhouette
        self._cluster_scores: Dict[int, float] = {}
        self._occupation_scores: Dict[str, float] = {}
        self._occupation_interaction_counts: Dict[str, int] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        occupation_skill_data: Dict[str, List[Dict[str, Any]]],
        occupation_scores: Optional[Dict[str, float]] = None,
        occupation_interaction_counts: Optional[Dict[str, int]] = None,
    ) -> "OccupationClusterModel":
        """Fit k-means on occupation skill vectors.

        Args:
            occupation_skill_data: Mapping from O*NET code to list of skill
                dicts.  Each skill dict must contain ``element_id`` and
                ``importance``.
            occupation_scores: Optional mapping of O*NET code to
                calibration score for that occupation (aggregated from
                interactions).  Used to compute cluster-level scores.
            occupation_interaction_counts: Optional mapping of O*NET code
                to total interaction count.

        Returns:
            ``self`` for chaining.
        """
        if not occupation_skill_data:
            raise ValueError("occupation_skill_data must be non-empty")

        self._onet_codes = sorted(occupation_skill_data.keys())
        self._occupation_scores = occupation_scores or {}
        self._occupation_interaction_counts = occupation_interaction_counts or {}

        # Build the union of all skill element IDs
        all_skill_ids: set = set()
        for skills in occupation_skill_data.values():
            for s in skills:
                eid = s.get("element_id")
                if eid:
                    all_skill_ids.add(eid)
        self._skill_ids = sorted(all_skill_ids)

        if not self._skill_ids:
            raise ValueError("No valid skill element IDs found in data")

        eid_to_idx = {eid: i for i, eid in enumerate(self._skill_ids)}

        # Build feature matrix (occupations x skills)
        n_occ = len(self._onet_codes)
        n_skills = len(self._skill_ids)
        X = np.zeros((n_occ, n_skills), dtype=np.float64)

        for row, code in enumerate(self._onet_codes):
            for s in occupation_skill_data[code]:
                eid = s.get("element_id")
                imp = s.get("importance")
                if eid and eid in eid_to_idx and imp is not None:
                    X[row, eid_to_idx[eid]] = float(imp)

        # Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._X_scaled = X_scaled  # keep for silhouette computation

        # Adjust k if we have fewer occupations than clusters
        effective_k = min(self.k, n_occ)
        if effective_k < 2:
            effective_k = 2 if n_occ >= 2 else 1

        self._kmeans = KMeans(
            n_clusters=effective_k,
            random_state=self.random_state,
            n_init=10,
        )
        self._cluster_labels = self._kmeans.fit_predict(X_scaled)

        # Compute cluster-level scores as the mean of member-occupation
        # scores (for occupations that have sufficient data).
        self._compute_cluster_scores()

        self._fitted = True
        logger.info(
            "OccupationClusterModel fitted: %d occupations, %d skills, k=%d",
            n_occ,
            n_skills,
            effective_k,
        )
        return self

    def _compute_cluster_scores(self) -> None:
        """Compute average calibration score per cluster."""
        if self._cluster_labels is None:
            return

        cluster_sums: Dict[int, List[float]] = {}
        for idx, code in enumerate(self._onet_codes):
            label = int(self._cluster_labels[idx])
            count = self._occupation_interaction_counts.get(code, 0)
            if count >= self.min_interactions and code in self._occupation_scores:
                cluster_sums.setdefault(label, []).append(
                    self._occupation_scores[code]
                )

        self._cluster_scores = {}
        for label, scores in cluster_sums.items():
            self._cluster_scores[label] = float(np.mean(scores))

        # For clusters with no data at all, use the global mean
        if self._occupation_scores:
            global_mean = float(np.mean(list(self._occupation_scores.values())))
        else:
            global_mean = 50.0  # neutral default

        n_clusters = self._kmeans.n_clusters if self._kmeans else 1
        for label in range(n_clusters):
            if label not in self._cluster_scores:
                self._cluster_scores[label] = global_mean

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def get_cluster(self, onet_code: str) -> Optional[int]:
        """Return the cluster label for a given occupation.

        Args:
            onet_code: O*NET-SOC code.

        Returns:
            Cluster label (int), or ``None`` if the code was not in the
            training set.
        """
        if not self._fitted or onet_code not in self._onet_codes:
            return None

        idx = self._onet_codes.index(onet_code)
        return int(self._cluster_labels[idx])

    def predict_cluster_for_skills(
        self,
        skills: List[Dict[str, Any]],
    ) -> int:
        """Predict the cluster for an unseen occupation given its skills.

        This is useful when the occupation was not in the training data but
        we have its skill vector (e.g. freshly fetched from O*NET).

        Args:
            skills: List of skill dicts with ``element_id`` and
                ``importance``.

        Returns:
            Predicted cluster label.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        vec = np.zeros(len(self._skill_ids), dtype=np.float64)
        eid_to_idx = {eid: i for i, eid in enumerate(self._skill_ids)}

        for s in skills:
            eid = s.get("element_id")
            imp = s.get("importance")
            if eid and eid in eid_to_idx and imp is not None:
                vec[eid_to_idx[eid]] = float(imp)

        vec_scaled = self._scaler.transform(vec.reshape(1, -1))
        label = self._kmeans.predict(vec_scaled)[0]
        return int(label)

    def get_cluster_score(self, cluster_label: int) -> float:
        """Return the calibration score associated with a cluster.

        Args:
            cluster_label: Cluster label (from :meth:`get_cluster` or
                :meth:`predict_cluster_for_skills`).

        Returns:
            Cluster-level calibration score (0-100 scale).
        """
        return self._cluster_scores.get(cluster_label, 50.0)

    def blend_score(
        self,
        onet_code: str,
        occupation_specific_score: float,
        interaction_count: int,
    ) -> float:
        """Blend cluster calibration with occupation-specific calibration.

        The blend weight for the cluster decreases linearly as the
        occupation accumulates interactions::

            cluster_weight = max(0, 1 - interaction_count / min_interactions)
            blended = cluster_weight * cluster_score
                      + (1 - cluster_weight) * occupation_score

        Args:
            onet_code: O*NET-SOC code for the target occupation.
            occupation_specific_score: Score computed from the occupation's
                own interaction data (0-100 scale).
            interaction_count: Number of recorded interactions for this
                occupation.

        Returns:
            Blended score (0-100 scale).
        """
        if not self._fitted:
            return occupation_specific_score

        cluster_label = self.get_cluster(onet_code)
        if cluster_label is None:
            return occupation_specific_score

        cluster_score = self.get_cluster_score(cluster_label)

        if interaction_count >= self.min_interactions:
            return occupation_specific_score

        # Linear blend
        cluster_weight = max(0.0, 1.0 - interaction_count / self.min_interactions)
        blended = (
            cluster_weight * cluster_score
            + (1.0 - cluster_weight) * occupation_specific_score
        )
        return float(blended)

    # ------------------------------------------------------------------
    # Cluster quality
    # ------------------------------------------------------------------

    @property
    def silhouette(self) -> Optional[float]:
        """Return the silhouette score for the fitted clustering.

        Returns ``None`` if the model is not fitted or has too few data
        points.
        """
        if not self._fitted:
            return None
        return compute_silhouette_score(self)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def n_clusters(self) -> int:
        """Effective number of clusters."""
        if self._kmeans is None:
            return 0
        return self._kmeans.n_clusters

    @property
    def cluster_sizes(self) -> Dict[int, int]:
        """Return the number of occupations in each cluster."""
        if self._cluster_labels is None:
            return {}
        unique, counts = np.unique(self._cluster_labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}


# ---------------------------------------------------------------------------
# cluster_quality.py function (co-located for convenience)
# ---------------------------------------------------------------------------


def compute_silhouette_score(
    model: OccupationClusterModel,
) -> Optional[float]:
    """Compute the silhouette score for the model's clustering.

    The silhouette score measures how similar each occupation is to its own
    cluster compared to the nearest neighbouring cluster.  Values range
    from -1 (poor) to +1 (excellent).

    Args:
        model: A fitted :class:`OccupationClusterModel`.

    Returns:
        Silhouette score (float), or ``None`` if computation is not
        possible (e.g. fewer than 2 clusters or 2 samples).
    """
    if not model._fitted:
        logger.warning("Cannot compute silhouette: model is not fitted")
        return None

    if model._X_scaled is None or model._cluster_labels is None:
        return None

    n_labels = len(set(model._cluster_labels))
    n_samples = len(model._onet_codes)

    # Silhouette requires at least 2 clusters and at least 2 samples
    if n_labels < 2 or n_samples < 2:
        logger.warning(
            "Cannot compute silhouette: n_labels=%d, n_samples=%d",
            n_labels,
            n_samples,
        )
        return None

    try:
        score = sk_silhouette_score(model._X_scaled, model._cluster_labels)
        return float(score)
    except Exception as e:
        logger.error("Silhouette computation failed: %s", e)
        return None
