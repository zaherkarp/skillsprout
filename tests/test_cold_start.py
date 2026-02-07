"""Tests for all cold-start modules.

Covers:
- cold_start_user.OccupationPriorModel
- cold_start_occupation.OccupationClusterModel  (+ silhouette)
- cold_start_combination.NoveltyDetector
"""

import pytest
import numpy as np

from ml.cold_start.cold_start_user import OccupationPriorModel
from ml.cold_start.cold_start_occupation import (
    OccupationClusterModel,
    compute_silhouette_score,
)
from ml.cold_start.cold_start_combination import (
    NoveltyDetector,
    NoveltyAssessment,
    UncertaintyLevel,
    should_use_fallback,
)


# ===================================================================
# Shared helpers
# ===================================================================

def _make_skill_data(n_occupations: int = 10, n_skills: int = 8):
    """Generate synthetic occupation skill data for cluster tests.

    Returns a dict mapping O*NET-style codes to skill dicts with distinct
    profiles so clusters are separable.
    """
    rng = np.random.RandomState(42)
    data = {}
    skill_ids = [f"skill_{i}" for i in range(n_skills)]

    for i in range(n_occupations):
        code = f"{10 + i}-0001.00"
        skills = []
        for sid in skill_ids:
            importance = float(rng.uniform(10, 100))
            skills.append({"element_id": sid, "importance": importance})
        data[code] = skills

    return data


# ===================================================================
# OccupationPriorModel tests
# ===================================================================

class TestOccupationPriorModel:
    """Tests for cold_start_user.OccupationPriorModel."""

    def test_uniform_fallback_below_min_users(self):
        """With < 10 total interactions the prior is uniform (0.5)."""
        store = {
            "15-1252.00": {
                "15-1299.08": (2, 3),  # 3 interactions
                "15-1244.00": (1, 2),  # 2 interactions
            },
        }
        model = OccupationPriorModel(interaction_store=store, min_users_for_prior=10)
        assert model.is_uniform_fallback is True
        assert model.lookup_prior("15-1252.00", "15-1299.08") == pytest.approx(0.5)
        assert model.lookup_prior("15-1252.00", "anything") == pytest.approx(0.5)

    def test_laplace_smoothing_on_zero_counts(self):
        """A (origin, target) pair with zero counts gets Laplace-smoothed prior."""
        store = {
            "15-1252.00": {
                "15-1299.08": (8, 10),
                "15-1244.00": (3, 10),
            },
        }
        # Total = 20, above min_users = 10
        model = OccupationPriorModel(interaction_store=store, min_users_for_prior=10)
        assert model.is_uniform_fallback is False

        # Unseen target pair: (0 + 1) / (0 + 2) = 0.5
        prior = model.lookup_prior("15-1252.00", "29-1141.00")
        assert prior == pytest.approx(0.5)

    def test_prior_matches_observed_rate(self):
        """Prior approximates observed positive rate with Laplace correction."""
        store = {
            "15-1252.00": {
                "15-1299.08": (8, 10),
            },
        }
        model = OccupationPriorModel(
            interaction_store=store,
            min_users_for_prior=5,
            laplace_alpha=1.0,
        )
        # (8 + 1) / (10 + 2) = 9/12 = 0.75
        prior = model.lookup_prior("15-1252.00", "15-1299.08")
        assert prior == pytest.approx(9.0 / 12.0)

    def test_prior_weight_decay(self):
        """Prior weight decays linearly to 0 at halflife interactions."""
        model = OccupationPriorModel(max_prior_weight=0.3, interaction_halflife=20)

        # No interactions -> full weight
        assert model.prior_weight(0) == pytest.approx(0.3)

        # 10 interactions -> 0.3 * (1 - 10/20) = 0.15
        assert model.prior_weight(10) == pytest.approx(0.15)

        # 20 interactions -> 0
        assert model.prior_weight(20) == pytest.approx(0.0)

        # 30 interactions -> still 0 (clamped)
        assert model.prior_weight(30) == pytest.approx(0.0)

    def test_blend_no_interactions(self):
        """With 0 interactions, blended score mixes prior and scorer."""
        store = {"A": {"B": (5, 10)}}
        model = OccupationPriorModel(
            interaction_store=store,
            min_users_for_prior=5,
            max_prior_weight=0.3,
        )
        # prior for A->B: (5+1)/(10+2) = 0.5 -> scaled to 50.0
        # scorer_output = 80.0
        # blended = 0.3 * 50 + 0.7 * 80 = 15 + 56 = 71.0
        blended = model.blend("A", "B", scorer_output=80.0, user_interaction_count=0)
        assert blended == pytest.approx(71.0)

    def test_blend_full_interactions(self):
        """With enough interactions, prior weight is 0 -> scorer pass-through."""
        store = {"A": {"B": (5, 10)}}
        model = OccupationPriorModel(
            interaction_store=store,
            min_users_for_prior=5,
            max_prior_weight=0.3,
            interaction_halflife=20,
        )
        blended = model.blend("A", "B", scorer_output=80.0, user_interaction_count=20)
        assert blended == pytest.approx(80.0)

    def test_update_store(self):
        """update_store() adds interactions and invalidates cache."""
        model = OccupationPriorModel(min_users_for_prior=1)
        model.update_store("A", "B", is_positive=True)
        model.update_store("A", "B", is_positive=False)

        # Now A->B has (1, 2)
        prior = model.lookup_prior("A", "B")
        # (1 + 1) / (2 + 2) = 0.5
        assert prior == pytest.approx(0.5)

    def test_empty_store(self):
        """Empty store -> uniform fallback."""
        model = OccupationPriorModel()
        assert model.is_uniform_fallback is True
        assert model.lookup_prior("any", "any") == pytest.approx(0.5)


# ===================================================================
# OccupationClusterModel tests
# ===================================================================

class TestOccupationClusterModel:
    """Tests for cold_start_occupation.OccupationClusterModel."""

    def test_fit_basic(self):
        """Fitting on synthetic data produces valid clusters."""
        data = _make_skill_data(n_occupations=20, n_skills=6)
        model = OccupationClusterModel(k=3, random_state=42)
        model.fit(data)

        assert model._fitted is True
        assert model.n_clusters == 3
        assert sum(model.cluster_sizes.values()) == 20

    def test_get_cluster_returns_valid_label(self):
        """get_cluster() returns a label for known codes."""
        data = _make_skill_data(n_occupations=10)
        model = OccupationClusterModel(k=3, random_state=42)
        model.fit(data)

        code = sorted(data.keys())[0]
        label = model.get_cluster(code)
        assert label is not None
        assert 0 <= label < 3

    def test_get_cluster_unknown_code(self):
        """get_cluster() returns None for unknown codes."""
        data = _make_skill_data(n_occupations=10)
        model = OccupationClusterModel(k=3, random_state=42)
        model.fit(data)

        assert model.get_cluster("99-9999.00") is None

    def test_predict_cluster_for_skills(self):
        """predict_cluster_for_skills() returns a valid label."""
        data = _make_skill_data(n_occupations=10, n_skills=4)
        model = OccupationClusterModel(k=2, random_state=42)
        model.fit(data)

        new_skills = [
            {"element_id": "skill_0", "importance": 50.0},
            {"element_id": "skill_1", "importance": 60.0},
        ]
        label = model.predict_cluster_for_skills(new_skills)
        assert 0 <= label < 2

    def test_predict_cluster_unfitted_raises(self):
        """predict_cluster_for_skills() raises if not fitted."""
        model = OccupationClusterModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_cluster_for_skills([{"element_id": "A", "importance": 50}])

    def test_blend_score_zero_interactions(self):
        """With 0 interactions, blended score equals cluster score."""
        data = _make_skill_data(n_occupations=10)
        scores = {code: 70.0 for code in data}
        counts = {code: 100 for code in data}  # all well-supported

        model = OccupationClusterModel(k=2, min_interactions=30, random_state=42)
        model.fit(data, occupation_scores=scores, occupation_interaction_counts=counts)

        code = sorted(data.keys())[0]
        cluster_score = model.get_cluster_score(model.get_cluster(code))

        blended = model.blend_score(code, occupation_specific_score=90.0, interaction_count=0)
        # With 0 interactions, cluster_weight = 1.0
        assert blended == pytest.approx(cluster_score)

    def test_blend_score_full_interactions(self):
        """With >= min_interactions, blended score equals occ-specific score."""
        data = _make_skill_data(n_occupations=10)
        model = OccupationClusterModel(k=2, min_interactions=30, random_state=42)
        model.fit(data)

        code = sorted(data.keys())[0]
        blended = model.blend_score(code, occupation_specific_score=85.0, interaction_count=30)
        assert blended == pytest.approx(85.0)

    def test_blend_score_partial_interactions(self):
        """With partial interactions, blended score is between cluster and occ."""
        data = _make_skill_data(n_occupations=10)
        scores = {code: 60.0 for code in data}
        counts = {code: 50 for code in data}

        model = OccupationClusterModel(k=2, min_interactions=30, random_state=42)
        model.fit(data, occupation_scores=scores, occupation_interaction_counts=counts)

        code = sorted(data.keys())[0]
        cluster_score = model.get_cluster_score(model.get_cluster(code))
        occ_score = 90.0

        blended = model.blend_score(code, occupation_specific_score=occ_score, interaction_count=15)
        # cluster_weight = 1 - 15/30 = 0.5
        expected = 0.5 * cluster_score + 0.5 * occ_score
        assert blended == pytest.approx(expected)

    def test_fit_empty_data_raises(self):
        """Fitting on empty data raises ValueError."""
        model = OccupationClusterModel()
        with pytest.raises(ValueError, match="non-empty"):
            model.fit({})

    def test_k_exceeds_occupations(self):
        """k is automatically reduced when fewer occupations than k."""
        data = _make_skill_data(n_occupations=3)
        model = OccupationClusterModel(k=50, random_state=42)
        model.fit(data)
        assert model.n_clusters == 3  # capped to n_occupations


# ===================================================================
# Silhouette score tests
# ===================================================================

class TestSilhouetteScore:
    """Tests for compute_silhouette_score()."""

    def test_silhouette_valid_clustering(self):
        """Silhouette score is in [-1, 1] for valid clustering."""
        # Generate 30 occupations with well-separated clusters
        rng = np.random.RandomState(42)
        data = {}
        for i in range(30):
            code = f"{10 + i}-0001.00"
            if i < 10:
                base = [80.0, 20.0, 20.0, 80.0]
            elif i < 20:
                base = [20.0, 80.0, 80.0, 20.0]
            else:
                base = [50.0, 50.0, 20.0, 80.0]
            skills = [
                {"element_id": f"s{j}", "importance": base[j] + rng.normal(0, 5)}
                for j in range(4)
            ]
            data[code] = skills

        model = OccupationClusterModel(k=3, random_state=42)
        model.fit(data)

        sil = compute_silhouette_score(model)
        assert sil is not None
        assert -1.0 <= sil <= 1.0

    def test_silhouette_unfitted_returns_none(self):
        """Unfitted model returns None."""
        model = OccupationClusterModel()
        assert compute_silhouette_score(model) is None

    def test_silhouette_property(self):
        """model.silhouette property works."""
        data = _make_skill_data(n_occupations=10)
        model = OccupationClusterModel(k=2, random_state=42)
        model.fit(data)
        sil = model.silhouette
        assert sil is not None
        assert -1.0 <= sil <= 1.0


# ===================================================================
# NoveltyDetector tests
# ===================================================================

class TestNoveltyDetector:
    """Tests for cold_start_combination.NoveltyDetector."""

    @pytest.fixture
    def fitted_detector(self):
        """Detector fitted on a Gaussian blob."""
        rng = np.random.RandomState(42)
        X = rng.multivariate_normal(
            mean=[0, 0, 0],
            cov=[[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]],
            size=200,
        )
        detector = NoveltyDetector(medium_percentile=90, high_percentile=99)
        detector.fit(X)
        return detector

    def test_fit_sets_thresholds(self, fitted_detector):
        """After fitting, thresholds are positive and ordered."""
        assert fitted_detector.medium_threshold > 0
        assert fitted_detector.high_threshold > fitted_detector.medium_threshold

    def test_inlier_low_uncertainty(self, fitted_detector):
        """A point near the centroid should be LOW uncertainty."""
        x = np.array([0.0, 0.0, 0.0])
        result = fitted_detector.assess(x)
        assert result.uncertainty == UncertaintyLevel.LOW
        assert result.use_fallback is False

    def test_outlier_high_uncertainty(self, fitted_detector):
        """A far-away point should be HIGH uncertainty."""
        x = np.array([100.0, 100.0, 100.0])
        result = fitted_detector.assess(x)
        assert result.uncertainty == UncertaintyLevel.HIGH
        assert result.use_fallback is True

    def test_moderate_point_medium_or_high(self, fitted_detector):
        """A moderately distant point should be MEDIUM or HIGH."""
        x = np.array([4.0, 4.0, 4.0])
        result = fitted_detector.assess(x)
        assert result.uncertainty in (UncertaintyLevel.MEDIUM, UncertaintyLevel.HIGH)

    def test_assess_handles_nan(self, fitted_detector):
        """NaN values are imputed with training mean."""
        x = np.array([np.nan, 0.0, 0.0])
        result = fitted_detector.assess(x)
        # Should not raise and should produce a valid assessment
        assert result.uncertainty in (
            UncertaintyLevel.LOW,
            UncertaintyLevel.MEDIUM,
            UncertaintyLevel.HIGH,
        )

    def test_assess_batch(self, fitted_detector):
        """assess_batch() returns one result per row."""
        X = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
            [1.0, 1.0, 1.0],
        ])
        results = fitted_detector.assess_batch(X)
        assert len(results) == 3
        assert results[0].uncertainty == UncertaintyLevel.LOW
        assert results[1].uncertainty == UncertaintyLevel.HIGH

    def test_unfitted_assess_raises(self):
        """assess() raises RuntimeError if not fitted."""
        detector = NoveltyDetector()
        with pytest.raises(RuntimeError, match="fitted"):
            detector.assess(np.array([1.0, 2.0]))

    def test_fit_single_feature(self):
        """Fitting with single feature works (edge case)."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, size=(50, 1))
        detector = NoveltyDetector()
        detector.fit(X)

        result = detector.assess(np.array([0.0]))
        assert result.uncertainty == UncertaintyLevel.LOW

    def test_fit_too_few_samples_raises(self):
        """Fitting with < 2 samples raises ValueError."""
        X = np.array([[1.0, 2.0]])
        detector = NoveltyDetector()
        with pytest.raises(ValueError, match="at least 2"):
            detector.fit(X)

    def test_percentile_near_zero_for_centroid(self, fitted_detector):
        """A point at the centroid should have a low percentile."""
        result = fitted_detector.assess(np.array([0.0, 0.0, 0.0]))
        assert result.percentile < 50.0

    def test_percentile_near_100_for_outlier(self, fitted_detector):
        """A far outlier should have percentile near 100."""
        result = fitted_detector.assess(np.array([100.0, 100.0, 100.0]))
        assert result.percentile >= 99.0

    def test_invalid_percentile_config(self):
        """Invalid percentile bounds raise ValueError."""
        with pytest.raises(ValueError, match="medium_percentile"):
            NoveltyDetector(medium_percentile=99, high_percentile=90)

    def test_fit_with_nan_values(self):
        """fit() handles NaN values via imputation."""
        X = np.array([
            [1.0, 2.0],
            [3.0, np.nan],
            [np.nan, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        detector = NoveltyDetector()
        detector.fit(X)
        assert detector.is_fitted is True


# ===================================================================
# should_use_fallback convenience function tests
# ===================================================================

class TestShouldUseFallback:
    """Tests for the should_use_fallback convenience function."""

    def test_inlier_no_fallback(self):
        """Inlier vector does not trigger fallback."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, size=(100, 3))
        detector = NoveltyDetector()
        detector.fit(X)

        assert should_use_fallback(detector, np.array([0.0, 0.0, 0.0])) is False

    def test_outlier_triggers_fallback(self):
        """Extreme outlier triggers fallback."""
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, size=(100, 3))
        detector = NoveltyDetector()
        detector.fit(X)

        assert should_use_fallback(detector, np.array([50.0, 50.0, 50.0])) is True
