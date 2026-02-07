"""Tests for the offline evaluation framework.

These tests validate the entire evaluation pipeline -- from synthetic data
generation through proxy label assignment, temporal splitting, and metric
computation.  They use synthetic data so no database or external services are
needed.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.eval_framework import (
    CalibrationBinData,
    BucketMetrics,
    EvaluationFramework,
    EvaluationReport,
    ProxyLabelDefinition,
    TemporalSplitter,
    TemporalSplit,
)
from ml.evaluation.generate_synthetic_interactions import (
    GeneratorConfig,
    SyntheticDataGenerator,
    OCCUPATION_ARCHETYPES,
    SKILL_CATALOG,
    USER_ARCHETYPES,
)
from ml.evaluation.eval_runner import (
    parse_args,
    run_evaluation,
    score_v1_baseline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> GeneratorConfig:
    """Small config for fast tests."""
    return GeneratorConfig(
        n_users=30,
        recommendations_per_user=2,
        occupations_per_event=4,
        seed=42,
    )


@pytest.fixture
def generator(small_config: GeneratorConfig) -> SyntheticDataGenerator:
    """Generator with small config."""
    return SyntheticDataGenerator(small_config)


@pytest.fixture
def synthetic_data(generator: SyntheticDataGenerator) -> pd.DataFrame:
    """Generated synthetic interaction data."""
    return generator.generate()


@pytest.fixture
def framework() -> EvaluationFramework:
    """Default evaluation framework."""
    return EvaluationFramework()


@pytest.fixture
def labeled_data(
    framework: EvaluationFramework, synthetic_data: pd.DataFrame
) -> pd.DataFrame:
    """Synthetic data with proxy labels assigned."""
    return framework.assign_proxy_labels(synthetic_data)


# ---------------------------------------------------------------------------
# Synthetic data generator tests
# ---------------------------------------------------------------------------


class TestSyntheticDataGenerator:
    """Tests for the synthetic data generator."""

    def test_generator_produces_dataframe(
        self, generator: SyntheticDataGenerator
    ) -> None:
        """Generator should return a non-empty DataFrame."""
        df = generator.generate()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns_present(
        self, synthetic_data: pd.DataFrame
    ) -> None:
        """All expected columns should be present."""
        required_columns = {
            "event_id",
            "user_id",
            "target_onet_code",
            "occupation_title",
            "event_timestamp",
            "action_type",
            "action_timestamp",
            "next_visit_timestamp",
            "bucket",
            "match_score",
            "gap_severity",
            "rank",
            "model_version",
            "user_archetype",
            "num_missing_skills",
            "sum_missing_weights",
            "mean_rating",
            "rating_variance",
            "num_rated_skills",
            "job_zone_diff",
            "target_job_zone",
        }
        assert required_columns.issubset(
            set(synthetic_data.columns)
        ), f"Missing columns: {required_columns - set(synthetic_data.columns)}"

    def test_bucket_values_valid(self, synthetic_data: pd.DataFrame) -> None:
        """Bucket values should be one of the three valid buckets."""
        valid_buckets = {"ready_now", "trainable", "long_reskill"}
        actual_buckets = set(synthetic_data["bucket"].unique())
        assert actual_buckets.issubset(
            valid_buckets
        ), f"Invalid buckets: {actual_buckets - valid_buckets}"

    def test_match_score_range(self, synthetic_data: pd.DataFrame) -> None:
        """Match scores should be in [0, 100]."""
        assert synthetic_data["match_score"].min() >= 0.0
        assert synthetic_data["match_score"].max() <= 100.0

    def test_user_count(
        self, small_config: GeneratorConfig, synthetic_data: pd.DataFrame
    ) -> None:
        """Number of unique users should match config."""
        assert (
            synthetic_data["user_id"].nunique() == small_config.n_users
        )

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical data."""
        config = GeneratorConfig(n_users=10, seed=99)
        df1 = SyntheticDataGenerator(config).generate()
        df2 = SyntheticDataGenerator(config).generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different data."""
        df1 = SyntheticDataGenerator(GeneratorConfig(n_users=10, seed=1)).generate()
        df2 = SyntheticDataGenerator(GeneratorConfig(n_users=10, seed=2)).generate()
        # At least some values should differ
        assert not df1["match_score"].equals(df2["match_score"])

    def test_has_multiple_action_types(
        self, synthetic_data: pd.DataFrame
    ) -> None:
        """Data should contain a variety of action types including no-action."""
        action_types = set(synthetic_data["action_type"].dropna().unique())
        # Should have at least some actions (the exact set varies with seed)
        assert len(action_types) >= 2, f"Only found actions: {action_types}"

    def test_timestamps_are_datetime(
        self, synthetic_data: pd.DataFrame
    ) -> None:
        """Timestamp columns should be proper datetime types."""
        assert pd.api.types.is_datetime64_any_dtype(
            synthetic_data["event_timestamp"]
        )

    def test_archetype_distribution(
        self, synthetic_data: pd.DataFrame
    ) -> None:
        """User archetypes should cover multiple types."""
        archetypes = synthetic_data["user_archetype"].unique()
        assert len(archetypes) >= 2, "Expected at least 2 archetypes in the data"


# ---------------------------------------------------------------------------
# Proxy label tests
# ---------------------------------------------------------------------------


class TestProxyLabels:
    """Tests for proxy label assignment."""

    def test_labels_assigned(self, labeled_data: pd.DataFrame) -> None:
        """Some labels should be assigned (not all NaN)."""
        assert labeled_data["label"].notna().any(), "No labels were assigned"

    def test_label_values(self, labeled_data: pd.DataFrame) -> None:
        """Labels should be 0.0, 1.0, or NaN."""
        valid_values = {0.0, 1.0}
        non_nan = labeled_data["label"].dropna().unique()
        actual = set(non_nan)
        assert actual.issubset(
            valid_values
        ), f"Unexpected label values: {actual - valid_values}"

    def test_apply_is_positive(
        self, framework: EvaluationFramework
    ) -> None:
        """An 'apply' action should always be labeled POSITIVE."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-01")],
            "action_type": ["apply"],
            "action_timestamp": [pd.Timestamp("2025-06-02")],
            "next_visit_timestamp": [pd.NaT],
            "bucket": ["ready_now"],
        })
        result = framework.assign_proxy_labels(df)
        assert result["label"].iloc[0] == 1.0

    def test_interview_is_positive(
        self, framework: EvaluationFramework
    ) -> None:
        """An 'interview' action should always be labeled POSITIVE."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-01")],
            "action_type": ["interview"],
            "action_timestamp": [pd.Timestamp("2025-06-15")],
            "next_visit_timestamp": [pd.NaT],
            "bucket": ["trainable"],
        })
        result = framework.assign_proxy_labels(df)
        assert result["label"].iloc[0] == 1.0

    def test_no_action_within_window_is_negative(
        self, framework: EvaluationFramework
    ) -> None:
        """No action within 14 days should be labeled NEGATIVE."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-01")],
            "action_type": [None],
            "action_timestamp": [pd.NaT],
            "next_visit_timestamp": [pd.NaT],
            "bucket": ["long_reskill"],
        })
        # reference_date far enough in the future
        result = framework.assign_proxy_labels(
            df, reference_date=pd.Timestamp("2025-07-01")
        )
        assert result["label"].iloc[0] == 0.0

    def test_hide_is_negative(
        self, framework: EvaluationFramework
    ) -> None:
        """A 'hide' action should be labeled NEGATIVE."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-01")],
            "action_type": ["hide"],
            "action_timestamp": [pd.Timestamp("2025-06-01 12:00")],
            "next_visit_timestamp": [pd.NaT],
            "bucket": ["trainable"],
        })
        result = framework.assign_proxy_labels(df)
        assert result["label"].iloc[0] == 0.0

    def test_save_with_quick_return_is_positive(
        self, framework: EvaluationFramework
    ) -> None:
        """Save + return within 7 days should be POSITIVE."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-01")],
            "action_type": ["save"],
            "action_timestamp": [pd.Timestamp("2025-06-02")],
            "next_visit_timestamp": [pd.Timestamp("2025-06-05")],  # 3 days after save
            "bucket": ["ready_now"],
        })
        result = framework.assign_proxy_labels(df)
        assert result["label"].iloc[0] == 1.0

    def test_save_without_return_is_excluded(
        self, framework: EvaluationFramework
    ) -> None:
        """Save without a return visit should be EXCLUDED (NaN), not positive."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-01")],
            "action_type": ["save"],
            "action_timestamp": [pd.Timestamp("2025-06-02")],
            "next_visit_timestamp": [pd.NaT],  # Never returned
            "bucket": ["ready_now"],
        })
        result = framework.assign_proxy_labels(df)
        # Should not be labeled as positive
        assert result["label"].iloc[0] != 1.0

    def test_save_with_late_return_not_positive(
        self, framework: EvaluationFramework
    ) -> None:
        """Save + return after 7 days should NOT be POSITIVE."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-01")],
            "action_type": ["save"],
            "action_timestamp": [pd.Timestamp("2025-06-02")],
            "next_visit_timestamp": [pd.Timestamp("2025-06-20")],  # 18 days after save
            "bucket": ["ready_now"],
        })
        result = framework.assign_proxy_labels(df)
        assert result["label"].iloc[0] != 1.0

    def test_recent_no_action_is_excluded(
        self, framework: EvaluationFramework
    ) -> None:
        """No action on a very recent event should be EXCLUDED (not enough time)."""
        df = pd.DataFrame({
            "event_id": [1],
            "user_id": [1],
            "target_onet_code": ["15-1252.00"],
            "event_timestamp": [pd.Timestamp("2025-06-25")],
            "action_type": [None],
            "action_timestamp": [pd.NaT],
            "next_visit_timestamp": [pd.NaT],
            "bucket": ["trainable"],
        })
        # reference_date is only 5 days after (not enough for 14-day window)
        result = framework.assign_proxy_labels(
            df, reference_date=pd.Timestamp("2025-06-30")
        )
        assert np.isnan(result["label"].iloc[0])


# ---------------------------------------------------------------------------
# Temporal splitting tests
# ---------------------------------------------------------------------------


class TestTemporalSplitter:
    """Tests for temporal splitting."""

    def test_fractions_must_sum_to_one(self) -> None:
        """Fractions that do not sum to 1 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            TemporalSplitter(train_frac=0.5, val_frac=0.3, test_frac=0.3)

    def test_too_small_raises(self) -> None:
        """DataFrame with fewer than 10 rows should raise ValueError."""
        splitter = TemporalSplitter()
        df = pd.DataFrame({
            "event_timestamp": pd.date_range("2025-01-01", periods=5),
            "value": range(5),
        })
        with pytest.raises(ValueError, match="at least 10"):
            splitter.split(df)

    def test_missing_column_raises(self) -> None:
        """Missing timestamp column should raise ValueError."""
        splitter = TemporalSplitter(timestamp_col="ts_col")
        df = pd.DataFrame({
            "other_column": range(20),
        })
        with pytest.raises(ValueError, match="ts_col"):
            splitter.split(df)

    def test_split_sizes(self, labeled_data: pd.DataFrame) -> None:
        """Split sizes should approximately match configured fractions."""
        splitter = TemporalSplitter()
        result = splitter.split(labeled_data)
        n = len(labeled_data)

        # Allow 10% tolerance due to integer rounding
        assert abs(len(result.train) / n - 0.70) < 0.10
        assert abs(len(result.val) / n - 0.15) < 0.10
        assert abs(len(result.test) / n - 0.15) < 0.10

        # Total should be preserved
        assert len(result.train) + len(result.val) + len(result.test) == n

    def test_temporal_ordering_respected(
        self, labeled_data: pd.DataFrame
    ) -> None:
        """All train timestamps should be before all test timestamps."""
        splitter = TemporalSplitter()
        result = splitter.split(labeled_data)

        train_max = result.train["event_timestamp"].max()
        test_min = result.test["event_timestamp"].min()
        assert train_max <= test_min, (
            f"Train max ({train_max}) should be <= test min ({test_min})"
        )

    def test_split_dates_populated(
        self, labeled_data: pd.DataFrame
    ) -> None:
        """Split dates dict should contain all expected keys."""
        splitter = TemporalSplitter()
        result = splitter.split(labeled_data)

        expected_keys = {
            "train_start", "train_end", "val_start", "val_end",
            "test_start", "test_end",
        }
        assert expected_keys == set(result.split_dates.keys())


# ---------------------------------------------------------------------------
# Metrics computation tests
# ---------------------------------------------------------------------------


class TestMetricsComputation:
    """Tests for metric computation."""

    def test_perfect_model_auc(self, framework: EvaluationFramework) -> None:
        """A perfect model should achieve AUC-ROC = 1.0."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        report = framework.compute_metrics(y_true, y_score, model_version="test")
        assert report.overall_auc_roc == 1.0

    def test_random_model_auc_near_half(
        self, framework: EvaluationFramework
    ) -> None:
        """A random model should have AUC-ROC near 0.5."""
        rng = np.random.RandomState(42)
        n = 1000
        y_true = rng.binomial(1, 0.3, size=n).astype(float)
        y_score = rng.uniform(0, 1, size=n)

        report = framework.compute_metrics(y_true, y_score, model_version="random")
        assert report.overall_auc_roc is not None
        assert 0.4 < report.overall_auc_roc < 0.6

    def test_single_class_returns_none(
        self, framework: EvaluationFramework
    ) -> None:
        """AUC should be None when only one class is present."""
        y_true = np.ones(10)
        y_score = np.random.uniform(0, 1, size=10)

        report = framework.compute_metrics(y_true, y_score, model_version="one_class")
        assert report.overall_auc_roc is None
        assert report.overall_auc_pr is None

    def test_length_mismatch_raises(
        self, framework: EvaluationFramework
    ) -> None:
        """Mismatched y_true and y_score lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            framework.compute_metrics(
                y_true=np.array([0, 1]),
                y_score=np.array([0.5, 0.5, 0.5]),
                model_version="bad",
            )

    def test_calibration_bins_populated(
        self, framework: EvaluationFramework
    ) -> None:
        """Calibration bins should be computed."""
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.binomial(1, 0.3, size=n).astype(float)
        y_score = rng.uniform(0, 1, size=n)

        report = framework.compute_metrics(y_true, y_score, model_version="test")
        assert len(report.calibration_bins) > 0
        assert all(isinstance(b, CalibrationBinData) for b in report.calibration_bins)

    def test_ece_perfectly_calibrated(
        self, framework: EvaluationFramework
    ) -> None:
        """A perfectly calibrated model should have ECE close to 0."""
        # Construct a case where predicted == observed
        n = 1000
        rng = np.random.RandomState(42)
        y_score = np.linspace(0.05, 0.95, n)
        y_true = (rng.uniform(0, 1, size=n) < y_score).astype(float)

        report = framework.compute_metrics(y_true, y_score, model_version="calibrated")
        # ECE should be low (not exactly 0 due to finite sample noise)
        assert report.expected_calibration_error < 0.1

    def test_bucket_metrics_computed(
        self, framework: EvaluationFramework
    ) -> None:
        """Per-bucket metrics should be computed when buckets are provided."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.binomial(1, 0.3, size=n).astype(float)
        y_score = rng.uniform(0, 1, size=n)
        buckets = np.array(["ready_now"] * 40 + ["trainable"] * 40 + ["long_reskill"] * 20)

        report = framework.compute_metrics(
            y_true, y_score, buckets=buckets, model_version="test"
        )
        assert len(report.bucket_metrics) == 3
        bucket_names = {bm.bucket_name for bm in report.bucket_metrics}
        assert bucket_names == {"ready_now", "trainable", "long_reskill"}

    def test_mrr_without_groups(
        self, framework: EvaluationFramework
    ) -> None:
        """MRR without query groups should work."""
        y_true = np.array([0, 0, 1, 0, 0])
        y_score = np.array([0.5, 0.4, 0.9, 0.3, 0.2])

        report = framework.compute_metrics(y_true, y_score, model_version="test")
        # The highest-scored positive (0.9) is at rank 1, so MRR = 1.0
        assert report.mrr == 1.0

    def test_mrr_with_groups(
        self, framework: EvaluationFramework
    ) -> None:
        """MRR with query groups should compute per-group reciprocal ranks."""
        y_true = np.array([0, 1, 0, 0, 1, 0])
        y_score = np.array([0.8, 0.5, 0.3, 0.9, 0.7, 0.2])
        groups = np.array([1, 1, 1, 2, 2, 2])

        report = framework.compute_metrics(
            y_true, y_score, model_version="test", query_groups=groups
        )

        # Group 1: scores [0.8, 0.5, 0.3], labels [0, 1, 0] -> sorted: [0, 1, 0] -> first relevant at rank 2 -> RR = 0.5
        # Group 2: scores [0.9, 0.7, 0.2], labels [0, 1, 0] -> sorted: [0, 1, 0] -> first relevant at rank 2 -> RR = 0.5
        # MRR = (0.5 + 0.5) / 2 = 0.5
        assert report.mrr is not None
        assert abs(report.mrr - 0.5) < 1e-6

    def test_n_counts(self, framework: EvaluationFramework) -> None:
        """Report should count total, positive, and negative correctly."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

        report = framework.compute_metrics(y_true, y_score, model_version="test")
        assert report.n_total == 5
        assert report.n_positive == 3
        assert report.n_negative == 2


# ---------------------------------------------------------------------------
# Report serialization tests
# ---------------------------------------------------------------------------


class TestReportSerialization:
    """Tests for report to JSON/Markdown conversion."""

    @pytest.fixture
    def sample_report(self, framework: EvaluationFramework) -> EvaluationReport:
        """Create a sample report for serialization tests."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = rng.binomial(1, 0.3, size=n).astype(float)
        y_score = rng.uniform(0, 1, size=n)
        buckets = np.array(["ready_now"] * 50 + ["trainable"] * 50)

        return framework.compute_metrics(
            y_true, y_score, buckets=buckets, model_version="test_v1"
        )

    def test_to_dict_is_serializable(
        self, sample_report: EvaluationReport
    ) -> None:
        """report_to_dict should produce JSON-serializable output."""
        d = EvaluationFramework.report_to_dict(sample_report)
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_to_dict_has_required_keys(
        self, sample_report: EvaluationReport
    ) -> None:
        """Dict should contain all top-level keys."""
        d = EvaluationFramework.report_to_dict(sample_report)
        required_keys = {
            "model_version",
            "overall_auc_roc",
            "overall_auc_pr",
            "mrr",
            "expected_calibration_error",
            "n_total",
            "n_positive",
            "n_negative",
            "calibration_bins",
            "bucket_metrics",
            "metadata",
        }
        assert required_keys == set(d.keys())

    def test_to_markdown_is_string(
        self, sample_report: EvaluationReport
    ) -> None:
        """report_to_markdown should produce a non-empty string."""
        md = EvaluationFramework.report_to_markdown(sample_report)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_markdown_contains_headers(
        self, sample_report: EvaluationReport
    ) -> None:
        """Markdown should contain expected section headers."""
        md = EvaluationFramework.report_to_markdown(sample_report)
        assert "# Evaluation Report" in md
        assert "## Overall Metrics" in md
        assert "## Per-Bucket Metrics" in md

    def test_markdown_contains_model_version(
        self, sample_report: EvaluationReport
    ) -> None:
        """Markdown should reference the model version."""
        md = EvaluationFramework.report_to_markdown(sample_report)
        assert "test_v1" in md


# ---------------------------------------------------------------------------
# V1 scorer adapter tests
# ---------------------------------------------------------------------------


class TestV1ScorerAdapter:
    """Tests for the v1 baseline scoring adapter."""

    def test_scores_in_unit_interval(
        self, synthetic_data: pd.DataFrame
    ) -> None:
        """V1 scores should be in [0, 1]."""
        scores = score_v1_baseline(synthetic_data)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_output_length_matches_input(
        self, synthetic_data: pd.DataFrame
    ) -> None:
        """Output should have same length as input."""
        scores = score_v1_baseline(synthetic_data)
        assert len(scores) == len(synthetic_data)


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLIParsing:
    """Tests for the CLI argument parser."""

    def test_default_args(self) -> None:
        """Default arguments should be v1 model, 200 users, seed 42."""
        args = parse_args([])
        assert args.model == "v1"
        assert args.n_users == 200
        assert args.seed == 42
        assert args.output_dir == "eval_reports"

    def test_model_v2(self) -> None:
        """--model v2 should be parsed correctly."""
        args = parse_args(["--model", "v2"])
        assert args.model == "v2"

    def test_model_both(self) -> None:
        """--model both should be parsed correctly."""
        args = parse_args(["--model", "both"])
        assert args.model == "both"

    def test_custom_users(self) -> None:
        """--n-users should set user count."""
        args = parse_args(["--n-users", "500"])
        assert args.n_users == 500

    def test_custom_seed(self) -> None:
        """--seed should set random seed."""
        args = parse_args(["--seed", "123"])
        assert args.seed == 123

    def test_verbose_flag(self) -> None:
        """--verbose flag should be captured."""
        args = parse_args(["--verbose"])
        assert args.verbose is True


# ---------------------------------------------------------------------------
# Integration test: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Integration test for the entire evaluation pipeline."""

    def test_end_to_end_v1(self, tmp_path: str) -> None:
        """Full v1 evaluation should complete without errors."""
        results = run_evaluation(
            model_name="v1",
            n_users=30,
            seed=42,
            output_dir=str(tmp_path),
        )

        assert "v1_baseline" in results
        report = results["v1_baseline"]
        assert report["n_total"] > 0
        assert report["overall_auc_roc"] is not None or report["n_positive"] == 0

    def test_end_to_end_produces_files(self, tmp_path: str) -> None:
        """Evaluation should produce JSON and Markdown files."""
        import os

        run_evaluation(
            model_name="v1",
            n_users=30,
            seed=42,
            output_dir=str(tmp_path),
        )

        files = os.listdir(str(tmp_path))
        json_files = [f for f in files if f.endswith(".json")]
        md_files = [f for f in files if f.endswith(".md")]
        assert len(json_files) >= 1, "Expected at least one JSON report"
        assert len(md_files) >= 1, "Expected at least one Markdown report"

    def test_v1_beats_random(self) -> None:
        """V1 baseline should achieve AUC-ROC significantly above 0.5.

        This is a critical sanity check: if the baseline model cannot beat
        random on synthetic data that was designed to have a learnable signal,
        something is fundamentally wrong with either the generator or the
        evaluation pipeline.
        """
        config = GeneratorConfig(n_users=100, seed=42)
        gen = SyntheticDataGenerator(config)
        data = gen.generate()

        framework = EvaluationFramework()
        labeled = framework.assign_proxy_labels(data)
        labeled_only = labeled[labeled["label"].notna()].copy()

        if len(labeled_only) < 20:
            pytest.skip("Not enough labeled data for this test")

        y_true = labeled_only["label"].values
        y_score = score_v1_baseline(labeled_only)

        if len(set(y_true)) < 2:
            pytest.skip("Only one class present -- cannot compute AUC")

        report = framework.compute_metrics(
            y_true, y_score, model_version="v1_sanity"
        )

        assert report.overall_auc_roc is not None
        assert report.overall_auc_roc > 0.5, (
            f"V1 baseline AUC-ROC ({report.overall_auc_roc:.3f}) should be > 0.5"
        )

    def test_calibration_data_shape(self) -> None:
        """Calibration bins should have consistent shape."""
        config = GeneratorConfig(n_users=50, seed=42)
        gen = SyntheticDataGenerator(config)
        data = gen.generate()

        framework = EvaluationFramework(calibration_n_bins=5)
        labeled = framework.assign_proxy_labels(data)
        labeled_only = labeled[labeled["label"].notna()].copy()

        if len(labeled_only) < 10:
            pytest.skip("Not enough labeled data")

        y_true = labeled_only["label"].values
        y_score = score_v1_baseline(labeled_only)

        if len(set(y_true)) < 2:
            pytest.skip("Only one class present")

        report = framework.compute_metrics(
            y_true, y_score, model_version="cal_test"
        )

        for bin_data in report.calibration_bins:
            assert 0.0 <= bin_data.bin_lower <= 1.0
            assert 0.0 <= bin_data.bin_upper <= 1.0
            assert bin_data.bin_lower < bin_data.bin_upper
            assert bin_data.count > 0
            assert 0.0 <= bin_data.mean_predicted <= 1.0
            assert 0.0 <= bin_data.mean_observed <= 1.0
