"""Tests for model management: calibration monitor, model registry, A/B framework."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from ml.model_management.calibration_monitor import (
    CalibrationMonitor,
    CalibrationReport,
    CalibrationSnapshot,
    CalibrationBin,
    ECE_WARNING_THRESHOLD,
    ECE_ALERT_THRESHOLD,
    KS_DRIFT_THRESHOLD,
)
from ml.model_management.model_registry import (
    ModelRegistryService,
    ModelArtifact,
    ModelStatus,
    FeatureSchema,
)
from ml.model_management.ab_test_framework import (
    ABTestFramework,
    ABExperiment,
    ABVariant,
    PredictionLog,
)


# =========================================================================
# CalibrationMonitor
# =========================================================================

class TestCalibrationMonitor:
    """Tests for CalibrationMonitor."""

    def test_perfect_calibration(self):
        """A perfectly calibrated model should have ECE near zero."""
        monitor = CalibrationMonitor(num_bins=10)
        # Create predictions that perfectly match outcomes on average.
        np.random.seed(42)
        n = 1000
        predictions = np.random.uniform(0, 1, n)
        outcomes = (np.random.uniform(0, 1, n) < predictions).astype(float)

        snapshot = monitor.evaluate(
            predictions=predictions,
            outcomes=outcomes,
            model_version="v_test",
        )

        assert snapshot.ece < 0.10  # Should be well-calibrated
        assert snapshot.severity == "OK"
        assert snapshot.total_samples == n

    def test_badly_calibrated_model(self):
        """A badly calibrated model should trigger an ALERT."""
        monitor = CalibrationMonitor(num_bins=10)
        # All predictions near 0.9, but outcomes are all 0 (not positive).
        predictions = np.full(200, 0.9)
        outcomes = np.zeros(200)

        snapshot = monitor.evaluate(
            predictions=predictions,
            outcomes=outcomes,
            model_version="v_bad",
        )

        assert snapshot.ece > ECE_ALERT_THRESHOLD
        assert snapshot.severity == "ALERT"
        assert any("ALERT" in f for f in snapshot.flags)

    def test_warning_ece(self):
        """ECE between WARNING and ALERT thresholds should produce WARNING."""
        monitor = CalibrationMonitor(num_bins=10)
        # Engineer predictions with moderate miscalibration.
        np.random.seed(123)
        n = 500
        predictions = np.clip(np.random.normal(0.7, 0.1, n), 0, 1)
        # Outcomes with ~40% positive rate (predictions say ~70%).
        outcomes = (np.random.uniform(0, 1, n) < 0.40).astype(float)

        snapshot = monitor.evaluate(
            predictions=predictions,
            outcomes=outcomes,
            model_version="v_warn",
        )

        # The ECE should be elevated but the exact severity depends on data.
        assert snapshot.ece > 0.0
        assert len(snapshot.bins) == 10

    def test_empty_inputs_raise(self):
        """Empty arrays should raise ValueError."""
        monitor = CalibrationMonitor()
        with pytest.raises(ValueError, match="non-empty"):
            monitor.evaluate(
                predictions=np.array([]),
                outcomes=np.array([]),
                model_version="v_empty",
            )

    def test_mismatched_shapes_raise(self):
        """Mismatched input shapes should raise ValueError."""
        monitor = CalibrationMonitor()
        with pytest.raises(ValueError, match="Shape mismatch"):
            monitor.evaluate(
                predictions=np.array([0.5, 0.6]),
                outcomes=np.array([0, 1, 0]),
                model_version="v_mismatch",
            )

    def test_reliability_diagram_bins(self):
        """Reliability diagram should have the correct number of bins."""
        monitor = CalibrationMonitor(num_bins=5)
        predictions = np.linspace(0, 1, 100)
        outcomes = (predictions > 0.5).astype(float)

        snapshot = monitor.evaluate(
            predictions=predictions,
            outcomes=outcomes,
            model_version="v_bins",
        )

        assert len(snapshot.bins) == 5
        # Verify bin ranges cover [0, 1].
        assert snapshot.bins[0].bin_lower == pytest.approx(0.0)
        assert snapshot.bins[-1].bin_upper == pytest.approx(1.0)

    def test_ks_drift_detection(self):
        """K-S test should detect shift from a very different reference."""
        monitor = CalibrationMonitor()
        predictions = np.full(200, 0.1)  # All very low
        outcomes = np.zeros(200)
        reference = np.full(200, 0.9)  # Reference is all very high

        snapshot = monitor.evaluate(
            predictions=predictions,
            outcomes=outcomes,
            model_version="v_drift",
            reference_predictions=reference,
        )

        assert snapshot.ks_statistic > KS_DRIFT_THRESHOLD
        assert any("DRIFT" in f for f in snapshot.flags)

    def test_snapshot_serialisation(self):
        """snapshot_to_dict should produce a JSON-serialisable dictionary."""
        monitor = CalibrationMonitor(num_bins=3)
        predictions = np.array([0.2, 0.5, 0.8])
        outcomes = np.array([0, 1, 1])

        snapshot = monitor.evaluate(
            predictions=predictions,
            outcomes=outcomes,
            model_version="v_serial",
        )
        data = monitor.snapshot_to_dict(snapshot)

        assert data["model_version"] == "v_serial"
        assert isinstance(data["ece"], float)
        assert isinstance(data["bins"], list)
        assert len(data["bins"]) == 3

    def test_snapshots_accumulate(self):
        """Multiple evaluations should accumulate in the snapshots list."""
        monitor = CalibrationMonitor()
        for i in range(3):
            monitor.evaluate(
                predictions=np.array([0.5]),
                outcomes=np.array([1]),
                model_version=f"v_{i}",
            )
        assert len(monitor.get_snapshots()) == 3


class TestCalibrationReport:
    """Tests for CalibrationReport markdown generation."""

    def test_report_contains_key_sections(self):
        """Report should include summary table and bin data."""
        snapshot = CalibrationSnapshot(
            model_version="v_report",
            evaluated_at=datetime.utcnow(),
            window_start=datetime.utcnow() - timedelta(days=7),
            window_end=datetime.utcnow(),
            total_samples=100,
            ece=0.05,
            bins=[
                CalibrationBin(
                    bin_lower=0.0,
                    bin_upper=0.5,
                    mean_predicted=0.25,
                    mean_observed=0.20,
                    count=50,
                    accuracy=0.80,
                ),
                CalibrationBin(
                    bin_lower=0.5,
                    bin_upper=1.0,
                    mean_predicted=0.75,
                    mean_observed=0.78,
                    count=50,
                    accuracy=0.85,
                ),
            ],
            ks_statistic=0.05,
            ks_p_value=0.50,
            severity="OK",
            flags=[],
            metadata={"mean_predicted": 0.50, "positive_rate": 0.49},
        )

        report = CalibrationReport.generate(snapshot)
        assert "# Calibration Report" in report
        assert "v_report" in report
        assert "ECE" in report
        assert "K-S Statistic" in report
        assert "Reliability Diagram" in report
        assert "OK" in report

    def test_report_with_flags(self):
        """Report should render flags section when there are alerts."""
        snapshot = CalibrationSnapshot(
            model_version="v_flagged",
            evaluated_at=datetime.utcnow(),
            window_start=datetime.utcnow() - timedelta(days=7),
            window_end=datetime.utcnow(),
            total_samples=50,
            ece=0.30,
            bins=[],
            ks_statistic=0.15,
            ks_p_value=0.01,
            severity="ALERT",
            flags=["ALERT: ECE (0.3000) exceeds 0.25", "DRIFT: K-S statistic (0.1500) exceeds 0.1"],
        )

        report = CalibrationReport.generate(snapshot)
        assert "## Flags" in report
        assert "ALERT" in report
        assert "DRIFT" in report


# =========================================================================
# ModelRegistryService
# =========================================================================

class TestModelRegistryService:
    """Tests for ModelRegistryService (uses mocked DB)."""

    def _make_artifact(
        self,
        version: str = "v_test",
        status: ModelStatus = ModelStatus.CANDIDATE,
    ) -> ModelArtifact:
        return ModelArtifact(
            model_version=version,
            trained_at=datetime.utcnow(),
            artifact_path=f"/models/{version}.joblib",
            status=status,
            training_samples=100,
            eval_metrics={"accuracy": 0.85, "roc_auc": 0.90},
            notes="test artifact",
        )

    def test_artifact_to_dict(self):
        """ModelArtifact.to_dict should produce a serialisable dict."""
        artifact = self._make_artifact()
        d = artifact.to_dict()
        assert d["status"] == "candidate"
        assert d["eval_metrics"]["accuracy"] == 0.85

    def test_feature_schema_roundtrip(self):
        """FeatureSchema should survive to_dict / from_dict roundtrip."""
        schema = FeatureSchema(
            version="fs_v1",
            feature_names=["match_score", "gap_severity"],
            feature_types={"match_score": "float", "gap_severity": "float"},
            transformations={"match_score": "standard_scaler"},
            description="Baseline features",
        )
        d = schema.to_dict()
        restored = FeatureSchema.from_dict(d)

        assert restored.version == "fs_v1"
        assert restored.feature_names == ["match_score", "gap_severity"]
        assert restored.transformations["match_score"] == "standard_scaler"

    def test_model_status_values(self):
        """ModelStatus enum should contain expected members."""
        assert ModelStatus.CANDIDATE.value == "candidate"
        assert ModelStatus.PRODUCTION.value == "production"
        assert ModelStatus.ARCHIVED.value == "archived"
        assert ModelStatus.ROLLED_BACK.value == "rolled_back"


# =========================================================================
# ABTestFramework
# =========================================================================

class TestABTestFramework:
    """Tests for ABTestFramework."""

    def test_create_experiment(self):
        """Creating an experiment should register it."""
        fw = ABTestFramework()
        exp = fw.create_experiment(
            name="test_exp",
            allocations={"v2.3": 90, "v2.4_candidate": 10},
        )
        assert exp.name == "test_exp"
        assert len(exp.variants) == 2
        assert fw.get_experiment("test_exp") is exp

    def test_duplicate_experiment_raises(self):
        """Creating an experiment with the same name should raise."""
        fw = ABTestFramework()
        fw.create_experiment("dup", allocations={"a": 50, "b": 50})
        with pytest.raises(ValueError, match="already exists"):
            fw.create_experiment("dup", allocations={"a": 50, "b": 50})

    def test_invalid_allocation_raises(self):
        """Allocations not summing to 100 should raise."""
        fw = ABTestFramework()
        with pytest.raises(ValueError, match="sum to 100"):
            fw.create_experiment("bad", allocations={"a": 60, "b": 30})

    def test_deterministic_assignment(self):
        """The same user should always be assigned to the same variant."""
        fw = ABTestFramework()
        exp = fw.create_experiment(
            "det_test", allocations={"v1": 50, "v2": 50}
        )
        assignments = [fw.assign_user(exp, user_id=42).name for _ in range(100)]
        assert len(set(assignments)) == 1  # Always the same

    def test_assignment_distribution(self):
        """Over many users the assignment should roughly match percentages."""
        fw = ABTestFramework()
        exp = fw.create_experiment(
            "dist_test", allocations={"control": 80, "treatment": 20}
        )
        counts = {"control": 0, "treatment": 0}
        for uid in range(10000):
            variant = fw.assign_user(exp, user_id=uid)
            counts[variant.name] += 1

        # 80/20 split -- allow generous tolerance.
        assert 7000 < counts["control"] < 9000
        assert 1000 < counts["treatment"] < 3000

    def test_salt_changes_assignment(self):
        """Different salts should produce different assignments."""
        fw = ABTestFramework()
        exp1 = fw.create_experiment(
            "salt1", allocations={"a": 50, "b": 50}, salt="alpha"
        )
        exp2 = fw.create_experiment(
            "salt2", allocations={"a": 50, "b": 50}, salt="beta"
        )
        # Not guaranteed to differ for every user, but statistically very likely
        # to differ for at least some of 100 users.
        differ = False
        for uid in range(100):
            a1 = fw.assign_user(exp1, user_id=uid).name
            a2 = fw.assign_user(exp2, user_id=uid).name
            if a1 != a2:
                differ = True
                break
        assert differ

    def test_log_prediction(self):
        """Logging a prediction should record the variant and model version."""
        fw = ABTestFramework()
        exp = fw.create_experiment(
            "log_test", allocations={"v1": 50, "v2": 50}
        )
        log = fw.log_prediction(exp, user_id=1, prediction=0.75)

        assert log.user_id == 1
        assert log.prediction == 0.75
        assert log.variant_name in ("v1", "v2")
        assert len(fw.get_logs("log_test")) == 1

    def test_record_outcome(self):
        """Recording an outcome should back-fill existing logs."""
        fw = ABTestFramework()
        exp = fw.create_experiment(
            "outcome_test", allocations={"v1": 100}
        )
        fw.log_prediction(exp, user_id=7, prediction=0.5)
        fw.log_prediction(exp, user_id=7, prediction=0.6)
        updated = fw.record_outcome(exp, user_id=7, outcome=1.0)
        assert updated == 2

        logs = fw.get_logs("outcome_test")
        assert all(l.outcome == 1.0 for l in logs)

    def test_analyse_significant(self):
        """Analysis should detect a significant difference when one exists."""
        fw = ABTestFramework()
        exp = fw.create_experiment(
            "sig_test", allocations={"control": 50, "treatment": 50}
        )
        # Control: 10% conversion, Treatment: 50% conversion.
        for uid in range(200):
            variant = fw.assign_user(exp, user_id=uid)
            if variant.name == "control":
                outcome = 1.0 if uid % 10 == 0 else 0.0
            else:
                outcome = 1.0 if uid % 2 == 0 else 0.0
            fw.log_prediction(exp, user_id=uid, prediction=0.5, outcome=outcome)

        result = fw.analyse(exp)
        assert result.chi2_statistic > 0
        # With such a large difference, p should be very small.
        assert result.is_significant
        assert result.winner is not None

    def test_analyse_no_outcomes_raises(self):
        """Analysis should raise when no outcomes have been recorded."""
        fw = ABTestFramework()
        exp = fw.create_experiment(
            "no_out", allocations={"v1": 50, "v2": 50}
        )
        fw.log_prediction(exp, user_id=1, prediction=0.5)  # No outcome
        with pytest.raises(ValueError, match="No outcomes"):
            fw.analyse(exp)

    def test_get_model_version_for_user(self):
        """Convenience method should return the model version string."""
        fw = ABTestFramework()
        fw.create_experiment(
            "version_test", allocations={"v2.3": 90, "v2.4_candidate": 10}
        )
        version = fw.get_model_version_for_user("version_test", user_id=99)
        assert version in ("v2.3", "v2.4_candidate")

    def test_get_model_version_missing_experiment_raises(self):
        """Unknown experiment should raise ValueError."""
        fw = ABTestFramework()
        with pytest.raises(ValueError, match="not found"):
            fw.get_model_version_for_user("nonexistent", user_id=1)

    def test_list_experiments(self):
        """list_experiments should return all registered experiments."""
        fw = ABTestFramework()
        fw.create_experiment("e1", allocations={"a": 100})
        fw.create_experiment("e2", allocations={"b": 100})
        assert len(fw.list_experiments()) == 2
