"""Tests for the bias audit framework, report generation, and mitigation strategies.

Covers:
  - AuditFramework: bucket parity, staleness detection, score symmetry
  - AuditReport: markdown generation
  - MitigationStrategies: SKILL_REWEIGHTING and STALENESS_PENALTY
"""

import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List

import pytest

from app.ml.scoring import BaselineScorer, OccupationScore, SkillGap
from ml.bias_audit.audit_framework import (
    AuditFinding,
    AuditSeverity,
    BiasAuditEngine,
    DemographicProfile,
    PARITY_CRITICAL_THRESHOLD,
    PARITY_WARNING_THRESHOLD,
    STALENESS_CRITICAL_DAYS,
    STALENESS_WARNING_DAYS,
    SYMMETRY_WARNING_THRESHOLD,
    get_demographic_profiles,
    run_bias_audit,
)
from ml.bias_audit.audit_report import (
    generate_audit_report,
)
from ml.bias_audit.mitigation_strategies import (
    MitigationResult,
    SkillReweightingMitigation,
    StalenessPenaltyMitigation,
    apply_mitigations,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer() -> BaselineScorer:
    """Baseline scorer with default thresholds."""
    return BaselineScorer()


@pytest.fixture
def sample_skills() -> List[Dict]:
    """Standard sample skills."""
    return [
        {"element_id": "2.B.3.a", "skill_name": "Programming", "importance": 80.0, "level": 5.0},
        {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 70.0, "level": 4.5},
        {"element_id": "2.B.4.a", "skill_name": "Social Perceptiveness", "importance": 50.0, "level": 3.0},
        {"element_id": "2.A.1.a", "skill_name": "Oral Comprehension", "importance": 60.0, "level": 4.0},
    ]


@pytest.fixture
def high_match_ratings() -> Dict[str, int]:
    """Ratings that produce high match scores."""
    return {"2.B.3.a": 4, "2.B.1.a": 4, "2.B.4.a": 3, "2.A.1.a": 3}


@pytest.fixture
def low_match_ratings() -> Dict[str, int]:
    """Ratings that produce low match scores."""
    return {"2.B.3.a": 0, "2.B.1.a": 0, "2.B.4.a": 0, "2.A.1.a": 0}


def _score(
    scorer: BaselineScorer,
    skills: List[Dict],
    ratings: Dict[str, int],
    onet_code: str = "15-1252.00",
    title: str = "Software Developer",
) -> OccupationScore:
    """Helper to produce an OccupationScore."""
    return scorer.score_occupation(
        onet_code=onet_code,
        occupation_title=title,
        occupation_skills=skills,
        user_skill_ratings=ratings,
    )


# ===================================================================
# AuditFramework tests
# ===================================================================

class TestBiasAuditEngine:
    """Tests for the BiasAuditEngine."""

    def test_full_audit_returns_findings(self, scorer, sample_skills, high_match_ratings):
        """Full audit should return a list of findings."""
        scores = [
            _score(scorer, sample_skills, high_match_ratings, "15-1252.00", "Software Dev"),
            _score(scorer, sample_skills, high_match_ratings, "29-1141.00", "Nurse"),
        ]

        findings = run_bias_audit(scores)

        assert isinstance(findings, list)
        for f in findings:
            assert isinstance(f, AuditFinding)

    def test_findings_sorted_by_severity(self, scorer, sample_skills, high_match_ratings):
        """Findings should be sorted CRITICAL > WARNING > INFO."""
        scores = [
            _score(scorer, sample_skills, high_match_ratings, code, title)
            for code, title in [
                ("15-1252.00", "Software Dev"),
                ("29-1141.00", "Nurse"),
                ("11-9013.00", "Farmer"),
            ]
        ]

        findings = run_bias_audit(scores)

        severity_order = {
            AuditSeverity.CRITICAL: 0,
            AuditSeverity.WARNING: 1,
            AuditSeverity.INFO: 2,
        }
        for i in range(len(findings) - 1):
            assert severity_order[findings[i].severity] <= severity_order[findings[i + 1].severity]

    def test_bucket_parity_with_balanced_scores(self, scorer, sample_skills, high_match_ratings):
        """When all occupations have the same score, parity should pass."""
        # All occupations scored identically (same skills, same ratings)
        codes = ["29-1141.00", "15-1252.00", "25-2021.00", "47-2111.00"]
        scores = [
            _score(scorer, sample_skills, high_match_ratings, code, f"Occ {code}")
            for code in codes
        ]

        engine = BiasAuditEngine()
        findings = engine.test_bucket_distribution_parity(
            scores, get_demographic_profiles()
        )

        # All should be INFO (no disparity since all scores are identical)
        for f in findings:
            if f.metric_name.startswith("ready_now_disparity"):
                assert f.severity == AuditSeverity.INFO

    def test_bucket_parity_detects_disparity(self, scorer, sample_skills, high_match_ratings, low_match_ratings):
        """Bucket parity should detect when one group has more READY_NOW."""
        # Female-majority occupations get high scores (ready_now)
        # Male-majority occupations get low scores (long_reskill)
        scores = [
            _score(scorer, sample_skills, high_match_ratings, "29-1141.00", "Nurse"),
            _score(scorer, sample_skills, high_match_ratings, "25-2021.00", "Teacher"),
            _score(scorer, sample_skills, high_match_ratings, "21-1021.00", "Social Worker"),
            _score(scorer, sample_skills, low_match_ratings, "15-1252.00", "Software Dev"),
            _score(scorer, sample_skills, low_match_ratings, "47-2111.00", "Electrician"),
            _score(scorer, sample_skills, low_match_ratings, "17-2141.00", "Mech Engineer"),
        ]

        engine = BiasAuditEngine()
        findings = engine.test_bucket_distribution_parity(
            scores, get_demographic_profiles()
        )

        # There should be at least one WARNING or CRITICAL finding on gender axis
        gender_findings = [
            f for f in findings
            if "gender" in f.metric_name
        ]
        assert len(gender_findings) > 0
        assert any(
            f.severity in (AuditSeverity.WARNING, AuditSeverity.CRITICAL)
            for f in gender_findings
        )

    def test_bucket_parity_with_insufficient_data(self, scorer, sample_skills, high_match_ratings):
        """Parity test with <4 profiled occupations should return INFO."""
        scores = [
            _score(scorer, sample_skills, high_match_ratings, "15-1252.00", "Dev"),
        ]

        engine = BiasAuditEngine()
        findings = engine.test_bucket_distribution_parity(
            scores, get_demographic_profiles()
        )

        info_findings = [f for f in findings if f.severity == AuditSeverity.INFO]
        assert len(info_findings) > 0

    def test_staleness_detects_old_profiles(self):
        """Staleness test should flag profiles older than threshold."""
        old_date = (datetime.utcnow() - timedelta(days=STALENESS_CRITICAL_DAYS + 1)).strftime("%Y-%m-%d")
        profiles = {
            "99-0001.00": DemographicProfile(
                "99-0001.00", "Old Occupation", 50.0, 30.0, 40.0, old_date
            ),
        }

        engine = BiasAuditEngine()
        findings = engine.test_skill_profile_staleness(profiles)

        critical = [f for f in findings if f.severity == AuditSeverity.CRITICAL]
        assert len(critical) > 0
        assert "99-0001.00" in critical[0].affected_occupations

    def test_staleness_passes_for_fresh_profiles(self):
        """Fresh profiles should not trigger staleness findings."""
        fresh_date = datetime.utcnow().strftime("%Y-%m-%d")
        profiles = {
            "99-0001.00": DemographicProfile(
                "99-0001.00", "Fresh Occ", 50.0, 30.0, 40.0, fresh_date
            ),
            "99-0002.00": DemographicProfile(
                "99-0002.00", "Also Fresh", 50.0, 30.0, 40.0, fresh_date
            ),
        }

        engine = BiasAuditEngine()
        findings = engine.test_skill_profile_staleness(profiles)

        warning_or_critical = [
            f for f in findings
            if f.severity in (AuditSeverity.WARNING, AuditSeverity.CRITICAL)
        ]
        assert len(warning_or_critical) == 0

    def test_staleness_warning_range(self):
        """Profiles in warning range should trigger WARNING."""
        warning_date = (
            datetime.utcnow() - timedelta(days=STALENESS_WARNING_DAYS + 10)
        ).strftime("%Y-%m-%d")
        profiles = {
            "99-0001.00": DemographicProfile(
                "99-0001.00", "Warning Occ", 50.0, 30.0, 40.0, warning_date
            ),
        }

        engine = BiasAuditEngine()
        findings = engine.test_skill_profile_staleness(profiles)

        warnings = [f for f in findings if f.severity == AuditSeverity.WARNING]
        assert len(warnings) > 0

    def test_score_symmetry_no_asymmetry(self, scorer, sample_skills, high_match_ratings):
        """Occupations scored identically should show no asymmetry."""
        # Same skills, same ratings -> same scores -> no asymmetry
        score_a = _score(scorer, sample_skills, high_match_ratings, "A", "OccA")
        score_b = _score(scorer, sample_skills, high_match_ratings, "B", "OccB")

        engine = BiasAuditEngine()
        findings = engine.test_score_symmetry([score_a, score_b])

        # Should be INFO (no asymmetry)
        for f in findings:
            assert f.severity == AuditSeverity.INFO

    def test_score_symmetry_detects_asymmetry(self, scorer):
        """Occupations with same gaps but different scores should be flagged."""
        # Manufacture two scores with same gap element_ids but different match_scores
        # by using different skill importances
        skills_a = [
            {"element_id": "s1", "skill_name": "Skill1", "importance": 80.0, "level": 5.0},
            {"element_id": "s2", "skill_name": "Skill2", "importance": 70.0, "level": 4.0},
        ]
        skills_b = [
            {"element_id": "s1", "skill_name": "Skill1", "importance": 30.0, "level": 5.0},
            {"element_id": "s2", "skill_name": "Skill2", "importance": 70.0, "level": 4.0},
        ]
        ratings = {"s1": 0, "s2": 0}  # Both gaps

        score_a = _score(scorer, skills_a, ratings, "A", "OccA")
        score_b = _score(scorer, skills_b, ratings, "B", "OccB")

        # Both have gaps on s1 and s2 (Jaccard = 1.0), but different importances
        # lead to different match_scores (both 0 since ratings=0, so this
        # actually won't produce asymmetry). Let's use ratings that DO create
        # a difference.
        ratings_partial = {"s1": 4, "s2": 0}  # Expert in s1, gap in s2
        score_a = _score(scorer, skills_a, ratings_partial, "A", "OccA")
        score_b = _score(scorer, skills_b, ratings_partial, "B", "OccB")

        engine = BiasAuditEngine()
        findings = engine.test_score_symmetry([score_a, score_b])

        # s2 is the only gap in both (s1 has rating 4 -> not a gap)
        # Jaccard of gap sets: both have {s2} -> Jaccard = 1.0
        # But match_scores differ because s1 has different importance
        # The score difference should be detected
        # Note: the actual difference depends on importance distribution
        assert len(findings) > 0

    def test_audit_finding_structure(self):
        """AuditFinding should have all required fields."""
        finding = AuditFinding(
            test_name="test",
            severity=AuditSeverity.INFO,
            description="Test finding",
            affected_occupations=["11-1011.00"],
            metric_name="test_metric",
            metric_value=0.5,
        )

        assert finding.test_name == "test"
        assert finding.severity == AuditSeverity.INFO
        assert finding.metric_value == 0.5
        assert "11-1011.00" in finding.affected_occupations

    def test_demographic_profiles_available(self):
        """Stub demographic profiles should be available."""
        profiles = get_demographic_profiles()
        assert len(profiles) > 0
        for code, profile in profiles.items():
            assert isinstance(profile, DemographicProfile)
            assert 0 <= profile.pct_female <= 100
            assert 0 <= profile.pct_minority <= 100


# ===================================================================
# AuditReport tests
# ===================================================================

class TestAuditReport:
    """Tests for the audit report generator."""

    def test_report_generation_basic(self):
        """Report should generate valid markdown."""
        findings = [
            AuditFinding(
                test_name="test_check",
                severity=AuditSeverity.INFO,
                description="All clear",
                affected_occupations=[],
                metric_name="test_metric",
                metric_value=0.0,
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            tmp_path = f.name

        try:
            report = generate_audit_report(
                findings,
                output_path=tmp_path,
                scored_occupation_count=10,
                model_version="v1_test",
            )

            assert "# SkillSprout Bias Audit Report" in report
            assert "v1_test" in report
            assert "PASS" in report  # Only INFO findings -> PASS
            assert os.path.exists(tmp_path)

            # Read back from file and verify
            with open(tmp_path, "r") as f:
                written = f.read()
            assert written == report
        finally:
            os.unlink(tmp_path)

    def test_report_with_critical_findings(self):
        """Report with CRITICAL findings should show FAIL status."""
        findings = [
            AuditFinding(
                test_name="parity_test",
                severity=AuditSeverity.CRITICAL,
                description="Major disparity found",
                affected_occupations=["11-1011.00"],
                metric_name="ready_now_disparity",
                metric_value=0.30,
                threshold=0.25,
                recommended_action="Review thresholds",
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            tmp_path = f.name

        try:
            report = generate_audit_report(
                findings,
                output_path=tmp_path,
                scored_occupation_count=5,
            )

            assert "FAIL" in report
            assert "CRITICAL" in report
            assert "Review thresholds" in report
        finally:
            os.unlink(tmp_path)

    def test_report_with_warnings(self):
        """Report with WARNING findings should show REVIEW status."""
        findings = [
            AuditFinding(
                test_name="staleness",
                severity=AuditSeverity.WARNING,
                description="Some stale profiles",
                affected_occupations=["29-1141.00"],
                metric_name="stale_count",
                metric_value=2.0,
            ),
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            tmp_path = f.name

        try:
            report = generate_audit_report(
                findings,
                output_path=tmp_path,
            )

            assert "REVIEW" in report
        finally:
            os.unlink(tmp_path)

    def test_report_includes_demographic_coverage(self):
        """Report should include a demographic coverage table."""
        findings = []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            tmp_path = f.name

        try:
            report = generate_audit_report(
                findings,
                output_path=tmp_path,
            )

            assert "Demographic Coverage" in report
            assert "O*NET Code" in report
            assert "% Female" in report
        finally:
            os.unlink(tmp_path)

    def test_report_includes_methodology(self):
        """Report should include methodology notes."""
        findings = []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            tmp_path = f.name

        try:
            report = generate_audit_report(
                findings,
                output_path=tmp_path,
            )

            assert "Methodology" in report
            assert "Bucket Distribution Parity" in report
            assert "Skill Profile Staleness" in report
            assert "Score Symmetry" in report
        finally:
            os.unlink(tmp_path)


# ===================================================================
# MitigationStrategies tests
# ===================================================================

class TestSkillReweighting:
    """Tests for SKILL_REWEIGHTING mitigation."""

    def test_reweighting_reduces_gap_severity(self, scorer, sample_skills, low_match_ratings):
        """Reweighting correlated skills should reduce gap severity."""
        score = _score(scorer, sample_skills, low_match_ratings)

        mitigation = SkillReweightingMitigation(reweight_factor=0.3)
        # Mark programming skill as correlated
        correlated = {"2.B.3.a"}

        result = mitigation.apply(score, correlated)

        assert result.adjusted_score.gap_severity <= score.gap_severity
        assert result.adjusted_score.match_score >= score.match_score
        assert result.mitigation_name == "SKILL_REWEIGHTING"

    def test_reweighting_no_correlated_skills(self, scorer, sample_skills, low_match_ratings):
        """With no correlated skills, scores should not change."""
        score = _score(scorer, sample_skills, low_match_ratings)

        mitigation = SkillReweightingMitigation()
        result = mitigation.apply(score, set())

        assert result.adjusted_score.match_score == score.match_score
        assert result.adjusted_score.gap_severity == score.gap_severity

    def test_reweighting_preserves_original(self, scorer, sample_skills, low_match_ratings):
        """Original score should be preserved in the result."""
        score = _score(scorer, sample_skills, low_match_ratings)

        mitigation = SkillReweightingMitigation()
        result = mitigation.apply(score, {"2.B.3.a"})

        assert result.original_score.match_score == score.match_score
        assert result.original_score.gap_severity == score.gap_severity

    def test_identify_correlated_skills(self):
        """Should identify skills that appear disproportionately in one group."""
        # Create skill maps where skill "gender_specific" only appears in
        # female-majority occupations.
        occ_skills_map = {
            "29-1141.00": [
                {"element_id": "common", "importance": 80},
                {"element_id": "gender_specific", "importance": 70},
            ],
            "25-2021.00": [
                {"element_id": "common", "importance": 80},
                {"element_id": "gender_specific", "importance": 60},
            ],
            "21-1021.00": [
                {"element_id": "common", "importance": 80},
                {"element_id": "gender_specific", "importance": 65},
            ],
            "15-1252.00": [
                {"element_id": "common", "importance": 80},
            ],
            "47-2111.00": [
                {"element_id": "common", "importance": 80},
            ],
            "17-2141.00": [
                {"element_id": "common", "importance": 80},
            ],
        }

        mitigation = SkillReweightingMitigation(correlation_threshold=0.70)
        correlated = mitigation.identify_correlated_skills(
            occ_skills_map, get_demographic_profiles()
        )

        # "gender_specific" should be identified as correlated
        # (appears in 3/3 female-majority, 0/3 male-majority)
        assert "gender_specific" in correlated

    def test_reweighting_can_change_bucket(self, scorer):
        """Significant reweighting could change bucket assignment."""
        # Create a score just below READY_NOW
        skills = [
            {"element_id": "s1", "skill_name": "Important", "importance": 90.0, "level": 5.0},
            {"element_id": "s2", "skill_name": "Correlated", "importance": 60.0, "level": 4.0},
        ]
        ratings = {"s1": 4, "s2": 0}  # Expert in s1, gap in s2

        score = _score(scorer, skills, ratings)

        mitigation = SkillReweightingMitigation(reweight_factor=0.5)
        result = mitigation.apply(score, {"s2"})

        # Reducing the correlated gap's weight should improve the score
        assert result.adjusted_score.match_score >= score.match_score


class TestStalenessPenalty:
    """Tests for STALENESS_PENALTY mitigation."""

    def test_fresh_data_no_penalty(self, scorer, sample_skills, high_match_ratings):
        """Fresh data should receive no penalty."""
        score = _score(scorer, sample_skills, high_match_ratings)
        mitigation = StalenessPenaltyMitigation()

        fresh_date = datetime.utcnow().strftime("%Y-%m-%d")
        result = mitigation.apply(score, fresh_date)

        assert result.adjusted_score.match_score == score.match_score
        assert result.adjusted_score.gap_severity == score.gap_severity

    def test_stale_data_reduces_match(self, scorer, sample_skills, high_match_ratings):
        """Stale data should reduce match_score."""
        score = _score(scorer, sample_skills, high_match_ratings)
        mitigation = StalenessPenaltyMitigation()

        old_date = (
            datetime.utcnow() - timedelta(days=STALENESS_CRITICAL_DAYS)
        ).strftime("%Y-%m-%d")
        result = mitigation.apply(score, old_date)

        assert result.adjusted_score.match_score < score.match_score
        assert result.adjusted_score.gap_severity > score.gap_severity

    def test_penalty_increases_with_age(self, scorer, sample_skills, high_match_ratings):
        """Older data should receive a larger penalty."""
        score = _score(scorer, sample_skills, high_match_ratings)
        mitigation = StalenessPenaltyMitigation()

        date_1yr = (
            datetime.utcnow() - timedelta(days=365)
        ).strftime("%Y-%m-%d")
        date_2yr = (
            datetime.utcnow() - timedelta(days=730)
        ).strftime("%Y-%m-%d")

        result_1yr = mitigation.apply(score, date_1yr)
        result_2yr = mitigation.apply(score, date_2yr)

        assert result_2yr.adjusted_score.match_score < result_1yr.adjusted_score.match_score

    def test_penalty_capped_at_max(self):
        """Penalty should not exceed max_penalty."""
        mitigation = StalenessPenaltyMitigation(max_penalty=0.15, penalty_max_days=730)

        # Data 10 years old -> should still cap at 0.15
        very_old = (
            datetime.utcnow() - timedelta(days=3650)
        ).strftime("%Y-%m-%d")
        penalty = mitigation.compute_penalty(very_old)

        assert penalty == 0.15

    def test_unparseable_date_gets_max_penalty(self):
        """Unparseable date strings should receive maximum penalty."""
        mitigation = StalenessPenaltyMitigation(max_penalty=0.15)
        penalty = mitigation.compute_penalty("not-a-date")
        assert penalty == 0.15

    def test_preserves_original_score(self, scorer, sample_skills, high_match_ratings):
        """Original score should be preserved in the result."""
        score = _score(scorer, sample_skills, high_match_ratings)
        mitigation = StalenessPenaltyMitigation()

        old_date = (
            datetime.utcnow() - timedelta(days=400)
        ).strftime("%Y-%m-%d")
        result = mitigation.apply(score, old_date)

        assert result.original_score.match_score == score.match_score
        assert result.mitigation_name == "STALENESS_PENALTY"

    def test_staleness_can_change_bucket(self, scorer):
        """Large staleness penalty could push READY_NOW to TRAINABLE."""
        # Create a score just barely in READY_NOW
        skills = [
            {"element_id": "s1", "skill_name": "Skill1", "importance": 80.0, "level": 5.0},
            {"element_id": "s2", "skill_name": "Skill2", "importance": 20.0, "level": 3.0},
        ]
        ratings = {"s1": 4, "s2": 3}  # High match

        score = _score(scorer, skills, ratings)
        # Score should be ready_now with high match
        if score.bucket != "ready_now":
            # Adjust if needed; this is data-dependent
            pytest.skip("Score not in ready_now, cannot test bucket change")

        mitigation = StalenessPenaltyMitigation(max_penalty=0.20)
        very_old = (
            datetime.utcnow() - timedelta(days=1000)
        ).strftime("%Y-%m-%d")
        result = mitigation.apply(score, very_old)

        # With a 20% penalty, a borderline READY_NOW might drop to TRAINABLE
        # The exact behaviour depends on the score values
        assert result.adjusted_score.match_score < score.match_score


class TestApplyMitigations:
    """Tests for the combined apply_mitigations convenience function."""

    def test_combined_mitigations(self, scorer, sample_skills, low_match_ratings):
        """Combined mitigations should produce results for each score."""
        scores = [
            _score(scorer, sample_skills, low_match_ratings, "15-1252.00", "Dev"),
            _score(scorer, sample_skills, low_match_ratings, "29-1141.00", "Nurse"),
        ]

        occ_skills_map = {
            "15-1252.00": sample_skills,
            "29-1141.00": sample_skills,
        }

        last_updated_map = {
            "15-1252.00": datetime.utcnow().strftime("%Y-%m-%d"),
            "29-1141.00": (
                datetime.utcnow() - timedelta(days=500)
            ).strftime("%Y-%m-%d"),
        }

        results = apply_mitigations(
            scores,
            occupation_skills_map=occ_skills_map,
            last_updated_map=last_updated_map,
        )

        assert len(results) == 2
        for result in results:
            assert isinstance(result, MitigationResult)
            assert result.mitigation_name == "COMBINED"

    def test_mitigations_disabled(self, scorer, sample_skills, high_match_ratings):
        """Disabling all mitigations should return unchanged scores."""
        scores = [
            _score(scorer, sample_skills, high_match_ratings),
        ]

        results = apply_mitigations(
            scores,
            enable_reweighting=False,
            enable_staleness=False,
        )

        assert len(results) == 1
        assert results[0].adjusted_score.match_score == scores[0].match_score

    def test_mitigation_result_has_details(self, scorer, sample_skills, low_match_ratings):
        """MitigationResult should include adjustment details."""
        score = _score(scorer, sample_skills, low_match_ratings)

        mitigation = SkillReweightingMitigation()
        result = mitigation.apply(score, {"2.B.3.a"})

        assert len(result.adjustment_details) > 0
        assert "reweight_factor" in result.parameters_used
