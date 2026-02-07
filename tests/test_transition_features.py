"""Tests for transition-aware feature engineering.

Each feature family is tested with at least three distinct origin->target
occupation pairs covering normal, edge, and missing-data scenarios.
"""

import math
import pytest
import numpy as np

from ml.features.transition_features import (
    skill_direction_vector,
    experience_transfer_ratio,
    occupation_demand_signal,
    salary_delta,
    credential_barrier,
    industry_distance,
    build_transition_features,
    TransitionFeatureVector,
    augment_calibration_array,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def software_dev_skills():
    """Skills for Software Developers (15-1252.00)."""
    return [
        {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 84.0},
        {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 84.0},
        {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81.0},
        {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78.0},
        {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75.0},
        {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72.0},
        {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69.0},
        {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69.0},
        {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 66.0},
    ]


@pytest.fixture
def web_dev_skills():
    """Skills for Web Developers (15-1299.08)."""
    return [
        {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 81.0},
        {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78.0},
        {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0},
        {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0},
        {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 72.0},
        {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66.0},
    ]


@pytest.fixture
def nurse_skills():
    """Skills for Registered Nurses (29-1141.00)."""
    return [
        {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 82.0},
        {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78.0},
        {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75.0},
        {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72.0},
        {"element_id": "2.B.9.b", "skill_name": "Service Orientation", "importance": 75.0},
        {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72.0},
        {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 66.0},
    ]


@pytest.fixture
def sysadmin_skills():
    """Skills for Network and Computer Systems Administrators (15-1244.00)."""
    return [
        {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84.0},
        {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81.0},
        {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78.0},
        {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0},
        {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 72.0},
        {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72.0},
        {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0},
        {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69.0},
    ]


# ===================================================================
# skill_direction_vector tests
# ===================================================================

class TestSkillDirectionVector:
    """Tests for skill_direction_vector()."""

    def test_same_domain_transition(self, software_dev_skills, web_dev_skills):
        """Software Dev -> Web Dev: small deltas for shared skills."""
        result = skill_direction_vector(software_dev_skills, web_dev_skills)
        assert result is not None

        # Programming: 81 - 84 = -3 (target needs slightly less)
        assert result["2.B.1.g"] == pytest.approx(-3.0)
        # Design appears only in target -> delta = 72 - 0 = 72
        assert result["2.B.5.c"] == pytest.approx(72.0)
        # Systems Analysis only in origin -> delta = 0 - 78 = -78
        assert result["2.B.8.d"] == pytest.approx(-78.0)

    def test_cross_domain_transition(self, software_dev_skills, nurse_skills):
        """Software Dev -> Nurse: large deltas across domains."""
        result = skill_direction_vector(software_dev_skills, nurse_skills)
        assert result is not None

        # Programming only in origin: 0 - 84 = -84
        assert result["2.B.1.g"] == pytest.approx(-84.0)
        # Service Orientation only in target: 75
        assert result["2.B.9.b"] == pytest.approx(75.0)
        # Critical Thinking shared: 82 - 81 = 1
        assert result["2.B.8.a"] == pytest.approx(1.0)

    def test_sysadmin_to_software_dev(self, sysadmin_skills, software_dev_skills):
        """SysAdmin -> Software Dev: moderate deltas."""
        result = skill_direction_vector(sysadmin_skills, software_dev_skills)
        assert result is not None

        # Programming: 84 - 72 = +12 (target needs more)
        assert result["2.B.1.g"] == pytest.approx(12.0)
        # Troubleshooting only in origin: 0 - 84 = -84
        assert result["2.B.9.a"] == pytest.approx(-84.0)

    def test_both_empty(self):
        """Empty skill lists return None."""
        assert skill_direction_vector([], []) is None

    def test_one_side_empty(self, software_dev_skills):
        """One empty list still produces a valid result."""
        result = skill_direction_vector([], software_dev_skills)
        assert result is not None
        # All deltas should be positive (target importance - 0)
        assert all(v >= 0 for v in result.values())


# ===================================================================
# experience_transfer_ratio tests
# ===================================================================

class TestExperienceTransferRatio:
    """Tests for experience_transfer_ratio()."""

    def test_high_transfer_same_domain(self, software_dev_skills, web_dev_skills):
        """Software Dev -> Web Dev should have high transfer."""
        ratio = experience_transfer_ratio(software_dev_skills, web_dev_skills, top_k=5)
        assert ratio is not None
        # Most top SW Dev skills overlap with Web Dev
        assert ratio >= 0.4  # at least 2 of top 5 should transfer

    def test_low_transfer_cross_domain(self, software_dev_skills, nurse_skills):
        """Software Dev -> Nurse should have lower transfer."""
        ratio = experience_transfer_ratio(software_dev_skills, nurse_skills, top_k=5)
        assert ratio is not None
        # Fewer top skills transfer across domains
        assert ratio <= 0.8

    def test_sysadmin_to_web_dev(self, sysadmin_skills, web_dev_skills):
        """SysAdmin -> Web Dev: moderate transfer."""
        ratio = experience_transfer_ratio(sysadmin_skills, web_dev_skills, top_k=5)
        assert ratio is not None
        assert 0.0 <= ratio <= 1.0

    def test_empty_origin(self, web_dev_skills):
        """Empty origin returns None."""
        assert experience_transfer_ratio([], web_dev_skills) is None

    def test_full_transfer(self):
        """When all top skills appear in target, ratio should be 1.0."""
        origin = [
            {"element_id": "A", "importance": 90},
            {"element_id": "B", "importance": 80},
        ]
        target = [
            {"element_id": "A", "importance": 70},
            {"element_id": "B", "importance": 60},
        ]
        assert experience_transfer_ratio(origin, target, top_k=2) == pytest.approx(1.0)


# ===================================================================
# occupation_demand_signal tests
# ===================================================================

class TestOccupationDemandSignal:
    """Tests for occupation_demand_signal()."""

    def test_bright_outlook_code(self):
        """Known Bright Outlook code returns True."""
        assert occupation_demand_signal("15-1252.00") is True

    def test_non_bright_outlook_known_code(self):
        """Known code that is NOT Bright Outlook returns False."""
        assert occupation_demand_signal("15-1244.00") is False

    def test_unknown_code(self):
        """Unknown code returns None (graceful missing data)."""
        assert occupation_demand_signal("99-9999.00") is None

    def test_empty_code(self):
        """Empty string returns None."""
        assert occupation_demand_signal("") is None


# ===================================================================
# salary_delta tests
# ===================================================================

class TestSalaryDelta:
    """Tests for salary_delta()."""

    def test_raise_transition(self):
        """Web Dev -> Software Dev should be a raise."""
        delta = salary_delta("15-1299.08", "15-1252.00")
        assert delta is not None
        assert delta > 0  # SW Dev pays more

    def test_pay_cut_transition(self):
        """Software Dev -> Web Dev should be a pay cut."""
        delta = salary_delta("15-1252.00", "15-1299.08")
        assert delta is not None
        assert delta < 0

    def test_same_occupation(self):
        """Same code -> delta should be 0."""
        delta = salary_delta("15-1252.00", "15-1252.00")
        assert delta is not None
        assert delta == pytest.approx(0.0)

    def test_unknown_origin(self):
        """Unknown origin returns None."""
        assert salary_delta("99-0000.00", "15-1252.00") is None

    def test_unknown_target(self):
        """Unknown target returns None."""
        assert salary_delta("15-1252.00", "99-0000.00") is None


# ===================================================================
# credential_barrier tests
# ===================================================================

class TestCredentialBarrier:
    """Tests for credential_barrier()."""

    def test_licensed_occupation(self):
        """Registered Nurse requires credentials."""
        result = credential_barrier("29-1141.00")
        assert result is not None
        assert result["required"] is True
        assert len(result["credentials"]) > 0
        assert any("RN" in c or "Nurse" in c for c in result["credentials"])

    def test_no_credentials_required(self):
        """Software Dev does not require specific credentials."""
        result = credential_barrier("15-1252.00")
        assert result is not None
        assert result["required"] is False
        assert len(result["credentials"]) == 0

    def test_accountant_credentials(self):
        """Accountants have optional CPA."""
        result = credential_barrier("13-2011.00")
        assert result is not None
        assert result["required"] is True
        assert any("CPA" in c for c in result["credentials"])

    def test_unknown_target(self):
        """Unknown target returns None."""
        assert credential_barrier("99-0000.00") is None


# ===================================================================
# industry_distance tests
# ===================================================================

class TestIndustryDistance:
    """Tests for industry_distance()."""

    def test_same_sector(self):
        """Software Dev -> Web Dev: same NAICS sector -> 0.0."""
        dist = industry_distance("15-1252.00", "15-1299.08")
        assert dist is not None
        assert dist == pytest.approx(0.0)

    def test_adjacent_sector(self):
        """Software Dev (51/Info) -> Accountant (52/Financial) -> 0.5."""
        dist = industry_distance("15-1252.00", "13-2011.00")
        assert dist is not None
        assert dist == pytest.approx(0.5)

    def test_distant_sector(self):
        """Software Dev (51/Info) -> Nurse (62/Healthcare) -> 1.0."""
        dist = industry_distance("15-1252.00", "29-1141.00")
        assert dist is not None
        assert dist == pytest.approx(1.0)

    def test_unknown_code(self):
        """Unknown SOC prefix returns None."""
        assert industry_distance("15-1252.00", "99-0000.00") is None

    def test_empty_codes(self):
        """Empty codes return None."""
        assert industry_distance("", "15-1252.00") is None


# ===================================================================
# TransitionFeatureVector tests
# ===================================================================

class TestTransitionFeatureVector:
    """Tests for the FeatureVector dataclass."""

    def test_to_array_shape(self):
        """to_array() returns correct shape."""
        vec = TransitionFeatureVector(
            skill_direction_mean=5.0,
            skill_direction_std=10.0,
            skill_direction_max_positive=72.0,
            skill_direction_max_negative=-84.0,
            experience_transfer_ratio=0.6,
            bright_outlook=True,
            salary_delta=42430.0,
            salary_delta_pct=54.2,
            credential_required=False,
            credential_count=0,
            industry_distance=0.0,
        )
        arr = vec.to_array()
        assert arr.shape == (11,)
        assert not np.any(np.isnan(arr))

    def test_to_array_none_handling(self):
        """None values become NaN in the array."""
        vec = TransitionFeatureVector()
        arr = vec.to_array()
        assert arr.shape == (11,)
        assert np.all(np.isnan(arr))

    def test_feature_explanations(self):
        """feature_explanations() returns all 11 fields."""
        explanations = TransitionFeatureVector.feature_explanations()
        assert len(explanations) == 11
        assert "skill_direction_mean" in explanations
        assert "industry_distance" in explanations

    def test_ordered_feature_names(self):
        """ordered_feature_names() length matches to_array()."""
        names = TransitionFeatureVector.ordered_feature_names()
        vec = TransitionFeatureVector()
        arr = vec.to_array()
        assert len(names) == arr.shape[0]


# ===================================================================
# build_transition_features integration test
# ===================================================================

class TestBuildTransitionFeatures:
    """Integration tests for build_transition_features()."""

    def test_sw_dev_to_web_dev(self, software_dev_skills, web_dev_skills):
        """Software Dev -> Web Dev produces a fully populated vector."""
        vec = build_transition_features(
            origin_code="15-1252.00",
            target_code="15-1299.08",
            origin_skills=software_dev_skills,
            target_skills=web_dev_skills,
        )
        assert vec.skill_direction_mean is not None
        assert vec.experience_transfer_ratio is not None
        assert vec.bright_outlook is True
        assert vec.salary_delta is not None
        assert vec.salary_delta < 0  # pay cut
        assert vec.credential_required is False
        assert vec.industry_distance == pytest.approx(0.0)

    def test_sw_dev_to_nurse(self, software_dev_skills, nurse_skills):
        """Software Dev -> Nurse: cross-domain transition."""
        vec = build_transition_features(
            origin_code="15-1252.00",
            target_code="29-1141.00",
            origin_skills=software_dev_skills,
            target_skills=nurse_skills,
        )
        assert vec.industry_distance == pytest.approx(1.0)
        assert vec.credential_required is True
        assert vec.salary_delta is not None and vec.salary_delta < 0

    def test_unknown_codes_graceful(self):
        """Unknown codes produce a vector with None fields, no crash."""
        vec = build_transition_features(
            origin_code="99-0001.00",
            target_code="99-0002.00",
            origin_skills=[],
            target_skills=[],
        )
        assert vec.skill_direction_mean is None
        assert vec.experience_transfer_ratio is None
        assert vec.bright_outlook is None
        assert vec.salary_delta is None
        assert vec.credential_required is None
        assert vec.industry_distance is None

    def test_sysadmin_to_sw_dev(self, sysadmin_skills, software_dev_skills):
        """SysAdmin -> Software Dev: moderate same-sector transition."""
        vec = build_transition_features(
            origin_code="15-1244.00",
            target_code="15-1252.00",
            origin_skills=sysadmin_skills,
            target_skills=software_dev_skills,
        )
        assert vec.industry_distance == pytest.approx(0.0)
        assert vec.salary_delta is not None and vec.salary_delta > 0


# ===================================================================
# augment_calibration_array test
# ===================================================================

class TestAugmentCalibrationArray:
    """Tests for augment_calibration_array()."""

    def test_concatenation(self):
        """Base + transition arrays concatenate correctly."""
        base = np.array([1.0, 2.0, 3.0])
        transition = TransitionFeatureVector(
            skill_direction_mean=5.0,
            experience_transfer_ratio=0.5,
        )
        result = augment_calibration_array(base, transition)
        assert result.shape[0] == 3 + 11
        # First 3 from base
        np.testing.assert_array_equal(result[:3], [1.0, 2.0, 3.0])
        # Fourth element is skill_direction_mean = 5.0
        assert result[3] == pytest.approx(5.0)

    def test_with_reshaped_base(self):
        """Works even if base_array has shape (1, N)."""
        base = np.array([[10.0, 20.0]])
        transition = TransitionFeatureVector(salary_delta=5000.0)
        result = augment_calibration_array(base, transition)
        assert result.ndim == 1
        assert result.shape[0] == 2 + 11
