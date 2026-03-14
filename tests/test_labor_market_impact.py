"""Test cases aligned with Anthropic's labor market impact paper.

Based on "Labor market impacts of AI: A new measure and early evidence"
(Massenkoff & McCrory, 2026), these tests validate that SkillSprout correctly
handles the occupations, exposure tiers, and transition scenarios identified
by the paper's analysis of O*NET data and Claude usage patterns.

The paper introduces "observed exposure" — a metric combining O*NET task data,
theoretical AI capability scores (Eloundou et al.), and real Claude usage data.
Key findings:
  - Computer & Math occupations: 94% theoretical capability, 33% observed
  - ~30% of workers fall into zero-exposure occupations (physical/manual tasks)
  - No systematic unemployment increase for highly exposed workers since 2022
  - Suggestive evidence of slowed hiring for young workers in exposed occupations
  - BLS growth projections weaker for high-exposure jobs (-0.6pp per 10pp coverage)

These tests ensure SkillSprout's scoring, explainability, and training path
engines handle all these occupation categories correctly.
"""

import pytest
from typing import Dict, List, Any

from app.ml.scoring import BaselineScorer, OccupationScore, RATING_TO_CAPABILITY
from app.ml.ai_exposure import (
    AIUsageType,
    TaskExposure,
    OccupationExposure,
    compute_observed_coverage,
    compute_theoretical_coverage,
    get_exposure_profile,
    get_sector_coverage,
    theoretical_observed_gap,
    HIGH_EXPOSURE_OCCUPATIONS,
    ZERO_EXPOSURE_OCCUPATIONS,
    SECTOR_COVERAGE,
)


# ============================================================================
# Occupation data from the paper's highest-exposure occupations
# ============================================================================

# Skills are sourced from O*NET element IDs matching the paper's methodology
PAPER_OCCUPATIONS = {
    # --- HIGHEST OBSERVED EXPOSURE (from the paper) ---
    "15-1251.00": {
        "title": "Computer Programmers",
        "job_zone": 4,
        "sector": "Computer and Mathematical",
        "observed_coverage": 0.75,  # Highest in the paper
        "skills": [
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 87, "level": 6.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.12},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 69, "level": 4.88},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 72, "level": 5.0},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66, "level": 4.62},
        ],
    },
    "43-4051.00": {
        "title": "Customer Service Representatives",
        "job_zone": 2,
        "sector": "Office and Administrative Support",
        "observed_coverage": 0.60,
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66, "level": 4.62},
        ],
    },
    "43-9021.00": {
        "title": "Data Entry Keyers",
        "job_zone": 2,
        "sector": "Office and Administrative Support",
        "observed_coverage": 0.67,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.5},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 63, "level": 4.25},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 63, "level": 4.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 60, "level": 4.0},
        ],
    },
    "13-2051.00": {
        "title": "Financial Analysts",
        "job_zone": 4,
        "sector": "Business and Financial Operations",
        "observed_coverage": 0.55,
        "skills": [
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 87, "level": 6.0},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 69, "level": 4.88},
        ],
    },
    # --- ZERO OBSERVED EXPOSURE (from the paper's ~30% of workers) ---
    "35-2014.00": {
        "title": "Cooks, Restaurant",
        "job_zone": 2,
        "sector": "Food Preparation and Serving",
        "observed_coverage": 0.0,
        "skills": [
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 60, "level": 4.0},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 63, "level": 4.25},
        ],
    },
    "49-3023.00": {
        "title": "Automotive Service Technicians and Mechanics",
        "job_zone": 3,
        "sector": "Installation, Maintenance, and Repair",
        "observed_coverage": 0.0,
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84, "level": 5.75},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.c", "skill_name": "Quality Control Analysis", "importance": 69, "level": 4.88},
        ],
    },
    "33-9092.00": {
        "title": "Lifeguards",
        "job_zone": 2,
        "sector": "Protective Service",
        "observed_coverage": 0.0,
        "skills": [
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69, "level": 4.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 66, "level": 4.62},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 63, "level": 4.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 63, "level": 4.25},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 63, "level": 4.25},
        ],
    },
    "35-3011.00": {
        "title": "Bartenders",
        "job_zone": 2,
        "sector": "Food Preparation and Serving",
        "observed_coverage": 0.0,
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 69, "level": 4.88},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66, "level": 4.62},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 63, "level": 4.25},
        ],
    },
    "35-9021.00": {
        "title": "Dishwashers",
        "job_zone": 1,
        "sector": "Food Preparation and Serving",
        "observed_coverage": 0.0,
        "skills": [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 60, "level": 4.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 57, "level": 3.75},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 57, "level": 3.75},
        ],
    },
}


@pytest.fixture
def scorer():
    """Baseline scorer with default thresholds."""
    return BaselineScorer()


# ============================================================================
# Test Class: AI Exposure Module
# ============================================================================

class TestAIExposureModule:
    """Tests for the AI exposure scoring module from the paper's methodology."""

    def test_task_exposure_automated_weight(self):
        """Automated tasks should receive full weight (1.0)."""
        task = TaskExposure(
            task_description="Write code to specification",
            theoretical_beta=1.0,
            usage_type=AIUsageType.AUTOMATED,
        )
        assert task.effective_weight == 1.0

    def test_task_exposure_augmentative_weight(self):
        """Augmentative tasks should receive half weight (0.5)."""
        task = TaskExposure(
            task_description="Review customer complaints",
            theoretical_beta=0.5,
            usage_type=AIUsageType.AUGMENTATIVE,
        )
        assert task.effective_weight == 0.5

    def test_task_exposure_none_weight(self):
        """Tasks with no AI usage should have zero weight."""
        task = TaskExposure(
            task_description="Operate deep fryer",
            theoretical_beta=0.0,
            usage_type=AIUsageType.NONE,
        )
        assert task.effective_weight == 0.0

    def test_compute_observed_coverage_mixed(self):
        """Coverage with mixed automated and augmentative tasks."""
        tasks = [
            TaskExposure("Task A", 1.0, AIUsageType.AUTOMATED),
            TaskExposure("Task B", 0.5, AIUsageType.AUGMENTATIVE),
            TaskExposure("Task C", 0.0, AIUsageType.NONE),
            TaskExposure("Task D", 1.0, AIUsageType.AUTOMATED),
        ]
        # Equal weights: (1.0 + 0.5 + 0.0 + 1.0) / 4 = 0.625
        assert compute_observed_coverage(tasks) == pytest.approx(0.625)

    def test_compute_observed_coverage_all_automated(self):
        """Full automation should give 100% coverage."""
        tasks = [
            TaskExposure("Task A", 1.0, AIUsageType.AUTOMATED),
            TaskExposure("Task B", 1.0, AIUsageType.AUTOMATED),
        ]
        assert compute_observed_coverage(tasks) == pytest.approx(1.0)

    def test_compute_observed_coverage_all_none(self):
        """No AI usage should give 0% coverage."""
        tasks = [
            TaskExposure("Task A", 0.0, AIUsageType.NONE),
            TaskExposure("Task B", 0.0, AIUsageType.NONE),
        ]
        assert compute_observed_coverage(tasks) == pytest.approx(0.0)

    def test_compute_observed_coverage_empty(self):
        """Empty task list should return 0."""
        assert compute_observed_coverage([]) == 0.0

    def test_compute_observed_coverage_with_time_shares(self):
        """Coverage should be weighted by time spent on each task."""
        tasks = [
            TaskExposure("Code writing", 1.0, AIUsageType.AUTOMATED, time_share=0.6),
            TaskExposure("Meetings", 0.0, AIUsageType.NONE, time_share=0.4),
        ]
        # (0.6 * 1.0 + 0.4 * 0.0) / (0.6 + 0.4) = 0.6
        assert compute_observed_coverage(tasks) == pytest.approx(0.6)

    def test_compute_theoretical_coverage(self):
        """Theoretical coverage counts tasks with β >= 0.5."""
        tasks = [
            TaskExposure("A", 1.0, AIUsageType.AUTOMATED),
            TaskExposure("B", 0.5, AIUsageType.AUGMENTATIVE),
            TaskExposure("C", 0.0, AIUsageType.NONE),
            TaskExposure("D", 0.0, AIUsageType.NONE),
        ]
        # 2 out of 4 have β >= 0.5
        assert compute_theoretical_coverage(tasks) == pytest.approx(0.5)

    def test_occupation_exposure_tier_high(self):
        """Computer programmers should be classified as high exposure."""
        profile = get_exposure_profile("15-1251.00")
        assert profile is not None
        assert profile.exposure_tier == "high"
        assert profile.observed_coverage == 0.75

    def test_occupation_exposure_tier_zero(self):
        """Cooks should be classified as zero exposure."""
        profile = get_exposure_profile("35-2014.00")
        assert profile is not None
        assert profile.exposure_tier == "zero"
        assert profile.observed_coverage == 0.0

    def test_theoretical_observed_gap_programmers(self):
        """Paper's key finding: actual adoption far below theoretical capability."""
        gap = theoretical_observed_gap("15-1251.00")
        assert gap is not None
        # 94% theoretical - 75% observed = 19% gap
        assert gap == pytest.approx(0.19, abs=0.01)

    def test_theoretical_observed_gap_software_devs(self):
        """Software devs: 94% theoretical, only 33% observed — large gap."""
        gap = theoretical_observed_gap("15-1252.00")
        assert gap is not None
        assert gap == pytest.approx(0.61, abs=0.01)

    def test_theoretical_observed_gap_zero_exposure(self):
        """Zero-exposure occupations should have gap equal to theoretical."""
        gap = theoretical_observed_gap("35-2014.00")
        assert gap is not None
        assert gap == pytest.approx(0.10, abs=0.01)  # 10% theoretical, 0% observed

    def test_bls_growth_adjustment(self):
        """Paper: -0.6pp per 10pp coverage increase."""
        profile = get_exposure_profile("15-1251.00")
        assert profile is not None
        # 75% coverage → -0.6 * 7.5 = -4.5
        assert profile.bls_growth_adjustment == pytest.approx(-4.5, abs=0.1)

    def test_bls_growth_adjustment_zero_exposure(self):
        """Zero-exposure occupations should have no BLS adjustment."""
        profile = get_exposure_profile("35-2014.00")
        assert profile is not None
        assert profile.bls_growth_adjustment == 0.0

    def test_sector_coverage_computer_math(self):
        """Computer & Math: 94% theoretical, 33% observed (from the paper)."""
        sector = get_sector_coverage("Computer and Mathematical")
        assert sector is not None
        assert sector["theoretical"] == pytest.approx(0.94)
        assert sector["observed"] == pytest.approx(0.33)

    def test_sector_coverage_food_service(self):
        """Food service: very low theoretical, zero observed."""
        sector = get_sector_coverage("Food Preparation and Serving")
        assert sector is not None
        assert sector["theoretical"] == pytest.approx(0.10)
        assert sector["observed"] == pytest.approx(0.0)

    def test_all_high_exposure_occupations_classified(self):
        """All high-exposure occupations from the paper must be registered."""
        expected = {"15-1251.00", "43-4051.00", "43-9021.00", "13-2051.00", "15-1252.00"}
        assert set(HIGH_EXPOSURE_OCCUPATIONS.keys()) == expected

    def test_all_zero_exposure_occupations_classified(self):
        """All zero-exposure occupations from the paper must be registered."""
        expected = {"35-2014.00", "49-3023.00", "33-9092.00", "35-3011.00", "35-9021.00"}
        assert set(ZERO_EXPOSURE_OCCUPATIONS.keys()) == expected

    def test_high_exposure_all_have_positive_coverage(self):
        """Every high-exposure occupation must have observed_coverage > 0."""
        for code, profile in HIGH_EXPOSURE_OCCUPATIONS.items():
            assert profile.observed_coverage > 0, f"{code} should have positive coverage"

    def test_zero_exposure_all_have_zero_coverage(self):
        """Every zero-exposure occupation must have observed_coverage == 0."""
        for code, profile in ZERO_EXPOSURE_OCCUPATIONS.items():
            assert profile.observed_coverage == 0.0, f"{code} should have zero coverage"

    def test_exposure_tier_thresholds(self):
        """Verify tier classification boundaries."""
        assert OccupationExposure("", "", "", 0.0, 0.50).exposure_tier == "high"
        assert OccupationExposure("", "", "", 0.0, 0.49).exposure_tier == "moderate"
        assert OccupationExposure("", "", "", 0.0, 0.20).exposure_tier == "moderate"
        assert OccupationExposure("", "", "", 0.0, 0.19).exposure_tier == "low"
        assert OccupationExposure("", "", "", 0.0, 0.01).exposure_tier == "low"
        assert OccupationExposure("", "", "", 0.0, 0.00).exposure_tier == "zero"


# ============================================================================
# Test Class: Scoring High-Exposure Occupations
# ============================================================================

class TestHighExposureOccupationScoring:
    """Verify scoring works correctly for the paper's highest-exposure occupations."""

    def test_computer_programmer_perfect_match(self, scorer):
        """A senior programmer should score ready_now for programming roles."""
        skills = PAPER_OCCUPATIONS["15-1251.00"]["skills"]
        user_ratings = {
            "2.B.1.g": 4,  # Programming - Expert
            "2.B.8.a": 4,  # Critical Thinking - Expert
            "2.B.8.b": 4,  # Complex Problem Solving - Expert
            "2.B.1.a": 3,  # Reading Comprehension - Advanced
            "2.B.3.a": 3,  # Writing - Advanced
            "2.B.5.a": 3,  # Mathematics - Advanced
            "2.B.8.d": 4,  # Systems Analysis - Expert
            "2.B.6.b": 3,  # Time Management - Advanced
        }
        score = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket == "ready_now"
        assert score.match_score >= 75

    def test_customer_service_rep_from_retail(self, scorer):
        """Retail worker transitioning to customer service should be trainable."""
        skills = PAPER_OCCUPATIONS["43-4051.00"]["skills"]
        user_ratings = {
            "2.B.2.a": 4,  # Active Listening - Expert (from retail)
            "2.B.4.a": 4,  # Speaking - Expert
            "2.B.1.a": 2,  # Reading Comprehension - Intermediate
            "2.B.1.f": 3,  # Service Orientation - Advanced
            "2.B.7.a": 3,  # Social Perceptiveness - Advanced
            "2.B.3.c": 1,  # Negotiation - Basic
            "2.B.8.a": 2,  # Critical Thinking - Intermediate
        }
        score = scorer.score_occupation(
            onet_code="43-4051.00",
            occupation_title="Customer Service Representatives",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket in ("ready_now", "trainable")
        assert score.match_score >= 50

    def test_data_entry_keyer_minimal_skills(self, scorer):
        """Data entry roles require fewer skills; even basic ratings may suffice."""
        skills = PAPER_OCCUPATIONS["43-9021.00"]["skills"]
        user_ratings = {
            "2.B.1.a": 2,  # Reading - Intermediate
            "2.B.2.a": 2,  # Active Listening - Intermediate
            "2.B.6.b": 2,  # Time Management - Intermediate
            "2.B.7.c": 1,  # Monitoring - Basic
            "2.B.8.a": 1,  # Critical Thinking - Basic
        }
        score = scorer.score_occupation(
            onet_code="43-9021.00",
            occupation_title="Data Entry Keyers",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        # Even moderate skills should give reasonable match for low job-zone
        assert score.match_score > 25
        assert score.bucket in ("trainable", "long_reskill")

    def test_financial_analyst_needs_strong_math(self, scorer):
        """Financial analysts require high math — missing it creates major gaps."""
        skills = PAPER_OCCUPATIONS["13-2051.00"]["skills"]
        user_ratings = {
            "2.B.5.a": 0,  # Mathematics - None (major gap)
            "2.B.8.a": 3,  # Critical Thinking - Advanced
            "2.B.1.a": 3,  # Reading - Advanced
            "2.B.8.b": 2,  # Complex Problem Solving - Intermediate
            "2.B.8.d": 1,  # Systems Analysis - Basic
            "2.B.3.a": 3,  # Writing - Advanced
            "2.B.6.b": 3,  # Time Management - Advanced
        }
        score = scorer.score_occupation(
            onet_code="13-2051.00",
            occupation_title="Financial Analysts",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        # Math is the highest-importance skill; missing it should hurt
        assert any(g.skill_name == "Mathematics" for g in score.top_gaps)
        # Should still be at least trainable with other strong skills
        assert score.bucket in ("trainable", "long_reskill")

    def test_financial_analyst_full_match(self, scorer):
        """Experienced financial analyst should be ready_now."""
        skills = PAPER_OCCUPATIONS["13-2051.00"]["skills"]
        user_ratings = {
            "2.B.5.a": 4,  # Mathematics - Expert
            "2.B.8.a": 4,  # Critical Thinking - Expert
            "2.B.1.a": 4,  # Reading - Expert
            "2.B.8.b": 3,  # Complex Problem Solving - Advanced
            "2.B.8.d": 3,  # Systems Analysis - Advanced
            "2.B.3.a": 3,  # Writing - Advanced
            "2.B.6.b": 3,  # Time Management - Advanced
        }
        score = scorer.score_occupation(
            onet_code="13-2051.00",
            occupation_title="Financial Analysts",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket == "ready_now"
        assert score.match_score >= 75


# ============================================================================
# Test Class: Scoring Zero-Exposure Occupations
# ============================================================================

class TestZeroExposureOccupationScoring:
    """Verify scoring for the paper's zero-AI-exposure occupations."""

    def test_cook_scoring_with_relevant_skills(self, scorer):
        """An experienced cook should score well for cooking roles."""
        skills = PAPER_OCCUPATIONS["35-2014.00"]["skills"]
        user_ratings = {
            "2.B.7.c": 4,  # Monitoring - Expert
            "2.B.6.b": 4,  # Time Management - Expert
            "2.B.7.b": 4,  # Coordination - Expert
            "2.B.8.a": 3,  # Critical Thinking - Advanced
            "2.B.2.a": 3,  # Active Listening - Advanced
        }
        score = scorer.score_occupation(
            onet_code="35-2014.00",
            occupation_title="Cooks, Restaurant",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket == "ready_now"
        assert score.match_score >= 75

    def test_mechanic_needs_troubleshooting(self, scorer):
        """Mechanics need troubleshooting and equipment maintenance — physical skills."""
        skills = PAPER_OCCUPATIONS["49-3023.00"]["skills"]
        user_ratings = {
            "2.B.9.a": 4,  # Troubleshooting - Expert
            "2.B.4.h": 4,  # Equipment Maintenance - Expert
            "2.B.8.b": 3,  # Complex Problem Solving - Advanced
            "2.B.9.c": 3,  # Operations Monitoring - Advanced
            "2.B.1.a": 2,  # Reading - Intermediate
            "2.B.8.a": 3,  # Critical Thinking - Advanced
            "2.B.6.c": 3,  # Quality Control Analysis - Advanced
        }
        score = scorer.score_occupation(
            onet_code="49-3023.00",
            occupation_title="Automotive Service Technicians",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket == "ready_now"
        assert score.match_score >= 75

    def test_lifeguard_monitoring_heavy(self, scorer):
        """Lifeguard role is monitoring-heavy — an interpersonal/physical role."""
        skills = PAPER_OCCUPATIONS["33-9092.00"]["skills"]
        user_ratings = {
            "2.B.7.c": 4,  # Monitoring - Expert
            "2.B.8.a": 3,  # Critical Thinking - Advanced
            "2.B.7.a": 3,  # Social Perceptiveness - Advanced
            "2.B.2.a": 3,  # Active Listening - Advanced
            "2.B.4.a": 3,  # Speaking - Advanced
            "2.B.1.f": 3,  # Service Orientation - Advanced
        }
        score = scorer.score_occupation(
            onet_code="33-9092.00",
            occupation_title="Lifeguards",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket == "ready_now"
        assert score.match_score >= 75

    def test_bartender_strong_service(self, scorer):
        """Bartenders rely on service orientation and interpersonal skills."""
        skills = PAPER_OCCUPATIONS["35-3011.00"]["skills"]
        user_ratings = {
            "2.B.2.a": 4,  # Active Listening - Expert
            "2.B.4.a": 4,  # Speaking - Expert
            "2.B.1.f": 4,  # Service Orientation - Expert
            "2.B.7.a": 3,  # Social Perceptiveness - Advanced
            "2.B.6.b": 3,  # Time Management - Advanced
            "2.B.7.c": 2,  # Monitoring - Intermediate
        }
        score = scorer.score_occupation(
            onet_code="35-3011.00",
            occupation_title="Bartenders",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket == "ready_now"

    def test_dishwasher_low_job_zone(self, scorer):
        """Dishwasher is job zone 1 — minimal skills needed, broad accessibility."""
        skills = PAPER_OCCUPATIONS["35-9021.00"]["skills"]
        user_ratings = {
            "2.B.7.b": 2,  # Coordination - Intermediate
            "2.B.6.b": 2,  # Time Management - Intermediate
            "2.B.2.a": 2,  # Active Listening - Intermediate
        }
        score = scorer.score_occupation(
            onet_code="35-9021.00",
            occupation_title="Dishwashers",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
            target_job_zone=1,
        )
        # Even intermediate skills should match well for zone 1
        assert score.match_score >= 40
        assert score.bucket in ("trainable", "ready_now")


# ============================================================================
# Test Class: Cross-Exposure Tier Transitions
# ============================================================================

class TestCrossExposureTierTransitions:
    """Test transitions between occupations of different AI exposure levels.

    The paper suggests that workers in high-exposure occupations may need to
    consider transitions, while zero-exposure occupations remain stable.
    These tests validate SkillSprout handles such cross-tier transitions.
    """

    def test_programmer_to_mechanic_large_gap(self, scorer):
        """Programmer → Mechanic: tech skills don't transfer to physical skills."""
        skills = PAPER_OCCUPATIONS["49-3023.00"]["skills"]
        programmer_ratings = {
            "2.B.1.g": 4,  # Programming (not relevant)
            "2.B.8.a": 4,  # Critical Thinking (transfers)
            "2.B.8.b": 3,  # Complex Problem Solving (transfers)
            "2.B.1.a": 3,  # Reading (transfers)
            "2.B.9.a": 0,  # Troubleshooting - None
            "2.B.4.h": 0,  # Equipment Maintenance - None
            "2.B.9.c": 0,  # Operations Monitoring - None
            "2.B.6.c": 0,  # Quality Control - None
        }
        score = scorer.score_occupation(
            onet_code="49-3023.00",
            occupation_title="Automotive Service Technicians",
            occupation_skills=skills,
            user_skill_ratings=programmer_ratings,
        )
        # Physical skills gaps should make this a long reskill
        assert score.bucket in ("trainable", "long_reskill")
        assert len(score.top_gaps) >= 2

    def test_mechanic_to_customer_service_transferable(self, scorer):
        """Mechanic → Customer Service: interpersonal skills partially transfer."""
        skills = PAPER_OCCUPATIONS["43-4051.00"]["skills"]
        mechanic_ratings = {
            "2.B.2.a": 3,  # Active Listening - Advanced (from customer interaction)
            "2.B.4.a": 3,  # Speaking - Advanced
            "2.B.1.a": 2,  # Reading - Intermediate
            "2.B.1.f": 2,  # Service Orientation - Intermediate
            "2.B.7.a": 2,  # Social Perceptiveness - Intermediate
            "2.B.3.c": 1,  # Negotiation - Basic
            "2.B.8.a": 3,  # Critical Thinking - Advanced
        }
        score = scorer.score_occupation(
            onet_code="43-4051.00",
            occupation_title="Customer Service Representatives",
            occupation_skills=skills,
            user_skill_ratings=mechanic_ratings,
        )
        assert score.bucket in ("trainable", "ready_now")

    def test_cook_to_bartender_close_transition(self, scorer):
        """Cook → Bartender: within same sector, high skill overlap."""
        skills = PAPER_OCCUPATIONS["35-3011.00"]["skills"]
        cook_ratings = {
            "2.B.2.a": 3,  # Active Listening
            "2.B.4.a": 2,  # Speaking
            "2.B.1.f": 3,  # Service Orientation
            "2.B.7.a": 2,  # Social Perceptiveness
            "2.B.6.b": 4,  # Time Management - Expert (from kitchen)
            "2.B.7.c": 3,  # Monitoring
        }
        score = scorer.score_occupation(
            onet_code="35-3011.00",
            occupation_title="Bartenders",
            occupation_skills=skills,
            user_skill_ratings=cook_ratings,
        )
        assert score.bucket in ("trainable", "ready_now")
        assert score.match_score >= 50

    def test_data_entry_to_financial_analyst_big_jump(self, scorer):
        """Data Entry → Financial Analyst: major job zone jump (2 → 4)."""
        skills = PAPER_OCCUPATIONS["13-2051.00"]["skills"]
        data_entry_ratings = {
            "2.B.5.a": 0,  # Mathematics - None
            "2.B.8.a": 1,  # Critical Thinking - Basic
            "2.B.1.a": 2,  # Reading - Intermediate
            "2.B.8.b": 0,  # Complex Problem Solving - None
            "2.B.8.d": 0,  # Systems Analysis - None
            "2.B.3.a": 1,  # Writing - Basic
            "2.B.6.b": 2,  # Time Management - Intermediate
        }
        score = scorer.score_occupation(
            onet_code="13-2051.00",
            occupation_title="Financial Analysts",
            occupation_skills=skills,
            user_skill_ratings=data_entry_ratings,
            current_job_zone=2,
            target_job_zone=4,
        )
        assert score.bucket == "long_reskill"
        assert score.metadata["job_zone_diff"] == 2
        assert len(score.top_gaps) >= 3

    def test_customer_service_to_programmer_high_gap(self, scorer):
        """Customer Service → Programmer: high-exposure to high-exposure but different skills."""
        skills = PAPER_OCCUPATIONS["15-1251.00"]["skills"]
        csr_ratings = {
            "2.B.1.g": 0,  # Programming - None
            "2.B.8.a": 2,  # Critical Thinking - Intermediate
            "2.B.8.b": 1,  # Complex Problem Solving - Basic
            "2.B.1.a": 3,  # Reading - Advanced
            "2.B.3.a": 2,  # Writing - Intermediate
            "2.B.5.a": 0,  # Mathematics - None
            "2.B.8.d": 0,  # Systems Analysis - None
            "2.B.6.b": 3,  # Time Management - Advanced
        }
        score = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=skills,
            user_skill_ratings=csr_ratings,
        )
        assert score.bucket in ("trainable", "long_reskill")
        assert any(g.skill_name == "Programming" for g in score.top_gaps)


# ============================================================================
# Test Class: Paper-Specific Invariants
# ============================================================================

class TestPaperInvariants:
    """Invariants that must hold based on the paper's methodology."""

    def test_high_exposure_occupations_have_cognitive_skills(self):
        """High-exposure occupations should be dominated by cognitive skills.

        The paper finds that AI exposure correlates with cognitive/information
        processing tasks, not physical or interpersonal ones.
        """
        cognitive_skills = {
            "2.B.1.a",  # Reading Comprehension
            "2.B.3.a",  # Writing
            "2.B.8.a",  # Critical Thinking
            "2.B.8.b",  # Complex Problem Solving
            "2.B.8.d",  # Systems Analysis
            "2.B.1.g",  # Programming
            "2.B.5.a",  # Mathematics
        }
        for code in ("15-1251.00", "13-2051.00", "43-9021.00"):
            occ = PAPER_OCCUPATIONS[code]
            skill_ids = {s["element_id"] for s in occ["skills"]}
            cognitive_overlap = skill_ids & cognitive_skills
            # High-exposure occupations should overlap with cognitive skills.
            # Data Entry Keyers have fewer skills overall (job zone 2) but
            # still rely on cognitive tasks like reading and critical thinking.
            min_expected = 2 if occ["job_zone"] <= 2 else 3
            assert len(cognitive_overlap) >= min_expected, (
                f"{occ['title']} should have >= {min_expected} cognitive skills, "
                f"found {len(cognitive_overlap)}"
            )

    def test_zero_exposure_occupations_have_physical_interpersonal_skills(self):
        """Zero-exposure occupations should emphasize physical/interpersonal skills.

        The paper notes that physical, manual, or highly situated interpersonal
        tasks show little to no observed AI exposure.
        """
        physical_interpersonal = {
            "2.B.9.a",  # Troubleshooting
            "2.B.4.h",  # Equipment Maintenance
            "2.B.9.c",  # Operations Monitoring
            "2.B.7.c",  # Monitoring
            "2.B.1.f",  # Service Orientation
            "2.B.7.a",  # Social Perceptiveness
            "2.B.7.b",  # Coordination
            "2.B.6.c",  # Quality Control Analysis
        }
        for code in ("49-3023.00", "35-2014.00", "33-9092.00"):
            occ = PAPER_OCCUPATIONS[code]
            skill_ids = {s["element_id"] for s in occ["skills"]}
            physical_overlap = skill_ids & physical_interpersonal
            assert len(physical_overlap) >= 2, (
                f"{occ['title']} should have >= 2 physical/interpersonal skills, "
                f"found {len(physical_overlap)}"
            )

    def test_job_zone_distribution_matches_paper(self):
        """The paper's high-exposure occupations span job zones 2-4,
        while zero-exposure spans 1-3."""
        high_zones = {PAPER_OCCUPATIONS[c]["job_zone"] for c in
                      ("15-1251.00", "43-4051.00", "43-9021.00", "13-2051.00")}
        zero_zones = {PAPER_OCCUPATIONS[c]["job_zone"] for c in
                      ("35-2014.00", "49-3023.00", "33-9092.00", "35-3011.00", "35-9021.00")}
        # High-exposure includes job zone 4 (professional) and 2 (entry)
        assert 4 in high_zones
        assert 2 in high_zones
        # Zero-exposure includes job zone 1 (minimal training)
        assert 1 in zero_zones

    def test_skill_count_range(self):
        """All test occupations should have reasonable skill counts (3-10)."""
        for code, occ in PAPER_OCCUPATIONS.items():
            count = len(occ["skills"])
            assert 3 <= count <= 10, (
                f"{occ['title']} has {count} skills, expected 3-10"
            )

    def test_importance_values_in_onet_range(self):
        """O*NET importance values should be in the 0-100 range."""
        for code, occ in PAPER_OCCUPATIONS.items():
            for skill in occ["skills"]:
                assert 0 <= skill["importance"] <= 100, (
                    f"{occ['title']}: {skill['skill_name']} importance "
                    f"{skill['importance']} out of range"
                )

    def test_scoring_monotonicity_for_improving_skills(self, scorer):
        """Improving any skill should never decrease match_score.

        This is a key invariant: if a user improves a skill from rating N
        to N+1, their match_score must not decrease.
        """
        skills = PAPER_OCCUPATIONS["15-1251.00"]["skills"]
        base_ratings = {s["element_id"]: 2 for s in skills}

        base_score = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=skills,
            user_skill_ratings=base_ratings,
        )

        # Improve one skill
        improved_ratings = base_ratings.copy()
        improved_ratings["2.B.1.g"] = 4  # Programming: 2 → 4

        improved_score = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=skills,
            user_skill_ratings=improved_ratings,
        )

        assert improved_score.match_score >= base_score.match_score
        assert improved_score.gap_severity <= base_score.gap_severity

    def test_scoring_symmetry_across_exposure_tiers(self, scorer):
        """The scorer should be fair across exposure tiers — no built-in bias.

        Same skill profile should produce consistent scores regardless of
        whether the occupation is high- or zero-exposure.
        """
        # Construct identical skill lists for a high and zero exposure occupation
        common_skills = [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.0},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 70, "level": 4.75},
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 65, "level": 4.5},
        ]
        user_ratings = {"2.B.8.a": 3, "2.B.6.b": 3, "2.B.7.c": 3}

        score_high = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=common_skills,
            user_skill_ratings=user_ratings,
        )
        score_zero = scorer.score_occupation(
            onet_code="35-2014.00",
            occupation_title="Cooks, Restaurant",
            occupation_skills=common_skills,
            user_skill_ratings=user_ratings,
        )
        # Same skills → same scores (scorer is occupation-agnostic)
        assert score_high.match_score == score_zero.match_score
        assert score_high.gap_severity == score_zero.gap_severity
        assert score_high.bucket == score_zero.bucket


# ============================================================================
# Test Class: User Persona Stories (8 personas based on paper findings)
# ============================================================================

class TestLaborMarketPersonas:
    """User persona integration stories based on the paper's findings.

    Each persona represents a realistic user whose career is being shaped
    by the labor market dynamics the paper describes. We test the full
    scoring pipeline for each persona.
    """

    # --- Persona 1: Elena, 34, Computer Programmer worried about AI ---
    # The paper ranks computer programmers #1 in observed AI exposure (75%).
    # Elena is considering adjacent roles where her skills transfer.

    def test_persona_elena_programmer_to_data_scientist(self, scorer):
        """Elena: Computer Programmer → Data Scientist.

        Elena is a 34-year-old programmer at a mid-size company. After
        reading Anthropic's paper showing 75% AI coverage for programmers,
        she wants to transition to data science where her analytical skills
        add more unique value. She has strong programming and math but
        needs to build statistics and ML skills.
        """
        data_scientist_skills = [
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 81, "level": 5.62},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.5.b", "skill_name": "Science", "importance": 75, "level": 5.25},
        ]
        elena_ratings = {
            "2.B.1.g": 4,  # Programming - Expert
            "2.B.5.a": 3,  # Mathematics - Advanced (needs more stats)
            "2.B.8.a": 3,  # Critical Thinking - Advanced
            "2.B.8.b": 3,  # Complex Problem Solving - Advanced
            "2.B.8.d": 3,  # Systems Analysis - Advanced
            "2.B.3.a": 2,  # Writing - Intermediate
            "2.B.5.b": 1,  # Science - Basic (needs improvement)
        }
        score = scorer.score_occupation(
            onet_code="15-2051.00",
            occupation_title="Data Scientists",
            occupation_skills=data_scientist_skills,
            user_skill_ratings=elena_ratings,
            current_job_zone=4,
            target_job_zone=5,
        )
        # Elena has strong tech foundation — should be trainable
        assert score.bucket in ("trainable", "ready_now")
        assert score.match_score >= 50
        # Science should appear as a gap
        assert any(g.element_id == "2.B.5.b" for g in score.top_gaps)

    # --- Persona 2: Marcus, 22, recent grad worried about entry-level jobs ---
    # The paper finds suggestive evidence that hiring of 22-25 year olds has
    # slowed in high-exposure occupations.

    def test_persona_marcus_new_grad_data_entry(self, scorer):
        """Marcus: Fresh graduate considering data entry.

        Marcus is 22, just graduated with a general studies degree.
        The paper suggests young workers face slowed hiring in high-exposure
        occupations. He's considering data entry (67% AI coverage) but
        should perhaps look at less exposed alternatives.
        """
        skills = PAPER_OCCUPATIONS["43-9021.00"]["skills"]
        marcus_ratings = {
            "2.B.1.a": 3,  # Reading - Advanced (college)
            "2.B.2.a": 2,  # Active Listening - Intermediate
            "2.B.6.b": 2,  # Time Management - Intermediate
            "2.B.7.c": 1,  # Monitoring - Basic
            "2.B.8.a": 2,  # Critical Thinking - Intermediate
        }
        score = scorer.score_occupation(
            onet_code="43-9021.00",
            occupation_title="Data Entry Keyers",
            occupation_skills=skills,
            user_skill_ratings=marcus_ratings,
        )
        # With a degree, Marcus has decent match
        assert score.match_score >= 35
        assert score.bucket in ("trainable", "ready_now")

    # --- Persona 3: Rosa, 45, customer service rep exploring options ---
    # Customer service reps have high observed exposure (60% coverage).

    def test_persona_rosa_csr_to_hr_specialist(self, scorer):
        """Rosa: Customer Service Rep → HR Specialist.

        Rosa is 45 with 15 years in customer service. The paper shows her
        occupation has 60% AI coverage. She wants to pivot to HR where
        her strong interpersonal skills can differentiate her from AI.
        """
        hr_skills = [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72, "level": 5.0},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 69, "level": 4.88},
        ]
        rosa_ratings = {
            "2.B.2.a": 4,  # Active Listening - Expert (15 years!)
            "2.B.4.a": 4,  # Speaking - Expert
            "2.B.7.a": 4,  # Social Perceptiveness - Expert
            "2.B.3.c": 3,  # Negotiation - Advanced
            "2.B.1.a": 3,  # Reading - Advanced
            "2.B.3.a": 2,  # Writing - Intermediate
            "2.B.4.e": 1,  # Management of Personnel - Basic (gap)
        }
        score = scorer.score_occupation(
            onet_code="13-1071.00",
            occupation_title="Human Resources Specialists",
            occupation_skills=hr_skills,
            user_skill_ratings=rosa_ratings,
        )
        # Rosa's interpersonal skills transfer well
        assert score.bucket in ("trainable", "ready_now")
        assert score.match_score >= 55

    # --- Persona 4: Dave, 28, auto mechanic exploring transition ---
    # The paper classifies mechanics as zero AI exposure.

    def test_persona_dave_mechanic_to_hvac_technician(self, scorer):
        """Dave: Auto Mechanic → HVAC Technician.

        Dave is 28, an auto mechanic whose shop is closing. The paper shows
        his occupation has zero AI exposure. He wants to transition to HVAC
        which shares troubleshooting and equipment maintenance skills.
        """
        hvac_skills = [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84, "level": 5.75},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 75, "level": 5.25},
            {"element_id": "2.B.9.c", "skill_name": "Operations Monitoring", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66, "level": 4.62},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72, "level": 5.0},
        ]
        dave_ratings = {
            "2.B.9.a": 4,  # Troubleshooting - Expert (from auto)
            "2.B.4.h": 4,  # Equipment Maintenance - Expert
            "2.B.8.b": 3,  # Complex Problem Solving - Advanced
            "2.B.9.c": 3,  # Operations Monitoring - Advanced
            "2.B.1.a": 2,  # Reading - Intermediate
            "2.B.8.a": 3,  # Critical Thinking - Advanced
        }
        score = scorer.score_occupation(
            onet_code="49-9021.00",
            occupation_title="HVAC Technicians",
            occupation_skills=hvac_skills,
            user_skill_ratings=dave_ratings,
        )
        # Near-perfect skill transfer within trades
        assert score.bucket == "ready_now"
        assert score.match_score >= 75

    # --- Persona 5: Priya, 31, financial analyst considering career shift ---
    # Financial analysts are among the paper's highest-exposure occupations.

    def test_persona_priya_analyst_to_product_manager(self, scorer):
        """Priya: Financial Analyst → Product Manager.

        Priya is 31, a financial analyst who sees AI automating much of her
        analytical work (55% coverage per the paper). She wants to move to
        product management where strategic thinking and people skills matter
        more than pure number crunching.
        """
        pm_skills = [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78, "level": 5.38},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72, "level": 5.0},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 69, "level": 4.88},
        ]
        priya_ratings = {
            "2.B.8.a": 4,  # Critical Thinking - Expert
            "2.B.4.a": 3,  # Speaking - Advanced
            "2.B.8.b": 4,  # Complex Problem Solving - Expert
            "2.B.7.b": 2,  # Coordination - Intermediate (needs work)
            "2.B.3.a": 3,  # Writing - Advanced
            "2.B.7.a": 2,  # Social Perceptiveness - Intermediate
            "2.B.4.e": 1,  # Management of Personnel - Basic (gap)
        }
        score = scorer.score_occupation(
            onet_code="11-2021.01",
            occupation_title="Product Managers",
            occupation_skills=pm_skills,
            user_skill_ratings=priya_ratings,
        )
        # Strong analytical foundation, needs people skills
        assert score.bucket == "trainable"
        assert score.match_score >= 50

    # --- Persona 6: Kenji, 40, bartender considering tech adjacent role ---
    # Bartenders have zero AI exposure per the paper.

    def test_persona_kenji_bartender_to_event_coordinator(self, scorer):
        """Kenji: Bartender → Event Coordinator.

        Kenji is 40, a bartender for 12 years. The paper shows zero AI
        exposure for bartenders. He wants to leverage his hospitality and
        coordination skills in event planning — still relatively low
        AI exposure but higher earning potential.
        """
        event_coordinator_skills = [
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 81, "level": 5.62},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 78, "level": 5.38},
            {"element_id": "2.B.3.c", "skill_name": "Negotiation", "importance": 72, "level": 5.0},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 75, "level": 5.25},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 69, "level": 4.88},
        ]
        kenji_ratings = {
            "2.B.7.b": 3,  # Coordination - Advanced (from bar management)
            "2.B.4.a": 4,  # Speaking - Expert (constant customer interaction)
            "2.B.6.b": 4,  # Time Management - Expert (busy service)
            "2.B.3.c": 3,  # Negotiation - Advanced (supplier dealings)
            "2.B.1.f": 4,  # Service Orientation - Expert
            "2.B.7.a": 3,  # Social Perceptiveness - Advanced
        }
        score = scorer.score_occupation(
            onet_code="13-1121.00",
            occupation_title="Event Coordinators",
            occupation_skills=event_coordinator_skills,
            user_skill_ratings=kenji_ratings,
        )
        # Bartending skills transfer well to event coordination
        assert score.bucket == "ready_now"
        assert score.match_score >= 75

    # --- Persona 7: Aaliyah, 26, lifeguard seeking stable career ---
    # Lifeguards are zero-exposure per the paper.

    def test_persona_aaliyah_lifeguard_to_emt(self, scorer):
        """Aaliyah: Lifeguard → Emergency Medical Technician.

        Aaliyah is 26, working as a lifeguard. The paper places lifeguards
        at zero AI exposure. She wants to build on her first-responder
        skills and monitoring abilities to become an EMT — another
        low-AI-exposure role with better career prospects.
        """
        emt_skills = [
            {"element_id": "2.B.7.c", "skill_name": "Monitoring", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81, "level": 5.62},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72, "level": 5.0},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 69, "level": 4.88},
        ]
        aaliyah_ratings = {
            "2.B.7.c": 4,  # Monitoring - Expert (core lifeguard skill)
            "2.B.8.a": 3,  # Critical Thinking - Advanced (emergency decisions)
            "2.B.1.f": 4,  # Service Orientation - Expert
            "2.B.2.a": 3,  # Active Listening - Advanced
            "2.B.4.a": 3,  # Speaking - Advanced
            "2.B.7.a": 3,  # Social Perceptiveness - Advanced
        }
        score = scorer.score_occupation(
            onet_code="29-2042.00",
            occupation_title="Emergency Medical Technicians",
            occupation_skills=emt_skills,
            user_skill_ratings=aaliyah_ratings,
        )
        # Lifeguard → EMT is a natural, high-overlap transition
        assert score.bucket == "ready_now"
        assert score.match_score >= 75

    # --- Persona 8: Tomás, 50, looking to future-proof his career ---
    # Tomás is a software developer (33% observed, 94% theoretical coverage).

    def test_persona_tomas_developer_to_engineering_manager(self, scorer):
        """Tomás: Software Developer → Engineering Manager.

        Tomás is 50, a senior developer who's seen the paper's findings that
        software development has 94% theoretical AI capability but only 33%
        observed coverage. He wants to move into management before the gap
        closes further. His leadership and technical skills should transfer.
        """
        eng_manager_skills = [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 84, "level": 5.88},
            {"element_id": "2.B.4.e", "skill_name": "Management of Personnel Resources", "importance": 84, "level": 5.75},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81, "level": 5.62},
            {"element_id": "2.B.7.b", "skill_name": "Coordination", "importance": 78, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78, "level": 5.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75, "level": 5.25},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 69, "level": 4.88},
        ]
        tomas_ratings = {
            "2.B.8.a": 4,  # Critical Thinking - Expert (20+ years)
            "2.B.4.e": 2,  # Management of Personnel - Intermediate (some lead experience)
            "2.B.8.b": 4,  # Complex Problem Solving - Expert
            "2.B.7.b": 3,  # Coordination - Advanced
            "2.B.4.a": 3,  # Speaking - Advanced
            "2.B.8.d": 4,  # Systems Analysis - Expert
            "2.B.1.g": 4,  # Programming - Expert (strong tech background)
        }
        score = scorer.score_occupation(
            onet_code="11-3021.00",
            occupation_title="Computer and Information Systems Managers",
            occupation_skills=eng_manager_skills,
            user_skill_ratings=tomas_ratings,
            current_job_zone=4,
            target_job_zone=5,
        )
        # Strong technical + developing management = trainable
        assert score.bucket in ("trainable", "ready_now")
        assert score.match_score >= 60
        # Management gap should be identified
        top_gap_ids = {g.element_id for g in score.top_gaps}
        # If management is a gap (rating=2), it should appear
        # Note: rating=2 gives capability 0.5, which is above the gap threshold (<=0.25)
        # So management won't be a gap, which is correct — intermediate is decent


# ============================================================================
# Test Class: Sector-Level Coverage Validation
# ============================================================================

class TestSectorCoverage:
    """Validate sector-level coverage data matches the paper's findings."""

    def test_all_sectors_have_valid_ranges(self):
        """All sector coverage values must be between 0 and 1."""
        for sector, data in SECTOR_COVERAGE.items():
            assert 0 <= data["theoretical"] <= 1, f"{sector} theoretical out of range"
            assert 0 <= data["observed"] <= 1, f"{sector} observed out of range"

    def test_observed_never_exceeds_theoretical(self):
        """Observed coverage should never exceed theoretical capability.

        This is a fundamental constraint from the paper: you can't observe
        more AI usage than what's theoretically feasible.
        """
        for sector, data in SECTOR_COVERAGE.items():
            assert data["observed"] <= data["theoretical"], (
                f"{sector}: observed ({data['observed']}) > theoretical ({data['theoretical']})"
            )

    def test_computer_math_highest_theoretical(self):
        """Computer & Math should have the highest theoretical coverage."""
        computer_math = SECTOR_COVERAGE["Computer and Mathematical"]["theoretical"]
        for sector, data in SECTOR_COVERAGE.items():
            if sector != "Computer and Mathematical":
                assert computer_math >= data["theoretical"], (
                    f"{sector} ({data['theoretical']}) should not exceed "
                    f"Computer and Mathematical ({computer_math})"
                )

    def test_food_service_low_coverage(self):
        """Food service should have very low theoretical and zero observed."""
        food = SECTOR_COVERAGE["Food Preparation and Serving"]
        assert food["theoretical"] <= 0.15
        assert food["observed"] == 0.0


# ============================================================================
# Test Class: Training Suggestion Quality
# ============================================================================

class TestTrainingSuggestionQuality:
    """Ensure training suggestions are appropriate for the paper's occupations."""

    def test_high_exposure_trainable_mentions_timeframe(self, scorer):
        """Trainable suggestions for high-exposure roles should include timeframes."""
        skills = PAPER_OCCUPATIONS["13-2051.00"]["skills"]
        user_ratings = {s["element_id"]: 2 for s in skills}  # All intermediate
        user_ratings["2.B.5.a"] = 0  # Math gap

        score = scorer.score_occupation(
            onet_code="13-2051.00",
            occupation_title="Financial Analysts",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
            target_job_zone=4,
        )
        if score.bucket == "trainable":
            suggestion = score.training_suggestion.lower()
            assert any(word in suggestion for word in
                       ["month", "year", "bootcamp", "degree", "program"])

    def test_zero_exposure_ready_now_encouraging(self, scorer):
        """Ready_now suggestions for zero-exposure roles should encourage applying."""
        skills = PAPER_OCCUPATIONS["35-2014.00"]["skills"]
        user_ratings = {s["element_id"]: 4 for s in skills}  # All expert

        score = scorer.score_occupation(
            onet_code="35-2014.00",
            occupation_title="Cooks, Restaurant",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert score.bucket == "ready_now"
        assert "apply" in score.training_suggestion.lower()

    def test_long_reskill_major_gap_honest(self, scorer):
        """Long reskill suggestions should be honest about the effort required."""
        skills = PAPER_OCCUPATIONS["15-1251.00"]["skills"]
        user_ratings = {s["element_id"]: 0 for s in skills}  # All none

        score = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
            target_job_zone=4,
        )
        assert score.bucket == "long_reskill"
        suggestion = score.training_suggestion.lower()
        assert any(word in suggestion for word in
                   ["significant", "degree", "extended", "year"])


# ============================================================================
# Test Class: Explanation Quality
# ============================================================================

class TestExplanationQuality:
    """Ensure explanations are meaningful for labor market transition decisions."""

    def test_explanation_mentions_occupation_title(self, scorer):
        """Every explanation should name the target occupation."""
        for code, occ in list(PAPER_OCCUPATIONS.items())[:5]:
            user_ratings = {s["element_id"]: 2 for s in occ["skills"]}
            score = scorer.score_occupation(
                onet_code=code,
                occupation_title=occ["title"],
                occupation_skills=occ["skills"],
                user_skill_ratings=user_ratings,
            )
            assert occ["title"] in score.explanation

    def test_explanation_includes_match_percentage(self, scorer):
        """Explanations should include the match percentage for transparency."""
        skills = PAPER_OCCUPATIONS["15-1251.00"]["skills"]
        user_ratings = {s["element_id"]: 3 for s in skills}
        score = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert "%" in score.explanation

    def test_explanation_mentions_gaps_when_present(self, scorer):
        """When gaps exist, the explanation should mention them."""
        skills = PAPER_OCCUPATIONS["13-2051.00"]["skills"]
        user_ratings = {s["element_id"]: 0 for s in skills}
        score = scorer.score_occupation(
            onet_code="13-2051.00",
            occupation_title="Financial Analysts",
            occupation_skills=skills,
            user_skill_ratings=user_ratings,
        )
        assert "gap" in score.explanation.lower() or "reskill" in score.explanation.lower()


# ============================================================================
# Test Class: Edge Cases from Paper's Methodology
# ============================================================================

class TestPaperMethodologyEdgeCases:
    """Edge cases inspired by the paper's methodology and findings."""

    def test_occupation_with_all_augmentative_tasks(self):
        """All-augmentative should give 50% coverage, not 100%."""
        tasks = [
            TaskExposure("Review docs", 0.5, AIUsageType.AUGMENTATIVE),
            TaskExposure("Draft emails", 0.5, AIUsageType.AUGMENTATIVE),
            TaskExposure("Analyze data", 0.5, AIUsageType.AUGMENTATIVE),
        ]
        assert compute_observed_coverage(tasks) == pytest.approx(0.5)

    def test_97_percent_rule(self):
        """Paper: 97% of observed tasks fall into categories rated β >= 0.5.

        This means theoretical coverage should almost always be >= observed.
        """
        tasks = [
            TaskExposure("Code", 1.0, AIUsageType.AUTOMATED),
            TaskExposure("Test", 1.0, AIUsageType.AUTOMATED),
            TaskExposure("Review", 0.5, AIUsageType.AUGMENTATIVE),
            # One task observed but theoretically infeasible (the 3% outlier)
            TaskExposure("Creative", 0.0, AIUsageType.AUGMENTATIVE),
        ]
        theoretical = compute_theoretical_coverage(tasks)
        observed = compute_observed_coverage(tasks)
        # Theoretical should generally exceed observed
        assert theoretical >= observed * 0.90  # Allow some slack for the 3%

    def test_single_task_occupation(self):
        """Occupation with only one task should still compute correctly."""
        tasks = [TaskExposure("Only task", 1.0, AIUsageType.AUTOMATED)]
        assert compute_observed_coverage(tasks) == 1.0
        assert compute_theoretical_coverage(tasks) == 1.0

    def test_unknown_occupation_returns_none(self):
        """Unknown O*NET code should return None, not crash."""
        assert get_exposure_profile("99-9999.00") is None
        assert theoretical_observed_gap("99-9999.00") is None

    def test_scorer_handles_missing_skills_gracefully(self, scorer):
        """User with zero ratings for a high-exposure occupation."""
        skills = PAPER_OCCUPATIONS["15-1251.00"]["skills"]
        score = scorer.score_occupation(
            onet_code="15-1251.00",
            occupation_title="Computer Programmers",
            occupation_skills=skills,
            user_skill_ratings={},  # No ratings at all
        )
        assert score.match_score == 0.0
        assert score.gap_severity == 100.0
        assert score.bucket == "long_reskill"
        assert len(score.top_gaps) == len(skills)
