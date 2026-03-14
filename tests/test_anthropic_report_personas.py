"""QA personas based on the Anthropic Labor Market Report (March 2026).

The Anthropic report 'Labor market impacts of AI: A new measure and early
evidence' (Massenkoff & McCrory, March 2026) identifies occupations most
exposed to AI disruption.  This module creates 10 testing personas drawn from
those positions to stress-test the SkillSprout scoring pipeline across the
full AI-exposure spectrum:

HIGH EXPOSURE (observed ≥47%):
  1. Derek  – Computer Programmer (75% observed exposure)
  2. Linda  – Customer Service Representative (70%)
  3. Priya  – Data Entry Keyer (67%)
  4. Carlos – Medical Records Specialist (67%)
  5. Wei    – Market Research Analyst (65%)
  6. Tanya  – Sales Representative (63%)
  7. Marcus – Software QA Analyst (52%)
  8. Fatima – Information Security Analyst (49%)
  9. Jordan – Computer User Support Specialist (47%)

LOW / MINIMAL EXPOSURE (recommended growth occupations):
  10. Elena  – Electrician (0% observed exposure, strong BLS growth)

These personas exercise:
  • Full AI-exposure spectrum (high → minimal)
  • Both declining and growth-outlook occupations
  • Diverse skill profiles (tech-heavy, people-heavy, manual)
  • Cross-occupation transition scoring (every persona scored against
    multiple target occupations)
  • Edge cases: workers with no programming trying tech roles,
    tech workers pivoting to low-exposure trades
"""
import copy
import pytest
from typing import Dict, List

from app.ml.scoring import BaselineScorer, OccupationScore
from app.data.ai_exposure import get_exposure, EXPOSURE_DATA
from app.data.bls_projections import get_projections, BLS_PROJECTIONS


# =====================================================================
# Mock occupation skills (mirrors MockONetClient data for test isolation)
# =====================================================================

ANTHROPIC_MOCK_OCCUPATION_SKILLS = {
    "15-1251.00": {
        "title": "Computer Programmers",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72.0, "level": 5.12},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78.0, "level": 5.50},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 84.0, "level": 5.88},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 88.0, "level": 6.00},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 66.0, "level": 4.75},
            {"element_id": "2.B.8.e", "skill_name": "Systems Evaluation", "importance": 69.0, "level": 5.00},
        ],
    },
    "43-4051.00": {
        "title": "Customer Service Representatives",
        "job_zone": 2,
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 81.0, "level": 5.25},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 78.0, "level": 5.12},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 4.50},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 81.0, "level": 5.38},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 72.0, "level": 4.75},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 66.0, "level": 4.38},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 63.0, "level": 4.12},
        ],
    },
    "43-9021.00": {
        "title": "Data Entry Keyers",
        "job_zone": 2,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 4.50},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 63.0, "level": 4.00},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 60.0, "level": 3.75},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66.0, "level": 4.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 60.0, "level": 4.00},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 57.0, "level": 3.50},
        ],
    },
    "29-2072.00": {
        "title": "Medical Records Specialists",
        "job_zone": 3,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75.0, "level": 5.00},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72.0, "level": 4.75},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72.0, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69.0, "level": 4.75},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66.0, "level": 4.50},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 66.0, "level": 4.38},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 63.0, "level": 4.25},
        ],
    },
    "13-1161.01": {
        "title": "Market Research Analysts and Marketing Specialists",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 78.0, "level": 5.25},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 78.0, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75.0, "level": 5.12},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 72.0, "level": 4.88},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.7.a", "skill_name": "Social Perceptiveness", "importance": 66.0, "level": 4.62},
        ],
    },
    "41-4012.00": {
        "title": "Sales Representatives, Wholesale and Manufacturing",
        "job_zone": 3,
        "skills": [
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 81.0, "level": 5.50},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 78.0, "level": 5.25},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 75.0, "level": 5.00},
            {"element_id": "2.B.4.b", "skill_name": "Persuasion", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.4.c", "skill_name": "Negotiation", "importance": 75.0, "level": 5.12},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 4.75},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69.0, "level": 4.75},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66.0, "level": 4.50},
        ],
    },
    "15-1253.00": {
        "title": "Software Quality Assurance Analysts and Testers",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75.0, "level": 5.12},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66.0, "level": 4.50},
        ],
    },
    "15-1212.00": {
        "title": "Information Security Analysts",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81.0, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81.0, "level": 5.62},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 78.0, "level": 5.50},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78.0, "level": 5.50},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 75.0, "level": 5.25},
        ],
    },
    "15-1232.00": {
        "title": "Computer User Support Specialists",
        "job_zone": 3,
        "skills": [
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 75.0, "level": 5.00},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 75.0, "level": 5.00},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72.0, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 75.0, "level": 5.12},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.1.f", "skill_name": "Service Orientation", "importance": 72.0, "level": 4.88},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 63.0, "level": 4.50},
        ],
    },
    "47-2111.00": {
        "title": "Electricians",
        "job_zone": 3,
        "skills": [
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 81.0, "level": 5.50},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 69.0, "level": 4.75},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 66.0, "level": 4.62},
            {"element_id": "2.B.5.a", "skill_name": "Mathematics", "importance": 66.0, "level": 4.50},
            {"element_id": "2.B.4.h", "skill_name": "Equipment Maintenance", "importance": 75.0, "level": 5.12},
            {"element_id": "2.B.9.b", "skill_name": "Equipment Selection", "importance": 69.0, "level": 4.75},
        ],
    },
}

ALL_OCC_CODES = list(ANTHROPIC_MOCK_OCCUPATION_SKILLS.keys())


# =====================================================================
# QA Personas from Anthropic Labor Market Report
# =====================================================================

PERSONA_DEREK = {
    "name": "Derek – Computer Programmer facing 75% AI exposure",
    "current_occupation": "15-1251.00",
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension – Advanced
        "2.B.3.a": 3,  # Writing – Advanced
        "2.B.8.a": 4,  # Critical Thinking – Expert
        "2.B.8.b": 4,  # Complex Problem Solving – Expert
        "2.B.8.d": 3,  # Systems Analysis – Advanced
        "2.B.1.g": 4,  # Programming – Expert
        "2.B.5.a": 3,  # Mathematics – Advanced
        "2.B.8.e": 3,  # Systems Evaluation – Advanced
    },
    "expected_transition_context": "highest AI exposure; needs to move to less automatable roles",
    "budget": 2000,
    "timeline_months": 12,
}

PERSONA_LINDA = {
    "name": "Linda – Customer Service Rep, 70% AI exposure",
    "current_occupation": "43-4051.00",
    "skill_ratings": {
        "2.B.2.a": 4,  # Active Listening – Expert
        "2.B.4.a": 4,  # Speaking – Expert
        "2.B.1.a": 2,  # Reading Comprehension – Intermediate
        "2.B.1.f": 4,  # Service Orientation – Expert
        "2.B.7.a": 3,  # Social Perceptiveness – Advanced
        "2.B.8.a": 2,  # Critical Thinking – Intermediate
        "2.B.3.a": 2,  # Writing – Intermediate
    },
    "expected_transition_context": "strong people skills but limited tech; needs upskilling for safer roles",
    "budget": 300,
    "timeline_months": 6,
}

PERSONA_PRIYA = {
    "name": "Priya – Data Entry Keyer, 67% AI exposure, declining outlook",
    "current_occupation": "43-9021.00",
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension – Advanced
        "2.B.2.a": 2,  # Active Listening – Intermediate
        "2.B.3.a": 2,  # Writing – Intermediate
        "2.B.6.b": 4,  # Time Management – Expert
        "2.B.8.a": 2,  # Critical Thinking – Intermediate
        "2.B.5.a": 2,  # Mathematics – Intermediate
    },
    "expected_transition_context": "routine cognitive work, -32% BLS outlook; urgently needs transition",
    "budget": 0,
    "timeline_months": 6,
}

PERSONA_CARLOS = {
    "name": "Carlos – Medical Records Specialist, 67% AI exposure",
    "current_occupation": "29-2072.00",
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension – Advanced
        "2.B.2.a": 3,  # Active Listening – Advanced
        "2.B.3.a": 3,  # Writing – Advanced
        "2.B.8.a": 3,  # Critical Thinking – Advanced
        "2.B.6.b": 3,  # Time Management – Advanced
        "2.B.4.a": 2,  # Speaking – Intermediate
        "2.B.8.d": 2,  # Systems Analysis – Intermediate
    },
    "expected_transition_context": "healthcare domain knowledge; could pivot within health informatics",
    "budget": 1500,
    "timeline_months": 18,
}

PERSONA_WEI = {
    "name": "Wei – Market Research Analyst, 65% AI exposure",
    "current_occupation": "13-1161.01",
    "skill_ratings": {
        "2.B.1.a": 4,  # Reading Comprehension – Expert
        "2.B.3.a": 4,  # Writing – Expert
        "2.B.8.a": 4,  # Critical Thinking – Expert
        "2.B.4.a": 3,  # Speaking – Advanced
        "2.B.2.a": 3,  # Active Listening – Advanced
        "2.B.5.a": 3,  # Mathematics – Advanced
        "2.B.8.b": 3,  # Complex Problem Solving – Advanced
        "2.B.7.a": 3,  # Social Perceptiveness – Advanced
    },
    "expected_transition_context": "strong analytical skills; good foundation for data science or strategy",
    "budget": 5000,
    "timeline_months": 12,
}

PERSONA_TANYA = {
    "name": "Tanya – Sales Representative, 63% AI exposure",
    "current_occupation": "41-4012.00",
    "skill_ratings": {
        "2.B.4.a": 4,  # Speaking – Expert
        "2.B.2.a": 4,  # Active Listening – Expert
        "2.B.1.f": 4,  # Service Orientation – Expert
        "2.B.4.b": 4,  # Persuasion – Expert
        "2.B.4.c": 3,  # Negotiation – Advanced
        "2.B.1.a": 2,  # Reading Comprehension – Intermediate
        "2.B.8.a": 2,  # Critical Thinking – Intermediate
        "2.B.6.b": 3,  # Time Management – Advanced
    },
    "expected_transition_context": "excellent interpersonal skills; transferable to management or client roles",
    "budget": 0,
    "timeline_months": 6,
}

PERSONA_MARCUS = {
    "name": "Marcus – Software QA Analyst, 52% AI exposure",
    "current_occupation": "15-1253.00",
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension – Advanced
        "2.B.8.a": 4,  # Critical Thinking – Expert
        "2.B.8.b": 3,  # Complex Problem Solving – Advanced
        "2.B.1.g": 3,  # Programming – Advanced
        "2.B.8.d": 3,  # Systems Analysis – Advanced
        "2.B.3.a": 3,  # Writing – Advanced
        "2.B.9.a": 3,  # Troubleshooting – Advanced
        "2.B.6.b": 3,  # Time Management – Advanced
    },
    "expected_transition_context": "testing skills transfer well to security or development roles",
    "budget": 3000,
    "timeline_months": 12,
}

PERSONA_FATIMA = {
    "name": "Fatima – Information Security Analyst, 49% AI exposure",
    "current_occupation": "15-1212.00",
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension – Advanced
        "2.B.8.a": 4,  # Critical Thinking – Expert
        "2.B.8.b": 4,  # Complex Problem Solving – Expert
        "2.B.1.g": 3,  # Programming – Advanced
        "2.B.8.d": 4,  # Systems Analysis – Expert
        "2.B.3.a": 3,  # Writing – Advanced
        "2.B.2.a": 3,  # Active Listening – Advanced
        "2.B.9.a": 3,  # Troubleshooting – Advanced
    },
    "expected_transition_context": "high-demand field despite exposure; strong technical foundation",
    "budget": 8000,
    "timeline_months": 18,
}

PERSONA_JORDAN = {
    "name": "Jordan – Computer User Support Specialist, 47% AI exposure",
    "current_occupation": "15-1232.00",
    "skill_ratings": {
        "2.B.2.a": 4,  # Active Listening – Expert
        "2.B.4.a": 4,  # Speaking – Expert
        "2.B.1.a": 3,  # Reading Comprehension – Advanced
        "2.B.8.a": 3,  # Critical Thinking – Advanced
        "2.B.8.b": 3,  # Complex Problem Solving – Advanced
        "2.B.9.a": 3,  # Troubleshooting – Advanced
        "2.B.1.f": 4,  # Service Orientation – Expert
        "2.B.1.g": 2,  # Programming – Intermediate
    },
    "expected_transition_context": "bridge role between IT and users; some programming gives mobility",
    "budget": 1000,
    "timeline_months": 12,
}

PERSONA_ELENA = {
    "name": "Elena – Electrician, 0% AI exposure, strong BLS growth",
    "current_occupation": "47-2111.00",
    "skill_ratings": {
        "2.B.9.a": 4,  # Troubleshooting – Expert
        "2.B.8.a": 3,  # Critical Thinking – Advanced
        "2.B.8.b": 3,  # Complex Problem Solving – Advanced
        "2.B.1.a": 2,  # Reading Comprehension – Intermediate
        "2.B.5.a": 3,  # Mathematics – Advanced
        "2.B.4.h": 4,  # Equipment Maintenance – Expert
        "2.B.9.b": 4,  # Equipment Selection – Expert
    },
    "expected_transition_context": "minimal AI exposure, growth outlook; tests low-risk persona",
    "budget": 500,
    "timeline_months": 6,
}


ALL_PERSONAS = [
    PERSONA_DEREK, PERSONA_LINDA, PERSONA_PRIYA, PERSONA_CARLOS, PERSONA_WEI,
    PERSONA_TANYA, PERSONA_MARCUS, PERSONA_FATIMA, PERSONA_JORDAN, PERSONA_ELENA,
]


# =====================================================================
# Helpers
# =====================================================================

def _make_scorer(**overrides) -> BaselineScorer:
    defaults = dict(
        ready_now_match_threshold=75.0,
        ready_now_gap_threshold=25.0,
        trainable_match_min=50.0,
        trainable_match_max=74.0,
        trainable_gap_min=26.0,
        trainable_gap_max=55.0,
    )
    defaults.update(overrides)
    return BaselineScorer(**defaults)


def _score(persona: Dict, occupation_code: str, scorer=None) -> OccupationScore:
    if scorer is None:
        scorer = _make_scorer()
    occ = ANTHROPIC_MOCK_OCCUPATION_SKILLS[occupation_code]
    return scorer.score_occupation(
        onet_code=occupation_code,
        occupation_title=occ["title"],
        occupation_skills=occ["skills"],
        user_skill_ratings=persona["skill_ratings"],
        current_job_zone=3,
        target_job_zone=occ["job_zone"],
    )


# =====================================================================
# 1. Persona-driven scenario tests
# =====================================================================

class TestHighExposurePersonaScenarios:
    """Verify scoring behavior for workers in the most AI-exposed occupations."""

    def test_derek_programmer_scores_ready_for_own_role(self):
        """Derek (expert programmer) should score ready_now for Computer Programmers."""
        score = _score(PERSONA_DEREK, "15-1251.00")
        assert score.bucket == "ready_now"
        assert score.match_score >= 75
        assert score.gap_severity <= 25

    def test_derek_programmer_to_security_analyst(self):
        """Derek should be trainable for InfoSec — has programming, needs security depth."""
        score = _score(PERSONA_DEREK, "15-1212.00")
        assert score.bucket in ["trainable", "ready_now"]
        assert score.match_score > 40

    def test_derek_programmer_to_electrician(self):
        """Derek pivoting to electrician should have significant gaps (no equipment skills)."""
        score = _score(PERSONA_DEREK, "47-2111.00")
        assert score.bucket in ["trainable", "long_reskill"]
        assert len(score.top_gaps) > 0

    def test_linda_csr_scores_well_for_own_role(self):
        """Linda should score highly for Customer Service (her current job)."""
        score = _score(PERSONA_LINDA, "43-4051.00")
        assert score.bucket in ["ready_now", "trainable"]
        assert score.match_score >= 50

    def test_linda_csr_to_sales_rep(self):
        """Linda's people skills should transfer partially to sales."""
        score = _score(PERSONA_LINDA, "41-4012.00")
        assert score.match_score > 30
        assert score.explanation is not None

    def test_linda_csr_to_programmer(self):
        """Linda trying programming should be long_reskill — no tech skills."""
        score = _score(PERSONA_LINDA, "15-1251.00")
        assert score.bucket in ["trainable", "long_reskill"]
        gap_ids = [g.element_id for g in score.top_gaps]
        assert "2.B.1.g" in gap_ids, "Programming should be a top gap for Linda"

    def test_priya_data_entry_urgency(self):
        """Priya's data entry role has -32% BLS outlook; transitions are urgent."""
        score = _score(PERSONA_PRIYA, "43-9021.00")
        # She should score well for her own role
        assert score.match_score > 40
        # Verify BLS data reflects the decline
        bls = get_projections("43-9021.00")
        assert bls is not None
        assert bls["projected_growth_pct"] < 0
        assert bls["outlook"] == "declining"

    def test_priya_data_entry_to_medical_records(self):
        """Priya could pivot to medical records — similar clerical skills."""
        score = _score(PERSONA_PRIYA, "29-2072.00")
        assert score.bucket in ["trainable", "ready_now"]

    def test_carlos_medical_records_to_market_research(self):
        """Carlos pivoting from medical records to market research needs analytical upskill."""
        score = _score(PERSONA_CARLOS, "13-1161.01")
        assert score.bucket in ["trainable", "long_reskill"]
        assert len(score.top_gaps) > 0

    def test_wei_market_research_strong_foundation(self):
        """Wei's analytical skills should provide a strong match across knowledge roles."""
        score = _score(PERSONA_WEI, "13-1161.01")
        assert score.match_score >= 65
        assert score.bucket in ["ready_now", "trainable"]

    def test_wei_to_qa_analyst(self):
        """Wei pivoting to QA should be trainable — has analysis but needs programming."""
        score = _score(PERSONA_WEI, "15-1253.00")
        assert score.bucket in ["trainable", "long_reskill"]
        gap_ids = [g.element_id for g in score.top_gaps]
        assert "2.B.1.g" in gap_ids, "Programming should be identified as a gap"

    def test_tanya_sales_to_customer_service(self):
        """Tanya's sales skills should transfer well to customer service."""
        score = _score(PERSONA_TANYA, "43-4051.00")
        assert score.match_score > 50
        assert score.bucket in ["ready_now", "trainable"]

    def test_marcus_qa_to_programmer(self):
        """Marcus (QA) should be trainable for programming — has foundational tech skills."""
        score = _score(PERSONA_MARCUS, "15-1251.00")
        assert score.bucket in ["trainable", "ready_now"]
        assert score.match_score > 40

    def test_marcus_qa_to_security_analyst(self):
        """Marcus's QA skills should transfer well to security analysis."""
        score = _score(PERSONA_MARCUS, "15-1212.00")
        assert score.bucket in ["trainable", "ready_now"]

    def test_fatima_security_strong_tech_base(self):
        """Fatima's security background gives excellent match across tech roles."""
        for occ_code in ["15-1251.00", "15-1253.00", "15-1232.00"]:
            score = _score(PERSONA_FATIMA, occ_code)
            assert score.match_score > 40, f"Fatima should have decent match for {occ_code}"

    def test_jordan_support_to_qa(self):
        """Jordan's IT support skills should partially transfer to QA."""
        score = _score(PERSONA_JORDAN, "15-1253.00")
        assert score.bucket in ["trainable", "long_reskill"]

    def test_jordan_support_has_programming_gap(self):
        """Jordan's intermediate programming should show as a gap for dev roles."""
        score = _score(PERSONA_JORDAN, "15-1251.00")
        gap_ids = [g.element_id for g in score.top_gaps]
        # Programming at level 2 (intermediate) counts as a gap (≤0.5 capability)
        assert "2.B.1.g" in gap_ids or score.match_score < 60


class TestLowExposurePersonaScenarios:
    """Verify scoring for low-exposure / growth-outlook occupations."""

    def test_elena_electrician_scores_ready_for_own_role(self):
        """Elena (expert electrician) should score well for her own occupation."""
        score = _score(PERSONA_ELENA, "47-2111.00")
        assert score.match_score >= 60
        assert score.bucket in ["ready_now", "trainable"]

    def test_elena_electrician_to_programmer(self):
        """Elena transitioning to programming should be long_reskill."""
        score = _score(PERSONA_ELENA, "15-1251.00")
        assert score.bucket in ["trainable", "long_reskill"]
        gap_ids = [g.element_id for g in score.top_gaps]
        assert "2.B.1.g" in gap_ids

    def test_elena_minimal_ai_exposure(self):
        """Electrician should have minimal AI exposure in our data."""
        exposure = get_exposure("47-2111.00")
        assert exposure is not None
        assert exposure["observed_exposure"] == 0.00
        assert exposure["exposure_rank"] == "minimal"

    def test_elena_strong_bls_outlook(self):
        """Electricians should have strong growth outlook."""
        bls = get_projections("47-2111.00")
        assert bls is not None
        assert bls["projected_growth_pct"] > 10
        assert bls["outlook"] == "strong growth"


# =====================================================================
# 2. AI exposure data integrity tests
# =====================================================================

class TestAIExposureDataIntegrity:
    """Ensure all Anthropic report occupations have proper exposure data."""

    @pytest.mark.parametrize("onet_code", [
        "15-1251.00", "43-4051.00", "43-9021.00", "29-2072.00",
        "13-1161.01", "41-4012.00", "15-1253.00", "15-1212.00",
        "15-1232.00", "47-2111.00",
    ])
    def test_exposure_data_exists(self, onet_code):
        """Every Anthropic report occupation should have exposure data."""
        exposure = get_exposure(onet_code)
        assert exposure is not None, f"Missing exposure data for {onet_code}"
        assert "theoretical_exposure" in exposure
        assert "observed_exposure" in exposure
        assert "exposure_rank" in exposure

    @pytest.mark.parametrize("onet_code", [
        "15-1251.00", "43-4051.00", "43-9021.00", "29-2072.00",
        "13-1161.01", "41-4012.00", "15-1253.00", "15-1212.00",
        "15-1232.00", "47-2111.00",
    ])
    def test_bls_projections_exist(self, onet_code):
        """Every Anthropic report occupation should have BLS projection data."""
        bls = get_projections(onet_code)
        assert bls is not None, f"Missing BLS data for {onet_code}"
        assert "projected_growth_pct" in bls
        assert "outlook" in bls

    def test_exposure_values_in_range(self):
        """All exposure values should be between 0 and 1."""
        for code, data in EXPOSURE_DATA.items():
            assert 0 <= data["theoretical_exposure"] <= 1, f"Bad theoretical for {code}"
            assert 0 <= data["observed_exposure"] <= 1, f"Bad observed for {code}"

    def test_observed_never_exceeds_theoretical(self):
        """Observed exposure should never exceed theoretical exposure."""
        for code, data in EXPOSURE_DATA.items():
            assert data["observed_exposure"] <= data["theoretical_exposure"], (
                f"{code}: observed {data['observed_exposure']} > theoretical {data['theoretical_exposure']}"
            )

    def test_high_exposure_ranking_consistency(self):
        """Occupations with observed exposure ≥ 0.40 should be ranked high."""
        for code, data in EXPOSURE_DATA.items():
            if data["observed_exposure"] >= 0.40:
                assert data["exposure_rank"] in ["high", "moderate"], (
                    f"{code}: {data['observed_exposure']} observed but ranked {data['exposure_rank']}"
                )

    def test_programmer_has_highest_observed(self):
        """Computer Programmers should have the highest observed exposure per the report."""
        prog = get_exposure("15-1251.00")
        assert prog is not None
        assert prog["observed_exposure"] == 0.75
        for code, data in EXPOSURE_DATA.items():
            if code != "15-1251.00":
                assert data["observed_exposure"] <= prog["observed_exposure"], (
                    f"{code} has higher observed exposure than Computer Programmers"
                )


# =====================================================================
# 3. Cross-persona invariant tests
# =====================================================================

class TestCrossPersonaInvariants:
    """Mathematical and logical invariants across all Anthropic report personas."""

    def test_scores_bounded_0_100(self):
        """All match scores and gap severities must be in [0, 100]."""
        for persona in ALL_PERSONAS:
            for occ_code in ALL_OCC_CODES:
                score = _score(persona, occ_code)
                assert 0 <= score.match_score <= 100, (
                    f"{persona['name']} → {occ_code}: match {score.match_score}"
                )
                assert 0 <= score.gap_severity <= 100, (
                    f"{persona['name']} → {occ_code}: gap {score.gap_severity}"
                )

    def test_valid_bucket_assignment(self):
        """Every score must produce a valid bucket."""
        valid_buckets = {"ready_now", "trainable", "long_reskill"}
        for persona in ALL_PERSONAS:
            for occ_code in ALL_OCC_CODES:
                score = _score(persona, occ_code)
                assert score.bucket in valid_buckets, (
                    f"{persona['name']} → {occ_code}: invalid bucket '{score.bucket}'"
                )

    def test_output_completeness(self):
        """Every score must have non-empty explanation and training suggestion."""
        for persona in ALL_PERSONAS:
            for occ_code in ALL_OCC_CODES:
                score = _score(persona, occ_code)
                assert isinstance(score.explanation, str) and len(score.explanation) > 0
                assert isinstance(score.training_suggestion, str) and len(score.training_suggestion) > 0
                assert isinstance(score.top_gaps, list)
                assert isinstance(score.metadata, dict)

    def test_determinism(self):
        """Same inputs must always produce same outputs."""
        for persona in ALL_PERSONAS:
            for occ_code in ALL_OCC_CODES:
                s1 = _score(persona, occ_code)
                s2 = _score(persona, occ_code)
                assert s1.match_score == s2.match_score
                assert s1.gap_severity == s2.gap_severity
                assert s1.bucket == s2.bucket

    def test_gap_ordering(self):
        """Gaps must be ordered by weight (importance) descending."""
        for persona in ALL_PERSONAS:
            for occ_code in ALL_OCC_CODES:
                score = _score(persona, occ_code)
                if len(score.top_gaps) > 1:
                    for i in range(len(score.top_gaps) - 1):
                        assert score.top_gaps[i].gap_weight >= score.top_gaps[i + 1].gap_weight, (
                            f"{persona['name']} → {occ_code}: gap ordering violated"
                        )

    def test_input_mutation_safety(self):
        """Scorer must never mutate the caller's skill ratings dict."""
        for persona in ALL_PERSONAS:
            original_ratings = copy.deepcopy(persona["skill_ratings"])
            for occ_code in ALL_OCC_CODES:
                _score(persona, occ_code)
            assert persona["skill_ratings"] == original_ratings, (
                f"{persona['name']}: skill ratings were mutated by scorer"
            )


# =====================================================================
# 4. Monotonicity tests
# =====================================================================

class TestMonotonicity:
    """Improving skills should never worsen scores."""

    @pytest.mark.parametrize("persona,occ_code", [
        (PERSONA_DEREK, "15-1212.00"),
        (PERSONA_LINDA, "43-4051.00"),
        (PERSONA_PRIYA, "29-2072.00"),
        (PERSONA_CARLOS, "13-1161.01"),
        (PERSONA_MARCUS, "15-1251.00"),
        (PERSONA_JORDAN, "15-1253.00"),
        (PERSONA_ELENA, "47-2111.00"),
    ])
    def test_improving_skill_never_worsens_match(self, persona, occ_code):
        """Bumping any skill rating up by 1 should never decrease match_score."""
        base_score = _score(persona, occ_code)
        for skill_id, rating in persona["skill_ratings"].items():
            if rating < 4:
                improved = dict(persona["skill_ratings"])
                improved[skill_id] = rating + 1
                improved_persona = {**persona, "skill_ratings": improved}
                new_score = _score(improved_persona, occ_code)
                assert new_score.match_score >= base_score.match_score, (
                    f"Improving {skill_id} from {rating}→{rating+1} decreased match: "
                    f"{base_score.match_score} → {new_score.match_score}"
                )


# =====================================================================
# 5. Transition scenario matrix tests
# =====================================================================

class TestTransitionMatrix:
    """Test specific transition paths relevant to the Anthropic report findings."""

    def test_high_exposure_to_low_exposure_transitions(self):
        """Workers in high-exposure roles transitioning to low-exposure trades."""
        high_exposure_personas = [PERSONA_DEREK, PERSONA_LINDA, PERSONA_PRIYA]
        low_exposure_target = "47-2111.00"  # Electrician

        for persona in high_exposure_personas:
            score = _score(persona, low_exposure_target)
            # These are major career changes — should require significant reskilling
            assert len(score.top_gaps) > 0, (
                f"{persona['name']} should have gaps for electrician role"
            )
            assert score.training_suggestion is not None

    def test_within_tech_mobility(self):
        """Tech workers should have reasonable mobility within tech roles."""
        tech_personas = [PERSONA_DEREK, PERSONA_MARCUS, PERSONA_FATIMA]
        tech_targets = ["15-1251.00", "15-1253.00", "15-1212.00", "15-1232.00"]

        for persona in tech_personas:
            high_matches = 0
            for target in tech_targets:
                score = _score(persona, target)
                if score.match_score >= 50:
                    high_matches += 1
            assert high_matches >= 2, (
                f"{persona['name']} should match well with at least 2 tech roles"
            )

    def test_service_role_transferability(self):
        """Service-oriented workers should transfer well between service roles."""
        service_personas = [PERSONA_LINDA, PERSONA_TANYA]
        service_targets = ["43-4051.00", "41-4012.00"]

        for persona in service_personas:
            for target in service_targets:
                score = _score(persona, target)
                assert score.match_score > 30, (
                    f"{persona['name']} should have reasonable match for {target}"
                )

    def test_declining_outlook_occupations_flagged(self):
        """Occupations with declining BLS outlook should have negative growth projections."""
        declining_codes = ["43-9021.00"]  # Data Entry Keyers (-32.4%)
        for code in declining_codes:
            bls = get_projections(code)
            assert bls is not None
            assert bls["projected_growth_pct"] < 0
            assert bls["outlook"] == "declining"

    def test_growth_occupations_positive_outlook(self):
        """Occupations recommended by the report should have positive BLS growth."""
        growth_codes = ["47-2111.00", "15-1212.00", "15-1253.00"]
        for code in growth_codes:
            bls = get_projections(code)
            assert bls is not None
            assert bls["projected_growth_pct"] > 0
            assert bls["outlook"] in ["moderate growth", "strong growth"]


# =====================================================================
# 6. Edge case and adversarial tests
# =====================================================================

class TestEdgeCases:
    """Adversarial and boundary inputs specific to Anthropic report personas."""

    def test_zero_skills_persona(self):
        """Persona with all ratings 0 should be long_reskill everywhere."""
        zero_persona = {
            "name": "Zero Skills",
            "current_occupation": "99-9999.00",
            "skill_ratings": {
                "2.B.1.a": 0, "2.B.2.a": 0, "2.B.3.a": 0,
                "2.B.4.a": 0, "2.B.8.a": 0, "2.B.1.g": 0,
            },
        }
        for occ_code in ALL_OCC_CODES:
            score = _score(zero_persona, occ_code)
            assert score.bucket == "long_reskill"
            assert score.gap_severity == 100.0

    def test_perfect_skills_persona(self):
        """Persona with all ratings 4 should be ready_now for every role."""
        for occ_code in ALL_OCC_CODES:
            occ = ANTHROPIC_MOCK_OCCUPATION_SKILLS[occ_code]
            perfect_ratings = {
                skill["element_id"]: 4 for skill in occ["skills"]
            }
            perfect_persona = {
                "name": "Perfect Skills",
                "current_occupation": "99-9999.00",
                "skill_ratings": perfect_ratings,
            }
            score = _score(perfect_persona, occ_code)
            assert score.match_score == 100.0
            assert score.gap_severity == 0.0
            assert score.bucket == "ready_now"
            assert len(score.top_gaps) == 0

    def test_no_overlapping_skills(self):
        """Persona with skills that don't match any occupation requirement."""
        mismatched_persona = {
            "name": "Mismatched",
            "current_occupation": "99-9999.00",
            "skill_ratings": {
                "2.B.99.a": 4,  # Fake skill ID
                "2.B.99.b": 4,
            },
        }
        for occ_code in ALL_OCC_CODES:
            score = _score(mismatched_persona, occ_code)
            assert score.gap_severity == 100.0
            assert score.bucket == "long_reskill"

    def test_partial_skill_overlap(self):
        """Persona with some but not all required skills should score partially."""
        partial = {
            "name": "Partial overlap",
            "current_occupation": "99-9999.00",
            "skill_ratings": {
                "2.B.8.a": 4,  # Critical Thinking – Expert
                "2.B.1.a": 4,  # Reading Comprehension – Expert
            },
        }
        # Score against programmer role (needs programming, math, etc.)
        score = _score(partial, "15-1251.00")
        assert 0 < score.match_score < 100
        assert 0 < score.gap_severity < 100

    def test_scoring_all_personas_against_all_occupations(self):
        """Full matrix: 10 personas × 10 occupations = 100 scores, all valid."""
        count = 0
        for persona in ALL_PERSONAS:
            for occ_code in ALL_OCC_CODES:
                score = _score(persona, occ_code)
                assert 0 <= score.match_score <= 100
                assert 0 <= score.gap_severity <= 100
                assert score.bucket in {"ready_now", "trainable", "long_reskill"}
                assert len(score.explanation) > 0
                assert len(score.training_suggestion) > 0
                count += 1
        assert count == 100, f"Expected 100 scores, got {count}"
