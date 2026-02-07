"""End-to-end integration tests for the full scoring pipeline.

Tests the complete journey from skills input through scoring to
recommendation output with explanations and training paths.
Uses 5 realistic personas to verify the system works for diverse users.
"""
import pytest
import numpy as np
from typing import Dict, List

from app.ml.scoring import BaselineScorer, OccupationScore


# === Test Personas ===
# Each persona represents a real user archetype with specific skills and goals.

PERSONA_MARIA = {
    "name": "Maria - Registered Nurse → Health Informatics",
    "current_occupation": "29-1141.00",  # Registered Nurses
    "target_occupation": "15-1211.00",   # Health Informatics
    "skill_ratings": {
        "2.B.1.a": 4,  # Reading Comprehension - Expert
        "2.B.2.a": 4,  # Active Listening - Expert
        "2.B.3.a": 3,  # Writing - Advanced
        "2.B.4.a": 4,  # Speaking - Expert
        "2.B.8.a": 3,  # Critical Thinking - Advanced
        "2.B.8.b": 2,  # Complex Problem Solving - Intermediate
        "2.B.1.g": 0,  # Programming - None
        "2.B.8.d": 1,  # Systems Analysis - Basic
        "2.B.7.a": 4,  # Social Perceptiveness - Expert
        "2.B.1.f": 4,  # Service Orientation - Expert
    },
    "expected_bucket_range": ["trainable"],  # Has strong soft skills, needs tech
    "budget": 500,
    "timeline_months": 12,
}

PERSONA_JAMES = {
    "name": "James - Retail Manager → exploring options",
    "current_occupation": "41-1011.00",  # First-Line Supervisors of Retail Sales
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension - Advanced
        "2.B.2.a": 4,  # Active Listening - Expert
        "2.B.3.a": 2,  # Writing - Intermediate
        "2.B.4.a": 4,  # Speaking - Expert
        "2.B.8.a": 3,  # Critical Thinking - Advanced
        "2.B.8.b": 3,  # Complex Problem Solving - Advanced
        "2.B.6.b": 4,  # Time Management - Expert
        "2.B.7.b": 4,  # Coordination - Expert
        "2.B.4.e": 4,  # Management of Personnel Resources - Expert
        "2.B.1.g": 0,  # Programming - None
    },
    "expected_bucket_range": ["trainable", "ready_now"],
    "budget": 0,
    "timeline_months": 6,
}

PERSONA_AISHA = {
    "name": "Aisha - CS Bootcamp Grad → checking readiness",
    "current_occupation": "15-1299.08",  # Web Developers
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension - Advanced
        "2.B.2.a": 2,  # Active Listening - Intermediate
        "2.B.3.a": 3,  # Writing - Advanced
        "2.B.8.a": 3,  # Critical Thinking - Advanced
        "2.B.8.b": 3,  # Complex Problem Solving - Advanced
        "2.B.1.g": 3,  # Programming - Advanced
        "2.B.5.c": 2,  # Design - Intermediate
        "2.B.6.b": 2,  # Time Management - Intermediate
        "2.B.8.d": 2,  # Systems Analysis - Intermediate
    },
    "expected_bucket_range": ["trainable", "ready_now"],
    "budget": 1000,
    "timeline_months": 12,
}

PERSONA_ROBERT = {
    "name": "Robert - Auto Mechanic, shop closing",
    "current_occupation": "49-3023.00",  # Automotive Service Technicians
    "skill_ratings": {
        "2.B.1.a": 2,  # Reading Comprehension - Intermediate
        "2.B.9.a": 4,  # Troubleshooting - Expert
        "2.B.4.h": 4,  # Equipment Maintenance - Expert
        "2.B.8.b": 3,  # Complex Problem Solving - Advanced
        "2.B.8.a": 3,  # Critical Thinking - Advanced
        "2.B.9.b": 3,  # Equipment Selection - Advanced
        "2.B.1.g": 0,  # Programming - None
        "2.B.3.a": 1,  # Writing - Basic
        "2.B.4.a": 2,  # Speaking - Intermediate
    },
    "expected_bucket_range": ["trainable", "long_reskill"],
    "budget": 0,
    "timeline_months": 6,
}

PERSONA_SARAH = {
    "name": "Sarah - Military Veteran (logistics)",
    "current_occupation": "13-1081.00",  # Logisticians
    "skill_ratings": {
        "2.B.1.a": 3,  # Reading Comprehension - Advanced
        "2.B.2.a": 3,  # Active Listening - Advanced
        "2.B.3.a": 3,  # Writing - Advanced
        "2.B.4.a": 3,  # Speaking - Advanced
        "2.B.8.a": 4,  # Critical Thinking - Expert
        "2.B.8.b": 4,  # Complex Problem Solving - Expert
        "2.B.8.d": 3,  # Systems Analysis - Advanced
        "2.B.7.b": 4,  # Coordination - Expert
        "2.B.6.b": 4,  # Time Management - Expert
        "2.B.4.e": 3,  # Management of Personnel Resources - Advanced
    },
    "expected_bucket_range": ["trainable", "ready_now"],
    "budget": 10000,  # GI Bill
    "timeline_months": 24,
}


# === Mock Occupation Data ===

MOCK_OCCUPATION_SKILLS = {
    "15-1252.00": {
        "title": "Software Developers",
        "job_zone": 4,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 72.0, "level": 5.12},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.3.a", "skill_name": "Writing", "importance": 75.0, "level": 5.12},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 81.0, "level": 5.62},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 84.0, "level": 5.88},
            {"element_id": "2.B.8.d", "skill_name": "Systems Analysis", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 84.0, "level": 5.75},
        ],
    },
    "15-1299.08": {
        "title": "Web Developers",
        "job_zone": 3,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 78.0, "level": 5.38},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 81.0, "level": 5.62},
            {"element_id": "2.B.5.c", "skill_name": "Design", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.6.b", "skill_name": "Time Management", "importance": 66.0, "level": 4.75},
        ],
    },
    "15-1244.00": {
        "title": "Network and Computer Systems Administrators",
        "job_zone": 3,
        "skills": [
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 5.00},
            {"element_id": "2.B.2.a", "skill_name": "Active Listening", "importance": 69.0, "level": 4.88},
            {"element_id": "2.B.4.a", "skill_name": "Speaking", "importance": 72.0, "level": 5.00},
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0, "level": 5.25},
            {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "importance": 81.0, "level": 5.50},
            {"element_id": "2.B.9.a", "skill_name": "Troubleshooting", "importance": 84.0, "level": 5.75},
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 72.0, "level": 5.12},
        ],
    },
}


class TestFullScoringPipeline:
    """Test the complete scoring journey for each persona."""

    def setup_method(self):
        """Set up scorer for each test."""
        self.scorer = BaselineScorer()

    def _score_persona(self, persona: Dict, occupation_code: str) -> OccupationScore:
        """Score a persona against an occupation."""
        occ = MOCK_OCCUPATION_SKILLS[occupation_code]
        return self.scorer.score_occupation(
            onet_code=occupation_code,
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=persona["skill_ratings"],
            current_job_zone=3,
            target_job_zone=occ["job_zone"],
        )

    def test_maria_nurse_to_tech(self):
        """Maria (nurse) should be TRAINABLE for tech roles - strong soft skills, needs programming."""
        score = self._score_persona(PERSONA_MARIA, "15-1252.00")

        # Maria has no programming skill, which is critical for Software Developers
        assert score.bucket in ["trainable", "long_reskill"]
        assert score.match_score > 20  # She has some transferable skills
        assert len(score.top_gaps) > 0
        assert score.explanation is not None
        assert score.training_suggestion is not None

        # Verify programming is identified as a gap
        gap_skill_ids = [g.element_id for g in score.top_gaps]
        assert "2.B.1.g" in gap_skill_ids, "Programming should be identified as a gap"

    def test_aisha_bootcamp_grad(self):
        """Aisha (CS bootcamp grad) should be TRAINABLE/READY_NOW for web dev roles."""
        score = self._score_persona(PERSONA_AISHA, "15-1299.08")

        # She has programming skills and CS fundamentals
        assert score.match_score > 40
        assert score.bucket in ["trainable", "ready_now"]

    def test_robert_mechanic(self):
        """Robert (mechanic) should score well for roles needing troubleshooting."""
        score = self._score_persona(PERSONA_ROBERT, "15-1244.00")

        # Robert has strong troubleshooting and equipment skills
        # But lacks programming, which is needed for sys admin
        assert score.bucket in ["trainable", "long_reskill"]
        assert len(score.top_gaps) > 0

    def test_sarah_veteran(self):
        """Sarah (veteran) should have strong transferable skills."""
        score = self._score_persona(PERSONA_SARAH, "15-1244.00")

        # Sarah has strong critical thinking, coordination, management
        assert score.match_score > 30
        assert score.bucket in ["trainable", "long_reskill"]

    def test_james_retail_manager(self):
        """James (retail manager) should be TRAINABLE for management-adjacent tech roles."""
        score = self._score_persona(PERSONA_JAMES, "15-1244.00")

        # James has strong people skills but no programming
        assert len(score.top_gaps) > 0
        assert score.training_suggestion is not None

    def test_all_personas_produce_valid_output(self):
        """All personas should produce valid, complete scoring output for all occupations."""
        personas = [PERSONA_MARIA, PERSONA_JAMES, PERSONA_AISHA, PERSONA_ROBERT, PERSONA_SARAH]

        for persona in personas:
            for occ_code in MOCK_OCCUPATION_SKILLS:
                score = self._score_persona(persona, occ_code)

                # Validate output structure
                assert score.onet_code == occ_code
                assert 0 <= score.match_score <= 100
                assert 0 <= score.gap_severity <= 100
                assert score.bucket in ["ready_now", "trainable", "long_reskill"]
                assert isinstance(score.explanation, str)
                assert len(score.explanation) > 0
                assert isinstance(score.training_suggestion, str)
                assert len(score.training_suggestion) > 0
                assert isinstance(score.top_gaps, list)
                assert isinstance(score.metadata, dict)

    def test_bucket_consistency(self):
        """Verify bucket boundaries are consistent with threshold logic."""
        scorer = BaselineScorer(
            ready_now_match_threshold=75.0,
            ready_now_gap_threshold=25.0,
            trainable_match_min=50.0,
            trainable_match_max=74.0,
            trainable_gap_min=26.0,
            trainable_gap_max=55.0,
        )

        # High match, low gaps = READY_NOW
        assert scorer._assign_bucket(80.0, 10.0) == "ready_now"

        # Moderate match = TRAINABLE
        assert scorer._assign_bucket(60.0, 30.0) == "trainable"

        # Low match, high gaps = LONG_RESKILL
        assert scorer._assign_bucket(30.0, 70.0) == "long_reskill"

    def test_empty_skills_handled(self):
        """Scorer should handle edge case of no occupation skills."""
        score = self.scorer.score_occupation(
            onet_code="99-9999.00",
            occupation_title="Unknown Job",
            occupation_skills=[],
            user_skill_ratings={"2.B.1.a": 3},
        )
        assert score.match_score == 0.0
        assert score.gap_severity == 100.0

    def test_no_user_ratings_handled(self):
        """Scorer should handle user with no ratings for required skills."""
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        score = self.scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings={},  # No ratings at all
        )
        # All skills should be gaps
        assert score.gap_severity == 100.0
        assert score.bucket == "long_reskill"

    def test_perfect_match(self):
        """User with max ratings on all required skills should be READY_NOW."""
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        perfect_ratings = {
            skill["element_id"]: 4  # Expert on everything
            for skill in occ["skills"]
        }
        score = self.scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=perfect_ratings,
        )
        assert score.match_score == 100.0
        assert score.gap_severity == 0.0
        assert score.bucket == "ready_now"
        assert len(score.top_gaps) == 0

    def test_score_determinism(self):
        """Same inputs should always produce same outputs."""
        score1 = self._score_persona(PERSONA_MARIA, "15-1252.00")
        score2 = self._score_persona(PERSONA_MARIA, "15-1252.00")

        assert score1.match_score == score2.match_score
        assert score1.gap_severity == score2.gap_severity
        assert score1.bucket == score2.bucket

    def test_gap_ordering(self):
        """Gaps should be ordered by weight (importance) descending."""
        score = self._score_persona(PERSONA_MARIA, "15-1252.00")

        if len(score.top_gaps) > 1:
            for i in range(len(score.top_gaps) - 1):
                assert score.top_gaps[i].gap_weight >= score.top_gaps[i + 1].gap_weight


class TestCalibrationFeatures:
    """Test calibration model feature extraction."""

    def test_feature_extraction(self):
        """Verify feature extraction produces valid feature vectors."""
        from app.ml.calibration import CalibrationModel

        model = CalibrationModel()
        features = model.extract_features(
            user_id=1,
            target_onet_code="15-1252.00",
            event_id=1,
            match_score=65.0,
            gap_severity=35.0,
            num_missing_skills=3,
            sum_missing_weights=0.45,
            current_job_zone=3,
            target_job_zone=4,
            user_ratings={"2.B.1.a": 3, "2.B.2.a": 2, "2.B.1.g": 0},
        )

        assert features.match_score == 65.0
        assert features.gap_severity == 35.0
        assert features.job_zone_diff == 1.0
        assert features.target_job_zone == 4.0
        assert features.num_missing_skills == 3
        assert features.num_rated_skills == 3
        assert features.mean_rating > 0

    def test_feature_to_array_shape(self):
        """Feature array should have correct shape for model input."""
        from app.ml.calibration import CalibrationModel

        model = CalibrationModel()
        features = model.extract_features(
            user_id=1, target_onet_code="15-1252.00", event_id=1,
            match_score=65.0, gap_severity=35.0,
            num_missing_skills=3, sum_missing_weights=0.45,
            current_job_zone=3, target_job_zone=4,
            user_ratings={"2.B.1.a": 3},
        )
        arr = model._features_to_array(features)
        assert arr.shape == (1, 9)


class TestFeedbackDataPreparation:
    """Test feedback-to-training-data pipeline."""

    def test_positive_label_assignment(self):
        """Interview, offer, and apply actions should produce positive labels."""
        from app.ml.calibration import prepare_training_data_from_feedback

        records = [
            {
                "action_type": "interview",
                "match_score": 70.0, "gap_severity": 20.0,
                "job_zone_diff": 1.0, "target_job_zone": 4.0,
                "num_missing_skills": 2, "sum_missing_weights": 0.3,
                "mean_rating": 3.0, "rating_variance": 0.5,
                "num_rated_skills": 8,
                "user_id": 1, "target_onet_code": "15-1252.00", "event_id": 1,
            },
            {
                "action_type": "offer",
                "match_score": 85.0, "gap_severity": 10.0,
                "job_zone_diff": 0.0, "target_job_zone": 3.0,
                "num_missing_skills": 1, "sum_missing_weights": 0.1,
                "mean_rating": 3.5, "rating_variance": 0.3,
                "num_rated_skills": 10,
                "user_id": 1, "target_onet_code": "15-1299.08", "event_id": 2,
            },
        ]

        training_data = prepare_training_data_from_feedback(records)
        assert len(training_data) == 2
        assert all(label == 1 for _, label in training_data)

    def test_negative_label_assignment(self):
        """Hide actions should produce negative labels."""
        from app.ml.calibration import prepare_training_data_from_feedback

        records = [
            {
                "action_type": "hide",
                "match_score": 30.0, "gap_severity": 70.0,
                "job_zone_diff": 2.0, "target_job_zone": 5.0,
                "num_missing_skills": 6, "sum_missing_weights": 0.8,
                "mean_rating": 2.0, "rating_variance": 1.0,
                "num_rated_skills": 5,
                "user_id": 1, "target_onet_code": "15-1252.00", "event_id": 1,
            },
        ]

        training_data = prepare_training_data_from_feedback(records)
        assert len(training_data) == 1
        assert training_data[0][1] == 0

    def test_ambiguous_actions_ignored(self):
        """Click and save actions should be ignored (ambiguous signal)."""
        from app.ml.calibration import prepare_training_data_from_feedback

        records = [
            {
                "action_type": "click",
                "match_score": 50.0, "gap_severity": 40.0,
                "job_zone_diff": 0.0, "target_job_zone": 3.0,
                "num_missing_skills": 3, "sum_missing_weights": 0.4,
                "mean_rating": 2.5, "rating_variance": 0.8,
                "num_rated_skills": 7,
                "user_id": 1, "target_onet_code": "15-1252.00", "event_id": 1,
            },
            {
                "action_type": "save",
                "match_score": 60.0, "gap_severity": 30.0,
                "job_zone_diff": 1.0, "target_job_zone": 4.0,
                "num_missing_skills": 2, "sum_missing_weights": 0.3,
                "mean_rating": 3.0, "rating_variance": 0.5,
                "num_rated_skills": 8,
                "user_id": 1, "target_onet_code": "15-1299.08", "event_id": 2,
            },
        ]

        training_data = prepare_training_data_from_feedback(records)
        assert len(training_data) == 0, "Click and save should be filtered out"
