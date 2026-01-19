"""Unit tests for baseline scoring engine."""
import pytest
from app.ml.scoring import BaselineScorer, RATING_TO_CAPABILITY


@pytest.fixture
def scorer():
    """Create a baseline scorer with default settings."""
    return BaselineScorer()


@pytest.fixture
def sample_occupation_skills():
    """Sample occupation skills."""
    return [
        {"element_id": "skill1", "skill_name": "Programming", "importance": 80.0, "level": 5.0},
        {"element_id": "skill2", "skill_name": "Critical Thinking", "importance": 70.0, "level": 4.5},
        {"element_id": "skill3", "skill_name": "Writing", "importance": 50.0, "level": 3.0},
        {"element_id": "skill4", "skill_name": "Mathematics", "importance": 60.0, "level": 4.0},
    ]


def test_rating_to_capability_mapping():
    """Test rating to capability scalar mapping."""
    assert RATING_TO_CAPABILITY[0] == 0.0
    assert RATING_TO_CAPABILITY[1] == 0.25
    assert RATING_TO_CAPABILITY[2] == 0.5
    assert RATING_TO_CAPABILITY[3] == 0.75
    assert RATING_TO_CAPABILITY[4] == 1.0


def test_perfect_match(scorer, sample_occupation_skills):
    """Test scoring with perfect skill match."""
    user_ratings = {
        "skill1": 4,  # Expert
        "skill2": 4,
        "skill3": 4,
        "skill4": 4,
    }

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
    )

    assert score.match_score == 100.0
    assert score.gap_severity == 0.0
    assert len(score.top_gaps) == 0
    assert score.bucket == "ready_now"


def test_no_skills_match(scorer, sample_occupation_skills):
    """Test scoring with no skill match."""
    user_ratings = {
        "skill1": 0,  # None
        "skill2": 0,
        "skill3": 0,
        "skill4": 0,
    }

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
    )

    assert score.match_score == 0.0
    assert score.gap_severity == 100.0
    assert len(score.top_gaps) == 4  # All skills are gaps
    assert score.bucket == "long_reskill"


def test_partial_match(scorer, sample_occupation_skills):
    """Test scoring with partial skill match."""
    user_ratings = {
        "skill1": 3,  # Advanced (0.75)
        "skill2": 2,  # Intermediate (0.5)
        "skill3": 0,  # None (0.0) - gap
        "skill4": 1,  # Basic (0.25) - gap
    }

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
    )

    # Expected weighted match:
    # Total importance: 80 + 70 + 50 + 60 = 260
    # skill1: (80/260) * 0.75 = 0.231
    # skill2: (70/260) * 0.5 = 0.135
    # skill3: (50/260) * 0.0 = 0
    # skill4: (60/260) * 0.25 = 0.058
    # Total: 0.423 * 100 = 42.3%

    assert 40 < score.match_score < 45
    assert score.gap_severity > 30  # skill3 and skill4 are gaps
    assert len(score.top_gaps) == 2
    assert score.bucket == "long_reskill"


def test_trainable_bucket(scorer, sample_occupation_skills):
    """Test assignment to trainable bucket."""
    user_ratings = {
        "skill1": 3,  # Advanced
        "skill2": 3,  # Advanced
        "skill3": 2,  # Intermediate
        "skill4": 2,  # Intermediate
    }

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
    )

    # Should have moderate match score
    assert 60 < score.match_score < 80
    assert score.bucket == "trainable"


def test_ready_now_bucket(scorer, sample_occupation_skills):
    """Test assignment to ready_now bucket."""
    user_ratings = {
        "skill1": 4,  # Expert
        "skill2": 4,  # Expert
        "skill3": 3,  # Advanced
        "skill4": 3,  # Advanced
    }

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
    )

    assert score.match_score >= 75
    assert score.gap_severity <= 25
    assert score.bucket == "ready_now"


def test_skill_gaps_sorted_by_importance(scorer, sample_occupation_skills):
    """Test that skill gaps are sorted by importance."""
    user_ratings = {
        "skill1": 0,  # Gap - importance 80
        "skill2": 0,  # Gap - importance 70
        "skill3": 0,  # Gap - importance 50
        "skill4": 0,  # Gap - importance 60
    }

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
    )

    # Should be sorted by importance (weight)
    assert score.top_gaps[0].skill_name == "Programming"  # importance 80
    assert score.top_gaps[1].skill_name == "Critical Thinking"  # importance 70
    assert score.top_gaps[2].skill_name == "Mathematics"  # importance 60
    assert score.top_gaps[3].skill_name == "Writing"  # importance 50


def test_training_suggestion_job_zone_1(scorer, sample_occupation_skills):
    """Test training suggestion for job zone 1."""
    user_ratings = {"skill1": 2, "skill2": 2, "skill3": 0, "skill4": 1}

    score = scorer.score_occupation(
        onet_code="11-9013.00",
        occupation_title="Test Occupation",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
        target_job_zone=1,
    )

    assert "certificate" in score.training_suggestion.lower() or "apprenticeship" in score.training_suggestion.lower()


def test_training_suggestion_job_zone_4(scorer, sample_occupation_skills):
    """Test training suggestion for job zone 4."""
    user_ratings = {"skill1": 2, "skill2": 2, "skill3": 0, "skill4": 1}

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
        target_job_zone=4,
    )

    # Job zone 4 should suggest longer training
    assert any(
        word in score.training_suggestion.lower()
        for word in ["associate", "degree", "comprehensive", "1-2 years"]
    )


def test_empty_occupation_skills(scorer):
    """Test scoring with empty occupation skills."""
    score = scorer.score_occupation(
        onet_code="99-9999.00",
        occupation_title="Empty Occupation",
        occupation_skills=[],
        user_skill_ratings={},
    )

    assert score.match_score == 0.0
    assert score.gap_severity == 100.0
    assert len(score.top_gaps) == 0


def test_missing_user_ratings(scorer, sample_occupation_skills):
    """Test scoring when user hasn't rated some skills."""
    # User only rated 2 out of 4 skills
    user_ratings = {
        "skill1": 3,
        "skill2": 2,
        # skill3 and skill4 not rated - should default to 0
    }

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
    )

    # Missing ratings treated as 0 (no capability)
    assert score.match_score < 50
    assert len(score.top_gaps) == 2  # skill3 and skill4


def test_bucket_boundaries():
    """Test bucket assignment at threshold boundaries."""
    scorer = BaselineScorer(
        ready_now_match_threshold=75.0,
        ready_now_gap_threshold=25.0,
        trainable_match_min=50.0,
        trainable_match_max=74.0,
    )

    # Exactly at READY_NOW threshold
    assert scorer._assign_bucket(75.0, 25.0) == "ready_now"
    assert scorer._assign_bucket(74.9, 25.0) == "trainable"

    # Exactly at TRAINABLE threshold
    assert scorer._assign_bucket(50.0, 30.0) == "trainable"
    assert scorer._assign_bucket(49.9, 30.0) == "long_reskill"


def test_metadata_in_score(scorer, sample_occupation_skills):
    """Test that metadata contains expected fields."""
    user_ratings = {"skill1": 3, "skill2": 2, "skill3": 1, "skill4": 0}

    score = scorer.score_occupation(
        onet_code="15-1252.00",
        occupation_title="Software Developer",
        occupation_skills=sample_occupation_skills,
        user_skill_ratings=user_ratings,
        current_job_zone=3,
        target_job_zone=4,
    )

    assert "total_skills" in score.metadata
    assert "skills_with_gaps" in score.metadata
    assert "job_zone_diff" in score.metadata
    assert score.metadata["job_zone_diff"] == 1
    assert score.metadata["total_skills"] == 4
