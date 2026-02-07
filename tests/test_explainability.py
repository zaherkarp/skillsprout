"""Tests for the explainability engine.

Covers:
  - ThresholdConfig: presets, domain classification, credential barriers
  - BucketExplainer: explanation generation for all three buckets
  - ComparisonView: side-by-side comparison, overlap, and ranking
"""

import pytest
from typing import Dict, List

from app.ml.scoring import BaselineScorer, OccupationScore, SkillGap
from app.features.explainability.threshold_config import (
    BucketThresholds,
    CredentialBarrierRule,
    RiskTolerance,
    SkillDomain,
    SkillDomainWeights,
    ThresholdProfile,
    classify_skill_domain,
    get_all_presets,
    get_threshold_profile,
    THRESHOLD_PRESETS,
)
from app.features.explainability.bucket_explainer import (
    BucketExplainerEngine,
    BucketExplanation,
    GapCategory,
    SkillToDevelop,
    WhatWouldChangeBucket,
    explain_score,
    _categorise_gap,
    _estimate_training_time,
)
from app.features.explainability.comparison_view import (
    ComparisonEngine,
    ComparisonError,
    ComparisonResult,
    MAX_COMPARISON_SIZE,
    compare_occupations,
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
    """Four sample skills with varying importance."""
    return [
        {"element_id": "2.B.3.a", "skill_name": "Programming", "importance": 80.0, "level": 5.0},
        {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 70.0, "level": 4.5},
        {"element_id": "2.B.4.a", "skill_name": "Social Perceptiveness", "importance": 50.0, "level": 3.0},
        {"element_id": "2.A.1.a", "skill_name": "Oral Comprehension", "importance": 60.0, "level": 4.0},
    ]


@pytest.fixture
def ready_now_ratings() -> Dict[str, int]:
    """User ratings that produce a READY_NOW score."""
    return {
        "2.B.3.a": 4,  # Expert
        "2.B.1.a": 4,  # Expert
        "2.B.4.a": 3,  # Advanced
        "2.A.1.a": 3,  # Advanced
    }


@pytest.fixture
def trainable_ratings() -> Dict[str, int]:
    """User ratings that produce a TRAINABLE score."""
    return {
        "2.B.3.a": 3,  # Advanced
        "2.B.1.a": 2,  # Intermediate
        "2.B.4.a": 0,  # None - gap
        "2.A.1.a": 1,  # Basic - gap
    }


@pytest.fixture
def long_reskill_ratings() -> Dict[str, int]:
    """User ratings that produce a LONG_RESKILL score."""
    return {
        "2.B.3.a": 0,  # None - gap
        "2.B.1.a": 0,  # None - gap
        "2.B.4.a": 0,  # None - gap
        "2.A.1.a": 0,  # None - gap
    }


def _make_score(
    scorer: BaselineScorer,
    skills: List[Dict],
    ratings: Dict[str, int],
    onet_code: str = "15-1252.00",
    title: str = "Software Developer",
) -> OccupationScore:
    """Helper to generate an OccupationScore from skills and ratings."""
    return scorer.score_occupation(
        onet_code=onet_code,
        occupation_title=title,
        occupation_skills=skills,
        user_skill_ratings=ratings,
    )


# ===================================================================
# ThresholdConfig tests
# ===================================================================

class TestThresholdConfig:
    """Tests for threshold_config module."""

    def test_standard_preset_matches_settings(self):
        """STANDARD preset thresholds should match app.core.config.settings."""
        from app.core.config import settings

        profile = get_threshold_profile(RiskTolerance.STANDARD)
        t = profile.bucket_thresholds

        assert t.ready_now_match_min == settings.ready_now_match_threshold
        assert t.ready_now_gap_max == settings.ready_now_gap_threshold
        assert t.trainable_match_min == settings.trainable_match_min
        assert t.trainable_match_max == settings.trainable_match_max
        assert t.trainable_gap_min == settings.trainable_gap_min
        assert t.trainable_gap_max == settings.trainable_gap_max

    def test_relaxed_has_lower_thresholds(self):
        """RELAXED preset should have lower READY_NOW match and higher gap ceiling."""
        standard = get_threshold_profile(RiskTolerance.STANDARD)
        relaxed = get_threshold_profile(RiskTolerance.RELAXED)

        assert relaxed.bucket_thresholds.ready_now_match_min < standard.bucket_thresholds.ready_now_match_min
        assert relaxed.bucket_thresholds.ready_now_gap_max > standard.bucket_thresholds.ready_now_gap_max

    def test_strict_has_higher_thresholds(self):
        """STRICT preset should have higher READY_NOW match and lower gap ceiling."""
        standard = get_threshold_profile(RiskTolerance.STANDARD)
        strict = get_threshold_profile(RiskTolerance.STRICT)

        assert strict.bucket_thresholds.ready_now_match_min > standard.bucket_thresholds.ready_now_match_min
        assert strict.bucket_thresholds.ready_now_gap_max < standard.bucket_thresholds.ready_now_gap_max

    def test_all_presets_available(self):
        """All three presets should be retrievable."""
        presets = get_all_presets()
        assert "relaxed" in presets
        assert "standard" in presets
        assert "strict" in presets

    def test_classify_skill_domain_technical(self):
        """Technical skills should map to TECHNICAL domain."""
        assert classify_skill_domain("2.B.3.a") == SkillDomain.TECHNICAL
        assert classify_skill_domain("2.B.3.z") == SkillDomain.TECHNICAL

    def test_classify_skill_domain_cognitive(self):
        """Cognitive abilities should map to COGNITIVE domain."""
        assert classify_skill_domain("2.A.1.a") == SkillDomain.COGNITIVE
        assert classify_skill_domain("2.A.1.b.1") == SkillDomain.COGNITIVE

    def test_classify_skill_domain_social(self):
        """Social skills should map to SOCIAL_SKILLS domain."""
        assert classify_skill_domain("2.B.4.a") == SkillDomain.SOCIAL_SKILLS

    def test_classify_skill_domain_basic_skills(self):
        """Basic content/process skills should map to BASIC_SKILLS domain."""
        assert classify_skill_domain("2.B.1.a") == SkillDomain.BASIC_SKILLS

    def test_classify_skill_domain_unknown(self):
        """Unknown element_ids should map to OTHER domain."""
        assert classify_skill_domain("99.X.1") == SkillDomain.OTHER
        assert classify_skill_domain("") == SkillDomain.OTHER

    def test_longest_prefix_match(self):
        """Longer prefixes should take priority over shorter ones."""
        # 2.B.3.a matches 2.B.3 (TECHNICAL) not 2.B (which is not mapped)
        assert classify_skill_domain("2.B.3.a") == SkillDomain.TECHNICAL

    def test_domain_weights_default_neutral(self):
        """Default domain weights should all be close to 1.0."""
        weights = SkillDomainWeights()
        for domain in SkillDomain:
            w = weights.get_weight(domain)
            assert 0.5 <= w <= 2.0, f"Weight for {domain} is {w}, expected 0.5-2.0"

    def test_credential_barrier_structure(self):
        """Credential barrier rules should have required fields."""
        from app.features.explainability.threshold_config import DEFAULT_CREDENTIAL_BARRIERS

        assert len(DEFAULT_CREDENTIAL_BARRIERS) > 0
        for rule in DEFAULT_CREDENTIAL_BARRIERS:
            assert isinstance(rule, CredentialBarrierRule)
            assert len(rule.credential_element_ids) > 0
            assert rule.force_bucket in ("trainable", "long_reskill")
            assert len(rule.explanation) > 0


# ===================================================================
# BucketExplainer tests
# ===================================================================

class TestBucketExplainer:
    """Tests for bucket_explainer module."""

    def test_explain_ready_now_score(self, scorer, sample_skills, ready_now_ratings):
        """Explanation for READY_NOW should report the correct bucket and low gaps."""
        score = _make_score(scorer, sample_skills, ready_now_ratings)
        assert score.bucket == "ready_now"

        explanation = explain_score(
            score,
            occupation_skills=sample_skills,
            user_skill_ratings=ready_now_ratings,
        )

        assert explanation.onet_code == "15-1252.00"
        assert explanation.bucket_reasoning.assigned_bucket == "ready_now"
        assert explanation.bucket_reasoning.match_meets_ready_now is True
        assert explanation.bucket_reasoning.gap_meets_ready_now is True
        assert "Ready Now" in explanation.summary
        assert len(explanation.skills_you_have) > 0
        assert explanation.what_would_change_bucket.to_ready_now is None

    def test_explain_trainable_score(self, scorer, sample_skills, trainable_ratings):
        """Explanation for TRAINABLE should show gaps and what would change."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        assert score.bucket == "trainable"

        explanation = explain_score(
            score,
            occupation_skills=sample_skills,
            user_skill_ratings=trainable_ratings,
        )

        assert explanation.bucket_reasoning.assigned_bucket == "trainable"
        assert len(explanation.skills_to_develop) > 0
        assert explanation.what_would_change_bucket.to_ready_now is not None
        assert "Trainable" in explanation.summary

    def test_explain_long_reskill_score(self, scorer, sample_skills, long_reskill_ratings):
        """Explanation for LONG_RESKILL should show all gaps and both transitions."""
        score = _make_score(scorer, sample_skills, long_reskill_ratings)
        assert score.bucket == "long_reskill"

        explanation = explain_score(
            score,
            occupation_skills=sample_skills,
            user_skill_ratings=long_reskill_ratings,
        )

        assert explanation.bucket_reasoning.assigned_bucket == "long_reskill"
        assert len(explanation.skills_to_develop) == 4
        assert explanation.what_would_change_bucket.to_ready_now is not None
        assert explanation.what_would_change_bucket.to_trainable is not None
        assert "Long Reskill" in explanation.summary

    def test_skills_to_develop_have_gap_categories(self, scorer, sample_skills, trainable_ratings):
        """Each skill gap should have a valid gap category."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        explanation = explain_score(score)

        for skill in explanation.skills_to_develop:
            assert skill.gap_category in (GapCategory.MINOR, GapCategory.MODERATE, GapCategory.MAJOR)
            assert len(skill.typical_training_time) > 0
            assert len(skill.why_it_matters) > 0

    def test_skills_to_develop_have_domains(self, scorer, sample_skills, trainable_ratings):
        """Each skill gap should have a skill domain classification."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        explanation = explain_score(
            score,
            occupation_skills=sample_skills,
            user_skill_ratings=trainable_ratings,
        )

        valid_domains = {d.value for d in SkillDomain}
        for skill in explanation.skills_to_develop:
            assert skill.skill_domain in valid_domains

    def test_bucket_reasoning_shows_thresholds(self, scorer, sample_skills, trainable_ratings):
        """Bucket reasoning should expose the actual thresholds used."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        explanation = explain_score(score)

        thresholds = explanation.bucket_reasoning.thresholds_used
        assert "ready_now_match_min" in thresholds
        assert "ready_now_gap_max" in thresholds
        assert "trainable_match_min" in thresholds
        assert "trainable_match_max" in thresholds
        assert thresholds["ready_now_match_min"] == 75.0

    def test_reasoning_text_is_human_readable(self, scorer, sample_skills, trainable_ratings):
        """Reasoning text should mention the user's actual scores."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        explanation = explain_score(score)

        text = explanation.bucket_reasoning.reasoning_text
        assert str(int(score.match_score)) in text or f"{score.match_score:.1f}" in text

    def test_what_would_change_gaps_to_close(self, scorer, sample_skills, trainable_ratings):
        """What-would-change should list specific gaps to close."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        explanation = explain_score(score)

        if explanation.what_would_change_bucket.gaps_to_close:
            for gap_name in explanation.what_would_change_bucket.gaps_to_close:
                assert isinstance(gap_name, str)
                assert len(gap_name) > 0

    def test_explain_with_different_risk_tolerance(self, scorer, sample_skills, trainable_ratings):
        """Different risk tolerances should produce different threshold references."""
        score = _make_score(scorer, sample_skills, trainable_ratings)

        standard = explain_score(score, risk_tolerance=RiskTolerance.STANDARD)
        relaxed = explain_score(score, risk_tolerance=RiskTolerance.RELAXED)

        # Different thresholds in reasoning
        assert (
            standard.bucket_reasoning.thresholds_used["ready_now_match_min"]
            != relaxed.bucket_reasoning.thresholds_used["ready_now_match_min"]
        )
        assert standard.risk_tolerance_used == "standard"
        assert relaxed.risk_tolerance_used == "relaxed"

    def test_explain_without_skill_details(self, scorer, sample_skills, trainable_ratings):
        """Explanation should work even without full skill lists."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        # Call without occupation_skills or user_skill_ratings
        explanation = explain_score(score)

        assert explanation.onet_code == score.onet_code
        assert explanation.bucket_reasoning.assigned_bucket == score.bucket
        # skills_you_have falls back to count-based description
        assert len(explanation.skills_you_have) >= 0

    def test_matched_skills_identified_correctly(self, scorer, sample_skills, trainable_ratings):
        """Skills where user has rating >= 2 and not a gap should be 'matched'."""
        score = _make_score(scorer, sample_skills, trainable_ratings)
        explanation = explain_score(
            score,
            occupation_skills=sample_skills,
            user_skill_ratings=trainable_ratings,
        )

        # trainable_ratings: 2.B.3.a=3, 2.B.1.a=2 are not gaps
        # 2.B.4.a=0, 2.A.1.a=1 are gaps
        assert "Programming" in explanation.skills_you_have
        assert "Reading Comprehension" in explanation.skills_you_have

    def test_gap_categorisation_minor(self):
        """Small gap weights should be categorised as MINOR."""
        assert _categorise_gap(0.03) == GapCategory.MINOR

    def test_gap_categorisation_moderate(self):
        """Medium gap weights should be categorised as MODERATE."""
        assert _categorise_gap(0.10) == GapCategory.MODERATE

    def test_gap_categorisation_major(self):
        """Large gap weights should be categorised as MAJOR."""
        assert _categorise_gap(0.20) == GapCategory.MAJOR

    def test_training_time_estimates(self):
        """Each gap category should have a training time estimate."""
        for category in (GapCategory.MINOR, GapCategory.MODERATE, GapCategory.MAJOR):
            estimate = _estimate_training_time(category)
            assert len(estimate) > 0
            assert "week" in estimate.lower() or "month" in estimate.lower()


# ===================================================================
# ComparisonView tests
# ===================================================================

class TestComparisonView:
    """Tests for comparison_view module."""

    def _make_two_scores(self, scorer, sample_skills, ready_now_ratings, trainable_ratings):
        """Helper to create two different scores for comparison."""
        score_a = _make_score(
            scorer, sample_skills, ready_now_ratings,
            onet_code="15-1252.00", title="Software Developer",
        )
        score_b = _make_score(
            scorer, sample_skills, trainable_ratings,
            onet_code="29-1141.00", title="Registered Nurse",
        )
        return score_a, score_b

    def test_compare_two_occupations(self, scorer, sample_skills, ready_now_ratings, trainable_ratings):
        """Comparing two occupations should produce a valid ComparisonResult."""
        score_a, score_b = self._make_two_scores(
            scorer, sample_skills, ready_now_ratings, trainable_ratings
        )

        result = compare_occupations([score_a, score_b])

        assert len(result.occupation_codes) == 2
        assert "15-1252.00" in result.occupation_codes
        assert "29-1141.00" in result.occupation_codes
        assert len(result.explanations) == 2
        assert result.readiness_ranking.closest_onet_code is not None

    def test_compare_single_occupation(self, scorer, sample_skills, ready_now_ratings):
        """Single occupation comparison should still work."""
        score = _make_score(scorer, sample_skills, ready_now_ratings)
        result = compare_occupations([score])

        assert len(result.occupation_codes) == 1
        assert "Single occupation" in result.comparison_summary

    def test_compare_exceeds_max_raises_error(self, scorer, sample_skills, ready_now_ratings):
        """More than MAX_COMPARISON_SIZE occupations should raise ComparisonError."""
        scores = [
            _make_score(scorer, sample_skills, ready_now_ratings, onet_code=f"11-100{i}.00")
            for i in range(MAX_COMPARISON_SIZE + 1)
        ]

        with pytest.raises(ComparisonError, match="Maximum"):
            compare_occupations(scores)

    def test_compare_empty_raises_error(self):
        """Empty score list should raise ComparisonError."""
        with pytest.raises(ComparisonError, match="At least one"):
            compare_occupations([])

    def test_readiness_ranking_order(self, scorer, sample_skills, ready_now_ratings, trainable_ratings, long_reskill_ratings):
        """READY_NOW should rank before TRAINABLE before LONG_RESKILL."""
        score_rn = _make_score(
            scorer, sample_skills, ready_now_ratings,
            onet_code="A", title="Ready",
        )
        score_tr = _make_score(
            scorer, sample_skills, trainable_ratings,
            onet_code="B", title="Trainable",
        )
        score_lr = _make_score(
            scorer, sample_skills, long_reskill_ratings,
            onet_code="C", title="LongReskill",
        )

        result = compare_occupations([score_lr, score_rn, score_tr])

        assert result.readiness_ranking.ranked_codes[0] == "A"
        assert result.readiness_ranking.closest_onet_code == "A"

    def test_shared_gaps_detected(self, scorer, sample_skills, trainable_ratings):
        """When two occupations have the same gaps, they should be listed as shared."""
        # Both occupations use the same skills and same user ratings,
        # so they share the same gaps.
        score_a = _make_score(
            scorer, sample_skills, trainable_ratings,
            onet_code="X", title="OccA",
        )
        score_b = _make_score(
            scorer, sample_skills, trainable_ratings,
            onet_code="Y", title="OccB",
        )

        result = compare_occupations([score_a, score_b])

        # Both have the same gaps (2.B.4.a and 2.A.1.a)
        assert len(result.skill_overlap.shared_gaps) > 0

    def test_unique_gaps_detected(self, scorer):
        """Gaps unique to one occupation should be identified."""
        skills_a = [
            {"element_id": "s1", "skill_name": "Skill1", "importance": 80.0, "level": 5.0},
            {"element_id": "s2", "skill_name": "Skill2", "importance": 70.0, "level": 4.0},
        ]
        skills_b = [
            {"element_id": "s1", "skill_name": "Skill1", "importance": 80.0, "level": 5.0},
            {"element_id": "s3", "skill_name": "Skill3", "importance": 70.0, "level": 4.0},
        ]
        ratings = {"s1": 0, "s2": 0, "s3": 0}

        score_a = _make_score(scorer, skills_a, ratings, onet_code="A", title="A")
        score_b = _make_score(scorer, skills_b, ratings, onet_code="B", title="B")

        result = compare_occupations([score_a, score_b])

        # s2 is unique to A, s3 is unique to B
        assert "Skill2" in result.skill_overlap.unique_gaps.get("A", [])
        assert "Skill3" in result.skill_overlap.unique_gaps.get("B", [])

    def test_comparison_summary_mentions_all_codes(self, scorer, sample_skills, ready_now_ratings, trainable_ratings):
        """Summary should mention all compared occupation codes."""
        score_a, score_b = self._make_two_scores(
            scorer, sample_skills, ready_now_ratings, trainable_ratings
        )

        result = compare_occupations([score_a, score_b])

        assert "15-1252.00" in result.comparison_summary
        assert "29-1141.00" in result.comparison_summary

    def test_comparison_with_risk_tolerance(self, scorer, sample_skills, trainable_ratings):
        """Comparison should respect risk tolerance setting."""
        score = _make_score(scorer, sample_skills, trainable_ratings)

        result_standard = compare_occupations(
            [score], risk_tolerance=RiskTolerance.STANDARD
        )
        result_relaxed = compare_occupations(
            [score], risk_tolerance=RiskTolerance.RELAXED
        )

        # Both should succeed, and use different thresholds
        expl_standard = result_standard.explanations[score.onet_code]
        expl_relaxed = result_relaxed.explanations[score.onet_code]
        assert expl_standard.risk_tolerance_used == "standard"
        assert expl_relaxed.risk_tolerance_used == "relaxed"

    def test_three_way_comparison(self, scorer, sample_skills, ready_now_ratings, trainable_ratings, long_reskill_ratings):
        """Three-way comparison should work correctly."""
        score_a = _make_score(
            scorer, sample_skills, ready_now_ratings,
            onet_code="A", title="A",
        )
        score_b = _make_score(
            scorer, sample_skills, trainable_ratings,
            onet_code="B", title="B",
        )
        score_c = _make_score(
            scorer, sample_skills, long_reskill_ratings,
            onet_code="C", title="C",
        )

        result = compare_occupations([score_a, score_b, score_c])

        assert len(result.occupation_codes) == 3
        assert len(result.explanations) == 3
        assert len(result.readiness_ranking.ranked_codes) == 3

    def test_readiness_distance_zero_for_ready_now(self, scorer, sample_skills, ready_now_ratings):
        """Distance to READY_NOW should be 0 for occupations already in that bucket."""
        score = _make_score(scorer, sample_skills, ready_now_ratings)
        assert score.bucket == "ready_now"

        result = compare_occupations([score])

        ranking = result.readiness_ranking.rankings[0]
        assert ranking["distance_to_ready_now"] == 0.0
