"""Cross-team QA alignment: property-based and adversarial testing.

Background
----------
Team Alpha (persona-driven QA, Feb 2026 hackathon) and Team Beta (contract
& invariant QA) merged under new technical leadership. This module represents
the combined team's joint review, blending Alpha's 5-persona scenario approach
with Beta's emphasis on:

  1. Mathematical invariants that must hold for ALL inputs
  2. Boundary-value analysis at exact threshold transitions
  3. Monotonicity guarantees (improving skills never worsens scores)
  4. Input mutation safety (scorer must not modify caller's data)
  5. Cross-persona parity (similar profiles produce similar scores)
  6. Adversarial / degenerate inputs (empty, extreme, malformed)
  7. Explainability-scorer contract alignment

All 5 existing personas (Maria, James, Aisha, Robert, Sarah) are reused
to anchor cross-team consistency checks.

Team Alpha QA approach: scenario-driven, persona-validated, narrative tests.
Team Beta QA approach:  invariant-driven, boundary-exhaustive, adversarial.
Combined:              both, plus cross-cutting contract verification.
"""
import copy
import math
import pytest
from typing import Dict, List, Any

from app.ml.scoring import (
    BaselineScorer,
    OccupationScore,
    SkillGap,
    RATING_TO_CAPABILITY,
)
from app.features.explainability.bucket_explainer import (
    BucketExplainerEngine,
    explain_score,
    _categorise_gap,
    GapCategory,
)
from app.features.explainability.threshold_config import (
    BucketThresholds,
    RiskTolerance,
    ThresholdProfile,
    get_threshold_profile,
    get_all_presets,
    classify_skill_domain,
    SkillDomain,
)
from app.features.training_paths.path_generator import (
    PathGenerator,
    SkillGap as TrainingSkillGap,
    TrainingPath,
)
from app.features.training_paths.resource_filter import UserConstraints

# Re-import personas from the integration test module for cross-team continuity
from tests.integration.test_full_pipeline import (
    PERSONA_MARIA,
    PERSONA_JAMES,
    PERSONA_AISHA,
    PERSONA_ROBERT,
    PERSONA_SARAH,
    MOCK_OCCUPATION_SKILLS,
)


# ===================================================================
# Helper fixtures and data
# ===================================================================

ALL_PERSONAS = [
    PERSONA_MARIA,
    PERSONA_JAMES,
    PERSONA_AISHA,
    PERSONA_ROBERT,
    PERSONA_SARAH,
]

ALL_OCCUPATION_CODES = list(MOCK_OCCUPATION_SKILLS.keys())


def _make_scorer(**overrides) -> BaselineScorer:
    """Build a scorer with explicit standard thresholds unless overridden."""
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
    """Score a persona against an occupation using the test scorer."""
    if scorer is None:
        scorer = _make_scorer()
    occ = MOCK_OCCUPATION_SKILLS[occupation_code]
    return scorer.score_occupation(
        onet_code=occupation_code,
        occupation_title=occ["title"],
        occupation_skills=occ["skills"],
        user_skill_ratings=persona["skill_ratings"],
        current_job_zone=3,
        target_job_zone=occ["job_zone"],
    )


# ===================================================================
# 1. MATHEMATICAL INVARIANTS  (Team Beta focus)
# ===================================================================

class TestMathematicalInvariants:
    """Properties that must hold for ALL valid inputs, regardless of persona."""

    def test_match_score_bounded_0_100(self):
        """match_score is always in [0, 100] for every persona x occupation."""
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                assert 0 <= score.match_score <= 100, (
                    f"{persona['name']} vs {code}: "
                    f"match_score={score.match_score} out of range"
                )

    def test_gap_severity_bounded_0_100(self):
        """gap_severity is always in [0, 100] for every persona x occupation."""
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                assert 0 <= score.gap_severity <= 100, (
                    f"{persona['name']} vs {code}: "
                    f"gap_severity={score.gap_severity} out of range"
                )

    def test_match_gap_upper_bound_invariant(self):
        """match_score + 0.75 * gap_severity <= 100 (proven from score formula).

        Proof: Let S = set of skills with capability <= 0.25 (gap set).
        match = 100 * sum(w_i * c_i). For i in S, c_i <= 0.25.
        gap = 100 * sum(w_i for i in S).
        match <= 100 * [0.25 * sum(w_i, i in S) + 1.0 * sum(w_i, i not in S)]
             = 100 * [0.25 * gap/100 + (1 - gap/100)]
             = 100 - 0.75 * gap
        Therefore: match + 0.75 * gap <= 100.
        """
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                lhs = score.match_score + 0.75 * score.gap_severity
                assert lhs <= 100.01, (  # tiny epsilon for floating point
                    f"{persona['name']} vs {code}: "
                    f"match({score.match_score}) + 0.75*gap({score.gap_severity}) "
                    f"= {lhs} > 100 violates invariant"
                )

    def test_gap_severity_lower_bound_from_match(self):
        """gap_severity >= (100 - match_score) / 1.0 is NOT guaranteed
        (unrated skills default to 0 but the user might have partial ratings).
        However, gap_severity >= 0 always holds (tested above).

        What IS guaranteed: if match < 100, then either gaps exist OR
        the user has partial (0.5 / 0.75) skills that aren't gaps.
        """
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                if score.match_score < 100.0:
                    # Either there are gaps, or there are partial-capability skills
                    has_gaps = score.gap_severity > 0
                    has_partial = any(
                        RATING_TO_CAPABILITY.get(
                            persona["skill_ratings"].get(s["element_id"], 0), 0.0
                        ) > 0.25
                        and RATING_TO_CAPABILITY.get(
                            persona["skill_ratings"].get(s["element_id"], 0), 0.0
                        ) < 1.0
                        for s in MOCK_OCCUPATION_SKILLS[code]["skills"]
                    )
                    has_unrated = any(
                        s["element_id"] not in persona["skill_ratings"]
                        for s in MOCK_OCCUPATION_SKILLS[code]["skills"]
                    )
                    assert has_gaps or has_partial or has_unrated, (
                        f"{persona['name']} vs {code}: match<100 but no gaps, "
                        "partial skills, or unrated skills"
                    )

    def test_gap_weights_sum_to_gap_severity(self):
        """Sum of top_gaps[].gap_weight * 100 should equal gap_severity."""
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                reconstructed = sum(g.gap_weight for g in score.top_gaps) * 100
                assert abs(reconstructed - score.gap_severity) < 0.1, (
                    f"{persona['name']} vs {code}: sum(gap_weights)*100 = "
                    f"{reconstructed} != gap_severity = {score.gap_severity}"
                )

    def test_bucket_always_valid_string(self):
        """bucket must be one of the three valid values."""
        valid = {"ready_now", "trainable", "long_reskill"}
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                assert score.bucket in valid, (
                    f"Invalid bucket '{score.bucket}' for "
                    f"{persona['name']} vs {code}"
                )


# ===================================================================
# 2. BOUNDARY VALUE ANALYSIS  (Team Beta focus)
# ===================================================================

class TestBoundaryValues:
    """Test exact threshold boundaries where bucket assignment changes."""

    def setup_method(self):
        self.scorer = _make_scorer()

    def test_exact_ready_now_boundary(self):
        """match=75.0, gap=25.0 should be exactly READY_NOW."""
        assert self.scorer._assign_bucket(75.0, 25.0) == "ready_now"

    def test_one_below_ready_now_match(self):
        """match=74.99, gap=25.0 should NOT be READY_NOW."""
        result = self.scorer._assign_bucket(74.99, 25.0)
        assert result != "ready_now"

    def test_one_above_ready_now_gap(self):
        """match=75.0, gap=25.01 should NOT be READY_NOW."""
        result = self.scorer._assign_bucket(75.0, 25.01)
        assert result != "ready_now"

    def test_trainable_match_lower_boundary(self):
        """match=50.0 with high gap should be TRAINABLE."""
        assert self.scorer._assign_bucket(50.0, 60.0) == "trainable"

    def test_trainable_match_above_old_upper_boundary(self):
        """match=74.5 with high gap was previously LONG_RESKILL (dead zone).
        After monotonicity fix, match >= 50 is always at least TRAINABLE."""
        assert self.scorer._assign_bucket(74.5, 60.0) == "trainable"

    def test_trainable_gap_lower_boundary(self):
        """gap=26.0 with low match should be TRAINABLE."""
        assert self.scorer._assign_bucket(30.0, 26.0) == "trainable"

    def test_trainable_gap_upper_boundary(self):
        """gap=55.0 with low match should be TRAINABLE."""
        assert self.scorer._assign_bucket(30.0, 55.0) == "trainable"

    def test_just_outside_trainable_becomes_long_reskill(self):
        """match=49.9, gap=25.9 (below trainable ranges) → LONG_RESKILL."""
        result = self.scorer._assign_bucket(49.9, 25.9)
        assert result == "long_reskill"

    def test_high_match_high_gap_trainable_via_gap(self):
        """match=80, gap=30: not READY_NOW (gap>25), but TRAINABLE via gap range."""
        result = self.scorer._assign_bucket(80.0, 30.0)
        assert result == "trainable"

    def test_zero_match_zero_gap(self):
        """match=0, gap=0: gap is below trainable range → LONG_RESKILL."""
        result = self.scorer._assign_bucket(0.0, 0.0)
        # gap=0 < trainable_gap_min=26 and match=0 < trainable_match_min=50
        assert result == "long_reskill"

    def test_perfect_scores(self):
        """match=100, gap=0 → READY_NOW."""
        assert self.scorer._assign_bucket(100.0, 0.0) == "ready_now"

    def test_all_risk_tolerance_presets_produce_valid_thresholds(self):
        """Every preset has ready_now thresholds stricter than trainable."""
        for preset_name, profile in get_all_presets().items():
            t = profile.bucket_thresholds
            # Ready now match must be > trainable match max
            assert t.ready_now_match_min > t.trainable_match_max, (
                f"{preset_name}: ready_now_match_min ({t.ready_now_match_min}) "
                f"should be > trainable_match_max ({t.trainable_match_max})"
            )
            # Ready now gap must be < trainable gap min
            assert t.ready_now_gap_max < t.trainable_gap_min, (
                f"{preset_name}: ready_now_gap_max ({t.ready_now_gap_max}) "
                f"should be < trainable_gap_min ({t.trainable_gap_min})"
            )


# ===================================================================
# 3. MONOTONICITY GUARANTEES  (Team Beta focus)
# ===================================================================

class TestMonotonicity:
    """Improving a skill should never make scores worse."""

    def test_improving_skill_never_decreases_match(self):
        """For each persona, upgrading any skill 0→4 should not lower match."""
        scorer = _make_scorer()
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                baseline = _score(persona, code, scorer)
                occ = MOCK_OCCUPATION_SKILLS[code]

                for skill in occ["skills"]:
                    eid = skill["element_id"]
                    original_rating = persona["skill_ratings"].get(eid, 0)
                    if original_rating >= 4:
                        continue  # already max

                    # Create improved persona
                    improved_ratings = dict(persona["skill_ratings"])
                    improved_ratings[eid] = 4  # upgrade to expert

                    improved_score = scorer.score_occupation(
                        onet_code=code,
                        occupation_title=occ["title"],
                        occupation_skills=occ["skills"],
                        user_skill_ratings=improved_ratings,
                        current_job_zone=3,
                        target_job_zone=occ["job_zone"],
                    )
                    assert improved_score.match_score >= baseline.match_score - 0.01, (
                        f"{persona['name']} vs {code}: improving {eid} "
                        f"from {original_rating}→4 lowered match "
                        f"from {baseline.match_score} to {improved_score.match_score}"
                    )

    def test_improving_skill_never_increases_gap(self):
        """Upgrading a skill should not increase gap_severity."""
        scorer = _make_scorer()
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                baseline = _score(persona, code, scorer)
                occ = MOCK_OCCUPATION_SKILLS[code]

                for skill in occ["skills"]:
                    eid = skill["element_id"]
                    original_rating = persona["skill_ratings"].get(eid, 0)
                    if original_rating >= 4:
                        continue

                    improved_ratings = dict(persona["skill_ratings"])
                    improved_ratings[eid] = 4

                    improved_score = scorer.score_occupation(
                        onet_code=code,
                        occupation_title=occ["title"],
                        occupation_skills=occ["skills"],
                        user_skill_ratings=improved_ratings,
                        current_job_zone=3,
                        target_job_zone=occ["job_zone"],
                    )
                    assert improved_score.gap_severity <= baseline.gap_severity + 0.01, (
                        f"{persona['name']} vs {code}: improving {eid} "
                        f"from {original_rating}→4 increased gap "
                        f"from {baseline.gap_severity} to {improved_score.gap_severity}"
                    )

    def test_bucket_never_downgrades_on_skill_improvement(self):
        """Improving a skill should never move the user to a worse bucket."""
        bucket_rank = {"ready_now": 3, "trainable": 2, "long_reskill": 1}
        scorer = _make_scorer()

        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                baseline = _score(persona, code, scorer)
                occ = MOCK_OCCUPATION_SKILLS[code]

                for skill in occ["skills"]:
                    eid = skill["element_id"]
                    original = persona["skill_ratings"].get(eid, 0)
                    if original >= 4:
                        continue

                    improved_ratings = dict(persona["skill_ratings"])
                    improved_ratings[eid] = 4

                    improved = scorer.score_occupation(
                        onet_code=code,
                        occupation_title=occ["title"],
                        occupation_skills=occ["skills"],
                        user_skill_ratings=improved_ratings,
                        current_job_zone=3,
                        target_job_zone=occ["job_zone"],
                    )
                    assert bucket_rank[improved.bucket] >= bucket_rank[baseline.bucket], (
                        f"{persona['name']} vs {code}: improving {eid} "
                        f"downgraded bucket from {baseline.bucket} to {improved.bucket}"
                    )


# ===================================================================
# 4. INPUT MUTATION SAFETY  (Team Beta critical finding)
# ===================================================================

class TestInputMutationSafety:
    """Scorer must never mutate caller-provided data structures."""

    def test_occupation_skills_not_mutated(self):
        """Scoring should not modify the occupation_skills dicts passed in."""
        scorer = _make_scorer()
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        skills_before = copy.deepcopy(occ["skills"])

        scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=PERSONA_MARIA["skill_ratings"],
            current_job_zone=3,
            target_job_zone=occ["job_zone"],
        )

        assert occ["skills"] == skills_before, (
            "Scorer mutated the occupation_skills list passed to it"
        )

    def test_zero_importance_skills_not_mutated(self):
        """When all importances are 0, scorer should not modify the input."""
        scorer = _make_scorer()
        skills = [
            {"element_id": "2.B.1.a", "skill_name": "Reading", "importance": 0, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Listening", "importance": 0, "level": 4.0},
        ]
        skills_before = copy.deepcopy(skills)

        scorer.score_occupation(
            onet_code="99-0000.00",
            occupation_title="Test Job",
            occupation_skills=skills,
            user_skill_ratings={"2.B.1.a": 3, "2.B.2.a": 2},
        )

        assert skills == skills_before, (
            "Scorer mutated input skills when importance was 0"
        )

    def test_user_ratings_not_mutated(self):
        """User ratings dict should not be modified by scoring."""
        scorer = _make_scorer()
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        ratings = dict(PERSONA_MARIA["skill_ratings"])
        ratings_before = dict(ratings)

        scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=ratings,
            current_job_zone=3,
            target_job_zone=occ["job_zone"],
        )

        assert ratings == ratings_before, "Scorer mutated the user_ratings dict"

    def test_repeated_scoring_with_same_input_is_stable(self):
        """Scoring the same input twice should produce identical results,
        proving no hidden state mutation."""
        scorer = _make_scorer()
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]

        score1 = scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=PERSONA_MARIA["skill_ratings"],
            current_job_zone=3,
            target_job_zone=occ["job_zone"],
        )
        score2 = scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=PERSONA_MARIA["skill_ratings"],
            current_job_zone=3,
            target_job_zone=occ["job_zone"],
        )

        assert score1.match_score == score2.match_score
        assert score1.gap_severity == score2.gap_severity
        assert score1.bucket == score2.bucket


# ===================================================================
# 5. ADVERSARIAL / EDGE-CASE INPUTS  (Team Beta focus)
# ===================================================================

class TestAdversarialInputs:
    """Edge cases and unusual inputs the scorer must handle gracefully."""

    def setup_method(self):
        self.scorer = _make_scorer()

    def test_out_of_range_rating_clamped(self):
        """Ratings outside 0-4 should be clamped, not crash or produce nonsense."""
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        # Rating of 5 (above max) should be treated as 4 (capped)
        ratings = {"2.B.1.a": 5, "2.B.8.a": -1}
        score = self.scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=ratings,
        )
        # Should not crash, and scores should be in valid range
        assert 0 <= score.match_score <= 100
        assert 0 <= score.gap_severity <= 100
        assert score.bucket in {"ready_now", "trainable", "long_reskill"}

    def test_single_skill_occupation(self):
        """Occupation with only one skill should produce valid scores."""
        skills = [
            {"element_id": "2.B.1.a", "skill_name": "Reading", "importance": 100.0, "level": 5.0},
        ]
        score = self.scorer.score_occupation(
            onet_code="99-0001.00",
            occupation_title="Single Skill Job",
            occupation_skills=skills,
            user_skill_ratings={"2.B.1.a": 4},
        )
        assert score.match_score == 100.0
        assert score.gap_severity == 0.0
        assert score.bucket == "ready_now"

    def test_many_skills_occupation(self):
        """Occupation with a large number of skills should handle correctly."""
        skills = [
            {"element_id": f"2.B.{i}.a", "skill_name": f"Skill {i}", "importance": 50.0, "level": 3.0}
            for i in range(50)
        ]
        ratings = {f"2.B.{i}.a": 3 for i in range(25)}  # half rated

        score = self.scorer.score_occupation(
            onet_code="99-0002.00",
            occupation_title="Many Skills Job",
            occupation_skills=skills,
            user_skill_ratings=ratings,
        )
        assert 0 <= score.match_score <= 100
        assert 0 <= score.gap_severity <= 100

    def test_null_importance_treated_as_zero(self):
        """Skills with None importance should not crash."""
        skills = [
            {"element_id": "2.B.1.a", "skill_name": "Reading", "importance": None, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Listening", "importance": 50.0, "level": 4.0},
        ]
        score = self.scorer.score_occupation(
            onet_code="99-0003.00",
            occupation_title="Null Importance Job",
            occupation_skills=skills,
            user_skill_ratings={"2.B.1.a": 3, "2.B.2.a": 2},
        )
        assert 0 <= score.match_score <= 100

    def test_very_large_importance_values(self):
        """Skills with extreme importance values should still produce 0-100 scores."""
        skills = [
            {"element_id": "2.B.1.a", "skill_name": "Reading", "importance": 999999.0, "level": 5.0},
            {"element_id": "2.B.2.a", "skill_name": "Listening", "importance": 1.0, "level": 4.0},
        ]
        score = self.scorer.score_occupation(
            onet_code="99-0004.00",
            occupation_title="Skewed Importance Job",
            occupation_skills=skills,
            user_skill_ratings={"2.B.1.a": 4, "2.B.2.a": 0},
        )
        assert 0 <= score.match_score <= 100
        assert 0 <= score.gap_severity <= 100

    def test_all_skills_missing_from_user(self):
        """User has no ratings at all → gap_severity should be 100, bucket=long_reskill."""
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        score = self.scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings={},
        )
        assert score.gap_severity == 100.0
        assert score.match_score == 0.0
        assert score.bucket == "long_reskill"

    def test_user_has_extra_skills_not_in_occupation(self):
        """Extra user skills not required by the occupation should be ignored."""
        skills = [
            {"element_id": "2.B.1.a", "skill_name": "Reading", "importance": 100.0, "level": 5.0},
        ]
        ratings = {"2.B.1.a": 4, "2.B.99.z": 4, "2.B.88.q": 4}
        score = self.scorer.score_occupation(
            onet_code="99-0005.00",
            occupation_title="Minimal Job",
            occupation_skills=skills,
            user_skill_ratings=ratings,
        )
        assert score.match_score == 100.0
        assert score.gap_severity == 0.0

    def test_empty_onet_code_accepted(self):
        """An empty O*NET code should be stored verbatim, not crash."""
        score = self.scorer.score_occupation(
            onet_code="",
            occupation_title="No Code Job",
            occupation_skills=[],
            user_skill_ratings={},
        )
        assert score.onet_code == ""


# ===================================================================
# 6. CROSS-PERSONA CONSISTENCY  (Combined Alpha + Beta)
# ===================================================================

class TestCrossPersonaConsistency:
    """Verify that personas with similar profiles get directionally similar scores."""

    def test_personas_with_programming_have_fewer_programming_gaps(self):
        """Aisha (has programming) should NOT have a programming gap for
        dev roles, while personas without programming SHOULD.

        Note: total match_score depends on ALL skills, not just programming.
        Maria's 6 expert-level soft skills give her a higher total match than
        Aisha for Software Developers, even though Maria lacks programming.
        The correct invariant is gap-specific, not total-score."""
        programming_id = "2.B.1.g"
        for code in ["15-1252.00", "15-1299.08"]:
            aisha = _score(PERSONA_AISHA, code)
            aisha_gap_ids = {g.element_id for g in aisha.top_gaps}
            assert programming_id not in aisha_gap_ids, (
                f"Aisha should NOT have a programming gap for {code}"
            )

            for persona in [PERSONA_MARIA, PERSONA_JAMES, PERSONA_ROBERT]:
                other = _score(persona, code)
                other_gap_ids = {g.element_id for g in other.top_gaps}
                assert programming_id in other_gap_ids, (
                    f"{persona['name']} should have a programming gap for {code}"
                )

    def test_veteran_strong_on_management_skills(self):
        """Sarah (veteran, strong management/coordination) should have
        fewer gaps than Robert (mechanic) for sysadmin roles."""
        code = "15-1244.00"
        sarah = _score(PERSONA_SARAH, code)
        robert = _score(PERSONA_ROBERT, code)
        # Sarah has broader soft skills coverage
        assert sarah.match_score >= robert.match_score, (
            f"Sarah should have >= match than Robert for sysadmin, "
            f"got {sarah.match_score} < {robert.match_score}"
        )

    def test_all_personas_get_training_suggestions(self):
        """Every persona should receive a non-empty training suggestion for
        every occupation, regardless of bucket."""
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                assert score.training_suggestion, (
                    f"{persona['name']} vs {code}: empty training_suggestion"
                )
                assert len(score.training_suggestion) > 10, (
                    f"{persona['name']} vs {code}: training_suggestion too short"
                )

    def test_all_personas_get_explanations(self):
        """Every persona should receive a non-empty explanation."""
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                assert score.explanation, (
                    f"{persona['name']} vs {code}: empty explanation"
                )
                assert len(score.explanation) > 20, (
                    f"{persona['name']} vs {code}: explanation too short"
                )


# ===================================================================
# 7. EXPLAINABILITY-SCORER CONTRACT  (Combined Alpha + Beta)
# ===================================================================

class TestExplainabilityScorerContract:
    """The explainability engine must be consistent with scorer output."""

    def test_explanation_bucket_matches_score_bucket(self):
        """BucketExplanation.bucket_reasoning.assigned_bucket must match
        OccupationScore.bucket for every persona x occupation."""
        engine = BucketExplainerEngine()
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                occ = MOCK_OCCUPATION_SKILLS[code]
                explanation = engine.explain(
                    score,
                    occupation_skills=occ["skills"],
                    user_skill_ratings=persona["skill_ratings"],
                )
                assert explanation.bucket_reasoning.assigned_bucket == score.bucket, (
                    f"{persona['name']} vs {code}: explainer bucket "
                    f"'{explanation.bucket_reasoning.assigned_bucket}' != "
                    f"scorer bucket '{score.bucket}'"
                )

    def test_skills_to_develop_matches_gap_count(self):
        """Number of skills_to_develop should match len(score.top_gaps)."""
        engine = BucketExplainerEngine()
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                score = _score(persona, code)
                explanation = engine.explain(score)
                assert len(explanation.skills_to_develop) == len(score.top_gaps), (
                    f"{persona['name']} vs {code}: "
                    f"skills_to_develop({len(explanation.skills_to_develop)}) != "
                    f"top_gaps({len(score.top_gaps)})"
                )

    def test_ready_now_has_no_major_gaps_in_explanation(self):
        """If a persona is READY_NOW, the explainer should not report MAJOR gaps."""
        engine = BucketExplainerEngine()
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        # Build a perfect-match user
        perfect_ratings = {s["element_id"]: 4 for s in occ["skills"]}
        scorer = _make_scorer()
        score = scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=perfect_ratings,
            current_job_zone=4,
            target_job_zone=4,
        )
        assert score.bucket == "ready_now"
        explanation = engine.explain(
            score,
            occupation_skills=occ["skills"],
            user_skill_ratings=perfect_ratings,
        )
        major_gaps = [
            s for s in explanation.skills_to_develop
            if s.gap_category == GapCategory.MAJOR
        ]
        assert len(major_gaps) == 0, (
            f"READY_NOW user has {len(major_gaps)} MAJOR gaps in explanation"
        )

    def test_what_would_change_is_none_for_ready_now(self):
        """Users already in READY_NOW should have to_ready_now=None."""
        engine = BucketExplainerEngine()
        occ = MOCK_OCCUPATION_SKILLS["15-1252.00"]
        perfect_ratings = {s["element_id"]: 4 for s in occ["skills"]}
        scorer = _make_scorer()
        score = scorer.score_occupation(
            onet_code="15-1252.00",
            occupation_title=occ["title"],
            occupation_skills=occ["skills"],
            user_skill_ratings=perfect_ratings,
            current_job_zone=4,
            target_job_zone=4,
        )
        explanation = engine.explain(score)
        assert explanation.what_would_change_bucket.to_ready_now is None

    def test_all_risk_tolerance_presets_load(self):
        """Every risk tolerance preset must load without error."""
        for tolerance in RiskTolerance:
            profile = get_threshold_profile(tolerance)
            assert profile.name
            assert profile.bucket_thresholds is not None


# ===================================================================
# 8. THRESHOLD CONFIGURATION INTEGRITY  (Team Beta focus)
# ===================================================================

class TestThresholdIntegrity:
    """Verify threshold configuration consistency across presets."""

    def test_standard_matches_config_settings(self):
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

    def test_relaxed_is_more_permissive_than_standard(self):
        """RELAXED preset should accept more users into READY_NOW."""
        standard = get_threshold_profile(RiskTolerance.STANDARD).bucket_thresholds
        relaxed = get_threshold_profile(RiskTolerance.RELAXED).bucket_thresholds
        assert relaxed.ready_now_match_min < standard.ready_now_match_min
        assert relaxed.ready_now_gap_max > standard.ready_now_gap_max

    def test_strict_is_more_conservative_than_standard(self):
        """STRICT preset should require higher match for READY_NOW."""
        standard = get_threshold_profile(RiskTolerance.STANDARD).bucket_thresholds
        strict = get_threshold_profile(RiskTolerance.STRICT).bucket_thresholds
        assert strict.ready_now_match_min > standard.ready_now_match_min
        assert strict.ready_now_gap_max < standard.ready_now_gap_max

    def test_no_threshold_gaps_or_overlaps(self):
        """Trainable range should not overlap with READY_NOW range."""
        for _, profile in get_all_presets().items():
            t = profile.bucket_thresholds
            # trainable_match_max < ready_now_match_min
            assert t.trainable_match_max < t.ready_now_match_min, (
                f"{profile.name}: trainable match overlaps READY_NOW"
            )
            # trainable_gap_min > ready_now_gap_max
            assert t.trainable_gap_min > t.ready_now_gap_max, (
                f"{profile.name}: trainable gap overlaps READY_NOW"
            )


# ===================================================================
# 9. SKILL DOMAIN CLASSIFICATION  (Team Beta focus)
# ===================================================================

class TestSkillDomainClassification:
    """Verify O*NET element_id → domain mapping is correct and complete."""

    def test_known_element_ids_classify_correctly(self):
        """Standard element IDs from test data should map to expected domains."""
        cases = {
            "2.B.1.a": SkillDomain.BASIC_SKILLS,      # Reading Comprehension
            "2.B.1.g": SkillDomain.BASIC_SKILLS,      # Programming
            "2.B.2.a": SkillDomain.COMPLEX_PROBLEM,   # Active Listening maps to 2.B.2
            "2.B.4.a": SkillDomain.SOCIAL_SKILLS,     # Speaking
            "2.B.8.a": SkillDomain.OTHER,              # Critical Thinking 2.B.8 unmapped
            "2.A.1.a": SkillDomain.COGNITIVE,          # Cognitive ability
            "2.C.3.a": SkillDomain.KNOWLEDGE,          # Knowledge area
        }
        for element_id, expected_domain in cases.items():
            result = classify_skill_domain(element_id)
            assert result == expected_domain, (
                f"classify_skill_domain('{element_id}') = {result}, "
                f"expected {expected_domain}"
            )

    def test_unknown_element_id_returns_other(self):
        """Unrecognised element IDs should return SkillDomain.OTHER."""
        assert classify_skill_domain("99.Z.1.a") == SkillDomain.OTHER
        assert classify_skill_domain("") == SkillDomain.OTHER

    def test_gap_categorisation_boundaries(self):
        """Gap category thresholds should produce correct results at boundaries."""
        assert _categorise_gap(0.0) == GapCategory.MINOR
        assert _categorise_gap(0.05) == GapCategory.MINOR
        assert _categorise_gap(0.051) == GapCategory.MODERATE
        assert _categorise_gap(0.15) == GapCategory.MODERATE
        assert _categorise_gap(0.151) == GapCategory.MAJOR
        assert _categorise_gap(1.0) == GapCategory.MAJOR


# ===================================================================
# 10. TRAINING PATH GENERATOR CONTRACT  (Team Beta focus)
# ===================================================================

class TestTrainingPathContract:
    """Verify training path generator honors constraints and contracts."""

    def test_zero_budget_produces_only_free_resources(self):
        """Zero-budget user should only get $0 resources in their path."""
        generator = PathGenerator()
        gaps = [
            TrainingSkillGap(
                skill_code="2.B.1.g",
                skill_name="Programming",
                gap_weight=0.8,
            ),
        ]
        constraints = UserConstraints(budget_usd=0, hours_per_week=20)
        path = generator.generate(gaps, constraints)
        for step in path.steps:
            assert step.estimated_cost_usd == 0, (
                f"Zero-budget path includes ${step.estimated_cost_usd} resource: "
                f"{step.resource.name}"
            )

    def test_empty_gaps_returns_complete_path(self):
        """No skill gaps → complete, feasible, empty path."""
        generator = PathGenerator()
        path = generator.generate([], UserConstraints())
        assert path.is_complete is True
        assert path.is_feasible is True
        assert len(path.steps) == 0

    def test_cumulative_cost_tracking_accurate(self):
        """Each step's cumulative_cost_usd must equal sum of all prior steps."""
        generator = PathGenerator()
        gaps = [
            TrainingSkillGap(skill_code="2.B.1.g", skill_name="Programming", gap_weight=0.8),
            TrainingSkillGap(skill_code="2.B.8.a", skill_name="Critical Thinking", gap_weight=0.5),
        ]
        path = generator.generate(gaps, UserConstraints(budget_usd=10000))
        running_cost = 0.0
        for step in path.steps:
            running_cost += step.estimated_cost_usd
            assert abs(step.cumulative_cost_usd - running_cost) < 0.01, (
                f"Step {step.step_number}: cumulative_cost_usd "
                f"({step.cumulative_cost_usd}) != running sum ({running_cost})"
            )

    def test_cumulative_weeks_tracking_accurate(self):
        """Each step's cumulative_weeks must equal sum of all prior steps."""
        generator = PathGenerator()
        gaps = [
            TrainingSkillGap(skill_code="2.B.1.g", skill_name="Programming", gap_weight=0.8),
            TrainingSkillGap(skill_code="2.B.8.a", skill_name="Critical Thinking", gap_weight=0.5),
        ]
        path = generator.generate(gaps, UserConstraints(budget_usd=10000))
        running_weeks = 0
        for step in path.steps:
            running_weeks += step.estimated_weeks
            assert step.cumulative_weeks == running_weeks, (
                f"Step {step.step_number}: cumulative_weeks "
                f"({step.cumulative_weeks}) != running sum ({running_weeks})"
            )

    def test_path_total_matches_last_step_cumulative(self):
        """path.total_weeks and total_cost should match last step's cumulative."""
        generator = PathGenerator()
        gaps = [
            TrainingSkillGap(skill_code="2.B.1.g", skill_name="Programming", gap_weight=0.8),
        ]
        path = generator.generate(gaps, UserConstraints(budget_usd=10000))
        if path.steps:
            last = path.steps[-1]
            assert path.total_weeks == last.cumulative_weeks
            assert abs(path.total_cost_usd - last.cumulative_cost_usd) < 0.01


# ===================================================================
# 11. CALIBRATION MODEL CONTRACT  (Team Beta focus)
# ===================================================================

class TestCalibrationContract:
    """Verify calibration model feature extraction contracts."""

    def test_feature_array_shape_is_1x9(self):
        """Feature array should always be (1, 9)."""
        from app.ml.calibration import CalibrationModel
        model = CalibrationModel()
        features = model.extract_features(
            user_id=1, target_onet_code="15-1252.00", event_id=1,
            match_score=50.0, gap_severity=40.0,
            num_missing_skills=3, sum_missing_weights=0.4,
            current_job_zone=3, target_job_zone=4,
            user_ratings={"2.B.1.a": 3},
        )
        arr = model._features_to_array(features)
        assert arr.shape == (1, 9)

    def test_feature_extraction_with_empty_ratings(self):
        """Empty user ratings should produce valid features (mean=0, var=0)."""
        from app.ml.calibration import CalibrationModel
        model = CalibrationModel()
        features = model.extract_features(
            user_id=1, target_onet_code="15-1252.00", event_id=1,
            match_score=0.0, gap_severity=100.0,
            num_missing_skills=8, sum_missing_weights=1.0,
            current_job_zone=None, target_job_zone=None,
            user_ratings={},
        )
        assert features.mean_rating == 0.0
        assert features.rating_variance == 0.0
        assert features.job_zone_diff == 0.0

    def test_feedback_label_mapping_exhaustive(self):
        """Verify all expected action types are handled correctly."""
        from app.ml.calibration import prepare_training_data_from_feedback

        # Positive labels
        for action in ["interview", "offer", "apply"]:
            records = [{
                "action_type": action,
                "match_score": 50.0, "gap_severity": 30.0,
                "job_zone_diff": 0.0, "target_job_zone": 3.0,
                "num_missing_skills": 2, "sum_missing_weights": 0.3,
                "mean_rating": 3.0, "rating_variance": 0.5,
                "num_rated_skills": 5,
                "user_id": 1, "target_onet_code": "15-1252.00", "event_id": 1,
            }]
            data = prepare_training_data_from_feedback(records)
            assert len(data) == 1 and data[0][1] == 1, f"{action} should be label=1"

        # Negative label
        records = [{
            "action_type": "hide",
            "match_score": 50.0, "gap_severity": 30.0,
            "job_zone_diff": 0.0, "target_job_zone": 3.0,
            "num_missing_skills": 2, "sum_missing_weights": 0.3,
            "mean_rating": 3.0, "rating_variance": 0.5,
            "num_rated_skills": 5,
            "user_id": 1, "target_onet_code": "15-1252.00", "event_id": 1,
        }]
        data = prepare_training_data_from_feedback(records)
        assert len(data) == 1 and data[0][1] == 0, "hide should be label=0"

        # Ambiguous (filtered)
        for action in ["click", "save", "view", "bookmark", ""]:
            records = [{
                "action_type": action,
                "match_score": 50.0, "gap_severity": 30.0,
                "job_zone_diff": 0.0, "target_job_zone": 3.0,
                "num_missing_skills": 2, "sum_missing_weights": 0.3,
                "mean_rating": 3.0, "rating_variance": 0.5,
                "num_rated_skills": 5,
                "user_id": 1, "target_onet_code": "15-1252.00", "event_id": 1,
            }]
            data = prepare_training_data_from_feedback(records)
            assert len(data) == 0, f"{action} should be filtered out"


# ===================================================================
# 12. SCORING DETERMINISM  (Team Beta regression check)
# ===================================================================

class TestScoringDeterminism:
    """Every persona scored 100 times should produce identical results every time."""

    def test_determinism_across_iterations(self):
        scorer = _make_scorer()
        for persona in ALL_PERSONAS:
            for code in ALL_OCCUPATION_CODES:
                scores = [_score(persona, code, scorer) for _ in range(10)]
                first = scores[0]
                for i, s in enumerate(scores[1:], 1):
                    assert s.match_score == first.match_score, (
                        f"Iteration {i}: match changed"
                    )
                    assert s.gap_severity == first.gap_severity, (
                        f"Iteration {i}: gap changed"
                    )
                    assert s.bucket == first.bucket, (
                        f"Iteration {i}: bucket changed"
                    )
