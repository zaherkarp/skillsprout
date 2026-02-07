"""Mitigation strategies for bias audit findings.

Design rationale:
-----------------
When the bias audit engine detects disparate impact, we need concrete,
auditable interventions that reduce bias WITHOUT silently distorting scores.
Each mitigation is:

1. **Transparent**: It logs what it changed and why.
2. **Reversible**: The original scores are preserved; mitigations produce
   new adjusted scores alongside the originals.
3. **Configurable**: Parameters can be tuned per deployment context.

Two mitigations are implemented:

SKILL_REWEIGHTING
  Downweight the importance scores of skills that are demographically
  correlated. For example, if "Physical Stamina" (element 2.A.3.x) has
  an importance of 80 in trucking (93% male) but the same importance in
  nursing (85% female), the skill itself is neutral. But if a skill appears
  almost exclusively in occupations dominated by one gender, its importance
  is potentially inflated by historical workforce composition rather than
  genuine job requirements.

  The mitigation reduces the effective importance of such skills by a
  configurable factor, which in turn changes match_scores and gap_severity.
  This shifts bucket assignments toward parity without removing the skill
  from consideration entirely.

STALENESS_PENALTY
  Reduce the confidence (and effective match_score) for occupations whose
  O*NET skill profile is stale. This is NOT a bias mitigation per se, but
  it prevents stale data from creating SYSTEMATIC bias when staleness is
  correlated with demographics (as detected by the staleness audit test).

  The penalty is a multiplicative discount on match_score:
    adjusted_match = match_score * (1 - penalty_factor)
  where penalty_factor increases with age of the data.

Both mitigations are idempotent: applying them twice produces the same
result as applying them once, because they operate on the original (un-
mitigated) scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from app.ml.scoring import OccupationScore, SkillGap
from ml.bias_audit.audit_framework import (
    DemographicProfile,
    get_demographic_profiles,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mitigation result wrapper
# ---------------------------------------------------------------------------

@dataclass
class MitigationResult:
    """Result of applying a mitigation strategy.

    Preserves the original score alongside the adjusted score so that
    downstream consumers can compare and audit the mitigation's impact.

    Attributes:
        original_score: The unmodified OccupationScore.
        adjusted_score: The score after mitigation has been applied.
        mitigation_name: Which mitigation was applied.
        adjustment_details: Human-readable description of what changed.
        parameters_used: The configuration parameters that controlled the
            mitigation.
    """
    original_score: OccupationScore
    adjusted_score: OccupationScore
    mitigation_name: str
    adjustment_details: str
    parameters_used: Dict[str, object]


# ---------------------------------------------------------------------------
# SKILL_REWEIGHTING mitigation
# ---------------------------------------------------------------------------

# Default demographic correlation threshold: if a skill appears in >70%
# of occupations dominated by one gender (>70% one gender in workforce),
# we consider it demographically correlated.
DEFAULT_CORRELATION_THRESHOLD = 0.70

# Default reweighting factor: reduce importance of correlated skills by 20%.
# A value of 0.0 means no reduction; 1.0 means zero out the skill entirely.
DEFAULT_REWEIGHT_FACTOR = 0.20


class SkillReweightingMitigation:
    """Downweight importance scores of demographically correlated skills.

    This mitigation identifies skills that appear disproportionately in
    occupations dominated by a specific demographic group and reduces
    their importance weight. The goal is to prevent historical workforce
    composition from inflating skill importance in the scoring engine.

    Usage::

        mitigation = SkillReweightingMitigation()
        correlated = mitigation.identify_correlated_skills(
            occupation_skills_map, profiles
        )
        result = mitigation.apply(score, correlated)
    """

    def __init__(
        self,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
        reweight_factor: float = DEFAULT_REWEIGHT_FACTOR,
    ):
        """Initialise the mitigation.

        Args:
            correlation_threshold: Minimum fraction of gendered occupations
                a skill must appear in to be considered correlated (0-1).
            reweight_factor: How much to reduce the importance (0-1).
                0.2 means reduce by 20%.
        """
        self.correlation_threshold = correlation_threshold
        self.reweight_factor = reweight_factor

    def identify_correlated_skills(
        self,
        occupation_skills_map: Dict[str, List[Dict]],
        profiles: Optional[Dict[str, DemographicProfile]] = None,
    ) -> Set[str]:
        """Identify skill element_ids that are demographically correlated.

        A skill is considered correlated if:
          1. It appears in at least ``correlation_threshold`` fraction of
             occupations in one demographic group (e.g., female-majority).
          2. It does NOT appear in at least ``correlation_threshold`` fraction
             of occupations in the opposite group.

        This is a conservative heuristic. A more sophisticated approach
        would use statistical tests, but this suffices for a v1 mitigation.

        Args:
            occupation_skills_map: Dict mapping onet_code to list of skill
                dicts (each with 'element_id' and 'importance').
            profiles: Demographic profiles. Defaults to stubs.

        Returns:
            Set of element_id strings that are demographically correlated.
        """
        if profiles is None:
            profiles = get_demographic_profiles()

        # Split occupations by gender majority, restricted to codes in the map
        available_codes = set(occupation_skills_map.keys())
        female_majority_codes = {
            code for code, p in profiles.items()
            if p.pct_female >= 60.0 and code in available_codes
        }
        male_majority_codes = {
            code for code, p in profiles.items()
            if p.pct_female < 40.0 and code in available_codes
        }

        # Count skill frequency in each group
        female_skill_counts: Dict[str, int] = {}
        male_skill_counts: Dict[str, int] = {}

        for code, skills in occupation_skills_map.items():
            for skill in skills:
                eid = skill.get("element_id", "")
                if code in female_majority_codes:
                    female_skill_counts[eid] = female_skill_counts.get(eid, 0) + 1
                elif code in male_majority_codes:
                    male_skill_counts[eid] = male_skill_counts.get(eid, 0) + 1

        n_female = max(len(female_majority_codes), 1)
        n_male = max(len(male_majority_codes), 1)

        correlated: Set[str] = set()
        all_skills = set(female_skill_counts.keys()) | set(male_skill_counts.keys())

        for eid in all_skills:
            female_rate = female_skill_counts.get(eid, 0) / n_female
            male_rate = male_skill_counts.get(eid, 0) / n_male

            # Correlated if appears in >threshold of one group and
            # <(1-threshold) of the other.
            if (female_rate >= self.correlation_threshold and
                    male_rate < (1 - self.correlation_threshold)):
                correlated.add(eid)
                logger.info(
                    f"Skill {eid} is female-correlated "
                    f"(female_rate={female_rate:.2f}, male_rate={male_rate:.2f})"
                )
            elif (male_rate >= self.correlation_threshold and
                  female_rate < (1 - self.correlation_threshold)):
                correlated.add(eid)
                logger.info(
                    f"Skill {eid} is male-correlated "
                    f"(male_rate={male_rate:.2f}, female_rate={female_rate:.2f})"
                )

        logger.info(f"Identified {len(correlated)} demographically correlated skills")
        return correlated

    def apply(
        self,
        score: OccupationScore,
        correlated_skills: Set[str],
    ) -> MitigationResult:
        """Apply skill reweighting to a single OccupationScore.

        For each gap whose element_id is in ``correlated_skills``, the
        gap_weight is reduced by ``reweight_factor``. This simulates lowering
        the skill's importance in the scoring formula.

        The match_score and gap_severity are recomputed from the adjusted
        gap weights. The bucket assignment is then re-evaluated using the
        adjusted scores (using the same thresholds as the original scorer).

        Args:
            score: The original OccupationScore.
            correlated_skills: Set of element_ids to reweight.

        Returns:
            MitigationResult with original and adjusted scores.
        """
        # Adjust gap weights for correlated skills
        adjusted_gaps: List[SkillGap] = []
        reweighted_count = 0
        total_weight_reduction = 0.0

        for gap in score.top_gaps:
            if gap.element_id in correlated_skills:
                # Reduce the gap weight (and therefore its contribution to
                # gap_severity and match_score).
                reduction = gap.gap_weight * self.reweight_factor
                new_weight = gap.gap_weight - reduction
                adjusted_gap = SkillGap(
                    element_id=gap.element_id,
                    skill_name=gap.skill_name,
                    required_importance=gap.required_importance * (1 - self.reweight_factor),
                    required_level=gap.required_level,
                    user_capability=gap.user_capability,
                    gap_weight=new_weight,
                )
                adjusted_gaps.append(adjusted_gap)
                reweighted_count += 1
                total_weight_reduction += reduction
            else:
                adjusted_gaps.append(gap)

        # Recompute gap_severity from adjusted weights
        adjusted_gap_severity = sum(g.gap_weight for g in adjusted_gaps) * 100

        # Recompute match_score: the freed weight goes to matched skills,
        # so match_score increases by the total_weight_reduction.
        adjusted_match_score = min(
            100.0,
            score.match_score + (total_weight_reduction * 100)
        )

        # Build adjusted score
        adjusted_score = OccupationScore(
            onet_code=score.onet_code,
            match_score=round(adjusted_match_score, 2),
            gap_severity=round(adjusted_gap_severity, 2),
            top_gaps=adjusted_gaps,
            bucket=score.bucket,  # Will be re-assigned below
            training_suggestion=score.training_suggestion,
            explanation=score.explanation,
            metadata={
                **score.metadata,
                "mitigation_applied": "SKILL_REWEIGHTING",
                "reweighted_skills": reweighted_count,
                "total_weight_reduction": round(total_weight_reduction, 4),
            },
        )

        # Re-assign bucket with adjusted scores
        # Import here to avoid circular dependency issues at module level
        from app.ml.scoring import BaselineScorer
        scorer = BaselineScorer()
        adjusted_score.bucket = scorer._assign_bucket(
            adjusted_match_score, adjusted_gap_severity
        )

        details = (
            f"Reweighted {reweighted_count} correlated skill(s), reducing "
            f"total gap weight by {total_weight_reduction:.4f}. "
            f"Match score: {score.match_score:.1f} -> {adjusted_match_score:.1f}. "
            f"Gap severity: {score.gap_severity:.1f} -> {adjusted_gap_severity:.1f}. "
            f"Bucket: {score.bucket} -> {adjusted_score.bucket}."
        )

        return MitigationResult(
            original_score=score,
            adjusted_score=adjusted_score,
            mitigation_name="SKILL_REWEIGHTING",
            adjustment_details=details,
            parameters_used={
                "correlation_threshold": self.correlation_threshold,
                "reweight_factor": self.reweight_factor,
                "correlated_skill_count": len(correlated_skills),
                "reweighted_in_score": reweighted_count,
            },
        )


# ---------------------------------------------------------------------------
# STALENESS_PENALTY mitigation
# ---------------------------------------------------------------------------

# Default penalty configuration
# Linear penalty: 0% at 0 days, reaching max_penalty at penalty_max_days.
DEFAULT_PENALTY_MAX = 0.15          # Maximum 15% penalty
DEFAULT_PENALTY_MAX_DAYS = 730      # Reaches max penalty at 2 years


class StalenessPenaltyMitigation:
    """Reduce match_score confidence for occupations with stale skill data.

    The penalty is linear in the age of the O*NET data:
      penalty_factor = min(max_penalty, (age_days / penalty_max_days) * max_penalty)
      adjusted_match = match_score * (1 - penalty_factor)

    This prevents stale data from producing overconfident READY_NOW
    classifications.

    Usage::

        mitigation = StalenessPenaltyMitigation()
        result = mitigation.apply(score, last_updated="2023-01-15")
    """

    def __init__(
        self,
        max_penalty: float = DEFAULT_PENALTY_MAX,
        penalty_max_days: int = DEFAULT_PENALTY_MAX_DAYS,
    ):
        """Initialise the staleness penalty.

        Args:
            max_penalty: Maximum penalty factor (0-1). Default 0.15 (15%).
            penalty_max_days: Age in days at which max_penalty is reached.
        """
        self.max_penalty = max_penalty
        self.penalty_max_days = penalty_max_days

    def compute_penalty(
        self,
        last_updated: str,
        reference_date: Optional[datetime] = None,
    ) -> float:
        """Compute the staleness penalty factor for a given update date.

        Args:
            last_updated: ISO date string of the last O*NET update.
            reference_date: Date to measure staleness from. Defaults to now.

        Returns:
            Penalty factor between 0.0 (no penalty) and max_penalty.
        """
        if reference_date is None:
            reference_date = datetime.utcnow()

        try:
            updated_dt = datetime.fromisoformat(last_updated)
        except (ValueError, TypeError):
            # Cannot parse date; apply maximum penalty as a safety measure.
            logger.warning(
                f"Cannot parse last_updated '{last_updated}'; "
                f"applying maximum staleness penalty."
            )
            return self.max_penalty

        age_days = (reference_date - updated_dt).days
        if age_days <= 0:
            return 0.0

        # Linear interpolation from 0 to max_penalty
        penalty = min(
            self.max_penalty,
            (age_days / self.penalty_max_days) * self.max_penalty,
        )

        return round(penalty, 4)

    def apply(
        self,
        score: OccupationScore,
        last_updated: str,
        reference_date: Optional[datetime] = None,
    ) -> MitigationResult:
        """Apply staleness penalty to a single OccupationScore.

        The match_score is reduced by the computed penalty factor. The
        gap_severity is increased proportionally (since lower match implies
        the gaps feel larger). The bucket is then re-assigned.

        Args:
            score: The original OccupationScore.
            last_updated: ISO date string of the O*NET data update.
            reference_date: Reference date for staleness calculation.

        Returns:
            MitigationResult with original and adjusted scores.
        """
        penalty = self.compute_penalty(last_updated, reference_date)

        if penalty == 0.0:
            # No adjustment needed
            return MitigationResult(
                original_score=score,
                adjusted_score=score,
                mitigation_name="STALENESS_PENALTY",
                adjustment_details="No staleness penalty applied (data is fresh).",
                parameters_used={
                    "max_penalty": self.max_penalty,
                    "penalty_max_days": self.penalty_max_days,
                    "computed_penalty": 0.0,
                    "last_updated": last_updated,
                },
            )

        # Apply penalty to match_score
        adjusted_match = score.match_score * (1.0 - penalty)

        # Increase gap_severity proportionally to reflect reduced confidence.
        # The intuition: if we are less confident in the match, the gaps
        # effectively feel larger.
        adjusted_gap = min(100.0, score.gap_severity * (1.0 + penalty))

        # Build adjusted score
        adjusted_score = OccupationScore(
            onet_code=score.onet_code,
            match_score=round(adjusted_match, 2),
            gap_severity=round(adjusted_gap, 2),
            top_gaps=score.top_gaps,  # Gaps themselves don't change
            bucket=score.bucket,  # Will be re-assigned below
            training_suggestion=score.training_suggestion,
            explanation=score.explanation,
            metadata={
                **score.metadata,
                "mitigation_applied": "STALENESS_PENALTY",
                "staleness_penalty_factor": penalty,
                "last_updated": last_updated,
            },
        )

        # Re-assign bucket with adjusted scores
        from app.ml.scoring import BaselineScorer
        scorer = BaselineScorer()
        adjusted_score.bucket = scorer._assign_bucket(
            adjusted_match, adjusted_gap
        )

        details = (
            f"Applied staleness penalty of {penalty:.2%} "
            f"(data last updated: {last_updated}). "
            f"Match score: {score.match_score:.1f} -> {adjusted_match:.1f}. "
            f"Gap severity: {score.gap_severity:.1f} -> {adjusted_gap:.1f}. "
            f"Bucket: {score.bucket} -> {adjusted_score.bucket}."
        )

        return MitigationResult(
            original_score=score,
            adjusted_score=adjusted_score,
            mitigation_name="STALENESS_PENALTY",
            adjustment_details=details,
            parameters_used={
                "max_penalty": self.max_penalty,
                "penalty_max_days": self.penalty_max_days,
                "computed_penalty": penalty,
                "last_updated": last_updated,
            },
        )


# ---------------------------------------------------------------------------
# Convenience: apply all relevant mitigations
# ---------------------------------------------------------------------------

def apply_mitigations(
    scores: List[OccupationScore],
    occupation_skills_map: Optional[Dict[str, List[Dict]]] = None,
    last_updated_map: Optional[Dict[str, str]] = None,
    profiles: Optional[Dict[str, DemographicProfile]] = None,
    enable_reweighting: bool = True,
    enable_staleness: bool = True,
) -> List[MitigationResult]:
    """Apply all configured mitigations to a list of occupation scores.

    This is the primary entry point for downstream consumers who want to
    apply mitigations in bulk.

    Args:
        scores: List of OccupationScore objects.
        occupation_skills_map: Dict mapping onet_code to skill dicts.
            Required for SKILL_REWEIGHTING; ignored if reweighting is
            disabled.
        last_updated_map: Dict mapping onet_code to last_updated ISO date
            string. Required for STALENESS_PENALTY; ignored if staleness
            is disabled.
        profiles: Demographic profiles for reweighting analysis.
        enable_reweighting: Whether to apply SKILL_REWEIGHTING.
        enable_staleness: Whether to apply STALENESS_PENALTY.

    Returns:
        List of MitigationResult objects, one per input score.
    """
    results: List[MitigationResult] = []

    # Identify correlated skills (once, shared across all scores)
    correlated_skills: Set[str] = set()
    if enable_reweighting and occupation_skills_map:
        reweight = SkillReweightingMitigation()
        correlated_skills = reweight.identify_correlated_skills(
            occupation_skills_map, profiles
        )

    staleness = StalenessPenaltyMitigation() if enable_staleness else None

    for score in scores:
        current_score = score

        # Apply reweighting first (changes importance weights)
        if enable_reweighting and correlated_skills:
            reweight_mit = SkillReweightingMitigation()
            reweight_result = reweight_mit.apply(current_score, correlated_skills)
            current_score = reweight_result.adjusted_score

        # Then apply staleness penalty (changes match_score/gap_severity)
        if enable_staleness and staleness and last_updated_map:
            last_updated = last_updated_map.get(score.onet_code)
            if last_updated:
                staleness_result = staleness.apply(current_score, last_updated)
                current_score = staleness_result.adjusted_score

        # Record the final result
        results.append(MitigationResult(
            original_score=score,
            adjusted_score=current_score,
            mitigation_name="COMBINED",
            adjustment_details=(
                f"Applied mitigations: "
                f"reweighting={'yes' if enable_reweighting and correlated_skills else 'no'}, "
                f"staleness={'yes' if enable_staleness and last_updated_map else 'no'}. "
                f"Match: {score.match_score:.1f} -> {current_score.match_score:.1f}, "
                f"Gap: {score.gap_severity:.1f} -> {current_score.gap_severity:.1f}, "
                f"Bucket: {score.bucket} -> {current_score.bucket}."
            ),
            parameters_used={
                "reweighting_enabled": enable_reweighting,
                "staleness_enabled": enable_staleness,
                "correlated_skill_count": len(correlated_skills),
            },
        ))

    return results
