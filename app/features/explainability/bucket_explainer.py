"""Structured explainability engine for scored occupation recommendations.

Design rationale:
-----------------
The baseline scorer (app.ml.scoring) produces an OccupationScore with a
human-readable ``explanation`` string. That is sufficient for a v1 UI, but
it does not support:

1. **Structured rendering**: front-ends need machine-parseable fields to build
   comparison tables, progress bars, and gap-closing checklists.
2. **Threshold transparency**: users should see *why* they landed in a bucket
   ("Your gap severity of 32% exceeds the 25% ceiling for Ready Now").
3. **Actionable next steps**: for each gap, we want to show the estimated
   training time and *what would change* if the user closed that gap.

This module generates a ``BucketExplanation`` dataclass that carries all of
this information in a structured, JSON-serialisable form.

Key design decisions:
  - We never duplicate threshold values. All thresholds are read from
    ``ThresholdProfile`` (which in turn reads from ``app.core.config.settings``
    for the STANDARD preset).
  - Gap categorisation uses a simple heuristic based on gap_weight. Production
    systems would refine this with O*NET-specific training-time data.
  - ``what_would_change_bucket`` is computed by simulating hypothetical score
    improvements, giving users concrete motivation ("If you closed 2 gaps,
    you would move from TRAINABLE to READY_NOW").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.ml.scoring import OccupationScore, SkillGap
from app.features.explainability.threshold_config import (
    BucketThresholds,
    RiskTolerance,
    SkillDomain,
    ThresholdProfile,
    classify_skill_domain,
    get_threshold_profile,
)


# ---------------------------------------------------------------------------
# Gap categorisation heuristics
# ---------------------------------------------------------------------------
# We bucket individual gaps into three severity tiers based on their
# gap_weight (normalised importance). The thresholds are chosen so that:
#   - MINOR: the skill matters but is not dominant. Usually a weekend course.
#   - MODERATE: meaningful gap. Requires sustained study (weeks to months).
#   - MAJOR: high-importance skill the user lacks entirely. Formal training.

_GAP_MINOR_CEILING = 0.05    # gap_weight <= 5% of total importance
_GAP_MODERATE_CEILING = 0.15  # gap_weight <= 15% of total importance
# Anything above 15% is MAJOR.


class GapCategory:
    """String constants for gap severity categories."""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


def _categorise_gap(gap_weight: float) -> str:
    """Assign a gap to a severity category based on its normalised weight.

    Args:
        gap_weight: The gap's weight (importance / total_importance), 0-1 range.

    Returns:
        One of GapCategory.MINOR, MODERATE, or MAJOR.
    """
    if gap_weight <= _GAP_MINOR_CEILING:
        return GapCategory.MINOR
    elif gap_weight <= _GAP_MODERATE_CEILING:
        return GapCategory.MODERATE
    else:
        return GapCategory.MAJOR


def _estimate_training_time(gap_category: str) -> str:
    """Return a human-readable training time estimate for a gap category.

    These are rough heuristics. In production we would use O*NET training
    time data or crowd-sourced estimates per skill.

    Args:
        gap_category: One of GapCategory constants.

    Returns:
        A string like "1-2 weeks" describing typical training time.
    """
    estimates = {
        GapCategory.MINOR: "1-4 weeks (self-study or short online course)",
        GapCategory.MODERATE: "1-3 months (structured course or bootcamp module)",
        GapCategory.MAJOR: "3-12 months (formal programme, degree module, or intensive bootcamp)",
    }
    return estimates.get(gap_category, "Unknown")


# ---------------------------------------------------------------------------
# Structured explanation dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkillToDevelop:
    """A single skill gap with actionable detail for the user.

    Attributes:
        element_id: O*NET element identifier.
        skill_name: Human-readable skill name.
        required_importance: How important this skill is to the target occupation
            (0-100 scale from O*NET).
        user_capability: User's current capability scalar (0.0-1.0).
        gap_weight: Normalised importance weight of this gap (0-1).
        gap_category: Severity tier: minor, moderate, or major.
        typical_training_time: Human-readable training time estimate.
        why_it_matters: Explanation of why this gap affects the bucket.
        skill_domain: Broad skill domain classification.
    """
    element_id: str
    skill_name: str
    required_importance: float
    user_capability: float
    gap_weight: float
    gap_category: str
    typical_training_time: str
    why_it_matters: str
    skill_domain: str


@dataclass
class BucketReasoning:
    """Transparent reasoning for why a score maps to a particular bucket.

    Shows the actual threshold values so the user can see exactly where their
    score sits relative to the decision boundary.

    Attributes:
        assigned_bucket: The bucket this occupation was assigned to.
        match_score: The user's match score for this occupation.
        gap_severity: The user's gap severity for this occupation.
        thresholds_used: The BucketThresholds that were applied.
        match_meets_ready_now: Whether match_score >= ready_now threshold.
        gap_meets_ready_now: Whether gap_severity <= ready_now threshold.
        match_in_trainable_range: Whether match_score is in trainable range.
        gap_in_trainable_range: Whether gap_severity is in trainable range.
        reasoning_text: Plain-English explanation of the assignment.
    """
    assigned_bucket: str
    match_score: float
    gap_severity: float
    thresholds_used: Dict[str, float]
    match_meets_ready_now: bool
    gap_meets_ready_now: bool
    match_in_trainable_range: bool
    gap_in_trainable_range: bool
    reasoning_text: str


@dataclass
class WhatWouldChangeBucket:
    """Hypothetical analysis showing what the user needs to reach each bucket.

    For each target bucket the user has NOT yet reached, we compute the
    gap closures required. This gives users concrete motivation and helps
    them prioritise skill development.

    Attributes:
        to_ready_now: Description of what would move the user to READY_NOW,
            or None if already there.
        to_trainable: Description of what would move the user to TRAINABLE,
            or None if already there or in a higher bucket.
        match_score_needed: The minimum match_score required for READY_NOW.
        gap_reduction_needed: How much gap_severity must decrease for READY_NOW.
        gaps_to_close: List of specific gaps that, if closed, would most
            efficiently move the user towards READY_NOW.
    """
    to_ready_now: Optional[str]
    to_trainable: Optional[str]
    match_score_needed: Optional[float]
    gap_reduction_needed: Optional[float]
    gaps_to_close: List[str]


@dataclass
class BucketExplanation:
    """Complete structured explanation for a scored occupation.

    This is the top-level object returned by the explainer. It contains
    everything a front-end needs to render a detailed, transparent explanation
    of why an occupation was scored and bucketed the way it was.

    Attributes:
        onet_code: The O*NET code of the explained occupation.
        summary: One-paragraph human-readable summary.
        skills_you_have: List of skill names where the user meets or exceeds
            the occupation requirement.
        skills_to_develop: Detailed gap information with training guidance.
        bucket_reasoning: Transparent threshold-based reasoning.
        what_would_change_bucket: Hypothetical bucket transitions.
        risk_tolerance_used: Which risk tolerance preset was applied.
        metadata: Additional context (total skills, job zone info, etc.).
    """
    onet_code: str
    summary: str
    skills_you_have: List[str]
    skills_to_develop: List[SkillToDevelop]
    bucket_reasoning: BucketReasoning
    what_would_change_bucket: WhatWouldChangeBucket
    risk_tolerance_used: str
    metadata: Dict[str, object]


# ---------------------------------------------------------------------------
# Explainer engine
# ---------------------------------------------------------------------------

class BucketExplainerEngine:
    """Generates structured explanations for OccupationScore results.

    The explainer is stateless: it takes a score and a threshold profile and
    produces a BucketExplanation. It can be instantiated once and reused
    across requests.

    Usage::

        engine = BucketExplainerEngine()
        explanation = engine.explain(score, profile=get_threshold_profile())
    """

    def explain(
        self,
        score: OccupationScore,
        occupation_skills: Optional[List[Dict]] = None,
        user_skill_ratings: Optional[Dict[str, int]] = None,
        profile: Optional[ThresholdProfile] = None,
    ) -> BucketExplanation:
        """Generate a full structured explanation for a scored occupation.

        Args:
            score: The OccupationScore from the baseline scorer.
            occupation_skills: Original occupation skill list (used to identify
                skills the user already has). If None, we derive what we can
                from score.top_gaps and score.metadata.
            user_skill_ratings: The user's skill ratings dict. If None, we can
                only explain gaps, not matched skills.
            profile: ThresholdProfile to use. Defaults to STANDARD.

        Returns:
            A fully populated BucketExplanation.
        """
        if profile is None:
            profile = get_threshold_profile(RiskTolerance.STANDARD)

        thresholds = profile.bucket_thresholds

        # --- Skills the user has ---
        skills_you_have = self._identify_matched_skills(
            score, occupation_skills, user_skill_ratings
        )

        # --- Skills to develop ---
        skills_to_develop = self._build_skills_to_develop(score.top_gaps, thresholds)

        # --- Bucket reasoning ---
        bucket_reasoning = self._build_bucket_reasoning(score, thresholds)

        # --- What would change bucket ---
        what_would_change = self._build_what_would_change(
            score, thresholds, score.top_gaps
        )

        # --- Summary ---
        summary = self._build_summary(
            score, len(skills_you_have), len(skills_to_develop)
        )

        return BucketExplanation(
            onet_code=score.onet_code,
            summary=summary,
            skills_you_have=skills_you_have,
            skills_to_develop=skills_to_develop,
            bucket_reasoning=bucket_reasoning,
            what_would_change_bucket=what_would_change,
            risk_tolerance_used=profile.risk_tolerance.value,
            metadata=score.metadata,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _identify_matched_skills(
        self,
        score: OccupationScore,
        occupation_skills: Optional[List[Dict]],
        user_skill_ratings: Optional[Dict[str, int]],
    ) -> List[str]:
        """Identify skills where the user meets or exceeds requirements.

        We consider a skill "matched" if the user's rating is >= 2 (Intermediate).
        This threshold is deliberately lower than the gap threshold (<=1) in the
        scorer to avoid a confusing gap between "you have it" and "you need it".

        Args:
            score: The occupation score.
            occupation_skills: List of skill dicts for the occupation.
            user_skill_ratings: User's skill ratings.

        Returns:
            List of skill names the user has.
        """
        if occupation_skills is None or user_skill_ratings is None:
            # Fallback: infer from metadata
            total = score.metadata.get("total_skills", 0)
            gap_count = len(score.top_gaps)
            matched_count = total - gap_count
            # We cannot name specific skills without the full skill list,
            # so return a placeholder.
            if matched_count > 0:
                return [f"{matched_count} skills matched (details unavailable)"]
            return []

        gap_element_ids = {gap.element_id for gap in score.top_gaps}
        matched = []
        for skill in occupation_skills:
            eid = skill.get("element_id", "")
            name = skill.get("skill_name", eid)
            rating = user_skill_ratings.get(eid, 0)
            # User has the skill if their rating > 1 (i.e. Intermediate+)
            # AND it is not in the gap list.
            if rating >= 2 and eid not in gap_element_ids:
                matched.append(name)
        return matched

    def _build_skills_to_develop(
        self,
        gaps: List[SkillGap],
        thresholds: BucketThresholds,
    ) -> List[SkillToDevelop]:
        """Build detailed gap descriptions for each skill gap.

        Args:
            gaps: List of SkillGap from the scorer.
            thresholds: Current bucket thresholds (for context in why_it_matters).

        Returns:
            List of SkillToDevelop dataclasses.
        """
        result = []
        for gap in gaps:
            category = _categorise_gap(gap.gap_weight)
            domain = classify_skill_domain(gap.element_id)

            why = self._explain_why_gap_matters(gap, category, thresholds)

            result.append(SkillToDevelop(
                element_id=gap.element_id,
                skill_name=gap.skill_name,
                required_importance=gap.required_importance,
                user_capability=gap.user_capability,
                gap_weight=round(gap.gap_weight, 4),
                gap_category=category,
                typical_training_time=_estimate_training_time(category),
                why_it_matters=why,
                skill_domain=domain.value,
            ))
        return result

    def _explain_why_gap_matters(
        self,
        gap: SkillGap,
        category: str,
        thresholds: BucketThresholds,
    ) -> str:
        """Generate a human-readable explanation of why a gap matters.

        Args:
            gap: The skill gap.
            category: The gap's severity category.
            thresholds: Current bucket thresholds.

        Returns:
            A string explaining the gap's impact on bucket assignment.
        """
        importance_pct = round(gap.gap_weight * 100, 1)

        if category == GapCategory.MAJOR:
            return (
                f"{gap.skill_name} accounts for {importance_pct}% of this "
                f"occupation's skill requirements. Closing this gap alone "
                f"could significantly improve your match score."
            )
        elif category == GapCategory.MODERATE:
            return (
                f"{gap.skill_name} contributes {importance_pct}% to the "
                f"skill profile. Developing this skill would meaningfully "
                f"reduce your gap severity (currently needs to be under "
                f"{thresholds.ready_now_gap_max}% for Ready Now)."
            )
        else:
            return (
                f"{gap.skill_name} is a smaller component ({importance_pct}%) "
                f"but still contributes to your overall gap severity. "
                f"Addressing it alongside larger gaps accelerates your path."
            )

    def _build_bucket_reasoning(
        self,
        score: OccupationScore,
        thresholds: BucketThresholds,
    ) -> BucketReasoning:
        """Build transparent reasoning for why a bucket was assigned.

        Args:
            score: The occupation score.
            thresholds: The thresholds used for assignment.

        Returns:
            BucketReasoning with all threshold comparisons visible.
        """
        match_meets_rn = score.match_score >= thresholds.ready_now_match_min
        gap_meets_rn = score.gap_severity <= thresholds.ready_now_gap_max
        match_in_trainable = (
            thresholds.trainable_match_min
            <= score.match_score
            <= thresholds.trainable_match_max
        )
        gap_in_trainable = (
            thresholds.trainable_gap_min
            <= score.gap_severity
            <= thresholds.trainable_gap_max
        )

        # Build reasoning text
        if score.bucket == "ready_now":
            reasoning = (
                f"Your match score ({score.match_score:.1f}%) meets the "
                f"Ready Now threshold (>= {thresholds.ready_now_match_min}%) "
                f"AND your gap severity ({score.gap_severity:.1f}%) is within "
                f"the ceiling (<= {thresholds.ready_now_gap_max}%)."
            )
        elif score.bucket == "trainable":
            parts = []
            if match_in_trainable:
                parts.append(
                    f"match score ({score.match_score:.1f}%) is in the "
                    f"Trainable range ({thresholds.trainable_match_min}-"
                    f"{thresholds.trainable_match_max}%)"
                )
            if gap_in_trainable:
                parts.append(
                    f"gap severity ({score.gap_severity:.1f}%) is in the "
                    f"Trainable range ({thresholds.trainable_gap_min}-"
                    f"{thresholds.trainable_gap_max}%)"
                )
            joined = " and ".join(parts) if parts else "threshold conditions met"
            reasoning = (
                f"You are in the Trainable bucket because your {joined}. "
                f"Ready Now requires match >= {thresholds.ready_now_match_min}% "
                f"AND gap <= {thresholds.ready_now_gap_max}%."
            )
        else:
            reasoning = (
                f"Your match score ({score.match_score:.1f}%) is below the "
                f"Trainable minimum ({thresholds.trainable_match_min}%) and "
                f"your gap severity ({score.gap_severity:.1f}%) is outside "
                f"the Trainable range ({thresholds.trainable_gap_min}-"
                f"{thresholds.trainable_gap_max}%). This places you in "
                f"Long Reskill."
            )

        thresholds_dict = {
            "ready_now_match_min": thresholds.ready_now_match_min,
            "ready_now_gap_max": thresholds.ready_now_gap_max,
            "trainable_match_min": thresholds.trainable_match_min,
            "trainable_match_max": thresholds.trainable_match_max,
            "trainable_gap_min": thresholds.trainable_gap_min,
            "trainable_gap_max": thresholds.trainable_gap_max,
        }

        return BucketReasoning(
            assigned_bucket=score.bucket,
            match_score=score.match_score,
            gap_severity=score.gap_severity,
            thresholds_used=thresholds_dict,
            match_meets_ready_now=match_meets_rn,
            gap_meets_ready_now=gap_meets_rn,
            match_in_trainable_range=match_in_trainable,
            gap_in_trainable_range=gap_in_trainable,
            reasoning_text=reasoning,
        )

    def _build_what_would_change(
        self,
        score: OccupationScore,
        thresholds: BucketThresholds,
        gaps: List[SkillGap],
    ) -> WhatWouldChangeBucket:
        """Compute hypothetical bucket changes if the user closed gaps.

        We simulate closing gaps in order of gap_weight (largest first) and
        report what it would take to reach READY_NOW or TRAINABLE.

        The simulation is approximate: closing a gap of weight W reduces
        gap_severity by W*100 and increases match_score by W*100. This is
        a simplification (the actual match_score formula depends on the
        capability scalar, not just 0/1), but it gives users directionally
        correct guidance.

        Args:
            score: The current occupation score.
            thresholds: The bucket thresholds.
            gaps: The list of skill gaps, sorted by weight descending.

        Returns:
            WhatWouldChangeBucket with actionable guidance.
        """
        if score.bucket == "ready_now":
            return WhatWouldChangeBucket(
                to_ready_now=None,
                to_trainable=None,
                match_score_needed=None,
                gap_reduction_needed=None,
                gaps_to_close=[],
            )

        # How far from READY_NOW?
        match_deficit = max(0, thresholds.ready_now_match_min - score.match_score)
        gap_excess = max(0, score.gap_severity - thresholds.ready_now_gap_max)

        # Simulate closing gaps in priority order
        simulated_match = score.match_score
        simulated_gap = score.gap_severity
        gaps_to_close_for_rn: List[str] = []

        for gap in gaps:
            if simulated_match >= thresholds.ready_now_match_min and \
               simulated_gap <= thresholds.ready_now_gap_max:
                break
            # Closing a gap: match goes up, gap goes down
            # The user goes from capability 0-0.25 to ~1.0, so match improves
            # by gap_weight * (1.0 - current_capability).
            improvement = gap.gap_weight * (1.0 - gap.user_capability) * 100
            simulated_match += improvement
            simulated_gap -= gap.gap_weight * 100
            gaps_to_close_for_rn.append(gap.skill_name)

        # Build to_ready_now message
        if gaps_to_close_for_rn:
            to_rn = (
                f"Close {len(gaps_to_close_for_rn)} gap(s) "
                f"({', '.join(gaps_to_close_for_rn)}) to potentially reach "
                f"Ready Now. You need match >= {thresholds.ready_now_match_min}% "
                f"(currently {score.match_score:.1f}%) and gap <= "
                f"{thresholds.ready_now_gap_max}% (currently "
                f"{score.gap_severity:.1f}%)."
            )
        else:
            to_rn = (
                f"Even closing all identified gaps may not reach Ready Now. "
                f"Consider the Relaxed risk tolerance for more opportunities."
            )

        # Build to_trainable message (only relevant if currently long_reskill)
        to_trainable = None
        if score.bucket == "long_reskill":
            # Find minimum gaps to close for TRAINABLE
            sim_match = score.match_score
            sim_gap = score.gap_severity
            gaps_for_trainable: List[str] = []

            for gap in gaps:
                in_match_range = (
                    thresholds.trainable_match_min
                    <= sim_match
                    <= thresholds.trainable_match_max
                )
                in_gap_range = (
                    thresholds.trainable_gap_min
                    <= sim_gap
                    <= thresholds.trainable_gap_max
                )
                if in_match_range or in_gap_range:
                    break
                improvement = gap.gap_weight * (1.0 - gap.user_capability) * 100
                sim_match += improvement
                sim_gap -= gap.gap_weight * 100
                gaps_for_trainable.append(gap.skill_name)

            if gaps_for_trainable:
                to_trainable = (
                    f"Close {len(gaps_for_trainable)} gap(s) "
                    f"({', '.join(gaps_for_trainable)}) to move from "
                    f"Long Reskill to Trainable."
                )
            else:
                to_trainable = (
                    "You are very close to the Trainable range. Small "
                    "improvements in any skill could shift your classification."
                )

        return WhatWouldChangeBucket(
            to_ready_now=to_rn,
            to_trainable=to_trainable,
            match_score_needed=thresholds.ready_now_match_min,
            gap_reduction_needed=round(gap_excess, 2) if gap_excess > 0 else None,
            gaps_to_close=gaps_to_close_for_rn,
        )

    def _build_summary(
        self,
        score: OccupationScore,
        matched_count: int,
        gap_count: int,
    ) -> str:
        """Build a one-paragraph summary of the explanation.

        Args:
            score: The occupation score.
            matched_count: Number of skills the user has.
            gap_count: Number of skill gaps.

        Returns:
            A human-readable summary paragraph.
        """
        bucket_labels = {
            "ready_now": "Ready Now",
            "trainable": "Trainable",
            "long_reskill": "Long Reskill",
        }
        label = bucket_labels.get(score.bucket, score.bucket)

        if score.bucket == "ready_now":
            return (
                f"This occupation is classified as {label}. You have "
                f"{matched_count} matching skill(s) and {gap_count} minor "
                f"gap(s). Your match score of {score.match_score:.1f}% and "
                f"gap severity of {score.gap_severity:.1f}% place you well "
                f"within the Ready Now thresholds. You can start applying "
                f"immediately."
            )
        elif score.bucket == "trainable":
            return (
                f"This occupation is classified as {label}. You have "
                f"{matched_count} matching skill(s) but {gap_count} gap(s) "
                f"that need development. Your match score is "
                f"{score.match_score:.1f}% with a gap severity of "
                f"{score.gap_severity:.1f}%. With focused training on the "
                f"identified gaps, this role is within reach."
            )
        else:
            return (
                f"This occupation is classified as {label}. You have "
                f"{matched_count} matching skill(s) but face {gap_count} "
                f"significant gap(s). Your match score of "
                f"{score.match_score:.1f}% and gap severity of "
                f"{score.gap_severity:.1f}% indicate that substantial "
                f"reskilling is needed. Consider this a longer-term goal "
                f"with a structured training plan."
            )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def explain_score(
    score: OccupationScore,
    occupation_skills: Optional[List[Dict]] = None,
    user_skill_ratings: Optional[Dict[str, int]] = None,
    risk_tolerance: RiskTolerance = RiskTolerance.STANDARD,
) -> BucketExplanation:
    """Convenience function to explain a single OccupationScore.

    Args:
        score: The OccupationScore to explain.
        occupation_skills: Optional full skill list for the occupation.
        user_skill_ratings: Optional user ratings dict.
        risk_tolerance: Risk tolerance preset to use for thresholds.

    Returns:
        A BucketExplanation with full structured detail.
    """
    engine = BucketExplainerEngine()
    profile = get_threshold_profile(risk_tolerance)
    return engine.explain(
        score,
        occupation_skills=occupation_skills,
        user_skill_ratings=user_skill_ratings,
        profile=profile,
    )
