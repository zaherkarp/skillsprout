"""Side-by-side comparison view for up to 3 scored occupations.

Design rationale:
-----------------
Users rarely evaluate occupations in isolation. The comparison view lets them
place up to three occupation recommendations next to each other and see:

1. **Skill overlap**: which skills are shared across all candidates, reducing
   the cognitive load of "which skills do I already have that transfer?"
2. **Unique gaps**: which gaps are specific to each occupation, highlighting
   the incremental cost of each path.
3. **Closest to READY_NOW**: a quick ranking of which occupation is nearest
   to the highest readiness bucket, so users can prioritise.

Limit of 3:
  UX research consistently shows that comparing more than 3 options at once
  leads to decision paralysis. We enforce this limit at the data layer so
  that the UI can always render a clean three-column table.

This module operates purely on OccupationScore objects (and optional skill
lists) and does NOT touch the database. The API layer is responsible for
fetching scores and passing them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.ml.scoring import OccupationScore, SkillGap
from app.features.explainability.bucket_explainer import (
    BucketExplainerEngine,
    BucketExplanation,
)
from app.features.explainability.threshold_config import (
    RiskTolerance,
    ThresholdProfile,
    get_threshold_profile,
)


# Maximum number of occupations that can be compared simultaneously.
MAX_COMPARISON_SIZE = 3

# Bucket priority for "closest to READY_NOW" ranking.
# Lower number = closer to ready.
_BUCKET_PRIORITY: Dict[str, int] = {
    "ready_now": 0,
    "trainable": 1,
    "long_reskill": 2,
}


class ComparisonError(Exception):
    """Raised when comparison inputs are invalid."""
    pass


# ---------------------------------------------------------------------------
# Comparison result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkillOverlapAnalysis:
    """Analysis of skill overlap and uniqueness across compared occupations.

    Attributes:
        shared_skills: Skills that appear as requirements in ALL compared
            occupations. These are the user's most transferable assets.
        shared_gaps: Skill gaps that the user has across ALL compared
            occupations. Closing these has the highest leverage.
        unique_gaps: Per-occupation dict mapping onet_code to gaps that
            are unique to that occupation (not shared with others).
        shared_skill_names: Human-readable names for shared skills.
        shared_gap_names: Human-readable names for shared gaps.
    """
    shared_skills: List[str]
    shared_gaps: List[str]
    unique_gaps: Dict[str, List[str]]
    shared_skill_names: List[str]
    shared_gap_names: List[str]


@dataclass
class ReadinessRanking:
    """Ranking of compared occupations by proximity to READY_NOW.

    Attributes:
        ranked_codes: List of onet_codes sorted from closest to READY_NOW
            to furthest.
        rankings: Per-occupation details including bucket, match_score, and
            the computed distance metric.
        closest_onet_code: The single occupation closest to READY_NOW.
    """
    ranked_codes: List[str]
    rankings: List[Dict[str, object]]
    closest_onet_code: str


@dataclass
class ComparisonResult:
    """Complete comparison result for up to 3 occupations.

    This is the top-level object returned by ``compare_occupations``.

    Attributes:
        occupation_codes: The onet_codes being compared.
        explanations: Full BucketExplanation for each occupation, keyed by code.
        skill_overlap: Analysis of shared skills and unique gaps.
        readiness_ranking: Which occupation is closest to READY_NOW.
        comparison_summary: Human-readable summary of the comparison.
    """
    occupation_codes: List[str]
    explanations: Dict[str, BucketExplanation]
    skill_overlap: SkillOverlapAnalysis
    readiness_ranking: ReadinessRanking
    comparison_summary: str


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

class ComparisonEngine:
    """Engine for building side-by-side occupation comparisons.

    Stateless: instantiate once and call ``compare`` repeatedly.
    """

    def __init__(self) -> None:
        self._explainer = BucketExplainerEngine()

    def compare(
        self,
        scores: List[OccupationScore],
        occupation_skills_map: Optional[Dict[str, List[Dict]]] = None,
        user_skill_ratings: Optional[Dict[str, int]] = None,
        risk_tolerance: RiskTolerance = RiskTolerance.STANDARD,
    ) -> ComparisonResult:
        """Compare up to 3 scored occupations side by side.

        Args:
            scores: List of OccupationScore objects (1-3 items).
            occupation_skills_map: Optional dict mapping onet_code to the
                full list of skill dicts for that occupation. Used for richer
                matched-skill identification.
            user_skill_ratings: Optional user skill ratings dict.
            risk_tolerance: Risk tolerance preset for threshold context.

        Returns:
            ComparisonResult with overlap analysis, ranking, and explanations.

        Raises:
            ComparisonError: If fewer than 1 or more than 3 scores are provided.
        """
        if not scores:
            raise ComparisonError("At least one occupation score is required.")
        if len(scores) > MAX_COMPARISON_SIZE:
            raise ComparisonError(
                f"Maximum {MAX_COMPARISON_SIZE} occupations can be compared "
                f"at once. Received {len(scores)}."
            )

        profile = get_threshold_profile(risk_tolerance)

        # Generate explanations for each occupation
        explanations: Dict[str, BucketExplanation] = {}
        for score in scores:
            occ_skills = (
                occupation_skills_map.get(score.onet_code)
                if occupation_skills_map
                else None
            )
            explanation = self._explainer.explain(
                score,
                occupation_skills=occ_skills,
                user_skill_ratings=user_skill_ratings,
                profile=profile,
            )
            explanations[score.onet_code] = explanation

        # Compute skill overlap
        skill_overlap = self._compute_skill_overlap(scores, explanations)

        # Compute readiness ranking
        readiness_ranking = self._compute_readiness_ranking(scores, profile)

        # Build summary
        summary = self._build_comparison_summary(
            scores, skill_overlap, readiness_ranking
        )

        return ComparisonResult(
            occupation_codes=[s.onet_code for s in scores],
            explanations=explanations,
            skill_overlap=skill_overlap,
            readiness_ranking=readiness_ranking,
            comparison_summary=summary,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _compute_skill_overlap(
        self,
        scores: List[OccupationScore],
        explanations: Dict[str, BucketExplanation],
    ) -> SkillOverlapAnalysis:
        """Compute which skills/gaps are shared across all occupations.

        We use element_ids for set operations (they are unique identifiers)
        but report human-readable skill names in the output.

        Args:
            scores: List of occupation scores.
            explanations: Dict of BucketExplanation keyed by onet_code.

        Returns:
            SkillOverlapAnalysis with shared and unique gap info.
        """
        # Collect gap element_ids per occupation
        gap_sets: Dict[str, Set[str]] = {}
        gap_names: Dict[str, str] = {}  # element_id -> skill_name

        for score in scores:
            gap_ids = set()
            for gap in score.top_gaps:
                gap_ids.add(gap.element_id)
                gap_names[gap.element_id] = gap.skill_name
            gap_sets[score.onet_code] = gap_ids

        # Shared gaps: intersection of all gap sets
        all_gap_sets = list(gap_sets.values())
        if all_gap_sets:
            shared_gap_ids = all_gap_sets[0].copy()
            for gs in all_gap_sets[1:]:
                shared_gap_ids &= gs
        else:
            shared_gap_ids = set()

        # Unique gaps: gaps in one occupation that are NOT in any other
        unique_gaps: Dict[str, List[str]] = {}
        for code, gap_ids in gap_sets.items():
            other_gaps = set()
            for other_code, other_gap_ids in gap_sets.items():
                if other_code != code:
                    other_gaps |= other_gap_ids
            unique = gap_ids - other_gaps
            unique_gaps[code] = [gap_names.get(eid, eid) for eid in unique]

        # Shared skills: skills_you_have that appear in ALL explanations
        shared_skill_sets: List[Set[str]] = []
        for code, expl in explanations.items():
            shared_skill_sets.append(set(expl.skills_you_have))

        if shared_skill_sets:
            shared_skills = shared_skill_sets[0].copy()
            for ss in shared_skill_sets[1:]:
                shared_skills &= ss
        else:
            shared_skills = set()

        return SkillOverlapAnalysis(
            shared_skills=sorted(shared_skills),
            shared_gaps=sorted(gap_names.get(eid, eid) for eid in shared_gap_ids),
            unique_gaps=unique_gaps,
            shared_skill_names=sorted(shared_skills),
            shared_gap_names=sorted(
                gap_names.get(eid, eid) for eid in shared_gap_ids
            ),
        )

    def _compute_readiness_ranking(
        self,
        scores: List[OccupationScore],
        profile: ThresholdProfile,
    ) -> ReadinessRanking:
        """Rank occupations by proximity to READY_NOW.

        The distance metric is:
          distance = max(
              max(0, ready_now_match_min - match_score),
              max(0, gap_severity - ready_now_gap_max)
          )
        Lower distance = closer to READY_NOW. Ties are broken by match_score.

        If the occupation is already READY_NOW, distance is 0 (or negative,
        clamped to 0).

        Args:
            scores: List of occupation scores.
            profile: The threshold profile for READY_NOW boundaries.

        Returns:
            ReadinessRanking with sorted codes and per-occupation details.
        """
        thresholds = profile.bucket_thresholds
        rankings = []

        for score in scores:
            # Compute distance to READY_NOW boundary
            match_deficit = max(
                0, thresholds.ready_now_match_min - score.match_score
            )
            gap_excess = max(
                0, score.gap_severity - thresholds.ready_now_gap_max
            )
            # Combined distance: the larger of the two deficits determines
            # how far the user is from READY_NOW.
            distance = max(match_deficit, gap_excess)

            rankings.append({
                "onet_code": score.onet_code,
                "bucket": score.bucket,
                "match_score": score.match_score,
                "gap_severity": score.gap_severity,
                "distance_to_ready_now": round(distance, 2),
                "bucket_priority": _BUCKET_PRIORITY.get(score.bucket, 99),
            })

        # Sort: first by bucket_priority (ready_now < trainable < long_reskill),
        # then by distance (ascending), then by match_score (descending).
        rankings.sort(
            key=lambda r: (
                r["bucket_priority"],
                r["distance_to_ready_now"],
                -r["match_score"],
            )
        )

        ranked_codes = [r["onet_code"] for r in rankings]
        closest = ranked_codes[0] if ranked_codes else ""

        return ReadinessRanking(
            ranked_codes=ranked_codes,
            rankings=rankings,
            closest_onet_code=closest,
        )

    def _build_comparison_summary(
        self,
        scores: List[OccupationScore],
        overlap: SkillOverlapAnalysis,
        ranking: ReadinessRanking,
    ) -> str:
        """Build a human-readable comparison summary.

        Args:
            scores: List of occupation scores.
            overlap: Skill overlap analysis.
            ranking: Readiness ranking.

        Returns:
            A paragraph summarising the comparison.
        """
        n = len(scores)
        codes = [s.onet_code for s in scores]

        if n == 1:
            return (
                f"Single occupation view for {codes[0]}. "
                f"No comparison data available."
            )

        shared_count = len(overlap.shared_gaps)
        closest = ranking.closest_onet_code
        closest_distance = 0.0
        for r in ranking.rankings:
            if r["onet_code"] == closest:
                closest_distance = r["distance_to_ready_now"]
                break

        parts = [
            f"Comparing {n} occupations: {', '.join(codes)}.",
        ]

        if shared_count > 0:
            parts.append(
                f" {shared_count} skill gap(s) are shared across all "
                f"options ({', '.join(overlap.shared_gap_names[:3])}), "
                f"meaning closing these gaps benefits every path."
            )

        if closest_distance == 0:
            parts.append(
                f" {closest} is already in the Ready Now bucket."
            )
        else:
            parts.append(
                f" {closest} is closest to Ready Now "
                f"(distance: {closest_distance:.1f} points)."
            )

        # Highlight unique gaps per occupation
        for code in codes:
            unique = overlap.unique_gaps.get(code, [])
            if unique:
                parts.append(
                    f" {code} has {len(unique)} unique gap(s) "
                    f"not shared with the others."
                )

        return "".join(parts)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def compare_occupations(
    scores: List[OccupationScore],
    occupation_skills_map: Optional[Dict[str, List[Dict]]] = None,
    user_skill_ratings: Optional[Dict[str, int]] = None,
    risk_tolerance: RiskTolerance = RiskTolerance.STANDARD,
) -> ComparisonResult:
    """Convenience function to compare up to 3 occupations.

    Args:
        scores: List of OccupationScore objects (1-3).
        occupation_skills_map: Optional per-occupation skill dicts.
        user_skill_ratings: Optional user ratings.
        risk_tolerance: Risk tolerance preset.

    Returns:
        ComparisonResult with full analysis.

    Raises:
        ComparisonError: If input validation fails.
    """
    engine = ComparisonEngine()
    return engine.compare(
        scores,
        occupation_skills_map=occupation_skills_map,
        user_skill_ratings=user_skill_ratings,
        risk_tolerance=risk_tolerance,
    )
