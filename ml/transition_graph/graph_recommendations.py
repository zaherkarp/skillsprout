"""Graph-enhanced scoring for recommendations.

Provides additive signals from the transition graph to supplement
skill-based scoring. Graph signals NEVER override the baseline scorer;
they adjust the calibration score up or down within bounded ranges.

Design principles:
1. Graph signals are additive: they boost or penalize the calibrated score,
   they do not replace the match_score or gap_severity.
2. Graph signals are bounded: the maximum adjustment is capped to prevent
   the graph from dominating the skill-based signal.
3. Unknown transitions get zero adjustment: if the graph has no data for
   a (origin, target) pair, the score is unchanged.
4. Novel paths are flagged, not penalized: an unexplored transition is
   annotated for the user's information but not scored lower.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx

from ml.transition_graph.graph_builder import (
    TransitionGraphBuilder,
    MIN_OBSERVATIONS_FOR_STATS,
)

logger = logging.getLogger(__name__)

# Maximum score adjustment from graph signals (in points on 0-100 scale).
# A well-traveled path with high success can boost by up to +10.
# A poorly-traveled path with low success can penalize by up to -5.
# The asymmetry is intentional: we are more conservative about penalizing
# than boosting, because the graph has limited data.
MAX_BOOST = 10.0
MAX_PENALTY = 5.0

# Minimum interactions for an edge to contribute any boost/penalty.
# Below this, the graph signal is zero (not enough data to trust).
MIN_INTERACTIONS_FOR_SIGNAL = 3


@dataclass
class GraphSignal:
    """A graph-derived signal for a single (origin, target) pair.

    Attributes:
        origin_onet_code: User's current occupation.
        target_onet_code: Recommended occupation.
        score_adjustment: Points to add to the calibrated score (-5 to +10).
        confidence: How confident the signal is ("high", "low", "none").
        is_well_traveled: Whether this transition has been frequently observed.
        is_novel: Whether this transition has never been observed.
        path_count: Number of multi-hop paths available (0 if none).
        explanation: Human-readable explanation of the graph signal.
        raw_factors: Dict of underlying graph metrics.
    """

    origin_onet_code: str
    target_onet_code: str
    score_adjustment: float
    confidence: str
    is_well_traveled: bool
    is_novel: bool
    path_count: int
    explanation: str
    raw_factors: Dict[str, Any]


def compute_graph_signal(
    graph: nx.DiGraph,
    origin_onet_code: str,
    target_onet_code: str,
    population_success_rate: Optional[float] = None,
) -> GraphSignal:
    """Compute the graph-derived score adjustment for a (origin, target) pair.

    The adjustment is calculated from three components:
    1. Success rate signal: transitions with high success get a boost.
    2. Volume signal: well-traveled paths get a small additional boost.
    3. Novelty signal: never-before-seen transitions get flagged (but no penalty).

    Args:
        graph: The transition DiGraph.
        origin_onet_code: User's current occupation code.
        target_onet_code: Target occupation code.
        population_success_rate: Overall success rate across all transitions.
            Used as a baseline for comparison. If None, uses 0.5.

    Returns:
        GraphSignal with the computed adjustment and metadata.
    """
    pop_sr = population_success_rate if population_success_rate is not None else 0.5
    factors: Dict[str, Any] = {}
    adjustment = 0.0

    # Check if both nodes exist in the graph
    origin_in_graph = origin_onet_code in graph
    target_in_graph = target_onet_code in graph

    if not origin_in_graph or not target_in_graph:
        return GraphSignal(
            origin_onet_code=origin_onet_code,
            target_onet_code=target_onet_code,
            score_adjustment=0.0,
            confidence="none",
            is_well_traveled=False,
            is_novel=True,
            path_count=0,
            explanation=(
                "This career transition has not been observed in our data. "
                "Your score is based entirely on skill matching."
            ),
            raw_factors=factors,
        )

    # Check for direct edge
    has_direct_edge = graph.has_edge(origin_onet_code, target_onet_code)

    if has_direct_edge:
        edge_data = graph.edges[origin_onet_code, target_onet_code]
        total_interactions = edge_data.get("total_interactions", 0)
        success_rate = edge_data.get("success_rate")
        has_sufficient_data = edge_data.get("has_sufficient_data", False)

        factors["total_interactions"] = total_interactions
        factors["success_rate"] = success_rate
        factors["has_sufficient_data"] = has_sufficient_data

        if total_interactions >= MIN_INTERACTIONS_FOR_SIGNAL:
            # ---- Component 1: Success rate signal ----
            if success_rate is not None and has_sufficient_data:
                # How much better (or worse) is this transition vs. population average?
                sr_delta = success_rate - pop_sr

                # Scale the delta to our adjustment range
                # A transition with 80% success vs 50% population = +0.3 delta
                # Scaled to: +0.3 * MAX_BOOST = +3.0 points
                if sr_delta >= 0:
                    sr_adjustment = sr_delta * MAX_BOOST
                else:
                    sr_adjustment = sr_delta * MAX_PENALTY

                adjustment += sr_adjustment
                factors["success_rate_adjustment"] = round(sr_adjustment, 2)

            # ---- Component 2: Volume signal ----
            # Well-traveled paths (many interactions) get a small boost.
            # This is logarithmic to prevent high-volume paths from dominating.
            import math

            volume_boost = min(
                math.log1p(total_interactions) * 0.5, MAX_BOOST * 0.3
            )
            adjustment += volume_boost
            factors["volume_boost"] = round(volume_boost, 2)

        # Determine if well-traveled
        is_well_traveled = total_interactions >= 10
        is_novel = False
        confidence = "high" if has_sufficient_data else "low"

    else:
        # No direct edge: check for indirect paths
        is_well_traveled = False
        is_novel = True
        confidence = "none"
        factors["direct_edge"] = False

    # ---- Count multi-hop paths ----
    path_count = 0
    try:
        paths = list(
            nx.all_simple_paths(
                graph, origin_onet_code, target_onet_code, cutoff=3
            )
        )
        path_count = len(paths)
        factors["alternative_paths"] = path_count

        # If no direct edge but indirect paths exist, give a tiny boost
        # to signal that this transition is reachable.
        if not has_direct_edge and path_count > 0:
            indirect_boost = min(path_count * 0.5, 2.0)
            adjustment += indirect_boost
            factors["indirect_path_boost"] = round(indirect_boost, 2)
            is_novel = False  # Not truly novel if indirect paths exist
            confidence = "low"

    except nx.NetworkXError:
        pass

    # ---- Clamp adjustment to bounds ----
    adjustment = max(-MAX_PENALTY, min(MAX_BOOST, adjustment))
    adjustment = round(adjustment, 2)

    # ---- Generate explanation ----
    explanation = _generate_explanation(
        is_well_traveled=is_well_traveled,
        is_novel=is_novel,
        adjustment=adjustment,
        factors=factors,
        origin_in_graph=origin_in_graph,
        path_count=path_count,
    )

    return GraphSignal(
        origin_onet_code=origin_onet_code,
        target_onet_code=target_onet_code,
        score_adjustment=adjustment,
        confidence=confidence,
        is_well_traveled=is_well_traveled,
        is_novel=is_novel,
        path_count=path_count,
        explanation=explanation,
        raw_factors=factors,
    )


def enhance_scores(
    graph: nx.DiGraph,
    origin_onet_code: str,
    scored_occupations: List[Dict[str, Any]],
    population_success_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Apply graph signals to a list of scored occupations.

    This function takes scored occupations from the baseline scorer and adds
    graph-derived adjustments. The graph signals are ADDITIVE: they modify
    a `graph_adjusted_score` field but never change the original match_score,
    gap_severity, or bucket assignment.

    Args:
        graph: The transition DiGraph.
        origin_onet_code: User's current occupation code.
        scored_occupations: List of dicts, each with at least:
            - target_onet_code: Target occupation code
            - match_score: Baseline match score (0-100)
            - calibrated_score: Calibrated score if available, else match_score
        population_success_rate: Overall success rate for normalization.

    Returns:
        The same list of dicts, each augmented with:
            - graph_adjusted_score: calibrated_score + graph adjustment
            - graph_signal: The full GraphSignal object as a dict
    """
    if graph.number_of_nodes() == 0:
        # No graph data: return scores unchanged
        for occ in scored_occupations:
            occ["graph_adjusted_score"] = occ.get(
                "calibrated_score", occ.get("match_score", 0.0)
            )
            occ["graph_signal"] = None
        return scored_occupations

    for occ in scored_occupations:
        target_code = occ["target_onet_code"]
        base_score = occ.get(
            "calibrated_score", occ.get("match_score", 0.0)
        )

        signal = compute_graph_signal(
            graph=graph,
            origin_onet_code=origin_onet_code,
            target_onet_code=target_code,
            population_success_rate=population_success_rate,
        )

        # Apply adjustment (clamped to 0-100)
        adjusted = max(0.0, min(100.0, base_score + signal.score_adjustment))

        occ["graph_adjusted_score"] = round(adjusted, 2)
        occ["graph_signal"] = {
            "score_adjustment": signal.score_adjustment,
            "confidence": signal.confidence,
            "is_well_traveled": signal.is_well_traveled,
            "is_novel": signal.is_novel,
            "path_count": signal.path_count,
            "explanation": signal.explanation,
        }

    return scored_occupations


def flag_unexplored_paths(
    graph: nx.DiGraph,
    origin_onet_code: str,
    scored_occupations: List[Dict[str, Any]],
    max_flags: int = 3,
) -> List[Dict[str, Any]]:
    """Identify interesting occupations with no observed transitions.

    These are occupations that score well on skill matching but have never
    been explored by any user starting from the given origin. They represent
    potential discoveries that the system has not yet learned about.

    This function does NOT penalize these occupations. It adds an annotation
    so the UI can present them as "unexplored possibilities."

    Args:
        graph: The transition DiGraph.
        origin_onet_code: User's current occupation code.
        scored_occupations: List of scored occupation dicts.
        max_flags: Maximum number of occupations to flag.

    Returns:
        List of dicts for flagged occupations, each containing:
            - target_onet_code: The occupation code
            - match_score: Its skill-based match score
            - reason: Why it was flagged
    """
    flagged = []

    for occ in scored_occupations:
        target_code = occ["target_onet_code"]
        match_score = occ.get("match_score", 0.0)

        # Only flag occupations with reasonable match scores
        if match_score < 50:
            continue

        # Check if this transition has been observed
        has_edge = graph.has_edge(origin_onet_code, target_code)

        if not has_edge:
            flagged.append(
                {
                    "target_onet_code": target_code,
                    "target_title": occ.get("title", "Unknown"),
                    "match_score": match_score,
                    "reason": (
                        "This role matches your skills well but no one in our "
                        "data has explored this transition yet. You could be "
                        "a pioneer."
                    ),
                }
            )

    # Sort by match score descending and take top N
    flagged.sort(key=lambda x: x["match_score"], reverse=True)
    return flagged[:max_flags]


def _generate_explanation(
    is_well_traveled: bool,
    is_novel: bool,
    adjustment: float,
    factors: Dict[str, Any],
    origin_in_graph: bool,
    path_count: int,
) -> str:
    """Generate a human-readable explanation of the graph signal.

    Args:
        is_well_traveled: Whether this is a well-traveled transition.
        is_novel: Whether this transition is novel (never observed).
        adjustment: The score adjustment applied.
        factors: Raw factors from the graph analysis.
        origin_in_graph: Whether the origin node exists in the graph.
        path_count: Number of multi-hop paths.

    Returns:
        Explanation string.
    """
    if not origin_in_graph:
        return (
            "We do not yet have transition data for your current occupation. "
            "Your score is based entirely on skill matching."
        )

    if is_novel and path_count == 0:
        return (
            "This career transition has not been observed in our data. "
            "Your score is based entirely on skill matching."
        )

    if is_novel and path_count > 0:
        return (
            f"No one has made this direct transition, but {path_count} "
            f"indirect path(s) through intermediate roles exist. "
            f"Score adjusted by {adjustment:+.1f} points."
        )

    parts = []

    total = factors.get("total_interactions", 0)
    sr = factors.get("success_rate")

    if is_well_traveled:
        parts.append(
            f"This is a well-traveled career path ({total} users have explored it)."
        )
    else:
        parts.append(
            f"Some users have explored this transition ({total} interactions)."
        )

    if sr is not None:
        pct = sr * 100
        if pct >= 60:
            parts.append(f"Success rate is strong ({pct:.0f}%).")
        elif pct >= 30:
            parts.append(f"Success rate is moderate ({pct:.0f}%).")
        else:
            parts.append(f"Success rate is low ({pct:.0f}%).")

    if adjustment != 0:
        direction = "boosted" if adjustment > 0 else "reduced"
        parts.append(f"Score {direction} by {abs(adjustment):.1f} points.")

    return " ".join(parts)
