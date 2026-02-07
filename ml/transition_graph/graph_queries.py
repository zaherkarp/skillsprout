"""Query functions for the transition graph.

Provides analytical queries over the directed graph of career transitions:
- Most common transitions from a given occupation
- Successful multi-hop paths between occupations
- Emerging transitions (recently popular)
- Transition difficulty assessment
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from ml.transition_graph.graph_builder import (
    TransitionGraphBuilder,
    MIN_OBSERVATIONS_FOR_STATS,
)

logger = logging.getLogger(__name__)


def most_common_transitions(
    graph: nx.DiGraph,
    origin_onet_code: str,
    top_n: int = 10,
    min_interactions: int = 1,
) -> List[Dict[str, Any]]:
    """Get the most common transitions from a given occupation.

    Transitions are ranked by total interaction count. This answers:
    "What do people in this role most commonly explore as their next job?"

    Args:
        graph: The transition DiGraph.
        origin_onet_code: O*NET code of the origin occupation.
        top_n: Maximum number of transitions to return.
        min_interactions: Minimum interaction count to include a transition.

    Returns:
        List of dicts sorted by total_interactions descending, each containing:
            - target_onet_code: Target occupation code
            - target_title: Target occupation title
            - total_interactions: Number of user interactions
            - success_rate: Fraction of positive outcomes (or None)
            - median_match_score: Median baseline match score
    """
    if origin_onet_code not in graph:
        logger.debug(f"Origin {origin_onet_code} not found in graph")
        return []

    transitions = []
    for target in graph.successors(origin_onet_code):
        edge_data = graph.edges[origin_onet_code, target]

        if edge_data.get("total_interactions", 0) < min_interactions:
            continue

        transitions.append(
            {
                "target_onet_code": target,
                "target_title": graph.nodes[target].get("title", "Unknown"),
                "total_interactions": edge_data.get("total_interactions", 0),
                "success_rate": edge_data.get("success_rate"),
                "median_match_score": edge_data.get("median_match_score"),
                "median_gap_severity": edge_data.get("median_gap_severity"),
                "has_sufficient_data": edge_data.get(
                    "has_sufficient_data", False
                ),
            }
        )

    # Sort by total interactions descending
    transitions.sort(key=lambda x: x["total_interactions"], reverse=True)

    return transitions[:top_n]


def successful_paths(
    graph: nx.DiGraph,
    origin_onet_code: str,
    target_onet_code: str,
    max_hops: int = 3,
    min_success_rate: float = 0.3,
) -> List[List[Dict[str, Any]]]:
    """Find successful multi-hop paths between two occupations.

    Sometimes a direct transition is rare or difficult, but a two-step
    path through an intermediate occupation is well-traveled. For example:
    Teacher -> Corporate Trainer -> HR Manager might be easier than
    Teacher -> HR Manager directly.

    Only paths where every edge has a success rate above min_success_rate
    are returned (when sufficient data exists to compute success rate).

    Args:
        graph: The transition DiGraph.
        origin_onet_code: Starting occupation code.
        target_onet_code: Destination occupation code.
        max_hops: Maximum number of edges in a path (default 3).
        min_success_rate: Minimum success rate per edge to include
            the path. Edges without sufficient data are allowed through.

    Returns:
        List of paths, each path being a list of step dicts:
            - from_code: Origin occupation for this step
            - to_code: Target occupation for this step
            - to_title: Title of target occupation
            - success_rate: Edge success rate (or None)
            - total_interactions: Number of interactions on this edge
    """
    if origin_onet_code not in graph or target_onet_code not in graph:
        return []

    paths = []

    try:
        # Find all simple paths up to max_hops edges
        all_simple_paths = nx.all_simple_paths(
            graph, origin_onet_code, target_onet_code, cutoff=max_hops
        )

        for node_path in all_simple_paths:
            path_steps = []
            path_viable = True

            for i in range(len(node_path) - 1):
                from_node = node_path[i]
                to_node = node_path[i + 1]

                edge_data = graph.edges[from_node, to_node]
                success_rate = edge_data.get("success_rate")

                # Filter by success rate when data exists
                if (
                    success_rate is not None
                    and success_rate < min_success_rate
                ):
                    path_viable = False
                    break

                path_steps.append(
                    {
                        "from_code": from_node,
                        "from_title": graph.nodes[from_node].get(
                            "title", "Unknown"
                        ),
                        "to_code": to_node,
                        "to_title": graph.nodes[to_node].get(
                            "title", "Unknown"
                        ),
                        "success_rate": success_rate,
                        "total_interactions": edge_data.get(
                            "total_interactions", 0
                        ),
                        "median_match_score": edge_data.get(
                            "median_match_score"
                        ),
                    }
                )

            if path_viable and path_steps:
                paths.append(path_steps)

    except nx.NetworkXError as e:
        logger.warning(
            f"Error finding paths from {origin_onet_code} "
            f"to {target_onet_code}: {e}"
        )

    # Sort paths by: fewest hops first, then by minimum success rate descending
    def path_sort_key(path_steps):
        min_sr = min(
            (s["success_rate"] for s in path_steps if s["success_rate"] is not None),
            default=0.0,
        )
        return (len(path_steps), -min_sr)

    paths.sort(key=path_sort_key)

    return paths


def emerging_transitions(
    graph: nx.DiGraph,
    lookback_days: int = 90,
    min_recent_interactions: int = 3,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Identify transitions that are gaining popularity recently.

    An emerging transition is one where a significant fraction of its
    total interactions occurred within the lookback period. This helps
    surface new career paths that users are starting to explore.

    Args:
        graph: The transition DiGraph.
        lookback_days: Number of days to consider as "recent."
        min_recent_interactions: Minimum interactions in the recent period.
        top_n: Maximum number of transitions to return.

    Returns:
        List of dicts sorted by recency_ratio descending:
            - origin_onet_code: Origin occupation code
            - origin_title: Origin occupation title
            - target_onet_code: Target occupation code
            - target_title: Target occupation title
            - total_interactions: Total interaction count
            - last_seen: Most recent interaction timestamp
            - recency_ratio: Approximate recency signal
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    emerging = []

    for origin, target, attrs in graph.edges(data=True):
        total = attrs.get("total_interactions", 0)
        last_seen_str = attrs.get("last_seen")

        if not last_seen_str or total < min_recent_interactions:
            continue

        # Parse last_seen timestamp
        try:
            if isinstance(last_seen_str, str):
                last_seen = datetime.fromisoformat(last_seen_str)
            elif isinstance(last_seen_str, datetime):
                last_seen = last_seen_str
            else:
                continue
        except (ValueError, TypeError):
            continue

        # Only consider transitions with recent activity
        if last_seen < cutoff:
            continue

        # Compute a recency ratio: newer edges with fewer total interactions
        # but recent activity are "more emerging" than old high-volume edges.
        first_seen_str = attrs.get("first_seen")
        try:
            if isinstance(first_seen_str, str):
                first_seen = datetime.fromisoformat(first_seen_str)
            elif isinstance(first_seen_str, datetime):
                first_seen = first_seen_str
            else:
                first_seen = last_seen
        except (ValueError, TypeError):
            first_seen = last_seen

        # Age of this transition in days (minimum 1 to avoid division by zero)
        age_days = max((datetime.utcnow() - first_seen).days, 1)

        # Recency ratio: interactions per day (higher = more actively explored)
        recency_ratio = total / age_days

        emerging.append(
            {
                "origin_onet_code": origin,
                "origin_title": graph.nodes[origin].get("title", "Unknown"),
                "target_onet_code": target,
                "target_title": graph.nodes[target].get("title", "Unknown"),
                "total_interactions": total,
                "last_seen": last_seen.isoformat()
                if isinstance(last_seen, datetime)
                else last_seen,
                "recency_ratio": round(recency_ratio, 4),
                "success_rate": attrs.get("success_rate"),
            }
        )

    # Sort by recency ratio descending
    emerging.sort(key=lambda x: x["recency_ratio"], reverse=True)

    return emerging[:top_n]


def transition_difficulty(
    graph: nx.DiGraph,
    origin_onet_code: str,
    target_onet_code: str,
) -> Dict[str, Any]:
    """Assess the difficulty of a specific transition.

    Combines graph-level signals (historical success rate, volume) with
    node-level signals (job zone difference) to produce a difficulty
    assessment.

    Args:
        graph: The transition DiGraph.
        origin_onet_code: Origin occupation code.
        target_onet_code: Target occupation code.

    Returns:
        Dict with difficulty assessment:
            - difficulty_level: "low", "moderate", "high", or "unknown"
            - confidence: "high", "low", or "none" (based on data volume)
            - factors: Dict of contributing signals
            - direct_edge_exists: Whether a direct transition has been observed
            - alternative_paths: Number of multi-hop paths available
    """
    result = {
        "origin_onet_code": origin_onet_code,
        "target_onet_code": target_onet_code,
        "difficulty_level": "unknown",
        "confidence": "none",
        "factors": {},
        "direct_edge_exists": False,
        "alternative_paths": 0,
    }

    # Check if nodes exist
    origin_exists = origin_onet_code in graph
    target_exists = target_onet_code in graph

    if not origin_exists or not target_exists:
        result["factors"]["reason"] = "one or both occupations not in graph"
        return result

    # Job zone difference
    origin_jz = graph.nodes[origin_onet_code].get("job_zone")
    target_jz = graph.nodes[target_onet_code].get("job_zone")
    jz_diff = None
    if origin_jz is not None and target_jz is not None:
        jz_diff = target_jz - origin_jz
        result["factors"]["job_zone_diff"] = jz_diff

    # Direct edge analysis
    if graph.has_edge(origin_onet_code, target_onet_code):
        result["direct_edge_exists"] = True
        edge_data = graph.edges[origin_onet_code, target_onet_code]

        result["factors"]["total_interactions"] = edge_data.get(
            "total_interactions", 0
        )
        result["factors"]["success_rate"] = edge_data.get("success_rate")
        result["factors"]["median_match_score"] = edge_data.get(
            "median_match_score"
        )
        result["factors"]["median_gap_severity"] = edge_data.get(
            "median_gap_severity"
        )

        has_data = edge_data.get("has_sufficient_data", False)
        result["confidence"] = "high" if has_data else "low"

        # Determine difficulty from success rate and match score
        success_rate = edge_data.get("success_rate")
        median_match = edge_data.get("median_match_score")

        if success_rate is not None and has_data:
            if success_rate >= 0.6:
                result["difficulty_level"] = "low"
            elif success_rate >= 0.3:
                result["difficulty_level"] = "moderate"
            else:
                result["difficulty_level"] = "high"
        elif median_match is not None:
            # Fall back to match score when success rate unavailable
            if median_match >= 75:
                result["difficulty_level"] = "low"
            elif median_match >= 50:
                result["difficulty_level"] = "moderate"
            else:
                result["difficulty_level"] = "high"
    else:
        # No direct edge: check for indirect paths
        result["confidence"] = "none"

    # Count alternative multi-hop paths (up to 3 hops)
    try:
        alt_paths = list(
            nx.all_simple_paths(
                graph,
                origin_onet_code,
                target_onet_code,
                cutoff=3,
            )
        )
        result["alternative_paths"] = len(alt_paths)

        # If no direct edge but paths exist, infer moderate-to-high difficulty
        if not result["direct_edge_exists"] and alt_paths:
            result["difficulty_level"] = "moderate"
            result["confidence"] = "low"
            result["factors"]["inferred_from"] = "indirect_paths"

    except nx.NetworkXError:
        pass

    # Adjust difficulty based on job zone difference
    if jz_diff is not None and result["difficulty_level"] != "unknown":
        if jz_diff >= 2:
            # Moving up 2+ job zones is inherently harder
            if result["difficulty_level"] == "low":
                result["difficulty_level"] = "moderate"
            result["factors"]["job_zone_penalty"] = (
                f"Moving up {jz_diff} job zone(s) increases difficulty"
            )

    return result
