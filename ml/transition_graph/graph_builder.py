"""Build directed transition graph from user behavior data.

This module constructs a directed graph where:
- Nodes represent O*NET occupations
- Edges represent observed career transitions between occupations
- Edge metadata captures median skill overlap, time to apply, and success rate

The graph is rebuilt nightly via a Celery task and stored as both
a NetworkX DiGraph and a JSON adjacency list for fast loading.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Default storage path for graph artifacts
GRAPH_DIR = Path("models/transition_graph")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Minimum observations required before computing transition-level statistics.
# Below this threshold, features default to population averages.
MIN_OBSERVATIONS_FOR_STATS = 5


class TransitionGraphBuilder:
    """Builds a directed graph of career transitions from user feedback data.

    The graph is constructed by analyzing RecommendationEvent and UserFeedback
    records. Each user's current occupation (at recommendation time) serves as
    the origin node, and each target occupation they interacted with serves as
    a potential destination node.

    Attributes:
        graph: NetworkX DiGraph with occupations as nodes and transitions as edges.
        built_at: Timestamp of the last build.
        stats: Summary statistics from the last build.
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.built_at: Optional[datetime] = None
        self.stats: Dict[str, Any] = {}

    def build_from_records(
        self,
        transition_records: List[Dict[str, Any]],
        occupation_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> nx.DiGraph:
        """Build the transition graph from raw transition records.

        Each record represents one user interaction with a recommended occupation.

        Args:
            transition_records: List of dicts, each containing:
                - origin_onet_code: User's current occupation code
                - origin_title: User's current occupation title
                - target_onet_code: Recommended occupation code
                - target_title: Recommended occupation title
                - action_type: User's feedback action (click, save, hide, apply,
                  interview, offer)
                - match_score: Baseline match score for this pair
                - gap_severity: Baseline gap severity for this pair
                - action_at: Timestamp of the action
                - origin_job_zone: Origin occupation job zone (optional)
                - target_job_zone: Target occupation job zone (optional)
            occupation_metadata: Optional dict mapping onet_code to metadata
                (title, job_zone, etc.) for node enrichment.

        Returns:
            NetworkX DiGraph with occupations as nodes and transitions as edges.
        """
        logger.info(
            f"Building transition graph from {len(transition_records)} records"
        )

        self.graph = nx.DiGraph()

        # Aggregate records by (origin, target) pair
        edge_data: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        node_titles: Dict[str, str] = {}
        node_job_zones: Dict[str, Optional[int]] = {}

        for record in transition_records:
            origin = record["origin_onet_code"]
            target = record["target_onet_code"]

            edge_data[(origin, target)].append(record)

            # Collect node metadata from records
            if origin not in node_titles and record.get("origin_title"):
                node_titles[origin] = record["origin_title"]
            if target not in node_titles and record.get("target_title"):
                node_titles[target] = record["target_title"]
            if origin not in node_job_zones and record.get("origin_job_zone"):
                node_job_zones[origin] = record["origin_job_zone"]
            if target not in node_job_zones and record.get("target_job_zone"):
                node_job_zones[target] = record["target_job_zone"]

        # Enrich node metadata from occupation_metadata if provided
        if occupation_metadata:
            for code, meta in occupation_metadata.items():
                if code not in node_titles and meta.get("title"):
                    node_titles[code] = meta["title"]
                if code not in node_job_zones and meta.get("job_zone"):
                    node_job_zones[code] = meta["job_zone"]

        # Add nodes
        all_nodes = set()
        for origin, target in edge_data:
            all_nodes.add(origin)
            all_nodes.add(target)

        for node in all_nodes:
            self.graph.add_node(
                node,
                title=node_titles.get(node, "Unknown"),
                job_zone=node_job_zones.get(node),
            )

        # Add edges with computed metadata
        for (origin, target), records in edge_data.items():
            edge_meta = self._compute_edge_metadata(records)
            self.graph.add_edge(origin, target, **edge_meta)

        self.built_at = datetime.utcnow()
        self.stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_records": len(transition_records),
            "built_at": self.built_at.isoformat(),
        }

        logger.info(
            f"Transition graph built: {self.stats['num_nodes']} nodes, "
            f"{self.stats['num_edges']} edges"
        )

        return self.graph

    def _compute_edge_metadata(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute aggregate metadata for a single edge (origin -> target).

        Args:
            records: All interaction records for this (origin, target) pair.

        Returns:
            Dict with edge metadata:
                - total_interactions: Total number of interactions
                - positive_outcomes: Count of apply/interview/offer
                - negative_outcomes: Count of hide
                - success_rate: positive / (positive + negative), or None
                - median_match_score: Median baseline match score
                - median_gap_severity: Median baseline gap severity
                - first_seen: Earliest interaction timestamp
                - last_seen: Most recent interaction timestamp
                - has_sufficient_data: Whether MIN_OBSERVATIONS_FOR_STATS is met
        """
        total = len(records)

        # Classify outcomes
        positive_actions = {"apply", "interview", "offer"}
        negative_actions = {"hide"}

        positive_count = sum(
            1 for r in records if r.get("action_type") in positive_actions
        )
        negative_count = sum(
            1 for r in records if r.get("action_type") in negative_actions
        )

        # Success rate (only from actionable signals)
        actionable_total = positive_count + negative_count
        success_rate = None
        if actionable_total >= MIN_OBSERVATIONS_FOR_STATS:
            success_rate = positive_count / actionable_total

        # Match score statistics
        match_scores = [
            r["match_score"] for r in records if r.get("match_score") is not None
        ]
        median_match_score = float(np.median(match_scores)) if match_scores else None

        # Gap severity statistics
        gap_severities = [
            r["gap_severity"] for r in records if r.get("gap_severity") is not None
        ]
        median_gap_severity = (
            float(np.median(gap_severities)) if gap_severities else None
        )

        # Timestamps
        timestamps = [
            r["action_at"] for r in records if r.get("action_at") is not None
        ]
        first_seen = min(timestamps) if timestamps else None
        last_seen = max(timestamps) if timestamps else None

        # Convert datetime objects to ISO strings for JSON serialization
        if isinstance(first_seen, datetime):
            first_seen = first_seen.isoformat()
        if isinstance(last_seen, datetime):
            last_seen = last_seen.isoformat()

        return {
            "total_interactions": total,
            "positive_outcomes": positive_count,
            "negative_outcomes": negative_count,
            "success_rate": success_rate,
            "median_match_score": median_match_score,
            "median_gap_severity": median_gap_severity,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "has_sufficient_data": actionable_total >= MIN_OBSERVATIONS_FOR_STATS,
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save the graph to disk as a JSON adjacency list.

        The graph is saved in two formats:
        1. JSON adjacency list (for fast loading and API serving)
        2. NetworkX GraphML (for analysis in graph tools)

        Args:
            path: Optional custom directory path. If None, uses default.

        Returns:
            Path to the saved JSON file.
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Saving empty graph")

        save_dir = Path(path) if path else GRAPH_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON adjacency list
        json_path = save_dir / "transition_graph.json"
        graph_data = {
            "nodes": {},
            "edges": {},
            "stats": self.stats,
            "built_at": self.built_at.isoformat() if self.built_at else None,
        }

        for node, attrs in self.graph.nodes(data=True):
            graph_data["nodes"][node] = dict(attrs)

        for origin, target, attrs in self.graph.edges(data=True):
            edge_key = f"{origin}->{target}"
            graph_data["edges"][edge_key] = {
                "origin": origin,
                "target": target,
                **{k: v for k, v in attrs.items()},
            }

        with open(json_path, "w") as f:
            json.dump(graph_data, f, indent=2, default=str)

        logger.info(f"Transition graph saved to {json_path}")
        return str(json_path)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "TransitionGraphBuilder":
        """Load a graph from a JSON adjacency list.

        Args:
            path: Path to the JSON file. If None, uses default location.

        Returns:
            TransitionGraphBuilder with the loaded graph.
        """
        json_path = Path(path) if path else GRAPH_DIR / "transition_graph.json"

        if not json_path.exists():
            logger.warning(f"No transition graph found at {json_path}")
            instance = cls()
            return instance

        with open(json_path, "r") as f:
            graph_data = json.load(f)

        instance = cls()
        instance.stats = graph_data.get("stats", {})
        built_at = graph_data.get("built_at")
        if built_at:
            try:
                instance.built_at = datetime.fromisoformat(built_at)
            except (ValueError, TypeError):
                instance.built_at = None

        # Reconstruct graph
        for node, attrs in graph_data.get("nodes", {}).items():
            instance.graph.add_node(node, **attrs)

        for _edge_key, edge_attrs in graph_data.get("edges", {}).items():
            origin = edge_attrs.pop("origin")
            target = edge_attrs.pop("target")
            instance.graph.add_edge(origin, target, **edge_attrs)

        logger.info(
            f"Transition graph loaded: {instance.graph.number_of_nodes()} nodes, "
            f"{instance.graph.number_of_edges()} edges"
        )

        return instance

    def get_adjacency_list(self) -> Dict[str, List[Dict[str, Any]]]:
        """Export the graph as an adjacency list.

        Returns:
            Dict mapping each origin node to a list of target dicts,
            each containing the target code and edge metadata.
        """
        adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for origin, target, attrs in self.graph.edges(data=True):
            adjacency[origin].append(
                {
                    "target_onet_code": target,
                    "target_title": self.graph.nodes[target].get(
                        "title", "Unknown"
                    ),
                    **attrs,
                }
            )

        return dict(adjacency)


def extract_transition_records_from_db(db_session) -> List[Dict[str, Any]]:
    """Extract transition records from the database for graph building.

    This function queries RecommendationEvent, RecommendedOccupation, and
    UserFeedback to build the raw records needed by TransitionGraphBuilder.

    Args:
        db_session: SQLAlchemy synchronous session.

    Returns:
        List of transition record dicts.
    """
    from app.models.models import (
        RecommendationEvent,
        RecommendedOccupation,
        UserFeedback,
        Occupation,
    )
    from sqlalchemy import select
    from sqlalchemy.orm import joinedload

    records = []

    # Query all feedback with associated recommendation event and occupation data
    query = (
        db_session.query(
            UserFeedback.action_type,
            UserFeedback.action_at,
            UserFeedback.target_onet_code,
            RecommendationEvent.current_onet_code,
            RecommendationEvent.user_id,
            RecommendedOccupation.score_json,
        )
        .join(
            RecommendationEvent,
            UserFeedback.event_id == RecommendationEvent.id,
        )
        .outerjoin(
            RecommendedOccupation,
            (
                RecommendedOccupation.event_id == UserFeedback.event_id
            )
            & (
                RecommendedOccupation.target_onet_code
                == UserFeedback.target_onet_code
            ),
        )
    )

    results = query.all()

    # Also get occupation titles and job zones
    occupations = {
        occ.onet_code: {
            "title": occ.title,
            "job_zone": occ.job_zone,
        }
        for occ in db_session.query(Occupation).all()
    }

    for row in results:
        score_json = row.score_json or {}
        metadata = score_json.get("metadata", {})

        origin_code = row.current_onet_code
        target_code = row.target_onet_code
        origin_meta = occupations.get(origin_code, {})
        target_meta = occupations.get(target_code, {})

        records.append(
            {
                "origin_onet_code": origin_code,
                "origin_title": origin_meta.get("title", "Unknown"),
                "target_onet_code": target_code,
                "target_title": target_meta.get("title", "Unknown"),
                "action_type": (
                    row.action_type.value
                    if hasattr(row.action_type, "value")
                    else row.action_type
                ),
                "match_score": score_json.get("match_score"),
                "gap_severity": score_json.get("gap_severity"),
                "action_at": row.action_at,
                "origin_job_zone": origin_meta.get("job_zone"),
                "target_job_zone": metadata.get("target_job_zone"),
            }
        )

    logger.info(f"Extracted {len(records)} transition records from database")
    return records


def build_transition_graph_task():
    """Celery-compatible task to rebuild the transition graph nightly.

    This function is designed to be called by a Celery task. It:
    1. Extracts transition records from the database
    2. Builds the directed graph
    3. Saves to disk

    Returns:
        Dict with build statistics.
    """
    from app.db.session import SyncSessionLocal

    logger.info("Starting transition graph build task")

    db = SyncSessionLocal()
    try:
        records = extract_transition_records_from_db(db)

        if not records:
            logger.warning("No transition records found. Skipping graph build.")
            return {
                "status": "skipped",
                "reason": "no_records",
            }

        builder = TransitionGraphBuilder()
        builder.build_from_records(records)
        save_path = builder.save()

        logger.info(
            f"Transition graph build complete: {builder.stats}"
        )

        return {
            "status": "success",
            "save_path": save_path,
            **builder.stats,
        }

    except Exception as e:
        logger.error(f"Error building transition graph: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        db.close()
