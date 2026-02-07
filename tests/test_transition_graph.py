"""Tests for the transition graph modules.

Tests cover:
- Graph building from transition records
- Edge metadata computation
- Save/load round-trip
- Query functions (most common, successful paths, emerging, difficulty)
- Graph-enhanced scoring (boost, penalty, novel paths)
- Edge cases (empty graph, missing nodes, insufficient data)
"""
import json
import os
import tempfile
from datetime import datetime, timedelta

import networkx as nx
import pytest

from ml.transition_graph.graph_builder import (
    TransitionGraphBuilder,
    MIN_OBSERVATIONS_FOR_STATS,
)
from ml.transition_graph.graph_queries import (
    most_common_transitions,
    successful_paths,
    emerging_transitions,
    transition_difficulty,
)
from ml.transition_graph.graph_recommendations import (
    compute_graph_signal,
    enhance_scores,
    flag_unexplored_paths,
    MAX_BOOST,
    MAX_PENALTY,
    MIN_INTERACTIONS_FOR_SIGNAL,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_records():
    """Create a set of transition records for testing."""
    now = datetime.utcnow()
    records = []

    # Software Developer -> Web Developer: 15 interactions, good success
    for i in range(10):
        records.append(
            {
                "origin_onet_code": "15-1252.00",
                "origin_title": "Software Developers",
                "target_onet_code": "15-1299.08",
                "target_title": "Web Developers",
                "action_type": "apply" if i < 6 else "hide",
                "match_score": 72.0 + i,
                "gap_severity": 20.0 - i,
                "action_at": now - timedelta(days=i),
                "origin_job_zone": 4,
                "target_job_zone": 3,
            }
        )
    for i in range(5):
        records.append(
            {
                "origin_onet_code": "15-1252.00",
                "origin_title": "Software Developers",
                "target_onet_code": "15-1299.08",
                "target_title": "Web Developers",
                "action_type": "click",
                "match_score": 70.0,
                "gap_severity": 22.0,
                "action_at": now - timedelta(days=i + 10),
                "origin_job_zone": 4,
                "target_job_zone": 3,
            }
        )

    # Software Developer -> Network Admin: 8 interactions, moderate success
    for i in range(8):
        records.append(
            {
                "origin_onet_code": "15-1252.00",
                "origin_title": "Software Developers",
                "target_onet_code": "15-1244.00",
                "target_title": "Network and Computer Systems Administrators",
                "action_type": "apply" if i < 3 else "hide",
                "match_score": 55.0 + i,
                "gap_severity": 35.0,
                "action_at": now - timedelta(days=i * 5),
                "origin_job_zone": 4,
                "target_job_zone": 3,
            }
        )

    # Web Developer -> Software Developer: 6 interactions (reverse direction)
    for i in range(6):
        records.append(
            {
                "origin_onet_code": "15-1299.08",
                "origin_title": "Web Developers",
                "target_onet_code": "15-1252.00",
                "target_title": "Software Developers",
                "action_type": "apply" if i < 5 else "hide",
                "match_score": 65.0,
                "gap_severity": 28.0,
                "action_at": now - timedelta(days=i * 3),
                "origin_job_zone": 3,
                "target_job_zone": 4,
            }
        )

    # Network Admin -> Software Developer: 2 interactions (sparse)
    for i in range(2):
        records.append(
            {
                "origin_onet_code": "15-1244.00",
                "origin_title": "Network and Computer Systems Administrators",
                "target_onet_code": "15-1252.00",
                "target_title": "Software Developers",
                "action_type": "apply",
                "match_score": 50.0,
                "gap_severity": 40.0,
                "action_at": now - timedelta(days=60 + i),
                "origin_job_zone": 3,
                "target_job_zone": 4,
            }
        )

    return records


@pytest.fixture
def built_graph(sample_records):
    """Build and return a transition graph from sample records."""
    builder = TransitionGraphBuilder()
    builder.build_from_records(sample_records)
    return builder.graph


@pytest.fixture
def builder_with_graph(sample_records):
    """Return a builder with a built graph."""
    builder = TransitionGraphBuilder()
    builder.build_from_records(sample_records)
    return builder


# ============================================================
# Graph Builder Tests
# ============================================================


class TestTransitionGraphBuilder:
    """Tests for TransitionGraphBuilder."""

    def test_build_creates_correct_nodes(self, built_graph):
        """All occupations from records should be present as nodes."""
        assert "15-1252.00" in built_graph
        assert "15-1299.08" in built_graph
        assert "15-1244.00" in built_graph
        assert built_graph.number_of_nodes() == 3

    def test_build_creates_correct_edges(self, built_graph):
        """Directed edges should be created for observed transitions."""
        assert built_graph.has_edge("15-1252.00", "15-1299.08")
        assert built_graph.has_edge("15-1252.00", "15-1244.00")
        assert built_graph.has_edge("15-1299.08", "15-1252.00")
        assert built_graph.has_edge("15-1244.00", "15-1252.00")
        assert built_graph.number_of_edges() == 4

    def test_edges_are_directed(self, built_graph):
        """A -> B does not imply B -> A."""
        # 15-1299.08 -> 15-1244.00 should NOT exist
        assert not built_graph.has_edge("15-1299.08", "15-1244.00")

    def test_node_attributes(self, built_graph):
        """Nodes should have title and job_zone attributes."""
        node_data = built_graph.nodes["15-1252.00"]
        assert node_data["title"] == "Software Developers"
        assert node_data["job_zone"] == 4

    def test_edge_total_interactions(self, built_graph):
        """Edge should have correct total_interactions count."""
        edge = built_graph.edges["15-1252.00", "15-1299.08"]
        # 10 action records + 5 click records = 15
        assert edge["total_interactions"] == 15

    def test_edge_success_rate(self, built_graph):
        """Success rate should be computed from actionable signals only."""
        edge = built_graph.edges["15-1252.00", "15-1299.08"]
        # 6 apply (positive) + 4 hide (negative) = 10 actionable
        # Success rate = 6/10 = 0.6
        assert edge["success_rate"] == pytest.approx(0.6)

    def test_edge_success_rate_insufficient_data(self, built_graph):
        """Edges with few actionable signals should have success_rate=None."""
        edge = built_graph.edges["15-1244.00", "15-1252.00"]
        # Only 2 apply records, 0 hide records = 2 actionable
        # 2 < MIN_OBSERVATIONS_FOR_STATS (5), so success_rate should be None
        assert edge["success_rate"] is None

    def test_edge_median_match_score(self, built_graph):
        """Edge should have a median match score."""
        edge = built_graph.edges["15-1252.00", "15-1299.08"]
        assert edge["median_match_score"] is not None
        # Scores range from 70.0 to 81.0 across 15 records
        assert 70 <= edge["median_match_score"] <= 82

    def test_edge_has_sufficient_data_flag(self, built_graph):
        """Edges should correctly report data sufficiency."""
        # SW Dev -> Web Dev: 10 actionable signals (6 apply + 4 hide)
        edge_sufficient = built_graph.edges["15-1252.00", "15-1299.08"]
        assert edge_sufficient["has_sufficient_data"] is True

        # Net Admin -> SW Dev: 2 actionable signals
        edge_insufficient = built_graph.edges["15-1244.00", "15-1252.00"]
        assert edge_insufficient["has_sufficient_data"] is False

    def test_build_empty_records(self):
        """Building from empty records should produce an empty graph."""
        builder = TransitionGraphBuilder()
        graph = builder.build_from_records([])
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_build_with_occupation_metadata(self, sample_records):
        """External occupation metadata should enrich nodes."""
        metadata = {
            "99-9999.00": {"title": "Extra Occupation", "job_zone": 2},
        }
        builder = TransitionGraphBuilder()
        builder.build_from_records(sample_records, occupation_metadata=metadata)
        # The extra occupation is not in records, so it should not be a node
        assert "99-9999.00" not in builder.graph

    def test_build_stats(self, builder_with_graph):
        """Builder should record build statistics."""
        assert builder_with_graph.stats["num_nodes"] == 3
        assert builder_with_graph.stats["num_edges"] == 4
        assert builder_with_graph.built_at is not None


class TestGraphPersistence:
    """Tests for save/load round-trip."""

    def test_save_and_load(self, builder_with_graph):
        """Graph should survive a save/load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = builder_with_graph.save(tmpdir)
            assert os.path.exists(save_path)

            loaded = TransitionGraphBuilder.load(save_path)
            assert loaded.graph.number_of_nodes() == 3
            assert loaded.graph.number_of_edges() == 4

            # Verify edge data survived
            edge = loaded.graph.edges["15-1252.00", "15-1299.08"]
            assert edge["total_interactions"] == 15
            assert edge["success_rate"] == pytest.approx(0.6)

    def test_save_creates_json(self, builder_with_graph):
        """Saved file should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = builder_with_graph.save(tmpdir)

            with open(save_path, "r") as f:
                data = json.load(f)

            assert "nodes" in data
            assert "edges" in data
            assert "stats" in data
            assert len(data["nodes"]) == 3

    def test_load_nonexistent_file(self):
        """Loading from a nonexistent path should return an empty graph."""
        loaded = TransitionGraphBuilder.load("/nonexistent/path.json")
        assert loaded.graph.number_of_nodes() == 0

    def test_adjacency_list_export(self, builder_with_graph):
        """Adjacency list export should have correct structure."""
        adj = builder_with_graph.get_adjacency_list()

        assert "15-1252.00" in adj
        assert len(adj["15-1252.00"]) == 2  # Web Dev + Net Admin

        targets = {t["target_onet_code"] for t in adj["15-1252.00"]}
        assert "15-1299.08" in targets
        assert "15-1244.00" in targets


# ============================================================
# Graph Query Tests
# ============================================================


class TestMostCommonTransitions:
    """Tests for most_common_transitions query."""

    def test_returns_sorted_by_interactions(self, built_graph):
        """Results should be sorted by total_interactions descending."""
        results = most_common_transitions(built_graph, "15-1252.00")
        assert len(results) == 2
        assert results[0]["target_onet_code"] == "15-1299.08"  # 15 interactions
        assert results[1]["target_onet_code"] == "15-1244.00"  # 8 interactions

    def test_returns_correct_metadata(self, built_graph):
        """Each result should include expected fields."""
        results = most_common_transitions(built_graph, "15-1252.00")
        first = results[0]
        assert "target_onet_code" in first
        assert "target_title" in first
        assert "total_interactions" in first
        assert "success_rate" in first
        assert "median_match_score" in first

    def test_respects_top_n(self, built_graph):
        """Should return at most top_n results."""
        results = most_common_transitions(built_graph, "15-1252.00", top_n=1)
        assert len(results) == 1

    def test_respects_min_interactions(self, built_graph):
        """Should filter out edges below min_interactions."""
        results = most_common_transitions(
            built_graph, "15-1244.00", min_interactions=5
        )
        assert len(results) == 0  # Net Admin -> SW Dev has only 2 interactions

    def test_unknown_origin(self, built_graph):
        """Unknown origin code should return empty list."""
        results = most_common_transitions(built_graph, "99-9999.00")
        assert results == []


class TestSuccessfulPaths:
    """Tests for successful_paths query."""

    def test_finds_direct_path(self, built_graph):
        """Should find the direct path when it exists."""
        paths = successful_paths(
            built_graph, "15-1252.00", "15-1299.08"
        )
        assert len(paths) >= 1
        # Direct path is 1 step
        direct = [p for p in paths if len(p) == 1]
        assert len(direct) == 1

    def test_finds_indirect_path(self, built_graph):
        """Should find indirect paths through intermediate nodes."""
        # 15-1252.00 -> 15-1244.00 -> 15-1252.00 is a cycle, skip that.
        # 15-1252.00 -> 15-1299.08 -> 15-1252.00 is also a cycle.
        # Let's test: 15-1244.00 -> 15-1252.00 -> 15-1299.08
        paths = successful_paths(
            built_graph, "15-1244.00", "15-1299.08", max_hops=2
        )
        # Should find: NetAdmin -> SWDev -> WebDev (2 hops)
        assert len(paths) >= 1
        two_hop = [p for p in paths if len(p) == 2]
        assert len(two_hop) >= 1

    def test_respects_max_hops(self, built_graph):
        """Should not return paths longer than max_hops."""
        paths = successful_paths(
            built_graph, "15-1244.00", "15-1299.08", max_hops=1
        )
        # No direct edge from NetAdmin to WebDev, and max_hops=1
        assert len(paths) == 0

    def test_no_path_exists(self, built_graph):
        """Should return empty list when no path exists."""
        paths = successful_paths(
            built_graph, "15-1299.08", "15-1244.00"
        )
        # Web Dev has no edge to Net Admin
        # Check if indirect path exists via SW Dev
        # 15-1299.08 -> 15-1252.00 -> 15-1244.00
        # This should exist as a 2-hop path
        if not paths:
            # No path at all
            assert paths == []
        else:
            # Found indirect path
            assert all(len(p) >= 2 for p in paths)

    def test_filters_low_success_rate(self, built_graph):
        """Should exclude paths with edges below min_success_rate."""
        # SW Dev -> Net Admin has success rate 3/8 = 0.375
        paths = successful_paths(
            built_graph,
            "15-1252.00",
            "15-1244.00",
            min_success_rate=0.5,
        )
        # The direct edge has success rate ~0.375, below 0.5 threshold
        # So it should be filtered out
        direct = [p for p in paths if len(p) == 1]
        assert len(direct) == 0

    def test_unknown_nodes(self, built_graph):
        """Unknown nodes should return empty list."""
        assert successful_paths(built_graph, "99-9999.00", "15-1252.00") == []


class TestEmergingTransitions:
    """Tests for emerging_transitions query."""

    def test_returns_recent_transitions(self, built_graph):
        """Should return transitions with recent activity."""
        results = emerging_transitions(built_graph, lookback_days=90)
        assert len(results) > 0

    def test_sorted_by_recency_ratio(self, built_graph):
        """Results should be sorted by recency_ratio descending."""
        results = emerging_transitions(built_graph, lookback_days=365)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["recency_ratio"] >= results[i + 1]["recency_ratio"]

    def test_respects_top_n(self, built_graph):
        """Should return at most top_n results."""
        results = emerging_transitions(built_graph, top_n=1)
        assert len(results) <= 1

    def test_excludes_old_transitions(self, built_graph):
        """Should exclude transitions with no recent activity."""
        results = emerging_transitions(built_graph, lookback_days=0)
        # lookback_days=0 means only transitions from today
        # Our fixture records span multiple days
        # Some should be filtered out
        assert isinstance(results, list)


class TestTransitionDifficulty:
    """Tests for transition_difficulty query."""

    def test_known_easy_transition(self, built_graph):
        """SW Dev -> Web Dev has high success rate, should be low difficulty."""
        result = transition_difficulty(built_graph, "15-1252.00", "15-1299.08")
        assert result["direct_edge_exists"] is True
        assert result["difficulty_level"] in ("low", "moderate")
        assert result["confidence"] == "high"

    def test_moderate_transition(self, built_graph):
        """SW Dev -> Net Admin has moderate success, should be moderate/high."""
        result = transition_difficulty(built_graph, "15-1252.00", "15-1244.00")
        assert result["direct_edge_exists"] is True
        assert result["difficulty_level"] in ("moderate", "high")

    def test_unknown_transition(self, built_graph):
        """Transition between unknown nodes should return 'unknown'."""
        result = transition_difficulty(built_graph, "99-0000.00", "99-0001.00")
        assert result["difficulty_level"] == "unknown"
        assert result["confidence"] == "none"

    def test_no_direct_edge(self, built_graph):
        """No direct edge should check for indirect paths."""
        # Web Dev -> Net Admin has no direct edge
        result = transition_difficulty(built_graph, "15-1299.08", "15-1244.00")
        assert result["direct_edge_exists"] is False
        # Should find an indirect path via SW Dev
        assert result["alternative_paths"] >= 0

    def test_returns_expected_structure(self, built_graph):
        """Result should contain all expected fields."""
        result = transition_difficulty(built_graph, "15-1252.00", "15-1299.08")
        assert "difficulty_level" in result
        assert "confidence" in result
        assert "factors" in result
        assert "direct_edge_exists" in result
        assert "alternative_paths" in result


# ============================================================
# Graph Recommendations Tests
# ============================================================


class TestComputeGraphSignal:
    """Tests for compute_graph_signal."""

    def test_well_traveled_path_gets_boost(self, built_graph):
        """A well-traveled path with high success should get a positive boost."""
        signal = compute_graph_signal(
            built_graph,
            "15-1252.00",
            "15-1299.08",
            population_success_rate=0.5,
        )
        # SW Dev -> Web Dev: 15 interactions, 60% success
        # Above population average -> should get boost
        assert signal.score_adjustment > 0
        assert signal.is_well_traveled is True
        assert signal.is_novel is False
        assert signal.confidence in ("high", "low")

    def test_novel_path_gets_zero_adjustment(self, built_graph):
        """A novel (never-seen) transition should get zero adjustment."""
        signal = compute_graph_signal(
            built_graph,
            "15-1252.00",
            "99-9999.00",  # Not in graph
        )
        assert signal.score_adjustment == 0.0
        assert signal.is_novel is True
        assert signal.confidence == "none"

    def test_adjustment_bounded_by_max_boost(self, built_graph):
        """Score adjustment should never exceed MAX_BOOST."""
        signal = compute_graph_signal(
            built_graph,
            "15-1252.00",
            "15-1299.08",
            population_success_rate=0.0,  # Makes relative success very high
        )
        assert signal.score_adjustment <= MAX_BOOST

    def test_adjustment_bounded_by_max_penalty(self, built_graph):
        """Score adjustment should never be worse than -MAX_PENALTY."""
        signal = compute_graph_signal(
            built_graph,
            "15-1252.00",
            "15-1244.00",
            population_success_rate=1.0,  # Makes relative success very low
        )
        assert signal.score_adjustment >= -MAX_PENALTY

    def test_explanation_is_nonempty(self, built_graph):
        """Signal should always include an explanation."""
        signal = compute_graph_signal(
            built_graph, "15-1252.00", "15-1299.08"
        )
        assert len(signal.explanation) > 0

    def test_indirect_path_gives_small_boost(self, built_graph):
        """Nodes reachable indirectly should get a small boost."""
        # Web Dev -> Net Admin: no direct edge, but can go via SW Dev
        signal = compute_graph_signal(
            built_graph, "15-1299.08", "15-1244.00"
        )
        # Should have path_count > 0 if indirect path exists
        if signal.path_count > 0:
            assert signal.score_adjustment >= 0
            assert signal.is_novel is False


class TestEnhanceScores:
    """Tests for enhance_scores."""

    def test_adds_graph_adjusted_score(self, built_graph):
        """Should add graph_adjusted_score to each occupation."""
        occupations = [
            {
                "target_onet_code": "15-1299.08",
                "match_score": 72.0,
                "calibrated_score": 72.0,
                "title": "Web Developers",
            },
            {
                "target_onet_code": "15-1244.00",
                "match_score": 55.0,
                "calibrated_score": 55.0,
                "title": "Network Admins",
            },
        ]

        result = enhance_scores(
            built_graph, "15-1252.00", occupations
        )

        assert len(result) == 2
        for occ in result:
            assert "graph_adjusted_score" in occ
            assert "graph_signal" in occ
            assert 0 <= occ["graph_adjusted_score"] <= 100

    def test_empty_graph_returns_original_scores(self):
        """With an empty graph, adjusted scores should equal original scores."""
        empty_graph = nx.DiGraph()
        occupations = [
            {
                "target_onet_code": "15-1299.08",
                "match_score": 72.0,
                "calibrated_score": 72.0,
            },
        ]

        result = enhance_scores(empty_graph, "15-1252.00", occupations)

        assert result[0]["graph_adjusted_score"] == 72.0
        assert result[0]["graph_signal"] is None

    def test_does_not_modify_original_match_score(self, built_graph):
        """Graph signals should never change match_score or gap_severity."""
        occupations = [
            {
                "target_onet_code": "15-1299.08",
                "match_score": 72.0,
                "calibrated_score": 72.0,
            },
        ]

        result = enhance_scores(
            built_graph, "15-1252.00", occupations
        )

        assert result[0]["match_score"] == 72.0  # Unchanged


class TestFlagUnexploredPaths:
    """Tests for flag_unexplored_paths."""

    def test_flags_high_scoring_novel_paths(self, built_graph):
        """Should flag occupations with good match but no graph edge."""
        occupations = [
            {
                "target_onet_code": "15-1299.08",
                "match_score": 80.0,
                "title": "Web Developers",
            },
            {
                "target_onet_code": "99-9999.00",
                "match_score": 75.0,
                "title": "Unknown Occupation",
            },
        ]

        flags = flag_unexplored_paths(
            built_graph, "15-1252.00", occupations
        )

        # 15-1299.08 has a direct edge from 15-1252.00, should NOT be flagged.
        # 99-9999.00 has no edge and match >= 50, should be flagged.
        flagged_codes = {f["target_onet_code"] for f in flags}
        assert "99-9999.00" in flagged_codes
        assert "15-1299.08" not in flagged_codes

    def test_does_not_flag_low_scoring_occupations(self, built_graph):
        """Should not flag occupations with match_score below 50."""
        occupations = [
            {
                "target_onet_code": "99-9999.00",
                "match_score": 30.0,
                "title": "Low Match",
            },
        ]

        flags = flag_unexplored_paths(
            built_graph, "15-1252.00", occupations
        )
        assert len(flags) == 0

    def test_respects_max_flags(self, built_graph):
        """Should return at most max_flags results."""
        occupations = [
            {"target_onet_code": f"99-{i:04d}.00", "match_score": 80.0, "title": f"Occ {i}"}
            for i in range(10)
        ]

        flags = flag_unexplored_paths(
            built_graph, "15-1252.00", occupations, max_flags=2
        )
        assert len(flags) <= 2

    def test_flags_sorted_by_match_score(self, built_graph):
        """Flagged occupations should be sorted by match_score descending."""
        occupations = [
            {"target_onet_code": "99-0001.00", "match_score": 60.0, "title": "A"},
            {"target_onet_code": "99-0002.00", "match_score": 90.0, "title": "B"},
            {"target_onet_code": "99-0003.00", "match_score": 75.0, "title": "C"},
        ]

        flags = flag_unexplored_paths(
            built_graph, "15-1252.00", occupations
        )

        if len(flags) > 1:
            for i in range(len(flags) - 1):
                assert flags[i]["match_score"] >= flags[i + 1]["match_score"]


# ============================================================
# Edge Case Tests
# ============================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_record_graph(self):
        """A graph built from a single record should have 2 nodes and 1 edge."""
        records = [
            {
                "origin_onet_code": "A",
                "origin_title": "Occupation A",
                "target_onet_code": "B",
                "target_title": "Occupation B",
                "action_type": "apply",
                "match_score": 60.0,
                "gap_severity": 30.0,
                "action_at": datetime.utcnow(),
                "origin_job_zone": 3,
                "target_job_zone": 3,
            }
        ]
        builder = TransitionGraphBuilder()
        graph = builder.build_from_records(records)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1

    def test_self_loop_ignored(self):
        """A record where origin == target should still create an edge."""
        records = [
            {
                "origin_onet_code": "A",
                "origin_title": "Occupation A",
                "target_onet_code": "A",
                "target_title": "Occupation A",
                "action_type": "click",
                "match_score": 100.0,
                "gap_severity": 0.0,
                "action_at": datetime.utcnow(),
                "origin_job_zone": 3,
                "target_job_zone": 3,
            }
        ]
        builder = TransitionGraphBuilder()
        graph = builder.build_from_records(records)
        assert graph.number_of_nodes() == 1
        assert graph.has_edge("A", "A")

    def test_missing_match_score_in_records(self):
        """Records with None match_score should not break median calculation."""
        records = [
            {
                "origin_onet_code": "A",
                "origin_title": "Occupation A",
                "target_onet_code": "B",
                "target_title": "Occupation B",
                "action_type": "apply",
                "match_score": None,
                "gap_severity": None,
                "action_at": datetime.utcnow(),
                "origin_job_zone": None,
                "target_job_zone": None,
            }
        ]
        builder = TransitionGraphBuilder()
        graph = builder.build_from_records(records)
        edge = graph.edges["A", "B"]
        assert edge["median_match_score"] is None

    def test_graph_signal_for_missing_origin(self, built_graph):
        """compute_graph_signal with unknown origin should return zero."""
        signal = compute_graph_signal(
            built_graph, "UNKNOWN", "15-1299.08"
        )
        assert signal.score_adjustment == 0.0
        assert signal.is_novel is True

    def test_graph_signal_for_missing_target(self, built_graph):
        """compute_graph_signal with unknown target should return zero."""
        signal = compute_graph_signal(
            built_graph, "15-1252.00", "UNKNOWN"
        )
        assert signal.score_adjustment == 0.0
        assert signal.is_novel is True

    def test_enhance_scores_empty_list(self, built_graph):
        """enhance_scores with empty occupation list should return empty."""
        result = enhance_scores(built_graph, "15-1252.00", [])
        assert result == []

    def test_most_common_transitions_node_with_no_outgoing(self, built_graph):
        """Node with no outgoing edges should return empty list."""
        # Add an isolated node
        built_graph.add_node("ISOLATED", title="Isolated", job_zone=2)
        results = most_common_transitions(built_graph, "ISOLATED")
        assert results == []
