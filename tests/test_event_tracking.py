"""Tests for event tracking: implicit signals, signal aggregation, and pairwise preferences.

Covers:
    - Heartbeat dwell-time tracking and sentiment classification
    - Explanation engagement scoring
    - Comparison behavior and preference inference
    - Search refinement direction classification
    - Signal aggregation for (user, occupation) pairs
    - Pairwise preference store operations
    - LambdaMART export format
"""
import pytest
from datetime import datetime, timedelta

from app.events.implicit_signals import (
    SignalType,
    DwellSentiment,
    HeartbeatRequest,
    ExplanationViewRequest,
    ComparisonEventRequest,
    SearchRefinementRequest,
    _classify_dwell,
    _compute_explanation_engagement_score,
    _infer_comparison_preference,
    _classify_refinement_direction,
    clear_signal_log,
    get_signal_log,
    router as events_router,
)
from app.events.signal_aggregator import (
    SignalAggregator,
    AggregatedSignals,
)
from app.events.pairwise_preference import (
    PairwisePreference,
    PairwisePreferenceStore,
    PreferenceSource,
    PreferenceStrengthTier,
    get_preference_store,
    reset_preference_store,
)

from fastapi.testclient import TestClient
from fastapi import FastAPI


# ==================== Fixtures ====================

@pytest.fixture(autouse=True)
def clean_signal_state():
    """Clear signal log and preference store before each test."""
    clear_signal_log()
    reset_preference_store()
    yield
    clear_signal_log()
    reset_preference_store()


@pytest.fixture
def app():
    """Create a test FastAPI app with the events router."""
    test_app = FastAPI()
    test_app.include_router(events_router, prefix="/api/v1")
    return test_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


# ==================== Dwell Time Classification Tests ====================

class TestDwellTimeClassification:
    """Tests for dwell time sentiment classification."""

    def test_negative_dwell_under_3_seconds(self):
        """Dwell under 3 seconds should be classified as negative."""
        assert _classify_dwell(0.0) == DwellSentiment.NEGATIVE
        assert _classify_dwell(1.0) == DwellSentiment.NEGATIVE
        assert _classify_dwell(2.9) == DwellSentiment.NEGATIVE

    def test_neutral_dwell_3_to_30_seconds(self):
        """Dwell between 3 and 30 seconds should be neutral."""
        assert _classify_dwell(3.0) == DwellSentiment.NEUTRAL
        assert _classify_dwell(15.0) == DwellSentiment.NEUTRAL
        assert _classify_dwell(29.9) == DwellSentiment.NEUTRAL

    def test_positive_dwell_30_to_120_seconds(self):
        """Dwell between 30 and 120 seconds should be positive."""
        assert _classify_dwell(30.0) == DwellSentiment.POSITIVE
        assert _classify_dwell(60.0) == DwellSentiment.POSITIVE
        assert _classify_dwell(120.0) == DwellSentiment.POSITIVE

    def test_extended_dwell_over_120_seconds(self):
        """Dwell over 120 seconds should be extended."""
        assert _classify_dwell(121.0) == DwellSentiment.EXTENDED
        assert _classify_dwell(300.0) == DwellSentiment.EXTENDED


# ==================== Heartbeat Endpoint Tests ====================

class TestHeartbeatEndpoint:
    """Tests for the heartbeat API endpoint."""

    def test_first_heartbeat_creates_session(self, client):
        """First heartbeat should create a new session."""
        response = client.post("/api/v1/events/heartbeat", json={
            "user_id": 1,
            "occupation_code": "15-1252.00",
            "session_id": "test-session-001",
            "sequence_number": 0,
            "viewport_visible": True,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-001"
        assert data["total_heartbeats"] == 1
        assert data["estimated_dwell_seconds"] == 5.0
        assert data["sentiment"] == "neutral"
        assert data["signal_logged"] is True

    def test_multiple_heartbeats_accumulate(self, client):
        """Subsequent heartbeats should accumulate dwell time."""
        for seq in range(7):  # 7 heartbeats = 35 seconds = positive
            response = client.post("/api/v1/events/heartbeat", json={
                "user_id": 1,
                "occupation_code": "15-1252.00",
                "session_id": "test-session-002",
                "sequence_number": seq,
                "viewport_visible": True,
            })

        data = response.json()
        assert data["total_heartbeats"] == 7
        assert data["estimated_dwell_seconds"] == 35.0
        assert data["sentiment"] == "positive"

    def test_heartbeat_logs_signal(self, client):
        """Heartbeat should log to the signal log."""
        client.post("/api/v1/events/heartbeat", json={
            "user_id": 42,
            "occupation_code": "29-1141.00",
            "session_id": "test-session-003",
            "sequence_number": 0,
        })

        log = get_signal_log()
        assert len(log) == 1
        assert log[0]["signal_type"] == SignalType.DWELL_TIME.value
        assert log[0]["user_id"] == 42
        assert log[0]["occupation_code"] == "29-1141.00"

    def test_heartbeat_with_scroll_depth(self, client):
        """Heartbeat should track scroll depth."""
        client.post("/api/v1/events/heartbeat", json={
            "user_id": 1,
            "occupation_code": "15-1252.00",
            "session_id": "test-session-004",
            "sequence_number": 0,
            "scroll_depth_pct": 75.5,
        })

        log = get_signal_log()
        assert log[0]["max_scroll_depth"] == 75.5


# ==================== Explanation Engagement Tests ====================

class TestExplanationEngagement:
    """Tests for explanation engagement scoring and tracking."""

    def test_expand_action_score(self):
        """Expanding an explanation should have a base score of 0.3."""
        score = _compute_explanation_engagement_score("expand", None)
        assert score == 0.3

    def test_click_training_link_high_score(self):
        """Clicking a training link should have a high score."""
        score = _compute_explanation_engagement_score("click_training_link", None)
        assert score == 0.8

    def test_collapse_action_low_score(self):
        """Collapsing without reading should have a low score."""
        score = _compute_explanation_engagement_score("collapse", None)
        assert score == 0.1

    def test_dwell_bonus_increases_score(self):
        """Longer dwell time on section should increase score."""
        score_no_dwell = _compute_explanation_engagement_score("expand", None)
        score_with_dwell = _compute_explanation_engagement_score("expand", 30000)

        assert score_with_dwell > score_no_dwell

    def test_score_capped_at_1(self):
        """Engagement score should never exceed 1.0."""
        score = _compute_explanation_engagement_score("click_training_link", 600000)
        assert score <= 1.0

    def test_explanation_view_endpoint(self, client):
        """Explanation view endpoint should log the signal."""
        response = client.post("/api/v1/events/explanation-view", json={
            "user_id": 1,
            "occupation_code": "15-1252.00",
            "action": "expand",
            "section": "explanation",
            "dwell_on_section_ms": 5000,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["logged"] is True
        assert data["occupation_code"] == "15-1252.00"
        assert data["engagement_score"] > 0

    def test_invalid_action_rejected(self, client):
        """Invalid explanation action should be rejected."""
        response = client.post("/api/v1/events/explanation-view", json={
            "user_id": 1,
            "occupation_code": "15-1252.00",
            "action": "invalid_action",
            "section": "explanation",
        })
        assert response.status_code == 422


# ==================== Comparison Behavior Tests ====================

class TestComparisonBehavior:
    """Tests for comparison event tracking and preference inference."""

    def test_explicit_choice_a(self):
        """Choosing occupation A should infer strong preference."""
        request = ComparisonEventRequest(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            action="choose_a",
            session_id="not-needed",  # Not needed for comparison
        )
        preferred, non_preferred, strength = _infer_comparison_preference(request)

        assert preferred == "15-1252.00"
        assert non_preferred == "29-1141.00"
        assert strength >= 0.8

    def test_explicit_choice_b(self):
        """Choosing occupation B should infer strong preference."""
        request = ComparisonEventRequest(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            action="choose_b",
            session_id="not-needed",
        )
        preferred, non_preferred, strength = _infer_comparison_preference(request)

        assert preferred == "29-1141.00"
        assert non_preferred == "15-1252.00"

    def test_save_a_infers_preference(self):
        """Saving occupation A should infer preference."""
        request = ComparisonEventRequest(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            action="save_a",
            session_id="not-needed",
        )
        preferred, non_preferred, strength = _infer_comparison_preference(request)

        assert preferred == "15-1252.00"
        assert strength >= 0.8

    def test_dwell_time_preference(self):
        """Large dwell time disparity should infer preference."""
        request = ComparisonEventRequest(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            action="compare_view",
            session_id="not-needed",
            dwell_a_ms=30000,
            dwell_b_ms=5000,
        )
        preferred, non_preferred, strength = _infer_comparison_preference(request)

        assert preferred == "15-1252.00"
        assert non_preferred == "29-1141.00"
        assert 0.0 < strength <= 0.7

    def test_equal_dwell_no_preference(self):
        """Equal dwell times should not infer a preference."""
        request = ComparisonEventRequest(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            action="compare_view",
            session_id="not-needed",
            dwell_a_ms=10000,
            dwell_b_ms=10000,
        )
        preferred, non_preferred, strength = _infer_comparison_preference(request)

        assert preferred is None
        assert strength == 0.0

    def test_dismiss_no_preference(self):
        """Dismissing comparison should not infer a preference."""
        request = ComparisonEventRequest(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            action="dismiss",
            session_id="not-needed",
        )
        preferred, non_preferred, strength = _infer_comparison_preference(request)

        assert preferred is None
        assert strength == 0.0

    def test_comparison_endpoint(self, client):
        """Comparison endpoint should log and return preference."""
        response = client.post("/api/v1/events/comparison", json={
            "user_id": 1,
            "occupation_a": "15-1252.00",
            "occupation_b": "29-1141.00",
            "action": "choose_a",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["logged"] is True
        assert data["preferred_code"] == "15-1252.00"
        assert data["non_preferred_code"] == "29-1141.00"
        assert data["preference_strength"] >= 0.8


# ==================== Search Refinement Tests ====================

class TestSearchRefinement:
    """Tests for search refinement tracking."""

    def test_budget_decrease_is_tightened(self):
        """Decreasing budget should be classified as tightening."""
        direction = _classify_refinement_direction("budget", 5000, 2000)
        assert direction == "tightened"

    def test_budget_increase_is_loosened(self):
        """Increasing budget should be classified as loosening."""
        direction = _classify_refinement_direction("budget", 2000, 5000)
        assert direction == "loosened"

    def test_min_salary_increase_is_tightened(self):
        """Increasing minimum salary should be classified as tightening."""
        direction = _classify_refinement_direction("min_salary", 40000, 60000)
        assert direction == "tightened"

    def test_non_numeric_is_changed(self):
        """Non-numeric changes should be classified as 'changed'."""
        direction = _classify_refinement_direction("preferred_format", "online", "in_person")
        assert direction == "changed"

    def test_search_refinement_endpoint(self, client):
        """Search refinement endpoint should log the signal."""
        response = client.post("/api/v1/events/search-refinement", json={
            "user_id": 1,
            "parameter_name": "risk_tolerance",
            "value_before": 0.8,
            "value_after": 0.5,
        })

        assert response.status_code == 200
        data = response.json()
        assert data["logged"] is True
        assert data["parameter_name"] == "risk_tolerance"
        assert data["direction"] == "tightened"


# ==================== Signal Aggregator Tests ====================

class TestSignalAggregator:
    """Tests for signal aggregation across signal types."""

    def _make_dwell_signals(
        self,
        user_id: int,
        occupation_code: str,
        heartbeat_count: int,
        session_id: str = "session-1",
    ):
        """Helper to create dwell-time signals in the log."""
        signals = get_signal_log()
        for i in range(heartbeat_count):
            signals.append({
                "signal_type": SignalType.DWELL_TIME.value,
                "user_id": user_id,
                "occupation_code": occupation_code,
                "session_id": session_id,
                "heartbeat_count": i + 1,
                "estimated_dwell_seconds": (i + 1) * 5.0,
                "max_scroll_depth": min(100.0, (i + 1) * 10.0),
                "visible_heartbeats": i + 1,
                "timestamp": (datetime.utcnow() - timedelta(minutes=5) + timedelta(seconds=i * 5)).isoformat(),
            })

    def _make_explanation_signal(
        self,
        user_id: int,
        occupation_code: str,
        action: str = "expand",
        score: float = 0.3,
    ):
        """Helper to create explanation engagement signals."""
        signals = get_signal_log()
        signals.append({
            "signal_type": SignalType.EXPLANATION_ENGAGEMENT.value,
            "user_id": user_id,
            "occupation_code": occupation_code,
            "action": action,
            "engagement_score": score,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _make_comparison_signal(
        self,
        user_id: int,
        occupation_a: str,
        occupation_b: str,
        preferred: str,
        non_preferred: str,
    ):
        """Helper to create comparison signals."""
        signals = get_signal_log()
        signals.append({
            "signal_type": SignalType.COMPARISON_BEHAVIOR.value,
            "user_id": user_id,
            "occupation_a": occupation_a,
            "occupation_b": occupation_b,
            "preferred_code": preferred,
            "non_preferred_code": non_preferred,
            "preference_strength": 0.9,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def test_aggregate_dwell_time(self):
        """Aggregator should compute total dwell and log transform."""
        self._make_dwell_signals(user_id=1, occupation_code="15-1252.00", heartbeat_count=8)

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.total_dwell_seconds == 40.0  # 8 * 5s
        assert result.total_dwell_log > 0
        assert result.max_scroll_depth > 0
        assert result.visible_ratio > 0

    def test_aggregate_explanation_engagement(self):
        """Aggregator should detect explanation expansion."""
        self._make_explanation_signal(
            user_id=1, occupation_code="15-1252.00",
            action="expand", score=0.3,
        )
        self._make_explanation_signal(
            user_id=1, occupation_code="15-1252.00",
            action="click_training_link", score=0.8,
        )

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.explanation_expanded is True
        assert result.clicked_training_link is True
        assert result.explanation_engagement_score == 0.8

    def test_aggregate_comparison_wins(self):
        """Aggregator should count comparison wins and losses."""
        self._make_comparison_signal(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            preferred="15-1252.00",
            non_preferred="29-1141.00",
        )
        self._make_comparison_signal(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="11-2021.00",
            preferred="15-1252.00",
            non_preferred="11-2021.00",
        )

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.comparison_wins == 2
        assert result.comparison_losses == 0
        assert result.comparison_win_rate == 1.0

    def test_aggregate_comparison_losses(self):
        """Aggregator should track losses correctly."""
        self._make_comparison_signal(
            user_id=1,
            occupation_a="15-1252.00",
            occupation_b="29-1141.00",
            preferred="29-1141.00",
            non_preferred="15-1252.00",
        )

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.comparison_wins == 0
        assert result.comparison_losses == 1
        assert result.comparison_win_rate == 0.0

    def test_aggregate_times_viewed(self):
        """Aggregator should count distinct viewing sessions."""
        self._make_dwell_signals(
            user_id=1, occupation_code="15-1252.00",
            heartbeat_count=3, session_id="session-a",
        )
        self._make_dwell_signals(
            user_id=1, occupation_code="15-1252.00",
            heartbeat_count=2, session_id="session-b",
        )

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.times_viewed == 2

    def test_aggregate_days_since_first_view(self):
        """Aggregator should compute days since first view."""
        signals = get_signal_log()
        old_time = datetime.utcnow() - timedelta(days=3)
        signals.append({
            "signal_type": SignalType.DWELL_TIME.value,
            "user_id": 1,
            "occupation_code": "15-1252.00",
            "session_id": "old-session",
            "heartbeat_count": 1,
            "estimated_dwell_seconds": 5.0,
            "max_scroll_depth": 10.0,
            "visible_heartbeats": 1,
            "timestamp": old_time.isoformat(),
        })

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.days_since_first_view >= 2.9

    def test_aggregate_save_after_explain(self):
        """save_after_explain should be True when both signals are present."""
        self._make_explanation_signal(
            user_id=1, occupation_code="15-1252.00",
            action="expand", score=0.3,
        )

        explicit_feedback = [
            {
                "user_id": 1,
                "target_onet_code": "15-1252.00",
                "action_type": "save",
            },
        ]

        aggregator = SignalAggregator(explicit_feedback=explicit_feedback)
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.save_after_explain is True

    def test_aggregate_no_save_after_explain(self):
        """save_after_explain should be False without save feedback."""
        self._make_explanation_signal(
            user_id=1, occupation_code="15-1252.00",
            action="expand", score=0.3,
        )

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")

        assert result.save_after_explain is False

    def test_feature_dict_output(self):
        """to_feature_dict should return all expected features as floats."""
        self._make_dwell_signals(user_id=1, occupation_code="15-1252.00", heartbeat_count=5)

        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")
        features = result.to_feature_dict()

        expected_keys = AggregatedSignals.feature_names()
        for key in expected_keys:
            assert key in features, f"Missing feature key: {key}"
            assert isinstance(features[key], float), (
                f"Feature {key} should be float, got {type(features[key])}"
            )

    def test_feature_vector_length(self):
        """to_feature_vector should return correct length."""
        result = AggregatedSignals(user_id=1, occupation_code="15-1252.00")
        vec = result.to_feature_vector()
        assert len(vec) == len(AggregatedSignals.feature_names())

    def test_aggregate_all_for_user(self):
        """aggregate_all_for_user should aggregate across all occupations."""
        self._make_dwell_signals(user_id=1, occupation_code="15-1252.00", heartbeat_count=3)
        self._make_dwell_signals(user_id=1, occupation_code="29-1141.00", heartbeat_count=2, session_id="session-2")

        aggregator = SignalAggregator()
        results = aggregator.aggregate_all_for_user(user_id=1)

        assert "15-1252.00" in results
        assert "29-1141.00" in results

    def test_no_signals_returns_defaults(self):
        """Aggregating with no signals should return zero-valued features."""
        aggregator = SignalAggregator()
        result = aggregator.aggregate(user_id=999, occupation_code="99-9999.00")

        assert result.total_dwell_seconds == 0.0
        assert result.explanation_expanded is False
        assert result.times_viewed == 0
        assert result.comparison_wins == 0
        assert result.signal_count == 0


# ==================== Pairwise Preference Store Tests ====================

class TestPairwisePreferenceStore:
    """Tests for the pairwise preference store."""

    def test_add_preference(self):
        """Should store and retrieve a preference."""
        store = PairwisePreferenceStore()
        pref_id = store.add(PairwisePreference(
            user_id=1,
            preferred_id="15-1252.00",
            non_preferred_id="29-1141.00",
            source=PreferenceSource.EXPLICIT_COMPARISON,
            strength=PreferenceStrengthTier.STRONG,
            confidence=0.9,
        ))

        assert pref_id == 1
        assert store.count == 1

    def test_add_from_comparison(self):
        """Should create preference from comparison behavior."""
        store = PairwisePreferenceStore()
        pref_id = store.add_from_comparison(
            user_id=1,
            preferred_code="15-1252.00",
            non_preferred_code="29-1141.00",
            confidence=0.9,
            event_id=5,
        )

        assert pref_id >= 1
        prefs = store.get_preferences_for_user(user_id=1)
        assert len(prefs) == 1
        assert prefs[0].preferred_id == "15-1252.00"
        assert prefs[0].strength == PreferenceStrengthTier.STRONG

    def test_add_from_feedback_pair(self):
        """Should create preference from save-vs-hide feedback."""
        store = PairwisePreferenceStore()
        store.add_from_feedback_pair(
            user_id=1,
            saved_code="15-1252.00",
            hidden_code="29-1141.00",
            event_id=3,
        )

        prefs = store.get_preferences_for_user(user_id=1)
        assert len(prefs) == 1
        assert prefs[0].source == PreferenceSource.SAVE_VS_HIDE
        assert prefs[0].confidence == 0.95

    def test_get_preferences_for_user(self):
        """Should filter preferences by user ID."""
        store = PairwisePreferenceStore()
        store.add_from_comparison(1, "A", "B", 0.9)
        store.add_from_comparison(2, "C", "D", 0.8)
        store.add_from_comparison(1, "E", "F", 0.7)

        user1_prefs = store.get_preferences_for_user(user_id=1)
        assert len(user1_prefs) == 2

        user2_prefs = store.get_preferences_for_user(user_id=2)
        assert len(user2_prefs) == 1

    def test_get_preferences_with_min_confidence(self):
        """Should filter by minimum confidence."""
        store = PairwisePreferenceStore()
        store.add_from_comparison(1, "A", "B", 0.9)
        store.add_from_comparison(1, "C", "D", 0.3)

        high_conf = store.get_preferences_for_user(user_id=1, min_confidence=0.5)
        assert len(high_conf) == 1
        assert high_conf[0].preferred_id == "A"

    def test_get_preferences_for_occupation(self):
        """Should find preferences involving a specific occupation."""
        store = PairwisePreferenceStore()
        store.add_from_comparison(1, "15-1252.00", "29-1141.00", 0.9)
        store.add_from_comparison(1, "11-2021.00", "15-1252.00", 0.8)

        prefs = store.get_preferences_for_occupation("15-1252.00")
        assert len(prefs) == 2  # one win, one loss

    def test_win_loss_record(self):
        """Should correctly compute win/loss record."""
        store = PairwisePreferenceStore()
        store.add_from_comparison(1, "15-1252.00", "29-1141.00", 0.9)
        store.add_from_comparison(1, "15-1252.00", "11-2021.00", 0.8)
        store.add_from_comparison(1, "43-9011.00", "15-1252.00", 0.7)

        record = store.get_win_loss_record("15-1252.00")
        assert record["wins"] == 2
        assert record["losses"] == 1
        assert record["total"] == 3

    def test_win_loss_record_for_user(self):
        """Should compute win/loss filtered by user."""
        store = PairwisePreferenceStore()
        store.add_from_comparison(1, "A", "B", 0.9)
        store.add_from_comparison(2, "B", "A", 0.8)

        record_user1 = store.get_win_loss_record("A", user_id=1)
        assert record_user1["wins"] == 1
        assert record_user1["losses"] == 0

        record_user2 = store.get_win_loss_record("A", user_id=2)
        assert record_user2["wins"] == 0
        assert record_user2["losses"] == 1

    def test_export_for_lambdamart(self):
        """Should export preferences in LambdaMART format."""
        store = PairwisePreferenceStore()
        store.add_from_comparison(1, "A", "B", 0.9, event_id=1)
        store.add_from_comparison(1, "A", "C", 0.8, event_id=1)

        rows = store.export_for_lambdamart()

        assert len(rows) >= 2
        # A should have relevance 2 (won twice), B and C should have 0
        a_rows = [r for r in rows if r["doc_id"] == "A"]
        assert len(a_rows) == 1
        assert a_rows[0]["relevance"] == 2

    def test_preference_to_dict(self):
        """PairwisePreference.to_dict should serialize correctly."""
        pref = PairwisePreference(
            user_id=1,
            preferred_id="15-1252.00",
            non_preferred_id="29-1141.00",
            source=PreferenceSource.EXPLICIT_COMPARISON,
            strength=PreferenceStrengthTier.STRONG,
            confidence=0.9,
            event_id=5,
            context={"page": "comparison"},
        )

        d = pref.to_dict()
        assert d["user_id"] == 1
        assert d["preferred_id"] == "15-1252.00"
        assert d["source"] == "explicit_comparison"
        assert d["confidence"] == 0.9
        assert "created_at" in d

    def test_store_clear(self):
        """clear() should empty the store."""
        store = PairwisePreferenceStore()
        store.add_from_comparison(1, "A", "B", 0.9)
        assert store.count == 1

        store.clear()
        assert store.count == 0

    def test_confidence_strength_mapping(self):
        """Confidence should map to correct strength tiers."""
        store = PairwisePreferenceStore()

        store.add_from_comparison(1, "A", "B", 0.9)  # strong
        store.add_from_comparison(1, "C", "D", 0.6)  # moderate
        store.add_from_comparison(1, "E", "F", 0.3)  # weak

        prefs = store.get_preferences_for_user(user_id=1)
        strengths = {p.preferred_id: p.strength for p in prefs}

        assert strengths["A"] == PreferenceStrengthTier.STRONG
        assert strengths["C"] == PreferenceStrengthTier.MODERATE
        assert strengths["E"] == PreferenceStrengthTier.WEAK
