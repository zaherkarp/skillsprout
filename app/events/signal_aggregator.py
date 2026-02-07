"""Signal aggregator for combining implicit and explicit signals.

Combines all behavioral signals for a (user, occupation) pair into a single
feature vector suitable for the calibration layer. This bridges raw event
data and the learned ranking model.

The aggregated features include:
    - total_dwell_seconds (log-transformed for normalization)
    - explanation_expanded (boolean: user read the explanation)
    - times_viewed (count of distinct viewing sessions)
    - comparison_wins (how many times this occupation was preferred in comparisons)
    - days_since_first_view (recency / familiarity)
    - save_after_explain (strong intent signal: saved after reading explanation)
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.events.implicit_signals import SignalType, get_signal_log

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSignals:
    """Aggregated signal features for a (user, occupation) pair.

    These features are designed to be consumed by the calibration layer
    (logistic regression) or a future LambdaMART model. All features
    are numeric and bounded to avoid scaling issues.
    """

    user_id: int
    occupation_code: str

    # Dwell time features
    total_dwell_seconds: float = 0.0
    total_dwell_log: float = 0.0  # log1p(total_dwell_seconds)
    max_scroll_depth: float = 0.0
    visible_ratio: float = 0.0  # fraction of heartbeats where card was visible

    # Explanation engagement
    explanation_expanded: bool = False
    explanation_engagement_score: float = 0.0
    clicked_training_link: bool = False

    # View frequency
    times_viewed: int = 0
    days_since_first_view: float = 0.0

    # Comparison outcomes
    comparison_wins: int = 0
    comparison_losses: int = 0
    comparison_win_rate: float = 0.0

    # Composite intent signals
    save_after_explain: bool = False

    # Metadata for debugging
    signal_count: int = 0
    first_signal_at: Optional[str] = None
    last_signal_at: Optional[str] = None

    def to_feature_dict(self) -> Dict[str, float]:
        """Convert to a flat dictionary of numeric features for ML consumption.

        Returns:
            Dictionary mapping feature names to float values. All values
            are numeric and suitable for direct use as model input.
        """
        return {
            "total_dwell_log": self.total_dwell_log,
            "max_scroll_depth": self.max_scroll_depth,
            "visible_ratio": self.visible_ratio,
            "explanation_expanded": 1.0 if self.explanation_expanded else 0.0,
            "explanation_engagement_score": self.explanation_engagement_score,
            "clicked_training_link": 1.0 if self.clicked_training_link else 0.0,
            "times_viewed": float(self.times_viewed),
            "days_since_first_view": self.days_since_first_view,
            "comparison_wins": float(self.comparison_wins),
            "comparison_losses": float(self.comparison_losses),
            "comparison_win_rate": self.comparison_win_rate,
            "save_after_explain": 1.0 if self.save_after_explain else 0.0,
        }

    def to_feature_vector(self) -> List[float]:
        """Convert to an ordered list of float features.

        Returns:
            List of floats in a fixed order matching ``feature_names()``.
        """
        d = self.to_feature_dict()
        return [d[name] for name in self.feature_names()]

    @staticmethod
    def feature_names() -> List[str]:
        """Return the ordered list of feature names.

        Returns:
            List of feature name strings in the canonical order.
        """
        return [
            "total_dwell_log",
            "max_scroll_depth",
            "visible_ratio",
            "explanation_expanded",
            "explanation_engagement_score",
            "clicked_training_link",
            "times_viewed",
            "days_since_first_view",
            "comparison_wins",
            "comparison_losses",
            "comparison_win_rate",
            "save_after_explain",
        ]


class SignalAggregator:
    """Aggregates raw signal events into per-(user, occupation) feature vectors.

    Usage::

        aggregator = SignalAggregator()
        signals = aggregator.aggregate(user_id=1, occupation_code="15-1252.00")
        features = signals.to_feature_dict()

    The aggregator reads from the shared signal log (in-memory for the demo,
    backed by a database or event store in production).
    """

    def __init__(
        self,
        signal_source: Optional[List[Dict[str, Any]]] = None,
        explicit_feedback: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the aggregator.

        Args:
            signal_source: Override signal source for testing. If ``None``,
                uses the global signal log from ``implicit_signals``.
            explicit_feedback: List of explicit feedback events (saves, clicks)
                for computing composite signals like save_after_explain.
        """
        self._signal_source = signal_source
        self._explicit_feedback = explicit_feedback or []

    @property
    def signal_log(self) -> List[Dict[str, Any]]:
        """Return the active signal source."""
        if self._signal_source is not None:
            return self._signal_source
        return get_signal_log()

    def aggregate(
        self,
        user_id: int,
        occupation_code: str,
        reference_time: Optional[datetime] = None,
    ) -> AggregatedSignals:
        """Aggregate all signals for a specific (user, occupation) pair.

        Args:
            user_id: The user to aggregate for.
            occupation_code: The O*NET occupation code to aggregate for.
            reference_time: Reference time for computing ``days_since_first_view``.
                Defaults to ``datetime.utcnow()``.

        Returns:
            ``AggregatedSignals`` instance with all features computed.
        """
        if reference_time is None:
            reference_time = datetime.utcnow()

        result = AggregatedSignals(
            user_id=user_id,
            occupation_code=occupation_code,
        )

        # Filter signals for this (user, occupation) pair
        relevant_signals = self._filter_signals(user_id, occupation_code)

        if not relevant_signals:
            return result

        result.signal_count = len(relevant_signals)

        # Track timestamps for recency
        timestamps: List[str] = []

        # Separate by signal type
        dwell_signals = []
        explanation_signals = []
        comparison_signals = []

        for signal in relevant_signals:
            signal_type = signal.get("signal_type")
            ts = signal.get("timestamp")
            if ts:
                timestamps.append(ts)

            if signal_type == SignalType.DWELL_TIME.value:
                dwell_signals.append(signal)
            elif signal_type == SignalType.EXPLANATION_ENGAGEMENT.value:
                explanation_signals.append(signal)
            elif signal_type == SignalType.COMPARISON_BEHAVIOR.value:
                comparison_signals.append(signal)

        # Aggregate dwell time
        self._aggregate_dwell(result, dwell_signals)

        # Aggregate explanation engagement
        self._aggregate_explanation(result, explanation_signals)

        # Aggregate comparison outcomes
        self._aggregate_comparisons(result, comparison_signals, occupation_code)

        # Compute recency
        if timestamps:
            timestamps.sort()
            result.first_signal_at = timestamps[0]
            result.last_signal_at = timestamps[-1]
            try:
                first_dt = datetime.fromisoformat(timestamps[0])
                delta = reference_time - first_dt
                result.days_since_first_view = max(0.0, delta.total_seconds() / 86400.0)
            except (ValueError, TypeError):
                result.days_since_first_view = 0.0

        # Compute view count from distinct sessions
        session_ids = {
            s.get("session_id")
            for s in dwell_signals
            if s.get("session_id")
        }
        result.times_viewed = max(len(session_ids), 1) if dwell_signals else 0

        # Compute composite: save_after_explain
        result.save_after_explain = self._check_save_after_explain(
            user_id, occupation_code, result.explanation_expanded,
        )

        return result

    def aggregate_all_for_user(
        self,
        user_id: int,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, AggregatedSignals]:
        """Aggregate signals for all occupations a user has interacted with.

        Args:
            user_id: The user to aggregate for.
            reference_time: Reference time for recency features.

        Returns:
            Dictionary mapping occupation codes to their ``AggregatedSignals``.
        """
        occupation_codes: set = set()
        for signal in self.signal_log:
            if signal.get("user_id") != user_id:
                continue
            occ = signal.get("occupation_code")
            if occ:
                occupation_codes.add(occ)
            # Comparison signals reference two occupations
            for key in ("occupation_a", "occupation_b"):
                occ = signal.get(key)
                if occ:
                    occupation_codes.add(occ)

        results: Dict[str, AggregatedSignals] = {}
        for occ_code in occupation_codes:
            results[occ_code] = self.aggregate(
                user_id=user_id,
                occupation_code=occ_code,
                reference_time=reference_time,
            )

        return results

    def _filter_signals(
        self, user_id: int, occupation_code: str,
    ) -> List[Dict[str, Any]]:
        """Filter signal log to entries relevant for (user, occupation).

        Args:
            user_id: Target user ID.
            occupation_code: Target O*NET code.

        Returns:
            List of matching signal dictionaries.
        """
        relevant = []
        for signal in self.signal_log:
            if signal.get("user_id") != user_id:
                continue
            # Direct match on occupation_code
            if signal.get("occupation_code") == occupation_code:
                relevant.append(signal)
            # Comparison signals: match on either side
            elif (
                signal.get("occupation_a") == occupation_code
                or signal.get("occupation_b") == occupation_code
            ):
                relevant.append(signal)
        return relevant

    def _aggregate_dwell(
        self,
        result: AggregatedSignals,
        dwell_signals: List[Dict[str, Any]],
    ) -> None:
        """Aggregate dwell-time signals.

        Uses the maximum dwell estimate per session (last heartbeat in each
        session carries the cumulative count).

        Args:
            result: Aggregated signals object to update in-place.
            dwell_signals: List of dwell-type signal dicts.
        """
        if not dwell_signals:
            return

        # Group by session_id, take max dwell per session
        sessions: Dict[str, Dict[str, Any]] = {}
        for signal in dwell_signals:
            sid = signal.get("session_id", "unknown")
            existing = sessions.get(sid)
            if (
                existing is None
                or signal.get("estimated_dwell_seconds", 0)
                > existing.get("estimated_dwell_seconds", 0)
            ):
                sessions[sid] = signal

        total_dwell = sum(
            s.get("estimated_dwell_seconds", 0) for s in sessions.values()
        )
        result.total_dwell_seconds = total_dwell
        result.total_dwell_log = math.log1p(total_dwell)

        # Max scroll depth across all sessions
        result.max_scroll_depth = max(
            (s.get("max_scroll_depth", 0) for s in sessions.values()),
            default=0.0,
        )

        # Visible ratio: aggregate across all signals
        total_heartbeats = sum(
            s.get("heartbeat_count", 0) for s in sessions.values()
        )
        total_visible = sum(
            s.get("visible_heartbeats", 0) for s in sessions.values()
        )
        if total_heartbeats > 0:
            result.visible_ratio = total_visible / total_heartbeats
        else:
            result.visible_ratio = 0.0

    def _aggregate_explanation(
        self,
        result: AggregatedSignals,
        explanation_signals: List[Dict[str, Any]],
    ) -> None:
        """Aggregate explanation engagement signals.

        Args:
            result: Aggregated signals object to update in-place.
            explanation_signals: List of explanation engagement signal dicts.
        """
        if not explanation_signals:
            return

        for signal in explanation_signals:
            action = signal.get("action", "")
            if action in ("expand", "scroll_to_gaps", "click_training_link", "copy_explanation"):
                result.explanation_expanded = True
            if action == "click_training_link":
                result.clicked_training_link = True

            score = signal.get("engagement_score", 0.0)
            if score > result.explanation_engagement_score:
                result.explanation_engagement_score = score

    def _aggregate_comparisons(
        self,
        result: AggregatedSignals,
        comparison_signals: List[Dict[str, Any]],
        occupation_code: str,
    ) -> None:
        """Aggregate comparison outcomes for this occupation.

        Args:
            result: Aggregated signals object to update in-place.
            comparison_signals: List of comparison signal dicts.
            occupation_code: The occupation to compute wins/losses for.
        """
        wins = 0
        losses = 0

        for signal in comparison_signals:
            preferred = signal.get("preferred_code")
            non_preferred = signal.get("non_preferred_code")

            if preferred == occupation_code:
                wins += 1
            elif non_preferred == occupation_code:
                losses += 1

        result.comparison_wins = wins
        result.comparison_losses = losses
        total = wins + losses
        result.comparison_win_rate = wins / total if total > 0 else 0.0

    def _check_save_after_explain(
        self,
        user_id: int,
        occupation_code: str,
        explanation_expanded: bool,
    ) -> bool:
        """Check whether the user saved this occupation after reading its explanation.

        This composite signal is a strong indicator of informed interest:
        the user not only read the detailed explanation but also chose to
        save the occupation for later.

        Args:
            user_id: The user ID.
            occupation_code: The O*NET code.
            explanation_expanded: Whether the explanation was expanded.

        Returns:
            True if user saved after expanding explanation.
        """
        if not explanation_expanded:
            return False

        # Check explicit feedback for a save action
        for fb in self._explicit_feedback:
            if (
                fb.get("user_id") == user_id
                and fb.get("target_onet_code") == occupation_code
                and fb.get("action_type") == "save"
            ):
                return True

        return False
