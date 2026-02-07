"""Implicit signal capture for enhanced event tracking.

Captures behavioral signals that go beyond explicit feedback (click, save, hide)
to build richer user preference profiles. These signals feed into the calibration
layer and future LambdaMART ranking model.

Signal types:
    - DWELL_TIME: Heartbeat-based view duration tracking
    - EXPLANATION_ENGAGEMENT: Whether users expand/read explanations
    - COMPARISON_BEHAVIOR: Pairwise preferences from compare actions
    - SEARCH_REFINEMENT: Tracking changes in user risk tolerance settings
"""
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


# ==================== Signal Type Enum ====================

class SignalType(str, Enum):
    """Types of implicit behavioral signals."""
    DWELL_TIME = "dwell_time"
    EXPLANATION_ENGAGEMENT = "explanation_engagement"
    COMPARISON_BEHAVIOR = "comparison_behavior"
    SEARCH_REFINEMENT = "search_refinement"


class DwellSentiment(str, Enum):
    """Sentiment derived from dwell time duration."""
    POSITIVE = "positive"      # >30 seconds: genuine interest
    NEUTRAL = "neutral"        # 3-30 seconds: browsing
    NEGATIVE = "negative"      # <3 seconds: quick dismissal
    EXTENDED = "extended"      # >120 seconds: deep engagement


# ==================== Request Schemas ====================

class HeartbeatRequest(BaseModel):
    """Heartbeat sent every 5 seconds while a user is viewing an occupation.

    The client sends heartbeats at a fixed interval. The server accumulates
    them into a session-level dwell time measurement. A session ends when
    no heartbeat arrives within 10 seconds (2x the interval).
    """
    user_id: int = Field(..., description="User identifier")
    occupation_code: str = Field(
        ...,
        description="O*NET occupation code being viewed",
        min_length=5,
        max_length=15,
    )
    event_id: Optional[int] = Field(
        None,
        description="Recommendation event ID if viewing from recommendations",
    )
    session_id: str = Field(
        ...,
        description="Client-generated session UUID for grouping heartbeats",
        min_length=10,
    )
    sequence_number: int = Field(
        ...,
        ge=0,
        description="Monotonically increasing sequence number within session",
    )
    viewport_visible: bool = Field(
        True,
        description="Whether the occupation card is in the visible viewport",
    )
    scroll_depth_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="How far down the detail page the user has scrolled",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Client-side timestamp of this heartbeat",
    )


class HeartbeatResponse(BaseModel):
    """Response to a heartbeat event."""
    session_id: str
    total_heartbeats: int
    estimated_dwell_seconds: float
    sentiment: DwellSentiment
    signal_logged: bool


class ExplanationViewRequest(BaseModel):
    """Tracks when a user expands or interacts with an explanation section.

    Explanation engagement is a strong positive signal: users who read
    explanations for why an occupation was recommended show genuine interest.
    """
    user_id: int = Field(..., description="User identifier")
    occupation_code: str = Field(
        ...,
        description="O*NET occupation code whose explanation was viewed",
    )
    event_id: Optional[int] = Field(
        None,
        description="Recommendation event ID",
    )
    action: str = Field(
        ...,
        description="Engagement action: expand, collapse, scroll_to_gaps, click_training_link",
    )
    section: str = Field(
        "explanation",
        description="Which section: explanation, skill_gaps, training_suggestion",
    )
    dwell_on_section_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Milliseconds spent with this section expanded",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator("action")
    def validate_action(cls, v: str) -> str:
        """Validate explanation engagement action."""
        valid_actions = {
            "expand", "collapse", "scroll_to_gaps",
            "click_training_link", "copy_explanation",
        }
        if v not in valid_actions:
            raise ValueError(
                f"action must be one of {valid_actions}, got '{v}'"
            )
        return v


class ExplanationViewResponse(BaseModel):
    """Response after logging explanation engagement."""
    logged: bool
    occupation_code: str
    action: str
    engagement_score: float = Field(
        ...,
        description="0.0-1.0 score reflecting depth of explanation engagement",
    )


class ComparisonEventRequest(BaseModel):
    """Tracks when a user compares two occupations.

    Comparison behavior provides natural pairwise preference data: the
    occupation the user spends more time on, saves, or clicks from the
    comparison view is the preferred item.
    """
    user_id: int = Field(..., description="User identifier")
    occupation_a: str = Field(..., description="First occupation O*NET code")
    occupation_b: str = Field(..., description="Second occupation O*NET code")
    event_id: Optional[int] = Field(None, description="Recommendation event ID")
    chosen_code: Optional[str] = Field(
        None,
        description="Code the user chose/saved/clicked from comparison (if any)",
    )
    dwell_a_ms: Optional[int] = Field(
        None, ge=0, description="Dwell time on occupation A in milliseconds",
    )
    dwell_b_ms: Optional[int] = Field(
        None, ge=0, description="Dwell time on occupation B in milliseconds",
    )
    action: str = Field(
        "compare_view",
        description="Action taken: compare_view, choose_a, choose_b, save_a, save_b, dismiss",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator("action")
    def validate_action(cls, v: str) -> str:
        """Validate comparison action."""
        valid_actions = {
            "compare_view", "choose_a", "choose_b",
            "save_a", "save_b", "dismiss",
        }
        if v not in valid_actions:
            raise ValueError(
                f"action must be one of {valid_actions}, got '{v}'"
            )
        return v


class ComparisonEventResponse(BaseModel):
    """Response after logging comparison behavior."""
    logged: bool
    preferred_code: Optional[str]
    non_preferred_code: Optional[str]
    preference_strength: float = Field(
        ...,
        description="0.0-1.0 confidence in the inferred preference",
    )


class SearchRefinementRequest(BaseModel):
    """Tracks when a user changes risk tolerance or filter settings.

    The before/after pattern reveals user intent: tightening filters
    shows the user wants to narrow down, while loosening shows
    they want more options. This informs personalization.
    """
    user_id: int = Field(..., description="User identifier")
    event_id: Optional[int] = Field(None, description="Recommendation event ID")
    parameter_name: str = Field(
        ...,
        description="Name of the parameter changed (e.g., risk_tolerance, budget, max_training_months)",
    )
    value_before: Any = Field(..., description="Previous parameter value")
    value_after: Any = Field(..., description="New parameter value")
    context: Optional[str] = Field(
        None,
        description="Context about what the user was viewing when they changed settings",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SearchRefinementResponse(BaseModel):
    """Response after logging a search refinement."""
    logged: bool
    parameter_name: str
    direction: str = Field(
        ...,
        description="Direction of change: tightened, loosened, or changed",
    )


# ==================== In-Memory Session Store ====================

class _DwellSession:
    """Tracks heartbeats for a single viewing session."""

    __slots__ = (
        "user_id", "occupation_code", "event_id",
        "heartbeat_count", "first_heartbeat", "last_heartbeat",
        "max_scroll_depth", "visible_heartbeats",
    )

    def __init__(
        self,
        user_id: int,
        occupation_code: str,
        event_id: Optional[int],
        timestamp: datetime,
    ) -> None:
        self.user_id = user_id
        self.occupation_code = occupation_code
        self.event_id = event_id
        self.heartbeat_count: int = 1
        self.first_heartbeat: datetime = timestamp
        self.last_heartbeat: datetime = timestamp
        self.max_scroll_depth: float = 0.0
        self.visible_heartbeats: int = 1


# Global session store. In production this would be backed by Redis.
_active_sessions: Dict[str, _DwellSession] = {}

# Global event log. In production this would persist to the database.
_signal_log: List[Dict[str, Any]] = []


def get_signal_log() -> List[Dict[str, Any]]:
    """Return a reference to the in-memory signal log for testing."""
    return _signal_log


def clear_signal_log() -> None:
    """Clear the in-memory signal log. Used in tests."""
    _signal_log.clear()
    _active_sessions.clear()


def _classify_dwell(seconds: float) -> DwellSentiment:
    """Classify dwell time into a sentiment bucket.

    Thresholds based on UX research on information-seeking behavior:
        - <3s: Quick scan / accidental view (negative signal)
        - 3-30s: Normal browsing (neutral)
        - 30-120s: Genuine interest (positive)
        - >120s: Deep engagement / comparison (extended positive)

    Args:
        seconds: Total dwell time in seconds.

    Returns:
        DwellSentiment classification.
    """
    if seconds < 3.0:
        return DwellSentiment.NEGATIVE
    elif seconds < 30.0:
        return DwellSentiment.NEUTRAL
    elif seconds <= 120.0:
        return DwellSentiment.POSITIVE
    else:
        return DwellSentiment.EXTENDED


def _compute_explanation_engagement_score(
    action: str,
    dwell_on_section_ms: Optional[int],
) -> float:
    """Compute a 0-1 engagement score for explanation interactions.

    Args:
        action: The engagement action taken.
        dwell_on_section_ms: Time spent on the section in milliseconds.

    Returns:
        Float between 0.0 and 1.0.
    """
    base_scores = {
        "expand": 0.3,
        "scroll_to_gaps": 0.5,
        "click_training_link": 0.8,
        "copy_explanation": 0.7,
        "collapse": 0.1,
    }
    score = base_scores.get(action, 0.1)

    if dwell_on_section_ms is not None and dwell_on_section_ms > 0:
        # Log-scaled dwell bonus: up to 0.2 extra for prolonged reading
        dwell_seconds = dwell_on_section_ms / 1000.0
        dwell_bonus = min(0.2, 0.2 * math.log1p(dwell_seconds) / math.log1p(60))
        score = min(1.0, score + dwell_bonus)

    return round(score, 3)


def _infer_comparison_preference(
    request: ComparisonEventRequest,
) -> tuple:
    """Infer pairwise preference from comparison behavior.

    Returns:
        (preferred_code, non_preferred_code, preference_strength)
        All None if no preference can be inferred.
    """
    preferred: Optional[str] = None
    non_preferred: Optional[str] = None
    strength: float = 0.0

    # Explicit choice or save is strongest signal
    if request.action in ("choose_a", "save_a"):
        preferred = request.occupation_a
        non_preferred = request.occupation_b
        strength = 0.9
    elif request.action in ("choose_b", "save_b"):
        preferred = request.occupation_b
        non_preferred = request.occupation_a
        strength = 0.9
    elif request.chosen_code:
        if request.chosen_code == request.occupation_a:
            preferred = request.occupation_a
            non_preferred = request.occupation_b
            strength = 0.8
        elif request.chosen_code == request.occupation_b:
            preferred = request.occupation_b
            non_preferred = request.occupation_a
            strength = 0.8
    elif request.dwell_a_ms is not None and request.dwell_b_ms is not None:
        # Dwell-time based preference: weaker signal
        total = request.dwell_a_ms + request.dwell_b_ms
        if total > 0:
            ratio = max(request.dwell_a_ms, request.dwell_b_ms) / total
            if ratio > 0.65:  # At least 65/35 split to infer preference
                if request.dwell_a_ms > request.dwell_b_ms:
                    preferred = request.occupation_a
                    non_preferred = request.occupation_b
                else:
                    preferred = request.occupation_b
                    non_preferred = request.occupation_a
                strength = min(0.7, ratio)

    return preferred, non_preferred, strength


def _classify_refinement_direction(
    param: str, before: Any, after: Any,
) -> str:
    """Classify whether a parameter change tightens or loosens constraints.

    Args:
        param: Parameter name.
        before: Previous value.
        after: New value.

    Returns:
        'tightened', 'loosened', or 'changed'.
    """
    try:
        before_num = float(before)
        after_num = float(after)
    except (TypeError, ValueError):
        return "changed"

    # Parameters where lower = tighter constraint
    tightening_when_lower = {
        "budget", "max_training_months", "hours_per_week", "risk_tolerance",
        "max_commute_miles",
    }
    # Parameters where higher = tighter constraint
    tightening_when_higher = {
        "min_salary", "min_match_score",
    }

    if param in tightening_when_lower:
        return "tightened" if after_num < before_num else "loosened"
    elif param in tightening_when_higher:
        return "tightened" if after_num > before_num else "loosened"
    else:
        return "changed"


# ==================== API Endpoints ====================

@router.post("/heartbeat", response_model=HeartbeatResponse)
async def record_heartbeat(request: HeartbeatRequest) -> HeartbeatResponse:
    """Record a dwell-time heartbeat for an occupation view.

    Clients should call this endpoint every 5 seconds while the user
    is viewing an occupation detail page. The server accumulates
    heartbeats into sessions and classifies dwell time sentiment.

    Dwell time thresholds:
        - >30s = positive signal (genuine interest)
        - 3-30s = neutral (normal browsing)
        - <3s = negative signal (quick dismissal)
        - >120s = extended engagement
    """
    session = _active_sessions.get(request.session_id)

    if session is None:
        session = _DwellSession(
            user_id=request.user_id,
            occupation_code=request.occupation_code,
            event_id=request.event_id,
            timestamp=request.timestamp,
        )
        _active_sessions[request.session_id] = session
    else:
        session.heartbeat_count += 1
        session.last_heartbeat = request.timestamp
        if request.viewport_visible:
            session.visible_heartbeats += 1

    if request.scroll_depth_pct is not None:
        session.max_scroll_depth = max(
            session.max_scroll_depth, request.scroll_depth_pct,
        )

    # Estimate dwell time: heartbeats * interval (5s)
    estimated_dwell = session.heartbeat_count * 5.0
    sentiment = _classify_dwell(estimated_dwell)

    # Log the signal
    signal_entry = {
        "signal_type": SignalType.DWELL_TIME.value,
        "user_id": request.user_id,
        "occupation_code": request.occupation_code,
        "event_id": request.event_id,
        "session_id": request.session_id,
        "heartbeat_count": session.heartbeat_count,
        "estimated_dwell_seconds": estimated_dwell,
        "sentiment": sentiment.value,
        "max_scroll_depth": session.max_scroll_depth,
        "visible_heartbeats": session.visible_heartbeats,
        "timestamp": request.timestamp.isoformat(),
    }
    _signal_log.append(signal_entry)

    logger.debug(
        "Heartbeat: user=%s occ=%s session=%s seq=%d dwell=%.0fs sentiment=%s",
        request.user_id, request.occupation_code, request.session_id,
        request.sequence_number, estimated_dwell, sentiment.value,
    )

    return HeartbeatResponse(
        session_id=request.session_id,
        total_heartbeats=session.heartbeat_count,
        estimated_dwell_seconds=estimated_dwell,
        sentiment=sentiment,
        signal_logged=True,
    )


@router.post("/explanation-view", response_model=ExplanationViewResponse)
async def record_explanation_view(
    request: ExplanationViewRequest,
) -> ExplanationViewResponse:
    """Record user engagement with an explanation section.

    Tracks when users expand explanations, scroll to skill gaps,
    click training links, or copy explanation text. These are strong
    positive signals indicating genuine interest in an occupation.
    """
    engagement_score = _compute_explanation_engagement_score(
        action=request.action,
        dwell_on_section_ms=request.dwell_on_section_ms,
    )

    signal_entry = {
        "signal_type": SignalType.EXPLANATION_ENGAGEMENT.value,
        "user_id": request.user_id,
        "occupation_code": request.occupation_code,
        "event_id": request.event_id,
        "action": request.action,
        "section": request.section,
        "dwell_on_section_ms": request.dwell_on_section_ms,
        "engagement_score": engagement_score,
        "timestamp": request.timestamp.isoformat(),
    }
    _signal_log.append(signal_entry)

    logger.debug(
        "Explanation view: user=%s occ=%s action=%s score=%.3f",
        request.user_id, request.occupation_code,
        request.action, engagement_score,
    )

    return ExplanationViewResponse(
        logged=True,
        occupation_code=request.occupation_code,
        action=request.action,
        engagement_score=engagement_score,
    )


@router.post("/comparison", response_model=ComparisonEventResponse)
async def record_comparison(
    request: ComparisonEventRequest,
) -> ComparisonEventResponse:
    """Record a comparison event between two occupations.

    When users compare two occupations side-by-side, their behavior
    (choosing one, saving one, spending more time on one) provides
    natural pairwise preference data for learning-to-rank models.
    """
    preferred, non_preferred, strength = _infer_comparison_preference(request)

    signal_entry = {
        "signal_type": SignalType.COMPARISON_BEHAVIOR.value,
        "user_id": request.user_id,
        "occupation_a": request.occupation_a,
        "occupation_b": request.occupation_b,
        "event_id": request.event_id,
        "action": request.action,
        "preferred_code": preferred,
        "non_preferred_code": non_preferred,
        "preference_strength": strength,
        "dwell_a_ms": request.dwell_a_ms,
        "dwell_b_ms": request.dwell_b_ms,
        "timestamp": request.timestamp.isoformat(),
    }
    _signal_log.append(signal_entry)

    logger.debug(
        "Comparison: user=%s a=%s b=%s action=%s preferred=%s strength=%.2f",
        request.user_id, request.occupation_a, request.occupation_b,
        request.action, preferred, strength,
    )

    return ComparisonEventResponse(
        logged=True,
        preferred_code=preferred,
        non_preferred_code=non_preferred,
        preference_strength=strength,
    )


@router.post("/search-refinement", response_model=SearchRefinementResponse)
async def record_search_refinement(
    request: SearchRefinementRequest,
) -> SearchRefinementResponse:
    """Record when a user changes search/filter parameters.

    Tracks the before/after state of parameter changes such as
    risk tolerance, budget constraints, and training timeline.
    The direction of change (tightened vs. loosened) reveals user intent.
    """
    direction = _classify_refinement_direction(
        request.parameter_name,
        request.value_before,
        request.value_after,
    )

    signal_entry = {
        "signal_type": SignalType.SEARCH_REFINEMENT.value,
        "user_id": request.user_id,
        "event_id": request.event_id,
        "parameter_name": request.parameter_name,
        "value_before": request.value_before,
        "value_after": request.value_after,
        "direction": direction,
        "context": request.context,
        "timestamp": request.timestamp.isoformat(),
    }
    _signal_log.append(signal_entry)

    logger.debug(
        "Search refinement: user=%s param=%s %s->%s direction=%s",
        request.user_id, request.parameter_name,
        request.value_before, request.value_after, direction,
    )

    return SearchRefinementResponse(
        logged=True,
        parameter_name=request.parameter_name,
        direction=direction,
    )
