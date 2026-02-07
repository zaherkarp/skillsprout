"""Pairwise preference collection for learning-to-rank.

Collects and stores pairwise preference data from comparison behavior
and explicit feedback. This data feeds a future LambdaMART ranking
model that learns to order occupations by user preference rather than
relying solely on deterministic skill-gap scores.

Data schema:
    (user_id, preferred_id, non_preferred_id, context)

The ``context`` field captures the source and strength of the preference
so the ranking model can weight observations appropriately.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ==================== Enums ====================

class PreferenceSource(str, Enum):
    """How the pairwise preference was derived."""
    EXPLICIT_COMPARISON = "explicit_comparison"
    DWELL_TIME_COMPARISON = "dwell_time_comparison"
    SAVE_VS_HIDE = "save_vs_hide"
    CLICK_VS_IGNORE = "click_vs_ignore"
    APPLY_VS_SKIP = "apply_vs_skip"
    EXPLANATION_ENGAGEMENT = "explanation_engagement"


class PreferenceStrengthTier(str, Enum):
    """Qualitative strength of the preference signal."""
    STRONG = "strong"      # Explicit choice, apply, save vs. hide
    MODERATE = "moderate"  # Click vs. ignore, dwell-time disparity
    WEAK = "weak"          # Minor dwell-time difference, scroll depth


# ==================== Core Data Structures ====================

@dataclass
class PairwisePreference:
    """A single pairwise preference observation.

    Represents the fact that a given user preferred ``preferred_id``
    over ``non_preferred_id`` in a specific context.

    Attributes:
        user_id: The user who expressed the preference.
        preferred_id: O*NET code of the preferred occupation.
        non_preferred_id: O*NET code of the non-preferred occupation.
        source: How the preference was derived.
        strength: Qualitative strength tier.
        confidence: Numeric confidence in [0, 1].
        event_id: Associated recommendation event ID, if any.
        context: Arbitrary context metadata.
        created_at: When the preference was recorded.
    """
    user_id: int
    preferred_id: str
    non_preferred_id: str
    source: PreferenceSource
    strength: PreferenceStrengthTier
    confidence: float  # 0.0 - 1.0
    event_id: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary for storage or transport.

        Returns:
            Dictionary representation of the preference.
        """
        return {
            "user_id": self.user_id,
            "preferred_id": self.preferred_id,
            "non_preferred_id": self.non_preferred_id,
            "source": self.source.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "event_id": self.event_id,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
        }


# ==================== Pydantic Schemas ====================

class PairwisePreferenceRecord(BaseModel):
    """Pydantic schema for a stored pairwise preference record.

    This schema mirrors the database table that will hold preference data
    for training the LambdaMART model.
    """
    id: Optional[int] = None
    user_id: int = Field(..., description="User who expressed the preference")
    preferred_id: str = Field(
        ..., description="O*NET code of the preferred occupation",
    )
    non_preferred_id: str = Field(
        ..., description="O*NET code of the non-preferred occupation",
    )
    source: str = Field(
        ..., description="Source of the preference signal",
    )
    strength: str = Field(
        ..., description="Strength tier: strong, moderate, weak",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score",
    )
    event_id: Optional[int] = Field(
        None, description="Recommendation event ID",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Context metadata",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("source")
    def validate_source(cls, v: str) -> str:
        """Validate preference source."""
        valid = {s.value for s in PreferenceSource}
        if v not in valid:
            raise ValueError(f"source must be one of {valid}, got '{v}'")
        return v

    @validator("strength")
    def validate_strength(cls, v: str) -> str:
        """Validate strength tier."""
        valid = {t.value for t in PreferenceStrengthTier}
        if v not in valid:
            raise ValueError(f"strength must be one of {valid}, got '{v}'")
        return v


class LambdaMARTTrainingRow(BaseModel):
    """Schema for a single row in LambdaMART training data export.

    LambdaMART expects query-document pairs with relevance labels.
    Here the 'query' is the user context (user_id + current occupation),
    the 'document' is a candidate occupation, and the relevance is derived
    from pairwise preferences.

    Fields:
        query_id: Identifier grouping preferences into ranking queries.
        doc_id: O*NET code of the candidate occupation.
        relevance: Ordinal relevance label (higher = more preferred).
        features: Feature vector from SignalAggregator + baseline scores.
    """
    query_id: str = Field(
        ..., description="Query identifier (e.g., 'user_1_event_5')",
    )
    doc_id: str = Field(..., description="Candidate occupation O*NET code")
    relevance: int = Field(
        ..., ge=0, description="Ordinal relevance label",
    )
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature vector for ranking",
    )


# ==================== Preference Store ====================

class PairwisePreferenceStore:
    """In-memory store for pairwise preferences.

    In production this would be backed by a database table. The in-memory
    implementation supports the full API so code can be developed and
    tested without database infrastructure.
    """

    def __init__(self) -> None:
        """Initialize an empty preference store."""
        self._preferences: List[PairwisePreference] = []
        self._next_id: int = 1

    @property
    def count(self) -> int:
        """Return the number of stored preferences."""
        return len(self._preferences)

    def add(self, preference: PairwisePreference) -> int:
        """Add a preference to the store.

        Args:
            preference: The pairwise preference to store.

        Returns:
            Integer ID assigned to the stored preference.
        """
        pref_id = self._next_id
        self._next_id += 1
        self._preferences.append(preference)

        logger.debug(
            "Stored preference #%d: user=%d preferred=%s over=%s source=%s",
            pref_id,
            preference.user_id,
            preference.preferred_id,
            preference.non_preferred_id,
            preference.source.value,
        )
        return pref_id

    def add_from_comparison(
        self,
        user_id: int,
        preferred_code: str,
        non_preferred_code: str,
        confidence: float,
        event_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Convenience method to add a preference from comparison behavior.

        Args:
            user_id: User expressing the preference.
            preferred_code: O*NET code of the preferred occupation.
            non_preferred_code: O*NET code of the non-preferred occupation.
            confidence: Confidence in the preference (0-1).
            event_id: Recommendation event ID.
            context: Additional context metadata.

        Returns:
            Integer ID of the stored preference.
        """
        if confidence >= 0.8:
            strength = PreferenceStrengthTier.STRONG
        elif confidence >= 0.5:
            strength = PreferenceStrengthTier.MODERATE
        else:
            strength = PreferenceStrengthTier.WEAK

        preference = PairwisePreference(
            user_id=user_id,
            preferred_id=preferred_code,
            non_preferred_id=non_preferred_code,
            source=PreferenceSource.EXPLICIT_COMPARISON,
            strength=strength,
            confidence=confidence,
            event_id=event_id,
            context=context or {},
        )
        return self.add(preference)

    def add_from_feedback_pair(
        self,
        user_id: int,
        saved_code: str,
        hidden_code: str,
        event_id: Optional[int] = None,
    ) -> int:
        """Add a preference derived from save-vs-hide feedback.

        When a user saves one occupation and hides another from the same
        recommendation set, this constitutes a strong pairwise preference.

        Args:
            user_id: User ID.
            saved_code: O*NET code that was saved.
            hidden_code: O*NET code that was hidden.
            event_id: Recommendation event ID.

        Returns:
            Integer ID of the stored preference.
        """
        preference = PairwisePreference(
            user_id=user_id,
            preferred_id=saved_code,
            non_preferred_id=hidden_code,
            source=PreferenceSource.SAVE_VS_HIDE,
            strength=PreferenceStrengthTier.STRONG,
            confidence=0.95,
            event_id=event_id,
            context={"derived_from": "save_vs_hide"},
        )
        return self.add(preference)

    def get_preferences_for_user(
        self,
        user_id: int,
        min_confidence: float = 0.0,
    ) -> List[PairwisePreference]:
        """Retrieve all preferences for a specific user.

        Args:
            user_id: The user to query for.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching preferences.
        """
        return [
            p for p in self._preferences
            if p.user_id == user_id and p.confidence >= min_confidence
        ]

    def get_preferences_for_occupation(
        self,
        occupation_code: str,
    ) -> List[PairwisePreference]:
        """Retrieve all preferences involving a specific occupation.

        Args:
            occupation_code: O*NET code to query.

        Returns:
            List of preferences where the occupation is either preferred
            or non-preferred.
        """
        return [
            p for p in self._preferences
            if (
                p.preferred_id == occupation_code
                or p.non_preferred_id == occupation_code
            )
        ]

    def get_win_loss_record(
        self,
        occupation_code: str,
        user_id: Optional[int] = None,
    ) -> Dict[str, int]:
        """Get win/loss record for an occupation across all preferences.

        Args:
            occupation_code: O*NET code.
            user_id: Optional filter by user.

        Returns:
            Dictionary with ``wins``, ``losses``, ``total``.
        """
        wins = 0
        losses = 0
        for p in self._preferences:
            if user_id is not None and p.user_id != user_id:
                continue
            if p.preferred_id == occupation_code:
                wins += 1
            elif p.non_preferred_id == occupation_code:
                losses += 1

        return {"wins": wins, "losses": losses, "total": wins + losses}

    def export_for_lambdamart(
        self,
        feature_provider: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Export preferences as LambdaMART training data.

        Converts pairwise preferences into query-document-relevance triples
        suitable for training a LambdaMART model.

        The query is defined as (user_id, event_id). Each occupation
        mentioned in preferences for that query receives a relevance
        score based on its win count.

        Args:
            feature_provider: Optional callable that accepts
                ``(user_id, occupation_code)`` and returns a feature dict.
                If None, features are left empty.

        Returns:
            List of training row dictionaries.
        """
        # Group preferences by query (user_id, event_id)
        queries: Dict[str, Dict[str, int]] = {}

        for pref in self._preferences:
            query_id = f"user_{pref.user_id}_event_{pref.event_id or 0}"

            if query_id not in queries:
                queries[query_id] = {}

            doc_scores = queries[query_id]

            # Increment relevance for preferred, ensure non-preferred exists
            doc_scores[pref.preferred_id] = doc_scores.get(pref.preferred_id, 0) + 1
            if pref.non_preferred_id not in doc_scores:
                doc_scores[pref.non_preferred_id] = 0

        # Convert to rows
        rows: List[Dict[str, Any]] = []
        for query_id, doc_scores in queries.items():
            for doc_id, relevance in doc_scores.items():
                features: Dict[str, float] = {}
                if feature_provider is not None:
                    # Extract user_id from query_id
                    parts = query_id.split("_")
                    try:
                        uid = int(parts[1])
                        features = feature_provider(uid, doc_id)
                    except (IndexError, ValueError, TypeError):
                        pass

                rows.append({
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": relevance,
                    "features": features,
                })

        return rows

    def clear(self) -> None:
        """Clear all stored preferences. Used in tests."""
        self._preferences.clear()
        self._next_id = 1


# ==================== Module-level Singleton ====================

_default_store: Optional[PairwisePreferenceStore] = None


def get_preference_store() -> PairwisePreferenceStore:
    """Get the module-level singleton preference store.

    Returns:
        The shared ``PairwisePreferenceStore`` instance.
    """
    global _default_store
    if _default_store is None:
        _default_store = PairwisePreferenceStore()
    return _default_store


def reset_preference_store() -> None:
    """Reset the singleton store. Used in tests."""
    global _default_store
    _default_store = None
