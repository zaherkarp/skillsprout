"""Core skills translation engine.

Converts free-text descriptions of work experience into standardised O*NET
skill identifiers using two complementary approaches:

1. **Rule-based**: Regex pattern matching against a curated phrase dictionary.
   Fast, deterministic, and high-precision for known phrases.

2. **TF-IDF similarity**: Cosine similarity between the input text and
   canonical O*NET skill descriptions.  Catches novel phrasing that the
   dictionary does not cover.

Results from both approaches are merged, deduplicated, and returned with a
confidence tier:

    - HIGH   (> 0.8): Near-certain match
    - MEDIUM (0.5 - 0.8): Probable match, may benefit from user confirmation
    - LOW    (0.3 - 0.5): Plausible but uncertain, always ask user to confirm
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.features.skills_translator.skill_dictionary import (
    ONET_SKILL_DESCRIPTIONS,
    ONET_SKILLS,
    PHRASE_TO_SKILL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE_THRESHOLD: float = 0.80
MEDIUM_CONFIDENCE_THRESHOLD: float = 0.50
LOW_CONFIDENCE_THRESHOLD: float = 0.30

# TF-IDF returns many near-zero scores; cap how many we consider.
MAX_TFIDF_MATCHES: int = 10


class ConfidenceLevel(str, Enum):
    """Human-readable confidence tiers for matched skills."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class MatchedSkill:
    """A single skill match produced by the translation engine.

    Attributes:
        element_id: O*NET element ID (e.g. ``"2.B.1.a"``).
        skill_name: Canonical O*NET skill name.
        confidence: Numeric confidence score in ``[0, 1]``.
        confidence_level: Categorical confidence tier.
        source: Which matching approach produced this result
                (``"rule"`` or ``"tfidf"``).
        matched_phrase: The phrase or text fragment that triggered the match.
    """

    element_id: str
    skill_name: str
    confidence: float
    confidence_level: ConfidenceLevel
    source: str
    matched_phrase: str = ""


@dataclass
class TranslationResult:
    """Aggregate result returned by the translator.

    Attributes:
        matched_skills: Skills with confidence >= ``MEDIUM_CONFIDENCE_THRESHOLD``.
        needs_confirmation: Skills with ``LOW`` confidence that the user
            should confirm or reject.
        all_matches: Full list before splitting (useful for debugging).
        input_text: The original free-text input.
    """

    matched_skills: List[MatchedSkill] = field(default_factory=list)
    needs_confirmation: List[MatchedSkill] = field(default_factory=list)
    all_matches: List[MatchedSkill] = field(default_factory=list)
    input_text: str = ""


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class SkillsTranslator:
    """Translates free-text work-experience descriptions into O*NET skills.

    The translator is stateless aside from a lazily-built TF-IDF model that
    is constructed once and reused across calls.

    Usage::

        translator = SkillsTranslator()
        result = translator.translate(
            "I managed a team of 10 retail employees and handled complaints"
        )
        for skill in result.matched_skills:
            print(skill.element_id, skill.skill_name, skill.confidence_level)
    """

    def __init__(self) -> None:
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._skill_ids_index: List[str] = []  # parallel to tfidf_matrix rows

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        confirmed_skill_ids: Optional[Sequence[str]] = None,
    ) -> TranslationResult:
        """Translate free-text into O*NET skills.

        Args:
            text: Free-text description of work experience.
            confirmed_skill_ids: Optional list of element IDs the user has
                already confirmed.  These are promoted to HIGH confidence.

        Returns:
            A ``TranslationResult`` with matched and needs-confirmation lists.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        confirmed = set(confirmed_skill_ids or [])
        text_normalised = self._normalise(text)

        # Phase 1 — rule-based matching
        rule_matches = self._rule_based_match(text_normalised)

        # Phase 2 — TF-IDF similarity matching
        tfidf_matches = self._tfidf_match(text_normalised)

        # Merge (rule-based wins on ties for same element_id)
        merged = self._merge_matches(rule_matches, tfidf_matches)

        # Promote any user-confirmed skills
        for match in merged:
            if match.element_id in confirmed:
                match.confidence = 1.0
                match.confidence_level = ConfidenceLevel.HIGH

        # Split into matched vs. needs_confirmation
        matched: List[MatchedSkill] = []
        needs_confirmation: List[MatchedSkill] = []

        for match in merged:
            if match.confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM):
                matched.append(match)
            elif match.confidence_level == ConfidenceLevel.LOW:
                needs_confirmation.append(match)

        # Sort by confidence descending within each group
        matched.sort(key=lambda m: m.confidence, reverse=True)
        needs_confirmation.sort(key=lambda m: m.confidence, reverse=True)

        result = TranslationResult(
            matched_skills=matched,
            needs_confirmation=needs_confirmation,
            all_matches=merged,
            input_text=text,
        )

        logger.info(
            "Translation complete: %d matched, %d need confirmation "
            "(input length=%d chars)",
            len(matched),
            len(needs_confirmation),
            len(text),
        )
        return result

    # ------------------------------------------------------------------
    # Rule-based matching
    # ------------------------------------------------------------------

    def _rule_based_match(self, text: str) -> List[MatchedSkill]:
        """Match input text against the curated phrase dictionary.

        Uses substring search with word-boundary awareness so that phrases
        like ``"managed a team"`` match inside longer sentences.

        Args:
            text: Normalised (lowercased) input text.

        Returns:
            List of ``MatchedSkill`` objects with ``source="rule"``.
        """
        matches: Dict[str, MatchedSkill] = {}

        for phrase, mapping in PHRASE_TO_SKILL.items():
            # Build a regex that allows the phrase to appear anywhere in
            # the text, respecting word boundaries.
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, text):
                eid = mapping["element_id"]
                # If we already matched this skill via a different phrase,
                # keep whichever match is longer (more specific).
                if eid not in matches or len(phrase) > len(matches[eid].matched_phrase):
                    matches[eid] = MatchedSkill(
                        element_id=eid,
                        skill_name=mapping["skill_name"],
                        confidence=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        source="rule",
                        matched_phrase=phrase,
                    )

        return list(matches.values())

    # ------------------------------------------------------------------
    # TF-IDF similarity matching
    # ------------------------------------------------------------------

    def _ensure_tfidf_model(self) -> None:
        """Lazily build the TF-IDF model on first use."""
        if self._tfidf_vectorizer is not None:
            return

        self._skill_ids_index = list(ONET_SKILL_DESCRIPTIONS.keys())
        corpus = [ONET_SKILL_DESCRIPTIONS[sid] for sid in self._skill_ids_index]

        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

        logger.debug(
            "TF-IDF model built: %d skills, vocabulary size %d",
            len(self._skill_ids_index),
            len(self._tfidf_vectorizer.vocabulary_),
        )

    def _tfidf_match(self, text: str) -> List[MatchedSkill]:
        """Compute cosine similarity of *text* against O*NET skill descriptions.

        Args:
            text: Normalised input text.

        Returns:
            List of ``MatchedSkill`` objects with ``source="tfidf"``
            and confidence above ``LOW_CONFIDENCE_THRESHOLD``.
        """
        self._ensure_tfidf_model()
        assert self._tfidf_vectorizer is not None
        assert self._tfidf_matrix is not None

        query_vec = self._tfidf_vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        matches: List[MatchedSkill] = []
        # Get indices sorted by descending similarity
        top_indices = np.argsort(similarities)[::-1][:MAX_TFIDF_MATCHES]

        for idx in top_indices:
            score = float(similarities[idx])
            if score < LOW_CONFIDENCE_THRESHOLD:
                break  # No point continuing; they are sorted descending

            eid = self._skill_ids_index[idx]
            matches.append(
                MatchedSkill(
                    element_id=eid,
                    skill_name=ONET_SKILLS[eid],
                    confidence=round(score, 4),
                    confidence_level=self._confidence_tier(score),
                    source="tfidf",
                    matched_phrase="(semantic similarity)",
                )
            )

        return matches

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_matches(
        rule_matches: List[MatchedSkill],
        tfidf_matches: List[MatchedSkill],
    ) -> List[MatchedSkill]:
        """Merge rule-based and TF-IDF matches, deduplicating by element ID.

        When both approaches match the same skill, the higher confidence
        wins; ties go to the rule-based match (more interpretable).

        Args:
            rule_matches: Matches from rule-based engine.
            tfidf_matches: Matches from TF-IDF engine.

        Returns:
            Deduplicated, sorted list of ``MatchedSkill``.
        """
        merged: Dict[str, MatchedSkill] = {}

        # Rule-based first (preferred on ties)
        for match in rule_matches:
            merged[match.element_id] = match

        for match in tfidf_matches:
            eid = match.element_id
            if eid not in merged:
                merged[eid] = match
            elif match.confidence > merged[eid].confidence:
                merged[eid] = match
            # else: keep the existing (rule) match

        result = list(merged.values())
        result.sort(key=lambda m: m.confidence, reverse=True)
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase and collapse whitespace.

        Args:
            text: Raw input text.

        Returns:
            Normalised string suitable for matching.
        """
        return " ".join(text.lower().split())

    @staticmethod
    def _confidence_tier(score: float) -> ConfidenceLevel:
        """Map a numeric score to a confidence tier.

        Args:
            score: A value in ``[0, 1]``.

        Returns:
            ``ConfidenceLevel`` enum member.
        """
        if score >= HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        if score >= MEDIUM_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_translator: Optional[SkillsTranslator] = None


def get_translator() -> SkillsTranslator:
    """Return a module-level singleton translator instance.

    The TF-IDF model is built lazily on the first call to ``translate()``,
    so creating the instance itself is cheap.

    Returns:
        A reusable ``SkillsTranslator`` instance.
    """
    global _default_translator
    if _default_translator is None:
        _default_translator = SkillsTranslator()
    return _default_translator
