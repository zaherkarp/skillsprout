"""Cold-start strategy for new users with no interaction history.

When a user has zero (or very few) recorded interactions we cannot rely on
the learned calibration model.  This module provides an
:class:`OccupationPriorModel` that uses *aggregated* interactions from all
users to estimate a prior score for each ``(origin, target)`` pair.

The prior is blended with the baseline scorer output so that as the user
accumulates interactions the prior weight decays to zero.

Integration point
-----------------
Call :meth:`OccupationPriorModel.blend` after the baseline scorer
(``app.ml.scoring.BaselineScorer``) has produced its raw score.  The
blended score can then be passed to the calibration layer.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for the aggregated interaction store.
# Mapping: origin_code -> {target_code -> (positive_count, total_count)}
InteractionStore = Dict[str, Dict[str, Tuple[int, int]]]


class OccupationPriorModel:
    """Lookup-based prior model for cold-start users.

    The model maintains a mapping of ``{origin: {target: prior_score}}``
    computed from aggregated user interactions.

    Parameters
    ----------
    interaction_store : InteractionStore
        Pre-aggregated interaction counts.  Each entry is
        ``origin_code -> target_code -> (positive_count, total_count)``.
    laplace_alpha : float
        Pseudo-count for Laplace smoothing (default 1.0).
    min_users_for_prior : int
        Minimum total users across the entire store before the model is
        considered reliable.  Below this threshold a uniform prior is used.
    max_prior_weight : float
        Maximum blending weight for the prior (caps at this value).
    interaction_halflife : int
        Number of user interactions at which the prior weight halves.
        The formula is ``weight = max_prior_weight * (1 - interactions / (2 * halflife))``.
        Default ``20`` yields the spec formula ``0.3 * (1 - interactions/20)``
        when ``max_prior_weight = 0.3``.
    """

    def __init__(
        self,
        interaction_store: Optional[InteractionStore] = None,
        laplace_alpha: float = 1.0,
        min_users_for_prior: int = 10,
        max_prior_weight: float = 0.3,
        interaction_halflife: int = 20,
    ) -> None:
        self._store: InteractionStore = interaction_store or {}
        self._alpha = laplace_alpha
        self._min_users = min_users_for_prior
        self._max_prior_weight = max_prior_weight
        self._halflife = interaction_halflife

        # Cache the total user count once on init so we can cheaply decide
        # whether to fall back to a uniform prior.
        self._total_users = self._count_total_users()
        self._prior_cache: Dict[str, Dict[str, float]] = {}

        logger.info(
            "OccupationPriorModel initialised — %d total users, "
            "min_users=%d, alpha=%.2f",
            self._total_users,
            self._min_users,
            self._alpha,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_prior(
        self,
        origin_code: str,
        target_code: str,
    ) -> float:
        """Return the prior score for an ``(origin, target)`` pair.

        When the store has fewer than ``min_users_for_prior`` total users,
        this falls back to a uniform prior of 0.5.

        Laplace smoothing is applied so that zero-count pairs still receive
        a small positive prior.

        Args:
            origin_code: O*NET-SOC code for the user's current occupation.
            target_code: O*NET-SOC code for the candidate target.

        Returns:
            Prior score in (0, 1).
        """
        if self._total_users < self._min_users:
            return 0.5  # uniform fallback

        # Check cache
        if origin_code in self._prior_cache:
            cached = self._prior_cache[origin_code]
            if target_code in cached:
                return cached[target_code]

        targets = self._store.get(origin_code, {})
        pos, total = targets.get(target_code, (0, 0))

        # Laplace-smoothed estimate: (pos + alpha) / (total + 2*alpha)
        score = (pos + self._alpha) / (total + 2.0 * self._alpha)

        # Populate cache
        self._prior_cache.setdefault(origin_code, {})[target_code] = score
        return score

    def prior_weight(self, user_interaction_count: int) -> float:
        """Compute the blending weight for the prior.

        The weight decays linearly as the user accumulates interactions::

            weight = min(max_prior_weight,
                         max_prior_weight * (1 - interactions / halflife))

        Once the user has ``halflife`` or more interactions the weight is 0.

        Args:
            user_interaction_count: Number of recorded interactions for the
                user (clicks, saves, applies, etc.).

        Returns:
            Weight in [0, max_prior_weight].
        """
        if user_interaction_count <= 0:
            return self._max_prior_weight

        ratio = 1.0 - (user_interaction_count / self._halflife)
        weight = self._max_prior_weight * max(ratio, 0.0)
        return min(weight, self._max_prior_weight)

    def blend(
        self,
        origin_code: str,
        target_code: str,
        scorer_output: float,
        user_interaction_count: int = 0,
    ) -> float:
        """Blend the prior score with a scorer output.

        Args:
            origin_code: O*NET-SOC code for the origin.
            target_code: O*NET-SOC code for the target.
            scorer_output: Raw score from the baseline or calibration scorer
                (expected 0-100 scale).
            user_interaction_count: How many interactions the user has
                recorded so far.

        Returns:
            Blended score on the same scale as ``scorer_output``.
        """
        w = self.prior_weight(user_interaction_count)

        if w <= 0.0:
            return scorer_output

        prior = self.lookup_prior(origin_code, target_code)
        # Scale prior to the same 0-100 range
        prior_scaled = prior * 100.0

        blended = w * prior_scaled + (1.0 - w) * scorer_output
        return float(blended)

    def update_store(
        self,
        origin_code: str,
        target_code: str,
        is_positive: bool,
    ) -> None:
        """Incrementally update the interaction store.

        This can be called as new feedback arrives so the prior stays
        current without a full recomputation.

        Args:
            origin_code: Origin O*NET-SOC code.
            target_code: Target O*NET-SOC code.
            is_positive: ``True`` for positive outcomes (apply / interview /
                offer), ``False`` for negative (hide).
        """
        if origin_code not in self._store:
            self._store[origin_code] = {}

        pos, total = self._store[origin_code].get(target_code, (0, 0))
        if is_positive:
            pos += 1
        total += 1
        self._store[origin_code][target_code] = (pos, total)

        # Invalidate cache for this origin
        self._prior_cache.pop(origin_code, None)

        # Update total user count (rough — assumes one interaction per user)
        self._total_users = self._count_total_users()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _count_total_users(self) -> int:
        """Sum all interaction counts as a proxy for total users."""
        total = 0
        for targets in self._store.values():
            for _pos, count in targets.values():
                total += count
        return total

    @property
    def is_uniform_fallback(self) -> bool:
        """Whether the model is using the uniform prior fallback."""
        return self._total_users < self._min_users
