"""Offline evaluation framework for SkillSprout recommendation models.

This module provides a rigorous offline evaluation harness that measures how well
our recommendation models predict which career transitions users will actually
pursue. It is the primary gate for deciding whether a model change ships.

Architecture Overview
---------------------
Because SkillSprout is a recommendation system -- not a simple classifier -- we
face the classic "partial feedback" problem: we only observe outcomes for items
we showed to the user. The framework addresses this through:

1. **Proxy labels**: We define POSITIVE and NEGATIVE using observable user
   behaviour rather than self-reported satisfaction (which is noisy and lagging).
2. **Temporal splitting**: We split data chronologically so the evaluation
   simulates how the model would have performed in production -- never leaking
   future information into training.
3. **Per-bucket metrics**: Different recommendation buckets (READY_NOW,
   TRAINABLE, LONG_RESKILL) serve different user intents, so we measure each
   bucket's quality independently.
4. **Calibration measurement**: A model can rank well but produce poorly
   calibrated probabilities. We track both discrimination (AUC) and calibration
   (expected calibration error, reliability diagrams).

Usage
-----
    from ml.evaluation.eval_framework import (
        EvaluationFramework,
        ProxyLabelDefinition,
        TemporalSplitter,
    )

    framework = EvaluationFramework()
    labels = framework.assign_proxy_labels(interactions_df)
    splits = framework.split_temporal(labels)
    report = framework.compute_metrics(y_true, y_score, buckets)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proxy label definitions
# ---------------------------------------------------------------------------

# WHY these thresholds?
#
# POSITIVE (label=1):
#   (user SAVED the recommendation AND returned within 7 days)
#   OR (user clicked APPLY)
#
#   Rationale: A "save" alone is weak signal -- users bookmark things they
#   never act on. But a save *followed by a return visit within a week*
#   indicates genuine interest. An "apply" click is the strongest in-product
#   signal we have short of an actual interview/offer.
#
# NEGATIVE (label=0):
#   No action (no click, save, apply, or hide) within 14 days of the
#   recommendation being shown.
#
#   Rationale: 14 days gives users enough time to act. Shorter windows
#   produce too many false negatives (users who were busy); longer windows
#   delay how quickly we can evaluate new models.
#
# EXCLUDED (label=None):
#   - Users who clicked but did not save or apply (ambiguous engagement)
#   - Users who hid the recommendation (explicit negative, but too rare to
#     balance the dataset -- we track it separately)
#   - Interactions younger than 14 days (label not yet resolvable)
#
# These definitions should be revisited as product surfaces change.

POSITIVE_RETURN_WINDOW_DAYS: int = 7
NEGATIVE_INACTION_WINDOW_DAYS: int = 14

POSITIVE_ACTION_TYPES: frozenset = frozenset({"apply", "interview", "offer"})
SAVE_ACTION_TYPE: str = "save"
NEGATIVE_EXCLUDED_ACTIONS: frozenset = frozenset({"click"})
HIDE_ACTION_TYPE: str = "hide"


@dataclass(frozen=True)
class ProxyLabelDefinition:
    """Documents and configures the proxy-label scheme.

    This is a frozen dataclass so that label definitions are immutable once
    created -- preventing accidental mutation during evaluation.

    Attributes:
        positive_return_window_days: After a SAVE, the user must return within
            this many days for the interaction to count as POSITIVE.
        negative_inaction_window_days: If the user takes no qualifying action
            within this many days, the interaction is labeled NEGATIVE.
        positive_action_types: Action types that immediately qualify as POSITIVE
            regardless of return behaviour.
        save_action_type: The action type representing a "save/bookmark".
    """

    positive_return_window_days: int = POSITIVE_RETURN_WINDOW_DAYS
    negative_inaction_window_days: int = NEGATIVE_INACTION_WINDOW_DAYS
    positive_action_types: frozenset = POSITIVE_ACTION_TYPES
    save_action_type: str = SAVE_ACTION_TYPE


# ---------------------------------------------------------------------------
# Temporal splitter
# ---------------------------------------------------------------------------


@dataclass
class TemporalSplit:
    """Result of a temporal train/val/test split.

    Attributes:
        train: Training set DataFrame.
        val: Validation set DataFrame.
        test: Held-out test set DataFrame.
        split_dates: Dict mapping split name to its boundary timestamp.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    split_dates: Dict[str, pd.Timestamp]


class TemporalSplitter:
    """Splits interaction data respecting temporal ordering.

    WHY temporal splits instead of random splits?
    ---------------------------------------------
    Random splits leak future information into the training set. In production,
    the model never sees tomorrow's data when making today's predictions. A
    temporal split simulates this: train on the past, evaluate on the future.

    Proportions (configurable):
        train = 70%  -- enough data to learn patterns
        val   = 15%  -- tune hyper-parameters without touching test
        test  = 15%  -- final, unbiased performance estimate

    The split is by *event timestamp*, not by user. This means a single user's
    interactions may span train and test, which mirrors reality.
    """

    def __init__(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        timestamp_col: str = "event_timestamp",
    ) -> None:
        if not np.isclose(train_frac + val_frac + test_frac, 1.0):
            raise ValueError(
                f"Split fractions must sum to 1.0, got "
                f"{train_frac} + {val_frac} + {test_frac} = "
                f"{train_frac + val_frac + test_frac}"
            )
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.timestamp_col = timestamp_col

    def split(self, df: pd.DataFrame) -> TemporalSplit:
        """Split DataFrame chronologically.

        Args:
            df: DataFrame with a timestamp column. Must contain at least 10
                rows to produce a meaningful split.

        Returns:
            TemporalSplit with train, val, test DataFrames.

        Raises:
            ValueError: If DataFrame is too small or missing timestamp column.
        """
        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"DataFrame must contain column '{self.timestamp_col}'. "
                f"Available columns: {list(df.columns)}"
            )
        if len(df) < 10:
            raise ValueError(
                f"Need at least 10 rows for a meaningful split, got {len(df)}"
            )

        sorted_df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        n = len(sorted_df)

        train_end = int(n * self.train_frac)
        val_end = int(n * (self.train_frac + self.val_frac))

        train = sorted_df.iloc[:train_end].copy()
        val = sorted_df.iloc[train_end:val_end].copy()
        test = sorted_df.iloc[val_end:].copy()

        split_dates = {
            "train_start": sorted_df[self.timestamp_col].iloc[0],
            "train_end": sorted_df[self.timestamp_col].iloc[train_end - 1],
            "val_start": sorted_df[self.timestamp_col].iloc[train_end],
            "val_end": sorted_df[self.timestamp_col].iloc[val_end - 1],
            "test_start": sorted_df[self.timestamp_col].iloc[val_end],
            "test_end": sorted_df[self.timestamp_col].iloc[-1],
        }

        logger.info(
            "Temporal split: train=%d (%s to %s), val=%d (%s to %s), "
            "test=%d (%s to %s)",
            len(train),
            split_dates["train_start"],
            split_dates["train_end"],
            len(val),
            split_dates["val_start"],
            split_dates["val_end"],
            len(test),
            split_dates["test_start"],
            split_dates["test_end"],
        )

        return TemporalSplit(
            train=train, val=val, test=test, split_dates=split_dates
        )


# ---------------------------------------------------------------------------
# Metric definitions with annotations
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBinData:
    """Data for a single calibration bin.

    Used to construct reliability diagrams (calibration plots).

    Attributes:
        bin_lower: Lower edge of the predicted-probability bin.
        bin_upper: Upper edge of the predicted-probability bin.
        mean_predicted: Average predicted probability in this bin.
        mean_observed: Fraction of actual positives in this bin (the "true"
            probability).
        count: Number of samples in this bin.
    """

    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_observed: float
    count: int


@dataclass
class BucketMetrics:
    """Metrics scoped to a single recommendation bucket.

    Attributes:
        bucket_name: One of 'ready_now', 'trainable', 'long_reskill'.
        n_samples: Number of scored interactions in this bucket.
        n_positives: Number of positive labels.
        auc_roc: Area under the ROC curve (None if insufficient positives).
        auc_pr: Area under the precision-recall curve.
        precision: Precision at the default 0.5 threshold.
        recall: Recall at the default 0.5 threshold.
    """

    bucket_name: str
    n_samples: int
    n_positives: int
    auc_roc: Optional[float]
    auc_pr: Optional[float]
    precision: Optional[float]
    recall: Optional[float]


@dataclass
class EvaluationReport:
    """Full evaluation report for a model.

    This is the single artifact that a reviewer inspects to decide ship/no-ship.

    Attributes:
        model_version: Identifier of the model being evaluated.
        overall_auc_roc: Aggregate AUC-ROC across all buckets.
        overall_auc_pr: Aggregate AUC-PR (better than AUC-ROC under class
            imbalance, which we expect because most recommendations are not
            acted upon).
        mrr: Mean Reciprocal Rank -- measures how high the first relevant item
            appears in the ranked list. A value of 1.0 means the top-ranked
            item is always relevant; 0.5 means on average the first relevant
            item is second.
        calibration_bins: Data for the reliability diagram.
        expected_calibration_error: Weighted average absolute gap between
            predicted probability and observed frequency across bins.
        bucket_metrics: Per-bucket breakdown.
        n_total: Total number of evaluated interactions.
        n_positive: Total positives.
        n_negative: Total negatives.
        metadata: Arbitrary extra information (thresholds, timestamps, etc.).
    """

    model_version: str
    overall_auc_roc: Optional[float]
    overall_auc_pr: Optional[float]
    mrr: Optional[float]
    calibration_bins: List[CalibrationBinData]
    expected_calibration_error: float
    bucket_metrics: List[BucketMetrics]
    n_total: int
    n_positive: int
    n_negative: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluation framework
# ---------------------------------------------------------------------------


class EvaluationFramework:
    """End-to-end offline evaluation harness for SkillSprout models.

    Typical workflow::

        fw = EvaluationFramework()

        # 1. Assign ground-truth proxy labels
        labeled = fw.assign_proxy_labels(interactions_df)

        # 2. Split temporally
        splits = TemporalSplitter().split(labeled)

        # 3. Compute metrics on test set predictions
        report = fw.compute_metrics(
            y_true=splits.test["label"].values,
            y_score=predictions,
            buckets=splits.test["bucket"].values,
            model_version="v2_calibrated",
        )
    """

    def __init__(
        self,
        label_def: Optional[ProxyLabelDefinition] = None,
        calibration_n_bins: int = 10,
    ) -> None:
        """Initialize the evaluation framework.

        Args:
            label_def: Proxy label definition. Uses defaults if not provided.
            calibration_n_bins: Number of bins for the calibration reliability
                diagram. 10 is standard; fewer bins reduce noise at the cost of
                resolution.
        """
        self.label_def = label_def or ProxyLabelDefinition()
        self.calibration_n_bins = calibration_n_bins

    # ------------------------------------------------------------------
    # Proxy label assignment
    # ------------------------------------------------------------------

    def assign_proxy_labels(
        self,
        interactions: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Assign POSITIVE / NEGATIVE / None proxy labels to interactions.

        The labeling rules encode our best understanding of what constitutes a
        "successful" recommendation:

        POSITIVE (label=1):
            The user performed one of the ``positive_action_types`` (apply,
            interview, offer) on this recommendation, **OR** the user saved it
            AND returned to the platform within ``positive_return_window_days``.

        NEGATIVE (label=0):
            The user performed *no qualifying action* on this recommendation
            within ``negative_inaction_window_days``.

        EXCLUDED (label=NaN):
            The interaction does not yet have enough elapsed time to assign a
            label, or the user's only action was a click (ambiguous signal).

        Expected columns in ``interactions``:
            - ``event_id``: Unique recommendation event identifier.
            - ``user_id``: User identifier.
            - ``target_onet_code``: Recommended occupation code.
            - ``event_timestamp``: When the recommendation was shown.
            - ``action_type``: The user action (may be None for no-action rows).
            - ``action_timestamp``: When the action occurred (may be NaT).
            - ``next_visit_timestamp``: Timestamp of the user's next platform
              visit after the action (may be NaT). Used for save-and-return.
            - ``bucket``: Recommendation bucket (ready_now / trainable /
              long_reskill).

        Args:
            interactions: DataFrame of user-recommendation interactions.
            reference_date: "Now" for computing elapsed time. Defaults to
                the maximum event_timestamp + 30 days (so all events have had
                time to mature).

        Returns:
            A copy of the input DataFrame with a new ``label`` column (float;
            1.0, 0.0, or NaN).
        """
        df = interactions.copy()

        if reference_date is None:
            reference_date = df["event_timestamp"].max() + pd.Timedelta(days=30)

        df["label"] = np.nan  # default: excluded

        # Days since the recommendation was shown
        df["days_since_event"] = (
            (reference_date - df["event_timestamp"]).dt.total_seconds() / 86400
        )

        # --- POSITIVE: direct strong actions ---
        # WHY: apply / interview / offer are unambiguous success signals.
        positive_mask = df["action_type"].isin(self.label_def.positive_action_types)
        df.loc[positive_mask, "label"] = 1.0

        # --- POSITIVE: save + return within window ---
        # WHY: A save alone is weak. Requiring a return visit within 7 days
        # filters out casual bookmarks and keeps genuine interest.
        save_mask = df["action_type"] == self.label_def.save_action_type
        if "next_visit_timestamp" in df.columns:
            return_mask = save_mask & df["next_visit_timestamp"].notna()
            if return_mask.any():
                days_to_return = (
                    (df.loc[return_mask, "next_visit_timestamp"]
                     - df.loc[return_mask, "action_timestamp"])
                    .dt.total_seconds() / 86400
                )
                returned_in_window = days_to_return <= self.label_def.positive_return_window_days
                positive_save_indices = returned_in_window[returned_in_window].index
                df.loc[positive_save_indices, "label"] = 1.0

        # --- NEGATIVE: no action within inaction window ---
        # WHY: 14 days is long enough for most users to act. Recommendations
        # with no action after 14 days are treated as irrelevant.
        no_action_mask = df["action_type"].isna() | (df["action_type"] == "")
        matured_mask = df["days_since_event"] >= self.label_def.negative_inaction_window_days
        df.loc[no_action_mask & matured_mask, "label"] = 0.0

        # --- Also NEGATIVE: explicit "hide" actions ---
        hide_mask = df["action_type"] == HIDE_ACTION_TYPE
        df.loc[hide_mask, "label"] = 0.0

        # Drop helper column
        df.drop(columns=["days_since_event"], inplace=True)

        n_pos = int((df["label"] == 1.0).sum())
        n_neg = int((df["label"] == 0.0).sum())
        n_excl = int(df["label"].isna().sum())
        logger.info(
            "Proxy labels assigned: %d positive, %d negative, %d excluded "
            "(%.1f%% positive rate among labeled)",
            n_pos,
            n_neg,
            n_excl,
            100 * n_pos / max(n_pos + n_neg, 1),
        )

        return df

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        buckets: Optional[np.ndarray] = None,
        model_version: str = "unknown",
        query_groups: Optional[np.ndarray] = None,
    ) -> EvaluationReport:
        """Compute the full suite of evaluation metrics.

        Args:
            y_true: Binary ground-truth labels (1 = positive, 0 = negative).
                Must not contain NaN -- filter excluded labels before calling.
            y_score: Model-predicted scores or probabilities. Higher = more
                likely to be positive. Must be in [0, 1] for calibration
                metrics to be meaningful.
            buckets: Optional array of bucket names (same length as y_true).
                If provided, per-bucket metrics are computed.
            model_version: Model identifier string for the report.
            query_groups: Optional array of query/event IDs. If provided, MRR
                is computed by grouping predictions into per-query ranked lists.

        Returns:
            EvaluationReport with all metrics populated.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)

        if len(y_true) != len(y_score):
            raise ValueError(
                f"y_true and y_score must have same length, got "
                f"{len(y_true)} and {len(y_score)}"
            )

        n_total = len(y_true)
        n_positive = int(y_true.sum())
        n_negative = n_total - n_positive

        # ---- AUC-ROC (overall) ----
        # WHY AUC-ROC?
        # It measures the model's ability to *discriminate* between positive and
        # negative interactions, independent of the classification threshold.
        #
        # GOOD: >= 0.75 means the model ranks positives above negatives most of
        #   the time. >= 0.85 is strong for a recommendation model.
        # BAD:  <= 0.55 is barely better than random. The model provides almost
        #   no useful ranking signal.
        # CAVEAT: AUC-ROC can be misleadingly high when the positive class is
        #   very rare (which it is here -- most recommendations are not acted
        #   upon). That is why we also track AUC-PR.
        overall_auc_roc = self._safe_roc_auc(y_true, y_score)

        # ---- AUC-PR (overall) ----
        # WHY AUC-PR (Average Precision)?
        # Under severe class imbalance (e.g., 5% positive rate), a model that
        # predicts "negative" for everything still gets 0.95 AUC-ROC. AUC-PR
        # focuses on how well the model retrieves the rare positives.
        #
        # GOOD: >= 0.30 with a 5% positive rate means the model is 6x better
        #   than random at surfacing actionable recommendations.
        # BAD:  <= 0.10 with a 5% positive rate means the model adds almost no
        #   precision above the base rate.
        overall_auc_pr = self._safe_average_precision(y_true, y_score)

        # ---- Calibration ----
        # WHY calibration?
        # Even if the model ranks well (high AUC), it may not produce
        # well-calibrated probabilities. If the model says P(positive) = 0.7,
        # we want ~70% of those interactions to actually be positive. Poor
        # calibration misleads downstream systems that consume raw probabilities
        # (e.g., expected-value-based ranking or risk displays to users).
        #
        # GOOD: ECE < 0.05 means predicted probabilities are accurate within
        #   5 percentage points on average.
        # BAD:  ECE > 0.15 means the model's confidence is systematically off
        #   by more than 15 percentage points.
        calibration_bins = self._compute_calibration_bins(y_true, y_score)
        ece = self._compute_ece(calibration_bins)

        # ---- MRR (Mean Reciprocal Rank) ----
        # WHY MRR?
        # MRR answers: "On average, where does the first relevant result
        # appear?" This is the metric users *feel* most directly -- if the
        # best recommendation is always #1, the user does not need to scroll.
        #
        # GOOD: >= 0.5 means the first relevant result is in the top 2 on
        #   average. >= 0.7 is excellent (usually the top item is relevant).
        # BAD:  <= 0.2 means users need to scroll past 5+ irrelevant items
        #   before finding something useful. This causes abandonment.
        mrr = self._compute_mrr(y_true, y_score, query_groups)

        # ---- Per-bucket metrics ----
        # WHY per-bucket?
        # A model might excel at ranking READY_NOW recommendations but fail
        # at TRAINABLE ones. Per-bucket metrics catch these blind spots.
        # The TRAINABLE bucket is the most important for SkillSprout because
        # it is where the most actionable career advice lives.
        bucket_metrics_list: List[BucketMetrics] = []
        if buckets is not None:
            buckets = np.asarray(buckets)
            for bucket_name in sorted(set(buckets)):
                mask = buckets == bucket_name
                bm = self._compute_bucket_metrics(
                    bucket_name, y_true[mask], y_score[mask]
                )
                bucket_metrics_list.append(bm)

        return EvaluationReport(
            model_version=model_version,
            overall_auc_roc=overall_auc_roc,
            overall_auc_pr=overall_auc_pr,
            mrr=mrr,
            calibration_bins=calibration_bins,
            expected_calibration_error=ece,
            bucket_metrics=bucket_metrics_list,
            n_total=n_total,
            n_positive=n_positive,
            n_negative=n_negative,
        )

    # ------------------------------------------------------------------
    # Private metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
        """Compute AUC-ROC, returning None if only one class is present."""
        if len(set(y_true)) < 2:
            logger.warning(
                "Cannot compute AUC-ROC: only one class present in y_true."
            )
            return None
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_average_precision(
        y_true: np.ndarray, y_score: np.ndarray
    ) -> Optional[float]:
        """Compute average precision, returning None if only one class."""
        if len(set(y_true)) < 2:
            logger.warning(
                "Cannot compute AUC-PR: only one class present in y_true."
            )
            return None
        return float(average_precision_score(y_true, y_score))

    def _compute_calibration_bins(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> List[CalibrationBinData]:
        """Bin predictions and compute calibration data.

        We use equal-width bins (not equal-count) because equal-width bins
        make the reliability diagram easier to interpret: each bin covers the
        same range of predicted probabilities.

        Args:
            y_true: Binary labels.
            y_score: Predicted probabilities in [0, 1].

        Returns:
            List of CalibrationBinData, one per non-empty bin.
        """
        bins: List[CalibrationBinData] = []
        bin_edges = np.linspace(0.0, 1.0, self.calibration_n_bins + 1)

        for i in range(self.calibration_n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i < self.calibration_n_bins - 1:
                mask = (y_score >= lower) & (y_score < upper)
            else:
                # Last bin includes the upper edge
                mask = (y_score >= lower) & (y_score <= upper)

            count = int(mask.sum())
            if count == 0:
                continue

            mean_predicted = float(y_score[mask].mean())
            mean_observed = float(y_true[mask].mean())

            bins.append(
                CalibrationBinData(
                    bin_lower=float(lower),
                    bin_upper=float(upper),
                    mean_predicted=mean_predicted,
                    mean_observed=mean_observed,
                    count=count,
                )
            )

        return bins

    @staticmethod
    def _compute_ece(bins: List[CalibrationBinData]) -> float:
        """Compute Expected Calibration Error from bin data.

        ECE = sum_b (|bin_b| / N) * |mean_predicted_b - mean_observed_b|

        This is a single number summarizing calibration quality. Lower is better.

        Args:
            bins: Calibration bin data.

        Returns:
            ECE value. 0.0 if no bins.
        """
        if not bins:
            return 0.0

        total_count = sum(b.count for b in bins)
        if total_count == 0:
            return 0.0

        ece = sum(
            (b.count / total_count) * abs(b.mean_predicted - b.mean_observed)
            for b in bins
        )
        return float(ece)

    @staticmethod
    def _compute_mrr(
        y_true: np.ndarray,
        y_score: np.ndarray,
        query_groups: Optional[np.ndarray],
    ) -> Optional[float]:
        """Compute Mean Reciprocal Rank.

        For each query (recommendation event), we sort the candidate
        occupations by predicted score (descending) and find the rank of the
        first relevant (positive-label) item. MRR is the mean of 1/rank
        across queries.

        If ``query_groups`` is None, we treat all items as belonging to a
        single query, which gives the reciprocal rank of the single highest-
        scored positive item.

        Args:
            y_true: Binary labels.
            y_score: Predicted scores.
            query_groups: Optional array mapping each item to its query.

        Returns:
            MRR value, or None if no queries contain positives.
        """
        if query_groups is None:
            # Treat everything as one query
            if y_true.sum() == 0:
                return None
            order = np.argsort(-y_score)
            ranked_labels = y_true[order]
            first_relevant = np.where(ranked_labels == 1.0)[0]
            if len(first_relevant) == 0:
                return 0.0
            return float(1.0 / (first_relevant[0] + 1))

        reciprocal_ranks: List[float] = []
        unique_groups = np.unique(query_groups)

        for group in unique_groups:
            mask = query_groups == group
            group_true = y_true[mask]
            group_score = y_score[mask]

            if group_true.sum() == 0:
                # No positives in this query -- skip (standard MRR practice)
                continue

            order = np.argsort(-group_score)
            ranked_labels = group_true[order]
            first_relevant = np.where(ranked_labels == 1.0)[0]

            if len(first_relevant) == 0:
                reciprocal_ranks.append(0.0)
            else:
                reciprocal_ranks.append(1.0 / (first_relevant[0] + 1))

        if not reciprocal_ranks:
            return None

        return float(np.mean(reciprocal_ranks))

    def _compute_bucket_metrics(
        self,
        bucket_name: str,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> BucketMetrics:
        """Compute precision, recall, AUC-ROC, and AUC-PR for one bucket.

        Per-bucket precision and recall are computed at a 0.5 threshold.
        This threshold is arbitrary but provides a simple default comparison
        point. In practice, you would tune the threshold on the validation set.

        Args:
            bucket_name: Name of the bucket.
            y_true: Binary labels for this bucket.
            y_score: Predicted scores for this bucket.

        Returns:
            BucketMetrics instance.
        """
        n_samples = len(y_true)
        n_positives = int(y_true.sum())

        auc_roc_val = self._safe_roc_auc(y_true, y_score)
        auc_pr_val = self._safe_average_precision(y_true, y_score)

        # Precision and recall at threshold = 0.5
        predicted_positive = y_score >= 0.5
        tp = int(((predicted_positive) & (y_true == 1.0)).sum())
        fp = int(((predicted_positive) & (y_true == 0.0)).sum())
        fn = int(((~predicted_positive) & (y_true == 1.0)).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None

        return BucketMetrics(
            bucket_name=bucket_name,
            n_samples=n_samples,
            n_positives=n_positives,
            auc_roc=auc_roc_val,
            auc_pr=auc_pr_val,
            precision=precision,
            recall=recall,
        )

    # ------------------------------------------------------------------
    # Report serialization
    # ------------------------------------------------------------------

    @staticmethod
    def report_to_dict(report: EvaluationReport) -> Dict[str, Any]:
        """Convert an EvaluationReport to a JSON-serializable dict.

        Args:
            report: The evaluation report to serialize.

        Returns:
            Dict suitable for ``json.dumps()``.
        """
        return {
            "model_version": report.model_version,
            "overall_auc_roc": report.overall_auc_roc,
            "overall_auc_pr": report.overall_auc_pr,
            "mrr": report.mrr,
            "expected_calibration_error": report.expected_calibration_error,
            "n_total": report.n_total,
            "n_positive": report.n_positive,
            "n_negative": report.n_negative,
            "calibration_bins": [
                {
                    "bin_lower": b.bin_lower,
                    "bin_upper": b.bin_upper,
                    "mean_predicted": b.mean_predicted,
                    "mean_observed": b.mean_observed,
                    "count": b.count,
                }
                for b in report.calibration_bins
            ],
            "bucket_metrics": [
                {
                    "bucket_name": bm.bucket_name,
                    "n_samples": bm.n_samples,
                    "n_positives": bm.n_positives,
                    "auc_roc": bm.auc_roc,
                    "auc_pr": bm.auc_pr,
                    "precision": bm.precision,
                    "recall": bm.recall,
                }
                for bm in report.bucket_metrics
            ],
            "metadata": report.metadata,
        }

    @staticmethod
    def report_to_markdown(report: EvaluationReport) -> str:
        """Render an EvaluationReport as a Markdown summary.

        This is intended for quick consumption in PR reviews or Slack
        notifications.

        Args:
            report: The evaluation report.

        Returns:
            Markdown-formatted string.
        """
        lines: List[str] = []
        lines.append(f"# Evaluation Report: {report.model_version}")
        lines.append("")
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append(f"| Metric | Value | Interpretation |")
        lines.append(f"|--------|-------|----------------|")

        # AUC-ROC
        auc_roc_str = f"{report.overall_auc_roc:.4f}" if report.overall_auc_roc is not None else "N/A"
        auc_roc_interp = _interpret_auc_roc(report.overall_auc_roc)
        lines.append(f"| AUC-ROC | {auc_roc_str} | {auc_roc_interp} |")

        # AUC-PR
        auc_pr_str = f"{report.overall_auc_pr:.4f}" if report.overall_auc_pr is not None else "N/A"
        auc_pr_interp = _interpret_auc_pr(report.overall_auc_pr, report.n_positive, report.n_total)
        lines.append(f"| AUC-PR | {auc_pr_str} | {auc_pr_interp} |")

        # MRR
        mrr_str = f"{report.mrr:.4f}" if report.mrr is not None else "N/A"
        mrr_interp = _interpret_mrr(report.mrr)
        lines.append(f"| MRR | {mrr_str} | {mrr_interp} |")

        # ECE
        lines.append(
            f"| ECE | {report.expected_calibration_error:.4f} | "
            f"{'Good' if report.expected_calibration_error < 0.05 else 'Needs improvement'} |"
        )

        lines.append("")
        lines.append(
            f"**Dataset**: {report.n_total} interactions "
            f"({report.n_positive} positive, {report.n_negative} negative, "
            f"{100 * report.n_positive / max(report.n_total, 1):.1f}% positive rate)"
        )

        # Per-bucket table
        if report.bucket_metrics:
            lines.append("")
            lines.append("## Per-Bucket Metrics")
            lines.append("")
            lines.append(
                "| Bucket | N | Positives | AUC-ROC | AUC-PR | Precision | Recall |"
            )
            lines.append(
                "|--------|---|-----------|---------|--------|-----------|--------|"
            )
            for bm in report.bucket_metrics:
                auc_r = f"{bm.auc_roc:.3f}" if bm.auc_roc is not None else "N/A"
                auc_p = f"{bm.auc_pr:.3f}" if bm.auc_pr is not None else "N/A"
                prec = f"{bm.precision:.3f}" if bm.precision is not None else "N/A"
                rec = f"{bm.recall:.3f}" if bm.recall is not None else "N/A"
                lines.append(
                    f"| {bm.bucket_name} | {bm.n_samples} | {bm.n_positives} | "
                    f"{auc_r} | {auc_p} | {prec} | {rec} |"
                )

        # Calibration
        if report.calibration_bins:
            lines.append("")
            lines.append("## Calibration (Reliability Diagram Data)")
            lines.append("")
            lines.append(
                "| Bin Range | Mean Predicted | Mean Observed | Count | Gap |"
            )
            lines.append(
                "|-----------|----------------|---------------|-------|-----|"
            )
            for b in report.calibration_bins:
                gap = abs(b.mean_predicted - b.mean_observed)
                lines.append(
                    f"| [{b.bin_lower:.2f}, {b.bin_upper:.2f}) | "
                    f"{b.mean_predicted:.3f} | {b.mean_observed:.3f} | "
                    f"{b.count} | {gap:.3f} |"
                )

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interpretation helpers (for markdown report)
# ---------------------------------------------------------------------------


def _interpret_auc_roc(value: Optional[float]) -> str:
    """Return a human-readable interpretation of AUC-ROC."""
    if value is None:
        return "Insufficient data"
    if value >= 0.85:
        return "Excellent discrimination"
    if value >= 0.75:
        return "Good discrimination"
    if value >= 0.65:
        return "Fair -- room for improvement"
    if value >= 0.55:
        return "Poor -- barely above random"
    return "Near-random -- model provides no useful signal"


def _interpret_auc_pr(
    value: Optional[float], n_positive: int, n_total: int
) -> str:
    """Return a human-readable interpretation of AUC-PR."""
    if value is None:
        return "Insufficient data"
    base_rate = n_positive / max(n_total, 1)
    lift = value / max(base_rate, 1e-9)
    if lift >= 5:
        return f"Strong ({lift:.1f}x lift over base rate {base_rate:.2%})"
    if lift >= 2:
        return f"Moderate ({lift:.1f}x lift over base rate {base_rate:.2%})"
    return f"Weak ({lift:.1f}x lift over base rate {base_rate:.2%})"


def _interpret_mrr(value: Optional[float]) -> str:
    """Return a human-readable interpretation of MRR."""
    if value is None:
        return "No queries with positives"
    if value >= 0.7:
        return "Excellent -- first relevant item is usually #1"
    if value >= 0.5:
        return "Good -- first relevant item is usually in top 2"
    if value >= 0.33:
        return "Fair -- first relevant item is usually in top 3"
    return "Poor -- users must scroll to find relevant items"
