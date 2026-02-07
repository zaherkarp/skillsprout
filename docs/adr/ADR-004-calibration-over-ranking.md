# ADR-004: Logistic Regression Calibration Over Learning-to-Rank

**Status:** Accepted
**Date:** 2026-02-07
**Authors:** SkillSprout Team

## Context

The calibration model (Model v2) needs to improve recommendation ranking beyond what the deterministic baseline provides. Two broad approaches exist:

1. **Pointwise calibration:** Train a model that predicts P(success | user, occupation) for each (user, occupation) pair independently, then rank by predicted probability. This is what logistic regression does.

2. **Learning-to-rank (LTR):** Train a model that directly optimizes the ranking of a list of items for a given user. LTR models (LambdaMART, ListNet, RankNet) are designed to maximize metrics like NDCG or MAP rather than pointwise accuracy.

LTR is the standard approach in mature recommendation systems (search engines, e-commerce, content feeds). But it has data requirements and infrastructure costs that matter for SkillSprout's current stage.

## Decision

We use logistic regression as a pointwise calibration model now, with a planned transition to learning-to-rank when data volume and infrastructure justify it.

The calibration model (`CalibrationModel` in `app/ml/calibration.py`) trains on individual (user, occupation, outcome) triples. Each triple is labeled positive (APPLY/INTERVIEW/OFFER) or negative (HIDE). The model outputs a probability that is used to re-rank within buckets.

### Why Logistic Regression Specifically

- **Probability calibration:** Logistic regression outputs well-calibrated probabilities by design. This matters because we use the predicted probability to set user expectations ("Strong match" vs. "Moderate match"), not just to rank items.

- **Sample efficiency:** Logistic regression produces meaningful results with 50-200 samples. LTR models typically need 1,000-10,000 query-document pairs with relevance labels.

- **Interpretability:** The model coefficients directly tell us which features matter. If `gap_severity` has a coefficient of -0.8, we know that higher gap severity strongly predicts failure. This interpretability is critical for debugging and for building trust with stakeholders.

- **Training speed:** Training takes milliseconds. We can retrain daily or even on-demand without scheduling concerns.

- **Scikit-learn maturity:** The implementation uses `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"` to handle the expected class imbalance (more HIDE signals than OFFER signals). No custom training infrastructure needed.

### Threshold for Switching to LTR

We plan to transition to a learning-to-rank approach when **all** of the following conditions are met:

1. **Data volume:** At least 5,000 labeled feedback events across at least 500 unique users.
2. **Pairwise data collection:** We have implemented a mechanism to collect explicit or implicit pairwise preferences (e.g., user clicked occupation A but not B within the same session).
3. **Evaluation infrastructure:** We have offline evaluation with held-out test sets and online A/B testing capability.
4. **Diminishing returns from calibration:** The logistic regression ROC-AUC has plateaued and feature engineering is no longer improving ranking quality.

## Consequences

### Positive

**Works now.** The model can train and produce useful re-rankings with the data we have today (or will have within weeks of launch).

**Stable predictions.** Logistic regression does not overfit dramatically on small datasets, especially with L2 regularization (the sklearn default). LTR models with complex architectures can produce erratic rankings when trained on insufficient data.

**Easy to validate.** We can compare the calibrated ranking against the baseline ranking using simple metrics (ROC-AUC, accuracy) without needing LTR-specific evaluation frameworks.

**Clear upgrade path.** The feature extraction pipeline, training data collection, and model registry infrastructure built for logistic regression will directly support LTR. The switch will be a model swap, not an architecture rewrite.

### Negative

**Suboptimal ranking.** Pointwise models do not directly optimize for ranking quality. A calibration model might assign P(success) = 0.7 to occupation A and P(success) = 0.68 to occupation B, ranking A first, even though B is a better recommendation for this specific user given the full context. LTR models can capture these relative preferences.

**No list-level context.** The calibration model scores each occupation independently. It cannot learn patterns like "showing too many LONG_RESKILL occupations discourages users" or "diversity in the recommendation list improves engagement." These are inherently list-level concerns that pointwise models cannot capture.

**Pairwise data not collected.** The current feedback system captures absolute signals (click, save, hide, apply) but not pairwise preferences (user preferred A over B). Collecting pairwise data requires UI changes (e.g., "Which of these two occupations interests you more?") or implicit inference from click patterns. This data is a prerequisite for the LTR transition.

## Alternatives Considered

### Learning-to-Rank from Day One

Implement LambdaMART or a neural LTR model as the initial calibration approach.

**Rejected because:**

- **Data requirements.** LTR models need pairwise or listwise training data. We currently collect only pointwise feedback (individual actions on individual occupations). Converting this to pairwise data requires assumptions about implicit negative signals (non-clicked items as negative) that are unreliable at small scale.
- **Evaluation complexity.** LTR evaluation requires NDCG, MAP, or other ranking metrics computed over ranked lists. These metrics are meaningful only with sufficient test queries (users). With 50 users, evaluation variance overwhelms the signal.
- **Infrastructure overhead.** LTR frameworks (LightGBM ranker, TensorFlow Ranking, XGBoost ranker) require more complex training pipelines, hyperparameter tuning, and serving infrastructure than a single logistic regression model.
- **Debugging difficulty.** When a ranking looks wrong, debugging an LTR model requires understanding the pairwise loss function and how individual features contributed to relative orderings. With logistic regression, we can inspect coefficients and trace each score directly.

### Gradient Boosted Trees (Pointwise)

Use XGBoost or LightGBM as a pointwise regression model instead of logistic regression.

**Rejected for now, planned as intermediate step because:**

- GBMs can capture non-linear feature interactions that logistic regression misses (e.g., the interaction between `job_zone_diff` and `gap_severity` may be non-linear).
- But GBMs need more data to avoid overfitting: typically 500-1,000+ samples minimum.
- GBMs are less naturally calibrated than logistic regression. Predicted probabilities from GBMs require post-hoc calibration (Platt scaling, isotonic regression).
- The plan is: logistic regression now (50-500 samples) -> gradient boosted trees (500-5,000 samples) -> LTR (5,000+ samples).

### Multi-Armed Bandits

Treat each occupation as an arm and use Thompson sampling or UCB to balance exploration and exploitation.

**Rejected as the primary approach because:**

- MAB models do not use features. They learn per-arm statistics, which means each occupation must be explored independently.
- With 1,000 occupations, convergence is extremely slow.
- However, we use an epsilon-greedy exploration policy (`ExplorationPolicy` class) on top of the calibration model to partially address exploration. A contextual bandit approach (which uses features) is planned for future work.

### No Calibration (Baseline Only)

Keep only the deterministic baseline and invest in better threshold tuning instead of building an ML pipeline.

**Rejected because:**

- The baseline cannot learn from outcomes. If we discover that users with high self-ratings consistently fail interviews, the baseline has no mechanism to adjust.
- Manual threshold tuning does not scale: different user populations and occupation domains may need different thresholds, and we cannot tune them individually without a learning system.
- The calibration infrastructure is already built (model registry, training task, feature extraction). The marginal cost of maintaining it is low.
