# ADR-001: Two-Stage Scoring Architecture (Deterministic Baseline + Learned Calibration)

**Status:** Accepted
**Date:** 2026-02-07
**Authors:** SkillSprout Team

## Context

SkillSprout needs to score how well a user's skills match target occupations and assign each occupation to a recommendation bucket (READY_NOW, TRAINABLE, LONG_RESKILL). The system must work from day one with zero training data and improve as user feedback accumulates.

We had to choose between:

1. A single end-to-end learned model that maps (user skills, occupation requirements) directly to a recommendation.
2. A two-stage architecture where a deterministic baseline produces interpretable scores and a learned calibration layer adjusts rankings over time.

The problem is harder than a typical recommendation task because the consequences of bad recommendations are high: a user who quits their job based on a "Ready Now" classification that was wrong faces real harm. This means we need to be able to explain and debug every recommendation from day one, not just after we have accumulated enough data to train a model.

## Decision

We use a two-stage architecture:

**Stage 1 (Baseline Scorer):** A deterministic, rule-based model (`BaselineScorer` in `app/ml/scoring.py`) that computes `match_score` and `gap_severity` from O*NET skill importance weights and user self-assessed ratings. It assigns buckets using configurable thresholds (e.g., READY_NOW requires match >= 75 and gap <= 25). This stage works immediately, requires no training data, and produces fully explainable results.

**Stage 2 (Calibration Model):** A logistic regression model (`CalibrationModel` in `app/ml/calibration.py`) that learns from user feedback (apply, interview, offer, hide) to predict P(success) for each (user, occupation) pair. The calibration model takes the baseline scores as input features (along with job zone, skill gap, and user confidence features) and re-ranks recommendations. It activates only after collecting at least 50 labeled feedback samples.

The baseline never goes away. The calibration model adjusts the ranking that the baseline produces; it does not replace the baseline's scoring logic or bucket assignment.

## Consequences

### Positive

**Explainability from day one.** Every recommendation can be explained in terms the user understands: "Your match score is 72% because you rated Programming as Advanced (0.75) and it carries 40% of the importance weight." There is no black-box opacity. The calibration layer adds a probability but does not remove the underlying explanation.

**Cold start is a non-problem.** The baseline scorer needs exactly zero historical data. It works the moment a user rates their skills. Systems that rely on collaborative filtering or learned embeddings cannot produce meaningful results for a new user or a new occupation without extensive warmup data.

**Debugging is tractable.** When a recommendation looks wrong, the debugging path is clear: check the O*NET skill weights, check the user's ratings, verify the threshold configuration. With an end-to-end model, the equivalent debugging process would require interpreting feature attributions from a neural network, which is slow and unreliable at small scale.

**Incremental sophistication.** The calibration layer can grow in complexity without touching the baseline. We can swap logistic regression for gradient boosting (at 1000+ samples), add transition-aware features, or introduce user embeddings. Each upgrade is isolated: if the calibration model degrades, we fall back to the baseline with no user-visible disruption.

**Threshold transparency.** All bucket boundaries are configurable via environment variables. A product decision like "we should be more conservative about READY_NOW" translates directly to changing `READY_NOW_MATCH_THRESHOLD` from 75 to 80. No retraining required.

### Negative

**Maintenance of two systems.** We maintain both the deterministic scoring code and the ML pipeline. The interaction between them (baseline scores as calibration features) creates a coupling that must be tested.

**Baseline ceiling.** The deterministic baseline treats all skills as independent and weights them solely by O*NET importance. It cannot learn that "Programming" and "Systems Analysis" are complementary, or that a user who rates themselves 4/4 on everything is probably over-confident. These patterns require the calibration layer.

**Transition complexity.** Deciding when to activate the calibration model and how to blend its output with the baseline requires careful monitoring. A poorly calibrated model could degrade results relative to the baseline.

## Alternatives Considered

### Single End-to-End Model

Train a neural network or gradient boosted model that takes raw skill vectors as input and predicts bucket assignment or match probability directly.

**Rejected because:**
- Requires hundreds or thousands of labeled examples before producing any output. SkillSprout has zero at launch.
- Explanations would require post-hoc methods (SHAP, LIME) which are computationally expensive and not always faithful to the model's actual decision process.
- A single model failure mode is catastrophic: there is no fallback.
- The O*NET skill importance weights are strong domain knowledge that would need to be re-learned from data.

### Pure Rule-Based System (No ML)

Use only the deterministic baseline with manually tuned thresholds, never introducing a learned component.

**Rejected because:**
- Cannot adapt to feedback patterns (e.g., users with high self-ratings who consistently fail interviews).
- Cannot capture non-linear interactions between features.
- Would require ongoing manual threshold tuning as the user base grows and diversifies.
- Misses the opportunity to learn from outcome data (interviews, offers) that reveals whether recommendations were actually useful.

### Collaborative Filtering

Recommend occupations based on what similar users have successfully transitioned to.

**Rejected because:**
- Requires a large user base with outcome data before producing useful recommendations.
- Suffers from popularity bias (well-known career paths dominate).
- Cannot explain recommendations in terms of specific skills.
- The "similar user" concept is hard to define meaningfully for career transitions.
