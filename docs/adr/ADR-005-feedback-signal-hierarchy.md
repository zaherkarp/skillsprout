# ADR-005: Feedback Signal Hierarchy (Outcome > Application > Save > Dwell > View)

**Status:** Accepted
**Date:** 2026-02-07
**Authors:** SkillSprout Team

## Context

SkillSprout collects six types of user feedback on recommendations: CLICK, SAVE, HIDE, APPLY, INTERVIEW, and OFFER. These signals vary dramatically in reliability, latency, and volume:

| Signal    | Reliability | Latency         | Expected Volume |
|-----------|-------------|-----------------|-----------------|
| CLICK     | Very low    | Immediate       | High            |
| SAVE      | Low         | Immediate       | Medium          |
| HIDE      | Moderate    | Immediate       | Medium          |
| APPLY     | Moderate    | Hours to days   | Low             |
| INTERVIEW | High        | Days to weeks   | Very low        |
| OFFER     | Very high   | Weeks to months | Extremely low   |

The question is: which signals should the calibration model learn from, and how should they be weighted?

The naive approach of treating all signals equally would let high-volume, low-reliability signals (clicks) dominate training. The ideal approach of using only OFFER signals would produce a model that can never train because offers are too rare.

## Decision

We adopt a signal hierarchy that determines which feedback types are used as training labels and how they are treated:

### Tier 1: Training Labels (Used for Model Training)

| Signal    | Label | Rationale |
|-----------|-------|-----------|
| OFFER     | 1 (positive) | Ground truth: the transition succeeded. |
| INTERVIEW | 1 (positive) | Strong signal: the user's skills were credible enough to get an interview. |
| APPLY     | 1 (positive) | Moderate signal: the user believed the recommendation was realistic enough to act on. |
| HIDE      | 0 (negative) | Clear negative: the user explicitly rejected this recommendation. |

### Tier 2: Monitoring Signals (Not Used for Training)

| Signal    | Use | Rationale |
|-----------|-----|-----------|
| SAVE      | Engagement metric, exploration trigger | Ambiguous intent: could mean "interested but not now" or "bookmarking for comparison." |
| CLICK     | UI analytics, dwell time proxy | Too noisy: position bias, curiosity clicks, accidental clicks. |

### Tier 3: Implicit Signals (Future)

| Signal    | Use | Rationale |
|-----------|-----|-----------|
| Dwell time | Future calibration feature | How long a user spent reading an occupation's details. Requires frontend instrumentation. |
| Non-interaction | Future negative signal | Occupations shown but never clicked. Requires careful handling of position bias. |

### Label Construction

For training the calibration model, we use **binary labels** rather than graded relevance:

- Label = 1: Any of {APPLY, INTERVIEW, OFFER}
- Label = 0: HIDE

We do **not** weight OFFER more heavily than APPLY in the current implementation. The `class_weight="balanced"` setting in logistic regression handles the class imbalance between positive and negative labels.

### Future: Composite Labels

When data volume permits (1,000+ labeled samples), we plan to construct composite labels that incorporate signal strength:

```
composite_label = (
    0.3 * applied +
    0.5 * interviewed +
    1.0 * offered
) / normalization_factor
```

This requires switching from classification (logistic regression) to regression or ordinal regression, which is planned for the gradient boosting phase (ADR-004).

## Consequences

### Positive

**Noise reduction.** By excluding CLICK and SAVE from training labels, we avoid the dominant source of noise. Clicks are heavily influenced by position (items shown first get more clicks regardless of quality) and by curiosity (clicking to learn about an unfamiliar occupation does not mean it is a good recommendation).

**Actionable signals only.** Every signal used for training corresponds to a meaningful user action: they applied, they got an interview, they got an offer, or they explicitly rejected the recommendation. There is no ambiguity about what these signals mean.

**SAVE and CLICK remain useful.** These signals are not discarded; they inform engagement metrics and exploration decisions. A recommendation that is frequently saved but never applied to might indicate an aspirational but unrealistic path -- useful for product insights even if not for model training.

**Clear upgrade path.** The hierarchy is designed to evolve. As we collect dwell time and non-interaction data, they slot into the existing framework without restructuring the training pipeline.

### Negative

**Low training volume.** Only 4 of 6 action types produce training labels, and the positive signals (APPLY, INTERVIEW, OFFER) are the rarest. At early stages, the model may have fewer than 50 usable labels for weeks after launch.

**Latency mismatch.** OFFER signals arrive weeks or months after the recommendation was generated. By the time we learn from an OFFER, the user's skill profile may have changed. We mitigate this by using the features recorded at recommendation time (stored in `score_json`), not the user's current features.

**APPLY is a weak positive.** A user who applies to a job but never gets an interview may have been over-optimistic about their fit. Treating APPLY as positive introduces some label noise. However, the alternative (waiting for INTERVIEW) would make positive labels too rare to train on.

**HIDE is an imperfect negative.** A user might hide a recommendation because they already know about that occupation, not because the recommendation was bad. This is an irreducible source of label noise that we accept.

## Alternatives Considered

### Use All Signals Equally

Treat every action type as a training signal: CLICK and SAVE as weak positive, HIDE as negative, APPLY/INTERVIEW/OFFER as strong positive.

**Rejected because:**
- Click and save volume would dominate training, and these signals are the noisiest.
- Position bias in clicks would teach the model to replicate whatever order the baseline already produces, creating a feedback loop that prevents improvement.
- Graded relevance labels (weak/strong positive) require more sophisticated loss functions than binary cross-entropy, adding complexity without proportionate benefit at low data volumes.

### Outcome Signals Only (INTERVIEW + OFFER)

Use only the highest-reliability signals for training, ignoring APPLY and HIDE.

**Rejected because:**
- At launch, we might wait months before accumulating 50 INTERVIEW/OFFER signals.
- No negative signal: without HIDE, the model sees only positive examples and cannot learn what makes a bad recommendation.
- The 50-sample training threshold would rarely be met in the first year of operation.

### Implicit Negative from Non-Interaction

Treat recommendations that were shown but never interacted with as implicit negatives.

**Rejected for now because:**
- A user might not interact with a recommendation because they did not scroll to it, not because they rejected it.
- Position bias strongly confounds non-interaction: items at the bottom of the list are not interacted with regardless of quality.
- Requires session-level tracking to know which recommendations were actually viewed (viewport tracking), which we have not implemented.
- Planned as a future Tier 3 signal when we add frontend instrumentation for viewport visibility.

### Weighted Binary Labels

Assign different weights to different positive signals: OFFER weight 3.0, INTERVIEW weight 2.0, APPLY weight 1.0.

**Rejected because:**
- Sample weighting in logistic regression changes the effective class balance in non-obvious ways, especially when combined with `class_weight="balanced"`.
- The weight ratios (3:2:1) are arbitrary and would need to be tuned on held-out data that we do not have.
- At small sample sizes, a single heavily-weighted OFFER signal could distort the model disproportionately.
- This is planned for the composite label phase when we have enough data to validate the weight ratios.
