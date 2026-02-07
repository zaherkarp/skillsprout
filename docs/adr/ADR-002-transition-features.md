# ADR-002: Transition-Aware Features for Calibration Model

**Status:** Accepted
**Date:** 2026-02-07
**Authors:** SkillSprout Team

## Context

The baseline scorer (`BaselineScorer`) evaluates each target occupation independently: it computes how well a user's skills match the target's requirements. But career transitions are not symmetric. Moving from Software Developer to Web Developer is a fundamentally different proposition than moving from Web Developer to Software Developer, even if the raw skill overlap is identical.

The calibration model (Model v2) needs features that capture the direction, difficulty, and historical success rate of specific transitions. Without these, the model treats "occupation A as a target" identically regardless of where the user is coming from.

This matters because:
- A Nurse moving into Healthcare Administration has institutional context that makes the transition easier than the skill overlap alone suggests.
- A Retail Manager moving into Software Development faces a steeper climb than a Systems Administrator making the same move, even if both have comparable skill gaps.
- Some transitions are well-traveled paths with established training pipelines; others are novel and risky.

## Decision

We add transition-aware features to the calibration model's feature set, derived from a directed graph of observed user transitions (see `ml/transition_graph/`). These features supplement the existing baseline features (match_score, gap_severity) and user confidence features (mean_rating, rating_variance).

The transition-aware features are:

1. **`transition_frequency`** -- How many users have attempted this specific origin-to-target transition. High frequency indicates a well-known path.

2. **`transition_success_rate`** -- Among users who attempted this transition, what fraction received positive outcomes (apply, interview, offer). This captures whether the path actually works, not just whether the skills overlap.

3. **`median_skill_overlap`** -- The median match_score for this transition across all users who attempted it. This normalizes for the fact that different user populations self-rate differently.

4. **`path_novelty`** -- Binary flag: 1 if this transition has never been observed in the data, 0 otherwise. Novel paths should be treated with more uncertainty.

5. **`job_zone_diff`** (already exists) -- The difference between target and current job zones. We retain this as it captures education/experience level changes.

These features are **additive** to the skill-based scoring. They are used only by the calibration model; the baseline scorer remains unchanged. The transition graph is rebuilt nightly via a Celery task.

## Consequences

### Positive

**Captures asymmetry.** The feature set now reflects that Software Developer -> Web Developer is not the same as Web Developer -> Software Developer. The calibration model can learn different success probabilities for each direction.

**Leverages collective intelligence.** If 200 users have successfully transitioned from Retail Manager to Sales Engineer, that signal improves recommendations for the next Retail Manager, even if their individual skill profile is mediocre.

**Identifies emerging paths.** By tracking transition frequency over time, we can detect new career paths that are gaining popularity (e.g., Data Analyst -> ML Engineer) before they become well-established.

**Uncertainty for novel transitions.** The `path_novelty` flag lets the model express lower confidence for transitions nobody has tried before, rather than confidently scoring them based solely on skill overlap.

### Negative

**Data sparsity.** With O*NET's ~1,000 occupations, the transition matrix has ~1M cells. Most will be empty for a long time. The graph is useful only for transitions that have been observed, and the features degrade gracefully to zero/null for unobserved transitions (the baseline still works).

**Computational cost.** Building the transition graph requires aggregating all historical feedback. The nightly Celery task makes this manageable, but the graph must be loaded into memory for scoring. With adjacency list representation and NetworkX, a graph of 1,000 nodes and 10,000 edges fits comfortably in memory (<10MB).

**Overfitting risk.** Transition-level features can overfit to small samples. If only 3 users have attempted a specific transition and all 3 succeeded, the success rate is 100% -- but that is not statistically meaningful. We mitigate this by:
  - Requiring a minimum of 5 observations before computing transition-level statistics (below that, features default to population averages).
  - Using the features as calibration inputs rather than hard filters, so the logistic regression can learn appropriate weights.
  - Regularization in the logistic regression (default L2 penalty) prevents any single feature from dominating.

**Feedback loop risk.** If the model boosts well-traveled paths, more users take those paths, generating more positive data, further boosting those paths. We address this partially through the exploration policy (epsilon-greedy), which occasionally recommends less-traveled transitions.

## Alternatives Considered

### Skill Overlap Only (No Transition Context)

Use only the baseline match_score and gap_severity without any information about the specific origin-target pair.

**Rejected because:**
- Treats all transitions as symmetric (origin does not matter).
- Cannot learn from historical success patterns.
- Misses the strong signal that "this specific transition has worked for many people before."

### Full Transition Matrix as Feature

Encode the entire transition history as a dense feature (e.g., frequency for every possible origin-target pair).

**Rejected because:**
- The feature space would be O(N^2) where N is the number of occupations (~1M features).
- Extreme sparsity would cause regularization to zero out nearly everything.
- Does not generalize: each transition is treated as completely independent.

### Occupation Embeddings

Learn dense vector representations of occupations from transition patterns and compute similarity in embedding space.

**Rejected because:**
- Requires substantial transition volume to learn meaningful embeddings.
- Adds significant model complexity (embedding layer, training procedure).
- Planned as a Model v3 enhancement when we have sufficient data (5,000+ transitions).
- For now, the simpler aggregate features (frequency, success rate) capture the most important signals without the infrastructure cost.

### SOC Code Hierarchy Features

Use the O*NET SOC code hierarchy (e.g., 15-12xx.xx shares a family with 15-13xx.xx) as a proxy for occupation similarity.

**Rejected as sole approach because:**
- SOC codes group occupations by administrative classification, not by skill transferability.
- A Data Scientist (15-2051.00) may have more skill overlap with a Statistician (15-2041.00) than with a Computer Programmer (15-1251.00), but they share a SOC family with the programmer.
- However, we may use SOC hierarchy as a fallback for smoothing transition statistics when direct observations are sparse. This is a future enhancement.
