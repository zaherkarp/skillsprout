# ADR-006: Demographic Parity Testing for Bias Auditing

**Status:** Accepted
**Date:** 2026-02-07
**Authors:** SkillSprout Team

## Context

SkillSprout recommends career transitions. The stakes are high: if the system systematically recommends lower-paying or lower-prestige occupations to users from underrepresented groups, it amplifies existing labor market inequities. Even a system that does not use demographic data directly can produce disparate outcomes through correlated features (e.g., current occupation and job zone are correlated with race, gender, and socioeconomic status in the U.S. labor market).

We need an approach to detecting and measuring bias in recommendations. The challenge is that:

1. We may not collect demographic data from users (privacy concerns, legal constraints, user trust).
2. Even when we do, sample sizes per demographic group will be small.
3. Different fairness definitions conflict with each other: satisfying demographic parity generally precludes satisfying equalized odds, and vice versa.
4. The relationship between "equal scores" and "equal outcomes" is not straightforward.

## Decision

We implement **demographic parity testing** as the primary bias audit mechanism, supplemented by **score distribution analysis** and **outcome disparity monitoring** when data permits.

### Demographic Parity Testing

For each protected attribute (when available), we test whether the distribution of recommendation buckets is independent of group membership:

```
P(bucket = READY_NOW | group = A) ~= P(bucket = READY_NOW | group = B)
```

We use a chi-squared test (or Fisher's exact test for small samples) to assess whether observed differences are statistically significant. The null hypothesis is that bucket assignment is independent of group membership.

**What we test:**
- Distribution of READY_NOW / TRAINABLE / LONG_RESKILL across groups.
- Mean match_score across groups.
- Mean gap_severity across groups.
- Number of recommendations per bucket across groups.

**When we test:**
- Nightly batch audit via Celery task.
- On-demand via admin endpoint.
- Before any model promotion (new calibration model must pass bias audit).

**Thresholds:**
- A disparity ratio (min group rate / max group rate) below 0.8 triggers an alert.
- A disparity ratio below 0.6 blocks model promotion.
- These thresholds are configurable and will be adjusted as we learn what is achievable.

### Score Distribution Analysis

Even without demographic data, we analyze score distributions for systematic patterns:

- Do users from certain occupation categories (e.g., service occupations vs. professional occupations) consistently receive lower match scores? This could indicate that the O*NET importance weights or the skill rating scale encode occupational prestige bias.
- Are there skill combinations that reliably produce LONG_RESKILL results? If so, are those combinations correlated with specific demographic patterns in the labor market?

### Outcome Disparity Monitoring

When outcome data (APPLY, INTERVIEW, OFFER) accumulates:

- Compare the success rate (P(positive outcome | recommendation)) across groups.
- A recommendation system that gives equal scores but produces unequal outcomes may be reflecting real-world barriers (hiring bias) rather than system bias. We document this distinction but do not automatically adjust for it.

## Consequences

### Positive

**Detectable bias.** Demographic parity is the simplest and most widely understood fairness metric. It directly answers the question "are different groups getting different recommendation distributions?" without requiring assumptions about base rates or qualification distributions.

**Actionable alerts.** The disparity ratio is a single number that stakeholders can understand without statistical training. "Group A gets READY_NOW recommendations at 60% the rate of Group B" is a clear and actionable finding.

**Model governance.** The requirement that new models must pass bias audit before promotion creates a structural safeguard against deploying a model that increases disparity.

**No demographic data required for basic checks.** Score distribution analysis can identify suspicious patterns (e.g., all users from occupation X get LONG_RESKILL) without any user-level demographic data.

### Negative

**Demographic parity is a limited metric.** It tests for equal rates across groups but does not distinguish between:
- **Justified differences:** If Group A has systematically different skill profiles from Group B (due to historical access to education and training), equal skill-based scoring will produce different bucket distributions. Demographic parity would flag this, but "fixing" it would mean recommending occupations that users are not qualified for.
- **Unjustified differences:** If the system penalizes skill combinations that are common in one group for reasons unrelated to job performance, that is genuine bias that should be corrected.

Demographic parity cannot tell us which situation we are in.

**Equal scores vs. equal outcomes tension.** If we adjust scores to achieve demographic parity (equal bucket distributions), the resulting recommendations may have different success rates across groups. A READY_NOW recommendation for a user from Group A might correspond to a 70% success rate, while the same label for Group B corresponds to a 50% rate. This is the fundamental fairness impossibility result: equal calibration and equal positive rates cannot both be satisfied simultaneously (except in trivial cases).

Our approach is to document this tension explicitly rather than pretending to resolve it. We aim for:
- Equal scores should mean equal likelihood of success (calibration).
- Differences in bucket distributions are flagged and investigated, but not automatically corrected.

**Cannot detect what it cannot see.** If we do not collect demographic data, the demographic parity test cannot run. We rely on proxy analysis (current occupation as a rough demographic proxy) which is crude and potentially misleading.

**Small sample sizes.** With few users per demographic group, statistical tests have low power. A disparity ratio of 0.7 with 20 users per group might not be statistically significant, even though it represents a real problem. We address this by tracking trends over time rather than relying on single-snapshot tests.

### What This Approach Can and Cannot Detect

**Can detect:**
- Systematic differences in bucket assignment rates across groups.
- Score distribution shifts correlated with current occupation category.
- Changes in disparity when a new model is deployed (before/after comparison).
- Occupations that are never recommended to any user (potential coverage gaps).

**Cannot detect:**
- Individual-level unfairness (a specific user getting a worse recommendation than they should).
- Bias in the O*NET data itself (if O*NET's skill importance weights reflect occupational biases, our system inherits them).
- Bias introduced by user self-assessment (if some groups systematically underrate their skills, the system will underestimate their match scores).
- Downstream hiring bias (the recommendation was good, but the employer was biased).
- Intersectional bias (disparities that appear only at the intersection of two or more protected attributes, e.g., Black women).

## Alternatives Considered

### Equalized Odds

Test whether the true positive rate and false positive rate are equal across groups: P(recommended READY_NOW | actually qualified) should be independent of group.

**Rejected as primary metric because:**
- Requires ground truth labels for "actually qualified," which we do not have. We have outcome data (APPLY/INTERVIEW/OFFER) which is a noisy proxy.
- Equalized odds conflicts with calibration: satisfying both simultaneously is provably impossible in most non-trivial cases.
- More complex to explain to stakeholders and more difficult to act on.
- Planned as a supplementary metric when outcome data volume permits.

### Individual Fairness

Ensure that similar users receive similar recommendations, regardless of group membership. "Similar" is defined by a task-specific distance metric.

**Rejected because:**
- Defining the similarity metric is the hard part. If "similar" means "similar skill profiles," then individual fairness reduces to the existing skill-based scoring. If "similar" means "similar career trajectories," we need transition data we do not yet have.
- Individual fairness is theoretically elegant but practically difficult to implement and evaluate.
- No standard tools or libraries for individual fairness in recommendation systems.

### Counterfactual Fairness

Test what the recommendation would have been if the user's demographic attributes were different, holding all other features constant.

**Rejected because:**
- We do not use demographic attributes as model features, so the counterfactual intervention is undefined in the current system.
- Counterfactual fairness requires a causal model of how demographic attributes influence skill ratings, occupation choice, and outcomes. Building this causal model is a research project, not an engineering task.
- Overkill for the current stage of the system.

### No Bias Auditing

Ship without any fairness testing and add it later when the system is more mature.

**Rejected because:**
- Bias in recommendation systems compounds over time: biased recommendations lead to biased feedback data, which trains biased models, which produce more biased recommendations.
- Retrofitting fairness into a system that has been deployed without it is much harder than building it in from the start.
- Even imperfect auditing (demographic parity with small samples) is better than no auditing. It creates organizational awareness and establishes the expectation that fairness is a first-class concern.
