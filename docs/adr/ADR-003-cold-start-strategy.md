# ADR-003: Three-Tier Cold Start Strategy

**Status:** Accepted
**Date:** 2026-02-07
**Authors:** SkillSprout Team

## Context

SkillSprout faces three simultaneous cold start problems:

1. **User cold start:** A new user has no profile, no current occupation, and no skill ratings. The system cannot score any occupations until the user provides all three. Currently, the API returns hard 400/404 errors at each missing step.

2. **System cold start:** Before any users exist, the calibration model has zero training data. It needs at least 50 labeled feedback samples (HIDE/APPLY/INTERVIEW/OFFER) to train. Until then, only the deterministic baseline operates.

3. **Occupation cold start:** Occupations not yet cached from O*NET have no skill data. They are silently skipped during scoring. A user searching for an uncommon occupation may see zero results.

The original implementation handled these with error messages: "User has no skill ratings" (400), "No occupations in cache" (503). This is technically correct but produces a terrible user experience. A user who installs the app and immediately hits three consecutive error screens will not return.

## Decision

We adopt a three-tier strategy that provides progressively better recommendations as more data becomes available:

### Tier 1: Immediate Value (Zero User Data)

When a user has set their current occupation but has not yet rated any skills:

- Show a **job zone neighbor list**: occupations in the same or adjacent job zones as the current occupation. These are sorted by category similarity (same SOC major group first) and annotated with "Rate your skills to see personalized match scores."
- Provide **quick-rate templates**: pre-selected skill subsets based on the current occupation's most important skills, with suggested ratings based on the occupation's typical profile. The user can accept, adjust, or dismiss.
- Display **population-level statistics**: "80% of Software Developers who used SkillSprout found at least 3 Ready Now matches." This creates motivation to complete the rating process.

### Tier 2: Baseline Recommendations (User Has Rated Skills)

Once the user has rated at least one skill:

- Run the full `BaselineScorer` pipeline against all cached occupations.
- Show results grouped by bucket (READY_NOW, TRAINABLE, LONG_RESKILL).
- Annotate unrated skills as "Not yet rated -- your match score may change" rather than silently treating them as zero capability.
- Prompt the user to rate additional skills that appear frequently in their top matches: "Rating 'Critical Thinking' would improve the accuracy of 12 of your recommendations."

### Tier 3: Calibrated Recommendations (System Has Feedback Data)

Once the system has accumulated sufficient feedback (50+ labeled samples):

- Train the calibration model and re-rank recommendations by P(success).
- Blend transition graph signals (if available) into the ranking.
- Show exploration items near decision boundaries to continue learning.

The tiers are not mutually exclusive. A user with ratings still benefits from Tier 1's quick-rate prompts for unrated skills. The calibration model uses the baseline scores as features, so Tier 2 never disappears.

## Consequences

### Positive

**No dead ends.** Every user sees something useful, even before rating a single skill. The error-to-value gap is eliminated.

**Progressive engagement.** Each tier creates a natural motivation to provide more data. The user sees their recommendations improve in real time as they rate more skills, which encourages completion.

**Graceful degradation.** If the calibration model fails or is retracted, the system falls back to Tier 2 (baseline) automatically. If the occupation cache is empty, Tier 1 still works with the user's current occupation metadata.

**Honest uncertainty.** Instead of hiding the fact that unrated skills are treated as zero, we surface it: "We assumed you have no experience with Critical Thinking because you haven't rated it yet." This builds trust and motivates users to correct the assumption.

### Negative

**Tier 1 quality is limited.** Job zone neighbors without skill matching are a weak signal. Some users may judge the system harshly based on Tier 1 results and leave before reaching Tier 2.

**Implementation complexity.** Three tiers means three code paths to test and maintain. The quick-rate template logic requires curating occupation-specific defaults.

**Population statistics can mislead.** "80% of users found Ready Now matches" is true but might not apply to a specific user whose background is unusual. We mitigate by not showing stats that imply a specific outcome for the individual user.

## Alternatives Considered

### Ask More Questions Upfront

Force users to rate all skills for their current occupation before showing any recommendations. This ensures Tier 2 quality from the first request.

**Rejected because:**
- O*NET occupations have 10-35 skills. Rating all of them is a significant time investment (5-15 minutes) before the user sees any value.
- Drop-off rates for multi-step onboarding flows are well-documented: every additional step loses 20-40% of users.
- Users who have never thought about their skills in O*NET's taxonomy will give low-quality ratings under time pressure.
- The quick-rate template approach gets to useful results faster by focusing on the highest-importance skills first and filling in others later.

### Collaborative Filtering for Cold Start

For new users, recommend what similar users (based on current occupation) found useful. This uses the system's accumulated data to bootstrap recommendations.

**Rejected because:**
- Requires a substantial user base (hundreds of users per occupation) to produce meaningful "similar user" signals.
- At launch, there are zero other users. Collaborative filtering does not solve the system cold start, it just shifts it.
- When it does work, it tends to recommend popular paths (popularity bias), which may not be the best fit for an individual user.
- Cannot explain recommendations in terms of the user's specific skills, only in terms of "users like you."
- Planned as a future enhancement for Tier 3 once we have sufficient user volume, but not as the primary cold start strategy.

### Show Everything

Display all occupations in the cache, sorted alphabetically or by SOC code, and let the user browse.

**Rejected because:**
- With 1,000+ occupations, this is overwhelming and provides no personalization signal.
- Users cannot meaningfully evaluate occupations without some scoring context.
- Browsing fatigue leads to abandonment.
- Does not leverage the one piece of data we do have (current occupation) to provide even basic relevance.

### Impute Ratings from Current Occupation

If a user is a Software Developer, assume they have the typical skill profile of a Software Developer (based on O*NET importance/level data) and score against that.

**Rejected as the default approach because:**
- The imputed profile may be wildly inaccurate for an individual. A Software Developer who is actually a junior frontend developer has a very different skill profile from a senior systems architect, even though O*NET classifies them identically.
- Users who see recommendations based on assumed skills they do not have will lose trust immediately.
- However, we do use this idea in a limited way: the quick-rate templates suggest typical ratings as starting points that the user can adjust. The key difference is that the user explicitly confirms or changes each rating, rather than having it silently assumed.
