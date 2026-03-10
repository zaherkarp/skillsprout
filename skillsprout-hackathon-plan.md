# SkillSprout Hackathon Plan: Claude Code Prompts

## Team Roles

| Engineer | Focus Area | Codename |
|----------|-----------|----------|
| **E1** | Backend / ML Pipeline | `ml-eng` |
| **E2** | Frontend / UX / Accessibility | `ux-eng` |
| **E3** | Infrastructure / Data Integrity / DevOps | `infra-eng` |

## How to Use This Document

Each prompt below is designed to be pasted directly into **Claude Code** (Anthropic's CLI agent). Prompts are sequenced so each day builds on the prior day's work. Engineers should coordinate at the **daily sync points** noted at the end of each day.

Prompts use `>>>` to denote the start of each Claude Code prompt. Everything between `>>>` markers is one prompt.

---

# Day 1 (Monday): Foundation & Audit

**Theme:** Understand what exists, identify structural problems, establish patterns for the week.

---

## E1 — ML Pipeline Audit & Eval Framework

### Prompt 1.1: Codebase Audit & Scoring Transparency Map

```
>>> Read the entire SkillSprout codebase. Produce a structured audit document
(markdown file: docs/ml-audit.md) that covers:

1. SCORING PIPELINE MAP: Trace the full path from user input to bucket
   assignment (READY_NOW, TRAINABLE, LONG_RESKILL). For every function in
   the chain, document:
   - Input shape and types
   - Transformation logic
   - Output shape and types
   - Hardcoded thresholds or magic numbers
   - Where O*NET data enters the pipeline

2. FEEDBACK LOOP INVENTORY: List every event type currently tracked
   (clicks, saves, applications, outcomes). For each, document:
   - Where it's captured (endpoint, model, table)
   - Current latency from user action to database write
   - Whether it's currently used by any model (v1 or v2)
   - Signal quality assessment (noise level, sparsity, delay)

3. COLD START ANALYSIS: Identify every place where missing data causes
   degraded behavior:
   - New users with no history
   - Rare occupations with few interactions
   - Unseen skill combinations
   - Document the current fallback behavior for each case

4. TECHNICAL DEBT: Flag any antipatterns, untested paths, or fragile
   coupling between the deterministic scorer and calibration layer.

Be thorough. This is the reference document for the rest of the week.
```

### Prompt 1.2: Offline Evaluation Pipeline

```
>>> Build an offline evaluation framework for SkillSprout's scoring models.
Create this in a new directory: ml/evaluation/

Requirements:
- eval_framework.py: Core evaluation harness that:
  * Loads historical interaction data from PostgreSQL (or generates
    synthetic data if insufficient volume exists — create a
    generate_synthetic_interactions.py helper for this)
  * Defines proxy labels:
    - POSITIVE: user saved occupation AND (returned to view within 7 days
      OR clicked external apply link)
    - NEGATIVE: user saw recommendation but took no action within 14 days
  * Splits data respecting temporal ordering (no future leakage):
    train = first 70% by timestamp, val = next 15%, test = final 15%
  * Computes per-model metrics:
    - AUC-ROC and AUC-PR (overall and per-bucket)
    - Calibration plot data (predicted probability vs observed frequency,
      10 bins)
    - Per-bucket precision and recall
    - Mean Reciprocal Rank for recommendation ordering within buckets

- eval_runner.py: CLI that runs eval for v1 baseline, v2 calibration, or
  both side-by-side. Output is a JSON report + markdown summary.

- tests/test_eval_framework.py: Test the evaluation pipeline itself —
  use synthetic data to verify metric computation is correct.

Annotate all code heavily explaining WHY each metric matters and what
actionable signal it provides. Include docstrings that explain what
"good" vs "bad" looks like for each metric in the SkillSprout context.
```

---

## E2 — UX Audit & Skills Translator Prototype

### Prompt 2.1: UX & Accessibility Audit

```
>>> Audit the SkillSprout frontend (and API responses that drive the UI)
for UX and accessibility issues. Create docs/ux-audit.md covering:

1. USER JOURNEY MAP: Document the current flow from first visit to
   receiving recommendations. For each step, note:
   - What the user must know/do to proceed
   - Assumed vocabulary level (does it require knowing O*NET terms?)
   - Where users without professional resume language would get stuck
   - Time-to-value: how many steps before they see useful output

2. ACCESSIBILITY GAPS:
   - Does the app work without JavaScript (progressive enhancement)?
   - Keyboard navigation completeness
   - Screen reader compatibility (ARIA labels, semantic HTML)
   - Color contrast and colorblind safety for any data visualization
   - Mobile / small screen usability
   - Low bandwidth / shared device considerations (session resumption,
     minimal data transfer)

3. INPUT BARRIER ANALYSIS: How does a user currently specify their
   skills? Document:
   - The exact input mechanism
   - What vocabulary it assumes
   - How someone who managed a church food bank for 10 years (with real
     project management and logistics skills) would fare
   - What's lost when non-traditional experience can't map to the
     expected input format

4. OUTPUT CLARITY: When a user sees their buckets, can they understand:
   - WHY each occupation is in that bucket?
   - WHAT specific skills they have vs. lack?
   - WHAT to do about the gaps?
   - Whether the recommendation is trustworthy?

Prioritize findings as P0 (blocks core usage), P1 (degrades experience
significantly), P2 (improvement opportunity).
```

### Prompt 2.2: Skills Translator — NLP Input Mode

```
>>> Build a "skills translator" feature that lets users describe their
work experience in plain language instead of selecting from O*NET skill
categories. Create this in: app/features/skills_translator/

Architecture:
- skills_translator.py: Core translation engine
  * Input: Free-text description of work experience (e.g., "I manage
    the schedule for 15 people, handle customer complaints, resolve
    billing disputes, and train new hires every quarter")
  * Processing: Use keyword extraction + semantic matching to map
    natural language to O*NET skill codes. Implement TWO approaches:
    1. Rule-based: Regex + keyword dictionary mapping common phrases
       to O*NET skills (e.g., "manage schedule" → "Scheduling",
       "Customer complaints" → "Customer Service", "train new hires"
       → "Training and Teaching Others")
    2. TF-IDF similarity: Compare input text against O*NET skill
       descriptions using cosine similarity. Return top-k matches
       above a threshold.
  * Output: List of matched skills with confidence scores, grouped as:
    - HIGH_CONFIDENCE (>0.8): Auto-selected
    - MEDIUM_CONFIDENCE (0.5-0.8): Suggested for user confirmation
    - LOW_CONFIDENCE (0.3-0.5): Shown as "did you mean...?"

- skill_dictionary.py: Curated mapping of ~200 common plain-language
  phrases to O*NET skills, covering:
  * Caregiving and volunteer work
  * Retail and food service
  * Trades and manual labor
  * Administrative and office work
  * Military service (MOS code common tasks)
  * Gig economy work

- API endpoint: POST /api/v1/skills/translate
  * Accepts: { "description": "free text", "confirm": ["skill_ids"] }
  * Returns: { "matched_skills": [...], "needs_confirmation": [...] }

- tests/test_skills_translator.py: Test with at least 10 diverse
  personas including:
  * Stay-at-home parent returning to work
  * Retail manager with 15 years experience
  * Military veteran (infantry → civilian)
  * Church volunteer coordinator
  * Gig worker (Uber + TaskRabbit + freelance)

Annotate the code heavily explaining the matching logic, why thresholds
are set where they are, and known limitations of each approach.
```

---

## E3 — Infrastructure Audit & Privacy Framework

### Prompt 3.1: Infrastructure & Security Audit

```
>>> Perform a comprehensive infrastructure and security audit of the
SkillSprout codebase. Create docs/infra-audit.md covering:

1. DATABASE ARCHITECTURE REVIEW:
   - Map all 10+ domain entities and their relationships
   - Identify missing indexes (query the current schema and flag any
     foreign key or frequently-filtered column without an index)
   - Assess migration history (Alembic) for reversibility
   - Check for N+1 query patterns in SQLAlchemy async relationships
   - Evaluate connection pooling configuration

2. CACHING LAYER (Redis):
   - What's currently cached and what should be?
   - Cache invalidation strategy (or lack thereof)
   - O*NET data freshness — how often is it refreshed?

3. ASYNC ARCHITECTURE:
   - Are there any sync-in-async violations (blocking calls inside
     async endpoints)?
   - Celery task configuration: retry policies, dead letter handling,
     task timeouts
   - Connection lifecycle management (are DB/Redis connections properly
     scoped?)

4. SECURITY REVIEW:
   - Authentication/authorization mechanism (or absence thereof)
   - SQL injection surface (raw queries, string interpolation)
   - Input validation coverage (are all endpoints using Pydantic?)
   - Secrets management (hardcoded credentials, env var handling)
   - CORS configuration
   - Rate limiting

5. DEPLOYMENT & CI/CD:
   - Docker configuration review (image size, multi-stage builds,
     non-root user)
   - GitHub Actions workflow completeness
   - Environment parity (dev/test/prod differences)
   - Health check and readiness probe configuration

6. DATA SENSITIVITY ASSESSMENT:
   - Classify every stored data field by sensitivity level
   - Career exploration data is HIGHLY sensitive (employer discovery
     risk) — flag everywhere this data is stored, logged, or
     transmitted
   - Identify any PII leakage in logs or error messages

Prioritize findings as CRITICAL (security risk or data loss),
HIGH (production reliability), MEDIUM (operational), LOW (cleanup).
```

### Prompt 3.2: Privacy-First Data Architecture

```
>>> Design and implement a privacy framework for SkillSprout. Career
exploration data is extraordinarily sensitive — if an employer
discovered an employee was exploring transitions, the consequences
could be severe. Create: app/core/privacy/

1. data_classification.py:
   - Define sensitivity tiers for all data types:
     * TIER_1_PUBLIC: O*NET occupation data, skill taxonomies
     * TIER_2_PSEUDONYMOUS: Aggregated usage patterns, model training
       features (no individual linkage)
     * TIER_3_PERSONAL: User skill profiles, saved occupations,
       search history
     * TIER_4_SENSITIVE: Application tracking, outcome data,
       transition intent signals
   - Decorator @data_tier(tier) for marking model fields and endpoints

2. retention_policy.py:
   - Implement time-based data retention rules:
     * Event-level tracking data: auto-purge after 90 days, retain
       only aggregated features for model training
     * User profiles: retained while active, full deletion within
       72 hours of request
     * Model training snapshots: retain de-identified feature vectors
       only, purge raw data
   - Celery task for nightly retention enforcement
   - Audit log for all deletions

3. private_mode.py:
   - Implement a "Private Mode" middleware that:
     * Returns recommendations WITHOUT any server-side logging
     * Uses session-only storage (no database writes for user actions)
     * Clearly communicates to the user what IS and ISN'T stored
   - API: Header-based toggle (X-Private-Mode: true)

4. data_export.py:
   - GDPR/CCPA-style data export: GET /api/v1/user/data-export
   - Returns all stored data about the user in structured JSON
   - Includes data lineage (where each piece of data came from)

5. data_deletion.py:
   - Full account deletion: DELETE /api/v1/user/data
   - Cascading deletion across all tables
   - Verification that deletion is complete (post-deletion audit query)

6. tests/test_privacy.py: Test all of the above, especially:
   - Private mode produces zero database writes
   - Deletion is truly cascading (no orphaned records)
   - Retention policy correctly purges old data
   - Data export includes everything stored

Include detailed comments explaining the RATIONALE for each privacy
decision — these comments serve as documentation for compliance reviews.
```

---

### Day 1 Sync Point
> **End of day coordination:** E1 shares ml-audit.md (scoring pipeline map), E2 shares ux-audit.md (journey map + accessibility gaps), E3 shares infra-audit.md (security + data sensitivity findings). All three review each other's documents. This creates shared context for the rest of the week.

---

# Day 2 (Tuesday): Core Model Improvements

**Theme:** Fix the scoring engine, add transition-aware features, build the skill transparency layer.

---

## E1 — Transition-Aware Scoring & Cold Start Handling

### Prompt 1.3: Transition-Aware Feature Engineering

```
>>> Enhance SkillSprout's scoring pipeline with transition-aware features.
The current model treats skill gaps as context-free, but the DIRECTION
of a career transition matters enormously. Going from nursing to health
informatics is very different from retail to health informatics, even
if the raw skill gaps look similar.

Create: ml/features/transition_features.py

Implement these feature families (each function should return a dict
of named features):

1. ORIGIN-AWARE FEATURES:
   - skill_direction_vector(): For each O*NET skill domain (cognitive,
     technical, social, physical), compute the signed delta between
     origin and target occupation. A nurse moving to data science has
     a very different cognitive↑/social↓ profile than a retail worker
     making the same move.
   - experience_transfer_ratio(): What fraction of the origin
     occupation's top-10 skills are relevant (importance > 0) in the
     target? This captures "how much of my experience carries over"
     vs raw gap counting.

2. LABOR MARKET FEATURES:
   - occupation_demand_signal(): Stub that returns a demand score for
     the target occupation. For now, use O*NET's "Bright Outlook" flag
     as a proxy. Document how this should eventually connect to BLS
     data or Indeed/LinkedIn job posting counts.
   - salary_delta(): Compute estimated salary change from origin to
     target using O*NET wage data. Include both absolute and
     percentage change. Users care about this.

3. TRANSITION DIFFICULTY FEATURES:
   - credential_barrier(): Flag whether the target occupation requires
     specific credentials (licenses, certifications, degrees) that
     represent hard gates vs. soft skill gaps. Use O*NET's education
     and certification data.
   - industry_distance(): How far apart are the origin and target
     industries? Use O*NET industry codes to compute a simple
     taxonomy distance (same 2-digit NAICS = 0, same 1-digit = 1,
     different = 2).

4. INTEGRATION:
   - Modify the scoring pipeline to include these features in the
     calibration layer's feature vector.
   - Add a FeatureVector dataclass that names every feature and
     includes a human-readable explanation string for each.
   - Ensure all features gracefully handle missing O*NET data
     (return None, not crash).

Tests: test each feature family with at least 3 origin→target pairs
that illustrate meaningfully different scores.

IMPORTANT: Annotate every feature with a comment block explaining:
- What signal this captures
- Why it matters for career transitions specifically
- Known limitations or biases
- Expected value range and interpretation
```

### Prompt 1.4: Cold Start Strategy Implementation

```
>>> Implement a comprehensive cold start strategy for SkillSprout.
Create: ml/cold_start/

The cold start problem exists at three levels. Handle all three:

1. cold_start_user.py — NEW USERS (no interaction history):
   Strategy: Occupation-level priors
   - When a user specifies their current occupation but has no click/
     save/apply history, use POPULATION-LEVEL transition patterns as
     priors.
   - Implement OccupationPriorModel:
     * Maintains a lookup: {origin_occupation: {target_occupation:
       prior_score}} built from aggregated user interaction data
     * Prior score = smoothed ratio of (users from this origin who
       engaged with this target) / (total users from this origin)
     * Use Laplace smoothing (add-1) to handle zero counts
     * Falls back to uniform prior when origin occupation has < 10
       total users in the system
   - Priors are BLENDED with the deterministic scorer, not replacing
     it. Weight: prior_weight = min(0.3, 0.3 * (1 - user_interactions / 20))
     So priors fade as the user builds history.

2. cold_start_occupation.py — RARE OCCUPATIONS (few interactions):
   Strategy: Cluster-level calibration
   - Group occupations by skill profile similarity using k-means on
     O*NET skill vectors (k=50 as a starting point)
   - Implement OccupationClusterModel:
     * Clusters occupations offline (store cluster assignments)
     * For rare occupations (< 30 total interactions), use the
       cluster's aggregate calibration parameters instead of
       occupation-specific ones
     * Transition from cluster-level to occupation-level calibration
       as data accumulates (linear blend based on interaction count)
   - Include a cluster_quality.py that computes silhouette scores
     and identifies occupations that are poorly represented by their
     cluster (high distance to centroid).

3. cold_start_combination.py — UNSEEN SKILL COMBINATIONS:
   Strategy: Deterministic baseline with uncertainty flag
   - When the calibration model encounters a feature vector outside
     its training distribution (implement a simple novelty detection:
     Mahalanobis distance from training data centroid > threshold),
     flag it and fall back to the v1 deterministic scorer.
   - Add an `uncertainty` field to scoring responses:
     * LOW: Calibration model has seen similar transitions, confident
     * MEDIUM: Limited data, blending calibration with baseline
     * HIGH: Novel combination, using baseline only
   - Surface this uncertainty to the API response so the frontend
     can communicate confidence to users.

Tests for each module with clear scenarios. Include edge cases:
- Brand new system with zero interaction data
- User whose occupation has only 5 total historical users
- Occupation that doesn't cleanly fit any cluster
```

---

## E2 — Scoring Transparency & Bucket Explainability

### Prompt 2.3: Bucket Explanation Engine

```
>>> Build a scoring transparency layer that explains WHY each occupation
lands in its bucket. Users making career decisions need to trust the
tool, and trust requires explainability.

Create: app/features/explainability/

1. bucket_explainer.py:
   - For each scored occupation, generate a structured explanation:
     {
       "occupation": "Data Analyst",
       "bucket": "TRAINABLE",
       "explanation": {
         "summary": "You have 7 of 10 key skills. The 3 gaps are
                      learnable within 6-12 months.",
         "skills_you_have": [
           {"skill": "Critical Thinking", "your_level": 4.2,
            "required_level": 4.0, "status": "MEETS"},
           ...
         ],
         "skills_to_develop": [
           {"skill": "Programming", "your_level": 1.5,
            "required_level": 3.8, "gap_size": 2.3,
            "gap_category": "SIGNIFICANT",
            "typical_training_time": "3-6 months",
            "why_it_matters": "Used daily for data manipulation
                               and analysis automation"},
           ...
         ],
         "bucket_reasoning": "Placed in TRAINABLE because: 70% skill
           coverage (threshold: >50%), largest gap is 2.3 (threshold:
           <3.0 for TRAINABLE), and no hard credential barriers.",
         "what_would_change_bucket": {
           "to_ready_now": "Close the Programming gap to level 3.0+
                            and the SQL gap to level 3.5+",
           "to_long_reskill": "N/A — you're well above the threshold"
         }
       }
     }

   - CRITICAL: Show the actual thresholds used for bucket boundaries.
     No black boxes. If READY_NOW requires >85% skill coverage, say so.

2. threshold_config.py:
   - Externalize ALL bucket thresholds into a configuration object:
     * skill_coverage_thresholds (per bucket)
     * max_gap_size_thresholds (per bucket)
     * credential_barrier_rules
     * weights for different skill domains
   - Include a "relaxed" and "strict" preset so users can adjust
     their risk tolerance:
     * RELAXED: "I'm a fast learner, show me stretch opportunities"
       (lowers READY_NOW threshold by 10%, expands TRAINABLE range)
     * STRICT: "Only show me safe bets"
       (raises thresholds by 10%)
   - This user preference feeds into scoring as a parameter, not
     a post-hoc filter.

3. API endpoint: GET /api/v1/recommendations/{occupation_id}/explain
   - Returns the full explanation object
   - Also: PATCH /api/v1/user/preferences with
     { "risk_tolerance": "relaxed" | "standard" | "strict" }

4. comparison_view.py:
   - Implement side-by-side comparison for up to 3 occupations:
     GET /api/v1/recommendations/compare?ids=101,205,312
   - Returns a unified view showing skill overlap, unique gaps per
     occupation, and which occupation is closest to READY_NOW.

Tests: Include scenarios where explanation correctly identifies:
- An occupation that's TRAINABLE but one skill gap away from READY_NOW
- An occupation in LONG_RESKILL due to a single hard credential barrier
  (not skill gaps)
- Two occupations in the same bucket but for very different reasons

Annotate threshold values with comments explaining how they were
chosen and what would justify changing them.
```

---

## E3 — Bias Audit Framework

### Prompt 3.3: O*NET Bias Audit Pipeline

```
>>> Build a bias audit framework that systematically tests whether
SkillSprout's scoring pipeline produces equitable results across
demographic lines. O*NET data is survey-derived from incumbent workers,
which means it reflects who CURRENTLY holds jobs, not who COULD.

Create: ml/bias_audit/

1. audit_framework.py — Core audit engine:
   - Define occupation demographic profiles using BLS data (or stubs
     if BLS integration isn't feasible this week):
     * Gender composition per occupation
     * Race/ethnicity composition per occupation
     * Typical education level per occupation
     * Median age per occupation
   - For each demographic dimension, test:
     * BUCKET DISTRIBUTION PARITY: Do occupations dominated by one
       demographic group systematically land in different buckets?
       Example: if "pink collar" occupations (healthcare support,
       education, admin) disproportionately appear as LONG_RESKILL
       targets while "white collar" occupations with equivalent
       skill profiles appear as TRAINABLE, that's a bias signal.
     * SKILL PROFILE STALENESS: Flag occupations where O*NET data
       was last updated more than 5 years ago. These are likely to
       have skill profiles that no longer match reality — and
       they're disproportionately occupations held by underrepresented
       groups.
     * SCORE SYMMETRY: Is the transition score from occupation A→B
       the same as B→A? If not, does the asymmetry correlate with
       demographic differences between A and B? (e.g., is it
       systematically easier to score well transitioning FROM
       male-dominated TO female-dominated occupations than vice versa?)

2. audit_report.py — Report generator:
   - Produces a markdown report (docs/bias-audit-report.md) with:
     * Summary statistics and distribution plots (as ASCII tables
       for now, note where D3 visualizations should go)
     * Flagged occupations with potential bias concerns
     * Specific skill domains where bias is most pronounced
     * Recommended mitigations for each finding

3. mitigation_strategies.py — Concrete fixes:
   - Implement at least two mitigation approaches:
     * SKILL_REWEIGHTING: Option to downweight O*NET skill importance
       scores that show high demographic correlation (configurable,
       not default — document the tradeoff)
     * STALENESS_PENALTY: Reduce confidence in scores for occupations
       with stale O*NET data (older than 3 years gets a warning flag,
       older than 5 years gets a confidence downgrade)
   - Each mitigation is toggleable via configuration and logged when
     active.

4. tests/test_bias_audit.py:
   - Test with synthetic occupation pairs that SHOULD produce
     symmetric scores
   - Test that staleness penalty correctly identifies old data
   - Test that the audit report generates without errors

IMPORTANT: Annotate extensively. Every bias detection method should
include comments explaining:
- What specific harm this detects
- Why this matters for career transition equity
- Limitations of the detection method
- What a human reviewer should do with the findings
```

---

### Day 2 Sync Point
> **End of day coordination:** E1 demos the transition-aware features and cold start handling. E2 demos the explainability engine (bucket explanations + threshold config). E3 presents bias audit findings (even preliminary). Discuss: do the bias findings require changes to E1's feature engineering or E2's explanation language?

---

# Day 3 (Wednesday): User Experience & Training Integration

**Theme:** Make the tool useful for real people, especially those who need it most.

---

## E1 — Implicit Feedback & Signal Enhancement

### Prompt 1.5: Enhanced Event Tracking & Implicit Signals

```
>>> The current event tracking captures explicit actions (clicks, saves,
applies), but these are sparse and delayed. Implement a richer implicit
feedback collection system.

Modify: app/events/ (or create if it doesn't exist)

1. implicit_signals.py — New signal types:
   - DWELL_TIME: How long a user spends viewing a specific occupation
     recommendation. Implement via heartbeat API:
     * POST /api/v1/events/heartbeat
       { "occupation_id": 123, "session_id": "abc", "seconds": 5 }
     * Called every 5 seconds while a recommendation card is in
       viewport. Aggregate server-side into total dwell time.
     * Dwell > 30 seconds on a single occupation = positive signal
     * Dwell < 3 seconds = negative signal (scrolled past)

   - EXPLANATION_ENGAGEMENT: Did the user expand the skill-gap details?
     * POST /api/v1/events/explanation-view
       { "occupation_id": 123, "expanded_sections": ["skills_gap",
         "training_paths"] }
     * Expanding explanation = stronger interest signal than just
       viewing the card

   - COMPARISON_BEHAVIOR: Which occupations did the user compare?
     * Track compare API usage as implicit preference signal
     * If user compares A vs B and then saves A → strong signal
       that A is preferred over B (pairwise preference data)

   - SEARCH_REFINEMENT: When a user adjusts risk tolerance or
     re-translates skills, track the before/after to understand
     what they're looking for.

2. signal_aggregator.py — Combine signals into training features:
   - Implement a SignalAggregator that combines all signals for a
     (user, occupation) pair into a feature vector:
     * total_dwell_seconds (log-transformed)
     * explanation_expanded (bool)
     * times_viewed (count)
     * comparison_wins (count of times chosen over alternatives)
     * days_since_first_view
     * save_after_explain (bool — did they save AFTER viewing
       explanation? Stronger signal than save without explanation)
   - Output: ready to feed into calibration layer as additional
     features

3. pairwise_preference.py — Prepare for learning-to-rank:
   - Collect pairwise preference data from comparison behavior
   - Store as (user_id, preferred_occupation_id,
     non_preferred_occupation_id, context)
   - This is FUTURE fuel for a LambdaMART ranking model.
     Document the schema and intended usage even though the model
     isn't built yet.

Tests: Simulate a realistic user session and verify all signals
are captured correctly and aggregated properly.
```

---

## E2 — Training Path Integration & Resource-Aware Recommendations

### Prompt 2.4: Actionable Training Recommendations

```
>>> "Learn Python" is not an actionable recommendation. Build a training
path system that gives users concrete, resource-aware next steps.

Create: app/features/training_paths/

1. training_catalog.py — Structured training resource database:
   - Define a TrainingResource model:
     {
       "id": "google-data-analytics-cert",
       "title": "Google Data Analytics Professional Certificate",
       "provider": "Coursera / Google",
       "skills_covered": ["SQL", "Data Analysis", "R Programming",
                          "Data Visualization", "Spreadsheets"],
       "format": "ONLINE_SELF_PACED",
       "cost_usd": 0,  // free on Coursera with financial aid
       "cost_tier": "FREE",  // FREE, LOW (<$200), MEDIUM (<$1000), HIGH
       "duration_weeks": 24,
       "hours_per_week": 10,
       "credential_type": "PROFESSIONAL_CERTIFICATE",
       "prerequisite_skills": [],
       "url": "https://...",
       "employer_recognition": "HIGH",  // how widely recognized
       "completion_rate": 0.15,  // if known
       "last_verified": "2025-01-15"
     }

   - Seed the catalog with 30-50 high-quality, verified resources
     across these categories:
     * Free certificates (Google, IBM, Meta via Coursera)
     * Community college programs (use placeholder URLs, note that
       these need per-geography customization)
     * Government-funded programs (WIOA, TAA, veteran benefits)
     * Bootcamps (note cost and outcomes data where available)
     * Self-directed (freeCodeCamp, Khan Academy, MIT OCW)

   - Map each resource to O*NET skill codes it develops.

2. path_generator.py — Personalized training paths:
   - Input: User's skill gaps (from bucket_explainer), constraints
   - User constraints model:
     {
       "budget_max_usd": 500,
       "hours_per_week_available": 10,
       "preferred_format": ["ONLINE_SELF_PACED", "HYBRID"],
       "has_reliable_internet": true,
       "has_personal_computer": true,
       "location_zip": "21201",  // for local program matching
       "timeline_months": 6
     }

   - Output: Ordered sequence of training resources that:
     * Covers all skill gaps for the target occupation
     * Respects budget and time constraints
     * Prioritizes free/low-cost options first
     * Orders by prerequisite chains (learn SQL before advanced
       data analysis)
     * Includes estimated timeline:
       "Weeks 1-4: Google Data Analytics (Module 1-2)
        Weeks 5-12: Complete certificate
        Weeks 13-16: Portfolio project
        Weeks 17-20: Apply to entry-level roles"

   - CRITICAL: If constraints make the path infeasible (e.g., $0
     budget but only paid options cover a required skill), say so
     explicitly and suggest alternatives (library resources, free
     MOOCs, local workforce board programs).

3. resource_filter.py — Constraint-aware filtering:
   - Filter catalog by user constraints
   - Flag resources that require prerequisites the user doesn't have
   - Handle the "no computer" case: filter to in-person only,
     suggest public library computer labs

4. API endpoints:
   - POST /api/v1/training-path/generate
     { "target_occupation_id": 123, "constraints": {...} }
   - GET /api/v1/training-resources?skill=programming&cost_tier=FREE

Tests: Include constraint scenarios:
- Zero budget, all gaps coverable by free resources → valid path
- Zero budget, one gap requires paid-only resource → explicit flag
- 5 hours/week available → extended timeline calculation
- No personal computer → in-person/library filtered results
```

---

## E3 — Monitoring, Health Checks & Operational Reliability

### Prompt 3.4: Production Monitoring & Observability

```
>>> Build a production monitoring and observability layer for SkillSprout.
The system needs to be monitorable before it can be reliable.

Create: app/core/monitoring/

1. health_checks.py — Comprehensive health check system:
   - GET /health → basic liveness (returns 200 if process is running)
   - GET /health/ready → readiness check that verifies:
     * PostgreSQL connection is alive and responsive (< 500ms)
     * Redis connection is alive
     * Celery worker is accepting tasks (submit a ping task, verify
       completion within 5 seconds)
     * O*NET data cache is populated (not empty)
     * ML model artifacts are loaded and valid
   - GET /health/detailed → returns JSON with individual component
     status and latency:
     {
       "status": "degraded",
       "components": {
         "database": {"status": "healthy", "latency_ms": 12},
         "redis": {"status": "healthy", "latency_ms": 3},
         "celery": {"status": "unhealthy", "error": "no workers"},
         "onet_cache": {"status": "healthy", "items": 974,
                        "last_refresh": "2025-01-15T..."},
         "ml_model": {"status": "healthy", "version": "v2.1",
                      "loaded_at": "2025-01-15T..."}
       }
     }

2. metrics.py — Application metrics collection:
   - Use prometheus_client to expose metrics at /metrics:
     * REQUEST METRICS: request_duration_seconds (histogram, by
       endpoint and status code), requests_total (counter)
     * SCORING METRICS: scoring_duration_seconds, scores_by_bucket
       (counter per bucket), cold_start_fallbacks_total (counter
       by type: user/occupation/combination)
     * CALIBRATION METRICS: calibration_model_version (info gauge),
       calibration_prediction_distribution (histogram of predicted
       scores — to detect drift)
     * FEEDBACK METRICS: events_received_total (by type),
       feedback_loop_latency_seconds (time from event to model
       training data)
     * SYSTEM METRICS: db_connection_pool_size, redis_connection_count,
       celery_queue_depth

3. alerting_rules.py — Define alerting thresholds:
   - Don't implement a full alerting system, but define the rules
     as code with clear thresholds and descriptions:
     * P1: Health check fails for > 60 seconds
     * P1: Scoring latency p99 > 2 seconds
     * P2: Cold start fallback rate > 50% of requests
     * P2: Calibration model not retrained in > 7 days
     * P3: O*NET cache older than 30 days
   - Output as a prometheus_rules.yml compatible format AND
     as documentation.

4. request_logging.py — Structured request logging:
   - Middleware that logs every request as structured JSON:
     { "timestamp", "request_id", "method", "path", "status_code",
       "duration_ms", "user_id" (hashed, not raw), "bucket_result",
       "cold_start_type", "error" }
   - PRIVACY: Never log raw user skill profiles, descriptions,
     or occupation exploration patterns. Log hashed user_id only.
   - Include correlation IDs that follow a request through
     async/Celery boundaries.

Tests: Verify health checks correctly report unhealthy when
components are down (mock the connections). Verify metrics
increment correctly. Verify that sensitive data never appears
in logs.
```

---

### Day 3 Sync Point
> **End of day coordination:** E1 demos implicit signal collection and how it feeds the calibration layer. E2 demos the training path generator with real constraint scenarios (show the "zero budget" and "no computer" paths). E3 demos the health check system and walks through the metrics/alerting rules. Key discussion: are E2's training resources mapped to the same O*NET skill codes that E1's scorer uses? Alignment check.

---

# Day 4 (Thursday): Integration, User Journey & Social Proof

**Theme:** Wire everything together into a coherent user experience. Add the features that drive retention.

---

## E1 — Calibration Monitoring & Model Versioning

### Prompt 1.6: Calibration Monitoring & Model Registry

```
>>> Build the infrastructure to monitor model quality in production and
manage model versions. Without this, you can't know if your models
are helping or hurting.

Create: ml/model_management/

1. calibration_monitor.py — Real-time calibration tracking:
   - Implement CalibrationMonitor that runs as a scheduled Celery task
     (weekly):
     * Pulls the last 7 days of (prediction, outcome) pairs
     * Computes:
       - Reliability diagram data (10 bins: predicted probability vs
         actual frequency)
       - Expected Calibration Error (ECE)
       - Per-bucket accuracy: of all READY_NOW recommendations, what
         % resulted in user engagement (save/apply)?
       - Score distribution shift: compare this week's prediction
         distribution to the training distribution using
         Kolmogorov-Smirnov test
     * Stores results in a calibration_snapshots table
     * Flags degradation:
       - ECE > 0.15 → WARNING
       - ECE > 0.25 → ALERT (model needs retraining)
       - K-S statistic > 0.1 → distribution drift detected

   - CalibrationReport: Generates a weekly markdown report saved to
     docs/calibration/report-{date}.md

2. model_registry.py — Version management:
   - ModelRegistry class that:
     * Stores model artifacts with metadata:
       { "version": "v2.3", "trained_at": "...",
         "training_data_range": "2025-01-01 to 2025-06-15",
         "feature_set": ["skill_overlap", "direction_vector", ...],
         "eval_metrics": {"auc": 0.82, "ece": 0.07},
         "status": "PRODUCTION" | "CANDIDATE" | "RETIRED" }
     * Supports atomic promotion: candidate → production (with
       automatic demotion of the previous production model to retired)
     * Stores the exact feature vector schema used by each model
       version (so old predictions can be reproduced)
     * Implements rollback: revert to previous production model in
       one operation

   - ModelArtifact: Serialized model + feature scaler + threshold
     config, all versioned together as one bundle.

3. ab_test_framework.py — A/B test infrastructure:
   - Implement simple traffic splitting:
     * Assign users to model variants based on hash(user_id) % 100
     * Support percentage-based allocation:
       { "v2.3": 90, "v2.4_candidate": 10 }
     * Log model version with every prediction for analysis
   - Analysis function:
     * Given a date range and model versions, compute per-version
       metrics and statistical significance (chi-squared test for
       bucket-level engagement rates)
   - This doesn't need a UI — CLI and Celery task based is fine.

Tests: Test model promotion/rollback, A/B split determinism (same
user always gets same variant), and calibration metric computation.
```

---

## E2 — User Profile, Save & Track, and Return Visit Loop

### Prompt 2.5: User Profile & Progress Tracking System

```
>>> Build the user profile and progress tracking system that creates
a reason for users to return. Career transitions happen over months.
A tool that only serves a single session will churn before collecting
meaningful data.

Create: app/features/user_profile/

1. profile.py — Lightweight user profile:
   - UserProfile model:
     {
       "user_id": "uuid",
       "display_name": "optional",
       "current_occupation_id": 123,
       "skills_snapshot": [...],  // their assessed skills at profile
                                  // creation
       "constraints": {
         "salary_minimum_usd": 50000,
         "location": "Baltimore, MD",
         "remote_preference": "HYBRID",
         "industry_interests": ["healthcare", "technology"],
         "timeline_months": 12
       },
       "risk_tolerance": "standard",  // relaxed | standard | strict
       "created_at": "...",
       "last_active_at": "..."
     }

   - API: POST /api/v1/profile (create)
          PATCH /api/v1/profile (update)
          GET /api/v1/profile (retrieve)

   - CRITICAL: Profile creation is OPTIONAL. The tool must provide
     full value without an account. Profile adds persistence and
     personalization but is never a gate.

2. saved_occupations.py — Bookmarking with progress:
   - SavedOccupation model:
     {
       "user_id": "uuid",
       "occupation_id": 123,
       "bucket_at_save": "TRAINABLE",
       "current_bucket": "TRAINABLE",  // recomputed periodically
       "saved_at": "...",
       "notes": "user's personal notes",
       "training_status": {
         "active_training": [
           {"resource_id": "google-data-cert", "started_at": "...",
            "progress_pct": 45}
         ],
         "completed_training": [...],
         "skills_gained_since_save": ["SQL", "Data Visualization"]
       }
     }

   - API: POST /api/v1/saved-occupations (save)
          GET /api/v1/saved-occupations (list all saved)
          PATCH /api/v1/saved-occupations/{id} (update progress)
          DELETE /api/v1/saved-occupations/{id} (remove)

   - Implement a Celery task that WEEKLY re-scores all saved
     occupations for active users. If a user's bucket changes
     (e.g., TRAINABLE → READY_NOW after completing training),
     flag it for notification.

3. progress_tracker.py — Skill development tracking:
   - Let users mark skills as "in progress" or "completed":
     POST /api/v1/skills/update
     { "skill_id": "programming", "new_level": 3.5,
       "evidence": "Completed Google cert module 3" }
   - Re-run scoring after skill updates to show bucket movement
   - Compute and expose:
     * skills_gained_count (since profile creation)
     * bucket_improvements: list of occupations that moved to a
       better bucket since the user started tracking
     * estimated_time_to_ready: for each TRAINABLE saved occupation,
       based on their training pace so far, estimate when they'll
       reach READY_NOW

4. return_engagement.py — Data that drives return visits:
   - "Your Progress" summary endpoint: GET /api/v1/progress/summary
     {
       "days_active": 45,
       "skills_developed": 3,
       "occupations_tracked": 5,
       "bucket_improvements": [
         {"occupation": "Data Analyst",
          "moved_from": "TRAINABLE", "moved_to": "READY_NOW",
          "date": "..."}
       ],
       "next_milestone": "Complete SQL module to unlock Data Analyst
                          as READY_NOW",
       "similar_users_stat": "43 users with your skill profile
                              transitioned to Data Analyst in the
                              last 6 months"
     }

   - This is the "hook" — the data that makes opening the app
     rewarding. Design the API response to be directly renderable
     as a dashboard card.

Tests: Full lifecycle test — create profile, save occupations,
update skills, verify bucket re-scoring triggers, verify progress
summary reflects changes.
```

---

## E3 — Low-Bandwidth Mode & Shared Device Support

### Prompt 3.5: Progressive Enhancement & Shared Device Support

```
>>> Build support for low-bandwidth and shared-device usage scenarios.
Many potential users access the internet through public library
computers, shared family phones, or spotty connections.

Create: app/core/progressive/

1. lightweight_api.py — Bandwidth-optimized API layer:
   - Implement response compression middleware (gzip/brotli for all
     JSON responses > 1KB)
   - Add a "lite" mode querystring parameter (?lite=true) that:
     * Returns only essential fields (no explanation details, no
       training catalogs — just occupation name, bucket, and top 3
       skill gaps)
     * Reduces payload size by ~70%
     * Designed for slow connections or data-metered devices

   - Implement response pagination for recommendation lists:
     GET /api/v1/recommendations?page=1&page_size=10
     (currently returns all results, which could be 50+ occupations)

   - Add ETag headers for cacheable responses (O*NET occupation data,
     training catalog). Clients can skip re-downloading unchanged data.

2. session_resumption.py — Survive interrupted sessions:
   - Problem: A user on a library computer starts exploring, their
     time expires, they come back the next day on a different
     computer.
   - Solution: Implement session state as a shareable token:
     * POST /api/v1/session/export → returns an encrypted,
       URL-safe token encoding:
       { "skills": [...], "saved_occupations": [...],
         "constraints": {...} }
     * POST /api/v1/session/import { "token": "..." } → restores
       session state
     * Token is short enough to write down, email to self, or
       encode as QR code (~200 chars for basic session)
   - This works WITHOUT an account. Users can resume their session
     without creating a profile.
   - Token encryption: Use Fernet symmetric encryption with a
     server-side key. Tokens expire after 30 days.

3. offline_capability.py — Static export for offline review:
   - GET /api/v1/recommendations/export?format=pdf
   - GET /api/v1/recommendations/export?format=csv
   - Generates a printable/downloadable summary of:
     * User's skill profile
     * Top 10 recommendations with explanations
     * Training paths for saved occupations
   - Users can print this at the library and review at home.

4. accessibility_middleware.py — Core accessibility enforcement:
   - Middleware that adds/enforces:
     * Proper Content-Type and charset headers
     * CORS headers for assistive technology browser extensions
     * Response time budget: log WARNING if any endpoint takes > 3
       seconds (critical for screen readers and low-bandwidth users
       who may assume the page is broken)
   - API response schema validation: ensure all responses include
     human-readable "description" fields alongside any coded values
     (never return just {"bucket": 2} — always include
     {"bucket": 2, "bucket_label": "TRAINABLE",
      "bucket_description": "Skills within reach with training"})

Tests:
- Verify lite mode reduces payload size significantly
- Verify session token roundtrip (export → import → same state)
- Verify CSV/PDF export contains all expected fields
- Verify accessibility middleware adds required headers
- Verify pagination works correctly at boundaries
```

---

### Day 4 Sync Point
> **End of day coordination:** This is the critical integration day. E2 demos the full user journey: translate skills → see explained buckets → save occupations → view training paths → track progress. E1 confirms the scoring pipeline accepts the new user profile constraints. E3 demos session resumption and lite mode. Key question: does the full journey work end-to-end with all three engineer's code connected?

---

# Day 5 (Friday): Testing, Documentation & Polish

**Theme:** Harden everything. Write the tests. Document the architecture. Prepare for real users.

---

## E1 — Integration Tests & Model Documentation

### Prompt 1.7: End-to-End Integration Tests

```
>>> Write comprehensive integration tests that verify the full scoring
pipeline works end-to-end with all the new components.

Create: tests/integration/

1. test_full_pipeline.py — Complete scoring journey:
   - Test: Plain text input → skills translator → scoring with
     transition features → bucket assignment with explanation →
     training path generation
   - Use 5 realistic personas:
     * "Maria": Registered nurse, 8 years experience, wants to
       transition to health informatics. Budget: $500. Timeline: 1yr.
     * "James": Retail store manager, 15 years. Exploring anything
       that uses his people management skills. No budget for training.
     * "Aisha": Recent CS bootcamp grad. Wants to understand which
       roles she's actually qualified for vs. which need more work.
     * "Robert": Auto mechanic, 20 years. Shop is closing. Needs to
       know what's realistic in 6 months.
     * "Sarah": Military veteran (logistics MOS). Transitioning to
       civilian workforce. Has GI Bill for training.
   - For each persona, verify:
     * Skills translator produces reasonable skill mappings
     * Bucket assignments pass a sanity check (nurse → health
       informatics should be TRAINABLE, not LONG_RESKILL)
     * Explanations reference the actual skill gaps
     * Training paths respect stated constraints
     * Cold start handling activates appropriately

2. test_feedback_loop.py — Verify the learning loop:
   - Simulate: user views recommendations → dwells on some →
     saves others → comes back and applies → outcome recorded
   - Verify: all events captured → signals aggregated →
     calibration training data correctly formed → model can
     retrain on this data without errors

3. test_privacy_compliance.py — Privacy guarantees under load:
   - Test: 100 simulated users in private mode produce zero
     database writes to event tables
   - Test: Data deletion removes ALL traces (query every table)
   - Test: Data export includes everything (compare export to
     raw database query)
   - Test: Retention policy correctly purges 91-day-old events

4. test_bias_detection.py — Regression tests for bias:
   - Test: Score symmetry for 20 occupation pairs (A→B vs B→A)
   - Test: No bucket has > 80% concentration of any one occupation
     demographic category (catch bucket segregation)
   - Test: Staleness warnings fire for occupations with old data

Include a conftest.py with fixtures for:
- Seeded test database with O*NET data
- Pre-configured user profiles for each persona
- Mock external services (O*NET API, training catalog)

All tests must pass AND produce a coverage report. Target: >80%
line coverage on new code from this week.
```

### Prompt 1.8: ML Architecture Decision Record

```
>>> Write comprehensive architecture decision records for all ML choices
made this week. These are for future engineers (including future you)
who need to understand WHY, not just WHAT.

Create: docs/adr/ (Architecture Decision Records)

Write one ADR for each major decision, following this template:
- Title
- Status (Accepted)
- Context (what problem were we solving?)
- Decision (what did we choose?)
- Consequences (what are the tradeoffs?)
- Alternatives Considered (what else could we have done?)

Required ADRs:

1. ADR-001-two-stage-scoring.md
   Why deterministic baseline + learned calibration instead of a single
   end-to-end model? Cover: explainability requirements, cold start
   behavior, debugging, the path to more sophisticated models.

2. ADR-002-transition-features.md
   Why transition-aware features? Cover: the asymmetry problem, what
   signals they capture that pure skill overlap misses, computational
   cost, the risk of overfitting on sparse transition data.

3. ADR-003-cold-start-strategy.md
   Why three-tier cold start (user priors, occupation clusters,
   uncertainty flags)? Cover: alternatives (ask users more questions
   upfront, use collaborative filtering, just show everything).

4. ADR-004-calibration-over-ranking.md
   Why logistic regression calibration now instead of learning-to-rank?
   Cover: data requirements for LTR, the pairwise data collection
   strategy, when to make the switch (what data volume threshold?).

5. ADR-005-feedback-signal-hierarchy.md
   Why this specific signal hierarchy (outcome > application > save >
   dwell > view)? Cover: noise levels, latency, volume, the
   composite label strategy.

6. ADR-006-bias-audit-approach.md
   Why demographic parity testing? Cover: limitations of the approach,
   what it can and can't detect, the philosophical tension between
   "equal scores" and "equal outcomes," and why transparency about
   limitations is itself a mitigation.

Make these genuinely useful — not checkbox compliance docs. Write them
as if explaining your reasoning to a sharp skeptic.
```

---

## E2 — API Documentation & Developer Experience

### Prompt 2.6: OpenAPI Documentation & Developer Onboarding

```
>>> Create comprehensive API documentation and a developer onboarding
guide.

1. Update/create the OpenAPI spec:
   - Ensure every endpoint added this week has complete OpenAPI 3.0
     documentation:
     * Request/response schemas with examples
     * Error response schemas (400, 404, 422, 500)
     * Authentication requirements (or lack thereof)
     * Rate limiting headers
   - Add descriptions that explain the PURPOSE of each endpoint, not
     just the mechanics. Example:
     BAD: "Returns recommendations for a user"
     GOOD: "Returns scored occupation recommendations grouped into
     three buckets (READY_NOW, TRAINABLE, LONG_RESKILL) based on
     the user's current skill profile. Each recommendation includes
     a confidence indicator and can be expanded with the /explain
     endpoint for full skill-gap breakdown."

   - Generate the spec file: docs/openapi.yaml

2. Create docs/developer-guide.md — Onboarding document:
   - Local development setup (Docker, env vars, database seeding)
   - Architecture overview with a request flow diagram (ASCII art
     or mermaid syntax):
     User Input → Skills Translator → Scoring Pipeline →
     [Deterministic Baseline → Feature Engineering →
      Calibration Layer → Cold Start Check] → Bucket Assignment →
     Explanation Engine → Training Path Generator → Response
   - How to add a new feature to the scoring pipeline
   - How to add a new training resource to the catalog
   - How to run the bias audit
   - How to run the evaluation framework
   - How to deploy (Docker + CI/CD pipeline)

3. Create docs/user-guide.md — End user documentation:
   - Plain language explanation of how SkillSprout works
   - What the three buckets mean and how to use them
   - How to interpret skill gap explanations
   - How to use the skills translator
   - Privacy: what data is stored, for how long, and how to delete it
   - Session resumption: how to save and restore your session
   - FAQ addressing common concerns:
     * "Why is [occupation] in LONG_RESKILL? I think I could do it."
     * "How accurate are these recommendations?"
     * "Who can see my data?"
     * "I don't have a computer at home — can I still use this?"

Keep all docs concise and scannable. Use the minimum formatting
needed for clarity.
```

---

## E3 — Deployment Hardening & Open Source Preparation

### Prompt 3.6: Docker Optimization & CI/CD Hardening

```
>>> Harden the deployment pipeline and prepare the codebase for
potential open-source release.

1. Docker optimization — Modify Dockerfile(s):
   - Multi-stage build:
     * Stage 1 (builder): Install all dependencies, run tests
     * Stage 2 (runtime): Copy only necessary artifacts, slim base
       image (python:3.12-slim)
   - Run as non-root user (appuser with UID 1000)
   - Pin ALL dependency versions (no floating versions in
     requirements.txt)
   - Add .dockerignore for: .git, __pycache__, .env, docs/,
     tests/, *.md
   - Health check instruction:
     HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
       CMD curl -f http://localhost:8000/health || exit 1
   - Target image size < 500MB (measure and optimize if larger)

2. CI/CD pipeline — Update .github/workflows/:
   - Pipeline stages:
     * lint (ruff + mypy)
     * unit-tests (pytest, fast, no external deps)
     * integration-tests (pytest with test database, Docker services)
     * bias-audit (run the bias audit framework, fail if critical
       issues found)
     * security-scan (pip-audit for known vulnerabilities,
       bandit for code issues)
     * build (Docker image build + push to registry)
     * deploy (only on main branch, after all checks pass)
   - Cache pip and Docker layers between runs
   - Fail fast: lint and security run in parallel, tests run after
   - Store test coverage report as artifact
   - Store bias audit report as artifact

3. Environment configuration — Create:
   - .env.example with ALL required variables documented:
     * Database connection (with explanation of each parameter)
     * Redis connection
     * O*NET API credentials (with link to registration page)
     * Feature flags (ENABLE_PRIVATE_MODE, ENABLE_BIAS_MITIGATIONS,
       ENABLE_LITE_MODE)
     * Model configuration (CALIBRATION_MODEL_VERSION,
       COLD_START_CLUSTER_K)
   - docker-compose.yml for local development (app + postgres +
     redis + celery worker)
   - docker-compose.test.yml for running full test suite

4. Open source preparation — Create:
   - LICENSE (MIT — discuss with team if different license needed)
   - CONTRIBUTING.md:
     * How to set up development environment
     * Code style and testing requirements
     * How to submit a bug report
     * How to add training resources to the catalog
     * How to contribute bias audit improvements
     * Code of conduct reference
   - README.md:
     * Project overview (one paragraph)
     * Quick start (5 commands from clone to running)
     * Architecture diagram
     * Link to full docs
     * Link to live demo (if applicable)
     * "Built with" badges
     * Current status and roadmap

Make the README compelling enough that a workforce development board
staffer could understand what this is and why they'd want it within
60 seconds of reading.
```

### Prompt 3.7: Transition Graph Data Pipeline (Stretch Goal)

```
>>> This is the stretch goal / bold bet. If time permits after
completing the hardening tasks, build the foundation for SkillSprout's
most defensible data asset: the career transition graph.

Create: ml/transition_graph/

1. graph_builder.py — Build a directed graph from user behavior:
   - Every user journey traces a path through career space:
     Origin Occupation → [Explored Occupation 1, 2, 3, ...] →
     [Saved Occupation(s)] → [Applied Occupation(s)] →
     [Outcome Occupation]
   - Aggregate these paths into a weighted directed graph:
     * Nodes = O*NET occupations
     * Edges = observed transitions (exploration, not just outcomes)
     * Edge weights = count of users who traversed this edge
     * Edge metadata:
       - median_skill_overlap for users on this edge
       - median_time_from_first_view_to_apply
       - success_rate (applications that led to positive outcomes)

   - Build incrementally: Celery task runs nightly, processes new
     events since last run, updates graph edges.

   - Store as both:
     * Adjacency list in PostgreSQL (for querying)
     * NetworkX graph object serialized to disk (for analysis)

2. graph_queries.py — Useful queries on the transition graph:
   - most_common_transitions(occupation_id, top_k=10):
     "People in your role most commonly explore..."
   - successful_paths(origin_id, target_id):
     "Users who made this transition typically went through..."
   - emerging_transitions(min_edge_weight=5, recent_days=30):
     "New transition patterns we're seeing..."
   - transition_difficulty(origin_id, target_id):
     Based on success_rate and time_to_transition for users
     who actually traversed this edge.

3. graph_recommendations.py — Graph-enhanced scoring:
   - Supplement the skill-based scorer with graph signals:
     * If many users from your origin occupation successfully
       transitioned to target X, boost X's score (social proof)
     * If a target occupation has zero observed transitions from
       your origin, flag it as "unexplored path" (not necessarily
       bad, but no social proof)
   - IMPORTANT: Graph signals are ADDITIVE to skill-based scores,
     never overriding. A well-traveled path with huge skill gaps
     is still LONG_RESKILL.

   - Weight graph signal by data freshness:
     graph_weight = base_weight * decay(days_since_most_recent_edge)
     So the graph stays current as patterns change.

Tests:
- Build a small synthetic graph (10 occupations, 20 edges)
- Verify queries return expected results
- Verify nightly update correctly adds new edges without duplicates
- Verify graph recommendations blend correctly with skill-based scores

Document this as experimental / alpha. The graph only becomes valuable
with sufficient user volume — estimate the minimum number of users
needed for the graph to produce meaningful signal (probably ~1000
active users with outcome data).
```

---

### Day 5 Sync Point (Final Demo)
> **End of week demo:** Each engineer demos their full contribution. Then run the integration test suite together. Review: test coverage report, bias audit report, calibration metrics, and the developer guide. Identify the top 3 items that didn't get finished and create tickets for them.

---

# Post-Hackathon: Priority Backlog

These items were identified during the week but are out of scope for the hackathon. Ordered by impact.

| Priority | Item | Owner | Notes |
|----------|------|-------|-------|
| P0 | Partner with one training provider for verified catalog data | Product | Google Certs or local community college |
| P0 | Define wedge user persona and test with 5 real users | Product | Veterans or bootcamp grads recommended |
| P1 | Learning-to-rank model (LambdaMART) using pairwise data | E1 | Blocked until ~500 pairwise preference records |
| P1 | Geographic training resource integration | E2 | Needs ZIP code → program mapping |
| P1 | Workforce development board pilot partnership | Product | American Job Centers, WIOA alignment |
| P2 | "I don't see myself here" feedback mechanism | E2 | Bias detection signal |
| P2 | BLS labor market data integration | E1 | Replace demand signal stubs |
| P2 | Load testing and performance optimization | E3 | Target: 100 concurrent users |
| P3 | Transition graph visualization (D3) | E2 | Depends on graph having enough data |
| P3 | Multi-language support | E2 | Spanish first, then Mandarin |
