# SkillSprout Modeling Notes

This document describes the modeling approach, formulas, thresholds, and evolution strategy for the SkillSprout recommendation system.

## Overview

SkillSprout uses a **progressive modeling strategy** that starts with a strong deterministic baseline and evolves into a learning system as user feedback accumulates.

### Model Versions

1. **Model v1 (Baseline)**: Deterministic, rule-based scoring
2. **Model v2 (Calibrated)**: Learned calibration layer on top of baseline
3. **Model v3 (Future)**: More sophisticated approaches (embeddings, LLMs, etc.)

---

## Model v1: Baseline Scoring

### Purpose

Establish a strong, interpretable baseline that works immediately without training data.

### Components

#### 1. User Capability Representation

User ratings (0-4) map to capability scalars:

```
Rating → Capability
  0    →    0.0     (None - no experience)
  1    →    0.25    (Basic - familiar with concepts)
  2    →    0.5     (Intermediate - can work independently)
  3    →    0.75    (Advanced - highly proficient)
  4    →    1.0     (Expert - mastery level)
```

**Rationale**: Linear mapping provides interpretable scores and reasonable discrimination between levels.

#### 2. Occupation Skill Representation

Each occupation is represented as a weighted vector of skills:

- **Element ID**: O*NET skill identifier (e.g., "2.B.1.a")
- **Skill Name**: Human-readable name (e.g., "Reading Comprehension")
- **Importance**: 0-100 scale (from O*NET)
- **Level**: 0-7 scale (from O*NET, optional)

**Weight Calculation**:
```
weight_i = importance_i / Σ(importance_j for all skills j)
```

#### 3. Match Score

Measures how well user capabilities align with job requirements.

**Formula**:
```
match_score = 100 × Σ(weight_i × capability_i) / Σ(weight_i)
            = 100 × Σ(weight_i × capability_i)  [since weights sum to 1]
```

Where:
- `weight_i` = normalized importance of skill i
- `capability_i` = user capability for skill i (0.0 to 1.0)

**Range**: 0-100
- 100 = Perfect match (expert in all required skills)
- 0 = No match (no capability in any required skill)

**Example**:
```
Occupation has 3 skills:
  - Programming (importance: 80) → weight: 0.4
  - Critical Thinking (importance: 70) → weight: 0.35
  - Writing (importance: 50) → weight: 0.25

User ratings:
  - Programming: 4 (expert) → capability: 1.0
  - Critical Thinking: 3 (advanced) → capability: 0.75
  - Writing: 2 (intermediate) → capability: 0.5

match_score = 100 × (0.4×1.0 + 0.35×0.75 + 0.25×0.5)
            = 100 × (0.4 + 0.2625 + 0.125)
            = 78.75
```

#### 4. Gap Severity

Measures the weighted importance of skills where user has low capability.

**Formula**:
```
gap_severity = 100 × Σ(weight_i for skills where capability_i ≤ 0.25) / Σ(weight_i)
             = 100 × Σ(weight_i for skills where capability_i ≤ 0.25)
```

**Gap Threshold**: capability ≤ 0.25 (ratings 0-1)

**Range**: 0-100
- 100 = All critical skills are gaps
- 0 = No significant gaps

**Rationale**: Focuses on severe gaps (none/basic level) rather than moderate skill deficiencies.

**Example** (continuing above):
```
Gaps (capability ≤ 0.25): None in this example

gap_severity = 0
```

**Example with gaps**:
```
User ratings:
  - Programming: 1 (basic) → capability: 0.25 → GAP
  - Critical Thinking: 4 (expert) → capability: 1.0
  - Writing: 0 (none) → capability: 0.0 → GAP

gap_severity = 100 × (0.4 + 0.25)  [weights of Programming and Writing]
             = 65
```

#### 5. Bucket Assignment

Categorizes occupations into three actionable buckets.

**Logic**:

```python
if match_score >= 75 and gap_severity <= 25:
    bucket = "READY_NOW"
elif (50 <= match_score <= 74) or (26 <= gap_severity <= 55):
    bucket = "TRAINABLE"
else:
    bucket = "LONG_RESKILL"
```

**Thresholds** (configurable via environment variables):

| Bucket | Match Score | Gap Severity | Interpretation |
|--------|-------------|--------------|----------------|
| READY_NOW | ≥ 75 | ≤ 25 | High match, low gaps → Apply now |
| TRAINABLE | 50-74 | 26-55 | Moderate match or gaps → Train 3-18mo |
| LONG_RESKILL | < 50 or gaps > 55 | > 55 or match < 50 | Significant reskilling needed → 1-4+ years |

**Decision Logic Explanation**:
- **READY_NOW**: Both conditions must be true (AND logic)
- **TRAINABLE**: Either condition can be true (OR logic)
  - Moderate match (50-74) even with low gaps, OR
  - Moderate gaps (26-55) even with lower match
- **LONG_RESKILL**: Everything else (catch-all)

#### 6. Training Suggestions

Heuristic-based recommendations for skill development.

**Job Zone Mapping**:

| Job Zone | Typical Education | Training Suggestion (TRAINABLE) | Training Suggestion (LONG_RESKILL) |
|----------|-------------------|--------------------------------|-----------------------------------|
| 1-2 | HS diploma / Some training | Certificate, apprenticeship (3-12mo) | Vocational training (1-2yr) |
| 3 | Associate's / Training | Bootcamp, certificate + portfolio (3-18mo) | Associate's degree, comprehensive program (1-3yr) |
| 4-5 | Bachelor's+ | Associate's, extended bootcamp (1-2yr) | Bachelor's degree, multi-year program (2-4yr) |

**Formula**:
```python
def generate_training_suggestion(bucket, job_zone, num_gaps):
    if bucket == "ready_now":
        return "Apply now! Refresh skills with online courses."

    if bucket == "trainable":
        if job_zone <= 2:
            return f"Fill {num_gaps} gaps via certificate/apprenticeship (3-12mo)"
        elif job_zone == 3:
            return f"Fill {num_gaps} gaps via bootcamp/certificate (3-18mo)"
        else:
            return f"Fill {num_gaps} gaps via associate's/extended bootcamp (1-2yr)"

    # long_reskill
    if job_zone <= 2:
        return f"Significant reskilling ({num_gaps} gaps): vocational (1-2yr)"
    elif job_zone == 3:
        return f"Significant reskilling ({num_gaps} gaps): associate's (1-3yr)"
    else:
        return f"Significant reskilling ({num_gaps} gaps): bachelor's (2-4yr)"
```

#### 7. Top Skill Gaps

Returns the N most important skills where user has gaps.

**Algorithm**:
```python
gaps = []
for skill in occupation_skills:
    if user_capability[skill] <= 0.25:  # Gap threshold
        gaps.append({
            "skill": skill,
            "importance": skill.importance,
            "weight": skill.weight,
            "user_capability": user_capability[skill]
        })

# Sort by weight (normalized importance) descending
gaps.sort(key=lambda x: x["weight"], reverse=True)
return gaps[:N]  # Top N
```

**Default N**: 10 (configurable)

---

## Model v2: Calibration Layer

### Purpose

Learn from user feedback to improve ranking and prediction of successful transitions.

### Architecture

```
User Features + Occupation Features
         ↓
  [Feature Extraction]
         ↓
   Baseline Scores (v1)
         ↓
  [Calibration Model]  ← Trained on feedback
         ↓
  P(success | user, occupation)
         ↓
  Re-ranked Recommendations
```

### Feature Set

**Baseline Features** (from Model v1):
1. `match_score`: Baseline match score (0-100)
2. `gap_severity`: Baseline gap severity (0-100)

**Job Transition Features**:
3. `job_zone_diff`: Target job zone - Current job zone
4. `target_job_zone`: Target occupation's job zone (1-5)

**Skill Gap Features**:
5. `num_missing_skills`: Count of skills with capability ≤ 0.25
6. `sum_missing_weights`: Sum of importance weights for gap skills

**User Confidence Features**:
7. `mean_rating`: Average of user's skill ratings
8. `rating_variance`: Variance of user's skill ratings
9. `num_rated_skills`: Number of skills user has rated

**Rationale**:
- **Baseline scores** provide strong signal from deterministic logic
- **Job zone** captures education/experience requirements
- **Skill gaps** quantify magnitude of reskilling needed
- **User confidence** captures self-assessment patterns (over/under-confidence)

### Labels

Derived from user feedback actions:

| Action Type | Label | Weight | Rationale |
|-------------|-------|--------|-----------|
| `interview` | 1 (positive) | High | Strong signal - got to interview stage |
| `offer` | 1 (positive) | High | Strongest signal - received offer |
| `apply` | 1 (positive) | Medium | Moderate signal - user applied |
| `hide` | 0 (negative) | High | Strong negative - explicitly rejected |
| `click` | - | - | Ambiguous - not used for training |
| `save` | - | - | Ambiguous - not used for training |

**Training Data Extraction**:
```sql
SELECT
    uf.action_type,
    ro.score_json,  -- Contains baseline features
    re.user_id,
    re.current_onet_code
FROM user_feedback uf
JOIN recommendation_event re ON uf.event_id = re.id
JOIN recommended_occupation ro ON uf.target_onet_code = ro.target_onet_code
WHERE uf.action_type IN ('interview', 'offer', 'apply', 'hide')
```

### Model Choice

**Initial**: Logistic Regression
- Simple, interpretable
- Works with small sample sizes (50+)
- Provides probability calibration
- Fast training and inference

**Future**: Gradient Boosting (XGBoost, LightGBM)
- Better at capturing non-linear interactions
- Requires more data (1000+ samples)
- Can incorporate feature importance

**Formula** (Logistic Regression):
```
P(success | x) = 1 / (1 + exp(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))

where x = [match_score, gap_severity, job_zone_diff, ...]
```

### Training Process

**Minimum Samples**: 50 (configurable via `MODEL_TRAINING_MIN_SAMPLES`)

**Training Pipeline**:
1. Extract feedback from database
2. Convert actions to labels
3. Extract features from score_json
4. Split train/test (80/20)
5. Standardize features (zero mean, unit variance)
6. Train logistic regression
7. Evaluate on test set (accuracy, ROC-AUC)
8. Save model artifact and register in model_registry table

**Evaluation Metrics**:
- **Accuracy**: Overall correctness
- **ROC-AUC**: Ranking quality (more important than accuracy)
- **Positive Rate**: % of positive labels (check for class imbalance)

**Frequency**: Daily at 2 AM (configurable via `PERIODIC_TRAINING_CRON`)

### Inference

**Re-ranking**:
1. Generate baseline recommendations (Model v1)
2. For each recommendation, predict P(success)
3. Re-rank by calibrated probability
4. Return top-N per bucket

**Calibrated Score**:
```
calibrated_score = 100 × P(success | user, occupation)
```

### Exploration Policy

**Epsilon-Greedy Strategy**:

```python
epsilon = 0.1  # Configurable

if random() < epsilon:
    # EXPLORE: Show 1-2 recommendations near decision boundary
    select_exploration_items()
else:
    # EXPLOIT: Show top recommendations by calibrated score
    select_top_items()
```

**Purpose**: Balance between:
- **Exploitation**: Show best recommendations based on current model
- **Exploration**: Show uncertain recommendations to gather more data

**Tracking**: `is_exploration` flag stored with each recommendation

**Benefits**:
- Accelerates learning
- Prevents filter bubble
- Discovers unexpected successful transitions

---

## Decision Flow

Guidance provided to users based on recommendation distribution.

**Logic**:

```python
if ready_now_count > 0:
    guidance = "Start applying to 'Ready Now' jobs while exploring trainable options."
elif trainable_count > 0:
    guidance = "Focus on training for 'Trainable' jobs. Consider part-time learning."
else:
    guidance = "Plan for long-term reskilling. Consider formal education or bootcamps."
```

**Display**: Shown prominently in recommendation response to guide user action.

---

## Thresholds and Configuration

All thresholds are configurable via environment variables to allow tuning without code changes.

### Baseline Model Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `READY_NOW_MATCH_THRESHOLD` | 75.0 | Min match score for READY_NOW |
| `READY_NOW_GAP_THRESHOLD` | 25.0 | Max gap severity for READY_NOW |
| `TRAINABLE_MATCH_MIN` | 50.0 | Min match score for TRAINABLE |
| `TRAINABLE_MATCH_MAX` | 74.0 | Max match score for TRAINABLE |
| `TRAINABLE_GAP_MIN` | 26.0 | Min gap severity for TRAINABLE |
| `TRAINABLE_GAP_MAX` | 55.0 | Max gap severity for TRAINABLE |

### Calibration Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_VERSION` | v1_baseline | Current model version identifier |
| `MODEL_TRAINING_MIN_SAMPLES` | 50 | Min feedback samples to train |
| `EXPLORATION_EPSILON` | 0.1 | Exploration probability (0-1) |
| `PERIODIC_TRAINING_CRON` | 0 2 * * * | Daily at 2 AM |

### Tuning Recommendations

**To make MORE jobs appear as READY_NOW**:
- Decrease `READY_NOW_MATCH_THRESHOLD` (e.g., 70)
- Increase `READY_NOW_GAP_THRESHOLD` (e.g., 30)

**To make FEWER jobs appear as READY_NOW** (more conservative):
- Increase `READY_NOW_MATCH_THRESHOLD` (e.g., 80)
- Decrease `READY_NOW_GAP_THRESHOLD` (e.g., 20)

**To broaden TRAINABLE bucket**:
- Decrease `TRAINABLE_MATCH_MIN` (e.g., 40)
- Increase `TRAINABLE_MATCH_MAX` (e.g., 79)

---

## Evolution Strategy

### Phase 1: Launch (Current)
- **Model**: v1 Baseline only
- **Data Collection**: All user interactions logged
- **Focus**: Establish baseline quality, gather feedback

### Phase 2: Calibration (50+ feedback samples)
- **Model**: v1 + v2 Calibration
- **Training**: Automatic daily
- **A/B Testing**: Compare v1 vs v2 performance
- **Monitoring**: Track ROC-AUC, feedback distribution

### Phase 3: Advanced Learning (500+ samples)
- **Explore**: Gradient boosting models
- **Features**: Add user trajectory features (past applications, outcomes)
- **Personalization**: User-specific models or embeddings

### Phase 4: Future Enhancements
- **Embeddings**: Skill embeddings for similarity
- **LLM Integration**: Career advice generation
- **Multi-armed Bandits**: More sophisticated exploration
- **Contextual Features**: Market demand, salary, location

---

## Monitoring and Metrics

### Key Metrics to Track

**Baseline Quality**:
- Distribution of match_scores
- Distribution of gap_severity
- % of users with READY_NOW options
- % of users with no TRAINABLE options

**Calibration Performance**:
- ROC-AUC (ranking quality)
- Calibration curve (predicted prob vs actual rate)
- Feature importance (which features matter most)

**User Behavior**:
- Click-through rate by bucket
- Application rate by match_score range
- Interview rate by calibrated_score
- Time-to-action distribution

**System Health**:
- Recommendation latency (target: < 3 seconds)
- Cache hit rate for occupations
- Model training success rate
- Feedback volume (actions per day)

### Quality Assurance

**Unit Tests**:
- Scoring functions with known inputs/outputs
- Bucket assignment at boundary cases
- Feature extraction logic

**Integration Tests**:
- End-to-end recommendation flow
- Feedback loop simulation
- Model training pipeline

**Manual Review**:
- Sample recommendations for known roles
- Edge cases (very high/low skills)
- Cross-domain transitions (e.g., teacher → developer)

---

## Limitations and Future Work

### Current Limitations

1. **No skill similarity**: Treats all skills as independent
   - Future: Use skill embeddings or taxonomy
2. **No market data**: Doesn't consider job availability, salary
   - Future: Integrate labor market data
3. **No temporal dynamics**: Doesn't model skill decay or learning curves
   - Future: Add time-based features
4. **Simple exploration**: Epsilon-greedy is basic
   - Future: Thompson sampling, contextual bandits
5. **No user segmentation**: One model for all users
   - Future: Personalized models or user clusters

### Research Questions

1. **Optimal thresholds**: Are default thresholds appropriate across all domains?
2. **Feature engineering**: What additional features improve calibration?
3. **Label quality**: How to handle ambiguous feedback (clicks, saves)?
4. **Cold start**: How to recommend for users with few ratings?
5. **Bias**: Does model amplify biases in feedback data?

---

## References

- O*NET Database: https://www.onetcenter.org/
- O*NET Web Services: https://services.onetcenter.org/
- Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- ROC-AUC: https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics

---

**Document Version**: 1.0
**Last Updated**: 2026-01-19
**Author**: SkillSprout Team
