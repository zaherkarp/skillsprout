# SkillSprout Developer Guide

## Local Development Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 13+ (or use SQLite for quick testing)
- Redis 6+
- O*NET Web Services credentials (optional -- demo mode works without them)

### Option A: Docker (Recommended)

Docker is the fastest path to a working environment:

```bash
# Start everything: PostgreSQL, Redis, FastAPI, Celery
make dev
# or: docker-compose up -d

# Verify services are running
curl http://localhost:8000/api/v1/health
```

This runs database migrations, seeds demo data, and starts all services. The app is available at `http://localhost:8000`.

### Option B: Local Python

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env -- at minimum:
#   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/skillsprout
#   DATABASE_URL_SYNC=postgresql+psycopg2://user:pass@localhost:5432/skillsprout
#   REDIS_URL=redis://localhost:6379/0
#   DEMO_MODE=true

# 4. Create database and run migrations
createdb skillsprout
alembic upgrade head

# 5. Seed demo data
python scripts/seed_demo.py

# 6. Start services (each in its own terminal)
uvicorn app.main:app --reload                              # API server
celery -A app.tasks.celery_app worker --loglevel=info      # Background worker
celery -A app.tasks.celery_app beat --loglevel=info        # Periodic scheduler
```

### Verify Your Setup

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Create a user, set occupation, rate skills, get recommendations
curl -X POST http://localhost:8000/api/v1/user/profile -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8000/api/v1/user/1/current-occupation -H "Content-Type: application/json" -d '{"onet_code": "15-1252.00"}'
curl -X POST http://localhost:8000/api/v1/user/1/skills/ratings -H "Content-Type: application/json" -d '{"ratings": [{"element_id": "2.B.1.g", "rating_0_4": 4}, {"element_id": "2.B.8.a", "rating_0_4": 3}]}'
curl -X POST http://localhost:8000/api/v1/user/1/recommendations -H "Content-Type: application/json" -d '{"limit_per_bucket": 5}'
```

---

## Architecture Overview

### System Components

```
                                    +-------------------+
                                    |   Web Browser /   |
                                    |   API Client      |
                                    +--------+----------+
                                             |
                                             v
+-------------------------------------------+-------------------------------------------+
|                          FastAPI Application (app/main.py)                              |
|                                                                                        |
|  +------------------+    +-------------------+    +------------------+                  |
|  | API Endpoints    |    | Pydantic Schemas  |    | Jinja2 Templates |                  |
|  | (app/api/)       |    | (app/schemas/)    |    | (templates/)     |                  |
|  +--------+---------+    +-------------------+    +------------------+                  |
|           |                                                                            |
|  +--------v---------+    +-------------------+                                         |
|  | Services          |    | ML Scoring        |                                         |
|  | (app/services/)   |    | (app/ml/)         |                                         |
|  | - O*NET client    |    | - BaselineScorer  |                                         |
|  +--------+----------+    | - CalibrationModel|                                         |
|           |               +--------+----------+                                         |
+-----------|------------------------|-------------------------------------------------+
            |                        |
    +-------v--------+     +--------v----------+     +--------------------------+
    | O*NET Web API  |     | PostgreSQL         |     | Redis                    |
    | (external)     |     | - Occupations      |     | - Celery broker/backend  |
    +----------------+     | - Skills           |     +-----------+--------------+
                           | - User profiles    |                 |
                           | - Feedback         |     +-----------v--------------+
                           | - Model registry   |     | Celery Worker            |
                           +--------------------+     | (app/tasks/)             |
                                                      | - Cache warming          |
                                                      | - Model training         |
                                                      | - Transition graph build  |
                                                      +--------------------------+
```

### Request Flow: Getting Recommendations

Here is the complete path from API request to response for `POST /api/v1/user/{user_id}/recommendations`:

```
1. Request arrives at endpoints.py::get_recommendations()
2. Verify user exists (UserProfile table)
3. Load active current occupation (UserCurrentOccupation table, is_active=True)
4. Load user skill ratings (UserSkillRating table -> dict of element_id: rating)
5. Fetch ALL cached occupations with skills (Occupation + OccupationSkill + Skill via joinedload)
6. For each occupation (excluding user's current):
   a. Build skill list: [{element_id, skill_name, importance, level}, ...]
   b. Call BaselineScorer.score_occupation(onet_code, title, skills, user_ratings, job_zones)
      i.   _calculate_scores(): compute weighted match and identify gaps
      ii.  _assign_bucket(): apply threshold logic (READY_NOW / TRAINABLE / LONG_RESKILL)
      iii. _generate_training_suggestion(): heuristic based on bucket + job zone
      iv.  _generate_explanation(): human-readable text
   c. Collect (occupation, OccupationScore) pair
7. Create RecommendationEvent record
8. Group scored occupations by bucket, sort by match_score descending
9. Persist RecommendedOccupation records (up to limit_per_bucket per bucket)
10. Build response with bucket labels, decision guidance, and occupation details
11. Return RecommendationResponse
```

### Key Data Models

| Model | Table | Purpose |
|-------|-------|---------|
| `Occupation` | `occupation` | Cached O*NET occupation metadata |
| `Skill` | `skill` | O*NET skill definitions |
| `OccupationSkill` | `occupation_skill` | Skills required by each occupation (importance, level) |
| `UserProfile` | `user_profile` | User accounts |
| `UserCurrentOccupation` | `user_current_occupation` | User's current job (one active at a time) |
| `UserSkillRating` | `user_skill_rating` | User's self-assessed skill ratings (0-4) |
| `RecommendationEvent` | `recommendation_event` | One record per recommendation request |
| `RecommendedOccupation` | `recommended_occupation` | Individual occupations in a recommendation set |
| `UserFeedback` | `user_feedback` | User actions on recommendations |
| `ModelRegistry` | `model_registry` | Trained model versions and artifacts |

---

## How to Add a New Feature to the Scoring Pipeline

### Example: Adding a "years of experience" feature

**Step 1: Add the data model.**

Edit `app/models/models.py` to add the field:

```python
class UserProfile(Base):
    # ... existing fields ...
    years_experience = Column(Integer, nullable=True)
```

Create a migration:

```bash
alembic revision --autogenerate -m "Add years_experience to user_profile"
alembic upgrade head
```

**Step 2: Add the API schema.**

Edit `app/schemas/schemas.py`:

```python
class UserProfileCreate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None
    years_experience: Optional[int] = None
```

**Step 3: Add to the calibration feature set.**

Edit `app/ml/calibration.py` to include the feature in `CalibrationFeatures`:

```python
@dataclass
class CalibrationFeatures:
    # ... existing fields ...
    years_experience: int
```

Update `_features_to_array()` to include it in the numpy array, and update `feature_names` in `train()`.

**Step 4: Wire it through the endpoint.**

In `app/api/endpoints.py`, pass the new feature through to the calibration model when extracting features.

**Step 5: Add tests.**

Add test cases in `tests/unit/test_scoring.py` or create a new test file. Test at least:
- Default behavior when the field is absent (backward compatibility).
- Boundary values (0 years, 30 years).
- Impact on scoring (if the feature should change scores, verify it does).

**Step 6: Update training data extraction.**

In `app/tasks/tasks.py`, update `train_calibration_model_task()` to extract the new feature from the database when building training data.

---

## Running the Bias Audit

The bias audit analyzes recommendation distributions for systematic disparities. It requires some accumulated recommendation and feedback data.

### Running Manually

```bash
# The bias audit module is in ml/bias_audit/
# Run the audit script (when implemented):
python -m ml.bias_audit.run_audit

# Or trigger via Celery:
celery -A app.tasks.celery_app call ml.bias_audit.audit_task
```

### What It Checks

1. **Bucket distribution by occupation category:** Are users from certain current occupations disproportionately assigned to LONG_RESKILL?
2. **Score distribution analysis:** Are match_score and gap_severity distributions consistent across user segments?
3. **Demographic parity (when data available):** If user demographics are collected, test whether bucket assignments are independent of protected attributes.

### Interpreting Results

- **Disparity ratio:** `min_group_rate / max_group_rate`. Values above 0.8 are acceptable. Values between 0.6-0.8 require investigation. Values below 0.6 block model promotion.
- Look at the raw numbers, not just the ratios. A disparity ratio of 0.5 based on 4 users in one group is not statistically meaningful.

See [ADR-006](adr/ADR-006-bias-audit-approach.md) for the full rationale.

---

## Running Evaluation

### Unit Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit/ -v

# Run only scoring tests
pytest tests/unit/test_scoring.py -v

# Run with coverage
pytest --cov=app --cov-report=html
```

### Integration Tests

```bash
# Run integration tests (requires database)
pytest tests/integration/ -v

# Run in Docker (recommended for integration tests)
make test
# or: docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Model Evaluation

The calibration model's training task automatically computes evaluation metrics on a held-out test set:

- **Accuracy:** Overall correctness of binary predictions.
- **ROC-AUC:** Ranking quality (more important than accuracy for recommendations).
- **Positive rate:** Fraction of positive labels in training data (monitors class imbalance).

These metrics are stored in `model_registry.metrics_json` and available via `GET /api/v1/model/status`.

### Smoke Tests

```bash
# End-to-end smoke test against a running instance
./scripts/smoke_test.sh

# Full E2E test suite
./scripts/test_e2e.sh
```

---

## Data Enrichment Pipeline

The enrichment pipeline automatically discovers new occupations, fetches their skills from O\*NET, scores all QA personas against them, and persists everything to a JSON registry file (`app/data/occupation_registry.json`).

### How It Works

1. **Startup hook**: The pipeline runs on app startup (`app/main.py` lifespan). In demo mode it skips O\*NET discovery but still seeds static data and scores.
2. **CLI**: `python -m app.services.enrichment_pipeline [--skip-discovery]`
3. **API**: `POST /api/v1/enrichment/run`

### Registry File

The registry is a JSON file that accumulates data across runs:

```json
{
  "version": 2,
  "occupations": {
    "15-1251.00": {
      "title": "Computer Programmers",
      "skills": [...],
      "ai_exposure": { "theoretical_exposure": 0.94, "observed_exposure": 0.75 },
      "bls_projections": { "projected_growth_pct": -10.6, "outlook": "declining" },
      "scoring_results": { "derek_programmer": { "match_score": 85.2, "bucket": "ready_now" } }
    }
  },
  "run_log": [...]
}
```

### Using the Registry in Code

```python
from app.services.occupation_registry import OccupationRegistry

registry = OccupationRegistry()

# Read data
entry = registry.get("15-1251.00")
skills = registry.get_skills("15-1251.00")
exposure = registry.get_exposure("15-1251.00")

# Write data
registry.upsert_occupation("99-0001.00", "New Occupation", source="manual")
registry.set_skills("99-0001.00", [...])
registry.save()

# Summary
print(registry.summary())
```

### Data Fallback Chain

`get_exposure()` and `get_projections()` check the registry first, then fall back to the static Python dictionaries. This means dynamically discovered data is immediately available without code changes.

### Adding New QA Personas

Add personas to `tests/test_anthropic_report_personas.py` following the existing pattern:

```python
PERSONA_NEW = {
    "name": "Name - Role, X% AI exposure",
    "current_occupation": "XX-XXXX.00",
    "skill_ratings": {
        "2.B.8.a": 3,  # Critical Thinking - Advanced
        # ... more skills
    },
    "expected_transition_context": "description of transition scenario",
    "budget": 1000,
    "timeline_months": 12,
}
```

Then add the persona to `ALL_PERSONAS` and write scenario tests in the appropriate test class.

---

## Deployment

### Environment Configuration

Required environment variables for production:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL async connection string |
| `DATABASE_URL_SYNC` | PostgreSQL sync connection string (for Celery) |
| `REDIS_URL` | Redis connection string |
| `ONET_USERNAME` | O*NET Web Services username |
| `ONET_PASSWORD` | O*NET Web Services password |
| `DEMO_MODE` | Set to `false` for production |
| `ENV` | `production` |
| `DEBUG` | `false` |

See `.env.example` for the complete list with defaults.

### Deployment Checklist

1. **Database:** Run `alembic upgrade head` to apply all migrations.
2. **Cache warming:** Trigger `warm_occupation_cache` with the desired occupation codes. The demo set includes 8 occupations; production should include all relevant O*NET codes.
3. **Celery worker:** Start at least one worker for background tasks.
4. **Celery beat:** Start the scheduler for periodic model training (daily at 2 AM by default).
5. **Health check:** Verify `GET /api/v1/health` returns `{"status": "healthy"}`.
6. **CORS:** Update `app/main.py` CORS settings to restrict allowed origins.
7. **HTTPS:** Ensure the API is served over HTTPS in production.
8. **Monitoring:** Set up log aggregation and alerting for error rates and latency.

### Docker Production Build

```bash
# Build production image
docker build -t skillsprout:latest .

# Run with docker-compose
docker-compose -f docker-compose.yml up -d
```

---

## Project Structure Reference

```
skillsprout/
+-- app/
|   +-- api/endpoints.py          # FastAPI route handlers
|   +-- core/config.py            # Settings (env vars, thresholds)
|   +-- db/session.py             # SQLAlchemy engine and session
|   +-- models/models.py          # ORM models (Occupation, User, Feedback, etc.)
|   +-- schemas/schemas.py        # Pydantic request/response schemas
|   +-- services/onet_client.py           # O*NET API client + mock client (14+ occupations)
|   +-- services/occupation_registry.py  # JSON-backed persistent occupation store
|   +-- services/enrichment_pipeline.py  # Auto-discovery, scoring, and persistence
|   +-- services/enrichment_api.py       # REST endpoints for enrichment
|   +-- ml/scoring.py             # BaselineScorer (Model v1)
|   +-- ml/calibration.py         # CalibrationModel (Model v2)
|   +-- tasks/celery_app.py       # Celery configuration
|   +-- tasks/tasks.py            # Background tasks (cache, training)
|   +-- main.py                   # FastAPI application entry point
+-- ml/
|   +-- transition_graph/         # Directed graph of career transitions
|   +-- bias_audit/               # Fairness testing
|   +-- cold_start/               # Cold start strategies
|   +-- evaluation/               # Model evaluation utilities
|   +-- features/                 # Feature engineering
+-- alembic/                      # Database migration scripts
+-- tests/
|   +-- unit/                     # Unit tests
|   +-- integration/              # Integration tests
|   +-- conftest.py               # Shared test fixtures
|   +-- test_anthropic_report_personas.py  # 10 Anthropic report QA personas (68 tests)
|   +-- test_enrichment_pipeline.py        # Registry + pipeline tests (31 tests)
+-- scripts/                      # Utility scripts (seed, smoke test)
+-- templates/                    # Jinja2 HTML templates
+-- static/                       # CSS, JS assets
+-- docs/                         # Documentation and ADRs
+-- requirements.txt              # Python dependencies
+-- docker-compose.yml            # Docker services
+-- Makefile                      # Common commands
```

---

## Key Design Decisions

For the reasoning behind major architectural choices, see the Architecture Decision Records:

- [ADR-001: Two-Stage Scoring](adr/ADR-001-two-stage-scoring.md) -- Why deterministic baseline + learned calibration
- [ADR-002: Transition Features](adr/ADR-002-transition-features.md) -- Why transition-aware features
- [ADR-003: Cold Start Strategy](adr/ADR-003-cold-start-strategy.md) -- Why three-tier progressive disclosure
- [ADR-004: Calibration Over Ranking](adr/ADR-004-calibration-over-ranking.md) -- Why logistic regression now
- [ADR-005: Feedback Signal Hierarchy](adr/ADR-005-feedback-signal-hierarchy.md) -- Why not all signals are equal
- [ADR-006: Bias Audit Approach](adr/ADR-006-bias-audit-approach.md) -- Why demographic parity testing
