# SkillSprout

A production-minded MVP web application that uses O*NET occupation skill data to help users discover job transition opportunities based on their current skills and experience.

## Features

- **Smart Job Matching**: Uses O*NET skill data to match users with transition opportunities
- **Three-Tier Recommendations**:
  - **Ready Now**: Jobs you can apply to immediately
  - **Trainable**: Jobs within reach with focused training (3-18 months)
  - **Long-term Reskill**: Jobs requiring significant reskilling (1-4+ years)
- **Skill Gap Analysis**: Identifies specific skills to develop for target roles
- **Training Path Suggestions**: Job-zone based recommendations for training approaches
- **Learning System**: Designed to improve over time using user feedback and outcomes
- **Demo Mode**: Works without O*NET credentials using mock data

## Tech Stack

### Backend
- **FastAPI**: Modern async web framework
- **PostgreSQL**: Primary data store (with SQLAlchemy 2.0 ORM)
- **Redis + Celery**: Background task processing and caching
- **Alembic**: Database migrations
- **Pydantic**: Request/response validation

### ML & Data
- **O*NET Web Services**: Occupation and skill data source
- **scikit-learn**: Calibration model training
- **Baseline Scoring**: Deterministic skill matching algorithm
- **Learnable Calibration**: Logistic regression model trained on user feedback

### Testing
- **pytest**: Unit and integration tests
- **httpx-mock**: API mocking for tests

## Architecture Overview

```
┌─────────────────┐
│   FastAPI App   │  ← API endpoints + minimal UI
└────────┬────────┘
         │
    ┌────┴───────┐
    │            │
┌───▼──────┐  ┌──▼──────┐
│ DB       │  │ O*NET   │
│(Postgres)│  │ API     │
└──────────┘  └─────────┘
    │
┌───▼──────────┐
│ Celery Worker│  ← Cache warming, model training
└──────────────┘
    │
┌───▼────┐
│ Redis  │
└────────┘
```

## Prerequisites

- Python 3.10+
- PostgreSQL 13+
- Redis 6+
- O*NET Web Services credentials (optional - demo mode available)

## Quick Start

### 🐳 Docker (Recommended for Testing)

The fastest way to get started is using Docker:

```bash
# One command to start everything
make dev

# Or without Make
docker-compose up -d
```

This will:
- Start PostgreSQL, Redis, FastAPI, and Celery
- Run database migrations
- Seed demo data
- Be ready at http://localhost:8000

**See [DOCKER_TESTING.md](DOCKER_TESTING.md) for complete Docker testing guide.**

### 💻 Local Development

### 1. Clone and Setup

```bash
git clone <repository-url>
cd skillsprout

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
# Minimum required for demo mode:
DATABASE_URL=postgresql+asyncpg://skillsprout:skillsprout@localhost:5432/skillsprout
DATABASE_URL_SYNC=postgresql+psycopg2://skillsprout:skillsprout@localhost:5432/skillsprout
REDIS_URL=redis://localhost:6379/0
DEMO_MODE=true  # Set to false when you have O*NET credentials
```

### 3. Set Up Database

```bash
# Create database
createdb skillsprout

# Run migrations
alembic upgrade head
```

### 4. Seed Demo Data

```bash
# This caches occupations and skills for faster development
python scripts/seed_demo.py
```

### 5. Start Services

```bash
# Terminal 1: Start API server
uvicorn app.main:app --reload

# Terminal 2: Start Celery worker
celery -A app.tasks.celery_app worker --loglevel=info

# Terminal 3 (optional): Start Celery beat for periodic tasks
celery -A app.tasks.celery_app beat --loglevel=info
```

### 6. Access the Application

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/api/v1/docs
- **Health Check**: http://localhost:8000/api/v1/health

## API Usage

### Complete Workflow Example

```bash
# 1. Create user profile
curl -X POST http://localhost:8000/api/v1/user/profile \
  -H "Content-Type: application/json" \
  -d '{}'

# Response: {"id": 1, "created_at": "2026-01-19T..."}

# 2. Search for occupations
curl "http://localhost:8000/api/v1/occupations/search?q=software"

# 3. Set current occupation
curl -X POST http://localhost:8000/api/v1/user/1/current-occupation \
  -H "Content-Type: application/json" \
  -d '{"onet_code": "15-1252.00"}'

# 4. Get occupation skills
curl "http://localhost:8000/api/v1/occupations/15-1252.00/skills"

# 5. Rate your skills
curl -X POST http://localhost:8000/api/v1/user/1/skills/ratings \
  -H "Content-Type: application/json" \
  -d '{
    "ratings": [
      {"element_id": "2.B.1.a", "rating_0_4": 3},
      {"element_id": "2.B.8.a", "rating_0_4": 4},
      {"element_id": "2.B.1.g", "rating_0_4": 4}
    ]
  }'

# 6. Get recommendations
curl -X POST http://localhost:8000/api/v1/user/1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "limit_per_bucket": 10,
    "use_calibration": false,
    "enable_exploration": false
  }'

# 7. Submit feedback
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": 1,
    "target_onet_code": "15-1299.08",
    "action_type": "click"
  }'

# 8. Check model status
curl "http://localhost:8000/api/v1/model/status"
```

## Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/occupations/search` | Search occupations |
| GET | `/api/v1/occupations/{code}` | Get occupation details |
| GET | `/api/v1/occupations/{code}/skills` | Get occupation skills |
| POST | `/api/v1/user/profile` | Create user profile |
| POST | `/api/v1/user/{id}/current-occupation` | Set current occupation |
| POST | `/api/v1/user/{id}/skills/ratings` | Update skill ratings |
| POST | `/api/v1/user/{id}/recommendations` | Get recommendations |
| POST | `/api/v1/feedback` | Submit feedback |
| GET | `/api/v1/model/status` | Model status and metrics |

## Scoring Model

### Baseline Model (v1)

The baseline model computes:

1. **Match Score** (0-100): Weighted overlap between user capabilities and job requirements
   - Uses skill importance as weights
   - User capability mapped from ratings: 0→0.0, 1→0.25, 2→0.5, 3→0.75, 4→1.0
   - Formula: `match_score = 100 * Σ(weight_i * capability_i) / Σ(weight_i)`

2. **Gap Severity** (0-100): Weighted importance of skills with low capability (≤0.25)
   - Formula: `gap_severity = 100 * Σ(weight_i for skills with capability ≤ 0.25) / Σ(weight_i)`

3. **Bucket Assignment**:
   - **READY_NOW**: match_score ≥ 75 AND gap_severity ≤ 25
   - **TRAINABLE**: match_score 50-74 OR gap_severity 26-55
   - **LONG_RESKILL**: everything else

4. **Training Suggestions**: Based on job zone and number of gaps
   - Job Zone 1-2: Certificate/apprenticeship (3-12 months)
   - Job Zone 3: Bootcamp/certificate (3-18 months)
   - Job Zone 4-5: Extended training/degree (1-4 years)

### Calibration Model (v2 - Framework)

The system includes a learnable calibration layer:

- **Model**: Logistic regression (ready to upgrade to gradient boosting)
- **Features**: match_score, gap_severity, job_zone_diff, skill gap metrics, user confidence
- **Labels**: Derived from user feedback
  - Positive: interview, offer, apply
  - Negative: hide
- **Training**: Periodic (daily at 2 AM) or on-demand via Celery
- **Exploration**: Epsilon-greedy policy for online learning

See [MODELING_NOTES.md](MODELING_NOTES.md) for detailed formulas and implementation notes.

## Background Tasks

### Cache Warming

```bash
# Warm cache for specific occupations
celery -A app.tasks.celery_app call app.tasks.tasks.warm_occupation_cache --args='[["15-1252.00", "15-1299.08"]]'

# Search and cache
celery -A app.tasks.celery_app call app.tasks.tasks.search_and_cache_occupations --args='[["software", "web developer", "data"]]'
```

### Model Training

```bash
# Train calibration model manually
celery -A app.tasks.celery_app call app.tasks.tasks.train_calibration_model_task
```

The training task runs automatically daily at 2 AM (configurable via `PERIODIC_TRAINING_CRON`).

## Testing

### Docker Testing (Recommended)

Full isolated testing environment with all services:

```bash
# Using Make (easiest)
make test

# Using test script (with cleanup)
./scripts/run_tests.sh

# Run E2E tests against running environment
make up
./scripts/test_e2e.sh

# With Docker Compose directly
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

**Benefits:**
- Isolated test database and Redis
- No local Python/PostgreSQL/Redis needed
- Consistent across all environments
- Perfect for CI/CD

**See [DOCKER_TESTING.md](DOCKER_TESTING.md) for comprehensive testing guide.**

### Local Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_scoring.py -v

# Run integration tests only
pytest tests/integration/ -v
```

## Development

### Project Structure

```
skillsprout/
├── app/
│   ├── api/            # API endpoints
│   ├── core/           # Configuration
│   ├── db/             # Database session management
│   ├── models/         # SQLAlchemy models
│   ├── schemas/        # Pydantic schemas
│   ├── services/       # O*NET client, external services
│   ├── ml/             # Scoring and calibration models
│   ├── tasks/          # Celery tasks
│   └── main.py         # FastAPI application
├── alembic/            # Database migrations
├── scripts/            # Utility scripts
├── templates/          # HTML templates
├── static/             # CSS, JS
├── tests/              # Test suite
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── models/             # Trained model artifacts (gitignored)
├── requirements.txt    # Python dependencies
└── README.md
```

### Creating Migrations

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Add new field"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1
```

### Adding New Occupations

```python
# In Python shell or script
from app.tasks.tasks import warm_occupation_cache

# Add specific occupations
warm_occupation_cache(["19-1021.00", "29-1141.00"])
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Async PostgreSQL connection string | Required |
| `DATABASE_URL_SYNC` | Sync PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `ONET_USERNAME` | O*NET Web Services username | - |
| `ONET_PASSWORD` | O*NET Web Services password | - |
| `DEMO_MODE` | Use mock O*NET data | `false` |
| `MODEL_VERSION` | Current model version | `v1_baseline` |
| `READY_NOW_MATCH_THRESHOLD` | Min match score for READY_NOW | `75.0` |
| `READY_NOW_GAP_THRESHOLD` | Max gap severity for READY_NOW | `25.0` |
| `MODEL_TRAINING_MIN_SAMPLES` | Min samples to train calibration | `50` |
| `EXPLORATION_EPSILON` | Exploration probability | `0.1` |

See `.env.example` for full list.

## O*NET Web Services

To use real O*NET data:

1. Register at https://services.onetcenter.org/
2. Obtain username and password
3. Set in `.env`:
   ```
   ONET_USERNAME=your_username
   ONET_PASSWORD=your_password
   DEMO_MODE=false
   ```

Without credentials, the app runs in demo mode with 3 mock occupations.

## Production Considerations

### Before Deploying

1. **Security**:
   - Set strong database passwords
   - Use HTTPS for API
   - Enable CORS restrictions (update `app/main.py`)
   - Add authentication/authorization

2. **Database**:
   - Use connection pooling
   - Set up regular backups
   - Monitor query performance

3. **Caching**:
   - Use Redis persistence
   - Consider Redis Cluster for scale

4. **Monitoring**:
   - Add logging aggregation (e.g., ELK, Datadog)
   - Set up alerting for failures
   - Monitor model performance metrics

5. **Scaling**:
   - Use multiple Celery workers
   - Scale API with load balancer
   - Consider read replicas for database

---

## Hackathon Sprint Summary (February 2026)

### What Happened

A 5-day virtual hackathon sprint executed by a single AI engineer (Claude, Opus 4.6) working from a structured plan designed for a 3-person team (ML engineer, UX engineer, infrastructure engineer). All three roles were executed by one agent operating across 12 parallel work streams. The sprint produced **92 new files** with **~30,600 lines of code** on top of the existing MVP foundation.

### Team Roles (All Executed by Claude / Opus 4.6)

| Role | Focus Area | Codename |
|------|-----------|----------|
| E1 | Backend / ML Pipeline | `ml-eng` |
| E2 | Frontend / UX / Accessibility | `ux-eng` |
| E3 | Infrastructure / Data Integrity / DevOps | `infra-eng` |

### What Was Built

#### Day 1: Foundation & Audit

| Deliverable | Owner | Files | Status |
|------------|-------|-------|--------|
| ML Pipeline Audit | E1 | `docs/ml-audit.md` | Complete |
| Offline Evaluation Framework | E1 | `ml/evaluation/eval_framework.py`, `eval_runner.py`, `generate_synthetic_interactions.py` | Complete |
| UX & Accessibility Audit | E2 | `docs/ux-audit.md` | Complete |
| Skills Translator (NLP input) | E2 | `app/features/skills_translator/` (4 files) | Complete |
| Infrastructure & Security Audit | E3 | `docs/infra-audit.md` | Complete |
| Privacy Framework | E3 | `app/core/privacy/` (5 files: classification, retention, private mode, export, deletion) | Complete |

#### Day 2: Core Model Improvements

| Deliverable | Owner | Files | Status |
|------------|-------|-------|--------|
| Transition-Aware Features | E1 | `ml/features/transition_features.py` | Complete |
| Cold Start Strategy (3-tier) | E1 | `ml/cold_start/` (3 files: user, occupation, combination) | Complete |
| Bucket Explanation Engine | E2 | `app/features/explainability/` (4 files: explainer, thresholds, comparison, API) | Complete |
| Bias Audit Framework | E3 | `ml/bias_audit/` (3 files: framework, report, mitigations) | Complete |

#### Day 3: User Experience & Training

| Deliverable | Owner | Files | Status |
|------------|-------|-------|--------|
| Enhanced Event Tracking | E1 | `app/events/` (3 files: implicit signals, aggregator, pairwise preferences) | Complete |
| Training Path System | E2 | `app/features/training_paths/` (4 files: catalog, generator, filter, API) | Complete |
| Monitoring & Observability | E3 | `app/core/monitoring/` (4 files: health checks, metrics, alerting, logging) | Complete |

#### Day 4: Integration & User Journey

| Deliverable | Owner | Files | Status |
|------------|-------|-------|--------|
| Model Management | E1 | `ml/model_management/` (3 files: calibration monitor, registry, A/B testing) | Complete |
| User Profile & Progress | E2 | `app/features/user_profile/` (4 files: profile, saved occupations, progress, engagement) | Complete |
| Progressive Enhancement | E3 | `app/core/progressive/` (4 files: lightweight API, session resumption, offline export, accessibility) | Complete |

#### Day 5: Testing, Docs & Polish

| Deliverable | Owner | Files | Status |
|------------|-------|-------|--------|
| Integration Tests (5 personas) | E1 | `tests/integration/test_full_pipeline.py` | Complete |
| 14 Test Files (all modules) | All | `tests/test_*.py` (14 files) | Complete |
| Architecture Decision Records | E1 | `docs/adr/` (6 ADRs) | Complete |
| Developer Guide | E2 | `docs/developer-guide.md` | Complete |
| User Guide | E2 | `docs/user-guide.md` | Complete |
| Docker Hardening | E3 | Updated `Dockerfile` (multi-stage, HEALTHCHECK, non-root) | Complete |
| CI/CD Pipeline | E3 | Updated `.github/workflows/test.yml` (lint, security, unit, integration, build) | Complete |
| Open Source Prep | E3 | `LICENSE`, `CONTRIBUTING.md`, `.dockerignore`, updated `.env.example` | Complete |
| Transition Graph (stretch) | E1 | `ml/transition_graph/` (3 files: builder, queries, recommendations) | Complete |

### New API Endpoints Added

| Method | Endpoint | Module |
|--------|----------|--------|
| POST | `/api/v1/skills/translate` | Skills Translator |
| GET | `/api/v1/recommendations/{id}/explain` | Explainability |
| GET | `/api/v1/recommendations/compare?ids=...` | Comparison View |
| PATCH | `/api/v1/user/preferences` | Risk Tolerance |
| POST | `/api/v1/training-path/generate` | Training Paths |
| GET | `/api/v1/training-resources` | Training Catalog |
| POST | `/api/v1/events/heartbeat` | Dwell Time Tracking |
| POST | `/api/v1/events/explanation-view` | Engagement Tracking |
| POST/GET/PATCH | `/api/v1/profile` | User Profile |
| POST/GET/PATCH/DELETE | `/api/v1/saved-occupations` | Saved Occupations |
| POST | `/api/v1/skills/update` | Progress Tracking |
| GET | `/api/v1/progress/summary` | Return Engagement |
| GET | `/api/v1/user/data-export` | GDPR Export |
| DELETE | `/api/v1/user/data` | Account Deletion |
| POST | `/api/v1/session/export` | Session Token |
| POST | `/api/v1/session/import` | Session Resume |
| GET | `/api/v1/recommendations/export` | Offline Export |
| GET | `/health`, `/health/ready`, `/health/detailed` | Health Checks |
| GET | `/metrics` | Prometheus Metrics |

### New Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| `prometheus-client` | 0.20.0 | Application metrics and monitoring |
| `cryptography` | 42.0.2 | Fernet encryption for session tokens |
| `networkx` | 3.2.1 | Career transition graph analysis |
| `pyyaml` | 6.0.1 | YAML configuration support |

### Where Things Stand Now

**What works (code complete, needs integration testing):**
- Full scoring pipeline with transition-aware features
- Three-tier cold start handling (user/occupation/combination)
- Skills translator with rule-based + TF-IDF matching
- Bucket explanations with transparent thresholds
- Training catalog with 30+ real resources and constraint-aware path generation
- Privacy framework with private mode, data export, and deletion
- Comprehensive health checks and Prometheus metrics
- Session resumption via encrypted tokens
- User profile with progress tracking and re-scoring
- Bias audit framework with staleness and symmetry checks

**What is NOT wired into `app/main.py` yet:**
- The new API routers (skills translator, explainability, training paths, user profile, events, privacy, monitoring, progressive) exist as standalone FastAPI routers but have **not been mounted** on the main app. Each needs an `app.include_router(...)` call in `main.py`.
- The new ML modules (transition features, cold start, bias audit, model management, transition graph) are standalone and need integration into the scoring endpoint and Celery tasks.
- Database migrations for any new models (saved occupations, progress tracking, event tables) have **not been generated** via Alembic.

**What needs validation:**
- All 14 new test files need to be run against a live database to confirm they pass
- The training catalog data (URLs, costs, durations) needs human verification
- Bias audit findings need review by someone with domain expertise in workforce equity
- Cold start clustering (k=50) needs tuning with real O*NET data

### QA Recommendations

**Priority 1 — Integration wiring (blocks all feature testing):**
1. Mount all new routers in `app/main.py` — each module under `app/features/*/api.py`, `app/events/implicit_signals.py`, `app/core/privacy/`, `app/core/monitoring/`, and `app/core/progressive/` has a FastAPI `APIRouter` that needs `app.include_router(router, prefix=...)`.
2. Generate Alembic migrations for any new database models (check if new SQLAlchemy models were added to `app/models/models.py` or if the feature modules define their own).
3. Run `pytest tests/ -v` locally to identify import errors and fixture issues across the 14 new test files.

**Priority 2 — Functional testing per feature:**
4. **Skills Translator**: Test with real user input — "I managed a church food bank for 10 years" should produce meaningful skill matches. Check that the TF-IDF approach returns reasonable confidence scores.
5. **Explainability**: Verify that `bucket_explainer.py` references the actual thresholds from `threshold_config.py` and that the "what would change bucket" logic is accurate.
6. **Training Paths**: Confirm the constraint solver handles the zero-budget case (all gaps coverable by free resources) and the no-computer case (filter to in-person only).
7. **Privacy/Private Mode**: Verify the middleware actually prevents database writes when `X-Private-Mode: true` is set. This is a correctness-critical feature.
8. **Session Resumption**: Test the token roundtrip — export session, close browser, import on new device — and verify all state is preserved.

**Priority 3 — Data and model quality:**
9. **Bias Audit**: Run `ml/bias_audit/audit_framework.py` against the full O*NET occupation set (not just demo data) and review the report for occupation pairs with asymmetric scores.
10. **Cold Start Clusters**: The `OccupationClusterModel` uses k=50 which is a guess. Run `cluster_quality` silhouette analysis on real O*NET skill vectors to validate.
11. **Training Catalog**: The 30+ resources in `training_catalog.py` include URLs and cost data that need human verification — prices change, programs sunset.

**Priority 4 — Non-functional:**
12. **Health Checks**: Verify `/health/ready` actually fails when PostgreSQL or Redis is down (test by stopping Docker containers).
13. **Prometheus Metrics**: Confirm `/metrics` endpoint returns valid Prometheus format that can be scraped.
14. **Dockerfile**: Build the production target and verify the image size is under 500MB.
15. **CI/CD**: Run the updated GitHub Actions workflow end-to-end on a PR.

### Open Questions for QA

1. **Router mounting strategy**: Should all new endpoints be mounted under `/api/v1/` or should some get a `/api/v2/` prefix to signal they're new/experimental?
2. **Authentication**: The infra audit flagged NO authentication as CRITICAL. Should QA block on this, or is it acceptable for the hackathon demo?
3. **Private mode scope**: The current implementation prevents DB writes for user actions, but the scoring pipeline itself still reads from the DB (O*NET data, cached occupations). Is "no writes" sufficient for the privacy guarantee, or do we need "no reads of user-specific data" as well?
4. **Bias audit thresholds**: The framework flags occupations with >5-year-old O*NET data as stale. Is 5 years the right cutoff, or should it be more aggressive (3 years)?
5. **Training resource verification**: Who owns ongoing maintenance of the training catalog? Resources go stale fast — should there be an automated link-checking job?
6. **Test database seeding**: Integration tests need O*NET data in the DB. Should we commit a SQLite fixture file with the demo occupations, or rely on the `MockONetClient`?
7. **Feature flags**: The `.env.example` includes flags like `ENABLE_PRIVATE_MODE` and `ENABLE_BIAS_MITIGATIONS` — should these default to `true` or `false` for the initial deployment?

---

## Roadmap

- [ ] Wire new routers into `app/main.py` and generate Alembic migrations
- [ ] User authentication and authorization (flagged CRITICAL in infra audit)
- [ ] Run full test suite and fix failures
- [ ] Partner with one training provider for verified catalog data
- [ ] Define wedge user persona and test with 5 real users
- [ ] Learning-to-rank model (LambdaMART) using pairwise data (needs ~500 records)
- [ ] Geographic training resource integration (ZIP → program mapping)
- [ ] BLS labor market data integration (replace demand signal stubs)
- [ ] Load testing (target: 100 concurrent users)
- [ ] Transition graph visualization (D3, needs sufficient user volume)
- [ ] Multi-language support (Spanish first)
- [ ] Embeddings-based similarity (complement skill matching)
- [ ] Resume parsing and skill extraction
- [ ] Mobile app

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, testing requirements, and how to submit changes.

## License

MIT License — see [LICENSE](LICENSE).

## Documentation

| Document | Purpose |
|----------|---------|
| [Developer Guide](docs/developer-guide.md) | Local setup, architecture, how to add features |
| [User Guide](docs/user-guide.md) | End-user documentation and FAQ |
| [ML Audit](docs/ml-audit.md) | Scoring pipeline analysis and technical debt |
| [UX Audit](docs/ux-audit.md) | Accessibility gaps and user journey analysis |
| [Infra Audit](docs/infra-audit.md) | Security review and deployment assessment |
| [ADR-001](docs/adr/ADR-001-two-stage-scoring.md) | Why two-stage scoring |
| [ADR-002](docs/adr/ADR-002-transition-features.md) | Why transition-aware features |
| [ADR-003](docs/adr/ADR-003-cold-start-strategy.md) | Why three-tier cold start |
| [ADR-004](docs/adr/ADR-004-calibration-over-ranking.md) | Why calibration over ranking |
| [ADR-005](docs/adr/ADR-005-feedback-signal-hierarchy.md) | Why this signal hierarchy |
| [ADR-006](docs/adr/ADR-006-bias-audit-approach.md) | Why demographic parity testing |
| [Modeling Notes](MODELING_NOTES.md) | Detailed scoring formulas |
| [Docker Testing](DOCKER_TESTING.md) | Docker test environment guide |

## Support

For issues or questions:
- GitHub Issues: [repository-url]/issues
- Documentation: http://localhost:8000/docs-page
- API Docs: http://localhost:8000/api/v1/docs

## Acknowledgments

- O*NET Web Services for occupation and skill data
- FastAPI for the excellent web framework
- scikit-learn for ML tools
