# SkillSprout

## Technical Leadership Orientation

Welcome. This section is written for the incoming technical manager to get oriented quickly: what SkillSprout does, how to verify it works, what decisions were made and why, and what open questions need your judgment.

### What This Project Is

SkillSprout is a job-transition recommendation engine. Users enter their current occupation and self-rate their skills (0-4 scale). The system scores target occupations using O\*NET skill data and assigns each to one of three buckets:

- **Ready Now** - match >= 75%, gap <= 25%: apply immediately
- **Trainable** - match >= 50%, or gap in 26-55%: reachable with focused training
- **Long Reskill** - everything else: requires significant reskilling

It then generates structured explanations, skill-gap analyses, and personalized training paths. The system is designed to learn from user feedback via a calibration layer (logistic regression, not yet active in production — needs ~500 labeled samples).

### Quickstart: Setup and Verify

```bash
# Option A: Docker (recommended, starts all services)
make dev                    # PostgreSQL, Redis, FastAPI, Celery
open http://localhost:8000  # Web UI
open http://localhost:8000/api/v1/docs  # Swagger

# Option B: Local with SQLite (no external services needed)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
DEMO_MODE=true DATABASE_URL=sqlite+aiosqlite:///./dev.db \
  DATABASE_URL_SYNC=sqlite:///./dev.db \
  uvicorn app.main:app --reload
```

### Running Tests

```bash
# Full suite (725 tests, ~12s on SQLite)
DEMO_MODE=true DATABASE_URL=sqlite+aiosqlite:///./test.db \
  DATABASE_URL_SYNC=sqlite:///./test.db \
  python -m pytest tests/ -v

# Docker-isolated (PostgreSQL, full integration)
make test

# Specific focus areas:
python -m pytest tests/test_cross_team_qa.py -v    # 58 invariant/contract tests
python -m pytest tests/integration/ -v              # 5-persona pipeline tests
python -m pytest tests/unit/test_scoring.py -v      # Scoring engine unit tests
```

All 725 tests pass as of this writing. The CI pipeline (`.github/workflows/test.yml`) runs lint, security scan, unit tests, integration tests, E2E tests, and a production image build on every push.

### Architecture at a Glance

| Layer | Key files | What to review |
|-------|-----------|----------------|
| **Scoring engine** | `app/ml/scoring.py` (344 lines) | `_assign_bucket()` for bucket logic, `_calculate_scores()` for the weighted matching formula |
| **Calibration model** | `app/ml/calibration.py` | Logistic regression layer — framework complete, awaiting production feedback data |
| **API surface** | `app/main.py`, `app/api/endpoints.py` | 46 routes across 14 routers; auth middleware in `app/core/auth.py` |
| **Explainability** | `app/features/explainability/` | Structured explanations with threshold transparency and "what would change" analysis |
| **Training paths** | `app/features/training_paths/` | Prerequisite-aware path generation with budget/timeline constraint tracking |
| **Test personas** | `tests/integration/test_full_pipeline.py` | 5 personas (Maria/nurse, James/retail, Aisha/bootcamp, Robert/mechanic, Sarah/veteran) |
| **Cross-team QA** | `tests/test_cross_team_qa.py` | 58 tests covering mathematical invariants, boundary values, monotonicity, input safety, contract alignment |
| **Architecture decisions** | `docs/adr/ADR-001` through `ADR-006` | Six ADRs documenting scoring, features, cold start, calibration, feedback signals, and bias audit design |

### What the Combined QA Review Found and Fixed

Two teams with different QA approaches (scenario-driven vs. invariant-driven) merged and conducted a joint technical review. Three issues were found and resolved:

| Issue | Severity | Root cause | Fix |
|-------|----------|------------|-----|
| **Input mutation** | Medium | `_calculate_scores()` modified the caller's `occupation_skills` list when all importance values were 0, setting them to 1.0. Repeated scoring of the same data produced different results. | Replaced in-place mutation with a local `importance_override` dict keyed by object identity. Caller data is never touched. (`app/ml/scoring.py`) |
| **Bucket monotonicity violation** | High | A dead zone existed between `trainable_match_max` (74) and `ready_now_match_threshold` (75). Improving a skill could reduce gap severity below the trainable range while keeping match just below ready_now — causing a downgrade from trainable to long_reskill. | Removed the upper bound on trainable match. Any match >= 50 that does not qualify for ready_now is now trainable. The ready_now check runs first, so this cannot produce false positives. (`app/ml/scoring.py`) |
| **Cross-persona test semantics** | Low | A test assumed Aisha (has programming) would have a higher total match than Maria for Software Developer roles. In reality, Maria's 6 expert-level soft skills give her broader skill coverage (63% vs 61%), even though Aisha has programming and Maria does not. | Test rewritten to assert gap-specific invariant: Aisha has no programming gap, while Maria/James/Robert do. (`tests/test_cross_team_qa.py`) |

### Key Design Decisions (for your review)

1. **Two-stage scoring** (ADR-001): deterministic baseline + learned calibration. Baseline ships now; calibration activates when feedback volume is sufficient (~500 samples).
2. **Trainable bucket uses OR logic** (match >= 50 OR gap in 26-55). This is intentional — users can enter trainable via either high match with some gaps or low match with targeted gaps.
3. **Auth disabled in dev** (`AUTH_ENABLED=false`). API key middleware is wired (`app/core/auth.py`); set `AUTH_ENABLED=true` and `API_KEY=<secret>` in production. OAuth2/JWT is roadmap.
4. **Demo mode** activates when O\*NET credentials are absent. Uses `MockONetClient` with 3 occupations. All tests run in demo mode by default.
5. **Privacy**: GDPR data export (`/api/v1/user/{id}/data-export`) and deletion endpoints are implemented. Private mode suppresses all writes. Review with legal before production.

### Outstanding Questions for Leadership Decision

These items require human judgment and cannot be resolved by engineering alone:

| # | Question | Context | Recommendation |
|---|----------|---------|----------------|
| 1 | **When to activate the calibration model?** | Framework is built; needs ~500 labeled feedback samples (interview/offer/apply/hide). Currently the system runs on deterministic baseline only. | Set up a feedback collection pipeline with early users. Monitor sample volume via `/api/v1/model/status`. Activate when AUC > 0.65 on held-out set. |
| 2 | **Training catalog data accuracy** | 40+ resources in `app/features/training_paths/training_catalog.py` with URLs, costs, durations. These were best-effort at time of build. | Assign quarterly review. Consider a Celery task for automated URL checking (roadmap item). |
| 3 | **Bias audit data source** | `ml/bias_audit/` uses stub demographic profiles. Real BLS data is needed for production demographic parity testing. | Decide which BLS datasets to integrate and whether demographic data collection from users is in scope. |
| 4 | **Production auth upgrade path** | API key auth is minimal. OAuth2/JWT with user accounts is on the roadmap. | Decide: build in-house, integrate with an identity provider (Auth0, Cognito), or defer until user testing validates the product? |
| 5 | **Cold start k=50 cluster count** | K-means clustering uses k=50 for ~970 O\*NET occupations. Appropriate for current data; may need revalidation as occupation data updates. | Validate silhouette score on production data after O\*NET refresh cycles. |
| 6 | **Load testing target** | Roadmap lists 100 concurrent users. No load testing has been performed yet. | Define production SLA (p99 latency, throughput) before investing in load testing infrastructure. |
| 7 | **`trainable_match_max` config** | The `trainable_match_max` setting in `app/core/config.py` is no longer used by the bucket logic (removed to fix monotonicity). It remains in the config for backward compatibility with explainability threshold profiles. | Decide whether to remove it from config or keep for future A/B experiments. |

### Where to Dig Deeper

| If you want to understand... | Start here |
|------------------------------|-----------|
| How scoring works, step by step | `app/ml/scoring.py`, then `MODELING_NOTES.md` |
| What got built in the hackathon | "Hackathon Sprint Summary" section below, plus `docs/adr/` |
| How the test suite is structured | `tests/` directory — 23 test files, organized by unit/integration, with `conftest.py` for shared fixtures |
| Infrastructure and deployment | `Dockerfile`, `docker-compose.yml`, `.github/workflows/test.yml`, `docs/infra-audit.md` |
| Privacy and compliance | `app/core/privacy/`, the GDPR endpoints in `app/api/endpoints.py` |
| What the users see | `templates/`, `static/`, `docs/user-guide.md` |

---

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
- **Health Check**: http://localhost:8000/health

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

### Core API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/occupations/search` | Search occupations |
| GET | `/api/v1/occupations/{code}` | Get occupation details |
| GET | `/api/v1/occupations/{code}/skills` | Get occupation skills |
| POST | `/api/v1/user/profile` | Create user profile |
| POST | `/api/v1/user/{id}/current-occupation` | Set current occupation |
| POST | `/api/v1/user/{id}/skills/ratings` | Update skill ratings |
| POST | `/api/v1/user/{id}/recommendations` | Get recommendations |
| POST | `/api/v1/feedback` | Submit feedback |
| GET | `/api/v1/model/status` | Model status and metrics |

### Feature Endpoints (Hackathon Sprint)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/skills/translate` | Translate natural language to O*NET skills |
| GET | `/api/v1/recommendations/{id}/explain` | Explain a recommendation |
| GET | `/api/v1/recommendations/compare?ids=...` | Compare occupations side-by-side |
| PATCH | `/api/v1/user/preferences` | Update risk tolerance |
| POST | `/api/v1/training-path/generate` | Generate a training path |
| GET | `/api/v1/training-resources` | Browse training catalog |
| POST/GET/PATCH | `/api/v1/profile` | User profile CRUD |
| POST/GET/PATCH/DELETE | `/api/v1/saved-occupations` | Saved occupations |
| POST | `/api/v1/events/heartbeat` | Dwell time tracking |
| GET | `/api/v1/user/{id}/data-export` | GDPR data export |
| DELETE | `/api/v1/user/{id}/data` | Account deletion |
| POST | `/api/v1/session/export` | Export session token |
| POST | `/api/v1/session/import` | Resume session |

### Infrastructure Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks DB, Redis) |
| GET | `/health/detailed` | Full component health |
| GET | `/metrics` | Prometheus metrics |

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
│   ├── api/              # Core API endpoints (search, recommend, feedback)
│   ├── core/
│   │   ├── auth.py       # API key authentication middleware
│   │   ├── config.py     # Application settings
│   │   ├── monitoring/   # Health checks, Prometheus metrics, alerting
│   │   ├── privacy/      # Data classification, retention, export, deletion
│   │   └── progressive/  # Lite mode, session resumption, offline export
│   ├── db/               # Database session management
│   ├── events/           # Implicit signals, pairwise preferences
│   ├── features/
│   │   ├── explainability/   # Bucket explanations, comparisons, thresholds
│   │   ├── skills_translator/ # NLP skill extraction (regex + TF-IDF)
│   │   ├── training_paths/   # Training catalog, path generation
│   │   └── user_profile/     # Profile, saved occupations, progress
│   ├── ml/               # Scoring and calibration models
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # O*NET client, external services
│   ├── tasks/            # Celery background tasks
│   └── main.py           # FastAPI application (46 routes)
├── ml/
│   ├── bias_audit/       # Demographic parity, mitigation strategies
│   ├── cold_start/       # User, occupation, and combination priors
│   ├── evaluation/       # Eval framework, synthetic data generation
│   ├── features/         # Transition-aware feature engineering
│   ├── model_management/ # Registry, calibration monitor, A/B testing
│   └── transition_graph/ # NetworkX career path graph
├── alembic/              # Database migrations
├── docs/                 # Audits, ADRs, guides
├── tests/                # 600+ unit and integration tests
├── templates/            # HTML templates
├── static/               # CSS, JS
└── requirements.txt      # Python dependencies
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
| `AUTH_ENABLED` | Enable API key authentication | `false` |
| `API_KEY` | API key for protected endpoints | - |
| `ONET_USERNAME` | O*NET Web Services username | - |
| `ONET_PASSWORD` | O*NET Web Services password | - |
| `DEMO_MODE` | Use mock O*NET data | `false` |
| `READY_NOW_MATCH_THRESHOLD` | Min match score for READY_NOW | `75.0` |
| `READY_NOW_GAP_THRESHOLD` | Max gap severity for READY_NOW | `25.0` |
| `ENABLE_PRIVATE_MODE` | Allow X-Private-Mode header | `true` |
| `ENABLE_BIAS_MITIGATIONS` | Enable bias mitigation strategies | `true` |

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

1. **Authentication** (CRITICAL):
   - Set `AUTH_ENABLED=true` and `API_KEY=<strong-random-key>` in production
   - API key middleware is in `app/core/auth.py`; OAuth2/JWT planned for v2
   - Health and metrics endpoints remain open for infrastructure probes

2. **Security**:
   - Set strong database passwords
   - Use HTTPS for API
   - Restrict CORS origins in `app/main.py` (currently `["*"]`)
   - Set `SKILLSPROUT_SESSION_KEY` env var for stable Fernet encryption

3. **Database**:
   - Run `alembic upgrade head` to apply DeletionAuditLog migration
   - Use connection pooling
   - Set up regular backups

4. **Privacy/GDPR**:
   - Review data export (`/api/v1/user/{id}/data-export`) and deletion endpoints with legal
   - Configure `DATA_RETENTION_DAYS=90` and `DELETION_COMPLIANCE_HOURS=72`
   - Bias audit demographic profiles are stubs; replace with real BLS data

5. **Monitoring**:
   - Prometheus metrics available at `/metrics`
   - Health probes at `/health` (liveness) and `/health/ready` (readiness)
   - Configure alerting rules in `app/core/monitoring/alerting_rules.py`

6. **Scaling**:
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

**Fully wired, tested, and working (46 routes, 600+ tests passing):**
- All 14 new API routers mounted in `app/main.py` with correct prefix handling
- API key authentication middleware (`app/core/auth.py`, disabled in dev, required in prod)
- Full scoring pipeline with transition-aware features
- Three-tier cold start handling (user/occupation/combination)
- Skills translator with rule-based + TF-IDF matching
- Bucket explanations with transparent thresholds
- Training catalog with 40+ real resources and constraint-aware path generation
- Privacy framework with private mode, GDPR data export, and account deletion
- Health checks at `/health`, `/health/ready`, `/health/detailed`
- Prometheus metrics at `/metrics`
- Session resumption via Fernet-encrypted tokens
- User profile with progress tracking and saved occupation re-scoring
- Bias audit framework with staleness and symmetry checks
- DeletionAuditLog model registered for Alembic migration
- Saved occupations API contract fixed (explicit `Query(...)` params)
- Retention policy SQL query fixed (removed incorrect `select()` wrapper)

**Remaining human-judgment items:**
- Training catalog data (URLs, costs, durations) needs periodic verification
- Bias audit demographic profiles are stubs — need real BLS data
- Cold start k=50 is appropriate for ~970 O*NET occupations; validate silhouette score on production data

### Sprint Decisions (Open Questions Resolved)

The following 7 open questions were resolved by the 3-engineer team (E1 ML, E2 UX, E3 Infra) with PM approval during the Week 2 QA sprint:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Router prefix: `/api/v1/` or `/api/v2/`? | **All `/api/v1/`** | E2: "Users shouldn't need to discover version boundaries." PM: "Ship it all as v1; version when we break backwards compat." |
| 2 | Authentication: block on it? | **Shipped API key middleware** | E3 implemented `app/core/auth.py`. Disabled in dev (`AUTH_ENABLED=false`), required in prod. OAuth2/JWT is v2 work. |
| 3 | Private mode scope: no-writes enough? | **Yes, no-writes is sufficient** | E1: "O*NET data is public. The privacy concern is user-specific data (ratings, saved occupations, events). Preventing those writes is the GDPR-relevant guarantee." |
| 4 | Bias audit staleness: 5 years or 3? | **Keep 5 years (730 days)** | E1: "O*NET updates occupations on a rolling basis. 3 years would flag too many valid occupations. 5 years matches the O*NET update cycle." E3: "Make it configurable via env var for operators who want tighter bounds." |
| 5 | Training catalog maintenance owner? | **Automated link checker (future roadmap)** | PM: "For now, catalog is best-effort. Add a Celery task that checks URLs monthly and flags dead links in the audit log. Human review quarterly." |
| 6 | Test DB seeding: fixture vs mock? | **MockONetClient for unit tests** | E1: "Unit tests should not need a database. Integration tests that need real data use `MockONetClient` with 10 representative occupations. No SQLite fixtures committed." |
| 7 | Feature flags defaults? | **All `true` except `AUTH_ENABLED`** | PM: "Ship with all features enabled. Auth is the one flag that should default to off in dev for developer experience. In prod, set `AUTH_ENABLED=true`." |

### QA Status (All Items Resolved)

**P1 — Integration wiring: DONE**
- [x] All 14 routers mounted in `app/main.py`
- [x] DeletionAuditLog registered in `alembic/env.py` for migration generation
- [x] Full test suite passes (600+ tests, 0 failures)

**P2 — Functional testing: DONE**
- [x] Skills Translator validated with real-world input strings
- [x] Explainability thresholds verified against scoring pipeline constants
- [x] Training Paths zero-budget and no-computer edge cases tested
- [x] Private Mode middleware verified (sets request state correctly)
- [x] Session resumption roundtrip tested (encrypt/decrypt preserves all fields)
- [x] Auth middleware tested (12 tests: open paths, protected routes, key validation)

**P3 — Data and model quality: VALIDATED**
- [x] Cold start k=50 confirmed appropriate; model auto-adjusts `effective_k = min(k, n_occ)`
- [x] Bias audit `identify_correlated_skills` fixed to scope to available occupation map
- [x] Feature flags all default to `true` (except `AUTH_ENABLED`)

**P4 — Non-functional: DONE**
- [x] Health endpoints at `/health`, `/health/ready`, `/health/detailed` (mounted at root)
- [x] `/metrics` endpoint wired and returning Prometheus format
- [x] Saved occupations API contract fixed (explicit `Query(...)` annotations)
- [x] Retention policy SQL fixed (removed unnecessary `select()` wrapper)
- [x] `getattr()` safety check added for retention rule column validation

---

## Roadmap

- [x] ~~Wire new routers into `app/main.py`~~ (done)
- [x] ~~API key authentication~~ (done, `app/core/auth.py`)
- [x] ~~Run full test suite and fix failures~~ (600+ tests passing)
- [x] ~~Generate Alembic migration for DeletionAuditLog~~ (registered in env.py)
- [ ] OAuth2/JWT authentication (upgrade from API key)
- [ ] Partner with one training provider for verified catalog data
- [ ] Define wedge user persona and test with 5 real users
- [ ] Learning-to-rank model (LambdaMART) using pairwise data (needs ~500 records)
- [ ] Geographic training resource integration (ZIP to program mapping)
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
This site incorporates information from O*NET Web Services by the U.S. Department of Labor, Employment and Training Administration (USDOL/ETA). O*NET® is a trademark of USDOL/ETA.

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
