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
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼──────┐
│ DB   │  │ O*NET   │
│(Postgres)│ │ API    │
└──────┘  └─────────┘
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

## Roadmap

- [ ] User authentication and authorization
- [ ] Embeddings-based similarity (in addition to skill matching)
- [ ] LLM-powered career advice
- [ ] Resume parsing and skill extraction
- [ ] Job posting integration
- [ ] Mobile app
- [ ] Advanced analytics dashboard

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run tests: `pytest`
5. Submit pull request

## License

MIT License

## Support

For issues or questions:
- GitHub Issues: [repository-url]/issues
- Documentation: http://localhost:8000/docs-page
- API Docs: http://localhost:8000/api/v1/docs

## Acknowledgments

- O*NET Web Services for occupation and skill data
- FastAPI for the excellent web framework
- scikit-learn for ML tools