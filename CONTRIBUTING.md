# Contributing to SkillSprout

Thank you for your interest in contributing to SkillSprout. This project helps people discover career transition opportunities using O*NET skill data.

## Development Setup

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Git

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd skillsprout

# Start all services (PostgreSQL, Redis, FastAPI, Celery)
docker-compose up -d

# Verify services are running
curl http://localhost:8000/api/v1/health

# Run tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (copy and modify)
cp .env.example .env

# Run migrations (requires PostgreSQL)
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

## Code Style

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines where possible
- Use descriptive variable names

## Testing

- Write tests for all new features
- Run the full test suite before submitting a PR
- Maintain test coverage above 80% for new code
- Test edge cases (empty inputs, missing data, boundary values)

```bash
# Run unit tests locally
pytest tests/unit/ -v

# Run full suite in Docker
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## How to Contribute

### Adding Training Resources

Training resources are defined in `app/features/training_paths/training_catalog.py`. To add a new resource:

1. Verify the resource is currently available and accessible
2. Map it to the O*NET skill codes it develops
3. Include accurate cost, duration, and format information
4. Add the `last_verified` date
5. Test that the path generator includes it appropriately

### Adding Bias Audit Improvements

The bias audit framework is in `ml/bias_audit/`. Contributions welcome:

1. New demographic data sources for occupation profiling
2. Additional bias detection methods
3. Mitigation strategies with documented tradeoffs
4. Test cases that exercise edge cases in bias detection

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes with clear commit messages
4. Run the test suite and verify it passes
5. Submit a pull request with a description of what and why

### Bug Reports

File issues with:
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, Docker version)
- Relevant log output

## Code of Conduct

Be respectful and constructive. This project serves people making career transitions, many of whom are in vulnerable situations. Keep that context in mind when making design decisions.

## Architecture Overview

See `docs/developer-guide.md` for the full architecture documentation including the scoring pipeline, feature engineering, and deployment topology.
