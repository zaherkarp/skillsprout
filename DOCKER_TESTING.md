# SkillSprout Docker Testing Guide

Complete guide for testing SkillSprout using Docker and automated testing environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Compose Setup](#docker-compose-setup)
3. [Running Tests](#running-tests)
4. [End-to-End Testing](#end-to-end-testing)
5. [Makefile Commands](#makefile-commands)
6. [CI/CD Integration](#cicd-integration)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Make (optional, for convenience commands)

### Fastest path — no Docker required

You do not need Docker (or Postgres, Redis, Celery, or an O*NET API key) to run
SkillSprout. The quickest way is:

```bash
make run    # venv + deps + API on SQLite, served from the bundled O*NET data
```

The rest of this guide covers the Docker workflow, which is useful for
integration testing and the optional background-job stack.

### Docker development environment

Two tiers:

```bash
make up        # ONE lightweight container: the API on SQLite (offline data).
               # No Postgres, Redis, or Celery.

make up-full   # The full stack: Postgres + Redis + Celery + API on Postgres.
               # Only needed for background cache-warming / nightly retrain.
```

`make dev` = `build` + `up` (build the image, then start the lightweight API).
The heavy services (`db`, `redis`, `celery-worker`, `celery-beat`) live behind
the Compose `full` profile, so a bare `docker compose up` starts only the API.

**Access Points:**
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/api/v1/docs
- Health Check: http://localhost:8000/api/v1/health

### Run Full Test Suite

```bash
# Using make
make test

# Or directly with the test script
./scripts/run_tests.sh

# Or with docker-compose
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

---

## Docker Compose Setup

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                        │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │PostgreSQL│  │  Redis   │  │     FastAPI App      │  │
│  │   :5432  │  │  :6379   │  │       :8000          │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
│                                                          │
│  ┌──────────────────────┐  ┌────────────────────────┐  │
│  │   Celery Worker      │  │   Celery Beat          │  │
│  │  (Background Tasks)  │  │  (Periodic Tasks)      │  │
│  └──────────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Services

> By default (`make up`) only the **api** service runs, on SQLite. The services
> below marked *(full profile)* start only with `make up-full` (or
> `docker compose --profile full up`). The **test** runner starts them via the
> `test` profile.

1. **db**: PostgreSQL 15 database *(full profile)*
   - Port: 5432
   - Credentials: skillsprout/skillsprout_password
   - Volume: `postgres_data`

2. **redis**: Redis 7 cache
   - Port: 6379
   - Volume: `redis_data`

3. **api**: FastAPI application
   - Port: 8000
   - Hot reload enabled in dev mode
   - Auto-runs migrations on startup

4. **celery-worker**: Background task processor
   - Processes cache warming, model training

5. **celery-beat**: Periodic task scheduler
   - Runs daily model training at 2 AM

6. **test**: Isolated test runner (profile)
   - Uses separate test database
   - Runs pytest with coverage

### Configuration Files

- `docker-compose.yml`: Main development environment
- `docker-compose.test.yml`: Isolated testing environment
- `Dockerfile`: Multi-stage build (dev and production targets)
- `.dockerignore`: Excludes unnecessary files from image

---

## Running Tests

### Option 1: Using Make (Recommended)

```bash
# Run all tests
make test

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run tests with coverage report
make test-coverage
```

### Option 2: Using Test Script

```bash
# Run full test suite with cleanup
./scripts/run_tests.sh
```

Features:
- Automatic environment setup and teardown
- Color-coded output
- Coverage report extraction
- Exit code propagation for CI/CD

### Option 3: Using Docker Compose Directly

```bash
# Start test environment and run tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Cleanup after tests
docker-compose -f docker-compose.test.yml down -v
```

### Test Environment Details

The test environment uses:
- **Isolated database**: `skillsprout_test` (separate from dev)
- **Isolated Redis**: Different Redis DB numbers
- **In-memory storage**: Uses tmpfs for faster tests
- **Demo mode**: Mocked O*NET data (no credentials needed)

### Coverage Reports

After running tests with coverage:

```bash
# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## End-to-End Testing

### Interactive E2E Test

Tests the complete user workflow via API:

```bash
# Ensure dev environment is running
make up

# Run E2E tests
./scripts/test_e2e.sh
```

### What E2E Tests Cover

1. ✓ Health check
2. ✓ Create user profile
3. ✓ Search occupations
4. ✓ Get occupation details
5. ✓ Get occupation skills
6. ✓ Set current occupation
7. ✓ Rate skills
8. ✓ Get recommendations
9. ✓ Submit feedback
10. ✓ Check model status

### Custom E2E Tests

You can customize the E2E test script:

```bash
# Test against different API URL
API_URL=http://localhost:8000 ./scripts/test_e2e.sh

# Test production endpoint
API_URL=https://api.example.com ./scripts/test_e2e.sh
```

---

## Makefile Commands

### Service Management

```bash
make build          # Build Docker images
make up             # Start all services
make down           # Stop all services
make restart        # Restart all services
make ps             # Show running containers
make stats          # Show resource usage
```

### Logs and Debugging

```bash
make logs           # Show logs from all services
make logs-api       # Show API logs only
make logs-celery    # Show Celery worker logs
make logs-db        # Show database logs
make shell-api      # Open shell in API container
make shell-db       # Open PostgreSQL shell
```

### Database Operations

```bash
make migrate        # Run database migrations
make migrate-create MSG="add field"  # Create new migration
make db-backup      # Backup database to backups/
make db-restore     # Restore from latest backup
```

### Data Management

```bash
make seed           # Seed demo data
make clean          # Stop services and remove volumes
make prune          # Remove all Docker resources (with confirmation)
```

### Testing

```bash
make test           # Run full test suite
make test-unit      # Run unit tests only
make test-integration  # Run integration tests only
make test-coverage  # Run tests with coverage report
```

### Health and Status

```bash
make health         # Check health of all services
make celery-status  # Show Celery worker status
make celery-tasks   # List registered Celery tasks
make celery-active  # Show active Celery tasks
make train-model    # Manually trigger model training
```

### Development Shortcuts

```bash
make dev            # Full dev setup (build + start + seed)
make quick-start    # Quick start (no build, uses cache)
make info           # Show environment information
make help           # Show all available commands
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Build and run tests
        run: |
          docker-compose -f docker-compose.test.yml up \
            --abort-on-container-exit \
            --exit-code-from test-runner

      - name: Upload coverage report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: coverage-report
          path: htmlcov/
```

### GitLab CI Example

```yaml
test:
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - apk add --no-cache docker-compose
  script:
    - docker-compose -f docker-compose.test.yml build
    - docker-compose -f docker-compose.test.yml up --abort-on-container-exit
  artifacts:
    paths:
      - htmlcov/
    expire_in: 1 week
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                sh '''
                    docker-compose -f docker-compose.test.yml build
                    docker-compose -f docker-compose.test.yml up \
                        --abort-on-container-exit \
                        --exit-code-from test-runner
                '''
            }
        }
    }

    post {
        always {
            sh 'docker-compose -f docker-compose.test.yml down -v'
            publishHTML([
                reportDir: 'htmlcov',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }
    }
}
```

---

## Troubleshooting

### Services Won't Start

**Problem**: Services fail to start or health checks fail

**Solutions**:

```bash
# Check if ports are already in use
lsof -i :8000  # API port
lsof -i :5432  # PostgreSQL port
lsof -i :6379  # Redis port

# Check Docker daemon
docker info

# Check logs
make logs

# Clean and rebuild
make clean
make build
make up
```

### Database Migrations Fail

**Problem**: Alembic migrations fail on startup

**Solutions**:

```bash
# Check database connection
make shell-db

# Reset database
make down
docker volume rm skillsprout_postgres_data
make up

# Run migrations manually
make migrate
```

### Tests Fail in Docker but Pass Locally

**Problem**: Tests pass locally but fail in Docker

**Solutions**:

```bash
# Ensure clean test environment
docker-compose -f docker-compose.test.yml down -v

# Rebuild test images
docker-compose -f docker-compose.test.yml build --no-cache

# Check environment variables
docker-compose -f docker-compose.test.yml config

# Run with verbose output
docker-compose -f docker-compose.test.yml run test-runner pytest -vv
```

### Out of Disk Space

**Problem**: Docker runs out of disk space

**Solutions**:

```bash
# Remove unused Docker resources
docker system prune -a

# Remove specific volumes
docker volume ls
docker volume rm <volume_name>

# Check disk usage
docker system df
```

### Performance Issues

**Problem**: Tests or API are slow in Docker

**Solutions**:

```bash
# Increase Docker resources (Docker Desktop)
# Settings > Resources > Adjust CPU/Memory

# Use tmpfs for databases (already configured in docker-compose.test.yml)

# Monitor resource usage
make stats

# Optimize volumes (avoid mounting large directories)
```

### Port Already in Use

**Problem**: Cannot start services due to port conflicts

**Solutions**:

```bash
# Find process using port
lsof -i :8000
# Kill the process or change port in docker-compose.yml

# Change ports in docker-compose.yml
# ports:
#   - "8001:8000"  # Map to different host port
```

### Can't Connect to Services

**Problem**: Cannot access API from host

**Solutions**:

```bash
# Check if services are running
make ps

# Check service health
make health

# Verify network
docker network ls
docker network inspect skillsprout_skillsprout-network

# Check API logs
make logs-api

# Try from inside container
docker-compose exec api curl http://localhost:8000/api/v1/health
```

---

## Best Practices

### For Development

1. **Use `make dev`** for initial setup
2. **Use `make logs-api`** for debugging
3. **Run `make test`** before commits
4. **Use `make clean`** to reset environment
5. **Back up data** with `make db-backup` before destructive operations

### For Testing

1. **Always use isolated test environment** (`docker-compose.test.yml`)
2. **Clean up after tests** with `-v` flag
3. **Check coverage reports** after test runs
4. **Run E2E tests** before major releases
5. **Use tmpfs** for test databases (faster)

### For CI/CD

1. **Cache Docker layers** to speed up builds
2. **Use `--abort-on-container-exit`** to get proper exit codes
3. **Save coverage reports** as artifacts
4. **Clean up volumes** after test runs
5. **Use health checks** to ensure services are ready

---

## Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Celery Testing](https://docs.celeryproject.org/en/stable/userguide/testing.html)

---

## Quick Reference Card

```bash
# DEVELOPMENT
make dev              # Full setup
make up               # Start services
make down             # Stop services
make logs-api         # View API logs
make shell-api        # Shell into API

# TESTING
make test             # Run all tests
make test-coverage    # Tests + coverage
./scripts/test_e2e.sh # E2E tests
./scripts/run_tests.sh # Full test suite

# DATABASE
make migrate          # Run migrations
make seed             # Seed data
make db-backup        # Backup DB
make shell-db         # DB shell

# CLEANUP
make clean            # Remove volumes
make prune            # Remove everything

# MONITORING
make health           # Check services
make ps               # Show containers
make stats            # Resource usage
```

---

**Questions or Issues?**

- Check logs: `make logs`
- Check health: `make health`
- Check [Troubleshooting](#troubleshooting) section
- Open an issue on GitHub
