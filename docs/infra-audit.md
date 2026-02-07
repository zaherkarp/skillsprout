# Infrastructure & Security Audit -- SkillSprout

**Date:** 2026-02-07
**Scope:** Full-stack review of application infrastructure, data layer, async architecture, security posture, deployment pipeline, and data sensitivity.
**Codebase:** FastAPI + async SQLAlchemy + PostgreSQL + Redis + Celery, deployed via Docker Compose with GitHub Actions CI/CD.

---

## Executive Summary

SkillSprout is a career-transition recommendation platform that ingests O\*NET occupational data, scores user-skill alignment, and delivers bucketed job recommendations. The application is architecturally sound in its data modeling and async design, but suffers from a **complete absence of authentication and authorization** -- the single most critical finding in this audit. Additional concerns include an overly permissive CORS policy, sync-in-async violations in background tasks, a full-table scan in the recommendation engine, and the handling of sensitive career exploration data without retention or encryption policies.

**Finding Distribution:**
| Severity | Count |
|----------|-------|
| CRITICAL | 2     |
| HIGH     | 6     |
| MEDIUM   | 8     |
| LOW      | 5     |

---

## 1. Database Architecture

### 1.1 Entity Map

The schema defines **10 tables** across 9 domain entities plus 1 system table:

| Table | Primary Key | Purpose |
|-------|-------------|---------|
| `occupation` | `onet_code` (String(10)) | O\*NET occupation cache |
| `skill` | `element_id` (String(20)) | O\*NET skill definitions |
| `occupation_skill` | `id` (auto-int) | Junction: occupation-to-skill with importance/level |
| `user_profile` | `id` (auto-int) | User identity and metadata |
| `user_current_occupation` | `id` (auto-int) | User's selected current occupation |
| `user_skill_rating` | `id` (auto-int) | User's self-assessed skill proficiency (0-4) |
| `recommendation_event` | `id` (auto-int) | One-per-request recommendation generation event |
| `recommended_occupation` | `id` (auto-int) | Individual scored recommendation within an event |
| `user_feedback` | `id` (auto-int) | User actions on recommendations (click/save/hide/apply/interview/offer) |
| `model_registry` | `id` (auto-int) | ML model version tracking and artifact paths |

**Relationships verified:** All foreign keys use `ondelete="CASCADE"`. Relationship back-references are properly configured. The `occupation_skill` and `user_skill_rating` tables enforce uniqueness constraints (`uq_occupation_skill`, `uq_user_skill`) to prevent duplicate entries.

*Source: `/home/user/skillsprout/app/models/models.py`*

### 1.2 Indexing Review

**Existing indexes (verified in Alembic migration `d1316f64dcb1`):**

| Table | Indexed Columns | Notes |
|-------|----------------|-------|
| `occupation` | `onet_code` (PK + index), `title` | Good for search-by-title |
| `skill` | `element_id` (PK + index), `name` | Good for lookup |
| `occupation_skill` | `onet_code`, `element_id` | Covers FK lookups; unique constraint also present |
| `user_profile` | `created_at` | Administrative queries |
| `user_current_occupation` | `user_id`, `selected_at` | Covers primary query pattern |
| `user_skill_rating` | `user_id` | Covers user-scoped queries |
| `recommendation_event` | `user_id`, `created_at` | Covers primary query patterns |
| `recommended_occupation` | `event_id`, `bucket` | Covers event-scoped and bucket-filtered queries |
| `user_feedback` | `event_id`, `action_type`, `action_at` | Covers primary query patterns |
| `model_registry` | `model_version` (unique) | Covers version lookups |

**Missing indexes identified:**

| Finding | Severity | Detail |
|---------|----------|--------|
| No index on `user_current_occupation.onet_code` | **MEDIUM** | FK column queried when looking up occupation relationship. PostgreSQL does not auto-index FK columns. |
| No index on `user_skill_rating.element_id` | **MEDIUM** | FK column; skill-scoped queries (e.g., "which users rated skill X") would require a full scan. |
| No index on `user_feedback.target_onet_code` | **MEDIUM** | Used in the feedback submission query (`RecommendedOccupation.target_onet_code == request.target_onet_code`). Would matter at scale. |
| No index on `recommended_occupation.target_onet_code` | **MEDIUM** | FK column. The feedback endpoint queries this in a compound WHERE clause. |
| Redundant index on `occupation.onet_code` | **LOW** | Primary key columns are already indexed; the explicit `index=True` on the PK creates a duplicate index. Same issue applies to `skill.element_id`. |

### 1.3 Alembic Migration Review

There is a **single initial migration** (`20260207_0430_d1316f64dcb1_initial_database_schema.py`) that creates all 10 tables in one shot.

| Finding | Severity | Detail |
|---------|----------|--------|
| Single monolithic migration | **LOW** | Acceptable for initial schema, but future changes should be incremental. A single migration means no ability to roll back partial schema changes. |
| No data migration support | **LOW** | No `op.execute()` for seed data. Seed data is handled by a separate `scripts/seed_demo.py`, which is appropriate. |
| Migration matches model definitions | **OK** | Verified: all columns, constraints, and indexes in the migration match the SQLAlchemy model definitions exactly. |

*Source: `/home/user/skillsprout/alembic/versions/20260207_0430_d1316f64dcb1_initial_database_schema.py`*

### 1.4 N+1 Query Patterns

| Finding | Severity | Detail |
|---------|----------|--------|
| Recommendation endpoint loads ALL occupations | **HIGH** | `endpoints.py` line 455-459: `select(Occupation).options(joinedload(...))` with no WHERE clause loads every occupation and every associated skill into memory. At O\*NET scale (~1,000 occupations x ~35 skills each = ~35,000 rows), this becomes a serious performance bottleneck. Should implement pagination, pre-filtered candidate sets, or materialized scoring views. |
| `joinedload` used correctly on targeted queries | **OK** | Single-occupation queries (e.g., `get_occupation_skills`) properly use `joinedload` to avoid N+1 on the occupation->skills->skill chain. |
| Skill ratings loop in `update_skill_ratings` | **MEDIUM** | `endpoints.py` lines 373-396: Each skill rating is upserted individually with a SELECT + INSERT/UPDATE per rating. For a user rating 35 skills, this is 35 round trips. Should use `INSERT ... ON CONFLICT` bulk upsert. |
| Cache warming task has per-occupation queries | **LOW** | `tasks.py` iterates through occupation codes with individual queries. Acceptable for a background task but could be batched. |

*Source: `/home/user/skillsprout/app/api/endpoints.py`, lines 454-459*

### 1.5 Connection Pooling

```python
# /home/user/skillsprout/app/db/session.py
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    pool_pre_ping=True,
)
```

| Finding | Severity | Detail |
|---------|----------|--------|
| `pool_pre_ping=True` enabled | **OK** | Connection health is validated before use, preventing stale connection errors after PostgreSQL restarts. |
| Default pool size (5) and max overflow (10) | **MEDIUM** | SQLAlchemy defaults are 5 pool + 10 overflow = 15 max connections. With 4 Uvicorn workers in production, this means up to 60 simultaneous connections. Should explicitly set `pool_size`, `max_overflow`, and `pool_recycle` based on expected load and PostgreSQL `max_connections`. |
| `echo=settings.debug` in production risk | **HIGH** | If `DEBUG=true` leaks into production, every SQL statement will be logged to stdout, including any data in WHERE clauses (user IDs, occupation codes). The `settings.debug` defaults to `True`. |
| Sync engine for Celery also uses defaults | **LOW** | The sync engine (`sync_engine`) uses the same default pool settings. Celery workers typically need fewer connections, but this should be tuned. |

*Source: `/home/user/skillsprout/app/db/session.py`*

---

## 2. Caching Layer

### 2.1 Redis Usage

Redis is configured at `redis://localhost:6379/0` but is used **exclusively as Celery's message broker and result backend**. No application-level caching (e.g., occupation lookups, search results, user sessions) uses Redis.

```python
# /home/user/skillsprout/app/core/config.py
redis_url: str = "redis://localhost:6379/0"
celery_broker_url: str = "redis://localhost:6379/0"
celery_result_backend: str = "redis://localhost:6379/1"
```

| Finding | Severity | Detail |
|---------|----------|--------|
| Redis underutilized | **MEDIUM** | Redis is deployed and running but provides no caching benefit to the application. Occupation data, search results, and recommendation scores are all candidates for Redis caching. |
| Celery broker and application share Redis DB 0 | **LOW** | The `redis_url` and `celery_broker_url` point to the same Redis database (DB 0). Namespace collisions are unlikely given current usage, but should be separated for clarity. |
| No Redis authentication | **MEDIUM** | Redis is running without `requirepass`. In the Docker network this is acceptable, but in any shared-network or production deployment, Redis must be password-protected. |

### 2.2 O\*NET Data Caching in PostgreSQL

O\*NET occupation and skill data is cached in PostgreSQL rather than Redis. The caching pattern is "cache-on-first-read":

1. Endpoint checks if the occupation exists in the `occupation` table.
2. If not found, fetches from O\*NET API and persists to PostgreSQL.
3. Subsequent requests are served from the database.

```python
# /home/user/skillsprout/app/api/endpoints.py, lines 82-86
result = await db.execute(
    select(Occupation).where(Occupation.onet_code == onet_code)
)
occupation = result.scalar_one_or_none()
```

| Finding | Severity | Detail |
|---------|----------|--------|
| No cache invalidation strategy | **HIGH** | Occupation data is written once and never refreshed. The `last_fetched_at` timestamp is stored but never checked. O\*NET updates its data periodically; stale data could lead to incorrect skill importance values and therefore incorrect recommendations. |
| No TTL or staleness check | **HIGH** | There is no mechanism to refetch data after a configurable TTL. A data freshness policy (e.g., refetch if `last_fetched_at` > 30 days) should be implemented. |
| Cache warming is manual/optional | **MEDIUM** | The `warm_occupation_cache` Celery task exists but is not scheduled in `beat_schedule`. Only `train_calibration_model_task` is periodic. Cache warming must be triggered manually or via the seed script. |

*Source: `/home/user/skillsprout/app/api/endpoints.py`, `/home/user/skillsprout/app/tasks/tasks.py`*

---

## 3. Async Architecture

### 3.1 Sync-in-Async Violation

The Celery tasks in `tasks.py` call `asyncio.run()` to invoke the async O\*NET client from within synchronous Celery task functions:

```python
# /home/user/skillsprout/app/tasks/tasks.py, lines 63, 84, 304
occ_data = asyncio.run(client.get_occupation_meta(onet_code))
skills_data = asyncio.run(client.get_occupation_skills(onet_code))
results = asyncio.run(client.search_occupations(query, limit=10))
```

| Finding | Severity | Detail |
|---------|----------|--------|
| `asyncio.run()` in Celery tasks | **HIGH** | Each `asyncio.run()` call creates and destroys an event loop. This is wasteful and can cause issues if Celery is running with an event loop already active (e.g., with gevent/eventlet pool). The correct approach is to either: (a) use a synchronous HTTP client (e.g., `httpx.Client` instead of `httpx.AsyncClient`) in Celery tasks, or (b) maintain a single event loop per worker with `asyncio.get_event_loop().run_until_complete()`. |
| Multiple `asyncio.run()` per task invocation | **MEDIUM** | In `warm_occupation_cache`, two `asyncio.run()` calls execute per occupation code (one for metadata, one for skills). For 50 occupations, this creates and destroys 100 event loops. |

### 3.2 Database Session Management

The async session lifecycle is well-scoped:

```python
# /home/user/skillsprout/app/db/session.py, lines 44-54
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

| Finding | Severity | Detail |
|---------|----------|--------|
| Auto-commit on success in dependency | **MEDIUM** | The `get_db` generator auto-commits after the endpoint function returns successfully. This means every endpoint implicitly commits, even read-only GET endpoints. This could cause issues if a GET endpoint accidentally modifies session state. However, `autoflush=False` mitigates accidental writes. |
| Double commit possible | **LOW** | Some endpoints explicitly call `await db.commit()` (e.g., `create_user_profile`), and then the `get_db` generator also commits on exit. This is harmless (committing a clean session is a no-op) but indicates inconsistent patterns. |
| Sync session for Celery tasks is correct | **OK** | Celery tasks use `SyncSessionLocal()` with manual lifecycle management (`db.close()` in finally block). This is the correct pattern for sync Celery workers. |

### 3.3 httpx Client Lifecycle

```python
# /home/user/skillsprout/app/services/onet_client.py, lines 105-106
async with httpx.AsyncClient(timeout=self.timeout) as client:
    response = await client.request(...)
```

| Finding | Severity | Detail |
|---------|----------|--------|
| New `httpx.AsyncClient` per request | **MEDIUM** | Every API call to O\*NET creates a new `httpx.AsyncClient` instance (and therefore a new connection pool). This prevents HTTP connection reuse and adds TLS handshake overhead per request. The client should be instantiated once and reused, or managed as a contextmanager at the application level. |
| ONetClient instantiated per endpoint call | **MEDIUM** | `get_onet_client()` is a factory that creates a new `ONetClient` (or `MockONetClient`) instance on every call. Combined with the above, there is zero connection reuse to the O\*NET API. |

*Source: `/home/user/skillsprout/app/services/onet_client.py`*

---

## 4. Security Review

### 4.1 Authentication & Authorization

| Finding | Severity | Detail |
|---------|----------|--------|
| **No authentication whatsoever** | **CRITICAL** | There is no authentication mechanism on any endpoint. All API routes (`/api/v1/*`) are fully open. Any client can create user profiles, submit skill ratings, generate recommendations, and submit feedback for any user ID. This is the highest-priority finding in this audit. |
| **No authorization / access control** | **CRITICAL** | User-scoped endpoints (e.g., `POST /user/{user_id}/recommendations`) accept any `user_id` in the URL path with no verification that the caller is authorized to act on behalf of that user. An attacker can enumerate user IDs and read/modify any user's data. |
| No API key requirement | **HIGH** | Even a simple API key or bearer token would provide a baseline level of access control. None exists. |

**Affected endpoints (all unauthenticated):**

| Method | Path | Risk |
|--------|------|------|
| POST | `/api/v1/user/profile` | Anyone can create unlimited user profiles |
| POST | `/api/v1/user/{user_id}/current-occupation` | Modify any user's occupation |
| POST | `/api/v1/user/{user_id}/skills/ratings` | Modify any user's skill ratings |
| POST | `/api/v1/user/{user_id}/recommendations` | Generate recommendations for any user |
| POST | `/api/v1/feedback` | Submit feedback for any event |
| GET | `/api/v1/model/status` | Expose internal model metrics |

*Source: `/home/user/skillsprout/app/api/endpoints.py`*

### 4.2 Input Validation

| Finding | Severity | Detail |
|---------|----------|--------|
| Pydantic validation on all request bodies | **OK** | All POST endpoints use typed Pydantic schemas with `Field` constraints (e.g., `rating_0_4: int = Field(..., ge=0, le=4)`, `limit: int = Field(10, ge=1, le=50)`). |
| `action_type` validated via `@validator` | **OK** | `UserFeedbackRequest.action_type` is validated against an explicit allowlist: `{"click", "save", "hide", "apply", "interview", "offer"}`. |
| Query parameter validation | **OK** | Search endpoint uses `Query(..., min_length=1)` and `Query(20, ge=1, le=100)` for bounds. |
| No SQL injection risk | **OK** | All database queries use SQLAlchemy ORM; no raw SQL is constructed from user input. |
| `onet_code` not format-validated | **LOW** | O\*NET codes follow a specific pattern (e.g., `15-1252.00`). The API accepts any string, which could lead to unnecessary O\*NET API calls for invalid codes. A regex validator would be appropriate. |

### 4.3 CORS Configuration

```python
# /home/user/skillsprout/app/main.py, lines 46-52
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

| Finding | Severity | Detail |
|---------|----------|--------|
| `allow_origins=["*"]` with `allow_credentials=True` | **HIGH** | This is an insecure combination. While browsers will technically block `credentials: true` with a wildcard origin, this configuration signals a lack of CORS policy. In production, `allow_origins` must be restricted to the specific frontend domain(s). The code comment `# Configure appropriately for production` acknowledges this but no environment-based override exists. |

### 4.4 Secrets Management

| Finding | Severity | Detail |
|---------|----------|--------|
| Secrets in environment variables | **OK** | O\*NET credentials, database passwords, and Redis URLs are loaded from environment variables via `pydantic-settings`. This is standard practice. |
| `.env` excluded from git | **OK** | `.gitignore` includes `.env` and `.env.local`. |
| `.env` excluded from Docker image | **OK** | `.dockerignore` includes `.env` and `.env.*`. |
| Hardcoded passwords in `docker-compose.yml` | **HIGH** | Database password `skillsprout_password` is hardcoded in plain text across all service definitions in `docker-compose.yml`. This file is committed to version control. Production credentials should use Docker secrets, a `.env` file referenced by `env_file:`, or a secrets manager. |
| `.env.example` contains placeholder credentials | **OK** | Contains `your_username_here` / `your_password_here` placeholders, not real credentials. |
| O\*NET credentials in plaintext env vars | **MEDIUM** | Standard for env-based config, but in production these should be injected from a secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault) rather than stored in compose files or `.env` files on disk. |

### 4.5 Rate Limiting

| Finding | Severity | Detail |
|---------|----------|--------|
| No rate limiting on any endpoint | **HIGH** | There is no rate limiting middleware (e.g., `slowapi`, `fastapi-limiter`). The recommendation endpoint is computationally expensive (loads all occupations, scores each one) and could be used for resource exhaustion attacks. The profile creation endpoint could be used to generate unlimited user records. |

### 4.6 Error Handling & Information Disclosure

| Finding | Severity | Detail |
|---------|----------|--------|
| Exception messages exposed to clients | **HIGH** | Multiple endpoints catch generic `Exception` and return `str(e)` in the HTTP response body (e.g., `endpoints.py` line 72: `raise HTTPException(status_code=500, detail=str(e))`). In production, this can leak internal implementation details, database errors, file paths, or stack traces. |
| `debug=True` by default | **MEDIUM** | `settings.debug` defaults to `True`. FastAPI in debug mode returns detailed error pages with stack traces. This must be explicitly set to `False` in production. |
| Health check returns hardcoded status | **LOW** | The `/health` endpoint returns `database: "connected"` and `redis: "connected"` as static strings without actually verifying connectivity. This makes the health check unreliable for orchestration (Docker health checks, load balancer probes). |

*Source: `/home/user/skillsprout/app/api/endpoints.py`, `/home/user/skillsprout/app/main.py`*

---

## 5. Deployment & CI/CD

### 5.1 Docker Configuration

**Dockerfile** (`/home/user/skillsprout/Dockerfile`):

```dockerfile
# Multi-stage: base -> development -> production
FROM python:3.11-slim as base        # Dependencies
FROM base as development             # Full source, --reload
FROM base as production              # Non-root user, 4 workers
```

| Finding | Severity | Detail |
|---------|----------|--------|
| Multi-stage build | **OK** | Properly separates dependency installation (base) from development and production stages. |
| Non-root user in production | **OK** | Production stage creates `appuser` (UID 1000) and runs as that user. Good security practice. |
| Development stage runs as root | **MEDIUM** | The development stage does not create a non-root user. While acceptable for local development, any development container exposed to a network runs with root privileges inside the container. |
| `COPY . .` copies entire context | **LOW** | Both development and production stages copy the entire build context. The `.dockerignore` mitigates this by excluding `.env`, tests, docs, and other non-essential files. |
| `.dockerignore` present and comprehensive | **OK** | Contrary to initial assessment, a `.dockerignore` file exists and properly excludes `.git`, `.env`, `tests/`, `docs/`, `*.md`, `docker-compose*.yml`, `.github/`, and model artifacts. |
| No `HEALTHCHECK` in Dockerfile | **LOW** | Health checks are defined in `docker-compose.yml` but not in the Dockerfile itself. This means standalone container runs (without Compose) lack health monitoring. |

**Docker Compose** (`/home/user/skillsprout/docker-compose.yml`):

| Finding | Severity | Detail |
|---------|----------|--------|
| 5 services defined | **OK** | `db` (PostgreSQL 15), `redis` (Redis 7), `api` (FastAPI), `celery-worker`, `celery-beat`. Architecture is appropriately decomposed. |
| Service health checks | **OK** | PostgreSQL and Redis have proper health checks. API depends on both with `condition: service_healthy`. |
| Source code mounted as volume | **MEDIUM** | `volumes: - .:/app` in all services means the entire source tree (including `.git`, any `.env` files, etc.) is accessible inside the container at runtime. The `.dockerignore` does not apply to volume mounts. |
| PostgreSQL data persisted | **OK** | `postgres_data` named volume ensures data survives container restarts. |
| Test database uses `tmpfs` | **OK** | `docker-compose.test.yml` uses `tmpfs` for PostgreSQL data, ensuring fast test execution and clean state. |

### 5.2 CI/CD Pipeline

**GitHub Actions** (`/home/user/skillsprout/.github/workflows/test.yml`):

The pipeline defines 4 jobs: `test`, `e2e-test`, `lint`, `security`.

| Finding | Severity | Detail |
|---------|----------|--------|
| `lint` job uses `continue-on-error: true` on all steps | **HIGH** | flake8, black, and isort checks all have `continue-on-error: true`, meaning the lint job always passes regardless of violations. Linting is effectively decorative. This should be changed to fail the build on lint errors. |
| `security` job uses `continue-on-error: true` | **HIGH** | The Trivy vulnerability scanner step has `continue-on-error: true`, meaning known vulnerabilities will never block a merge. At minimum, critical/high severity vulnerabilities should fail the build. |
| No deployment stage | **MEDIUM** | The pipeline tests and scans but has no deployment job. There is no automated path from a merged PR to a deployed environment. |
| `e2e-test` depends on `test` | **OK** | End-to-end tests only run if unit/integration tests pass first. This is a good gating strategy. |
| Docker layer caching | **OK** | Uses `actions/cache@v4` for Docker Buildx layers, reducing build times. |
| Coverage report uploaded as artifact | **OK** | Coverage HTML report is uploaded with 30-day retention. |
| No branch protection enforcement | **LOW** | The workflow triggers on pushes to `main`, `develop`, and `claude/*` branches, and on PRs to `main` and `develop`. However, CI passing is not enforced as a merge requirement (that is a GitHub repository settings concern, not a workflow issue). |
| E2E test uses `sleep 30` for service readiness | **LOW** | A fixed 30-second sleep is fragile. Should use a retry loop with health check polling instead. |

*Source: `/home/user/skillsprout/.github/workflows/test.yml`*

---

## 6. Data Sensitivity & Privacy

### 6.1 Sensitive Data Classification

SkillSprout processes **career exploration data**, which is inherently sensitive:

| Data Category | Tables | Sensitivity | Rationale |
|---------------|--------|-------------|-----------|
| Occupational interests | `user_current_occupation`, `recommended_occupation` | **HIGH** | Reveals what jobs a user is considering leaving/transitioning to. Could be used by employers to identify flight-risk employees. |
| Skill self-assessments | `user_skill_rating` | **HIGH** | Self-reported competency levels. Could be used to infer qualifications, education level, and professional weaknesses. |
| Career action signals | `user_feedback` | **HIGH** | Actions like `apply`, `interview`, and `offer` directly indicate active job-seeking behavior. Combined with `target_onet_code`, this reveals exactly which jobs a user is pursuing. |
| User metadata | `user_profile.metadata_json` | **MEDIUM** | Unstructured JSON field that could contain demographic or identifying information depending on what the frontend sends. |
| Model training data | `model_registry.metrics_json` | **LOW** | Aggregate model performance metrics. Not individually identifying. |

### 6.2 Privacy & Compliance Findings

| Finding | Severity | Detail |
|---------|----------|--------|
| No data retention policy | **HIGH** | User data, feedback events, and recommendation history are stored indefinitely. There is no mechanism to age-out old data, no TTL on records, and no scheduled cleanup task. Under GDPR, CCPA, and similar regulations, data retention must be limited to what is necessary. |
| No user data deletion capability | **HIGH** | There is no endpoint or mechanism for a user to request deletion of their data (right to erasure / right to be forgotten). While CASCADE deletes on `user_profile` would propagate deletions, no API endpoint or admin tool exists to trigger this. |
| No encryption at rest | **MEDIUM** | PostgreSQL data is stored unencrypted on the `postgres_data` Docker volume. In production, the database should use Transparent Data Encryption (TDE) or the underlying storage should be encrypted (e.g., encrypted EBS volumes on AWS). |
| No encryption in transit (internal) | **MEDIUM** | Communication between services (API <-> PostgreSQL, API <-> Redis, Worker <-> Redis) uses unencrypted connections. In a Docker bridge network this is lower risk, but in any multi-host deployment, TLS should be configured for all inter-service communication. |
| Logs may contain PII | **MEDIUM** | Logger statements include user IDs and occupation codes in f-strings throughout the codebase. Examples: `logger.error(f"Error setting current occupation: {e}")` may include user-identifying context in the exception. `logger.info(f"Caching {onet_code}")` logs occupation exploration patterns. In production, structured logging with PII redaction should be used. |
| `metadata_json` is unstructured | **LOW** | Both `user_profile.metadata_json` and `user_feedback.metadata_json` accept arbitrary JSON. Without schema validation on these fields, there is no way to audit what data is being stored or ensure compliance with data minimization principles. |
| No consent tracking | **MEDIUM** | There is no mechanism to record user consent for data processing, which is required under GDPR and similar frameworks. |

*Source: `/home/user/skillsprout/app/models/models.py`, `/home/user/skillsprout/app/api/endpoints.py`*

---

## 7. Prioritized Remediation Plan

### CRITICAL (Address Immediately)

| # | Finding | Remediation |
|---|---------|-------------|
| C-1 | No authentication | Implement JWT-based authentication with a proper identity provider. At minimum, add API key authentication as a stopgap. Every user-scoped endpoint must verify the caller's identity. |
| C-2 | No authorization / access control | After authentication is in place, add ownership checks: a user can only access/modify their own data (`user_id` in JWT must match `user_id` in path). Admin endpoints (e.g., model status) should require an admin role. |

### HIGH (Address Before Production)

| # | Finding | Remediation |
|---|---------|-------------|
| H-1 | Recommendation endpoint loads all occupations | Implement candidate pre-filtering (e.g., by job zone proximity, pre-computed similarity) or pagination. Consider a materialized view or precomputed scoring table. |
| H-2 | No cache invalidation for O\*NET data | Add a staleness check against `last_fetched_at` with a configurable TTL (e.g., 30 days). Schedule periodic cache refresh via Celery Beat. |
| H-3 | Exception details exposed to clients | Replace `detail=str(e)` with generic error messages in production. Log the full exception server-side. Use a custom exception handler middleware. |
| H-4 | CI lint/security jobs never fail builds | Remove `continue-on-error: true` from lint and security steps, or at minimum configure severity thresholds (e.g., only continue-on-error for LOW findings). |
| H-5 | No rate limiting | Add `slowapi` or equivalent middleware. Recommended limits: profile creation (10/min/IP), recommendations (5/min/user), feedback (30/min/user), search (60/min/IP). |
| H-6 | Hardcoded database passwords in docker-compose.yml | Move credentials to a `.env` file referenced by `env_file:` or use Docker secrets. Never commit credentials to version control. |
| H-7 | CORS allows all origins with credentials | Restrict `allow_origins` to the specific frontend domain(s). Make it configurable via environment variable. |
| H-8 | No data retention policy or user deletion | Implement a data retention schedule and a `DELETE /user/{user_id}` endpoint. Add a Celery task to purge data older than the retention period. |

### MEDIUM (Address in Next Sprint)

| # | Finding | Remediation |
|---|---------|-------------|
| M-1 | `asyncio.run()` in Celery tasks | Refactor O\*NET client to provide synchronous methods for use in Celery tasks, or use a single event loop per worker. |
| M-2 | Missing FK indexes | Add indexes on `user_current_occupation.onet_code`, `user_skill_rating.element_id`, `user_feedback.target_onet_code`, `recommended_occupation.target_onet_code`. |
| M-3 | Default connection pool sizing | Explicitly configure `pool_size`, `max_overflow`, and `pool_recycle` based on expected concurrent load and PostgreSQL `max_connections`. |
| M-4 | `debug=True` by default | Change default to `False`. Ensure production environment explicitly sets `DEBUG=false`. |
| M-5 | httpx client created per request | Instantiate `httpx.AsyncClient` once (e.g., as an application-level singleton or a lifespan-managed resource) and reuse across requests. |
| M-6 | No encryption at rest or in transit (internal) | Enable PostgreSQL TDE or storage-level encryption. Configure TLS for PostgreSQL and Redis connections in production. |
| M-7 | Logs may contain PII | Adopt structured logging (e.g., `structlog`) with PII field redaction. Avoid logging user IDs and occupation codes at INFO level. |
| M-8 | No deployment stage in CI/CD | Add a deployment job that triggers on merges to `main`, using environment-specific configurations. |

### LOW (Backlog)

| # | Finding | Remediation |
|---|---------|-------------|
| L-1 | Redundant PK indexes | Remove explicit `index=True` from primary key columns (`occupation.onet_code`, `skill.element_id`). |
| L-2 | Single monolithic Alembic migration | No action needed for existing migration. Ensure future schema changes are incremental. |
| L-3 | Skill rating upsert is per-row | Refactor to use bulk `INSERT ... ON CONFLICT DO UPDATE` for better performance. |
| L-4 | Health check does not verify connectivity | Make the `/health` endpoint actually ping PostgreSQL (`SELECT 1`) and Redis (`PING`). |
| L-5 | `onet_code` path parameter not format-validated | Add a regex validator (e.g., `r"^\d{2}-\d{4}\.\d{2}$"`) to O\*NET code path parameters. |

---

## Appendix A: File Reference

| File | Role |
|------|------|
| `/home/user/skillsprout/app/models/models.py` | SQLAlchemy ORM models (10 tables) |
| `/home/user/skillsprout/app/api/endpoints.py` | FastAPI route handlers |
| `/home/user/skillsprout/app/core/config.py` | pydantic-settings configuration |
| `/home/user/skillsprout/app/db/session.py` | Database engine and session factories |
| `/home/user/skillsprout/app/tasks/tasks.py` | Celery background tasks |
| `/home/user/skillsprout/app/tasks/celery_app.py` | Celery application configuration |
| `/home/user/skillsprout/app/services/onet_client.py` | O\*NET API client (async + mock) |
| `/home/user/skillsprout/app/main.py` | FastAPI application entry point |
| `/home/user/skillsprout/app/schemas/schemas.py` | Pydantic request/response schemas |
| `/home/user/skillsprout/app/ml/scoring.py` | Baseline scoring engine |
| `/home/user/skillsprout/app/ml/calibration.py` | Calibration model (logistic regression) |
| `/home/user/skillsprout/Dockerfile` | Multi-stage Docker build |
| `/home/user/skillsprout/docker-compose.yml` | Development service orchestration |
| `/home/user/skillsprout/docker-compose.test.yml` | Test environment orchestration |
| `/home/user/skillsprout/.github/workflows/test.yml` | CI/CD pipeline |
| `/home/user/skillsprout/.dockerignore` | Docker build context exclusions |
| `/home/user/skillsprout/.gitignore` | Git tracked file exclusions |
| `/home/user/skillsprout/alembic/versions/20260207_0430_d1316f64dcb1_initial_database_schema.py` | Initial database migration |

## Appendix B: Threat Model Summary

| Threat | Likelihood | Impact | Current Mitigation | Gap |
|--------|-----------|--------|-------------------|-----|
| Unauthorized data access | **Certain** | **High** | None | No authentication |
| User impersonation | **Certain** | **High** | None | No authorization |
| Resource exhaustion (DoS) | **Likely** | **Medium** | None | No rate limiting |
| Data exfiltration via API | **Likely** | **High** | None | No access control |
| Stale recommendation data | **Likely** | **Low** | `last_fetched_at` stored | No TTL enforcement |
| SQL injection | **Unlikely** | **High** | SQLAlchemy ORM | Adequate |
| XSS via API responses | **Unlikely** | **Medium** | Pydantic serialization | Adequate |
| Credential leakage | **Possible** | **High** | `.gitignore`, `.dockerignore` | Hardcoded compose passwords |
| Internal exception leakage | **Certain** | **Low** | None | `str(e)` in HTTP responses |
