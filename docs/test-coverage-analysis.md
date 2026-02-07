# Test Coverage Analysis

**Date:** 2026-02-07
**Overall Coverage:** 68% (614 tests passing, 6730 statements, 2150 missed)

## Executive Summary

The SkillSprout test suite has solid coverage of feature modules (ML, events, explainability, training paths) but significant gaps in the **API layer**, **background tasks**, **privacy/GDPR compliance**, and **model lifecycle management**. Eight source files have **0% coverage**, and several GDPR-critical modules sit below 40%.

---

## Current Coverage by Module

### Well-Tested (>80% coverage)

| Module | Coverage | Notes |
|--------|----------|-------|
| `app/models/models.py` | 100% | ORM models fully exercised |
| `app/core/config.py` | 100% | Settings validated |
| `app/tasks/celery_app.py` | 100% | Celery config |
| `app/ml/scoring.py` | 97% | Core scoring engine |
| `app/core/monitoring/request_logging.py` | 97% | |
| `app/features/training_paths/training_catalog.py` | 97% | |
| `ml/model_management/ab_test_framework.py` | 97% | |
| `ml/features/transition_features.py` | 96% | |
| `ml/bias_audit/mitigation_strategies.py` | 96% | |
| `app/features/explainability/bucket_explainer.py` | 96% | |
| `ml/cold_start/cold_start_user.py` | 95% | |
| `app/features/skills_translator/skills_translator.py` | 95% | |
| `ml/transition_graph/graph_recommendations.py` | 95% | |
| `app/core/privacy/data_classification.py` | 95% | |
| `app/core/progressive/session_resumption.py` | 95% | |
| `app/core/progressive/accessibility_middleware.py` | 94% | |
| `app/events/implicit_signals.py` | 94% | |
| `ml/evaluation/eval_framework.py` | 94% | |
| `ml/bias_audit/audit_framework.py` | 93% | |
| `ml/bias_audit/audit_report.py` | 93% | |
| `app/core/progressive/lightweight_api.py` | 93% | |
| `ml/cold_start/cold_start_occupation.py` | 90% | |
| `app/features/training_paths/path_generator.py` | 90% | |
| `app/events/pairwise_preference.py` | 87% | |
| `ml/evaluation/generate_synthetic_interactions.py` | 87% | |
| `app/features/training_paths/resource_filter.py` | 85% | |
| `ml/transition_graph/graph_queries.py` | 81% | |

### Under-Tested (40-80% coverage)

| Module | Coverage | Key Gaps |
|--------|----------|----------|
| `app/features/user_profile/return_engagement.py` | 78% | Engagement metric edge cases |
| `ml/transition_graph/graph_builder.py` | 77% | Bulk graph building, serialization |
| `ml/model_management/calibration_monitor.py` | 75% | Weekly Celery task, report generation |
| `app/core/privacy/private_mode.py` | 71% | Full session lifecycle |
| `app/core/monitoring/metrics.py` | 68% | Counter/histogram recording |
| `app/features/user_profile/profile.py` | 67% | Profile update/patch logic |
| `ml/evaluation/eval_runner.py` | 59% | V2 calibration scoring, CLI entry point |
| `app/ml/calibration.py` | 58% | Model training, prediction, persistence |
| `app/core/privacy/data_deletion.py` | 56% | Cascading deletion, verification |
| `app/core/privacy/data_export.py` | 55% | Entire export endpoint |
| `app/core/progressive/offline_capability.py` | 53% | CSV export endpoint |
| `app/features/user_profile/progress_tracker.py` | 45% | Skill update, re-scoring, metrics |
| `app/db/session.py` | 45% | Exception handling, rollback |
| `app/core/monitoring/health_checks.py` | 42% | All error/degraded paths |
| `ml/model_management/ab_test_framework.py` | 42%* | (see note) |

### Zero Coverage (0%)

| Module | Statements | Description |
|--------|------------|-------------|
| `app/api/endpoints.py` | 250 | **Core REST API endpoints** |
| `app/main.py` | 63 | FastAPI app initialization, routing |
| `app/schemas/schemas.py` | 141 | Pydantic request/response models |
| `app/services/onet_client.py` | 155 | O*NET external API client |
| `app/tasks/tasks.py` | 132 | Celery background tasks |
| `app/features/explainability/api.py` | 179 | Explainability API routes |
| `app/features/skills_translator/api.py` | 35 | Skills translator API routes |
| `app/features/training_paths/api.py` | 79 | Training paths API routes |

### Critically Low (<40%)

| Module | Coverage | Description |
|--------|----------|-------------|
| `app/core/privacy/retention_policy.py` | 36% | GDPR retention enforcement |
| `app/features/user_profile/saved_occupations.py` | 35% | Saved occupation CRUD |
| `ml/model_management/model_registry.py` | 29% | Model lifecycle management |

---

## Priority Improvement Areas

### Priority 1 (Critical) -- API Endpoint Tests

**Files:** `app/api/endpoints.py` (0%), all `*/api.py` routes (0%)

**Why:** The entire HTTP API layer has zero test coverage. This means:
- Request validation is untested (malformed inputs could crash the server)
- Response schemas are never verified against the contract
- Error handling paths (404, 400, 502) are unexercised
- Database transaction behavior under API calls is unverified

**Recommended tests:**
- Use `fastapi.testclient.TestClient` (or `httpx.AsyncClient`) to test each endpoint
- Test the full recommendation flow: create user -> set occupation -> rate skills -> get recommendations
- Test error responses: missing user (404), missing skills (400), O*NET failure (502)
- Test request validation: invalid ratings, empty search queries, out-of-bound limits
- Test response schema compliance (verify JSON structure matches Pydantic schemas)

**Estimated scope:** ~40-50 test cases covering all 30+ endpoints

---

### Priority 2 (Critical) -- GDPR/Privacy Compliance Tests

**Files:**
- `app/core/privacy/retention_policy.py` (36%)
- `app/core/privacy/data_deletion.py` (56%)
- `app/core/privacy/data_export.py` (55%)

**Why:** These modules implement legally required GDPR rights (Articles 5, 17, 20). Bugs here could mean:
- User data not actually deleted when requested (Article 17 violation)
- Expired data retained past policy limits (Article 5(1)(e) violation)
- Incomplete data exports missing user information (Article 20 violation)

**Recommended tests:**
- **Retention policy:** Test `purge_expired_records()` and `enforce_all_retention_rules()` with records at/past cutoff dates. Verify dry-run mode is truly read-only. Verify cascade deletions propagate correctly.
- **Data deletion:** Test cascading deletion across all 6+ tables. Verify post-deletion verification catches incomplete deletes. Test partial failure recovery.
- **Data export:** Test full export endpoint with populated user data across all tables. Verify no internal fields leak. Verify correct counts per category.

**Estimated scope:** ~30-40 test cases

---

### Priority 3 (High) -- Model Lifecycle Management

**Files:**
- `ml/model_management/model_registry.py` (29%)
- `app/ml/calibration.py` (58%)

**Why:** The model registry manages production ML model deployment. With only 29% coverage:
- Model promotion (candidate -> production) is untested -- could deploy broken models
- Model rollback is untested -- can't recover from bad deployments
- Duplicate version registration is not verified
- Active model queries are untested -- serving could use wrong model

**Recommended tests:**
- Test full model lifecycle: register -> promote -> rollback
- Test atomic promotion (demotion of current + promotion of new in single transaction)
- Test duplicate version rejection
- Test `get_active_model()` returns the production model
- Test model persistence save/load roundtrip for calibration models

**Estimated scope:** ~20-25 test cases

---

### Priority 4 (High) -- Background Task Tests

**Files:**
- `app/tasks/tasks.py` (0%)

**Why:** Celery tasks run asynchronously and failures may not surface immediately:
- `warm_occupation_cache` writes to the database -- untested DB consistency
- `train_calibration_model_task` trains and persists models -- untested training pipeline
- Partial failures (one occupation fails, others succeed) could corrupt cache

**Recommended tests:**
- Test `warm_occupation_cache` with mocked O*NET client, verify DB records created
- Test partial failure handling (one occupation fails, others still cached)
- Test `train_calibration_model_task` with sufficient/insufficient samples
- Test feature extraction from score JSON with various data shapes

**Estimated scope:** ~15-20 test cases

---

### Priority 5 (High) -- User Profile Feature Tests

**Files:**
- `app/features/user_profile/saved_occupations.py` (35%)
- `app/features/user_profile/progress_tracker.py` (45%)

**Why:** These are core user-facing features:
- Saved occupations CRUD is mostly untested (save, list, update, delete)
- Progress tracking re-scoring logic is untested
- Skill gain counting and bucket improvement detection are untested

**Recommended tests:**
- Test full saved occupation lifecycle: save -> list -> update -> delete
- Test duplicate save prevention (409 conflict)
- Test progress tracking: new skill gains, bucket improvements, time estimates
- Test re-scoring of saved occupations when skills change

**Estimated scope:** ~20-25 test cases

---

### Priority 6 (Medium) -- Health Check Error Paths

**File:** `app/core/monitoring/health_checks.py` (42%)

**Why:** All component failure paths (Postgres down, Redis down, Celery unresponsive) are untested. In production, these are the exact paths that execute during incidents -- when reliability matters most.

**Recommended tests:**
- Test each component check with simulated failures
- Test status aggregation (single failure -> degraded, critical failure -> unhealthy)
- Test `/health/ready` returns 503 when critical components fail
- Test uptime calculation edge cases

**Estimated scope:** ~15 test cases

---

### Priority 7 (Medium) -- O*NET Client and Schema Validation

**Files:**
- `app/services/onet_client.py` (0%)
- `app/schemas/schemas.py` (0%)

**Why:**
- The O*NET client has retry logic, error classification, and response parsing that are completely untested
- Pydantic schemas define the API contract but validators and constraints are never exercised

**Recommended tests:**
- Mock httpx responses to test success, 401, 429, 500, and timeout scenarios
- Verify retry logic with exponential backoff
- Test MockONetClient returns valid data matching real API schema
- Test Pydantic validators (action_type validation, rating bounds, length constraints)
- Test ORM-to-schema serialization with `from_attributes=True`

**Estimated scope:** ~25-30 test cases

---

## Structural Test Gaps

### 1. No API-Level Integration Tests

The existing integration tests (`test_full_pipeline.py`, `test_recommendation_flow.py`) test the scoring logic with direct function calls but never go through the HTTP layer. There is no `TestClient` usage anywhere in the test suite. This means:
- Middleware (CORS, request logging) is untested
- Route mounting in `main.py` is unverified
- Request/response serialization through FastAPI is unexercised

### 2. No Negative/Error Path Testing for APIs

All existing tests focus on happy paths. There are no tests verifying:
- 404 responses for missing resources
- 400 responses for invalid input
- 502 responses for external service failures
- 409 responses for duplicate resources

### 3. No Celery Task Tests

Background tasks are entirely untested. The `tasks.py` module handles cache warming and model training -- both are write-heavy operations that modify database state.

### 4. Missing Database Transaction Safety Tests

The `get_db()` and `get_sync_db()` generators in `session.py` have untested exception/rollback paths. If these fail silently, database connections could leak.

### 5. No End-to-End Flow Tests Through HTTP

The e2e test script (`scripts/test_e2e.sh`) exists but uses curl against a live service. There are no pytest-based e2e tests that exercise the full stack (HTTP -> API -> Service -> DB) in an isolated test environment.

---

## Recommended Test Implementation Order

| Order | Area | New Tests | Coverage Impact |
|-------|------|-----------|-----------------|
| 1 | API endpoints (`TestClient`) | ~50 | +250 stmts covered |
| 2 | GDPR privacy modules | ~35 | +160 stmts covered |
| 3 | Model registry + calibration | ~25 | +220 stmts covered |
| 4 | Celery tasks | ~18 | +132 stmts covered |
| 5 | User profile features | ~25 | +150 stmts covered |
| 6 | Health check error paths | ~15 | +90 stmts covered |
| 7 | O*NET client + schemas | ~30 | +296 stmts covered |
| **Total** | | **~198** | **~1,298 stmts (+19% coverage -> ~87%)** |

---

## Quick Wins

These changes would immediately improve coverage with minimal effort:

1. **Add `TestClient` smoke tests** for each API route (just check 200/201 status codes) -- covers `endpoints.py`, `main.py`, and all `api.py` files
2. **Test Pydantic schema validation** -- pure unit tests, no DB needed, covers `schemas.py`
3. **Test `MockONetClient`** -- already exists, just needs assertions on return values
4. **Add error path tests to existing test files** -- e.g., test `user_not_found` 404 in `test_user_profile.py`
