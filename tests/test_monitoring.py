"""Tests for the monitoring subsystem.

Covers:
  - Health checks (liveness, readiness, detailed)
  - Prometheus metrics (recording, middleware, /metrics endpoint)
  - Alerting rules (YAML generation, threshold constants)
  - Request logging middleware (structured logs, hashed user IDs, correlation IDs)
"""
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Health Checks
# ---------------------------------------------------------------------------


class TestHealthChecks:
    """Tests for app.core.monitoring.health_checks."""

    def _make_app(self):
        """Create a minimal FastAPI app with health routes."""
        from app.core.monitoring.health_checks import router

        app = FastAPI()
        app.include_router(router)
        return app

    def test_liveness_returns_200(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"

    def test_liveness_response_schema(self):
        app = self._make_app()
        client = TestClient(app)
        data = client.get("/health").json()
        assert set(data.keys()) == {"status", "timestamp", "version"}

    @patch("app.core.monitoring.health_checks._run_all_checks")
    def test_readiness_all_healthy(self, mock_checks):
        from app.core.monitoring.health_checks import ComponentHealth

        mock_checks.return_value = [
            ComponentHealth(name="postgresql", status="healthy"),
            ComponentHealth(name="redis", status="healthy"),
            ComponentHealth(name="celery", status="healthy"),
            ComponentHealth(name="onet_cache", status="healthy"),
            ComponentHealth(name="ml_model", status="healthy"),
        ]
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["components"]["postgresql"] == "healthy"

    @patch("app.core.monitoring.health_checks._run_all_checks")
    def test_readiness_critical_unhealthy_returns_503(self, mock_checks):
        from app.core.monitoring.health_checks import ComponentHealth

        mock_checks.return_value = [
            ComponentHealth(name="postgresql", status="unhealthy", detail="connection refused"),
            ComponentHealth(name="redis", status="healthy"),
            ComponentHealth(name="celery", status="healthy"),
            ComponentHealth(name="onet_cache", status="healthy"),
            ComponentHealth(name="ml_model", status="healthy"),
        ]
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"

    @patch("app.core.monitoring.health_checks._run_all_checks")
    def test_readiness_non_critical_degraded_still_ready(self, mock_checks):
        from app.core.monitoring.health_checks import ComponentHealth

        mock_checks.return_value = [
            ComponentHealth(name="postgresql", status="healthy"),
            ComponentHealth(name="redis", status="degraded"),
            ComponentHealth(name="celery", status="degraded"),
            ComponentHealth(name="onet_cache", status="degraded"),
            ComponentHealth(name="ml_model", status="healthy"),
        ]
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    @patch("app.core.monitoring.health_checks._run_all_checks")
    def test_detailed_health_includes_latency(self, mock_checks):
        from app.core.monitoring.health_checks import ComponentHealth

        mock_checks.return_value = [
            ComponentHealth(name="postgresql", status="healthy", latency_ms=1.5),
            ComponentHealth(name="redis", status="healthy", latency_ms=0.8),
            ComponentHealth(name="celery", status="healthy", latency_ms=50.0),
            ComponentHealth(name="onet_cache", status="healthy", latency_ms=2.0),
            ComponentHealth(name="ml_model", status="healthy", latency_ms=0.1),
        ]
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health/detailed")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert len(data["components"]) == 5
        pg = next(c for c in data["components"] if c["name"] == "postgresql")
        assert pg["latency_ms"] == 1.5

    @patch("app.core.monitoring.health_checks._run_all_checks")
    def test_detailed_health_unhealthy_returns_503(self, mock_checks):
        from app.core.monitoring.health_checks import ComponentHealth

        mock_checks.return_value = [
            ComponentHealth(name="postgresql", status="unhealthy"),
            ComponentHealth(name="redis", status="healthy"),
            ComponentHealth(name="celery", status="healthy"),
            ComponentHealth(name="onet_cache", status="healthy"),
            ComponentHealth(name="ml_model", status="healthy"),
        ]
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health/detailed")
        assert resp.status_code == 503
        assert resp.json()["status"] == "unhealthy"

    def test_mark_startup_sets_uptime(self):
        from app.core.monitoring.health_checks import mark_startup, _startup_time

        mark_startup()
        from app.core.monitoring import health_checks

        assert health_checks._startup_time is not None


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for app.core.monitoring.metrics."""

    def test_record_scoring(self):
        from app.core.monitoring.metrics import (
            SCORING_DURATION,
            SCORES_BY_BUCKET,
            record_scoring,
        )

        record_scoring("v1_baseline", "ready_now", 0.05)
        # Histogram should have observed a value
        sample = SCORING_DURATION.labels(model_version="v1_baseline")
        assert sample is not None

    def test_record_cold_start_fallback(self):
        from app.core.monitoring.metrics import (
            COLD_START_FALLBACKS_TOTAL,
            record_cold_start_fallback,
        )

        before = COLD_START_FALLBACKS_TOTAL._value.get()
        record_cold_start_fallback()
        after = COLD_START_FALLBACKS_TOTAL._value.get()
        assert after == before + 1

    def test_record_feedback(self):
        from app.core.monitoring.metrics import (
            FEEDBACK_EVENTS_TOTAL,
            record_feedback,
        )

        record_feedback("click")
        sample = FEEDBACK_EVENTS_TOTAL.labels(action_type="click")
        assert sample is not None

    def test_record_feedback_latency(self):
        from app.core.monitoring.metrics import record_feedback_latency

        record_feedback_latency(120.5)

    def test_set_model_version_info(self):
        from app.core.monitoring.metrics import MODEL_VERSION_INFO, set_model_version

        set_model_version("v2_calibrated", "2025-01-01", "true")
        # Info metric should be set

    def test_normalise_path_numeric(self):
        from app.core.monitoring.metrics import _normalise_path

        assert _normalise_path("/api/v1/user/42/recommendations") == "/api/v1/user/:id/recommendations"

    def test_normalise_path_uuid(self):
        from app.core.monitoring.metrics import _normalise_path

        path = "/api/v1/session/550e8400-e29b-41d4-a716-446655440000"
        assert ":id" in _normalise_path(path)

    def test_metrics_endpoint(self):
        from app.core.monitoring.metrics import metrics_router

        app = FastAPI()
        app.include_router(metrics_router)
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "request_duration_seconds" in resp.text or "# HELP" in resp.text

    def test_metrics_middleware(self):
        from app.core.monitoring.metrics import metrics_middleware

        app = FastAPI()
        app.middleware("http")(metrics_middleware)

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Alerting Rules
# ---------------------------------------------------------------------------


class TestAlertingRules:
    """Tests for app.core.monitoring.alerting_rules."""

    def test_alerting_rules_list_not_empty(self):
        from app.core.monitoring.alerting_rules import ALERTING_RULES

        assert len(ALERTING_RULES) >= 5

    def test_all_rules_have_required_fields(self):
        from app.core.monitoring.alerting_rules import ALERTING_RULES

        for rule in ALERTING_RULES:
            assert rule.alert
            assert rule.expr
            assert rule.for_duration
            assert rule.severity in ("P1", "P2", "P3")
            assert rule.summary
            assert rule.description

    def test_p1_rules_exist(self):
        from app.core.monitoring.alerting_rules import ALERTING_RULES

        p1 = [r for r in ALERTING_RULES if r.severity == "P1"]
        assert len(p1) >= 2

    def test_generate_yaml_output(self):
        from app.core.monitoring.alerting_rules import generate_prometheus_rules_yaml

        yaml_str = generate_prometheus_rules_yaml()
        assert "groups:" in yaml_str
        assert "skillsprout_alerts" in yaml_str
        assert "HealthEndpointDown" in yaml_str
        assert "ScoringLatencyP99TooHigh" in yaml_str
        assert "severity: P1" in yaml_str

    def test_generate_yaml_valid_structure(self):
        import yaml
        from app.core.monitoring.alerting_rules import generate_prometheus_rules_yaml

        yaml_str = generate_prometheus_rules_yaml()
        parsed = yaml.safe_load(yaml_str)
        assert "groups" in parsed
        assert len(parsed["groups"]) == 1
        rules = parsed["groups"][0]["rules"]
        assert len(rules) >= 5

        for rule in rules:
            assert "alert" in rule
            assert "expr" in rule
            assert "for" in rule
            assert "labels" in rule
            assert "severity" in rule["labels"]

    def test_threshold_constants(self):
        from app.core.monitoring.alerting_rules import (
            HEALTH_FAIL_MAX_SECONDS,
            SCORING_P99_MAX_SECONDS,
            COLD_START_FALLBACK_MAX_RATIO,
            CALIBRATION_STALE_MAX_SECONDS,
            ONET_CACHE_STALE_MAX_SECONDS,
        )

        assert HEALTH_FAIL_MAX_SECONDS == 60
        assert SCORING_P99_MAX_SECONDS == 2.0
        assert COLD_START_FALLBACK_MAX_RATIO == 0.50
        assert CALIBRATION_STALE_MAX_SECONDS == 7 * 86400
        assert ONET_CACHE_STALE_MAX_SECONDS == 30 * 86400

    def test_write_rules_file(self, tmp_path):
        from app.core.monitoring.alerting_rules import write_prometheus_rules_file

        dest = str(tmp_path / "rules.yml")
        result_path = write_prometheus_rules_file(path=dest)
        assert result_path.endswith("rules.yml")

        import yaml

        with open(dest) as fh:
            parsed = yaml.safe_load(fh)
        assert "groups" in parsed


# ---------------------------------------------------------------------------
# Request Logging
# ---------------------------------------------------------------------------


class TestRequestLogging:
    """Tests for app.core.monitoring.request_logging."""

    def test_hash_user_id_deterministic(self):
        from app.core.monitoring.request_logging import _hash_user_id

        h1 = _hash_user_id("42")
        h2 = _hash_user_id("42")
        assert h1 == h2
        assert h1 is not None
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_user_id_none(self):
        from app.core.monitoring.request_logging import _hash_user_id

        assert _hash_user_id(None) is None

    def test_hash_user_id_different_users(self):
        from app.core.monitoring.request_logging import _hash_user_id

        assert _hash_user_id("1") != _hash_user_id("2")

    def test_sensitive_path_detection(self):
        from app.core.monitoring.request_logging import _is_sensitive_path

        assert _is_sensitive_path("/api/v1/user/42/skills/ratings")
        assert _is_sensitive_path("/api/v1/user/99/recommendations")
        assert not _is_sensitive_path("/api/v1/health")
        assert not _is_sensitive_path("/api/v1/occupations/search")

    def test_structured_access_record(self):
        from app.core.monitoring.request_logging import StructuredAccessRecord

        record = StructuredAccessRecord(
            request_id="req-123",
            correlation_id="corr-456",
            method="GET",
            path="/api/v1/health",
            query="",
            status_code=200,
            duration_ms=12.5,
            user_id_hash="abc123",
            client_ip="127.0.0.1",
        )
        log_json = record.to_json()
        parsed = json.loads(log_json)
        assert parsed["request_id"] == "req-123"
        assert parsed["status_code"] == 200
        assert parsed["duration_ms"] == 12.5

    def test_correlation_id_context(self):
        from app.core.monitoring.request_logging import (
            get_correlation_id,
            set_correlation_id,
        )

        set_correlation_id("test-cid-789")
        assert get_correlation_id() == "test-cid-789"

    def test_middleware_injects_headers(self):
        from app.core.monitoring.request_logging import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/ping")
        async def ping():
            return {"pong": True}

        client = TestClient(app)
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        assert "x-correlation-id" in resp.headers

    def test_middleware_propagates_request_id(self):
        from app.core.monitoring.request_logging import RequestLoggingMiddleware

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/ping", headers={"X-Request-ID": "my-req-id"})
        assert resp.headers["x-request-id"] == "my-req-id"

    def test_extract_user_id_from_path(self):
        from app.core.monitoring.request_logging import _extract_user_id_from_path

        assert _extract_user_id_from_path("/api/v1/user/42/skills") == "42"
        assert _extract_user_id_from_path("/api/v1/health") is None

    def test_sensitive_path_redacts_query(self):
        from app.core.monitoring.request_logging import StructuredAccessRecord

        record = StructuredAccessRecord(
            request_id="r1",
            correlation_id="c1",
            method="POST",
            path="/api/v1/user/42/skills/ratings",
            query="debug=true",
            status_code=200,
            duration_ms=10.0,
        )
        parsed = json.loads(record.to_json())
        assert parsed["query"] == "[REDACTED]"
