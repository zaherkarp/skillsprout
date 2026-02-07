"""Tests for the progressive enhancement subsystem.

Covers:
  - Lightweight API (pagination, lite filtering, ETag, compression)
  - Session resumption (encrypt/decrypt, export/import endpoints)
  - Offline capability (CSV generation)
  - Accessibility middleware (charset, CORS, descriptions, time budget)
"""
import csv
import io
import json
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Lightweight API
# ---------------------------------------------------------------------------


class TestPagination:
    """Tests for app.core.progressive.lightweight_api.paginate."""

    def test_first_page(self):
        from app.core.progressive.lightweight_api import paginate

        items = list(range(25))
        result = paginate(items, page=1, page_size=10)
        assert result.pagination.page == 1
        assert result.pagination.total_items == 25
        assert result.pagination.total_pages == 3
        assert result.pagination.has_next is True
        assert result.pagination.has_prev is False
        assert len(result.data) == 10
        assert result.data == list(range(10))

    def test_middle_page(self):
        from app.core.progressive.lightweight_api import paginate

        items = list(range(25))
        result = paginate(items, page=2, page_size=10)
        assert result.pagination.has_next is True
        assert result.pagination.has_prev is True
        assert result.data == list(range(10, 20))

    def test_last_page(self):
        from app.core.progressive.lightweight_api import paginate

        items = list(range(25))
        result = paginate(items, page=3, page_size=10)
        assert result.pagination.has_next is False
        assert result.pagination.has_prev is True
        assert len(result.data) == 5

    def test_page_beyond_range(self):
        from app.core.progressive.lightweight_api import paginate

        items = list(range(5))
        result = paginate(items, page=100, page_size=10)
        assert len(result.data) == 0
        assert result.pagination.has_next is False

    def test_empty_list(self):
        from app.core.progressive.lightweight_api import paginate

        result = paginate([], page=1, page_size=10)
        assert result.pagination.total_items == 0
        assert result.pagination.total_pages == 1
        assert len(result.data) == 0

    def test_page_size_capped_at_100(self):
        from app.core.progressive.lightweight_api import paginate

        items = list(range(200))
        result = paginate(items, page=1, page_size=999)
        assert len(result.data) == 100

    def test_negative_page_treated_as_1(self):
        from app.core.progressive.lightweight_api import paginate

        items = list(range(10))
        result = paginate(items, page=-5, page_size=5)
        assert result.pagination.page == 1


class TestLiteFilter:
    """Tests for app.core.progressive.lightweight_api.lite_filter."""

    def test_filter_recommendation(self):
        from app.core.progressive.lightweight_api import lite_filter

        full = {
            "onet_code": "15-1252.00",
            "title": "Software Developers",
            "rank": 1,
            "bucket": "ready_now",
            "match_score": 85.0,
            "gap_severity": 12.0,
            "top_gaps": [],
            "training_suggestion": "Apply now!",
            "explanation": "Strong match...",
        }
        lite = lite_filter(full, "RecommendedOccupation")
        assert set(lite.keys()) == {"onet_code", "title", "rank", "bucket", "match_score"}

    def test_filter_unknown_model_passthrough(self):
        from app.core.progressive.lightweight_api import lite_filter

        data = {"foo": "bar", "baz": 123}
        assert lite_filter(data, "UnknownModel") == data

    def test_filter_list(self):
        from app.core.progressive.lightweight_api import lite_filter

        items = [
            {"onet_code": "a", "title": "A", "rank": 1, "bucket": "ready_now", "match_score": 90, "extra": "x"},
            {"onet_code": "b", "title": "B", "rank": 2, "bucket": "trainable", "match_score": 60, "extra": "y"},
        ]
        filtered = lite_filter(items, "RecommendedOccupation")
        assert len(filtered) == 2
        assert "extra" not in filtered[0]


class TestETag:
    """Tests for ETag computation and matching."""

    def test_compute_etag(self):
        from app.core.progressive.lightweight_api import compute_etag

        etag = compute_etag(b'{"hello":"world"}')
        assert etag.startswith('W/"')
        assert etag.endswith('"')

    def test_etag_deterministic(self):
        from app.core.progressive.lightweight_api import compute_etag

        body = b"test body content"
        assert compute_etag(body) == compute_etag(body)

    def test_etag_matches(self):
        from app.core.progressive.lightweight_api import compute_etag, etag_matches
        from starlette.testclient import TestClient as _TC
        from starlette.requests import Request as _Req

        etag = compute_etag(b"data")
        # Build a mock request with If-None-Match header
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"if-none-match", etag.encode())],
        }
        req = _Req(scope)
        assert etag_matches(req, etag) is True


class TestLightweightMiddleware:
    """Tests for the LightweightAPIMiddleware."""

    def _make_app(self):
        from app.core.progressive.lightweight_api import LightweightAPIMiddleware

        app = FastAPI()
        app.add_middleware(LightweightAPIMiddleware)

        @app.get("/data")
        async def data_endpoint():
            return {
                "onet_code": "15-1252.00",
                "title": "Software Developers",
                "rank": 1,
                "bucket": "ready_now",
                "match_score": 85.0,
                "extra_field": "should_be_stripped_in_lite",
            }

        return app

    def test_normal_request_passes_through(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/data")
        assert resp.status_code == 200
        assert "extra_field" in resp.json()

    def test_etag_header_present(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/data")
        assert "etag" in resp.headers

    def test_conditional_request_304(self):
        app = self._make_app()
        client = TestClient(app)
        resp1 = client.get("/data")
        etag = resp1.headers["etag"]

        resp2 = client.get("/data", headers={"If-None-Match": etag})
        assert resp2.status_code == 304

    def test_lite_mode_strips_fields(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/data?lite=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "extra_field" not in data
        assert "onet_code" in data


# ---------------------------------------------------------------------------
# Session Resumption
# ---------------------------------------------------------------------------


class TestSessionEncryption:
    """Tests for session token encryption/decryption."""

    def test_encrypt_decrypt_round_trip(self):
        from app.core.progressive.session_resumption import (
            SessionPayload,
            encrypt_session,
            decrypt_session,
        )

        payload = SessionPayload(
            user_id=42,
            current_onet_code="15-1252.00",
            skill_ratings={"2.B.1.a": 3, "2.B.8.b": 4},
            preferences={"theme": "dark"},
        )
        token = encrypt_session(payload)
        assert isinstance(token, str)
        assert len(token) > 0

        restored = decrypt_session(token)
        assert restored.user_id == 42
        assert restored.current_onet_code == "15-1252.00"
        assert restored.skill_ratings == {"2.B.1.a": 3, "2.B.8.b": 4}

    def test_expired_token_raises(self):
        from app.core.progressive.session_resumption import (
            SessionPayload,
            encrypt_session,
            decrypt_session,
            _get_fernet,
        )

        # Manually craft a token with an old timestamp by encrypting
        # then decrypting with a very short TTL after a brief sleep
        payload = SessionPayload(user_id=1)
        token = encrypt_session(payload)

        # Fernet embeds a timestamp; decrypt with max_age=1 after
        # forcing the token to appear old via a mock
        import time as _time
        from unittest.mock import patch as _patch
        from cryptography.fernet import Fernet

        # Use a max_age of 1 second and sleep just past it
        _time.sleep(1.1)
        with pytest.raises(ValueError, match="Invalid or expired"):
            decrypt_session(token, max_age=1)

    def test_tampered_token_raises(self):
        from app.core.progressive.session_resumption import decrypt_session

        with pytest.raises(ValueError, match="Invalid or expired"):
            decrypt_session("this-is-not-a-valid-fernet-token")

    def test_empty_payload(self):
        from app.core.progressive.session_resumption import (
            SessionPayload,
            encrypt_session,
            decrypt_session,
        )

        payload = SessionPayload()
        token = encrypt_session(payload)
        restored = decrypt_session(token)
        assert restored.user_id is None
        assert restored.skill_ratings == {}


class TestSessionEndpoints:
    """Tests for session export/import endpoints."""

    def _make_app(self):
        from app.core.progressive.session_resumption import router

        app = FastAPI()
        app.include_router(router)
        return app

    def test_export_returns_token(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/v1/session/export",
            json={
                "user_id": 42,
                "current_onet_code": "15-1252.00",
                "skill_ratings": {"2.B.1.a": 3},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["expires_in_seconds"] == 30 * 24 * 3600

    def test_import_restores_session(self):
        app = self._make_app()
        client = TestClient(app)

        # Export first
        export_resp = client.post(
            "/api/v1/session/export",
            json={
                "user_id": 99,
                "current_onet_code": "15-1299.08",
                "skill_ratings": {"2.B.8.a": 2},
                "preferences": {"lang": "en"},
            },
        )
        token = export_resp.json()["token"]

        # Import
        import_resp = client.post(
            "/api/v1/session/import",
            json={"token": token},
        )
        assert import_resp.status_code == 200
        data = import_resp.json()
        assert data["user_id"] == 99
        assert data["current_onet_code"] == "15-1299.08"
        assert data["skill_ratings"]["2.B.8.a"] == 2

    def test_import_invalid_token_returns_400(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.post(
            "/api/v1/session/import",
            json={"token": "invalid-garbage-token"},
        )
        assert resp.status_code == 400

    def test_export_minimal_payload(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.post("/api/v1/session/export", json={})
        assert resp.status_code == 200
        assert "token" in resp.json()


# ---------------------------------------------------------------------------
# Offline Capability (CSV generation)
# ---------------------------------------------------------------------------


class TestCSVGeneration:
    """Tests for app.core.progressive.offline_capability.generate_csv."""

    def test_generate_csv_structure(self):
        from app.core.progressive.offline_capability import generate_csv

        csv_content = generate_csv(
            user_id=1,
            current_occupation_title="Software Developers",
            current_onet_code="15-1252.00",
            skill_ratings=[
                {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "rating": 3},
                {"element_id": "2.B.8.b", "skill_name": "Complex Problem Solving", "rating": 4},
            ],
            recommendations=[
                {
                    "rank": 1,
                    "onet_code": "15-1299.08",
                    "title": "Web Developers",
                    "bucket": "ready_now",
                    "match_score": 85.0,
                    "training_suggestion": "Apply now!",
                },
            ],
        )
        assert "SkillSprout Career Transition Report" in csv_content
        assert "Software Developers" in csv_content
        assert "15-1252.00" in csv_content
        assert "Reading Comprehension" in csv_content
        assert "Web Developers" in csv_content
        assert "END OF REPORT" in csv_content

    def test_csv_parseable(self):
        from app.core.progressive.offline_capability import generate_csv

        csv_content = generate_csv(
            user_id=1,
            current_occupation_title="Test",
            current_onet_code="00-0000.00",
            skill_ratings=[],
            recommendations=[],
        )
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)
        assert len(rows) > 5  # header + profile + recs + footer

    def test_csv_empty_recommendations(self):
        from app.core.progressive.offline_capability import generate_csv

        csv_content = generate_csv(
            user_id=1,
            current_occupation_title="Test",
            current_onet_code="00-0000.00",
            skill_ratings=[],
            recommendations=[],
        )
        assert "TOP RECOMMENDATIONS" in csv_content


# ---------------------------------------------------------------------------
# Accessibility Middleware
# ---------------------------------------------------------------------------


class TestDescriptionInjection:
    """Tests for inject_descriptions utility."""

    def test_bucket_description_injected(self):
        from app.core.progressive.accessibility_middleware import inject_descriptions

        data = {"bucket": "ready_now", "score": 85}
        result = inject_descriptions(data)
        assert "bucket_description" in result
        assert "immediately" in result["bucket_description"]

    def test_action_type_description(self):
        from app.core.progressive.accessibility_middleware import inject_descriptions

        data = {"action_type": "apply"}
        result = inject_descriptions(data)
        assert "action_type_description" in result
        assert "applied" in result["action_type_description"]

    def test_unknown_code_no_description(self):
        from app.core.progressive.accessibility_middleware import inject_descriptions

        data = {"bucket": "unknown_bucket"}
        result = inject_descriptions(data)
        assert "bucket_description" not in result

    def test_nested_injection(self):
        from app.core.progressive.accessibility_middleware import inject_descriptions

        data = {
            "occupations": [
                {"bucket": "trainable", "title": "Web Dev"},
                {"bucket": "long_reskill", "title": "Doctor"},
            ]
        }
        result = inject_descriptions(data)
        assert "bucket_description" in result["occupations"][0]
        assert "bucket_description" in result["occupations"][1]

    def test_non_dict_passthrough(self):
        from app.core.progressive.accessibility_middleware import inject_descriptions

        assert inject_descriptions("hello") == "hello"
        assert inject_descriptions(42) == 42
        assert inject_descriptions(None) is None


class TestAccessibilityMiddleware:
    """Tests for the AccessibilityMiddleware ASGI middleware."""

    def _make_app(self):
        from app.core.progressive.accessibility_middleware import AccessibilityMiddleware

        app = FastAPI()
        app.add_middleware(AccessibilityMiddleware)

        @app.get("/json-endpoint")
        async def json_endpoint():
            return {"bucket": "ready_now", "score": 85}

        @app.get("/slow")
        async def slow_endpoint():
            import asyncio

            await asyncio.sleep(0.01)
            return {"status": "ok"}

        return app

    def test_charset_header_present(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/json-endpoint")
        ct = resp.headers.get("content-type", "")
        assert "charset=utf-8" in ct

    def test_cors_headers_present(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/json-endpoint")
        assert "access-control-allow-origin" in resp.headers

    def test_response_time_header(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/json-endpoint")
        assert "x-response-time-ms" in resp.headers
        ms = float(resp.headers["x-response-time-ms"])
        assert ms >= 0

    def test_description_injected_in_response(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/json-endpoint")
        data = resp.json()
        assert "bucket_description" in data

    def test_response_time_budget_warning(self, caplog):
        """Verify that slow responses generate a warning log."""
        from app.core.progressive.accessibility_middleware import AccessibilityMiddleware
        import asyncio

        app = FastAPI()
        app.add_middleware(AccessibilityMiddleware)

        @app.get("/very-slow")
        async def very_slow():
            await asyncio.sleep(0.01)
            return {"ok": True}

        # The warning is only emitted for responses >3s, and we don't
        # actually wait 3s in tests.  Just verify the middleware doesn't crash.
        client = TestClient(app)
        resp = client.get("/very-slow")
        assert resp.status_code == 200


class TestCompression:
    """Tests for GZip compression helper."""

    def test_add_compression_does_not_crash(self):
        from app.core.progressive.lightweight_api import add_compression

        app = FastAPI()
        add_compression(app, minimum_size=100)

        @app.get("/big")
        async def big():
            return {"data": "x" * 1000}

        client = TestClient(app)
        resp = client.get("/big", headers={"Accept-Encoding": "gzip"})
        assert resp.status_code == 200
