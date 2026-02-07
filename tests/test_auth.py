"""Tests for API key authentication middleware.

Sprint context:
  E3 (Infra) wrote the middleware, E2 (UX) wrote the tests, PM verified
  coverage of open paths, protected paths, and edge cases.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.auth import APIKeyAuthMiddleware


def _make_app(auth_enabled: bool = True, api_key: str = "test-key-123"):
    """Create a test app with auth middleware."""
    app = FastAPI()
    app.add_middleware(APIKeyAuthMiddleware)

    @app.get("/")
    async def home():
        return {"page": "home"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def health_ready():
        return {"ready": True}

    @app.get("/health/detailed")
    async def health_detailed():
        return {"detailed": True}

    @app.get("/metrics")
    async def metrics():
        return {"metrics": "data"}

    @app.get("/docs-page")
    async def docs_page():
        return {"docs": True}

    @app.get("/api/v1/occupations/search")
    async def protected_search():
        return {"results": []}

    @app.post("/api/v1/user/1/recommendations")
    async def protected_recs():
        return {"recommendations": []}

    return app


class TestAuthDisabled:
    """When auth_enabled=False, all requests pass through."""

    def test_protected_route_accessible_without_key(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.auth_enabled = False
            mock_settings.api_key = "test-key"
            app = _make_app()
            client = TestClient(app)
            response = client.get("/api/v1/occupations/search")
            assert response.status_code == 200


class TestAuthEnabled:
    """When auth_enabled=True, protected routes require X-API-Key header."""

    @pytest.fixture
    def client(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.auth_enabled = True
            mock_settings.api_key = "valid-key-123"
            app = _make_app()
            yield TestClient(app)

    def test_protected_route_without_key_returns_401(self, client):
        response = client.get("/api/v1/occupations/search")
        assert response.status_code == 401
        assert "API key" in response.json()["detail"]

    def test_protected_route_with_wrong_key_returns_401(self, client):
        response = client.get(
            "/api/v1/occupations/search",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_protected_route_with_valid_key_passes(self, client):
        response = client.get(
            "/api/v1/occupations/search",
            headers={"X-API-Key": "valid-key-123"},
        )
        assert response.status_code == 200

    def test_post_route_requires_key(self, client):
        response = client.post("/api/v1/user/1/recommendations")
        assert response.status_code == 401

    def test_post_route_with_valid_key_passes(self, client):
        response = client.post(
            "/api/v1/user/1/recommendations",
            headers={"X-API-Key": "valid-key-123"},
        )
        assert response.status_code == 200


class TestOpenPaths:
    """Certain paths are always accessible without authentication."""

    @pytest.fixture
    def client(self):
        with patch("app.core.auth.settings") as mock_settings:
            mock_settings.auth_enabled = True
            mock_settings.api_key = "valid-key-123"
            app = _make_app()
            yield TestClient(app)

    def test_root_is_open(self, client):
        assert client.get("/").status_code == 200

    def test_health_is_open(self, client):
        assert client.get("/health").status_code == 200

    def test_health_ready_is_open(self, client):
        assert client.get("/health/ready").status_code == 200

    def test_health_detailed_is_open(self, client):
        assert client.get("/health/detailed").status_code == 200

    def test_metrics_is_open(self, client):
        assert client.get("/metrics").status_code == 200

    def test_docs_page_is_open(self, client):
        assert client.get("/docs-page").status_code == 200
