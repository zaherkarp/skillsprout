"""Integration tests for the AI exposure and BLS projections API endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestAIExposureEndpoint:
    def test_get_exposure_valid_code(self):
        """GET /api/v1/occupations/{code}/ai-exposure returns 200 with correct schema."""
        resp = client.get("/api/v1/occupations/15-1251.00/ai-exposure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["onet_code"] == "15-1251.00"
        assert data["theoretical_exposure"] == 0.94
        assert data["observed_exposure"] == 0.75
        assert data["exposure_rank"] == "high"
        assert data["ai_resilience_score"] == 25.0
        assert data["ai_headroom"] == 0.19

    def test_get_exposure_invalid_code(self):
        """GET /api/v1/occupations/{code}/ai-exposure returns 404 for unknown code."""
        resp = client.get("/api/v1/occupations/99-9999.00/ai-exposure")
        assert resp.status_code == 404


class TestBLSProjectionsEndpoint:
    def test_get_bls_valid_code(self):
        """GET /api/v1/occupations/{code}/bls-projections returns 200 with correct data."""
        resp = client.get("/api/v1/occupations/29-1141.00/bls-projections")
        assert resp.status_code == 200
        data = resp.json()
        assert data["onet_code"] == "29-1141.00"
        assert data["projected_growth_pct"] == 5.6
        assert data["projected_openings_annual"] == 193100
        assert data["current_employment"] == 3175400
        assert data["outlook"] == "moderate growth"

    def test_get_bls_invalid_code(self):
        """GET /api/v1/occupations/{code}/bls-projections returns 404 for unknown code."""
        resp = client.get("/api/v1/occupations/99-9999.00/bls-projections")
        assert resp.status_code == 404
