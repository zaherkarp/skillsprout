"""Tests for the occupation registry and data enrichment pipeline.

Covers:
  1. OccupationRegistry CRUD operations and persistence
  2. EnrichmentPipeline discovery, skills fetch, and scoring
  3. Static data seeding and registry-based lookups
  4. Edge cases (empty registry, missing data, concurrent saves)
"""

import json
import pytest
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock

from app.services.occupation_registry import OccupationRegistry, REGISTRY_VERSION
from app.services.enrichment_pipeline import EnrichmentPipeline, EnrichmentResult
from app.data.ai_exposure import EXPOSURE_DATA
from app.data.bls_projections import BLS_PROJECTIONS


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def tmp_registry(tmp_path) -> OccupationRegistry:
    """Create a registry backed by a temporary file."""
    return OccupationRegistry(path=tmp_path / "test_registry.json")


@pytest.fixture
def populated_registry(tmp_registry) -> OccupationRegistry:
    """Registry with a few occupations pre-loaded."""
    tmp_registry.upsert_occupation("15-1251.00", "Computer Programmers", source="test")
    tmp_registry.set_skills("15-1251.00", [
        {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 88.0, "level": 6.0},
        {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 78.0, "level": 5.5},
    ])
    tmp_registry.set_exposure("15-1251.00", {
        "theoretical_exposure": 0.94, "observed_exposure": 0.75, "exposure_rank": "high",
    })
    tmp_registry.set_projections("15-1251.00", {
        "projected_growth_pct": -10.6, "projected_openings_annual": 9600,
        "current_employment": 147400, "outlook": "declining",
    })
    tmp_registry.upsert_occupation("47-2111.00", "Electricians", source="test")
    tmp_registry.save()
    return tmp_registry


@pytest.fixture
def mock_onet_client():
    """Mock O*NET client for pipeline tests."""
    client = AsyncMock()
    client.search_occupations = AsyncMock(return_value=[
        {"code": "99-0001.00", "title": "Test Occupation Alpha"},
        {"code": "99-0002.00", "title": "Test Occupation Beta"},
    ])
    client.get_occupation_meta = AsyncMock(return_value={
        "title": "Test Occupation", "description": "A test", "job_zone": 3, "education": "Bachelor's",
    })
    client.get_occupation_skills = AsyncMock(return_value=[
        {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0, "level": 5.0},
        {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 4.5},
    ])
    return client


@pytest.fixture
def test_personas() -> Dict[str, Dict]:
    return {
        "test_persona_a": {
            "name": "Test Persona A",
            "skill_ratings": {"2.B.8.a": 3, "2.B.1.a": 2, "2.B.1.g": 0},
        },
        "test_persona_b": {
            "name": "Test Persona B",
            "skill_ratings": {"2.B.8.a": 4, "2.B.1.a": 4, "2.B.1.g": 4},
        },
    }


# =====================================================================
# 1. OccupationRegistry tests
# =====================================================================

class TestOccupationRegistry:

    def test_empty_registry_init(self, tmp_registry):
        assert tmp_registry.count() == 0
        assert tmp_registry.codes() == []

    def test_upsert_and_get(self, tmp_registry):
        tmp_registry.upsert_occupation("15-1251.00", "Computer Programmers")
        entry = tmp_registry.get("15-1251.00")
        assert entry is not None
        assert entry["title"] == "Computer Programmers"
        assert entry["code"] == "15-1251.00"

    def test_upsert_updates_existing(self, tmp_registry):
        tmp_registry.upsert_occupation("15-1251.00", "Old Title")
        tmp_registry.upsert_occupation("15-1251.00", "New Title")
        assert tmp_registry.count() == 1
        assert tmp_registry.get("15-1251.00")["title"] == "New Title"

    def test_has_check(self, tmp_registry):
        assert not tmp_registry.has("15-1251.00")
        tmp_registry.upsert_occupation("15-1251.00", "Test")
        assert tmp_registry.has("15-1251.00")

    def test_set_skills(self, tmp_registry):
        tmp_registry.upsert_occupation("15-1251.00", "Test")
        skills = [{"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 88.0, "level": 6.0}]
        tmp_registry.set_skills("15-1251.00", skills)
        assert tmp_registry.get_skills("15-1251.00") == skills

    def test_set_skills_raises_for_unknown(self, tmp_registry):
        with pytest.raises(KeyError):
            tmp_registry.set_skills("99-9999.00", [])

    def test_set_exposure(self, tmp_registry):
        tmp_registry.upsert_occupation("15-1251.00", "Test")
        exposure = {"theoretical_exposure": 0.94, "observed_exposure": 0.75, "exposure_rank": "high"}
        tmp_registry.set_exposure("15-1251.00", exposure)
        assert tmp_registry.get_exposure("15-1251.00") == exposure

    def test_set_projections(self, tmp_registry):
        tmp_registry.upsert_occupation("15-1251.00", "Test")
        bls = {"projected_growth_pct": -10.6, "outlook": "declining"}
        tmp_registry.set_projections("15-1251.00", bls)
        assert tmp_registry.get_projections("15-1251.00") == bls

    def test_record_score(self, tmp_registry):
        tmp_registry.upsert_occupation("15-1251.00", "Test")
        tmp_registry.record_score("15-1251.00", "persona_a", 72.5, 27.5, "trainable")
        entry = tmp_registry.get("15-1251.00")
        assert "persona_a" in entry["scoring_results"]
        assert entry["scoring_results"]["persona_a"]["match_score"] == 72.5
        assert entry["scoring_results"]["persona_a"]["bucket"] == "trainable"

    def test_record_score_overwrites(self, tmp_registry):
        tmp_registry.upsert_occupation("15-1251.00", "Test")
        tmp_registry.record_score("15-1251.00", "p", 50.0, 50.0, "trainable")
        tmp_registry.record_score("15-1251.00", "p", 80.0, 20.0, "ready_now")
        assert tmp_registry.get("15-1251.00")["scoring_results"]["p"]["match_score"] == 80.0

    def test_get_returns_none_for_missing(self, tmp_registry):
        assert tmp_registry.get("99-9999.00") is None
        assert tmp_registry.get_exposure("99-9999.00") is None
        assert tmp_registry.get_projections("99-9999.00") is None
        assert tmp_registry.get_skills("99-9999.00") is None

    def test_summary(self, populated_registry):
        summary = populated_registry.summary()
        assert summary["total_occupations"] == 2
        assert summary["with_skills"] == 1
        assert summary["with_exposure"] == 1
        assert summary["with_bls"] == 1

    def test_log_run(self, tmp_registry):
        tmp_registry.log_run(added=5, scored=10, notes="test run")
        assert len(tmp_registry.run_log) == 1
        assert tmp_registry.run_log[0]["occupations_added"] == 5


class TestRegistryPersistence:

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "reg.json"

        # Create and save
        r1 = OccupationRegistry(path)
        r1.upsert_occupation("15-1251.00", "Computer Programmers")
        r1.set_skills("15-1251.00", [
            {"element_id": "2.B.1.g", "skill_name": "Programming", "importance": 88.0, "level": 6.0}
        ])
        r1.record_score("15-1251.00", "test", 72.0, 28.0, "trainable")
        r1.save()

        # Reload
        r2 = OccupationRegistry(path)
        assert r2.count() == 1
        assert r2.get("15-1251.00")["title"] == "Computer Programmers"
        assert len(r2.get_skills("15-1251.00")) == 1
        assert r2.get("15-1251.00")["scoring_results"]["test"]["match_score"] == 72.0

    def test_file_format_valid_json(self, tmp_path):
        path = tmp_path / "reg.json"
        r = OccupationRegistry(path)
        r.upsert_occupation("15-1251.00", "Test")
        r.save()

        with open(path) as f:
            data = json.load(f)
        assert data["version"] == REGISTRY_VERSION
        assert "occupations" in data
        assert "15-1251.00" in data["occupations"]

    def test_atomic_save(self, tmp_path):
        """Save uses a tmp file for atomicity — no .tmp left behind."""
        path = tmp_path / "reg.json"
        r = OccupationRegistry(path)
        r.upsert_occupation("15-1251.00", "Test")
        r.save()
        assert not (tmp_path / "reg.tmp").exists()
        assert path.exists()

    def test_seed_from_static(self, tmp_registry):
        added = tmp_registry.seed_from_static(EXPOSURE_DATA, BLS_PROJECTIONS)
        assert added > 0
        assert tmp_registry.count() >= len(EXPOSURE_DATA)
        # Verify exposure data was set
        for code in EXPOSURE_DATA:
            assert tmp_registry.get_exposure(code) == EXPOSURE_DATA[code]

    def test_seed_idempotent(self, tmp_registry):
        first = tmp_registry.seed_from_static(EXPOSURE_DATA, BLS_PROJECTIONS)
        second = tmp_registry.seed_from_static(EXPOSURE_DATA, BLS_PROJECTIONS)
        assert second == 0  # No new codes on second run

    def test_export_for_scoring(self, populated_registry):
        result = populated_registry.export_for_scoring("15-1251.00")
        assert result is not None
        assert result["code"] == "15-1251.00"
        assert result["title"] == "Computer Programmers"
        assert len(result["skills"]) == 2

    def test_export_for_scoring_missing_skills(self, populated_registry):
        """Occupation without skills should return None for scoring."""
        result = populated_registry.export_for_scoring("47-2111.00")
        assert result is None


# =====================================================================
# 2. EnrichmentPipeline tests
# =====================================================================

class TestEnrichmentPipeline:

    @pytest.mark.asyncio
    async def test_pipeline_discovers_occupations(self, tmp_path, mock_onet_client):
        registry = OccupationRegistry(tmp_path / "reg.json")
        pipeline = EnrichmentPipeline(
            registry=registry,
            onet_client=mock_onet_client,
            discovery_queries=["test query"],
        )
        result = await pipeline.run()
        assert result.occupations_discovered == 2
        assert registry.has("99-0001.00")
        assert registry.has("99-0002.00")

    @pytest.mark.asyncio
    async def test_pipeline_fetches_skills(self, tmp_path, mock_onet_client):
        registry = OccupationRegistry(tmp_path / "reg.json")
        registry.upsert_occupation("99-0001.00", "Test", source="test")
        pipeline = EnrichmentPipeline(
            registry=registry,
            onet_client=mock_onet_client,
            discovery_queries=[],
        )
        result = await pipeline.run(skip_discovery=True)
        # Skills are fetched for the test occupation plus all seeded static codes
        assert result.skills_fetched >= 1
        skills = registry.get_skills("99-0001.00")
        assert len(skills) == 2

    @pytest.mark.asyncio
    async def test_pipeline_scores_personas(self, tmp_path, mock_onet_client, test_personas):
        registry = OccupationRegistry(tmp_path / "reg.json")
        registry.upsert_occupation("99-0001.00", "Test", source="test")
        registry.set_skills("99-0001.00", [
            {"element_id": "2.B.8.a", "skill_name": "Critical Thinking", "importance": 75.0, "level": 5.0},
            {"element_id": "2.B.1.a", "skill_name": "Reading Comprehension", "importance": 69.0, "level": 4.5},
        ])
        pipeline = EnrichmentPipeline(
            registry=registry,
            onet_client=mock_onet_client,
            personas=test_personas,
            discovery_queries=[],
        )
        result = await pipeline.run(skip_discovery=True)
        # 2 personas scored against all occupations with skills (test occ + seeded ones)
        assert result.scores_computed >= 2
        entry = registry.get("99-0001.00")
        assert "test_persona_a" in entry["scoring_results"]
        assert "test_persona_b" in entry["scoring_results"]

    @pytest.mark.asyncio
    async def test_pipeline_persists_to_disk(self, tmp_path, mock_onet_client):
        path = tmp_path / "reg.json"
        registry = OccupationRegistry(path)
        pipeline = EnrichmentPipeline(
            registry=registry,
            onet_client=mock_onet_client,
            discovery_queries=["test"],
        )
        await pipeline.run()
        assert path.exists()

        # Reload and verify
        r2 = OccupationRegistry(path)
        assert r2.count() >= 2
        assert len(r2.run_log) == 1

    @pytest.mark.asyncio
    async def test_pipeline_seeds_static_data(self, tmp_path, mock_onet_client):
        registry = OccupationRegistry(tmp_path / "reg.json")
        pipeline = EnrichmentPipeline(
            registry=registry,
            onet_client=mock_onet_client,
            discovery_queries=[],
        )
        await pipeline.run(skip_discovery=True)
        # Static data should be seeded
        for code in EXPOSURE_DATA:
            assert registry.has(code), f"Static code {code} missing after seed"

    @pytest.mark.asyncio
    async def test_pipeline_handles_api_errors(self, tmp_path):
        """Pipeline should handle API errors gracefully."""
        client = AsyncMock()
        client.search_occupations = AsyncMock(side_effect=Exception("API down"))
        registry = OccupationRegistry(tmp_path / "reg.json")
        pipeline = EnrichmentPipeline(
            registry=registry,
            onet_client=client,
            discovery_queries=["test"],
        )
        result = await pipeline.run()
        assert len(result.errors) > 0
        assert "API down" in result.errors[0]

    @pytest.mark.asyncio
    async def test_pipeline_skip_discovery(self, tmp_path, mock_onet_client):
        registry = OccupationRegistry(tmp_path / "reg.json")
        pipeline = EnrichmentPipeline(
            registry=registry,
            onet_client=mock_onet_client,
            discovery_queries=["test"],
        )
        result = await pipeline.run(skip_discovery=True)
        assert result.occupations_discovered == 0
        mock_onet_client.search_occupations.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pipeline_idempotent(self, tmp_path, mock_onet_client, test_personas):
        """Running the pipeline twice should not duplicate data."""
        path = tmp_path / "reg.json"
        pipeline = EnrichmentPipeline(
            registry=OccupationRegistry(path),
            onet_client=mock_onet_client,
            personas=test_personas,
            discovery_queries=["test"],
        )
        r1 = await pipeline.run()
        count_after_first = OccupationRegistry(path).count()

        pipeline2 = EnrichmentPipeline(
            registry=OccupationRegistry(path),
            onet_client=mock_onet_client,
            personas=test_personas,
            discovery_queries=["test"],
        )
        r2 = await pipeline2.run()
        count_after_second = OccupationRegistry(path).count()

        assert count_after_second == count_after_first
        assert r2.occupations_discovered == 0  # already known


class TestEnrichmentResult:

    def test_success_when_no_errors(self):
        r = EnrichmentResult(occupations_discovered=5, scores_computed=10)
        assert r.success is True

    def test_not_success_with_errors(self):
        r = EnrichmentResult(errors=["something failed"])
        assert r.success is False

    def test_str_representation(self):
        r = EnrichmentResult(occupations_discovered=3, skills_fetched=2, scores_computed=10)
        s = str(r)
        assert "discovered=3" in s
        assert "scores=10" in s
