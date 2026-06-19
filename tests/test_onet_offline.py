"""Tests for the file-backed offline O*NET client (full vendored snapshot).

These exercise the real ``OfflineONetClient`` against
``client/data/onet-full.json`` — the source that runs when the app starts with
no O*NET credentials. The rest of the suite is pinned to ``MockONetClient`` via
``conftest.py``; this module instantiates ``OfflineONetClient`` directly, so
that pin does not affect it.
"""
import pytest

from app.services.onet_offline import (
    OfflineONetClient,
    _element_id_for,
    _importance_to_pct,
)
from app.services.onet_client import ONetClientError, get_onet_client


@pytest.fixture
def client() -> OfflineONetClient:
    return OfflineONetClient()


async def test_dataset_is_a_real_catalog(client):
    # The whole point of the file source: a full catalog, not the ~21-row mock.
    assert len(client._dataset.by_code) > 500


async def test_search_returns_matches(client):
    results = await client.search_occupations("nurse", limit=20)
    assert len(results) > 0
    assert all(set(r.keys()) == {"code", "title"} for r in results)
    assert all("nurse" in r["title"].lower() for r in results)


async def test_search_empty_and_limit(client):
    assert await client.search_occupations("", limit=10) == []
    capped = await client.search_occupations("a", limit=5)
    assert len(capped) <= 5


async def test_meta_shape(client):
    code = next(iter(client._dataset.by_code))
    meta = await client.get_occupation_meta(code)
    assert {"code", "title", "description", "job_zone", "education", "raw_data"} <= set(meta)
    assert meta["code"] == code
    assert meta["title"]


async def test_meta_unknown_code_raises(client):
    with pytest.raises(ONetClientError):
        await client.get_occupation_meta("00-0000.00")


async def test_skills_shape_and_scale(client):
    code = next(iter(client._dataset.by_code))
    skills = await client.get_occupation_skills(code)
    assert len(skills) > 0
    for s in skills:
        assert set(s.keys()) == {"element_id", "skill_name", "importance", "level"}
        assert len(s["element_id"]) <= 20  # Skill.element_id is String(20)
        assert s["skill_name"]
        imp = s["importance"]
        assert imp is None or 0.0 <= imp <= 100.0


def test_element_id_is_stable_and_short():
    a = _element_id_for("Critical Thinking")
    b = _element_id_for("  critical thinking ")
    assert a == b  # case/whitespace-insensitive: one Skill row per skill name
    assert len(a) <= 20


def test_importance_conversion_bounds():
    assert _importance_to_pct(1) == 0.0
    assert _importance_to_pct(5) == 100.0
    assert _importance_to_pct(None) is None


async def test_factory_selects_offline_file(monkeypatch):
    # Blank credentials in the test env => is_demo_mode is True.
    from app.core.config import settings

    monkeypatch.setattr(settings, "onet_offline_source", "file")
    assert isinstance(get_onet_client(), OfflineONetClient)
