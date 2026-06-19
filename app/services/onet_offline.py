"""File-backed offline O*NET client.

Serves occupation and skill data from the O*NET 28.3 snapshot vendored in the
repo (``client/data/onet-full.json``, ~1,000 occupations) with **zero network
and zero credentials**, so the application runs fully independent of the O*NET
Web Services API. This is the default offline source; the smaller hand-written
``MockONetClient`` remains available for deterministic tests
(``ONET_OFFLINE_SOURCE=mock``).

Fidelity note. The vendored snapshot is the compact client dataset
(``{code, title, zone, category, skills:[{name, importance}]}``). Four fields
the live API returns are **synthesized** here and are explicit approximations:

* ``description`` - rendered as ``"<title> (<category>)"``; the snapshot has no
  prose blurb.
* ``education`` - derived from the O*NET Job Zone (the snapshot has no
  per-occupation education value); phrasings match those in ``MockONetClient``.
* skill ``element_id`` - a deterministic short hash of the skill name (the
  snapshot carries no O*NET element ids). Stable across runs and across
  occupations, so the same skill name maps to one ``Skill`` row.
* skill ``level`` - derived from importance on the 0-7 scale.

Importance is converted from the O*NET 1-5 "IM" scale to the backend's 0-100
scale via ``((IM - 1) / 4) * 100`` (the percentage O*NET OnLine itself
displays). The scoring engine normalizes importance **within** an occupation
(``app/ml/scoring.py``), so the precise transform does not change which
occupations are recommended.
"""
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.onet_client import ONetClient, ONetClientError

logger = logging.getLogger(__name__)

# O*NET Job Zone -> typical education. Canonical five-zone mapping; phrasings
# match the values used in MockONetClient so the two offline sources agree.
JOB_ZONE_EDUCATION: Dict[int, str] = {
    1: "No formal educational credential",
    2: "High school diploma or equivalent",
    3: "Associate's degree",
    4: "Bachelor's degree",
    5: "Graduate degree",
}


def _element_id_for(skill_name: str) -> str:
    """Deterministic id for a skill name.

    ``Skill.element_id`` is ``String(20)`` and a primary key, so this must be
    short and stable: the same name always yields the same id (one shared
    ``Skill`` row), and distinct names effectively never collide.
    """
    digest = hashlib.sha1(skill_name.strip().lower().encode("utf-8")).hexdigest()
    return f"x{digest[:15]}"  # 16 chars, within String(20)


def _importance_to_pct(im: Optional[float]) -> Optional[float]:
    """Convert an O*NET IM rating (1-5) to the backend's 0-100 importance."""
    if im is None:
        return None
    pct = (float(im) - 1.0) / 4.0 * 100.0
    return round(max(0.0, min(100.0, pct)), 1)


def _level_from_importance(im: Optional[float]) -> Optional[float]:
    """Synthesize a 0-7 ``level`` from importance (snapshot has no level)."""
    if im is None:
        return None
    return round(float(im) / 5.0 * 7.0, 2)


class _OfflineDataset:
    """Parsed and indexed view of ``onet-full.json`` (built once)."""

    def __init__(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        occupations = raw.get("occupations", [])
        self.by_code: Dict[str, Dict[str, Any]] = {}
        self.search_index: List[Dict[str, str]] = []
        for occ in occupations:
            code = occ.get("code")
            if not code:
                continue
            self.by_code[code] = occ
            self.search_index.append({"code": code, "title": occ.get("title", "")})
        self.version = raw.get("version")
        logger.info(
            "Loaded offline O*NET dataset: %d occupations (v%s) from %s",
            len(self.by_code),
            self.version,
            path,
        )


_dataset: Optional[_OfflineDataset] = None
_dataset_lock = threading.Lock()


def _resolve_data_path() -> Path:
    path = Path(settings.onet_offline_data_path)
    if not path.is_absolute():
        # Resolve relative to the repo root (app/services/onet_offline.py -> repo).
        path = Path(__file__).resolve().parents[2] / path
    return path


def _load_dataset() -> _OfflineDataset:
    global _dataset
    if _dataset is None:
        with _dataset_lock:
            if _dataset is None:
                path = _resolve_data_path()
                if not path.exists():
                    raise ONetClientError(
                        f"Offline O*NET dataset not found at {path}. "
                        f"Set ONET_OFFLINE_DATA_PATH, or use ONET_OFFLINE_SOURCE=mock."
                    )
                _dataset = _OfflineDataset(path)
    return _dataset


class OfflineONetClient(ONetClient):
    """O*NET client backed by the vendored snapshot (no network, no creds)."""

    def __init__(self) -> None:
        # Intentionally skip ONetClient.__init__: no credentials, no warning.
        self.username = ""
        self.password = ""
        self.base_url = ""
        self.timeout = 0
        self.max_retries = 0
        self._dataset = _load_dataset()

    async def _request(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise ONetClientError("OfflineONetClient makes no network requests")

    async def search_occupations(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        q = (query or "").strip().lower()
        if not q:
            return []
        results = [r for r in self._dataset.search_index if q in r["title"].lower()]
        logger.info("[OFFLINE] Found %d occupations for query: %s", len(results), query)
        return results[:limit]

    async def get_occupation_meta(self, onet_code: str) -> Dict[str, Any]:
        occ = self._dataset.by_code.get(onet_code)
        if occ is None:
            raise ONetClientError(f"Occupation {onet_code} not found in offline dataset")
        title = occ.get("title", "")
        category = occ.get("category")
        zone = occ.get("zone")
        return {
            "code": onet_code,
            "title": title,
            "description": f"{title} ({category})" if category else title,
            "job_zone": zone,
            "education": JOB_ZONE_EDUCATION.get(zone),
            "raw_data": occ,
        }

    async def get_occupation_skills(self, onet_code: str) -> List[Dict[str, Any]]:
        occ = self._dataset.by_code.get(onet_code)
        if occ is None:
            raise ONetClientError(f"Skills for {onet_code} not found in offline dataset")
        skills: List[Dict[str, Any]] = []
        for s in occ.get("skills", []):
            name = s.get("name")
            if not name:
                continue
            im = s.get("importance")
            skills.append(
                {
                    "element_id": _element_id_for(name),
                    "skill_name": name,
                    "importance": _importance_to_pct(im),
                    "level": _level_from_importance(im),
                }
            )
        logger.info("[OFFLINE] Fetched %d skills for %s", len(skills), onet_code)
        return skills
