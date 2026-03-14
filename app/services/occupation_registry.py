"""Persistent occupation registry backed by a JSON file.

Accumulates occupation data (metadata, skills, AI exposure, BLS projections,
scoring results) across runs so the system grows its knowledge base over time.

The registry is the single source of truth for dynamically discovered data.
Static data in ai_exposure.py and bls_projections.py serves as seed data;
the registry layer merges both.

File format (occupation_registry.json):
{
  "version": 2,
  "last_updated": "2026-03-14T12:00:00Z",
  "occupations": {
    "15-1251.00": {
      "code": "15-1251.00",
      "title": "Computer Programmers",
      "description": "...",
      "job_zone": 4,
      "education": "Bachelor's degree",
      "skills": [ { "element_id": "...", ... } ],
      "ai_exposure": { "theoretical_exposure": 0.94, ... },
      "bls_projections": { "projected_growth_pct": -10.6, ... },
      "scoring_results": { "<persona_key>": { "match": 72.5, ... } },
      "discovered_at": "2026-03-14T12:00:00Z",
      "last_scored_at": "2026-03-14T12:01:00Z",
      "source": "onet_api"
    }
  },
  "run_log": [
    { "timestamp": "...", "occupations_added": 5, "occupations_scored": 10 }
  ]
}
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "occupation_registry.json"

REGISTRY_VERSION = 2


class OccupationRegistry:
    """Thread-safe JSON file registry for occupation data."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_REGISTRY_PATH
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        """Load registry from disk or create empty structure."""
        if self.path.exists():
            with open(self.path, "r") as f:
                data = json.load(f)
            if data.get("version", 1) < REGISTRY_VERSION:
                data = self._migrate(data)
            return data
        return self._empty()

    def save(self) -> None:
        """Write current state to disk."""
        self._data["last_updated"] = _now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        tmp.replace(self.path)

    def _empty(self) -> Dict[str, Any]:
        return {
            "version": REGISTRY_VERSION,
            "last_updated": _now_iso(),
            "occupations": {},
            "run_log": [],
        }

    def _migrate(self, data: Dict) -> Dict:
        """Migrate from older registry versions."""
        if "run_log" not in data:
            data["run_log"] = []
        data["version"] = REGISTRY_VERSION
        return data

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    @property
    def occupations(self) -> Dict[str, Dict]:
        return self._data["occupations"]

    @property
    def run_log(self) -> List[Dict]:
        return self._data["run_log"]

    def get(self, onet_code: str) -> Optional[Dict]:
        """Get a single occupation entry or None."""
        return self.occupations.get(onet_code)

    def has(self, onet_code: str) -> bool:
        return onet_code in self.occupations

    def codes(self) -> List[str]:
        return list(self.occupations.keys())

    def count(self) -> int:
        return len(self.occupations)

    def get_exposure(self, onet_code: str) -> Optional[Dict]:
        """Get AI exposure data for an occupation."""
        entry = self.get(onet_code)
        if entry and "ai_exposure" in entry:
            return entry["ai_exposure"]
        return None

    def get_projections(self, onet_code: str) -> Optional[Dict]:
        """Get BLS projections for an occupation."""
        entry = self.get(onet_code)
        if entry and "bls_projections" in entry:
            return entry["bls_projections"]
        return None

    def get_skills(self, onet_code: str) -> Optional[List[Dict]]:
        """Get skills list for an occupation."""
        entry = self.get(onet_code)
        if entry and "skills" in entry:
            return entry["skills"]
        return None

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the registry state."""
        occ = self.occupations
        return {
            "total_occupations": len(occ),
            "with_skills": sum(1 for o in occ.values() if o.get("skills")),
            "with_exposure": sum(1 for o in occ.values() if o.get("ai_exposure")),
            "with_bls": sum(1 for o in occ.values() if o.get("bls_projections")),
            "with_scores": sum(1 for o in occ.values() if o.get("scoring_results")),
            "total_runs": len(self.run_log),
            "last_updated": self._data.get("last_updated"),
        }

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def upsert_occupation(
        self,
        code: str,
        title: str,
        description: str = "",
        job_zone: int = 0,
        education: str = "",
        source: str = "unknown",
    ) -> Dict:
        """Add or update basic occupation metadata."""
        if code in self.occupations:
            entry = self.occupations[code]
            entry["title"] = title
            if description:
                entry["description"] = description
            if job_zone:
                entry["job_zone"] = job_zone
            if education:
                entry["education"] = education
        else:
            entry = {
                "code": code,
                "title": title,
                "description": description,
                "job_zone": job_zone,
                "education": education,
                "skills": [],
                "ai_exposure": None,
                "bls_projections": None,
                "scoring_results": {},
                "discovered_at": _now_iso(),
                "last_scored_at": None,
                "source": source,
            }
            self.occupations[code] = entry
        return entry

    def set_skills(self, code: str, skills: List[Dict]) -> None:
        """Set skills for an occupation (must already exist)."""
        if code not in self.occupations:
            raise KeyError(f"Occupation {code} not in registry; upsert first")
        self.occupations[code]["skills"] = skills

    def set_exposure(self, code: str, exposure: Dict) -> None:
        """Set AI exposure data for an occupation."""
        if code not in self.occupations:
            raise KeyError(f"Occupation {code} not in registry; upsert first")
        self.occupations[code]["ai_exposure"] = exposure

    def set_projections(self, code: str, projections: Dict) -> None:
        """Set BLS projections for an occupation."""
        if code not in self.occupations:
            raise KeyError(f"Occupation {code} not in registry; upsert first")
        self.occupations[code]["bls_projections"] = projections

    def record_score(
        self,
        target_code: str,
        persona_key: str,
        match_score: float,
        gap_severity: float,
        bucket: str,
        top_gaps: Optional[List[Dict]] = None,
    ) -> None:
        """Record a scoring result for a persona → occupation pair."""
        if target_code not in self.occupations:
            raise KeyError(f"Occupation {target_code} not in registry")
        entry = self.occupations[target_code]
        if "scoring_results" not in entry:
            entry["scoring_results"] = {}
        entry["scoring_results"][persona_key] = {
            "match_score": match_score,
            "gap_severity": gap_severity,
            "bucket": bucket,
            "top_gaps": top_gaps or [],
            "scored_at": _now_iso(),
        }
        entry["last_scored_at"] = _now_iso()

    def log_run(self, added: int, scored: int, errors: int = 0, notes: str = "") -> None:
        """Append an entry to the run log."""
        self.run_log.append({
            "timestamp": _now_iso(),
            "occupations_added": added,
            "occupations_scored": scored,
            "errors": errors,
            "notes": notes,
        })

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def seed_from_static(
        self,
        exposure_data: Dict[str, Dict],
        bls_data: Dict[str, Dict],
    ) -> int:
        """Import all static seed data (ai_exposure + bls_projections).

        Returns the number of new codes added.
        """
        all_codes = set(exposure_data.keys()) | set(bls_data.keys())
        added = 0
        for code in all_codes:
            if not self.has(code):
                self.upsert_occupation(
                    code=code,
                    title=code,  # placeholder; enrichment fills real title
                    source="static_seed",
                )
                added += 1
            if code in exposure_data:
                self.set_exposure(code, exposure_data[code])
            if code in bls_data:
                self.set_projections(code, bls_data[code])
        return added

    def export_for_scoring(self, code: str) -> Optional[Dict]:
        """Return occupation data in the format expected by BaselineScorer."""
        entry = self.get(code)
        if not entry or not entry.get("skills"):
            return None
        return {
            "code": entry["code"],
            "title": entry["title"],
            "job_zone": entry.get("job_zone", 3),
            "skills": entry["skills"],
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
