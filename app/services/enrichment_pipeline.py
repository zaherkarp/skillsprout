"""Data enrichment pipeline — discovers, fetches, scores, and persists occupations.

This pipeline is designed to run:
  1. On app startup (via lifespan hook)
  2. On-demand via CLI: python -m app.services.enrichment_pipeline
  3. Periodically via Celery beat (background task)

Each run:
  - Seeds the registry with static data (ai_exposure, bls_projections)
  - Discovers new occupations via O*NET API search queries
  - Fetches skills for any occupation missing them
  - Scores all personas against all occupations in the registry
  - Persists everything to occupation_registry.json
  - Optionally syncs to the database (Occupation / OccupationSkill tables)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.data.ai_exposure import EXPOSURE_DATA
from app.data.bls_projections import BLS_PROJECTIONS
from app.ml.scoring import BaselineScorer, OccupationScore
from app.services.occupation_registry import OccupationRegistry

logger = logging.getLogger(__name__)


# Default search queries to discover occupations — covers the Anthropic report
# categories and common career transition targets.
DEFAULT_DISCOVERY_QUERIES = [
    "computer programmer",
    "customer service",
    "data entry",
    "medical records",
    "market research analyst",
    "sales representative",
    "software quality assurance",
    "information security",
    "computer support specialist",
    "electrician",
    "registered nurse",
    "software developer",
    "web developer",
    "network administrator",
    "financial analyst",
    "project manager",
    "graphic designer",
    "teacher",
    "accountant",
    "administrative assistant",
]


@dataclass
class EnrichmentResult:
    """Summary of what happened during an enrichment run."""
    occupations_discovered: int = 0
    occupations_enriched: int = 0
    skills_fetched: int = 0
    scores_computed: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        return (
            f"Enrichment: discovered={self.occupations_discovered}, "
            f"enriched={self.occupations_enriched}, skills_fetched={self.skills_fetched}, "
            f"scores={self.scores_computed}, errors={len(self.errors)}"
        )


class EnrichmentPipeline:
    """Orchestrates occupation discovery, data fetching, and scoring."""

    def __init__(
        self,
        registry: Optional[OccupationRegistry] = None,
        onet_client=None,
        scorer: Optional[BaselineScorer] = None,
        personas: Optional[Dict[str, Dict]] = None,
        discovery_queries: Optional[List[str]] = None,
    ):
        self.registry = registry or OccupationRegistry()
        self._onet_client = onet_client
        self.scorer = scorer or BaselineScorer()
        self.personas = personas or {}
        self.discovery_queries = discovery_queries or DEFAULT_DISCOVERY_QUERIES

    @property
    def onet_client(self):
        if self._onet_client is None:
            from app.services.onet_client import get_onet_client
            self._onet_client = get_onet_client()
        return self._onet_client

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run(self, skip_discovery: bool = False) -> EnrichmentResult:
        """Execute the full enrichment pipeline.

        Steps:
            1. Seed static data into registry
            2. Discover new occupations via O*NET search
            3. Fetch skills for occupations missing them
            4. Score all personas against all occupations
            5. Persist to disk
        """
        result = EnrichmentResult()

        # Step 1: Seed static data
        seeded = self.registry.seed_from_static(EXPOSURE_DATA, BLS_PROJECTIONS)
        logger.info(f"Seeded {seeded} new codes from static data")

        # Step 2: Discover occupations
        if not skip_discovery:
            discovered = await self._discover(result)
            result.occupations_discovered = discovered

        # Step 3: Fetch missing skills
        fetched = await self._fetch_missing_skills(result)
        result.skills_fetched = fetched

        # Step 4: Score personas
        scored = self._score_all_personas(result)
        result.scores_computed = scored

        # Step 5: Persist
        self.registry.log_run(
            added=result.occupations_discovered,
            scored=result.scores_computed,
            errors=len(result.errors),
            notes=str(result),
        )
        self.registry.save()

        result.occupations_enriched = self.registry.count()
        logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    async def _discover(self, result: EnrichmentResult) -> int:
        """Search O*NET for new occupations."""
        discovered = 0
        for query in self.discovery_queries:
            try:
                results = await self.onet_client.search_occupations(query, limit=5)
                for occ in results:
                    code = occ["code"]
                    if not self.registry.has(code):
                        self.registry.upsert_occupation(
                            code=code,
                            title=occ["title"],
                            source="onet_api",
                        )
                        discovered += 1
                        logger.info(f"Discovered: {code} — {occ['title']}")
            except Exception as e:
                msg = f"Discovery query '{query}' failed: {e}"
                logger.warning(msg)
                result.errors.append(msg)
        return discovered

    async def _fetch_missing_skills(self, result: EnrichmentResult) -> int:
        """Fetch skills for occupations that don't have them yet."""
        fetched = 0
        for code in self.registry.codes():
            entry = self.registry.get(code)
            if entry and not entry.get("skills"):
                try:
                    # Fetch metadata
                    meta = await self.onet_client.get_occupation_meta(code)
                    if meta:
                        self.registry.upsert_occupation(
                            code=code,
                            title=meta.get("title", code),
                            description=meta.get("description", ""),
                            job_zone=meta.get("job_zone", 0),
                            education=meta.get("education", ""),
                            source=entry.get("source", "onet_api"),
                        )

                    # Fetch skills
                    skills = await self.onet_client.get_occupation_skills(code)
                    if skills:
                        self.registry.set_skills(code, skills)
                        fetched += 1
                        logger.info(f"Fetched {len(skills)} skills for {code}")
                except Exception as e:
                    msg = f"Skill fetch for {code} failed: {e}"
                    logger.warning(msg)
                    result.errors.append(msg)
        return fetched

    def _score_all_personas(self, result: EnrichmentResult) -> int:
        """Score every persona against every occupation with skills."""
        scored = 0
        for code in self.registry.codes():
            occ_data = self.registry.export_for_scoring(code)
            if not occ_data:
                continue

            for persona_key, persona in self.personas.items():
                try:
                    score: OccupationScore = self.scorer.score_occupation(
                        onet_code=occ_data["code"],
                        occupation_title=occ_data["title"],
                        occupation_skills=occ_data["skills"],
                        user_skill_ratings=persona["skill_ratings"],
                        current_job_zone=3,
                        target_job_zone=occ_data["job_zone"],
                    )
                    self.registry.record_score(
                        target_code=code,
                        persona_key=persona_key,
                        match_score=score.match_score,
                        gap_severity=score.gap_severity,
                        bucket=score.bucket,
                        top_gaps=[
                            {"element_id": g.element_id, "skill_name": g.skill_name,
                             "gap_weight": g.gap_weight}
                            for g in score.top_gaps
                        ],
                    )
                    scored += 1
                except Exception as e:
                    msg = f"Scoring {persona_key} → {code} failed: {e}"
                    logger.warning(msg)
                    result.errors.append(msg)
        return scored

    # ------------------------------------------------------------------
    # Database sync
    # ------------------------------------------------------------------

    async def sync_to_database(self, db_session) -> int:
        """Sync registry data into the Occupation and OccupationSkill tables.

        Returns the number of rows upserted.
        """
        from app.models.models import Occupation, OccupationSkill, Skill

        upserted = 0
        for code, entry in self.registry.occupations.items():
            # Upsert occupation
            existing = await db_session.get(Occupation, code)
            if existing:
                existing.title = entry["title"]
                existing.description = entry.get("description", "")
                existing.job_zone = entry.get("job_zone", 0)
                existing.education_level = entry.get("education", "")
            else:
                db_session.add(Occupation(
                    onet_code=code,
                    title=entry["title"],
                    description=entry.get("description", ""),
                    job_zone=entry.get("job_zone", 0),
                    education_level=entry.get("education", ""),
                    raw_json=entry,
                ))

            # Upsert skills
            for skill in entry.get("skills", []):
                eid = skill["element_id"]
                # Ensure skill exists
                existing_skill = await db_session.get(Skill, eid)
                if not existing_skill:
                    db_session.add(Skill(
                        element_id=eid,
                        name=skill.get("skill_name", eid),
                    ))

                # Upsert occupation-skill link
                from sqlalchemy import select
                stmt = select(OccupationSkill).where(
                    OccupationSkill.onet_code == code,
                    OccupationSkill.element_id == eid,
                )
                result = await db_session.execute(stmt)
                occ_skill = result.scalar_one_or_none()
                if occ_skill:
                    occ_skill.importance = skill.get("importance", 0)
                    occ_skill.level = skill.get("level", 0)
                else:
                    db_session.add(OccupationSkill(
                        onet_code=code,
                        element_id=eid,
                        importance=skill.get("importance", 0),
                        level=skill.get("level", 0),
                    ))
                upserted += 1

        await db_session.commit()
        logger.info(f"Synced {upserted} occupation-skill rows to database")
        return upserted


# ------------------------------------------------------------------
# Convenience: run from CLI
# ------------------------------------------------------------------

def run_enrichment(
    registry_path=None,
    personas: Optional[Dict[str, Dict]] = None,
    skip_discovery: bool = False,
) -> EnrichmentResult:
    """Synchronous wrapper for the enrichment pipeline."""
    from app.services.occupation_registry import OccupationRegistry, DEFAULT_REGISTRY_PATH
    from pathlib import Path

    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    registry = OccupationRegistry(path)
    pipeline = EnrichmentPipeline(
        registry=registry,
        personas=personas or {},
        discovery_queries=DEFAULT_DISCOVERY_QUERIES,
    )
    return asyncio.run(pipeline.run(skip_discovery=skip_discovery))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Import Anthropic report personas for scoring
    try:
        from tests.test_anthropic_report_personas import (
            PERSONA_DEREK, PERSONA_LINDA, PERSONA_PRIYA, PERSONA_CARLOS,
            PERSONA_WEI, PERSONA_TANYA, PERSONA_MARCUS, PERSONA_FATIMA,
            PERSONA_JORDAN, PERSONA_ELENA,
        )
        personas = {
            "derek_programmer": PERSONA_DEREK,
            "linda_csr": PERSONA_LINDA,
            "priya_data_entry": PERSONA_PRIYA,
            "carlos_medical_records": PERSONA_CARLOS,
            "wei_market_research": PERSONA_WEI,
            "tanya_sales": PERSONA_TANYA,
            "marcus_qa": PERSONA_MARCUS,
            "fatima_security": PERSONA_FATIMA,
            "jordan_support": PERSONA_JORDAN,
            "elena_electrician": PERSONA_ELENA,
        }
    except ImportError:
        personas = {}
        logger.warning("Could not import test personas; running without scoring")

    skip = "--skip-discovery" in sys.argv
    result = run_enrichment(personas=personas, skip_discovery=skip)
    print(f"\n{result}")
    sys.exit(0 if result.success else 1)
