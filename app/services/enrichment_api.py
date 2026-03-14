"""API endpoints for the data enrichment pipeline and occupation registry."""

from fastapi import APIRouter, Query
from typing import Optional

from app.services.occupation_registry import OccupationRegistry
from app.services.enrichment_pipeline import EnrichmentPipeline

router = APIRouter(prefix="/enrichment", tags=["enrichment"])


@router.get("/status")
async def registry_status():
    """Return a summary of the occupation registry."""
    registry = OccupationRegistry()
    return registry.summary()


@router.get("/occupations")
async def list_occupations(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    has_skills: Optional[bool] = None,
    has_scores: Optional[bool] = None,
):
    """List occupations in the registry with optional filters."""
    registry = OccupationRegistry()
    items = list(registry.occupations.values())

    if has_skills is True:
        items = [o for o in items if o.get("skills")]
    elif has_skills is False:
        items = [o for o in items if not o.get("skills")]

    if has_scores is True:
        items = [o for o in items if o.get("scoring_results")]
    elif has_scores is False:
        items = [o for o in items if not o.get("scoring_results")]

    total = len(items)
    items = items[offset:offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "occupations": [
            {
                "code": o.get("code"),
                "title": o.get("title"),
                "job_zone": o.get("job_zone"),
                "has_skills": bool(o.get("skills")),
                "has_exposure": o.get("ai_exposure") is not None,
                "has_bls": o.get("bls_projections") is not None,
                "scores_count": len(o.get("scoring_results", {})),
                "source": o.get("source"),
                "discovered_at": o.get("discovered_at"),
            }
            for o in items
        ],
    }


@router.get("/occupations/{onet_code}")
async def get_occupation(onet_code: str):
    """Get full details for a single occupation."""
    registry = OccupationRegistry()
    entry = registry.get(onet_code)
    if entry is None:
        return {"error": f"Occupation {onet_code} not found in registry"}
    return entry


@router.post("/run")
async def run_enrichment(skip_discovery: bool = Query(False)):
    """Trigger an enrichment pipeline run on-demand."""
    registry = OccupationRegistry()
    pipeline = EnrichmentPipeline(registry=registry)
    result = await pipeline.run(skip_discovery=skip_discovery)
    return {
        "success": result.success,
        "occupations_discovered": result.occupations_discovered,
        "occupations_enriched": result.occupations_enriched,
        "skills_fetched": result.skills_fetched,
        "scores_computed": result.scores_computed,
        "errors": result.errors,
        "registry_summary": registry.summary(),
    }


@router.get("/run-log")
async def get_run_log(limit: int = Query(20, ge=1, le=100)):
    """Return the most recent enrichment run log entries."""
    registry = OccupationRegistry()
    log = registry.run_log[-limit:]
    log.reverse()
    return {"total_runs": len(registry.run_log), "entries": log}
