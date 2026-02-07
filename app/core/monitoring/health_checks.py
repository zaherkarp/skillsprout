"""Health check endpoints for SkillSprout.

Provides liveness, readiness, and detailed health checks for all
application components: PostgreSQL, Redis, Celery, O*NET cache, and ML model.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    """Health status for a single component."""

    name: str
    status: str = Field(description="healthy | degraded | unhealthy")
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class LivenessResponse(BaseModel):
    """Minimal liveness probe response."""

    status: str
    timestamp: datetime
    version: str


class ReadinessResponse(BaseModel):
    """Readiness probe response with component roll-up."""

    status: str
    timestamp: datetime
    components: Dict[str, str]


class DetailedHealthResponse(BaseModel):
    """Detailed health response including per-component latency."""

    status: str
    timestamp: datetime
    version: str
    uptime_seconds: Optional[float] = None
    demo_mode: bool
    components: list[ComponentHealth]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_startup_time: Optional[datetime] = None


def mark_startup() -> None:
    """Record application startup time.  Call once during lifespan startup."""
    global _startup_time
    _startup_time = datetime.utcnow()


async def _check_postgres() -> ComponentHealth:
    """Check PostgreSQL connectivity via a lightweight SELECT 1."""
    start = time.monotonic()
    try:
        from sqlalchemy import text
        from app.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        elapsed = (time.monotonic() - start) * 1000
        return ComponentHealth(name="postgresql", status="healthy", latency_ms=round(elapsed, 2))
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("PostgreSQL health check failed: %s", exc)
        return ComponentHealth(
            name="postgresql",
            status="unhealthy",
            latency_ms=round(elapsed, 2),
            detail=str(exc),
        )


async def _check_redis() -> ComponentHealth:
    """Check Redis connectivity via PING."""
    start = time.monotonic()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        try:
            pong = await client.ping()
            elapsed = (time.monotonic() - start) * 1000
            status = "healthy" if pong else "unhealthy"
            return ComponentHealth(name="redis", status=status, latency_ms=round(elapsed, 2))
        finally:
            await client.aclose()
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("Redis health check failed: %s", exc)
        return ComponentHealth(
            name="redis",
            status="unhealthy",
            latency_ms=round(elapsed, 2),
            detail=str(exc),
        )


async def _check_celery() -> ComponentHealth:
    """Check Celery worker availability via inspect ping."""
    start = time.monotonic()
    try:
        from app.tasks.celery_app import celery_app

        inspector = celery_app.control.inspect(timeout=2.0)
        ping_result = inspector.ping()
        elapsed = (time.monotonic() - start) * 1000
        if ping_result:
            return ComponentHealth(
                name="celery",
                status="healthy",
                latency_ms=round(elapsed, 2),
                detail=f"{len(ping_result)} worker(s) online",
            )
        return ComponentHealth(
            name="celery",
            status="degraded",
            latency_ms=round(elapsed, 2),
            detail="No workers responding",
        )
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("Celery health check failed: %s", exc)
        return ComponentHealth(
            name="celery",
            status="unhealthy",
            latency_ms=round(elapsed, 2),
            detail=str(exc),
        )


async def _check_onet_cache() -> ComponentHealth:
    """Check O*NET occupation cache freshness."""
    start = time.monotonic()
    try:
        from sqlalchemy import select, func
        from app.db.session import AsyncSessionLocal
        from app.models.models import Occupation

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(func.count(Occupation.onet_code)))
            count = result.scalar() or 0

            result = await session.execute(select(func.max(Occupation.last_fetched_at)))
            last_fetched = result.scalar()

        elapsed = (time.monotonic() - start) * 1000
        if count == 0:
            return ComponentHealth(
                name="onet_cache",
                status="degraded",
                latency_ms=round(elapsed, 2),
                detail="Cache is empty",
            )

        status = "healthy"
        detail = f"{count} occupations cached"
        if last_fetched:
            age_days = (datetime.utcnow() - last_fetched).days
            detail += f", last updated {age_days}d ago"
            if age_days > 30:
                status = "degraded"
                detail += " (stale)"

        return ComponentHealth(name="onet_cache", status=status, latency_ms=round(elapsed, 2), detail=detail)
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("O*NET cache health check failed: %s", exc)
        return ComponentHealth(
            name="onet_cache",
            status="unhealthy",
            latency_ms=round(elapsed, 2),
            detail=str(exc),
        )


async def _check_ml_model() -> ComponentHealth:
    """Check ML model availability."""
    start = time.monotonic()
    try:
        from app.ml.scoring import get_baseline_scorer

        scorer = get_baseline_scorer()
        # Verify the scorer can be instantiated and has thresholds configured
        has_thresholds = (
            scorer.ready_now_match_threshold is not None
            and scorer.ready_now_gap_threshold is not None
        )
        elapsed = (time.monotonic() - start) * 1000
        if has_thresholds:
            return ComponentHealth(
                name="ml_model",
                status="healthy",
                latency_ms=round(elapsed, 2),
                detail=f"Baseline scorer (model_version={settings.model_version})",
            )
        return ComponentHealth(
            name="ml_model",
            status="degraded",
            latency_ms=round(elapsed, 2),
            detail="Scorer loaded but thresholds missing",
        )
    except Exception as exc:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("ML model health check failed: %s", exc)
        return ComponentHealth(
            name="ml_model",
            status="unhealthy",
            latency_ms=round(elapsed, 2),
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=LivenessResponse)
async def liveness():
    """Basic liveness probe.

    Returns 200 if the process is alive and capable of serving traffic.
    Intended for Kubernetes ``livenessProbe`` or a load-balancer ping.
    """
    return LivenessResponse(
        status="alive",
        timestamp=datetime.utcnow(),
        version="1.0.0",
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(response: Response):
    """Readiness probe.

    Checks PostgreSQL, Redis, Celery, O*NET cache, and the ML model.
    Returns 200 when all critical components are healthy, 503 otherwise.
    """
    checks = await _run_all_checks()
    components = {c.name: c.status for c in checks}

    critical = {"postgresql", "ml_model"}
    all_healthy = all(
        c.status == "healthy" for c in checks if c.name in critical
    )

    status = "ready" if all_healthy else "not_ready"
    if not all_healthy:
        response.status_code = 503

    return ReadinessResponse(
        status=status,
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health(response: Response):
    """Detailed health check with per-component latency.

    Returns a full JSON breakdown of every component including response
    latency, error detail, and uptime information.
    """
    checks = await _run_all_checks()

    any_unhealthy = any(c.status == "unhealthy" for c in checks)
    any_degraded = any(c.status == "degraded" for c in checks)

    if any_unhealthy:
        overall = "unhealthy"
        response.status_code = 503
    elif any_degraded:
        overall = "degraded"
    else:
        overall = "healthy"

    uptime = None
    if _startup_time:
        uptime = (datetime.utcnow() - _startup_time).total_seconds()

    return DetailedHealthResponse(
        status=overall,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=round(uptime, 2) if uptime is not None else None,
        demo_mode=settings.is_demo_mode,
        components=checks,
    )


async def _run_all_checks() -> list[ComponentHealth]:
    """Execute every component health check and return results."""
    import asyncio

    results = await asyncio.gather(
        _check_postgres(),
        _check_redis(),
        _check_celery(),
        _check_onet_cache(),
        _check_ml_model(),
        return_exceptions=True,
    )

    checked: list[ComponentHealth] = []
    names = ["postgresql", "redis", "celery", "onet_cache", "ml_model"]
    for name, result in zip(names, results):
        if isinstance(result, Exception):
            checked.append(
                ComponentHealth(name=name, status="unhealthy", detail=str(result))
            )
        else:
            checked.append(result)
    return checked
