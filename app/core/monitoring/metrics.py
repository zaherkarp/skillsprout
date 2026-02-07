"""Prometheus metrics for SkillSprout.

Defines application-level metrics across five categories:
  - REQUEST:      HTTP request duration and counts
  - SCORING:      ML scoring pipeline performance
  - CALIBRATION:  Model versioning and prediction distribution
  - FEEDBACK:     User feedback event tracking
  - SYSTEM:       Infrastructure-level gauges

Usage::

    from app.core.monitoring.metrics import REQUEST_DURATION, REQUESTS_TOTAL
    from app.core.monitoring.metrics import metrics_middleware

    # Add middleware to FastAPI app
    app.middleware("http")(metrics_middleware)
"""
import time
import logging
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom registry (avoids polluting the default with test artefacts)
# ---------------------------------------------------------------------------

REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# REQUEST metrics
# ---------------------------------------------------------------------------

REQUEST_DURATION = Histogram(
    "request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["method", "path", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)

REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total HTTP requests",
    labelnames=["method", "path", "status_code"],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# SCORING metrics
# ---------------------------------------------------------------------------

SCORING_DURATION = Histogram(
    "scoring_duration_seconds",
    "Time spent scoring a single occupation against user capabilities",
    labelnames=["model_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    registry=REGISTRY,
)

SCORES_BY_BUCKET = Counter(
    "scores_by_bucket",
    "Number of occupation scores produced, by recommendation bucket",
    labelnames=["bucket"],
    registry=REGISTRY,
)

COLD_START_FALLBACKS_TOTAL = Counter(
    "cold_start_fallbacks_total",
    "Number of times the system fell back to cold-start baseline scoring",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# CALIBRATION metrics
# ---------------------------------------------------------------------------

MODEL_VERSION_INFO = Info(
    "model_version",
    "Currently active ML model version and metadata",
    registry=REGISTRY,
)

PREDICTION_DISTRIBUTION = Histogram(
    "prediction_distribution",
    "Distribution of calibrated prediction probabilities (0-1)",
    labelnames=["model_version"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# FEEDBACK metrics
# ---------------------------------------------------------------------------

FEEDBACK_EVENTS_TOTAL = Counter(
    "feedback_events_received_total",
    "Total user feedback events received",
    labelnames=["action_type"],
    registry=REGISTRY,
)

FEEDBACK_LOOP_LATENCY = Histogram(
    "feedback_loop_latency_seconds",
    "Latency from recommendation generation to user feedback",
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600, 86400),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# SYSTEM metrics
# ---------------------------------------------------------------------------

DB_CONNECTION_POOL_SIZE = Gauge(
    "db_connection_pool_size",
    "Current size of the database connection pool",
    labelnames=["state"],  # active, idle, overflow
    registry=REGISTRY,
)

REDIS_CONNECTION_COUNT = Gauge(
    "redis_connection_count",
    "Number of active Redis connections",
    registry=REGISTRY,
)

CELERY_QUEUE_DEPTH = Gauge(
    "celery_queue_depth",
    "Number of tasks waiting in the Celery task queue",
    labelnames=["queue"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def record_scoring(model_version: str, bucket: str, duration_seconds: float) -> None:
    """Record a scoring event with duration and bucket classification.

    Args:
        model_version: The model version used for scoring.
        bucket: Recommendation bucket (ready_now, trainable, long_reskill).
        duration_seconds: Wall-clock time for the scoring call.
    """
    SCORING_DURATION.labels(model_version=model_version).observe(duration_seconds)
    SCORES_BY_BUCKET.labels(bucket=bucket).inc()


def record_cold_start_fallback() -> None:
    """Increment the cold-start fallback counter."""
    COLD_START_FALLBACKS_TOTAL.inc()


def record_feedback(action_type: str) -> None:
    """Record a feedback event.

    Args:
        action_type: One of click, save, hide, apply, interview, offer.
    """
    FEEDBACK_EVENTS_TOTAL.labels(action_type=action_type).inc()


def record_feedback_latency(seconds: float) -> None:
    """Record latency between recommendation and feedback.

    Args:
        seconds: Elapsed seconds from recommendation creation to feedback.
    """
    FEEDBACK_LOOP_LATENCY.observe(seconds)


def set_model_version(version: str, trained_at: str = "", is_calibrated: str = "false") -> None:
    """Update the model version info metric.

    Args:
        version: Active model version string.
        trained_at: ISO-format timestamp of last training run.
        is_calibrated: 'true' or 'false'.
    """
    MODEL_VERSION_INFO.info({
        "version": version,
        "trained_at": trained_at,
        "is_calibrated": is_calibrated,
    })


def update_system_gauges() -> None:
    """Refresh system-level gauges (DB pool, Redis, Celery).

    Best called periodically (e.g. every 15 s) rather than on every request.
    """
    _update_db_pool_gauge()
    _update_redis_gauge()
    _update_celery_gauge()


def _update_db_pool_gauge() -> None:
    """Read the SQLAlchemy pool status and publish gauges."""
    try:
        from app.db.session import async_engine

        pool = async_engine.pool
        DB_CONNECTION_POOL_SIZE.labels(state="checked_in").set(pool.checkedin())
        DB_CONNECTION_POOL_SIZE.labels(state="checked_out").set(pool.checkedout())
        DB_CONNECTION_POOL_SIZE.labels(state="overflow").set(pool.overflow())
    except Exception:
        pass  # pool may be NullPool in tests


def _update_redis_gauge() -> None:
    """Read Redis INFO to publish connection count."""
    try:
        import redis as sync_redis

        client = sync_redis.from_url(settings.redis_url)
        try:
            info = client.info("clients")
            REDIS_CONNECTION_COUNT.set(info.get("connected_clients", 0))
        finally:
            client.close()
    except Exception:
        pass


def _update_celery_gauge() -> None:
    """Inspect Celery queues and publish depth."""
    try:
        from app.tasks.celery_app import celery_app

        inspector = celery_app.control.inspect(timeout=1.0)
        reserved = inspector.reserved() or {}
        for worker_name, tasks in reserved.items():
            CELERY_QUEUE_DEPTH.labels(queue=worker_name).set(len(tasks))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Middleware that instruments every HTTP request
# ---------------------------------------------------------------------------

async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """Starlette-compatible middleware that records request metrics.

    Records ``request_duration_seconds`` and ``requests_total`` for each
    inbound HTTP request.  Mount via ``app.middleware("http")(metrics_middleware)``.
    """
    start = time.monotonic()
    response: Response = await call_next(request)
    elapsed = time.monotonic() - start

    # Normalise path to avoid cardinality explosion (strip IDs)
    path = _normalise_path(request.url.path)
    method = request.method
    status = str(response.status_code)

    REQUEST_DURATION.labels(method=method, path=path, status_code=status).observe(elapsed)
    REQUESTS_TOTAL.labels(method=method, path=path, status_code=status).inc()

    return response


def _normalise_path(path: str) -> str:
    """Collapse numeric and UUID path segments to ``:id`` to limit label cardinality."""
    import re

    # Replace numeric IDs
    path = re.sub(r"/\d+", "/:id", path)
    # Replace UUIDs
    path = re.sub(
        r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "/:id",
        path,
        flags=re.IGNORECASE,
    )
    return path


# ---------------------------------------------------------------------------
# /metrics endpoint (for Prometheus scraping)
# ---------------------------------------------------------------------------

from fastapi import APIRouter as _APIRouter  # noqa: E402
from fastapi.responses import Response as _FastAPIResponse  # noqa: E402

from app.core.config import settings  # noqa: E402

metrics_router = _APIRouter(tags=["metrics"])


@metrics_router.get("/metrics")
async def prometheus_metrics() -> _FastAPIResponse:
    """Expose Prometheus metrics for scraping."""
    body = generate_latest(REGISTRY)
    return _FastAPIResponse(content=body, media_type=CONTENT_TYPE_LATEST)
