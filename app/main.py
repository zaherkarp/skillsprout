"""FastAPI application main entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.core.config import settings

# -- Feature routers (no prefix — mounted at /api/v1) ---------------------
from app.features.skills_translator.api import router as skills_translator_router
from app.features.explainability.api import router as explainability_router
from app.features.training_paths.api import router as training_paths_router

# -- Feature routers (self-prefixed — mounted at root) --------------------
from app.features.user_profile.profile import router as profile_router
from app.features.user_profile.saved_occupations import router as saved_occupations_router
from app.features.user_profile.progress_tracker import router as progress_tracker_router
from app.features.user_profile.return_engagement import router as return_engagement_router

# -- Event tracking (prefix="/events" — mounted at /api/v1) ---------------
from app.events.implicit_signals import router as events_router

# -- Privacy routers (no prefix — mounted at /api/v1) ---------------------
from app.core.privacy.data_export import router as data_export_router
from app.core.privacy.data_deletion import router as data_deletion_router

# -- Health checks (mounted at root for k8s probes) -----------------------
from app.core.monitoring.health_checks import router as health_router
from app.core.monitoring.health_checks import mark_startup

# -- Prometheus metrics (mounted at root for scraping) --------------------
from app.core.monitoring.metrics import metrics_router

# -- Data enrichment pipeline ---------------------------------------------
from app.services.enrichment_api import router as enrichment_router

# -- Authentication middleware --------------------------------------------
from app.core.auth import APIKeyAuthMiddleware
from app.core.profile_ownership import TOKEN_HEADER as PROFILE_TOKEN_HEADER, require_profile_owner

# -- Progressive enhancement (various mounting strategies) -----------------
from app.core.progressive.session_resumption import router as session_router
from app.core.progressive.offline_capability import router as offline_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _add_owner_token_column_if_missing(sync_conn) -> None:
    """SQLite only: add user_profile.owner_token_hash to a pre-existing table."""
    from sqlalchemy import inspect, text

    columns = {c["name"] for c in inspect(sync_conn).get_columns("user_profile")}
    if "owner_token_hash" not in columns:
        sync_conn.execute(text("ALTER TABLE user_profile ADD COLUMN owner_token_hash VARCHAR(64)"))
        logger.info(
            "Added user_profile.owner_token_hash; existing profiles are locked until "
            "an owner token is issued (python -m app.core.profile_ownership unclaimed)"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Demo mode: {settings.is_demo_mode}")
    mark_startup()

    # First-run schema bootstrap for SQLite (local / no-Docker dev). Postgres
    # deployments create tables via Alembic migrations, so this is skipped there.
    try:
        from app.db.session import async_engine, Base
        import app.models.models  # noqa: F401  register core tables on Base.metadata
        import app.core.privacy.retention_policy  # noqa: F401  deletion_audit_log table
        if async_engine.url.get_backend_name().startswith("sqlite"):
            async with async_engine.begin() as conn:
                await conn.run_sync(lambda sync_conn: Base.metadata.create_all(bind=sync_conn))
                # create_all makes missing tables but never adds a column to an
                # existing one, so a skillsprout.db from before owner tokens
                # would lack user_profile.owner_token_hash and every profile
                # query would fail. Add the column in place. Old rows get NULL,
                # which keeps them locked until an owner token is issued (see
                # app/core/profile_ownership.py). Postgres gets the same column
                # from the Alembic migration instead.
                await conn.run_sync(_add_owner_token_column_if_missing)
            logger.info("SQLite schema ensured (Base.metadata.create_all)")
    except Exception as e:
        logger.warning(f"Schema bootstrap skipped: {e}")

    # Optionally run the data enrichment pipeline on startup. Off by default: a
    # normal boot stays fast and does NOT rewrite the committed registry file.
    # Enable with RUN_STARTUP_ENRICHMENT=true (also runs via Celery beat, the
    # /api/v1/enrichment/run endpoint, or the CLI).
    if settings.run_startup_enrichment:
        try:
            from app.services.enrichment_pipeline import EnrichmentPipeline
            from app.services.occupation_registry import OccupationRegistry
            pipeline = EnrichmentPipeline(
                registry=OccupationRegistry(),
            )
            result = await pipeline.run(skip_discovery=settings.is_demo_mode)
            logger.info(f"Enrichment pipeline: {result}")
        except Exception as e:
            logger.warning(f"Enrichment pipeline skipped: {e}")
    else:
        logger.info("Startup enrichment disabled (set RUN_STARTUP_ENRICHMENT=true to enable)")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Job transition discovery app using O*NET skill data",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
    # Per-profile ownership, checked on every route. Registered app-wide (not
    # per router) so a new route that takes a user_id is protected without
    # anyone remembering to add a check; it ignores non-API paths and requests
    # that name no user. See app/core/profile_ownership.py.
    dependencies=[Depends(require_profile_owner)],
)

# Authentication middleware (must be added before CORS so it runs after CORS)
app.add_middleware(APIKeyAuthMiddleware)

# CORS middleware
# Only origins listed in CORS_ALLOW_ORIGINS may call the API from another
# site; by default none may (see Settings.cors_allow_origins for why).
# - allow_credentials=False: the API uses no cookies, and Starlette answers a
#   credentialed request under "*" by echoing the caller's origin back, which
#   is what made the old setting allow every site.
# - Methods and headers are the ones the API and its UI actually use, so a
#   listed origin still cannot send arbitrary headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", PROFILE_TOKEN_HEADER],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# ==================== API Routers ====================

# Original core API router
app.include_router(router, prefix="/api/v1", tags=["api"])

# Feature routers without prefix — mount at /api/v1
app.include_router(skills_translator_router, prefix="/api/v1")
app.include_router(explainability_router, prefix="/api/v1")
app.include_router(training_paths_router, prefix="/api/v1")
app.include_router(events_router, prefix="/api/v1")

# Privacy routers — mount at /api/v1
app.include_router(data_export_router, prefix="/api/v1")
app.include_router(data_deletion_router, prefix="/api/v1")

# Self-prefixed routers (already include /api/v1) — mount at root
app.include_router(profile_router)
app.include_router(saved_occupations_router)
app.include_router(progress_tracker_router)
app.include_router(return_engagement_router)
app.include_router(session_router)

# Offline capability (route path already includes /api/v1) — mount at root
app.include_router(offline_router)

# Health checks — mount at root for k8s/load-balancer probes
app.include_router(health_router)

# Prometheus metrics — mount at root for scraping
app.include_router(metrics_router)

# Data enrichment pipeline
app.include_router(enrichment_router, prefix="/api/v1")


# ==================== UI Routes ====================

# Starlette 1.x (pulled in by the pinned FastAPI) takes the request as the
# first argument of TemplateResponse and no longer accepts the old
# TemplateResponse(name, {"request": request, ...}) form: every page here
# answered 500 until this was changed. The request is still available to the
# templates as `request`.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse(
        request,
        "pages/index.html",
        {"app_name": settings.app_name, "demo_mode": settings.is_demo_mode},
    )


@app.get("/flow/{user_id}", response_class=HTMLResponse)
async def user_flow(request: Request, user_id: int):
    """User flow page."""
    return templates.TemplateResponse(
        request,
        "pages/flow.html",
        {"user_id": user_id, "app_name": settings.app_name},
    )


@app.get("/docs-page", response_class=HTMLResponse)
async def docs_page(request: Request):
    """Documentation page."""
    return templates.TemplateResponse(
        request,
        "pages/docs.html",
        {"app_name": settings.app_name},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
