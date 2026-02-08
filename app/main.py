"""FastAPI application main entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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

# -- Authentication middleware --------------------------------------------
from app.core.auth import APIKeyAuthMiddleware

# -- Progressive enhancement (various mounting strategies) -----------------
from app.core.progressive.session_resumption import router as session_router
from app.core.progressive.offline_capability import router as offline_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info(f"Starting {settings.app_name}")
    logger.info(f"Environment: {settings.env}")
    logger.info(f"Demo mode: {settings.is_demo_mode}")
    mark_startup()

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
)

# Authentication middleware (must be added before CORS so it runs after CORS)
app.add_middleware(APIKeyAuthMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# ==================== UI Routes ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse(
        "pages/index.html",
        {"request": request, "app_name": settings.app_name, "demo_mode": settings.is_demo_mode},
    )


@app.get("/flow/{user_id}", response_class=HTMLResponse)
async def user_flow(request: Request, user_id: int):
    """User flow page."""
    return templates.TemplateResponse(
        "pages/flow.html",
        {"request": request, "user_id": user_id, "app_name": settings.app_name},
    )


@app.get("/docs-page", response_class=HTMLResponse)
async def docs_page(request: Request):
    """Documentation page."""
    return templates.TemplateResponse(
        "pages/docs.html",
        {"request": request, "app_name": settings.app_name},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
