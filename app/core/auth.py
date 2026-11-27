"""API key authentication middleware for SkillSprout.

Sprint Decision (Week 1, Engineers E1/E2/E3 + PM):
  - E1 (ML): "We need auth that doesn't break existing test fixtures."
  - E2 (UX): "Health and metrics endpoints must remain open for k8s probes."
  - E3 (Infra): "API key auth is the minimum viable gate. OAuth2/JWT is Week 3."
  - PM: "Ship API key now. Auth disabled by default for dev. Enabled in prod via
    AUTH_ENABLED=true + API_KEY=<secret>. Open endpoints: /, /health*, /metrics,
    /docs, /openapi.json, /static, /flow, /docs-page."

This middleware checks for a valid API key in the X-API-Key header on all
protected routes. It is intentionally simple -- the infra audit flagged
"no authentication" as CRITICAL, and this is the minimal fix that unblocks
deployment without breaking the development workflow.
"""

import hmac
import logging
from typing import Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import settings

logger = logging.getLogger(__name__)

# Paths that never require authentication
OPEN_PATHS: Set[str] = {
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/docs-page",
    "/health",
    "/health/ready",
    "/health/detailed",
    "/metrics",
}

# Path prefixes that never require authentication
OPEN_PREFIXES = (
    "/static/",
    "/flow/",
)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces API key authentication on protected routes.

    When ``settings.auth_enabled`` is False (default in dev), all requests
    pass through without checks. When enabled, requests to protected
    endpoints must include a valid ``X-API-Key`` header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not settings.auth_enabled:
            return await call_next(request)

        path = request.url.path

        # Allow open paths
        if path in OPEN_PATHS or path.startswith(OPEN_PREFIXES):
            return await call_next(request)

        # Check API key
        api_key = request.headers.get("X-API-Key", "")
        # hmac.compare_digest takes the same time however many leading
        # characters match, so response timing does not leak the key. The
        # empty checks keep an unset API_KEY from matching an empty header
        # (auth on with no key configured stays closed).
        if (
            not api_key
            or not settings.api_key
            or not hmac.compare_digest(api_key.encode(), settings.api_key.encode())
        ):
            logger.warning(
                "Unauthorized request to %s from %s",
                path,
                request.client.host if request.client else "unknown",
            )
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
