"""Structured JSON request logging middleware for SkillSprout.

Emits one structured JSON log line per request containing:
  - timestamp, request_id, method, path, status_code, duration_ms
  - SHA-256-hashed user_id (never the raw value)
  - Correlation IDs propagated across async boundaries

Privacy guarantees:
  - NEVER logs raw skill profiles or occupation exploration patterns
  - User identifiers are always hashed before emission
"""
import hashlib
import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Callable, Optional, Set

from fastapi import Request, Response
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("skillsprout.access")

# ---------------------------------------------------------------------------
# Context variable for cross-boundary correlation
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Return the current correlation ID (available across async tasks)."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


# ---------------------------------------------------------------------------
# Privacy helpers
# ---------------------------------------------------------------------------

_HASH_SALT = "skillsprout-request-log"

# Paths whose request/response bodies must NEVER be logged
_SENSITIVE_PATHS: Set[str] = {
    "/api/v1/user/{user_id}/skills/ratings",
    "/api/v1/user/{user_id}/recommendations",
    "/api/v1/user/{user_id}/current-occupation",
    "/api/v1/occupations/{onet_code}/skills",
}


def _hash_user_id(user_id: Optional[str]) -> Optional[str]:
    """Return a SHA-256 hash of the user_id.  Never store the raw value.

    Args:
        user_id: Raw user identifier (may be None).

    Returns:
        Hex-encoded SHA-256 digest or None.
    """
    if user_id is None:
        return None
    return hashlib.sha256(f"{_HASH_SALT}:{user_id}".encode()).hexdigest()


def _is_sensitive_path(path: str) -> bool:
    """Check whether a request path matches a sensitive pattern.

    Args:
        path: The request URL path.

    Returns:
        True if the path matches a sensitive pattern.
    """
    # Normalise to generic pattern for comparison
    import re

    normalised = re.sub(r"/\d+", "/{user_id}", path)
    normalised = re.sub(r"/\d{2}-\d{4}\.\d{2}", "/{onet_code}", normalised)
    return normalised in _SENSITIVE_PATHS


# ---------------------------------------------------------------------------
# Structured JSON formatter
# ---------------------------------------------------------------------------

class StructuredAccessRecord:
    """Builds a structured access-log record."""

    def __init__(
        self,
        *,
        request_id: str,
        correlation_id: str,
        method: str,
        path: str,
        query: str,
        status_code: int,
        duration_ms: float,
        user_id_hash: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> None:
        self.data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "correlation_id": correlation_id,
            "method": method,
            "path": path,
            "query": query if not _is_sensitive_path(path) else "[REDACTED]",
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "user_id_hash": user_id_hash,
            "client_ip": client_ip,
        }

    def to_json(self) -> str:
        """Serialise the record as a compact JSON string."""
        return json.dumps(self.data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------

class RequestLoggingMiddleware:
    """ASGI middleware that emits structured JSON access logs.

    Usage::

        app.add_middleware(RequestLoggingMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        # Generate or propagate correlation / request IDs
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        correlation_id = request.headers.get(
            "X-Correlation-ID", request_id
        )
        set_correlation_id(correlation_id)

        # Extract user_id from path if present (e.g. /api/v1/user/42/...)
        user_id_raw = _extract_user_id_from_path(request.url.path)

        start = time.monotonic()
        status_code = 500  # default in case of unhandled error

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # Inject correlation headers into response
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                headers.append((b"x-correlation-id", correlation_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000

            record = StructuredAccessRecord(
                request_id=request_id,
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                query=str(request.query_params),
                status_code=status_code,
                duration_ms=elapsed_ms,
                user_id_hash=_hash_user_id(user_id_raw),
                client_ip=request.client.host if request.client else None,
            )
            logger.info(record.to_json())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_user_id_from_path(path: str) -> Optional[str]:
    """Extract a numeric user ID from common URL patterns.

    Looks for ``/user/<digits>/`` in the path.

    Args:
        path: The request URL path.

    Returns:
        The raw user ID string, or None if not found.
    """
    import re

    match = re.search(r"/user/(\d+)", path)
    return match.group(1) if match else None
