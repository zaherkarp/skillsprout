"""Bandwidth-optimised API utilities for SkillSprout.

Provides:
  - GZip response compression middleware
  - ``?lite=true`` mode that strips responses to essential fields only
  - Cursor/offset pagination via ``?page=1&page_size=10``
  - ETag headers for cache-friendly responses
"""
import gzip
import hashlib
import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lightweight"])


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------

class PaginationMeta(BaseModel):
    """Pagination metadata included in paginated responses."""

    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel):
    """Generic wrapper for paginated list responses."""

    data: List[Any]
    pagination: PaginationMeta


def paginate(
    items: Sequence[Any],
    page: int = 1,
    page_size: int = 10,
) -> PaginatedResponse:
    """Apply offset-based pagination to a list of items.

    Args:
        items: Full sequence of items.
        page: 1-based page number.
        page_size: Number of items per page.

    Returns:
        A ``PaginatedResponse`` containing the page slice and metadata.
    """
    page = max(1, page)
    page_size = max(1, min(page_size, 100))  # cap at 100

    total_items = len(items)
    total_pages = max(1, math.ceil(total_items / page_size))

    start = (page - 1) * page_size
    end = start + page_size
    page_data = list(items[start:end])

    return PaginatedResponse(
        data=page_data,
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        ),
    )


# ---------------------------------------------------------------------------
# Lite-mode field filtering
# ---------------------------------------------------------------------------

# Which fields to keep when ``?lite=true`` is requested.
# Keys are response-model names; values are sets of field names to retain.
_LITE_FIELDS: Dict[str, set] = {
    "RecommendedOccupation": {
        "onet_code",
        "title",
        "rank",
        "bucket",
        "match_score",
    },
    "RecommendationResponse": {
        "event_id",
        "user_id",
        "model_version",
        "buckets",
        "total_recommendations",
    },
    "RecommendationBucket": {
        "bucket_name",
        "bucket_label",
        "occupations",
    },
    "OccupationDetail": {
        "code",
        "title",
    },
    "OccupationWithSkills": {
        "code",
        "title",
        "skills",
    },
}


def lite_filter(obj: Any, model_name: Optional[str] = None) -> Any:
    """Recursively strip a Pydantic-model-like dict to its lite fields.

    Args:
        obj: A dict (or list of dicts) derived from a Pydantic model.
        model_name: Optional model name hint for field lookup.

    Returns:
        The filtered structure.
    """
    if isinstance(obj, list):
        return [lite_filter(item, model_name) for item in obj]

    if not isinstance(obj, dict):
        return obj

    # Try to infer model name from keys present
    resolved_name = model_name
    if resolved_name is None:
        for name, fields in _LITE_FIELDS.items():
            if fields.issubset(set(obj.keys())):
                resolved_name = name
                break

    keep = _LITE_FIELDS.get(resolved_name or "", None)
    if keep is None:
        return obj  # no lite spec -- return as-is

    filtered: Dict[str, Any] = {}
    for key in keep:
        if key in obj:
            value = obj[key]
            # Recursively filter nested structures
            if isinstance(value, list):
                filtered[key] = [lite_filter(v) for v in value]
            elif isinstance(value, dict):
                filtered[key] = lite_filter(value)
            else:
                filtered[key] = value
    return filtered


# ---------------------------------------------------------------------------
# ETag helpers
# ---------------------------------------------------------------------------

def compute_etag(body: bytes) -> str:
    """Compute a weak ETag from the response body.

    Args:
        body: The raw response body bytes.

    Returns:
        A weak ETag string (e.g. ``W/"abc123"``).
    """
    digest = hashlib.md5(body).hexdigest()  # nosec -- not for security
    return f'W/"{digest}"'


def etag_matches(request: Request, etag: str) -> bool:
    """Check whether the client already has this ETag cached.

    Args:
        request: The incoming request (checks ``If-None-Match``).
        etag: The current ETag value.

    Returns:
        True if the client sent a matching ``If-None-Match`` header.
    """
    if_none_match = request.headers.get("If-None-Match", "")
    return etag in if_none_match


# ---------------------------------------------------------------------------
# ASGI Middleware: Lite mode + ETag injection
# ---------------------------------------------------------------------------

class LightweightAPIMiddleware:
    """ASGI middleware that applies lite-mode filtering and ETag headers.

    When a request includes ``?lite=true``, the JSON response body is
    filtered to retain only essential fields.

    An ``ETag`` header is always set on JSON responses so clients can
    use conditional requests to save bandwidth.

    Usage::

        app.add_middleware(LightweightAPIMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        lite_mode = request.query_params.get("lite", "").lower() == "true"

        # If no lite mode and no caching optimisation needed, pass through
        if not lite_mode:
            # Still inject ETag on JSON responses
            await self._pass_with_etag(scope, receive, send, request)
            return

        # Buffer the response to apply lite filtering
        response_started = False
        response_headers: list = []
        response_status: int = 200
        body_parts: list[bytes] = []

        async def capture_send(message: dict) -> None:
            nonlocal response_started, response_headers, response_status
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))

        await self.app(scope, receive, capture_send)

        # Reassemble body
        full_body = b"".join(body_parts)

        # Apply lite filtering if the content is JSON
        content_type = ""
        for name, value in response_headers:
            if name.lower() == b"content-type":
                content_type = value.decode("utf-8", errors="replace")
                break

        if "application/json" in content_type and full_body:
            try:
                data = json.loads(full_body)
                data = lite_filter(data)
                full_body = json.dumps(data, separators=(",", ":")).encode("utf-8")
            except (json.JSONDecodeError, Exception):
                pass  # leave body unchanged

        # Compute ETag
        etag = compute_etag(full_body)

        # Check If-None-Match
        if etag_matches(request, etag):
            await send({"type": "http.response.start", "status": 304, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return

        # Update content-length and add ETag
        new_headers = []
        for name, value in response_headers:
            if name.lower() != b"content-length":
                new_headers.append((name, value))
        new_headers.append((b"content-length", str(len(full_body)).encode()))
        new_headers.append((b"etag", etag.encode()))

        await send({
            "type": "http.response.start",
            "status": response_status,
            "headers": new_headers,
        })
        await send({"type": "http.response.body", "body": full_body})

    async def _pass_with_etag(
        self, scope: Scope, receive: Receive, send: Send, request: Request
    ) -> None:
        """Forward the response while injecting an ETag header."""
        response_headers: list = []
        body_parts: list[bytes] = []
        response_status: int = 200

        async def capture_send(message: dict) -> None:
            nonlocal response_headers, response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))

        await self.app(scope, receive, capture_send)

        full_body = b"".join(body_parts)

        # Only add ETag for JSON responses
        content_type = ""
        for name, value in response_headers:
            if name.lower() == b"content-type":
                content_type = value.decode("utf-8", errors="replace")
                break

        if "application/json" in content_type and full_body:
            etag = compute_etag(full_body)
            if etag_matches(request, etag):
                await send({"type": "http.response.start", "status": 304, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            response_headers.append((b"etag", etag.encode()))

        await send({
            "type": "http.response.start",
            "status": response_status,
            "headers": response_headers,
        })
        await send({"type": "http.response.body", "body": full_body})


def add_compression(app: Any, minimum_size: int = 500) -> None:
    """Add GZip compression middleware to a FastAPI/Starlette application.

    Args:
        app: The FastAPI or Starlette application instance.
        minimum_size: Minimum response size in bytes before compression kicks in.
    """
    app.add_middleware(GZipMiddleware, minimum_size=minimum_size)
