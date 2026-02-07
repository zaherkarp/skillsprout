"""Accessibility and standards-compliance middleware for SkillSprout.

Enforces:
  - Proper ``Content-Type`` and ``charset`` headers on all responses
  - CORS headers (defence-in-depth alongside FastAPI's CORSMiddleware)
  - Response-time budget logging (warns when responses exceed 3 seconds)
  - Human-readable descriptions injected alongside coded values in JSON
    responses (e.g. bucket codes get a ``_description`` sibling key)
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from starlette.types import ASGIApp, Receive, Scope, Send
from fastapi import Request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESPONSE_TIME_BUDGET_SECONDS = 3.0

# Human-readable descriptions for coded values used in the API
_CODE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "bucket": {
        "ready_now": "Jobs you can apply to immediately based on your current skills",
        "trainable": "Jobs within reach with focused training (months, not years)",
        "long_reskill": "Jobs requiring significant reskilling or formal education",
    },
    "action_type": {
        "click": "User clicked to view more details",
        "save": "User saved the recommendation for later",
        "hide": "User chose to hide this recommendation",
        "apply": "User applied for this position",
        "interview": "User received an interview invitation",
        "offer": "User received a job offer",
    },
}


# ---------------------------------------------------------------------------
# Description injection
# ---------------------------------------------------------------------------

def inject_descriptions(data: Any) -> Any:
    """Recursively inject human-readable ``_description`` fields.

    For every key in ``_CODE_DESCRIPTIONS``, if the value is a known code,
    a sibling key ``<key>_description`` is added with the human-readable text.

    Args:
        data: A JSON-compatible Python object (dict, list, or scalar).

    Returns:
        The augmented data structure.
    """
    if isinstance(data, list):
        return [inject_descriptions(item) for item in data]

    if not isinstance(data, dict):
        return data

    augmented: Dict[str, Any] = {}
    for key, value in data.items():
        augmented[key] = inject_descriptions(value)
        # Add description sibling if applicable
        if key in _CODE_DESCRIPTIONS and isinstance(value, str):
            desc = _CODE_DESCRIPTIONS[key].get(value)
            if desc:
                augmented[f"{key}_description"] = desc

    return augmented


# ---------------------------------------------------------------------------
# ASGI middleware
# ---------------------------------------------------------------------------

class AccessibilityMiddleware:
    """ASGI middleware that enforces accessibility and standards compliance.

    Responsibilities:
      1. Ensures ``Content-Type`` includes ``charset=utf-8`` for text/* and
         application/json responses.
      2. Adds CORS headers as a defence-in-depth layer.
      3. Logs a warning when the response exceeds the time budget.
      4. Injects human-readable descriptions into JSON responses.

    Usage::

        app.add_middleware(AccessibilityMiddleware)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start = time.monotonic()

        response_headers: list = []
        response_status: int = 200
        body_parts: list[bytes] = []

        async def capture_send(message: dict) -> None:
            nonlocal response_headers, response_status
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))

        await self.app(scope, receive, capture_send)

        elapsed = time.monotonic() - start

        # --- Time budget warning ---
        if elapsed > RESPONSE_TIME_BUDGET_SECONDS:
            logger.warning(
                "Response time budget exceeded: %.2f s for %s %s (budget: %.1f s)",
                elapsed,
                request.method,
                request.url.path,
                RESPONSE_TIME_BUDGET_SECONDS,
            )

        full_body = b"".join(body_parts)

        # --- Detect JSON content type ---
        content_type = ""
        content_type_idx: Optional[int] = None
        for idx, (name, value) in enumerate(response_headers):
            if name.lower() == b"content-type":
                content_type = value.decode("utf-8", errors="replace")
                content_type_idx = idx
                break

        is_json = "application/json" in content_type

        # --- Inject descriptions into JSON responses ---
        if is_json and full_body:
            try:
                data = json.loads(full_body)
                data = inject_descriptions(data)
                full_body = json.dumps(data, separators=(",", ":")).encode("utf-8")
            except (json.JSONDecodeError, Exception):
                pass

        # --- Ensure charset=utf-8 ---
        if content_type and "charset" not in content_type.lower():
            if is_json or content_type.startswith("text/"):
                new_ct = f"{content_type}; charset=utf-8"
                if content_type_idx is not None:
                    response_headers[content_type_idx] = (
                        b"content-type",
                        new_ct.encode(),
                    )

        # --- CORS defence-in-depth headers ---
        existing_header_names = {name.lower() for name, _ in response_headers}
        cors_headers = [
            (b"access-control-allow-origin", b"*"),
            (b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"),
            (b"access-control-allow-headers", b"Content-Type, Authorization, X-Request-ID, X-Correlation-ID"),
            (b"access-control-expose-headers", b"X-Request-ID, X-Correlation-ID, ETag"),
        ]
        for name, value in cors_headers:
            if name not in existing_header_names:
                response_headers.append((name, value))

        # --- Add response-time header ---
        response_headers.append(
            (b"x-response-time-ms", str(round(elapsed * 1000, 2)).encode())
        )

        # --- Update content-length ---
        new_headers = []
        for name, value in response_headers:
            if name.lower() != b"content-length":
                new_headers.append((name, value))
        new_headers.append((b"content-length", str(len(full_body)).encode()))

        await send({
            "type": "http.response.start",
            "status": response_status,
            "headers": new_headers,
        })
        await send({"type": "http.response.body", "body": full_body})
