"""
Private Mode Middleware for SkillSprout
========================================

RATIONALE: Some users want to explore career transitions without leaving
ANY trace on the server. This is not paranoia -- it is a legitimate need.
Consider:

  - An employee using a company-provided laptop during lunch break to
    explore new careers. If the employer monitors network traffic, they
    can see API calls to SkillSprout, but private mode ensures there is
    no server-side record that could be subpoenaed or accessed via a
    data breach.

  - A user who wants to "try before they commit" -- explore the tool
    casually without creating a persistent profile, then decide later
    whether to sign up for full tracking.

  - Compliance with the privacy principle of "data minimization" (GDPR
    Art. 5(1)(c)): do not collect data you do not strictly need. If the
    user explicitly opts out of collection, honor that.

HOW IT WORKS:

  1. The client sends the header `X-Private-Mode: true` with any request.

  2. This middleware intercepts the request BEFORE it reaches the endpoint.

  3. It sets `request.state.private_mode = True`, which downstream code
     checks before performing any database writes.

  4. In private mode:
     - Recommendations are computed in-memory and returned WITHOUT creating
       a RecommendationEvent or RecommendedOccupation in the database.
     - No UserFeedback records are created.
     - No search history is logged.
     - The response includes an `X-Private-Mode: active` header and a
       JSON field explaining what IS and IS NOT stored.

  5. What IS still stored in private mode:
     - Nothing. Zero server-side persistence.
     - O*NET data that was already cached (public TIER_1 data) is read
       but not modified.

  6. What IS NOT stored:
     - No recommendation events
     - No feedback records
     - No user profile creation or updates
     - No search/browsing history
     - No analytics events

TRADE-OFFS:

  - Users in private mode do not benefit from personalized calibration
    (the model cannot learn from their feedback if it is not recorded).
  - Session continuity is lost on page refresh (no server-side state).
  - This is BY DESIGN. Privacy and personalization are in tension, and
    we let the user decide where they fall on that spectrum.
"""

import logging
from typing import Any, Callable, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Header constants
# ---------------------------------------------------------------------------

# RATIONALE for header-based toggle: HTTP headers are the standard mechanism
# for per-request metadata. Using a header (vs. query param or cookie) keeps
# the toggle invisible in URL logs and browser history, which is itself a
# privacy benefit. The header name uses the X- prefix convention for custom
# application headers.

PRIVATE_MODE_REQUEST_HEADER = "X-Private-Mode"
PRIVATE_MODE_RESPONSE_HEADER = "X-Private-Mode"
PRIVATE_MODE_HEADER_VALUE = "true"

# ---------------------------------------------------------------------------
# Privacy disclosure: what IS and IS NOT stored
# ---------------------------------------------------------------------------
# RATIONALE: Transparency is a core GDPR principle (Art. 5(1)(a)). When
# private mode is active, we tell the user exactly what the system is and
# is not doing with their data. This is returned as a structured dict in
# the response so the frontend can display it.

PRIVATE_MODE_DISCLOSURE: Dict[str, Any] = {
    "private_mode": True,
    "storage_policy": {
        "stored": {
            "items": [],
            "explanation": (
                "In private mode, SkillSprout stores NOTHING about your "
                "session on the server. No database writes occur for any "
                "user-specific data."
            ),
        },
        "not_stored": {
            "items": [
                "Recommendation events (which occupations were suggested to you)",
                "Your feedback (clicks, saves, applications)",
                "Search history (what occupations you looked up)",
                "User profile data (skill ratings, current occupation)",
                "Analytics or tracking events",
                "Session identifiers linking requests together",
            ],
            "explanation": (
                "All computation happens in-memory and results are returned "
                "directly in the HTTP response. Once the response is sent, "
                "the server retains no record of the interaction."
            ),
        },
        "still_accessible": {
            "items": [
                "Public O*NET occupation data (already cached, not user-specific)",
                "Skill taxonomy data (public reference data)",
            ],
            "explanation": (
                "Public reference data (Tier 1) is read from the cache but "
                "is not modified or linked to your session in any way."
            ),
        },
    },
    "trade_offs": [
        "Recommendations are not personalized by calibration model feedback",
        "Session state is not preserved between requests",
        "You cannot save occupations or track applications in this mode",
    ],
}


# ---------------------------------------------------------------------------
# Helper: check if a request is in private mode
# ---------------------------------------------------------------------------


def is_private_mode(request: Request) -> bool:
    """
    Check whether the current request has private mode enabled.

    This can be called from any endpoint to conditionally skip DB writes.

    RATIONALE: Providing a simple boolean helper avoids duplicating header
    parsing logic across every endpoint. Endpoints call this function and
    branch accordingly.
    """
    # Check the header first (primary mechanism).
    header_value = request.headers.get(PRIVATE_MODE_REQUEST_HEADER, "").lower()
    if header_value == PRIVATE_MODE_HEADER_VALUE:
        return True

    # Also check request.state in case the middleware has already processed
    # the request (useful for nested calls / dependency injection).
    return getattr(request.state, "private_mode", False)


def get_private_mode_disclosure() -> Dict[str, Any]:
    """
    Return the privacy disclosure dict for inclusion in API responses
    when private mode is active.
    """
    return dict(PRIVATE_MODE_DISCLOSURE)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class PrivateModeMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that detects the X-Private-Mode header and configures
    the request context accordingly.

    RATIONALE for middleware approach: Using middleware ensures that private
    mode is detected ONCE at the beginning of the request lifecycle, before
    any endpoint code runs. This is more reliable than checking the header
    in every individual endpoint, which is error-prone (a developer might
    forget to check in a new endpoint).

    The middleware:
      1. Reads the X-Private-Mode header.
      2. Sets request.state.private_mode = True/False.
      3. Passes the request to the next handler.
      4. Adds the X-Private-Mode response header so the client can confirm
         that private mode was honored.
      5. Logs that a private-mode request was processed (without logging
         any request details -- that would defeat the purpose).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Detect private mode from the request header.
        private_mode = (
            request.headers.get(PRIVATE_MODE_REQUEST_HEADER, "").lower()
            == PRIVATE_MODE_HEADER_VALUE
        )

        # Store on request.state so all downstream code can check it.
        request.state.private_mode = private_mode

        if private_mode:
            # RATIONALE: We log that a private-mode request occurred, but
            # we deliberately do NOT log the path, query params, user agent,
            # or any other identifying information. The log entry exists
            # solely for operational monitoring (e.g., "how many private-mode
            # requests are we getting?").
            logger.debug("Private mode request received")

        # Process the request through the normal handler chain.
        response = await call_next(request)

        if private_mode:
            # Add response header so the client can confirm private mode
            # was active for this request.
            response.headers[PRIVATE_MODE_RESPONSE_HEADER] = "active"

            # RATIONALE: Adding the disclosure as a header would be unwieldy
            # (it is a large JSON structure). Instead, endpoints that support
            # private mode include the disclosure in their response body.
            # The response header is a simple signal for the client.

        return response


# ---------------------------------------------------------------------------
# Session-only storage helper
# ---------------------------------------------------------------------------
# RATIONALE: In private mode, we may need temporary storage for the duration
# of a single request (e.g., to pass data between dependencies). This dict
# lives on request.state and is garbage-collected when the request completes.
# It is NEVER written to a database or persisted in any way.


def get_session_store(request: Request) -> Dict[str, Any]:
    """
    Get or create a session-only in-memory store for private mode.

    This store exists only for the lifetime of the current HTTP request.
    It is NOT persisted to any database, cache, or file system.

    Usage in endpoints:
        if is_private_mode(request):
            store = get_session_store(request)
            store["recommendations"] = computed_results
            # These results are returned in the response and then discarded.
    """
    if not hasattr(request.state, "_private_session_store"):
        request.state._private_session_store = {}
    return request.state._private_session_store


# ---------------------------------------------------------------------------
# Decorator for endpoints that support private mode
# ---------------------------------------------------------------------------


def supports_private_mode(func: Callable) -> Callable:
    """
    Decorator that marks an endpoint as private-mode aware.

    RATIONALE: This decorator serves as documentation and enables automated
    discovery of which endpoints properly handle private mode. It does not
    change the endpoint's behavior -- that responsibility belongs to the
    endpoint implementation, which checks is_private_mode(request) and
    branches accordingly.

    In the future, this could be extended to automatically inject the
    privacy disclosure into the response body.
    """
    func._supports_private_mode = True  # type: ignore[attr-defined]
    return func
