"""Per-profile ownership: only the holder of a profile's owner token may use it.

Why this exists (security audit, Nov 2026)
------------------------------------------
Profiles are anonymous and identified by a sequential integer id. Before this
module, any caller who could reach the API could read, change, export or
delete *any* profile just by counting: ``GET /api/v1/user/2/data-export``,
``DELETE /api/v1/user/2/data`` and so on. The shared ``X-API-Key`` does not
help: it is one key for the whole app, and the bundled web page would have to
embed it, where anyone can read it.

The model
---------
* When a profile is created, the server generates a random **owner token**
  (``secrets.token_urlsafe(32)``, 256 bits) and returns it **once**, in the
  creation response. It is never shown again.
* Only a SHA-256 hash of the token is stored (``user_profile.owner_token_hash``),
  so a leaked database or backup does not leak working tokens. A fast hash is
  fine here (unlike passwords) because the token is random and long, so there
  is nothing to brute-force.
* Every request under ``/api/`` that names a user id, whether in the path
  (``/user/{user_id}/...``), the query string (``?user_id=``) or a JSON body
  field (``"user_id": ...``), must carry the matching token in the
  ``X-Profile-Token`` header. :func:`require_profile_owner` enforces this as an
  app-wide dependency, so a new route that takes ``user_id`` is covered
  without anyone remembering to add a check. Routes that reach a user through
  another id (feedback, by recommendation event) call
  :func:`ensure_profile_owner` themselves.
* The bundled web page keeps the token in the browser's ``localStorage`` and
  sends it on every API call.

Profiles created before this change (legacy)
--------------------------------------------
They have no hash, so **no token matches and they are locked**. That is the
safe default: there is no way to tell their real owner from anyone else. To
hand one back to its owner, run on the server::

    python -m app.core.profile_ownership issue <user_id>

which stores a fresh token's hash and prints the token plus a one-time link
(``/flow/<id>#token=...``). Opening the link in the owner's browser saves the
token there. ``python -m app.core.profile_ownership unclaimed`` lists the
locked profiles. The token in the link sits after ``#``, so browsers never
send it to the server or put it in access logs.

Turning it off
--------------
``PROFILE_TOKENS_REQUIRED=false`` disables the check (single-user local use,
or while migrating). It defaults to on.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import secrets
import sys
from collections.abc import Iterable

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models.models import UserProfile

logger = logging.getLogger(__name__)

# Header the client sends the token in. Also listed in the CORS allow_headers
# (app/main.py) so a configured cross-site frontend can send it.
TOKEN_HEADER = "X-Profile-Token"

# 32 bytes from the OS CSPRNG, URL-safe base64: about 43 characters, 256 bits.
_TOKEN_BYTES = 32


# ---------------------------------------------------------------------------
# Token primitives
# ---------------------------------------------------------------------------


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a token. This is what the database stores."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_owner_token() -> tuple[str, str]:
    """Return ``(token, token_hash)`` for a new profile.

    The caller stores the hash and gives the token to the client exactly once.
    """
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    return token, hash_token(token)


def token_matches(presented: str | None, stored_hash: str | None) -> bool:
    """Constant-time check of a presented token against a stored hash.

    * No stored hash (legacy profile) never matches, so those stay locked
      until an operator issues a token.
    * No presented token never matches.
    * ``hmac.compare_digest`` keeps the timing independent of how many
      leading characters match.
    """
    if not presented or not stored_hash:
        return False
    return hmac.compare_digest(hash_token(presented), stored_hash)


# ---------------------------------------------------------------------------
# Request-time enforcement
# ---------------------------------------------------------------------------


def _as_int(value) -> int | None:
    """Coerce a user id candidate to int, or None if it is not one.

    A value that cannot be an id is left for the route's own validation (it
    answers 422).

    This errs on the side of accepting: Pydantic's lax mode turns "2", " 2 "
    and 2.0 into the int 2, so all of those must be treated as id 2 here too,
    or ``{"user_id": 2.0}`` would reach profile 2 without a check. Reading one
    too many values as ids is harmless (an id with no profile passes).
    """
    if isinstance(value, bool):  # bool is an int subclass; never a user id
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            pass
        try:
            as_float = float(text)
        except ValueError:
            return None
        return int(as_float) if as_float.is_integer() else None
    return None


async def _user_ids_named_in(request: Request) -> set[int]:
    """Every user id the request names: path, query string, or JSON body."""
    candidates: list = []

    # 1. Path parameter, e.g. /api/v1/user/{user_id}/data. Filled in by the
    #    router before dependencies run.
    if "user_id" in request.path_params:
        candidates.append(request.path_params["user_id"])

    # 2. Query string, e.g. ?user_id=2. getlist() so that ?user_id=1&user_id=2
    #    cannot sneak a second id past a check of the first.
    candidates.extend(request.query_params.getlist("user_id"))

    # 3. Top-level field of a JSON body, e.g. {"user_id": 2, ...}. FastAPI has
    #    already read and cached the body for routes that take one, so reading
    #    it here does not consume it. Anything that is not a JSON object is
    #    ignored (the route's own validation rejects it).
    content_type = request.headers.get("content-type", "")
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and "json" in content_type:
        try:
            body = await request.json()
        # Not valid JSON: json.JSONDecodeError and UnicodeDecodeError are both
        # ValueErrors. Narrow on purpose, so a real fault (say the DB) is not hidden.
        except ValueError:
            body = None
        if isinstance(body, dict) and "user_id" in body:
            candidates.append(body["user_id"])

    return {uid for uid in (_as_int(c) for c in candidates) if uid is not None}


async def ensure_profile_owner(
    db: AsyncSession, user_id: int, presented: str | None
) -> None:
    """Raise 403 unless ``presented`` is the owner token of profile ``user_id``.

    A profile that does not exist passes, so the route can answer its usual
    404. Ids are sequential, so hiding existence behind a 403 would protect
    nothing.
    """
    if not settings.profile_tokens_required:
        return
    result = await db.execute(
        select(UserProfile.id, UserProfile.owner_token_hash).where(
            UserProfile.id == user_id
        )
    )
    row = result.first()
    if row is None:
        return
    if not token_matches(presented, row.owner_token_hash):
        # Log the id, never the presented token.
        logger.warning(
            "Rejected request for profile %d: owner token missing or wrong", user_id
        )
        if row.owner_token_hash is None:
            detail = (
                "This profile was created before owner tokens existed and is locked. "
                "Whoever runs this server can issue a token for it with "
                "`python -m app.core.profile_ownership issue <user_id>`."
            )
        else:
            detail = f"This profile's owner token is missing or wrong (send it in the {TOKEN_HEADER} header)."
        raise HTTPException(status_code=403, detail=detail)


async def require_profile_owner(
    request: Request,
    db: AsyncSession = Depends(get_db),  # noqa: B008  (FastAPI idiom, used repo-wide)
) -> None:
    """App-wide dependency: check ownership of every user id an API call names.

    Registered on the FastAPI app (app/main.py), so it runs for every route.
    It only acts on paths under ``/api/``: the HTML pages (``/flow/{user_id}``)
    are plain browser navigations that cannot send a header, and they render
    no profile data themselves; the page's own API calls carry the token.

    ``get_db`` is shared with the route (FastAPI caches a dependency per
    request), so this costs one small extra query, not a second connection.
    """
    if not settings.profile_tokens_required:
        return
    if not request.url.path.startswith("/api/"):
        return
    presented = request.headers.get(TOKEN_HEADER)
    for user_id in sorted(await _user_ids_named_in(request)):
        await ensure_profile_owner(db, user_id, presented)


# ---------------------------------------------------------------------------
# Operator CLI: hand legacy profiles back to their owners
# ---------------------------------------------------------------------------


def _sync_session():
    """A plain synchronous session on DATABASE_URL_SYNC, for the CLI."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine(settings.database_url_sync)
    return Session(engine)


def issue_token(user_id: int, rotate: bool = False) -> str:
    """Store a fresh token hash for ``user_id`` and return the token.

    Refuses a profile that already has a token unless ``rotate`` is set,
    because rotating locks out whoever holds the current token.
    """
    with _sync_session() as session:
        row = session.get(UserProfile, user_id)
        if row is None:
            raise SystemExit(f"No profile with id {user_id}.")
        if row.owner_token_hash and not rotate:
            raise SystemExit(
                f"Profile {user_id} already has an owner token. Pass --rotate to replace it "
                "(the current holder will be locked out)."
            )
        token, token_hash = new_owner_token()
        row.owner_token_hash = token_hash
        session.commit()
        return token


def unclaimed_ids() -> Iterable[int]:
    """Ids of profiles with no owner token (locked legacy profiles)."""
    with _sync_session() as session:
        return [
            r[0]
            for r in session.execute(
                select(UserProfile.id)
                .where(UserProfile.owner_token_hash.is_(None))
                .order_by(UserProfile.id)
            )
        ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.core.profile_ownership",
        description="Issue owner tokens for SkillSprout profiles (mainly ones created before tokens existed).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_issue = sub.add_parser(
        "issue", help="issue a token for one profile and print it once"
    )
    p_issue.add_argument("user_id", type=int)
    p_issue.add_argument(
        "--rotate", action="store_true", help="replace an existing token"
    )
    p_issue.add_argument(
        "--base-url",
        default=f"http://127.0.0.1:{settings.api_port}",
        help="where the app is served, for the printed link (default: %(default)s)",
    )

    sub.add_parser("unclaimed", help="list profiles that have no owner token yet")

    args = parser.parse_args(argv)
    if args.command == "issue":
        token = issue_token(args.user_id, rotate=args.rotate)
        # Printed once and not logged anywhere. Give it only to the profile's owner.
        print(f"Owner token for profile {args.user_id} (shown once):")
        print(f"  {token}")
        print("Open this link in the owner's browser to save the token there:")
        print(f"  {args.base_url.rstrip('/')}/flow/{args.user_id}#token={token}")
        return 0
    if args.command == "unclaimed":
        ids = list(unclaimed_ids())
        print(f"{len(ids)} profile(s) without an owner token" + (":" if ids else "."))
        for uid in ids:
            print(f"  {uid}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
