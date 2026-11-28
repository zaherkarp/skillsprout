"""Per-profile ownership: a profile's data is reachable only with its owner token.

Background (security audit, Nov 2026): profiles are anonymous and numbered
1, 2, 3..., and every per-user route trusted the id it was given, so anyone
who could reach the API could read, change, export or delete any profile. The
fix (app/core/profile_ownership.py) issues a random owner token at creation,
stores only its hash, and checks it on every API call that names a user.

How these tests are built
-------------------------
* Each "rejected" case is paired with an "accepted" one using the right
  token, so a check that refused everything would fail too.
* ``test_every_route_naming_a_user_is_guarded`` walks the app's real route
  table instead of a hand-written list, so a route added later that takes a
  ``user_id`` (path, query or JSON body) is tested automatically.
* Each test gets its own in-memory SQLite database (StaticPool keeps one
  connection, so the schema survives across requests).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.core.privacy.retention_policy
import app.models.models
from app.core import profile_ownership as po
from app.core.config import settings
from app.db.session import Base, get_db
from app.main import app
from app.models.models import RecommendationEvent, UserProfile

HEADER = po.TOKEN_HEADER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory():
    """A fresh in-memory database per test, wired into the app via get_db."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    async def _init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init())
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _db
    yield factory
    app.dependency_overrides.pop(get_db, None)
    asyncio.run(engine.dispose())


@pytest.fixture
def client(session_factory, monkeypatch):
    # Enforcement on (the shipped default), API-key auth off (also default),
    # so these tests isolate the ownership check.
    monkeypatch.setattr(settings, "profile_tokens_required", True)
    monkeypatch.setattr(settings, "auth_enabled", False)
    with TestClient(app) as c:
        yield c


def _create(client, path="/api/v1/user/profile", body=None):
    """Create a profile and return (id, token)."""
    r = client.post(path, json=body if body is not None else {})
    assert r.status_code == 201, r.text
    data = r.json()
    uid = data["id"] if "id" in data else data["user_id"]
    return uid, data["owner_token"]


# ---------------------------------------------------------------------------
# Token primitives
# ---------------------------------------------------------------------------


def test_new_tokens_are_long_random_and_stored_only_as_a_hash():
    t1, h1 = po.new_owner_token()
    t2, h2 = po.new_owner_token()
    assert t1 != t2 and h1 != h2
    assert len(t1) >= 43  # 32 random bytes, base64url
    assert len(h1) == 64 and t1 not in h1  # hex SHA-256, not the token
    assert po.hash_token(t1) == h1


@pytest.mark.parametrize(
    "presented,stored,ok",
    [
        ("right", po.hash_token("right"), True),
        ("wrong", po.hash_token("right"), False),
        ("", po.hash_token("right"), False),
        (None, po.hash_token("right"), False),
        ("anything", None, False),  # legacy profile: locked, nothing matches
        ("", None, False),
    ],
)
def test_token_matches(presented, stored, ok):
    assert po.token_matches(presented, stored) is ok


@pytest.mark.parametrize(
    "value,expected",
    [
        (2, 2),
        ("2", 2),
        (" 2 ", 2),
        (2.0, 2),
        ("2.0", 2),
        ("+2", 2),
        ("-1", -1),
        (2.5, None),
        ("abc", None),
        (None, None),
        (True, None),
        ([2], None),
    ],
)
def test_user_id_coercion_matches_pydantic_lax_mode(value, expected):
    # Anything Pydantic would turn into int 2 must be checked as id 2, or
    # {"user_id": 2.0} would slip past the check.
    assert po._as_int(value) == expected


# ---------------------------------------------------------------------------
# Creation returns the token once
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/v1/user/profile", "/api/v1/profile"])
def test_both_creation_routes_return_a_token_and_store_only_its_hash(
    client, session_factory, path
):
    uid, token = _create(client, path)
    assert token

    async def stored():
        async with session_factory() as s:
            return (await s.get(UserProfile, uid)).owner_token_hash

    assert asyncio.run(stored()) == po.hash_token(token)

    # The token is never returned again.
    r = client.get(f"/api/v1/profile/{uid}", headers={HEADER: token})
    assert r.status_code == 200
    assert r.json().get("owner_token") is None


# ---------------------------------------------------------------------------
# The audit's reproduction, now blocked
# ---------------------------------------------------------------------------


def test_another_profile_cannot_be_exported_or_deleted(client):
    a, token_a = _create(client)
    b, token_b = _create(client)

    # No token, and A's token, are both refused for B.
    for headers in ({}, {HEADER: token_a}, {HEADER: "x" * 43}):
        assert (
            client.get(f"/api/v1/user/{b}/data-export", headers=headers).status_code
            == 403
        )
        assert (
            client.delete(f"/api/v1/user/{b}/data", headers=headers).status_code == 403
        )

    # B's own token works, and B still exists after the refused delete.
    assert (
        client.get(
            f"/api/v1/user/{b}/data-export", headers={HEADER: token_b}
        ).status_code
        == 200
    )
    assert (
        client.delete(f"/api/v1/user/{b}/data", headers={HEADER: token_b}).status_code
        == 200
    )
    # Gone now: a missing profile answers its normal 404, not 403.
    assert (
        client.get(
            f"/api/v1/user/{b}/data-export", headers={HEADER: token_b}
        ).status_code
        == 404
    )
    # A is untouched.
    assert (
        client.get(
            f"/api/v1/user/{a}/data-export", headers={HEADER: token_a}
        ).status_code
        == 200
    )


@pytest.mark.parametrize("where", ["query", "body"])
def test_query_and_body_user_ids_are_checked(client, where):
    _a, token_a = _create(client)
    b, token_b = _create(client)
    if where == "query":

        def call(token):
            return client.patch(
                f"/api/v1/user/preferences?user_id={b}",
                json={"risk_tolerance": "standard"},
                headers={HEADER: token},
            )
    else:

        def call(token):
            return client.post(
                "/api/v1/saved-occupations",
                json={"user_id": b, "onet_code": "15-1252.00"},
                headers={HEADER: token},
            )

    assert call(token_a).status_code == 403
    assert call(token_b).status_code != 403


def test_two_ids_in_one_request_must_both_be_owned(client):
    a, token_a = _create(client)
    b, _ = _create(client)
    # ?user_id=a&user_id=b: holding a's token must not unlock b.
    r = client.get(
        f"/api/v1/recommendations/export?user_id={a}&user_id={b}",
        headers={HEADER: token_a},
    )
    assert r.status_code == 403


def test_float_id_in_body_does_not_bypass(client):
    _a, token_a = _create(client)
    b, _ = _create(client)
    r = client.post(
        "/api/v1/saved-occupations",
        json={"user_id": float(b), "onet_code": "15-1252.00"},
        headers={HEADER: token_a},
    )
    assert r.status_code == 403


def test_feedback_is_checked_through_its_event(client, session_factory):
    _a, token_a = _create(client)
    b, token_b = _create(client)

    async def add_event():
        async with session_factory() as s:
            ev = RecommendationEvent(
                user_id=b, current_onet_code="15-1252.00", model_version="t"
            )
            s.add(ev)
            await s.commit()
            return ev.id

    event_id = asyncio.run(add_event())
    body = {
        "event_id": event_id,
        "target_onet_code": "15-1251.00",
        "action_type": "click",
    }
    assert (
        client.post(
            "/api/v1/feedback", json=body, headers={HEADER: token_a}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/v1/feedback", json=body, headers={HEADER: token_b}
        ).status_code
        == 201
    )


def test_html_page_needs_no_header(client):
    uid, _ = _create(client)
    # A browser navigation cannot send a header; the page shows no data itself.
    assert client.get(f"/flow/{uid}").status_code == 200


def test_switch_off_restores_open_access(client, monkeypatch):
    _create(client)
    b, _ = _create(client)
    monkeypatch.setattr(settings, "profile_tokens_required", False)
    assert client.get(f"/api/v1/user/{b}/data-export").status_code == 200


# ---------------------------------------------------------------------------
# Every route that names a user is guarded (walks the real route table)
# ---------------------------------------------------------------------------


def _routes_naming_a_user():
    """(method, path, where) for every /api route that takes a user_id."""
    found = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/api/"):
            continue
        d = route.dependant
        where = None
        if any(p.name == "user_id" for p in d.path_params):
            where = "path"
        elif any(p.name == "user_id" for p in d.query_params):
            where = "query"
        else:
            for p in d.body_params:
                model = getattr(p.field_info, "annotation", None)
                if "user_id" in getattr(model, "model_fields", {}):
                    where = "body"
        if where:
            for method in sorted(route.methods):
                found.append((method, route.path, where))
    return found


ROUTES = _routes_naming_a_user()


def test_route_inventory_is_not_empty():
    # Guards the walk itself: if FastAPI internals change and it finds
    # nothing, the parametrized test below would pass vacuously.
    paths = {p for _, p, _ in ROUTES}
    assert "/api/v1/user/{user_id}/data" in paths
    assert "/api/v1/user/preferences" in paths  # query
    assert "/api/v1/saved-occupations" in paths  # body
    assert len(ROUTES) >= 15


@pytest.mark.parametrize(
    "method,path,where", ROUTES, ids=[f"{m} {p}" for m, p, _ in ROUTES]
)
def test_every_route_naming_a_user_is_guarded(client, method, path, where):
    owner, owner_token = _create(client)
    _, other_token = _create(client)

    # Fill other path parameters with harmless placeholders.
    url = (
        path.replace("{user_id}", str(owner))
        .replace("{onet_code}", "15-1252.00")
        .replace("{occupation_id}", "15-1252.00")
        .replace("{saved_id}", "none")
    )
    kwargs = {}
    if where == "query":
        url += f"?user_id={owner}"
    if method in ("POST", "PATCH", "PUT", "DELETE"):
        kwargs["content"] = json.dumps({"user_id": owner} if where == "body" else {})
        kwargs["headers"] = {"Content-Type": "application/json"}

    def call(token):
        headers = dict(kwargs.get("headers", {}))
        if token:
            headers[HEADER] = token
        return client.request(
            method, url, content=kwargs.get("content"), headers=headers
        )

    assert call(None).status_code == 403
    assert call(other_token).status_code == 403
    # With the right token the ownership check passes (the route may still
    # answer 404/422 for the placeholder values; it must not be 403).
    assert call(owner_token).status_code != 403


# ---------------------------------------------------------------------------
# Legacy profiles and the operator CLI
# ---------------------------------------------------------------------------


def test_legacy_profile_is_locked_until_a_token_is_issued(
    tmp_path, monkeypatch, capsys
):
    """End to end on a SQLite file, as `make run` uses."""
    from sqlalchemy import create_engine, text

    db_file = tmp_path / "legacy.db"
    sync_url = f"sqlite:///{db_file}"
    monkeypatch.setattr(settings, "database_url_sync", sync_url)

    # A database from before owner tokens: the old user_profile shape, one row.
    eng = create_engine(sync_url)
    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE user_profile (id INTEGER PRIMARY KEY, created_at DATETIME NOT NULL, "
                "updated_at DATETIME NOT NULL, metadata_json JSON)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO user_profile (id, created_at, updated_at) VALUES (7, '2026-01-01', '2026-01-01')"
            )
        )

        # The startup bootstrap adds the missing column in place.
        from app.main import _add_owner_token_column_if_missing

        _add_owner_token_column_if_missing(conn)
        _add_owner_token_column_if_missing(conn)  # idempotent
        hash_before = conn.execute(
            text("SELECT owner_token_hash FROM user_profile WHERE id = 7")
        ).scalar()
    assert hash_before is None  # locked: no token can match NULL
    assert po.token_matches("anything", hash_before) is False

    assert po.main(["unclaimed"]) == 0
    assert "7" in capsys.readouterr().out

    assert po.main(["issue", "7", "--base-url", "http://127.0.0.1:8000"]) == 0
    out = capsys.readouterr().out
    token = out.split("shown once):")[1].split()[0]
    assert f"http://127.0.0.1:8000/flow/7#token={token}" in out

    with eng.begin() as conn:
        stored = conn.execute(
            text("SELECT owner_token_hash FROM user_profile WHERE id = 7")
        ).scalar()
    assert po.token_matches(token, stored)

    # A second issue is refused without --rotate (it would lock out the holder) ...
    with pytest.raises(SystemExit):
        po.main(["issue", "7"])
    # ... and with --rotate the old token stops working.
    assert po.main(["issue", "7", "--rotate"]) == 0
    with eng.begin() as conn:
        rotated = conn.execute(
            text("SELECT owner_token_hash FROM user_profile WHERE id = 7")
        ).scalar()
    assert not po.token_matches(token, rotated)

    assert po.main(["unclaimed"]) == 0
    assert "0 profile(s)" in capsys.readouterr().out


def test_locked_profile_gets_an_explanatory_403(client, session_factory):
    async def add_legacy():
        async with session_factory() as s:
            row = UserProfile(owner_token_hash=None)
            s.add(row)
            await s.commit()
            return row.id

    uid = asyncio.run(add_legacy())
    r = client.get(f"/api/v1/user/{uid}/data-export", headers={HEADER: "guess"})
    assert r.status_code == 403
    assert "profile_ownership issue" in r.json()["detail"]
