"""Safe network defaults: CORS, bind address, and the API key comparison.

Background (security audit, Nov 2026): CORS was `*` with credentials allowed.
Starlette answers that by echoing any caller's Origin back, so any web page
the user visited could read, change and DELETE profiles on an instance
running on their machine (reproduced: a preflight from https://evil.example
was allowed and the cross-site DELETE succeeded). These tests pin the fix and
pair each "blocked" case with an "allowed" case, so a CORS setup that
refused everything would fail too.
"""

import importlib

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.core.config import Settings

EVIL = "https://evil.example"


def _cors_app(origins: list[str]) -> FastAPI:
    """A tiny app with the SAME CORS arguments app/main.py uses."""
    import inspect
    import app.main as main_module

    # Guard against this helper drifting from main.py: the arguments below
    # must be the ones main.py passes.
    src = inspect.getsource(main_module)
    assert "allow_origins=settings.cors_origin_list" in src
    assert "allow_credentials=False" in src

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Profile-Token"],
    )

    @app.delete("/api/v1/user/{user_id}/data")
    async def delete(user_id: int):
        return {"deleted": user_id}

    return app


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/user/2/data",
        headers={"Origin": origin, "Access-Control-Request-Method": "DELETE"},
    )


# --- settings defaults -----------------------------------------------------


def test_default_allows_no_cross_site_origin(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    assert Settings(_env_file=None).cors_origin_list == []


def test_origin_list_parses_commas_and_blanks(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", " https://a.example, ,http://localhost:5173 ")
    assert Settings(_env_file=None).cors_origin_list == [
        "https://a.example", "http://localhost:5173"]


def test_default_bind_is_loopback(monkeypatch):
    monkeypatch.delenv("API_HOST", raising=False)
    assert Settings(_env_file=None).api_host == "127.0.0.1"


# --- CORS behaviour ---------------------------------------------------------


def test_unlisted_origin_preflight_is_refused():
    client = TestClient(_cors_app([]))
    r = _preflight(client, EVIL)
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_unlisted_origin_gets_no_cors_headers_on_a_simple_request():
    client = TestClient(_cors_app([]))
    r = client.delete("/api/v1/user/2/data", headers={"Origin": EVIL})
    # The browser would block a cross-site DELETE at the preflight above; this
    # checks the response also carries nothing a browser would accept.
    assert "access-control-allow-origin" not in r.headers


def test_listed_origin_is_allowed_without_credentials():
    client = TestClient(_cors_app(["https://app.example"]))
    r = _preflight(client, "https://app.example")
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://app.example"
    assert "access-control-allow-credentials" not in r.headers
    # A listed origin does not unlock others.
    assert _preflight(client, EVIL).status_code == 400


def test_real_app_refuses_an_unlisted_origin():
    """End to end on app.main with the default (empty) list."""
    import app.main as main_module

    client = TestClient(main_module.app)
    r = _preflight(client, EVIL)
    assert "access-control-allow-origin" not in r.headers


# --- API key comparison ---------------------------------------------------


@pytest.mark.parametrize("header,configured,expected", [
    ("right-key", "right-key", 200),
    ("wrong-key", "right-key", 401),
    ("", "right-key", 401),
    ("", "", 401),          # auth on but no key configured stays closed
    ("anything", "", 401),
])
def test_api_key_check(monkeypatch, header, configured, expected):
    from unittest.mock import patch
    from app.core.auth import APIKeyAuthMiddleware

    app = FastAPI()
    app.add_middleware(APIKeyAuthMiddleware)

    @app.get("/api/v1/thing")
    async def thing():
        return {"ok": True}

    with patch("app.core.auth.settings") as s:
        s.auth_enabled = True
        s.api_key = configured
        headers = {"X-API-Key": header} if header else {}
        assert TestClient(app).get("/api/v1/thing", headers=headers).status_code == expected
