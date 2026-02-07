"""Session state export / import as encrypted shareable tokens.

Allows users to:
  - Export their current session state (skills, occupation, preferences)
    as a URL-safe encrypted token.
  - Import a previously exported token on any device to resume their
    session -- no account required.

Encryption uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256)
with a 30-day expiry baked into the token.
"""
import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/session", tags=["session"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Token lifetime in seconds (30 days)
SESSION_TOKEN_TTL_SECONDS: int = 30 * 24 * 3600

# Encryption key -- generate deterministically from a secret or env var.
# In production, set SKILLSPROUT_SESSION_KEY to a Fernet key.
_ENV_KEY = os.environ.get("SKILLSPROUT_SESSION_KEY")
_FERNET_KEY: bytes = (
    _ENV_KEY.encode() if _ENV_KEY else Fernet.generate_key()
)
_fernet = Fernet(_FERNET_KEY)


def _get_fernet() -> Fernet:
    """Return the module-level Fernet instance (testable seam)."""
    return _fernet


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SessionPayload(BaseModel):
    """The cleartext session state that gets encrypted into the token."""

    user_id: Optional[int] = None
    current_onet_code: Optional[str] = None
    skill_ratings: Dict[str, int] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    exported_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: int = 1


class ExportRequest(BaseModel):
    """Request body for session export."""

    user_id: Optional[int] = None
    current_onet_code: Optional[str] = None
    skill_ratings: Dict[str, int] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)


class ExportResponse(BaseModel):
    """Response containing the encrypted session token."""

    token: str = Field(description="URL-safe Fernet-encrypted session token")
    expires_in_seconds: int
    exported_at: str


class ImportRequest(BaseModel):
    """Request body for session import."""

    token: str = Field(description="Previously exported session token")


class ImportResponse(BaseModel):
    """Response after successfully importing a session token."""

    user_id: Optional[int] = None
    current_onet_code: Optional[str] = None
    skill_ratings: Dict[str, int]
    preferences: Dict[str, Any]
    exported_at: str
    token_age_seconds: float


# ---------------------------------------------------------------------------
# Encryption / decryption
# ---------------------------------------------------------------------------

def encrypt_session(payload: SessionPayload) -> str:
    """Encrypt a session payload into a URL-safe token.

    Args:
        payload: The session data to encrypt.

    Returns:
        A URL-safe string token.
    """
    plaintext = json.dumps(payload.model_dump(), separators=(",", ":")).encode("utf-8")
    token_bytes = _get_fernet().encrypt(plaintext)
    # Fernet tokens are already URL-safe base64
    return token_bytes.decode("ascii")


def decrypt_session(token: str, max_age: int = SESSION_TOKEN_TTL_SECONDS) -> SessionPayload:
    """Decrypt and validate a session token.

    Args:
        token: The encrypted token string.
        max_age: Maximum allowed age in seconds (default 30 days).

    Returns:
        The decrypted ``SessionPayload``.

    Raises:
        ValueError: If the token is expired, tampered with, or malformed.
    """
    try:
        plaintext = _get_fernet().decrypt(token.encode("ascii"), ttl=max_age)
    except InvalidToken:
        raise ValueError("Invalid or expired session token")

    try:
        data = json.loads(plaintext.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Malformed session token payload: {exc}")

    return SessionPayload(**data)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/export", response_model=ExportResponse)
async def export_session(request: ExportRequest) -> ExportResponse:
    """Export current session state as an encrypted URL-safe token.

    The token contains the user's skill ratings, current occupation,
    and preferences.  It is encrypted with Fernet (AES-128-CBC + HMAC)
    and automatically expires after 30 days.

    No account is required -- the token IS the session.
    """
    payload = SessionPayload(
        user_id=request.user_id,
        current_onet_code=request.current_onet_code,
        skill_ratings=request.skill_ratings,
        preferences=request.preferences,
    )

    token = encrypt_session(payload)

    return ExportResponse(
        token=token,
        expires_in_seconds=SESSION_TOKEN_TTL_SECONDS,
        exported_at=payload.exported_at,
    )


@router.post("/import", response_model=ImportResponse)
async def import_session(request: ImportRequest) -> ImportResponse:
    """Restore session state from a previously exported token.

    Validates the Fernet token (checks expiry and HMAC), decrypts the
    payload, and returns the session data so the client can restore
    its local state.

    No account is required.
    """
    try:
        payload = decrypt_session(request.token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Calculate token age
    try:
        exported_dt = datetime.fromisoformat(payload.exported_at)
        age_seconds = (datetime.now(timezone.utc) - exported_dt).total_seconds()
    except Exception:
        age_seconds = 0.0

    return ImportResponse(
        user_id=payload.user_id,
        current_onet_code=payload.current_onet_code,
        skill_ratings=payload.skill_ratings,
        preferences=payload.preferences,
        exported_at=payload.exported_at,
        token_age_seconds=round(age_seconds, 2),
    )
