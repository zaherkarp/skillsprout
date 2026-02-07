"""User profile with constraints for SkillSprout.

Profile creation is **optional** -- the application works without one.
When a profile exists it stores display preferences, job search
constraints (salary, location, remote preference, timeline), risk
tolerance, and a snapshot of current skills.

API endpoints:
    POST   /api/v1/profile          -- create profile (optional)
    GET    /api/v1/profile/{user_id} -- retrieve profile
    PATCH  /api/v1/profile/{user_id} -- partial update
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models.models import UserProfile as UserProfileRow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskTolerance(str, Enum):
    """How aggressively the user wants to explore career changes."""

    RELAXED = "relaxed"
    STANDARD = "standard"
    STRICT = "strict"


class RemotePreference(str, Enum):
    """User's preference for remote work."""

    REMOTE_ONLY = "remote_only"
    HYBRID = "hybrid"
    ON_SITE = "on_site"
    NO_PREFERENCE = "no_preference"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ProfileConstraints(BaseModel):
    """Job search constraints embedded in the user profile."""

    salary_minimum: Optional[float] = Field(
        None,
        ge=0,
        description="Minimum acceptable annual salary in USD.",
    )
    location: Optional[str] = Field(
        None,
        max_length=255,
        description="Preferred work location (city, state, or region).",
    )
    remote_preference: RemotePreference = Field(
        RemotePreference.NO_PREFERENCE,
        description="Remote work preference.",
    )
    industry_interests: List[str] = Field(
        default_factory=list,
        description="NAICS industry codes or free-text industry names.",
    )
    timeline_months: Optional[int] = Field(
        None,
        ge=1,
        le=60,
        description="Target timeline for career transition (months).",
    )


class SkillSnapshot(BaseModel):
    """Point-in-time snapshot of user's skill ratings."""

    element_id: str
    rating: int = Field(..., ge=0, le=4)
    skill_name: Optional[str] = None


class ProfileCreateRequest(BaseModel):
    """Request body for creating a user profile (optional)."""

    display_name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Display name (optional, no PII requirement).",
    )
    current_occupation_id: Optional[str] = Field(
        None,
        max_length=10,
        description="O*NET code of the user's current occupation.",
    )
    skills_snapshot: List[SkillSnapshot] = Field(
        default_factory=list,
        description="Optional snapshot of current skill ratings.",
    )
    constraints: Optional[ProfileConstraints] = Field(
        None,
        description="Job search constraints.",
    )
    risk_tolerance: RiskTolerance = Field(
        RiskTolerance.STANDARD,
        description="Risk tolerance for recommendations.",
    )

    @field_validator("display_name")
    @classmethod
    def strip_display_name(cls, v: Optional[str]) -> Optional[str]:
        """Strip leading/trailing whitespace from display_name."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class ProfileUpdateRequest(BaseModel):
    """Request body for partially updating a user profile (PATCH)."""

    display_name: Optional[str] = Field(None, min_length=1, max_length=100)
    current_occupation_id: Optional[str] = Field(None, max_length=10)
    skills_snapshot: Optional[List[SkillSnapshot]] = None
    constraints: Optional[ProfileConstraints] = None
    risk_tolerance: Optional[RiskTolerance] = None


class ProfileResponse(BaseModel):
    """Response body for profile endpoints."""

    user_id: int
    display_name: Optional[str] = None
    current_occupation_id: Optional[str] = None
    skills_snapshot: List[SkillSnapshot] = Field(default_factory=list)
    constraints: Optional[ProfileConstraints] = None
    risk_tolerance: RiskTolerance = RiskTolerance.STANDARD
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helper: pack / unpack metadata JSON
# ---------------------------------------------------------------------------

def _pack_profile_metadata(
    display_name: Optional[str] = None,
    current_occupation_id: Optional[str] = None,
    skills_snapshot: Optional[List[SkillSnapshot]] = None,
    constraints: Optional[ProfileConstraints] = None,
    risk_tolerance: RiskTolerance = RiskTolerance.STANDARD,
) -> Dict[str, Any]:
    """Serialise profile fields into the ``metadata_json`` column."""
    data: Dict[str, Any] = {
        "display_name": display_name,
        "current_occupation_id": current_occupation_id,
        "risk_tolerance": risk_tolerance.value,
    }
    if skills_snapshot is not None:
        data["skills_snapshot"] = [s.model_dump() for s in skills_snapshot]
    else:
        data["skills_snapshot"] = []
    if constraints is not None:
        data["constraints"] = constraints.model_dump()
    else:
        data["constraints"] = None
    return data


def _unpack_profile_response(row: UserProfileRow) -> ProfileResponse:
    """Build a ``ProfileResponse`` from a ``UserProfileRow``."""
    meta = row.metadata_json or {}

    skills_raw = meta.get("skills_snapshot", [])
    skills = [SkillSnapshot(**s) for s in skills_raw] if skills_raw else []

    constraints_raw = meta.get("constraints")
    constraints = ProfileConstraints(**constraints_raw) if constraints_raw else None

    risk_raw = meta.get("risk_tolerance", RiskTolerance.STANDARD.value)
    try:
        risk = RiskTolerance(risk_raw)
    except ValueError:
        risk = RiskTolerance.STANDARD

    return ProfileResponse(
        user_id=row.id,
        display_name=meta.get("display_name"),
        current_occupation_id=meta.get("current_occupation_id"),
        skills_snapshot=skills,
        constraints=constraints,
        risk_tolerance=risk,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ProfileResponse, status_code=201)
async def create_profile(
    request: ProfileCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Create a new user profile.

    Profile creation is **optional** -- the core recommendation flow does
    not require one.  A profile lets users store display preferences,
    constraints, and skill snapshots for a richer experience.
    """
    try:
        metadata = _pack_profile_metadata(
            display_name=request.display_name,
            current_occupation_id=request.current_occupation_id,
            skills_snapshot=request.skills_snapshot,
            constraints=request.constraints,
            risk_tolerance=request.risk_tolerance,
        )

        now = datetime.utcnow()
        user = UserProfileRow(
            created_at=now,
            updated_at=now,
            metadata_json=metadata,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        logger.info("Created profile for user %d", user.id)
        return _unpack_profile_response(user)

    except Exception as exc:
        logger.error("Error creating profile: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{user_id}", response_model=ProfileResponse)
async def get_profile(
    user_id: int,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Retrieve a user profile by ID."""
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return _unpack_profile_response(user)


@router.patch("/{user_id}", response_model=ProfileResponse)
async def update_profile(
    user_id: int,
    request: ProfileUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Partially update a user profile.

    Only fields present in the request body are modified; omitted fields
    retain their current values.
    """
    result = await db.execute(
        select(UserProfileRow).where(UserProfileRow.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    try:
        meta = user.metadata_json or {}

        if request.display_name is not None:
            meta["display_name"] = request.display_name.strip()
        if request.current_occupation_id is not None:
            meta["current_occupation_id"] = request.current_occupation_id
        if request.skills_snapshot is not None:
            meta["skills_snapshot"] = [s.model_dump() for s in request.skills_snapshot]
        if request.constraints is not None:
            meta["constraints"] = request.constraints.model_dump()
        if request.risk_tolerance is not None:
            meta["risk_tolerance"] = request.risk_tolerance.value

        user.metadata_json = meta
        user.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(user)

        logger.info("Updated profile for user %d", user_id)
        return _unpack_profile_response(user)

    except Exception as exc:
        logger.error("Error updating profile for user %d: %s", user_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
