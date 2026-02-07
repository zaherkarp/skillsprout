"""FastAPI router for the Skills Translator feature.

Provides a REST endpoint that accepts free-text descriptions of work
experience and returns matched O*NET skills with confidence levels.

Endpoints
---------
POST /api/v1/skills/translate
    Translate a plain-language description into O*NET skills.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.features.skills_translator.skills_translator import (
    ConfidenceLevel,
    MatchedSkill,
    SkillsTranslator,
    get_translator,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SkillsTranslateRequest(BaseModel):
    """Request body for the skills translation endpoint.

    Attributes:
        description: Free-text description of work experience.
        confirm: Optional list of O*NET element IDs the user has
            previously confirmed as accurate.
    """

    description: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Free-text description of work experience.",
        json_schema_extra={"examples": [
            "I managed a retail store for five years, trained new employees, "
            "handled customer complaints, and tracked inventory."
        ]},
    )
    confirm: Optional[List[str]] = Field(
        default=None,
        description=(
            "O*NET element IDs the user confirms as accurate matches. "
            "These are promoted to HIGH confidence."
        ),
        json_schema_extra={"examples": [["2.B.7.e", "2.B.6.b"]]},
    )


class MatchedSkillResponse(BaseModel):
    """A single matched skill in the response."""

    element_id: str = Field(
        ..., description="O*NET element ID (e.g. '2.B.1.a')."
    )
    skill_name: str = Field(
        ..., description="Canonical O*NET skill name."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1."
    )
    confidence_level: str = Field(
        ..., description="Categorical confidence tier: HIGH, MEDIUM, or LOW."
    )
    source: str = Field(
        ..., description="Match source: 'rule' (dictionary) or 'tfidf' (semantic)."
    )
    matched_phrase: str = Field(
        default="",
        description="The phrase or indicator that triggered this match.",
    )


class SkillsTranslateResponse(BaseModel):
    """Response body for the skills translation endpoint.

    Attributes:
        matched_skills: Skills matched with MEDIUM or HIGH confidence.
        needs_confirmation: Skills matched with LOW confidence that the
            user should review.
    """

    matched_skills: List[MatchedSkillResponse] = Field(
        default_factory=list,
        description="Skills matched with HIGH or MEDIUM confidence.",
    )
    needs_confirmation: List[MatchedSkillResponse] = Field(
        default_factory=list,
        description="Skills matched with LOW confidence; user should confirm.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(match: MatchedSkill) -> MatchedSkillResponse:
    """Convert an internal ``MatchedSkill`` to an API response model.

    Args:
        match: Internal match dataclass.

    Returns:
        Pydantic response model.
    """
    return MatchedSkillResponse(
        element_id=match.element_id,
        skill_name=match.skill_name,
        confidence=round(match.confidence, 4),
        confidence_level=match.confidence_level.value,
        source=match.source,
        matched_phrase=match.matched_phrase,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/skills/translate",
    response_model=SkillsTranslateResponse,
    summary="Translate work experience into O*NET skills",
    description=(
        "Accepts a free-text description of work experience and returns "
        "matched O*NET skills with confidence levels.  Skills the user "
        "confirms via the `confirm` field are promoted to HIGH confidence."
    ),
    tags=["skills-translator"],
)
async def translate_skills(
    request: SkillsTranslateRequest,
) -> SkillsTranslateResponse:
    """Translate free-text work experience into O*NET skills.

    Args:
        request: The translation request containing the description and
            optional confirmed skill IDs.

    Returns:
        A response with matched skills and skills needing confirmation.

    Raises:
        HTTPException: 400 if the description is empty.
        HTTPException: 500 on unexpected internal errors.
    """
    try:
        translator = get_translator()
        result = translator.translate(
            text=request.description,
            confirmed_skill_ids=request.confirm,
        )
    except ValueError as exc:
        logger.warning("Invalid translation request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Skills translation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during skills translation.",
        )

    return SkillsTranslateResponse(
        matched_skills=[_to_response(m) for m in result.matched_skills],
        needs_confirmation=[_to_response(m) for m in result.needs_confirmation],
    )
