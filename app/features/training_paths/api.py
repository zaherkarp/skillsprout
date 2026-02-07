"""FastAPI router for training path endpoints.

Provides two main endpoints:

    POST /api/v1/training-path/generate
        Generate a personalized training path from skill gaps and constraints.

    GET /api/v1/training-resources
        Browse and filter the training resource catalog.
"""
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.features.training_paths.path_generator import (
    PathGenerator,
    SkillGap,
    TrainingPath,
)
from app.features.training_paths.resource_filter import (
    UserConstraints,
    filter_resources,
)
from app.features.training_paths.training_catalog import (
    CostTier,
    DeliveryFormat,
    ResourceCategory,
    TrainingResource,
    get_catalog,
    get_catalog_stats,
    get_resource_by_id,
    get_resources_by_skill_name,
    SKILL_CODES,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["training-paths"])


# ==================== Request / Response Schemas ====================

class SkillGapInput(BaseModel):
    """Input schema for a single skill gap."""
    skill_code: str = Field(
        ..., description="O*NET element ID (e.g., '2.B.3.e' for Programming)",
    )
    skill_name: str = Field(..., description="Human-readable skill name")
    current_level: float = Field(
        0.0, ge=0.0, le=1.0, description="Current capability (0-1)",
    )
    required_level: float = Field(
        1.0, ge=0.0, le=1.0, description="Required capability (0-1)",
    )
    gap_weight: float = Field(
        1.0, ge=0.0, description="Importance weight of this gap",
    )


class TrainingPathRequest(BaseModel):
    """Request to generate a personalized training path.

    Combines skill gaps with user constraints to produce an ordered
    sequence of training resources.
    """
    skill_gaps: List[SkillGapInput] = Field(
        ..., min_length=1, description="Skill gaps to address",
    )
    budget_usd: Optional[float] = Field(
        None, ge=0.0, description="Maximum total budget in USD. None = no limit.",
    )
    hours_per_week: Optional[float] = Field(
        None, gt=0.0, le=80.0, description="Maximum hours per week for training.",
    )
    preferred_formats: List[str] = Field(
        default_factory=list,
        description="Acceptable delivery formats (e.g., 'online_self_paced', 'in_person').",
    )
    has_computer: bool = Field(True, description="User has a personal computer.")
    has_internet: bool = Field(True, description="User has reliable internet.")
    max_weeks: Optional[int] = Field(
        None, gt=0, description="Maximum training timeline in weeks.",
    )


class TrainingPathResponse(BaseModel):
    """Response containing the generated training path."""
    path: TrainingPath
    catalog_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics about the training catalog.",
    )


class TrainingResourceListResponse(BaseModel):
    """Response for catalog browsing."""
    resources: List[TrainingResource]
    total: int
    filters_applied: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ==================== Endpoints ====================

@router.post("/training-path/generate", response_model=TrainingPathResponse)
async def generate_training_path(
    request: TrainingPathRequest,
) -> TrainingPathResponse:
    """Generate a personalized training path based on skill gaps and constraints.

    Produces an ordered sequence of training resources that respects
    prerequisites, budget, timeline, and access constraints. When
    constraints make a full path infeasible, returns a partial path
    with explanations and suggestions.

    Args:
        request: Training path generation request with skill gaps and constraints.

    Returns:
        Training path with steps, feasibility analysis, and catalog stats.
    """
    # Convert input to domain models
    skill_gaps = [
        SkillGap(
            skill_code=gap.skill_code,
            skill_name=gap.skill_name,
            current_level=gap.current_level,
            required_level=gap.required_level,
            gap_weight=gap.gap_weight,
        )
        for gap in request.skill_gaps
    ]

    constraints = UserConstraints(
        budget_usd=request.budget_usd,
        hours_per_week=request.hours_per_week,
        preferred_formats=request.preferred_formats,
        has_computer=request.has_computer,
        has_internet=request.has_internet,
        max_weeks=request.max_weeks,
        target_skill_codes=[g.skill_code for g in request.skill_gaps],
    )

    generator = PathGenerator()
    path = generator.generate(skill_gaps=skill_gaps, constraints=constraints)

    return TrainingPathResponse(
        path=path,
        catalog_stats=get_catalog_stats(),
    )


@router.get("/training-resources", response_model=TrainingResourceListResponse)
async def list_training_resources(
    skill: Optional[str] = Query(
        None,
        description="Filter by skill name (case-insensitive, e.g., 'programming').",
    ),
    skill_code: Optional[str] = Query(
        None,
        description="Filter by O*NET skill code (e.g., '2.B.3.e').",
    ),
    cost_tier: Optional[str] = Query(
        None,
        description="Filter by cost tier: free, low, moderate, high.",
    ),
    category: Optional[str] = Query(
        None,
        description="Filter by category (e.g., 'free_certificate', 'bootcamp').",
    ),
    format: Optional[str] = Query(
        None,
        description="Filter by delivery format (e.g., 'online_self_paced').",
    ),
    has_computer: Optional[bool] = Query(
        None,
        description="Filter by computer requirement. Set to false for no-computer resources.",
    ),
    max_cost: Optional[float] = Query(
        None, ge=0.0, description="Maximum cost in USD.",
    ),
    max_weeks: Optional[int] = Query(
        None, gt=0, description="Maximum duration in weeks.",
    ),
) -> TrainingResourceListResponse:
    """Browse and filter the training resource catalog.

    Returns training resources matching the specified filters. All filters
    are optional and combined with AND logic.

    Args:
        skill: Filter by skill name.
        skill_code: Filter by O*NET skill code.
        cost_tier: Filter by cost tier.
        category: Filter by resource category.
        format: Filter by delivery format.
        has_computer: Filter by computer requirement.
        max_cost: Maximum cost filter.
        max_weeks: Maximum duration filter.

    Returns:
        Matching resources with filter metadata.
    """
    filters_applied: List[str] = []
    warnings: List[str] = []

    # Build constraints for the filter
    constraints = UserConstraints(
        has_computer=has_computer if has_computer is not None else True,
        has_internet=True,
    )

    if skill:
        constraints.target_skill_names = [skill]
        filters_applied.append(f"skill={skill}")

    if skill_code:
        constraints.target_skill_codes = [skill_code]
        filters_applied.append(f"skill_code={skill_code}")

    if max_cost is not None:
        constraints.budget_usd = max_cost
        filters_applied.append(f"max_cost=${max_cost:.0f}")

    if max_weeks is not None:
        constraints.max_weeks = max_weeks
        filters_applied.append(f"max_weeks={max_weeks}")

    if has_computer is not None and not has_computer:
        filters_applied.append("no_computer")

    # Apply constraint-based filtering
    result = filter_resources(constraints)
    resources = result.matching_resources
    warnings.extend(result.warnings)

    # Apply additional filters not covered by UserConstraints
    if cost_tier:
        valid_tiers = {t.value for t in CostTier}
        if cost_tier not in valid_tiers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cost_tier. Must be one of: {valid_tiers}",
            )
        resources = [r for r in resources if r.cost_tier == cost_tier]
        filters_applied.append(f"cost_tier={cost_tier}")

    if category:
        valid_categories = {c.value for c in ResourceCategory}
        if category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category. Must be one of: {valid_categories}",
            )
        resources = [r for r in resources if r.category == category]
        filters_applied.append(f"category={category}")

    if format:
        valid_formats = {f.value for f in DeliveryFormat}
        if format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid format. Must be one of: {valid_formats}",
            )
        resources = [r for r in resources if r.delivery_format == format]
        filters_applied.append(f"format={format}")

    return TrainingResourceListResponse(
        resources=resources,
        total=len(resources),
        filters_applied=filters_applied,
        warnings=warnings,
    )
