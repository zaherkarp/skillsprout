"""Constraint-aware resource filtering.

Filters the training catalog based on user constraints such as budget,
available hours per week, computer access, internet access, preferred
format, and timeline. Handles edge cases including the no-computer
scenario by recommending library and community resources.
"""
import logging
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from app.features.training_paths.training_catalog import (
    CostTier,
    DeliveryFormat,
    TrainingResource,
    get_catalog,
    get_no_computer_resources,
)

logger = logging.getLogger(__name__)


# ==================== Constraint Model ====================

class UserConstraints(BaseModel):
    """User constraints for filtering training resources.

    Attributes:
        budget_usd: Maximum total budget in USD. Use 0 for free-only.
        hours_per_week: Maximum hours per week available for training.
        preferred_formats: Acceptable delivery formats. Empty means all.
        has_computer: Whether the user has access to a personal computer.
        has_internet: Whether the user has reliable internet access.
        max_weeks: Maximum acceptable program duration in weeks.
        target_skill_codes: O*NET skill codes to address (empty = all).
        target_skill_names: Human-readable skill names to match (empty = all).
        exclude_categories: Resource categories to exclude.
        geographic_scope: Required geographic availability.
    """
    budget_usd: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum total budget in USD. None means no budget constraint.",
    )
    hours_per_week: Optional[float] = Field(
        None,
        gt=0.0,
        le=80.0,
        description="Maximum hours per week available. None means no constraint.",
    )
    preferred_formats: List[str] = Field(
        default_factory=list,
        description="Acceptable delivery formats. Empty means all formats.",
    )
    has_computer: bool = Field(
        True,
        description="Whether the user has access to a personal computer.",
    )
    has_internet: bool = Field(
        True,
        description="Whether the user has reliable internet access.",
    )
    max_weeks: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum program duration in weeks. None means no constraint.",
    )
    target_skill_codes: List[str] = Field(
        default_factory=list,
        description="O*NET skill codes the user needs. Empty means any.",
    )
    target_skill_names: List[str] = Field(
        default_factory=list,
        description="Skill names to match (case-insensitive). Empty means any.",
    )
    exclude_categories: List[str] = Field(
        default_factory=list,
        description="Resource categories to exclude.",
    )
    geographic_scope: Optional[str] = Field(
        None,
        description="Required geographic scope (e.g., 'national', 'state').",
    )


class FilterResult(BaseModel):
    """Result of filtering the training catalog.

    Includes both the matching resources and diagnostic information
    about why resources were excluded.
    """
    matching_resources: List[TrainingResource]
    total_catalog_size: int
    filters_applied: List[str]
    excluded_count: int
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# ==================== Filter Functions ====================

def filter_by_budget(
    resources: List[TrainingResource],
    budget_usd: float,
) -> List[TrainingResource]:
    """Filter resources by maximum budget.

    Args:
        resources: Resources to filter.
        budget_usd: Maximum total cost in USD.

    Returns:
        Resources within budget.
    """
    return [r for r in resources if r.estimated_cost_usd <= budget_usd]


def filter_by_hours(
    resources: List[TrainingResource],
    max_hours_per_week: float,
) -> List[TrainingResource]:
    """Filter resources by maximum weekly time commitment.

    Args:
        resources: Resources to filter.
        max_hours_per_week: Maximum acceptable hours per week.

    Returns:
        Resources requiring at most ``max_hours_per_week``.
    """
    return [r for r in resources if r.hours_per_week <= max_hours_per_week]


def filter_by_format(
    resources: List[TrainingResource],
    formats: List[str],
) -> List[TrainingResource]:
    """Filter resources by acceptable delivery formats.

    Args:
        resources: Resources to filter.
        formats: List of acceptable format strings.

    Returns:
        Resources matching at least one format.
    """
    format_set = set(formats)
    return [r for r in resources if r.delivery_format in format_set]


def filter_by_computer_access(
    resources: List[TrainingResource],
    has_computer: bool,
) -> List[TrainingResource]:
    """Filter resources based on computer access.

    When the user has no computer, only resources that do not require
    a computer are returned. This includes in-person programs, library
    programs (which provide computer access), and offline materials.

    Args:
        resources: Resources to filter.
        has_computer: Whether the user has a personal computer.

    Returns:
        Accessible resources.
    """
    if has_computer:
        return resources
    return [r for r in resources if not r.requires_computer]


def filter_by_internet_access(
    resources: List[TrainingResource],
    has_internet: bool,
) -> List[TrainingResource]:
    """Filter resources based on internet access.

    Args:
        resources: Resources to filter.
        has_internet: Whether the user has reliable internet.

    Returns:
        Accessible resources.
    """
    if has_internet:
        return resources
    return [r for r in resources if not r.requires_internet]


def filter_by_duration(
    resources: List[TrainingResource],
    max_weeks: int,
) -> List[TrainingResource]:
    """Filter resources by maximum program duration.

    Args:
        resources: Resources to filter.
        max_weeks: Maximum acceptable duration in weeks.

    Returns:
        Resources completable within the timeline.
    """
    return [r for r in resources if r.total_weeks <= max_weeks]


def filter_by_skill_codes(
    resources: List[TrainingResource],
    skill_codes: List[str],
) -> List[TrainingResource]:
    """Filter resources that address at least one of the target skill codes.

    Args:
        resources: Resources to filter.
        skill_codes: O*NET element IDs the user needs to develop.

    Returns:
        Resources addressing at least one target skill.
    """
    code_set = set(skill_codes)
    return [
        r for r in resources
        if code_set.intersection(r.skill_codes)
    ]


def filter_by_skill_names(
    resources: List[TrainingResource],
    skill_names: List[str],
) -> List[TrainingResource]:
    """Filter resources matching skill names (case-insensitive).

    Args:
        resources: Resources to filter.
        skill_names: Human-readable skill names to match.

    Returns:
        Resources addressing at least one named skill.
    """
    name_set = {name.lower() for name in skill_names}
    return [
        r for r in resources
        if any(sn.lower() in name_set for sn in r.skill_names)
    ]


def filter_by_category_exclusion(
    resources: List[TrainingResource],
    exclude_categories: List[str],
) -> List[TrainingResource]:
    """Exclude resources in specified categories.

    Args:
        resources: Resources to filter.
        exclude_categories: Category strings to exclude.

    Returns:
        Resources not in excluded categories.
    """
    excluded_set = set(exclude_categories)
    return [r for r in resources if r.category not in excluded_set]


# ==================== Composite Filter ====================

def filter_resources(
    constraints: UserConstraints,
    catalog: Optional[List[TrainingResource]] = None,
) -> FilterResult:
    """Apply all user constraints to filter the training catalog.

    Applies filters in a specific order designed to fail gracefully:
    hard constraints (computer, internet) are applied first, then
    soft constraints (budget, hours, duration). If constraints are
    too restrictive, warnings and suggestions are provided.

    Args:
        constraints: User constraints to apply.
        catalog: Override catalog for testing. Uses global catalog if None.

    Returns:
        ``FilterResult`` with matching resources and diagnostics.
    """
    if catalog is None:
        catalog = get_catalog()

    total_size = len(catalog)
    filtered = list(catalog)
    filters_applied: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []

    # ---- Hard constraints: access ----

    if not constraints.has_computer:
        filtered = filter_by_computer_access(filtered, False)
        filters_applied.append("no_computer")
        suggestions.append(
            "Visit your local public library for free computer access "
            "and digital literacy programs."
        )

    if not constraints.has_internet:
        filtered = filter_by_internet_access(filtered, False)
        filters_applied.append("no_internet")
        suggestions.append(
            "Many libraries and community centers offer free Wi-Fi. "
            "Ask about offline learning materials."
        )

    # ---- Skill targeting ----

    if constraints.target_skill_codes:
        filtered = filter_by_skill_codes(filtered, constraints.target_skill_codes)
        filters_applied.append(f"skill_codes({len(constraints.target_skill_codes)})")

    if constraints.target_skill_names:
        filtered = filter_by_skill_names(filtered, constraints.target_skill_names)
        filters_applied.append(f"skill_names({len(constraints.target_skill_names)})")

    # ---- Category exclusions ----

    if constraints.exclude_categories:
        filtered = filter_by_category_exclusion(
            filtered, constraints.exclude_categories,
        )
        filters_applied.append(f"exclude_categories({len(constraints.exclude_categories)})")

    # ---- Soft constraints ----

    pre_soft_count = len(filtered)

    if constraints.budget_usd is not None:
        budget_filtered = filter_by_budget(filtered, constraints.budget_usd)
        if not budget_filtered and filtered:
            warnings.append(
                f"No resources found within ${constraints.budget_usd:.0f} budget. "
                f"Consider WIOA funding, Pell Grants, or library resources."
            )
            suggestions.append(
                "Apply at your local American Job Center for WIOA training funding."
            )
            # Keep free resources as fallback
            budget_filtered = filter_by_budget(filtered, 0.0)
        filtered = budget_filtered
        filters_applied.append(f"budget(${constraints.budget_usd:.0f})")

    if constraints.hours_per_week is not None:
        hours_filtered = filter_by_hours(filtered, constraints.hours_per_week)
        if not hours_filtered and filtered:
            warnings.append(
                f"No resources fit within {constraints.hours_per_week:.0f} hours/week. "
                f"Consider self-paced options that allow flexible scheduling."
            )
        else:
            filtered = hours_filtered
        filters_applied.append(f"hours_per_week({constraints.hours_per_week:.0f})")

    if constraints.preferred_formats:
        format_filtered = filter_by_format(filtered, constraints.preferred_formats)
        if not format_filtered and filtered:
            warnings.append(
                "No resources match your preferred format. "
                "Showing results in all available formats."
            )
        else:
            filtered = format_filtered
        filters_applied.append(f"formats({','.join(constraints.preferred_formats)})")

    if constraints.max_weeks is not None:
        duration_filtered = filter_by_duration(filtered, constraints.max_weeks)
        if not duration_filtered and filtered:
            warnings.append(
                f"No resources completable within {constraints.max_weeks} weeks. "
                f"Showing shortest available options."
            )
            # Fallback: sort by duration and return shortest
            filtered.sort(key=lambda r: r.total_weeks)
        else:
            filtered = duration_filtered
        filters_applied.append(f"max_weeks({constraints.max_weeks})")

    # ---- Final diagnostics ----

    if not filtered:
        warnings.append(
            "No resources match all your constraints. "
            "Try relaxing budget, timeline, or format requirements."
        )
        suggestions.append(
            "Contact your local American Job Center for personalized "
            "training guidance: https://www.careeronestop.org/LocalHelp/"
        )

    excluded_count = total_size - len(filtered)

    return FilterResult(
        matching_resources=filtered,
        total_catalog_size=total_size,
        filters_applied=filters_applied,
        excluded_count=excluded_count,
        warnings=warnings,
        suggestions=suggestions,
    )
