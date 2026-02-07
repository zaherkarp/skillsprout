"""Personalized training path generator.

Given a set of skill gaps and user constraints, generates an ordered
sequence of training resources that respects:

    - Prerequisite ordering (foundational skills before advanced)
    - Budget limits (cumulative cost tracking)
    - Timeline limits (cumulative weeks)
    - Format and access constraints
    - Explicit handling of infeasible constraint combinations

The output is a training path: an ordered list of steps, each
containing a resource, the skills it addresses, and estimated
completion time. When constraints make a full path infeasible,
the generator returns a partial path with clear explanations.
"""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.features.training_paths.resource_filter import (
    UserConstraints,
    filter_resources,
    FilterResult,
)
from app.features.training_paths.training_catalog import (
    CostTier,
    DeliveryFormat,
    TrainingResource,
    get_catalog,
    SKILL_CODES,
)

logger = logging.getLogger(__name__)


# ==================== Prerequisite Graph ====================

# Skill codes that are prerequisites for other skills.
# Maps: skill_code -> list of prerequisite skill codes
# This encodes common educational dependencies.
PREREQUISITE_GRAPH: Dict[str, List[str]] = {
    # Programming prerequisites
    SKILL_CODES["programming"]: [
        SKILL_CODES["mathematics"],
        SKILL_CODES["critical_thinking"],
    ],
    SKILL_CODES["technology_design"]: [
        SKILL_CODES["critical_thinking"],
        SKILL_CODES["programming"],
    ],
    SKILL_CODES["systems_analysis"]: [
        SKILL_CODES["critical_thinking"],
        SKILL_CODES["reading_comprehension"],
    ],
    SKILL_CODES["systems_evaluation"]: [
        SKILL_CODES["systems_analysis"],
        SKILL_CODES["critical_thinking"],
    ],
    SKILL_CODES["operations_analysis"]: [
        SKILL_CODES["mathematics"],
        SKILL_CODES["critical_thinking"],
    ],
    # Management prerequisites
    SKILL_CODES["management_personnel"]: [
        SKILL_CODES["social_perceptiveness"],
        SKILL_CODES["coordination"],
    ],
    SKILL_CODES["management_financial"]: [
        SKILL_CODES["mathematics"],
        SKILL_CODES["judgment_decision_making"],
    ],
    # Communication prerequisites
    SKILL_CODES["persuasion"]: [
        SKILL_CODES["speaking"],
        SKILL_CODES["active_listening"],
    ],
    SKILL_CODES["negotiation"]: [
        SKILL_CODES["speaking"],
        SKILL_CODES["social_perceptiveness"],
    ],
    SKILL_CODES["instructing"]: [
        SKILL_CODES["speaking"],
        SKILL_CODES["active_listening"],
    ],
    # Technical prerequisites
    SKILL_CODES["repairing"]: [
        SKILL_CODES["troubleshooting"],
    ],
    SKILL_CODES["installation"]: [
        SKILL_CODES["equipment_selection"],
    ],
}


# ==================== Data Models ====================

class SkillGap(BaseModel):
    """A single skill gap to address.

    Attributes:
        skill_code: O*NET element ID.
        skill_name: Human-readable skill name.
        current_level: User's current capability (0.0-1.0).
        required_level: Required capability for target occupation.
        gap_weight: Importance weight of this gap.
        priority: Computed priority (higher = address first).
    """
    skill_code: str
    skill_name: str
    current_level: float = Field(0.0, ge=0.0, le=1.0)
    required_level: float = Field(1.0, ge=0.0, le=1.0)
    gap_weight: float = Field(1.0, ge=0.0)
    priority: float = 0.0


class TrainingStep(BaseModel):
    """A single step in a training path.

    Attributes:
        step_number: 1-based ordinal position in the path.
        resource: The training resource for this step.
        skills_addressed: Skill codes this step addresses.
        skill_names_addressed: Human-readable names for display.
        estimated_weeks: Duration of this step in weeks.
        estimated_cost_usd: Cost for this step.
        cumulative_weeks: Total weeks including all prior steps.
        cumulative_cost_usd: Total cost including all prior steps.
        rationale: Why this step is placed here.
        is_prerequisite: Whether this step is a prerequisite for later steps.
    """
    step_number: int
    resource: TrainingResource
    skills_addressed: List[str]
    skill_names_addressed: List[str]
    estimated_weeks: int
    estimated_cost_usd: float
    cumulative_weeks: int
    cumulative_cost_usd: float
    rationale: str
    is_prerequisite: bool = False


class InfeasibilityReason(BaseModel):
    """Explanation of why constraints make a full path infeasible.

    Attributes:
        constraint_name: Name of the binding constraint.
        detail: Human-readable explanation.
        suggestion: Actionable suggestion to resolve.
    """
    constraint_name: str
    detail: str
    suggestion: str


class TrainingPath(BaseModel):
    """A complete personalized training path.

    Attributes:
        steps: Ordered list of training steps.
        total_weeks: Total estimated duration.
        total_cost_usd: Total estimated cost.
        skills_covered: Set of skill codes addressed.
        skills_not_covered: Skill codes that could not be addressed.
        is_complete: Whether all skill gaps are covered.
        is_feasible: Whether the path fits within all constraints.
        infeasibility_reasons: Explanations when path is not feasible.
        warnings: General warnings about the path.
        suggestions: Actionable suggestions.
    """
    steps: List[TrainingStep] = Field(default_factory=list)
    total_weeks: int = 0
    total_cost_usd: float = 0.0
    skills_covered: List[str] = Field(default_factory=list)
    skills_not_covered: List[str] = Field(default_factory=list)
    is_complete: bool = False
    is_feasible: bool = True
    infeasibility_reasons: List[InfeasibilityReason] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


# ==================== Path Generator ====================

class PathGenerator:
    """Generates personalized training paths from skill gaps and constraints.

    The generator works in phases:
        1. Prioritize skill gaps using prerequisite ordering and weights
        2. Filter catalog by user constraints
        3. Greedily assign resources to gaps in priority order
        4. Track cumulative budget and timeline
        5. Report infeasibility when constraints bind

    Usage::

        generator = PathGenerator()
        path = generator.generate(
            skill_gaps=[
                SkillGap(skill_code="2.B.3.e", skill_name="Programming", gap_weight=0.8),
                SkillGap(skill_code="2.A.1.e", skill_name="Mathematics", gap_weight=0.5),
            ],
            constraints=UserConstraints(budget_usd=0, hours_per_week=10),
        )
    """

    def __init__(
        self,
        catalog: Optional[List[TrainingResource]] = None,
    ) -> None:
        """Initialize the path generator.

        Args:
            catalog: Override catalog for testing. Uses global catalog if None.
        """
        self._catalog = catalog

    @property
    def catalog(self) -> List[TrainingResource]:
        """Return the active training catalog."""
        if self._catalog is not None:
            return self._catalog
        return get_catalog()

    def generate(
        self,
        skill_gaps: List[SkillGap],
        constraints: UserConstraints,
    ) -> TrainingPath:
        """Generate a personalized training path.

        Args:
            skill_gaps: List of skill gaps to address, with priorities.
            constraints: User constraints for filtering and ordering.

        Returns:
            A ``TrainingPath`` with ordered steps, feasibility analysis,
            and suggestions.
        """
        if not skill_gaps:
            return TrainingPath(
                is_complete=True,
                is_feasible=True,
                warnings=["No skill gaps provided. No training needed."],
            )

        # Phase 1: Prioritize gaps using prerequisite ordering
        ordered_gaps = self._prioritize_gaps(skill_gaps)

        # Phase 2: Filter catalog by constraints
        filter_result = filter_resources(constraints, self.catalog)
        available_resources = filter_result.matching_resources

        # Phase 3: Build path greedily
        path = self._build_path(
            ordered_gaps=ordered_gaps,
            available_resources=available_resources,
            constraints=constraints,
            filter_result=filter_result,
        )

        return path

    def _prioritize_gaps(
        self,
        skill_gaps: List[SkillGap],
    ) -> List[SkillGap]:
        """Order skill gaps respecting prerequisites and weights.

        Gaps whose skill codes appear as prerequisites for other gaps
        are prioritized first. Within the same prerequisite level,
        gaps are ordered by weight (descending).

        Args:
            skill_gaps: Unordered skill gaps.

        Returns:
            Skill gaps in recommended learning order.
        """
        gap_codes = {g.skill_code for g in skill_gaps}
        gap_map = {g.skill_code: g for g in skill_gaps}

        # Compute priority: prerequisite depth + weight
        for gap in skill_gaps:
            prereq_boost = 0.0
            # Check if this gap's skill is a prerequisite for other gaps
            for other_code in gap_codes:
                if other_code == gap.skill_code:
                    continue
                prereqs = PREREQUISITE_GRAPH.get(other_code, [])
                if gap.skill_code in prereqs:
                    prereq_boost += 1.0

            gap.priority = prereq_boost * 10.0 + gap.gap_weight

        # Sort by priority descending (highest priority = learn first)
        return sorted(skill_gaps, key=lambda g: g.priority, reverse=True)

    def _build_path(
        self,
        ordered_gaps: List[SkillGap],
        available_resources: List[TrainingResource],
        constraints: UserConstraints,
        filter_result: FilterResult,
    ) -> TrainingPath:
        """Build a training path from ordered gaps and available resources.

        Args:
            ordered_gaps: Prioritized skill gaps.
            available_resources: Catalog resources passing constraint filters.
            constraints: User constraints for tracking limits.
            filter_result: Filter diagnostics for warnings.

        Returns:
            Constructed ``TrainingPath``.
        """
        path = TrainingPath()
        path.warnings.extend(filter_result.warnings)
        path.suggestions.extend(filter_result.suggestions)

        if not available_resources:
            path.is_feasible = False
            path.is_complete = False
            path.skills_not_covered = [g.skill_code for g in ordered_gaps]
            path.infeasibility_reasons.append(
                InfeasibilityReason(
                    constraint_name="no_resources",
                    detail="No training resources match your constraints.",
                    suggestion=(
                        "Try relaxing budget, timeline, or format constraints. "
                        "Visit your local American Job Center for in-person assistance."
                    ),
                )
            )
            return path

        cumulative_weeks = 0
        cumulative_cost = 0.0
        skills_covered: Set[str] = set()
        used_resource_ids: Set[str] = set()
        step_number = 0

        budget_limit = constraints.budget_usd
        timeline_limit = constraints.max_weeks

        for gap in ordered_gaps:
            if gap.skill_code in skills_covered:
                continue

            # Find best resource for this gap
            best_resource = self._select_resource(
                skill_code=gap.skill_code,
                available_resources=available_resources,
                used_ids=used_resource_ids,
                remaining_budget=(
                    budget_limit - cumulative_cost
                    if budget_limit is not None
                    else None
                ),
                remaining_weeks=(
                    timeline_limit - cumulative_weeks
                    if timeline_limit is not None
                    else None
                ),
            )

            if best_resource is None:
                path.skills_not_covered.append(gap.skill_code)
                continue

            # Check budget constraint
            new_cost = cumulative_cost + best_resource.estimated_cost_usd
            if budget_limit is not None and new_cost > budget_limit:
                path.skills_not_covered.append(gap.skill_code)
                path.infeasibility_reasons.append(
                    InfeasibilityReason(
                        constraint_name="budget",
                        detail=(
                            f"Cannot afford {best_resource.name} "
                            f"(${best_resource.estimated_cost_usd:.0f}) "
                            f"within remaining budget of "
                            f"${budget_limit - cumulative_cost:.0f}."
                        ),
                        suggestion=(
                            "Consider WIOA funding, Pell Grants, or "
                            "free alternatives like freeCodeCamp or Khan Academy."
                        ),
                    )
                )
                continue

            # Check timeline constraint
            new_weeks = cumulative_weeks + best_resource.total_weeks
            if timeline_limit is not None and new_weeks > timeline_limit:
                path.skills_not_covered.append(gap.skill_code)
                path.infeasibility_reasons.append(
                    InfeasibilityReason(
                        constraint_name="timeline",
                        detail=(
                            f"Cannot fit {best_resource.name} "
                            f"({best_resource.total_weeks} weeks) "
                            f"within remaining timeline of "
                            f"{timeline_limit - cumulative_weeks} weeks."
                        ),
                        suggestion=(
                            "Consider shorter programs, or extend your "
                            "training timeline. Some skills can be learned "
                            "concurrently."
                        ),
                    )
                )
                continue

            # Add step to path
            step_number += 1
            cumulative_cost = new_cost
            cumulative_weeks = new_weeks

            # Determine which of the user's gaps this resource addresses
            addressed_codes = [
                g.skill_code for g in ordered_gaps
                if g.skill_code in best_resource.skill_codes
                and g.skill_code not in skills_covered
            ]
            addressed_names = [
                g.skill_name for g in ordered_gaps
                if g.skill_code in addressed_codes
            ]

            skills_covered.update(addressed_codes)
            used_resource_ids.add(best_resource.id)

            # Determine if this step is a prerequisite for later steps
            is_prereq = any(
                gap.skill_code in PREREQUISITE_GRAPH.get(other_gap.skill_code, [])
                for other_gap in ordered_gaps
                if other_gap.skill_code not in skills_covered
                for gap_item in ordered_gaps
                if gap_item.skill_code in addressed_codes
            )

            rationale = self._build_rationale(
                gap=gap,
                resource=best_resource,
                is_prereq=is_prereq,
                step_number=step_number,
            )

            step = TrainingStep(
                step_number=step_number,
                resource=best_resource,
                skills_addressed=addressed_codes,
                skill_names_addressed=addressed_names,
                estimated_weeks=best_resource.total_weeks,
                estimated_cost_usd=best_resource.estimated_cost_usd,
                cumulative_weeks=cumulative_weeks,
                cumulative_cost_usd=cumulative_cost,
                rationale=rationale,
                is_prerequisite=is_prereq,
            )
            path.steps.append(step)

        # Finalize path
        path.total_weeks = cumulative_weeks
        path.total_cost_usd = cumulative_cost
        path.skills_covered = list(skills_covered)

        all_gap_codes = {g.skill_code for g in ordered_gaps}
        uncovered = all_gap_codes - skills_covered
        path.skills_not_covered = list(uncovered)
        path.is_complete = len(uncovered) == 0
        path.is_feasible = len(path.infeasibility_reasons) == 0

        if not path.is_complete and not path.infeasibility_reasons:
            path.warnings.append(
                f"{len(uncovered)} skill gap(s) could not be addressed "
                f"with available resources."
            )

        return path

    def _select_resource(
        self,
        skill_code: str,
        available_resources: List[TrainingResource],
        used_ids: Set[str],
        remaining_budget: Optional[float],
        remaining_weeks: Optional[int],
    ) -> Optional[TrainingResource]:
        """Select the best resource for a specific skill gap.

        Selection criteria (in priority order):
            1. Must address the target skill code
            2. Must not already be used in the path
            3. Must fit within remaining budget and timeline
            4. Prefer lower cost
            5. Prefer shorter duration
            6. Prefer resources that address multiple user gaps

        Args:
            skill_code: O*NET element ID to address.
            available_resources: Resources passing constraint filters.
            used_ids: Resource IDs already used in the path.
            remaining_budget: Remaining budget, or None.
            remaining_weeks: Remaining weeks, or None.

        Returns:
            Best matching resource, or None if nothing fits.
        """
        candidates = []

        for resource in available_resources:
            if resource.id in used_ids:
                continue
            if skill_code not in resource.skill_codes:
                continue
            if (
                remaining_budget is not None
                and resource.estimated_cost_usd > remaining_budget
            ):
                continue
            if (
                remaining_weeks is not None
                and resource.total_weeks > remaining_weeks
            ):
                continue
            candidates.append(resource)

        if not candidates:
            return None

        # Score candidates: lower cost + shorter duration = better
        def score_resource(r: TrainingResource) -> Tuple[float, int, int]:
            cost_score = r.estimated_cost_usd
            duration_score = r.total_weeks
            # Bonus for addressing multiple skills (negative = better)
            breadth = -len(r.skill_codes)
            return (cost_score, duration_score, breadth)

        candidates.sort(key=score_resource)
        return candidates[0]

    def _build_rationale(
        self,
        gap: SkillGap,
        resource: TrainingResource,
        is_prereq: bool,
        step_number: int,
    ) -> str:
        """Build a human-readable rationale for why this step is placed here.

        Args:
            gap: The primary skill gap being addressed.
            resource: The selected resource.
            is_prereq: Whether this is a prerequisite step.
            step_number: The step's ordinal position.

        Returns:
            Rationale string.
        """
        parts = []

        if step_number == 1:
            parts.append(f"Start with {resource.name}")
        else:
            parts.append(f"Continue with {resource.name}")

        if is_prereq:
            parts.append(
                f"to build foundational {gap.skill_name} skills "
                f"needed for later steps"
            )
        else:
            parts.append(f"to address your {gap.skill_name} skill gap")

        if resource.estimated_cost_usd == 0:
            parts.append("(free)")
        else:
            parts.append(f"(${resource.estimated_cost_usd:.0f})")

        return " ".join(parts) + "."
