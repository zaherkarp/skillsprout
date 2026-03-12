"""AI exposure scoring aligned with Anthropic's labor market impact methodology.

Based on the paper "Labor market impacts of AI: A new measure and early evidence"
(Massenkoff & McCrory, 2026), this module implements an observed exposure metric
that distinguishes between theoretical AI capability and real-world AI adoption
for O*NET occupations.

Key concepts from the paper:
- Theoretical exposure (β): whether an LLM could speed up a task (Eloundou et al.)
- Observed exposure: actual AI usage mapped to O*NET tasks from Claude usage data
- Coverage: fraction of an occupation's tasks where AI is actively used
- Weighting: automated uses get full weight, augmentative uses get half weight

The paper found that computer programmers (75% coverage), customer service reps,
data entry keyers (67%), and financial analysts have the highest observed exposure,
while ~30% of workers (cooks, mechanics, lifeguards, bartenders) show zero exposure.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AIUsageType(Enum):
    """Type of AI usage for a task, per the paper's weighting scheme."""
    AUTOMATED = "automated"       # Full weight: AI performs the task independently
    AUGMENTATIVE = "augmentative"  # Half weight: AI assists a human doing the task
    NONE = "none"                 # No observed AI usage for this task


@dataclass
class TaskExposure:
    """AI exposure assessment for a single O*NET task.

    Attributes:
        task_description: Brief description of the task.
        theoretical_beta: Eloundou et al. β score (0, 0.5, or 1.0).
        usage_type: Whether AI usage is automated, augmentative, or none.
        effective_weight: Computed weight (1.0 for automated, 0.5 for augmentative, 0 for none).
        time_share: Fraction of work time spent on this task (0-1).
    """
    task_description: str
    theoretical_beta: float
    usage_type: AIUsageType
    effective_weight: float = 0.0
    time_share: float = 0.0

    def __post_init__(self):
        if self.effective_weight == 0.0:
            self.effective_weight = self._compute_weight()

    def _compute_weight(self) -> float:
        if self.usage_type == AIUsageType.AUTOMATED:
            return 1.0
        elif self.usage_type == AIUsageType.AUGMENTATIVE:
            return 0.5
        return 0.0


@dataclass
class OccupationExposure:
    """Full AI exposure profile for an occupation.

    Attributes:
        onet_code: O*NET-SOC code.
        title: Occupation title.
        soc_sector: Broad SOC major group name.
        theoretical_coverage: Fraction of tasks theoretically feasible for AI (β >= 0.5).
        observed_coverage: Weighted fraction of tasks with actual AI usage.
        task_exposures: Individual task-level exposure assessments.
        exposure_tier: Classification: high, moderate, low, or zero.
        bls_growth_adjustment: Estimated BLS growth projection adjustment
            (-0.6pp per 10pp coverage increase, per the paper).
    """
    onet_code: str
    title: str
    soc_sector: str
    theoretical_coverage: float
    observed_coverage: float
    task_exposures: List[TaskExposure] = field(default_factory=list)
    exposure_tier: str = ""
    bls_growth_adjustment: float = 0.0

    def __post_init__(self):
        if not self.exposure_tier:
            self.exposure_tier = self._classify_tier()
        if self.bls_growth_adjustment == 0.0:
            self.bls_growth_adjustment = self._estimate_bls_adjustment()

    def _classify_tier(self) -> str:
        if self.observed_coverage >= 0.50:
            return "high"
        elif self.observed_coverage >= 0.20:
            return "moderate"
        elif self.observed_coverage > 0.0:
            return "low"
        return "zero"

    def _estimate_bls_adjustment(self) -> float:
        """Per the paper: -0.6pp per 10pp increase in coverage."""
        return round(-0.6 * (self.observed_coverage * 100 / 10), 2)


def compute_observed_coverage(task_exposures: List[TaskExposure]) -> float:
    """Compute observed coverage from task-level exposures.

    Coverage = sum(time_share * effective_weight) for all tasks.
    If time_share is not set, uses equal weighting across tasks.

    Args:
        task_exposures: List of task-level exposure assessments.

    Returns:
        Observed coverage as a fraction (0-1).
    """
    if not task_exposures:
        return 0.0

    has_time_shares = any(t.time_share > 0 for t in task_exposures)

    if has_time_shares:
        total_time = sum(t.time_share for t in task_exposures)
        if total_time == 0:
            return 0.0
        return sum(t.time_share * t.effective_weight for t in task_exposures) / total_time
    else:
        n = len(task_exposures)
        return sum(t.effective_weight for t in task_exposures) / n


def compute_theoretical_coverage(task_exposures: List[TaskExposure]) -> float:
    """Compute theoretical coverage from task-level β scores.

    Theoretical coverage = fraction of tasks where β >= 0.5.

    Args:
        task_exposures: List of task-level exposure assessments.

    Returns:
        Theoretical coverage as a fraction (0-1).
    """
    if not task_exposures:
        return 0.0

    feasible = sum(1 for t in task_exposures if t.theoretical_beta >= 0.5)
    return feasible / len(task_exposures)


# ============================================================================
# Reference data: AI exposure profiles for key occupations from the paper
# ============================================================================

# Occupations identified as HIGHEST observed exposure by the paper
HIGH_EXPOSURE_OCCUPATIONS: Dict[str, OccupationExposure] = {
    "15-1251.00": OccupationExposure(
        onet_code="15-1251.00",
        title="Computer Programmers",
        soc_sector="Computer and Mathematical",
        theoretical_coverage=0.94,
        observed_coverage=0.75,
    ),
    "43-4051.00": OccupationExposure(
        onet_code="43-4051.00",
        title="Customer Service Representatives",
        soc_sector="Office and Administrative Support",
        theoretical_coverage=0.90,
        observed_coverage=0.60,
    ),
    "43-9021.00": OccupationExposure(
        onet_code="43-9021.00",
        title="Data Entry Keyers",
        soc_sector="Office and Administrative Support",
        theoretical_coverage=0.92,
        observed_coverage=0.67,
    ),
    "13-2051.00": OccupationExposure(
        onet_code="13-2051.00",
        title="Financial Analysts",
        soc_sector="Business and Financial Operations",
        theoretical_coverage=0.88,
        observed_coverage=0.55,
    ),
    "15-1252.00": OccupationExposure(
        onet_code="15-1252.00",
        title="Software Developers",
        soc_sector="Computer and Mathematical",
        theoretical_coverage=0.94,
        observed_coverage=0.33,
    ),
}

# Occupations identified as ZERO observed exposure by the paper
ZERO_EXPOSURE_OCCUPATIONS: Dict[str, OccupationExposure] = {
    "35-2014.00": OccupationExposure(
        onet_code="35-2014.00",
        title="Cooks, Restaurant",
        soc_sector="Food Preparation and Serving",
        theoretical_coverage=0.10,
        observed_coverage=0.0,
    ),
    "49-3023.00": OccupationExposure(
        onet_code="49-3023.00",
        title="Automotive Service Technicians and Mechanics",
        soc_sector="Installation, Maintenance, and Repair",
        theoretical_coverage=0.15,
        observed_coverage=0.0,
    ),
    "33-9092.00": OccupationExposure(
        onet_code="33-9092.00",
        title="Lifeguards, Ski Patrol, and Other Recreational Protective Service Workers",
        soc_sector="Protective Service",
        theoretical_coverage=0.08,
        observed_coverage=0.0,
    ),
    "35-3011.00": OccupationExposure(
        onet_code="35-3011.00",
        title="Bartenders",
        soc_sector="Food Preparation and Serving",
        theoretical_coverage=0.12,
        observed_coverage=0.0,
    ),
    "35-9021.00": OccupationExposure(
        onet_code="35-9021.00",
        title="Dishwashers",
        soc_sector="Food Preparation and Serving",
        theoretical_coverage=0.05,
        observed_coverage=0.0,
    ),
}

# Sector-level theoretical vs observed coverage (from the paper's findings)
SECTOR_COVERAGE: Dict[str, Dict[str, float]] = {
    "Computer and Mathematical": {
        "theoretical": 0.94,
        "observed": 0.33,
    },
    "Office and Administrative Support": {
        "theoretical": 0.90,
        "observed": 0.20,
    },
    "Business and Financial Operations": {
        "theoretical": 0.88,
        "observed": 0.25,
    },
    "Food Preparation and Serving": {
        "theoretical": 0.10,
        "observed": 0.0,
    },
    "Installation, Maintenance, and Repair": {
        "theoretical": 0.15,
        "observed": 0.0,
    },
    "Protective Service": {
        "theoretical": 0.08,
        "observed": 0.0,
    },
}


def get_exposure_profile(onet_code: str) -> Optional[OccupationExposure]:
    """Look up the AI exposure profile for an occupation.

    Args:
        onet_code: O*NET-SOC code.

    Returns:
        OccupationExposure if available, else None.
    """
    if onet_code in HIGH_EXPOSURE_OCCUPATIONS:
        return HIGH_EXPOSURE_OCCUPATIONS[onet_code]
    if onet_code in ZERO_EXPOSURE_OCCUPATIONS:
        return ZERO_EXPOSURE_OCCUPATIONS[onet_code]
    return None


def get_sector_coverage(sector_name: str) -> Optional[Dict[str, float]]:
    """Look up sector-level coverage data.

    Args:
        sector_name: SOC major group name.

    Returns:
        Dict with 'theoretical' and 'observed' keys, or None.
    """
    return SECTOR_COVERAGE.get(sector_name)


def theoretical_observed_gap(onet_code: str) -> Optional[float]:
    """Compute the gap between theoretical and observed coverage.

    The paper's key finding is that actual adoption is far below theoretical
    capability. This function quantifies that gap for a given occupation.

    Args:
        onet_code: O*NET-SOC code.

    Returns:
        Gap as a fraction (theoretical - observed), or None if not found.
    """
    profile = get_exposure_profile(onet_code)
    if profile is None:
        return None
    return round(profile.theoretical_coverage - profile.observed_coverage, 4)
