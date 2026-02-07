"""Externalized threshold configuration for bucket assignment and scoring.

Design rationale:
-----------------
The baseline scorer (app.ml.scoring.BaselineScorer) hard-codes thresholds from
app.core.config.settings. This is adequate for a single deployment, but fails
when we need to:

1. Let users choose their own risk tolerance ("I want aggressive matches" vs.
   "Only show me sure things").
2. Run A/B experiments with different threshold regimes.
3. Weight skill domains differently (e.g. penalise missing safety credentials
   more heavily than missing soft skills).

This module externalises ALL bucket logic into composable, serialisable
dataclasses so that thresholds can be swapped at runtime without touching
scorer internals.

Presets:
  RELAXED  - Favours more READY_NOW classifications. Useful for users who
             are comfortable with risk and want the widest opportunity set.
  STANDARD - Matches the production defaults from config.py.
  STRICT   - Conservative. Requires very high match and very low gaps for
             READY_NOW. Appropriate for regulated professions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from app.core.config import settings


# ---------------------------------------------------------------------------
# Skill domain taxonomy
# ---------------------------------------------------------------------------
# O*NET element_id prefixes map to broad skill domains. We assign weights so
# that missing a safety-critical skill is penalised more than missing a
# nice-to-have soft skill.
# ---------------------------------------------------------------------------

class SkillDomain(str, Enum):
    """Broad skill domain categories derived from O*NET element_id prefixes.

    The O*NET Content Model organises elements hierarchically:
      2.A = Abilities, 2.B = Skills (occupational), 2.C = Knowledge,
      4.A = Work Activities, etc.

    We collapse these into actionable domains for weighting purposes.
    """
    COGNITIVE = "cognitive"          # 2.A.1 - Cognitive abilities
    PSYCHOMOTOR = "psychomotor"      # 2.A.2 - Psychomotor abilities
    PHYSICAL = "physical"            # 2.A.3 - Physical abilities
    SENSORY = "sensory"              # 2.A.4 - Sensory abilities
    BASIC_SKILLS = "basic_skills"    # 2.B.1 - Content, Process
    SOCIAL_SKILLS = "social_skills"  # 2.B.4 - Social skills
    TECHNICAL = "technical"          # 2.B.3 - Technical skills
    COMPLEX_PROBLEM = "complex_problem"  # 2.B.2 - Complex problem solving
    SYSTEMS = "systems"              # 2.B.5 - Systems skills
    RESOURCE_MGMT = "resource_mgmt"  # 2.B.6 - Resource management
    KNOWLEDGE = "knowledge"          # 2.C - Knowledge domains
    WORK_ACTIVITY = "work_activity"  # 4.A - Work activities
    OTHER = "other"                  # Everything else


# Mapping from O*NET element_id prefix to domain.  Longest prefix wins.
ELEMENT_PREFIX_TO_DOMAIN: Dict[str, SkillDomain] = {
    "2.A.1": SkillDomain.COGNITIVE,
    "2.A.2": SkillDomain.PSYCHOMOTOR,
    "2.A.3": SkillDomain.PHYSICAL,
    "2.A.4": SkillDomain.SENSORY,
    "2.B.1": SkillDomain.BASIC_SKILLS,
    "2.B.2": SkillDomain.COMPLEX_PROBLEM,
    "2.B.3": SkillDomain.TECHNICAL,
    "2.B.4": SkillDomain.SOCIAL_SKILLS,
    "2.B.5": SkillDomain.SYSTEMS,
    "2.B.6": SkillDomain.RESOURCE_MGMT,
    "2.C":   SkillDomain.KNOWLEDGE,
    "4.A":   SkillDomain.WORK_ACTIVITY,
}


def classify_skill_domain(element_id: str) -> SkillDomain:
    """Map an O*NET element_id to its broad skill domain.

    Uses longest-prefix matching so that '2.B.3.a' matches TECHNICAL (prefix
    '2.B.3') rather than falling through to a shorter prefix.

    Args:
        element_id: O*NET element identifier, e.g. '2.B.1.a'.

    Returns:
        The matching SkillDomain, or OTHER if no prefix matches.
    """
    # Sort by prefix length descending so longest match wins.
    for prefix in sorted(ELEMENT_PREFIX_TO_DOMAIN, key=len, reverse=True):
        if element_id.startswith(prefix):
            return ELEMENT_PREFIX_TO_DOMAIN[prefix]
    return SkillDomain.OTHER


# ---------------------------------------------------------------------------
# Bucket threshold configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BucketThresholds:
    """Numeric thresholds that control bucket assignment.

    The baseline scorer assigns buckets using two axes:
      - match_score  (0-100): how well user skills cover occupation requirements
      - gap_severity (0-100): weighted mass of skill deficits

    READY_NOW requires match >= ready_now_match_min AND gap <= ready_now_gap_max.
    TRAINABLE requires match in [trainable_match_min, trainable_match_max]
      OR gap in [trainable_gap_min, trainable_gap_max].
    Everything else falls to LONG_RESKILL.

    Attributes:
        ready_now_match_min: Minimum match_score for READY_NOW.
        ready_now_gap_max:   Maximum gap_severity for READY_NOW.
        trainable_match_min: Lower bound of match_score for TRAINABLE.
        trainable_match_max: Upper bound of match_score for TRAINABLE.
        trainable_gap_min:   Lower bound of gap_severity for TRAINABLE.
        trainable_gap_max:   Upper bound of gap_severity for TRAINABLE.
    """
    ready_now_match_min: float
    ready_now_gap_max: float
    trainable_match_min: float
    trainable_match_max: float
    trainable_gap_min: float
    trainable_gap_max: float


@dataclass(frozen=True)
class CredentialBarrierRule:
    """Rule that can force a bucket override when a credential barrier exists.

    Some occupations have hard credential requirements (e.g. nursing licence,
    commercial driving licence) that no amount of skill overlap can satisfy.
    When the user lacks the credential, we may want to force the bucket to
    TRAINABLE or LONG_RESKILL regardless of the numeric score.

    Attributes:
        credential_element_ids: O*NET element_ids that represent the credential.
        required_min_rating: Minimum user rating (0-4) to satisfy the barrier.
        force_bucket: Bucket to assign if the barrier is not met.
        explanation: Human-readable reason for the override.
    """
    credential_element_ids: tuple  # Tuple of element_id strings
    required_min_rating: int
    force_bucket: str  # "trainable" or "long_reskill"
    explanation: str


@dataclass(frozen=True)
class SkillDomainWeights:
    """Per-domain multipliers applied to gap severity calculations.

    A weight of 1.0 means neutral (no adjustment). Higher weights make gaps in
    that domain contribute more to gap_severity, pushing the occupation towards
    LONG_RESKILL. Lower weights reduce the penalty.

    Rationale: a missing safety skill (e.g. psychomotor for a surgeon) is far
    more consequential than a missing soft skill. Weights let us encode this
    domain knowledge without retraining the scorer.
    """
    weights: Dict[SkillDomain, float] = field(default_factory=lambda: {
        SkillDomain.COGNITIVE: 1.0,
        SkillDomain.PSYCHOMOTOR: 1.2,
        SkillDomain.PHYSICAL: 1.1,
        SkillDomain.SENSORY: 1.1,
        SkillDomain.BASIC_SKILLS: 1.0,
        SkillDomain.SOCIAL_SKILLS: 0.8,
        SkillDomain.TECHNICAL: 1.3,
        SkillDomain.COMPLEX_PROBLEM: 1.1,
        SkillDomain.SYSTEMS: 1.0,
        SkillDomain.RESOURCE_MGMT: 0.9,
        SkillDomain.KNOWLEDGE: 1.0,
        SkillDomain.WORK_ACTIVITY: 0.9,
        SkillDomain.OTHER: 1.0,
    })

    def get_weight(self, domain: SkillDomain) -> float:
        """Return the multiplier for a given domain, defaulting to 1.0."""
        return self.weights.get(domain, 1.0)


# ---------------------------------------------------------------------------
# Risk tolerance presets
# ---------------------------------------------------------------------------

class RiskTolerance(str, Enum):
    """User risk tolerance level.

    Controls how aggressively the system classifies occupations into
    higher-readiness buckets.

    RELAXED:  User is comfortable with uncertainty; show more READY_NOW.
    STANDARD: Production defaults.
    STRICT:   User wants high confidence; only show very strong matches.
    """
    RELAXED = "relaxed"
    STANDARD = "standard"
    STRICT = "strict"


# ---------------------------------------------------------------------------
# Complete threshold profile
# ---------------------------------------------------------------------------

@dataclass
class ThresholdProfile:
    """Complete threshold configuration combining all tunable parameters.

    This is the top-level configuration object that the explainer and scorer
    consume. It bundles bucket thresholds, domain weights, credential rules,
    and metadata about which preset (if any) was used.

    Attributes:
        name: Human-readable name for this profile (e.g. "standard", "relaxed").
        risk_tolerance: The RiskTolerance enum value.
        bucket_thresholds: Numeric thresholds for bucket assignment.
        domain_weights: Per-domain gap severity multipliers.
        credential_barriers: List of credential barrier rules that can force
            bucket overrides.
        description: Free-text description of this profile for the UI.
    """
    name: str
    risk_tolerance: RiskTolerance
    bucket_thresholds: BucketThresholds
    domain_weights: SkillDomainWeights
    credential_barriers: list  # List[CredentialBarrierRule]
    description: str = ""


# ---------------------------------------------------------------------------
# Default credential barrier rules
# ---------------------------------------------------------------------------
# These are illustrative. In production, these would be loaded from a database
# or configuration file keyed by occupation SOC codes.

DEFAULT_CREDENTIAL_BARRIERS: list = [
    CredentialBarrierRule(
        credential_element_ids=("2.C.3.a",),  # Medicine and Dentistry knowledge
        required_min_rating=3,
        force_bucket="long_reskill",
        explanation=(
            "This occupation requires formal medical credentials. Without "
            "advanced knowledge in Medicine and Dentistry, licensure is not "
            "possible and the role is a long-term reskill target."
        ),
    ),
    CredentialBarrierRule(
        credential_element_ids=("2.C.5.b",),  # Law and Government knowledge
        required_min_rating=3,
        force_bucket="trainable",
        explanation=(
            "This occupation involves regulated legal practice. Without "
            "advanced knowledge of Law and Government, additional training "
            "and certification are required."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

def _standard_thresholds() -> BucketThresholds:
    """Build STANDARD thresholds from app.core.config.settings.

    This ensures the explainability module stays in sync with the production
    scorer without duplicating magic numbers.
    """
    return BucketThresholds(
        ready_now_match_min=settings.ready_now_match_threshold,
        ready_now_gap_max=settings.ready_now_gap_threshold,
        trainable_match_min=settings.trainable_match_min,
        trainable_match_max=settings.trainable_match_max,
        trainable_gap_min=settings.trainable_gap_min,
        trainable_gap_max=settings.trainable_gap_max,
    )


# Presets keyed by RiskTolerance enum.  The RELAXED preset lowers the
# READY_NOW match bar and raises the gap ceiling; STRICT does the opposite.

THRESHOLD_PRESETS: Dict[RiskTolerance, ThresholdProfile] = {
    RiskTolerance.RELAXED: ThresholdProfile(
        name="relaxed",
        risk_tolerance=RiskTolerance.RELAXED,
        bucket_thresholds=BucketThresholds(
            ready_now_match_min=65.0,   # 10 points lower than standard 75
            ready_now_gap_max=35.0,     # 10 points higher than standard 25
            trainable_match_min=40.0,   # 10 points lower than standard 50
            trainable_match_max=64.0,   # Adjusted to meet ready_now boundary
            trainable_gap_min=36.0,     # Adjusted to meet ready_now boundary
            trainable_gap_max=65.0,     # 10 points higher than standard 55
        ),
        domain_weights=SkillDomainWeights(),
        credential_barriers=DEFAULT_CREDENTIAL_BARRIERS,
        description=(
            "Relaxed thresholds show more opportunities by lowering the bar "
            "for READY_NOW and TRAINABLE. Best for users comfortable with "
            "uncertainty who want the widest funnel of options."
        ),
    ),
    RiskTolerance.STANDARD: ThresholdProfile(
        name="standard",
        risk_tolerance=RiskTolerance.STANDARD,
        bucket_thresholds=_standard_thresholds(),
        domain_weights=SkillDomainWeights(),
        credential_barriers=DEFAULT_CREDENTIAL_BARRIERS,
        description=(
            "Standard thresholds match SkillSprout production defaults. "
            "Balanced between showing enough options and maintaining accuracy."
        ),
    ),
    RiskTolerance.STRICT: ThresholdProfile(
        name="strict",
        risk_tolerance=RiskTolerance.STRICT,
        bucket_thresholds=BucketThresholds(
            ready_now_match_min=85.0,   # 10 points higher than standard 75
            ready_now_gap_max=15.0,     # 10 points lower than standard 25
            trainable_match_min=60.0,   # 10 points higher than standard 50
            trainable_match_max=84.0,   # Adjusted to meet ready_now boundary
            trainable_gap_min=16.0,     # Adjusted to meet ready_now boundary
            trainable_gap_max=45.0,     # 10 points lower than standard 55
        ),
        domain_weights=SkillDomainWeights(),
        credential_barriers=DEFAULT_CREDENTIAL_BARRIERS,
        description=(
            "Strict thresholds require very high match scores and very low "
            "gap severity for READY_NOW. Best for regulated professions or "
            "users who want only high-confidence recommendations."
        ),
    ),
}


def get_threshold_profile(
    risk_tolerance: RiskTolerance = RiskTolerance.STANDARD,
) -> ThresholdProfile:
    """Retrieve the ThresholdProfile for a given risk tolerance level.

    Args:
        risk_tolerance: The desired risk tolerance level.

    Returns:
        The corresponding ThresholdProfile.

    Raises:
        KeyError: If the risk tolerance has no registered preset.
    """
    return THRESHOLD_PRESETS[risk_tolerance]


def get_all_presets() -> Dict[str, ThresholdProfile]:
    """Return all available threshold presets for UI display.

    Returns:
        Dict mapping preset name strings to ThresholdProfile instances.
    """
    return {profile.name: profile for profile in THRESHOLD_PRESETS.values()}
