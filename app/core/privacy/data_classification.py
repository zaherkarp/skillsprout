"""
Data Classification Framework for SkillSprout
==============================================

RATIONALE: Career transition data is inherently sensitive. A person exploring
new occupations may be signaling intent to leave their current employer, which
could have real-world consequences (retaliation, termination, loss of benefits).
This tiered classification system ensures that every piece of data stored by
SkillSprout is explicitly categorized by sensitivity level, and that handling
rules (retention, access, encryption, logging) are enforced programmatically
rather than left to developer judgment.

The four tiers map to concrete risk scenarios:

  TIER 1 (Public):      No user linkage. O*NET data is already public domain.
                         Breach impact: NONE.

  TIER 2 (Pseudonymous): Aggregated patterns that cannot identify individuals
                         without additional linkage. Used for model training
                         and platform analytics.
                         Breach impact: LOW -- statistical patterns only.

  TIER 3 (Personal):    Data tied to a specific user profile. Skill ratings,
                         saved occupations, and search history reveal career
                         interests and self-assessed competencies.
                         Breach impact: MODERATE -- could embarrass or
                         disadvantage a user if their employer discovers
                         they are exploring transitions.

  TIER 4 (Sensitive):   Data that reveals active job-seeking behavior:
                         applications submitted, interview outcomes, offers
                         received. This is the highest-risk category because
                         it proves intent to leave, not just curiosity.
                         Breach impact: HIGH -- direct employment risk.

Each tier carries enforceable rules for retention, access control, and logging
that are checked at the ORM field level and at the API endpoint level.
"""

import enum
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitivity tier definitions
# ---------------------------------------------------------------------------


class DataTier(enum.IntEnum):
    """
    Sensitivity tiers ordered by increasing risk.

    Using IntEnum so tiers can be compared numerically:
        if field_tier >= DataTier.TIER_3_PERSONAL:
            enforce_encryption()

    RATIONALE for numeric ordering: privacy controls should be monotonically
    stricter as tier increases. Numeric comparison makes it trivial to write
    guards like "if tier >= X, require audit logging" without enumerating
    every tier explicitly.
    """

    TIER_1_PUBLIC = 1
    TIER_2_PSEUDONYMOUS = 2
    TIER_3_PERSONAL = 3
    TIER_4_SENSITIVE = 4


# ---------------------------------------------------------------------------
# Human-readable metadata for each tier
# ---------------------------------------------------------------------------

TIER_METADATA: Dict[DataTier, Dict[str, Any]] = {
    DataTier.TIER_1_PUBLIC: {
        "label": "Public",
        "description": "Publicly available reference data with no user linkage.",
        "examples": [
            "O*NET occupation titles and codes",
            "Skill taxonomy definitions (element IDs, names, descriptions)",
            "Job zone and education level metadata",
        ],
        # RATIONALE: Public data has no retention limit because it is not
        # user-generated and carries zero privacy risk. Purging it would
        # degrade the service without any privacy benefit.
        "max_retention_days": None,  # No limit -- public reference data
        "requires_encryption_at_rest": False,
        "requires_audit_log": False,
        "included_in_data_export": False,  # Not user data
        "deleted_on_account_removal": False,  # Shared reference data
    },
    DataTier.TIER_2_PSEUDONYMOUS: {
        "label": "Pseudonymous",
        "description": (
            "Aggregated or de-identified usage patterns that cannot be traced "
            "to a specific individual without additional linkage keys."
        ),
        "examples": [
            "Aggregated recommendation acceptance rates by occupation pair",
            "De-identified model training snapshots",
            "Platform-wide skill gap distributions",
        ],
        # RATIONALE: 365 days balances model training needs against the
        # principle of storage limitation. Aggregated data older than a year
        # is unlikely to improve model accuracy (labor market shifts).
        "max_retention_days": 365,
        "requires_encryption_at_rest": False,
        "requires_audit_log": False,
        "included_in_data_export": False,  # Cannot be linked to user
        "deleted_on_account_removal": False,  # Already de-identified
    },
    DataTier.TIER_3_PERSONAL: {
        "label": "Personal",
        "description": (
            "Data directly linked to a user profile that reveals career "
            "interests and self-assessed competencies."
        ),
        "examples": [
            "User skill self-ratings (UserSkillRating)",
            "Saved / bookmarked occupations",
            "Search history and browsing patterns",
            "User profile metadata",
            "Current occupation selection (UserCurrentOccupation)",
        ],
        # RATIONALE: 180 days for event-level personal data. Profiles
        # themselves persist until the user requests deletion, but
        # granular activity logs are purged to limit exposure.
        "max_retention_days": 180,
        "requires_encryption_at_rest": True,
        "requires_audit_log": True,
        "included_in_data_export": True,
        "deleted_on_account_removal": True,
    },
    DataTier.TIER_4_SENSITIVE: {
        "label": "Sensitive",
        "description": (
            "Data that reveals active job-seeking intent: applications, "
            "interview tracking, outcome data, and transition planning."
        ),
        "examples": [
            "Application tracking feedback (action_type: apply, interview, offer)",
            "Recommendation events with outcome data",
            "Transition intent signals",
        ],
        # RATIONALE: 90 days is the strictest retention window. Sensitive
        # outcome data (did the user apply? get an offer?) is the most
        # dangerous if breached. We keep it only long enough for the
        # recommendation feedback loop, then purge.
        "max_retention_days": 90,
        "requires_encryption_at_rest": True,
        "requires_audit_log": True,
        "included_in_data_export": True,
        "deleted_on_account_removal": True,
    },
}


# ---------------------------------------------------------------------------
# Registry: tracks which models/fields/endpoints belong to which tier
# ---------------------------------------------------------------------------

# Global registry populated by the @data_tier decorator.
# Structure:  { "model_field": { "ModelName.field_name": DataTier, ... },
#               "endpoint":    { "GET /api/v1/...": DataTier, ... } }
_classification_registry: Dict[str, Dict[str, DataTier]] = {
    "model_field": {},
    "endpoint": {},
}


def get_classification_registry() -> Dict[str, Dict[str, DataTier]]:
    """
    Return a read-only snapshot of the classification registry.

    RATIONALE: Exposing the registry allows audit tooling to verify that every
    model field and every endpoint has been explicitly classified. Unclassified
    fields/endpoints should be treated as a compliance gap.
    """
    return {
        category: dict(entries)
        for category, entries in _classification_registry.items()
    }


# ---------------------------------------------------------------------------
# @data_tier decorator
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def data_tier(tier: DataTier, reason: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator that marks a model field, endpoint function, or class with a
    data sensitivity tier.

    Usage on an endpoint:

        @router.get("/user/{user_id}/profile")
        @data_tier(DataTier.TIER_3_PERSONAL, reason="Returns user skill profile")
        async def get_profile(user_id: int, db=Depends(get_db)):
            ...

    Usage on a model class:

        @data_tier(DataTier.TIER_4_SENSITIVE, reason="Tracks job applications")
        class UserFeedback(Base):
            ...

    RATIONALE for decorator approach: Keeping the classification co-located
    with the code it describes ensures that classification stays up to date
    as the codebase evolves. A separate spreadsheet or wiki page would drift
    out of sync within weeks. The decorator also enables runtime introspection
    for automated compliance checks.

    Args:
        tier: The DataTier classification level.
        reason: Optional human-readable explanation for WHY this tier was chosen.
                Encouraged for audit trail clarity.
    """

    def decorator(func_or_class: F) -> F:
        # Attach tier metadata directly to the object for runtime inspection.
        func_or_class._data_tier = tier  # type: ignore[attr-defined]
        func_or_class._data_tier_reason = reason  # type: ignore[attr-defined]

        tier_meta = TIER_METADATA[tier]

        # Register in global classification registry.
        if isinstance(func_or_class, type):
            # It is a class (e.g., an ORM model).
            key = func_or_class.__name__
            _classification_registry["model_field"][key] = tier
            logger.debug(
                "Classified model %s as %s (reason: %s)",
                key,
                tier.name,
                reason or "not specified",
            )
        else:
            # It is a function (e.g., a FastAPI endpoint).
            key = getattr(func_or_class, "__qualname__", func_or_class.__name__)
            _classification_registry["endpoint"][key] = tier
            logger.debug(
                "Classified endpoint %s as %s (reason: %s)",
                key,
                tier.name,
                reason or "not specified",
            )

        @functools.wraps(func_or_class)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # RATIONALE: For TIER_3+ endpoints we log access at DEBUG level.
            # This does NOT log request/response bodies (that would be a
            # privacy violation in itself). It logs only that the endpoint
            # was invoked, enabling audit trail reconstruction.
            if tier >= DataTier.TIER_3_PERSONAL and tier_meta["requires_audit_log"]:
                logger.info(
                    "Access to %s-classified resource: %s",
                    tier.name,
                    key,
                )
            return func_or_class(*args, **kwargs)

        # For classes, return the original class (not the wrapper function).
        if isinstance(func_or_class, type):
            return func_or_class  # type: ignore[return-value]

        # Preserve tier attributes on the wrapper.
        wrapper._data_tier = tier  # type: ignore[attr-defined]
        wrapper._data_tier_reason = reason  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_tier(obj: Any) -> Optional[DataTier]:
    """
    Retrieve the DataTier assigned to an object via @data_tier.

    Returns None if the object has not been classified. Unclassified objects
    should be flagged during compliance audits.
    """
    return getattr(obj, "_data_tier", None)


def get_tier_reason(obj: Any) -> Optional[str]:
    """Retrieve the human-readable reason for an object's tier classification."""
    return getattr(obj, "_data_tier_reason", None)


def get_tier_policy(tier: DataTier) -> Dict[str, Any]:
    """
    Return the full policy metadata for a given tier.

    This is the canonical source of truth for retention limits, encryption
    requirements, and deletion behavior for each classification level.
    """
    return dict(TIER_METADATA[tier])


def list_classified_models() -> Dict[str, DataTier]:
    """Return all model classes that have been classified."""
    return dict(_classification_registry["model_field"])


def list_classified_endpoints() -> Dict[str, DataTier]:
    """Return all endpoint functions that have been classified."""
    return dict(_classification_registry["endpoint"])


def find_unclassified_fields(
    known_models: List[str],
) -> List[str]:
    """
    Given a list of known model class names, return any that have NOT been
    classified with @data_tier.

    RATIONALE: This function is designed to be called in CI/CD pipelines or
    startup health checks. Any unclassified model is a compliance gap that
    should block deployment until resolved.
    """
    classified = set(_classification_registry["model_field"].keys())
    return [model for model in known_models if model not in classified]


# ---------------------------------------------------------------------------
# Pre-classify the existing SkillSprout models
# ---------------------------------------------------------------------------
# RATIONALE: Rather than modifying the model source files (which live in
# app/models/models.py and are shared with Alembic migrations), we register
# classifications here in the privacy module. This keeps privacy policy
# centralized and avoids circular import issues.

MODEL_CLASSIFICATIONS: Dict[str, DataTier] = {
    # --- TIER 1: Public reference data ---
    "Occupation": DataTier.TIER_1_PUBLIC,
    "Skill": DataTier.TIER_1_PUBLIC,
    "OccupationSkill": DataTier.TIER_1_PUBLIC,

    # --- TIER 2: Pseudonymous / aggregate ---
    "ModelRegistry": DataTier.TIER_2_PSEUDONYMOUS,

    # --- TIER 3: Personal user data ---
    "UserProfile": DataTier.TIER_3_PERSONAL,
    "UserCurrentOccupation": DataTier.TIER_3_PERSONAL,
    "UserSkillRating": DataTier.TIER_3_PERSONAL,

    # --- TIER 4: Sensitive outcome / intent data ---
    "RecommendationEvent": DataTier.TIER_4_SENSITIVE,
    "RecommendedOccupation": DataTier.TIER_4_SENSITIVE,
    "UserFeedback": DataTier.TIER_4_SENSITIVE,
}

# Register all model classifications in the global registry.
for _model_name, _tier in MODEL_CLASSIFICATIONS.items():
    _classification_registry["model_field"][_model_name] = _tier


def get_model_tier(model_name: str) -> Optional[DataTier]:
    """
    Look up the classification tier for a model by name.

    Returns None if the model has not been classified -- which should be
    treated as a compliance gap in production.
    """
    return MODEL_CLASSIFICATIONS.get(model_name)


def get_models_for_tier(tier: DataTier) -> List[str]:
    """Return all model names classified at the given tier."""
    return [
        name for name, t in MODEL_CLASSIFICATIONS.items()
        if t == tier
    ]


def get_deletable_models() -> List[str]:
    """
    Return model names that must be deleted when a user requests account
    removal (GDPR Art. 17 / CCPA right to delete).

    RATIONALE: Only models at TIER_3+ are linked to a specific user.
    TIER_1 is public reference data, and TIER_2 is already de-identified.
    Deleting those would degrade service without privacy benefit.
    """
    return [
        name
        for name, tier in MODEL_CLASSIFICATIONS.items()
        if TIER_METADATA[tier]["deleted_on_account_removal"]
    ]


def get_exportable_models() -> List[str]:
    """
    Return model names that must be included in a GDPR data portability
    export (Art. 20) or CCPA right-to-know response.
    """
    return [
        name
        for name, tier in MODEL_CLASSIFICATIONS.items()
        if TIER_METADATA[tier]["included_in_data_export"]
    ]
