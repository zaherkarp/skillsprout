"""
SkillSprout Privacy Framework
==============================

This package implements the privacy-by-design framework for SkillSprout.
All privacy controls are centralized here to ensure consistency and
auditability.

Modules:
    data_classification - Sensitivity tiers and @data_tier decorator
    retention_policy    - Time-based data retention rules with Celery enforcement
    private_mode        - Header-based middleware for zero-logging sessions
    data_export         - GDPR/CCPA data portability endpoint
    data_deletion       - Cascading account deletion endpoint
"""

from app.core.privacy.data_classification import (
    DataTier,
    data_tier,
    get_tier,
    get_tier_reason,
    get_tier_policy,
    get_model_tier,
    get_models_for_tier,
    get_deletable_models,
    get_exportable_models,
    MODEL_CLASSIFICATIONS,
    TIER_METADATA,
)

from app.core.privacy.private_mode import (
    is_private_mode,
    get_private_mode_disclosure,
    PrivateModeMiddleware,
    PRIVATE_MODE_REQUEST_HEADER,
)

__all__ = [
    # Data classification
    "DataTier",
    "data_tier",
    "get_tier",
    "get_tier_reason",
    "get_tier_policy",
    "get_model_tier",
    "get_models_for_tier",
    "get_deletable_models",
    "get_exportable_models",
    "MODEL_CLASSIFICATIONS",
    "TIER_METADATA",
    # Private mode
    "is_private_mode",
    "get_private_mode_disclosure",
    "PrivateModeMiddleware",
    "PRIVATE_MODE_REQUEST_HEADER",
]
