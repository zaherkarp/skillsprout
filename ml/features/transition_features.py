"""Transition-aware feature engineering for career transition scoring.

This module computes features that capture the *direction* and *difficulty*
of a career transition between an origin occupation and a target occupation.
Features are designed to integrate with the calibration layer
(``app.ml.calibration.CalibrationModel``) by extending its feature vector.

All feature functions accept raw O*NET-style dicts so they can be called
both from live API responses and from cached database rows.  When upstream
data is unavailable the functions return ``None`` rather than raising, which
lets downstream consumers decide on imputation strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder / stub data used when live O*NET look-ups are unavailable.
# In production these would come from a database or the O*NET API.
# ---------------------------------------------------------------------------

# Bright Outlook flag keyed by O*NET-SOC code (stub subset).
_BRIGHT_OUTLOOK_CODES: Set[str] = {
    "15-1252.00",  # Software Developers
    "15-1299.08",  # Web Developers
    "29-1141.00",  # Registered Nurses
    "13-2011.00",  # Accountants and Auditors
    "11-9013.00",  # Farmers, Ranchers, and Agricultural Managers
    "15-1211.00",  # Computer Systems Analysts
    "15-1212.00",  # Information Security Analysts
    "29-1071.00",  # Physician Assistants
    "21-1021.00",  # Child, Family, and School Social Workers
    "25-1011.00",  # Business Teachers, Postsecondary
}

# Placeholder median annual wage (USD) keyed by O*NET-SOC code.
_PLACEHOLDER_WAGES: Dict[str, float] = {
    "15-1252.00": 120_730.0,
    "15-1299.08": 78_300.0,
    "15-1244.00": 80_600.0,
    "29-1141.00": 77_600.0,
    "13-2011.00": 73_560.0,
    "11-9013.00": 73_060.0,
    "15-1211.00": 99_270.0,
    "15-1212.00": 102_600.0,
    "29-1071.00": 115_390.0,
    "21-1021.00": 48_430.0,
    "25-1011.00": 97_290.0,
    "43-9061.00": 36_360.0,
    "35-2014.00": 28_800.0,
    "53-3032.00": 45_760.0,
    "41-2031.00": 29_120.0,
    "51-4121.00": 44_420.0,
}

# Credentials / licenses required — stub mapping.
_CREDENTIAL_REQUIRED: Dict[str, List[str]] = {
    "29-1141.00": ["Registered Nurse License (RN)"],
    "29-1071.00": ["Physician Assistant License"],
    "13-2011.00": ["CPA (optional but preferred)"],
    "15-1212.00": ["CISSP (optional but preferred)"],
    "21-1021.00": ["Licensed Clinical Social Worker (LCSW)"],
    "25-1011.00": ["Doctoral or Professional Degree"],
}

# NAICS-style industry sector codes (2-digit) per O*NET-SOC prefix.
# The first two digits of the SOC code loosely map to a sector.
_SOC_PREFIX_TO_NAICS: Dict[str, int] = {
    "11": 55,  # Management
    "13": 52,  # Financial
    "15": 51,  # Information / Tech
    "17": 23,  # Engineering / Construction
    "19": 54,  # Science / Professional
    "21": 62,  # Community / Social / Healthcare-adjacent
    "23": 54,  # Legal / Professional
    "25": 61,  # Education
    "27": 71,  # Arts / Entertainment
    "29": 62,  # Healthcare
    "31": 62,  # Healthcare Support
    "33": 92,  # Protective Service / Government
    "35": 72,  # Food Service / Hospitality
    "37": 56,  # Building / Grounds Maintenance
    "39": 81,  # Personal Care
    "41": 44,  # Retail / Sales
    "43": 56,  # Office / Admin Support
    "45": 11,  # Farming / Agriculture
    "47": 23,  # Construction
    "49": 81,  # Installation / Maintenance
    "51": 31,  # Production / Manufacturing
    "53": 48,  # Transportation
}


# ---------------------------------------------------------------------------
# Feature computation functions
# ---------------------------------------------------------------------------


def skill_direction_vector(
    origin_skills: List[Dict[str, Any]],
    target_skills: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Compute the signed delta between origin and target per skill domain.

    For every skill element that appears in *either* occupation, the delta is
    defined as ``target_importance - origin_importance`` (normalised to 0-100
    scale).  Positive values indicate the target requires *more* of that
    skill; negative values indicate the user already exceeds the target need.

    Args:
        origin_skills: Skill dicts for the user's current occupation.
            Each dict must contain ``element_id`` and ``importance``.
        target_skills: Skill dicts for the target occupation.

    Returns:
        Dict mapping ``element_id`` to the signed importance delta, or
        ``None`` if both skill lists are empty.
    """
    if not origin_skills and not target_skills:
        return None

    origin_map: Dict[str, float] = {
        s["element_id"]: float(s.get("importance", 0) or 0)
        for s in origin_skills
        if "element_id" in s
    }
    target_map: Dict[str, float] = {
        s["element_id"]: float(s.get("importance", 0) or 0)
        for s in target_skills
        if "element_id" in s
    }

    all_ids = set(origin_map) | set(target_map)
    if not all_ids:
        return None

    return {
        eid: target_map.get(eid, 0.0) - origin_map.get(eid, 0.0)
        for eid in sorted(all_ids)
    }


def experience_transfer_ratio(
    origin_skills: List[Dict[str, Any]],
    target_skills: List[Dict[str, Any]],
    top_k: int = 5,
) -> Optional[float]:
    """Fraction of the origin's top-k skills that are relevant in the target.

    A skill from the origin is considered *relevant* in the target if it
    appears with importance >= 50 in the target skill list.

    Args:
        origin_skills: Skill dicts for the origin occupation.
        target_skills: Skill dicts for the target occupation.
        top_k: Number of top origin skills to consider (by importance).

    Returns:
        Float in [0, 1] representing the transfer ratio, or ``None`` if the
        origin has no skills.
    """
    if not origin_skills:
        return None

    # Sort origin skills by importance descending and take top k
    sorted_origin = sorted(
        origin_skills,
        key=lambda s: float(s.get("importance", 0) or 0),
        reverse=True,
    )
    top_origin = sorted_origin[:top_k]

    if not top_origin:
        return None

    target_important: Set[str] = {
        s["element_id"]
        for s in target_skills
        if float(s.get("importance", 0) or 0) >= 50.0 and "element_id" in s
    }

    transferable = sum(
        1 for s in top_origin
        if s.get("element_id") in target_important
    )

    return transferable / len(top_origin)


def occupation_demand_signal(onet_code: str) -> Optional[bool]:
    """Return whether the occupation is flagged as Bright Outlook by O*NET.

    This is a stub implementation using a hard-coded set of codes.  In
    production this would query the O*NET API or a local cache table.

    Args:
        onet_code: O*NET-SOC code (e.g. ``"15-1252.00"``).

    Returns:
        ``True`` if the occupation is Bright Outlook, ``False`` if it is
        known but not Bright Outlook, or ``None`` if the code is not
        recognised in the stub data.
    """
    if not onet_code:
        return None

    # Normalise code format (strip whitespace)
    code = onet_code.strip()

    # If we have wage data for the code we consider it "known" even if
    # it is not in the Bright Outlook set.
    is_known = code in _BRIGHT_OUTLOOK_CODES or code in _PLACEHOLDER_WAGES
    if not is_known:
        return None

    return code in _BRIGHT_OUTLOOK_CODES


def salary_delta(
    origin_code: str,
    target_code: str,
) -> Optional[float]:
    """Estimated salary change when transitioning from origin to target.

    Uses placeholder median wage data.  Returns the absolute dollar
    difference (positive means a raise, negative means a pay cut).

    Args:
        origin_code: O*NET-SOC code for the origin occupation.
        target_code: O*NET-SOC code for the target occupation.

    Returns:
        Dollar delta (target - origin), or ``None`` if either code lacks
        wage data.
    """
    origin_wage = _PLACEHOLDER_WAGES.get(origin_code)
    target_wage = _PLACEHOLDER_WAGES.get(target_code)

    if origin_wage is None or target_wage is None:
        return None

    return target_wage - origin_wage


def credential_barrier(target_code: str) -> Optional[Dict[str, Any]]:
    """Flag credentials or licences required for the target occupation.

    Args:
        target_code: O*NET-SOC code for the target occupation.

    Returns:
        Dict with ``required`` (bool) and ``credentials`` (list of str),
        or ``None`` if the code is not recognised.
    """
    if not target_code:
        return None

    code = target_code.strip()
    is_known = code in _CREDENTIAL_REQUIRED or code in _PLACEHOLDER_WAGES
    if not is_known:
        return None

    creds = _CREDENTIAL_REQUIRED.get(code, [])
    return {
        "required": len(creds) > 0,
        "credentials": creds,
    }


def industry_distance(
    origin_code: str,
    target_code: str,
) -> Optional[float]:
    """Compute NAICS-based taxonomy distance between two occupations.

    Uses a simplified mapping from the first two digits of the SOC code to
    a 2-digit NAICS sector.  The distance is defined as:

    * 0.0  -- same NAICS sector
    * 0.5  -- adjacent sectors (absolute NAICS difference <= 10)
    * 1.0  -- distant sectors

    Args:
        origin_code: O*NET-SOC code for the origin.
        target_code: O*NET-SOC code for the target.

    Returns:
        Distance in [0, 1], or ``None`` if either code cannot be mapped.
    """
    origin_prefix = _extract_soc_prefix(origin_code)
    target_prefix = _extract_soc_prefix(target_code)

    if origin_prefix is None or target_prefix is None:
        return None

    origin_naics = _SOC_PREFIX_TO_NAICS.get(origin_prefix)
    target_naics = _SOC_PREFIX_TO_NAICS.get(target_prefix)

    if origin_naics is None or target_naics is None:
        return None

    if origin_naics == target_naics:
        return 0.0

    diff = abs(origin_naics - target_naics)
    if diff <= 10:
        return 0.5

    return 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_soc_prefix(onet_code: str) -> Optional[str]:
    """Extract the 2-digit SOC major-group prefix from an O*NET-SOC code.

    Examples:
        ``"15-1252.00"`` -> ``"15"``
        ``"29-1141.00"`` -> ``"29"``

    Returns ``None`` for malformed codes.
    """
    if not onet_code or not isinstance(onet_code, str):
        return None

    code = onet_code.strip()
    if len(code) < 2:
        return None

    prefix = code[:2]
    if not prefix.isdigit():
        return None

    return prefix


# ---------------------------------------------------------------------------
# FeatureVector dataclass — names every feature with a human-readable
# explanation.  This is the contract between the feature layer and the
# calibration model.
# ---------------------------------------------------------------------------


@dataclass
class TransitionFeatureVector:
    """Complete feature vector for a career transition.

    Each field carries a human-readable ``metadata`` entry in
    :pyattr:`feature_explanations` describing what the value represents.
    """

    # --- Skill direction ---
    skill_direction_mean: Optional[float] = field(
        default=None,
        metadata={"explanation": "Mean signed importance delta across all skill domains (positive = target needs more)."},
    )
    skill_direction_std: Optional[float] = field(
        default=None,
        metadata={"explanation": "Std-dev of importance deltas; high values indicate uneven skill shifts."},
    )
    skill_direction_max_positive: Optional[float] = field(
        default=None,
        metadata={"explanation": "Largest positive delta — the single biggest new skill demand."},
    )
    skill_direction_max_negative: Optional[float] = field(
        default=None,
        metadata={"explanation": "Most negative delta — the skill most over-supplied by origin."},
    )

    # --- Experience transfer ---
    experience_transfer_ratio: Optional[float] = field(
        default=None,
        metadata={"explanation": "Fraction (0-1) of the origin's top skills that are relevant in the target."},
    )

    # --- Demand signal ---
    bright_outlook: Optional[bool] = field(
        default=None,
        metadata={"explanation": "True if the target is flagged as O*NET Bright Outlook (growing demand)."},
    )

    # --- Salary ---
    salary_delta: Optional[float] = field(
        default=None,
        metadata={"explanation": "Estimated annual salary change in USD (positive = raise)."},
    )
    salary_delta_pct: Optional[float] = field(
        default=None,
        metadata={"explanation": "Salary delta as percentage of origin salary."},
    )

    # --- Credential barrier ---
    credential_required: Optional[bool] = field(
        default=None,
        metadata={"explanation": "True if the target occupation requires specific credentials/licences."},
    )
    credential_count: Optional[int] = field(
        default=None,
        metadata={"explanation": "Number of credentials/licences required for the target."},
    )

    # --- Industry distance ---
    industry_distance: Optional[float] = field(
        default=None,
        metadata={"explanation": "NAICS-based taxonomy distance (0=same sector, 0.5=adjacent, 1=distant)."},
    )

    @classmethod
    def feature_explanations(cls) -> Dict[str, str]:
        """Return a mapping of field name to human-readable explanation."""
        explanations: Dict[str, str] = {}
        for f in cls.__dataclass_fields__.values():
            meta = f.metadata
            if meta and "explanation" in meta:
                explanations[f.name] = meta["explanation"]
        return explanations

    def to_array(self) -> np.ndarray:
        """Convert to a numpy array suitable for the calibration model.

        ``None`` values are replaced with ``np.nan`` so that downstream
        imputers can handle them.

        Returns:
            1-D numpy array with one element per feature.
        """
        values = [
            self.skill_direction_mean,
            self.skill_direction_std,
            self.skill_direction_max_positive,
            self.skill_direction_max_negative,
            self.experience_transfer_ratio,
            1.0 if self.bright_outlook else (0.0 if self.bright_outlook is not None else None),
            self.salary_delta,
            self.salary_delta_pct,
            1.0 if self.credential_required else (0.0 if self.credential_required is not None else None),
            float(self.credential_count) if self.credential_count is not None else None,
            self.industry_distance,
        ]
        return np.array(
            [v if v is not None else np.nan for v in values],
            dtype=np.float64,
        )

    @classmethod
    def ordered_feature_names(cls) -> List[str]:
        """Return feature names in the same order as :meth:`to_array`."""
        return [
            "skill_direction_mean",
            "skill_direction_std",
            "skill_direction_max_positive",
            "skill_direction_max_negative",
            "experience_transfer_ratio",
            "bright_outlook",
            "salary_delta",
            "salary_delta_pct",
            "credential_required",
            "credential_count",
            "industry_distance",
        ]


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------


def build_transition_features(
    origin_code: str,
    target_code: str,
    origin_skills: List[Dict[str, Any]],
    target_skills: List[Dict[str, Any]],
) -> TransitionFeatureVector:
    """Build the complete transition feature vector.

    This is the primary entry point for downstream consumers.  It calls
    every feature-family function and assembles the results into a
    :class:`TransitionFeatureVector`.

    Args:
        origin_code: O*NET-SOC code for the origin occupation.
        target_code: O*NET-SOC code for the target occupation.
        origin_skills: Skill dicts for the origin occupation.
        target_skills: Skill dicts for the target occupation.

    Returns:
        Fully populated :class:`TransitionFeatureVector` (fields may be
        ``None`` when upstream data is missing).
    """
    # --- Skill direction ---
    direction = skill_direction_vector(origin_skills, target_skills)
    dir_mean: Optional[float] = None
    dir_std: Optional[float] = None
    dir_max_pos: Optional[float] = None
    dir_max_neg: Optional[float] = None

    if direction:
        vals = np.array(list(direction.values()), dtype=np.float64)
        dir_mean = float(np.mean(vals))
        dir_std = float(np.std(vals)) if len(vals) > 1 else 0.0
        dir_max_pos = float(np.max(vals))
        dir_max_neg = float(np.min(vals))

    # --- Experience transfer ---
    transfer = experience_transfer_ratio(origin_skills, target_skills)

    # --- Demand signal ---
    demand = occupation_demand_signal(target_code)

    # --- Salary ---
    sal = salary_delta(origin_code, target_code)
    sal_pct: Optional[float] = None
    if sal is not None:
        origin_wage = _PLACEHOLDER_WAGES.get(origin_code)
        if origin_wage and origin_wage > 0:
            sal_pct = (sal / origin_wage) * 100.0

    # --- Credential barrier ---
    cred = credential_barrier(target_code)
    cred_required: Optional[bool] = None
    cred_count: Optional[int] = None
    if cred is not None:
        cred_required = cred["required"]
        cred_count = len(cred["credentials"])

    # --- Industry distance ---
    ind_dist = industry_distance(origin_code, target_code)

    return TransitionFeatureVector(
        skill_direction_mean=dir_mean,
        skill_direction_std=dir_std,
        skill_direction_max_positive=dir_max_pos,
        skill_direction_max_negative=dir_max_neg,
        experience_transfer_ratio=transfer,
        bright_outlook=demand,
        salary_delta=sal,
        salary_delta_pct=sal_pct,
        credential_required=cred_required,
        credential_count=cred_count,
        industry_distance=ind_dist,
    )


# ---------------------------------------------------------------------------
# Calibration integration helper
# ---------------------------------------------------------------------------


def augment_calibration_array(
    base_array: np.ndarray,
    transition_vector: TransitionFeatureVector,
) -> np.ndarray:
    """Concatenate base calibration features with transition features.

    The calibration model (``app.ml.calibration``) expects a 1-D feature
    array.  This helper appends the transition features so that a retrained
    calibration model can leverage them.

    Args:
        base_array: 1-D array from ``CalibrationModel._features_to_array``.
        transition_vector: A :class:`TransitionFeatureVector`.

    Returns:
        Concatenated 1-D numpy array.
    """
    transition_arr = transition_vector.to_array()
    # Flatten base_array in case it has extra dims from reshape(1, -1)
    base_flat = base_array.flatten()
    return np.concatenate([base_flat, transition_arr])
