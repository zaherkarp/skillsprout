"""Core bias audit engine for SkillSprout's recommendation system.

Design rationale:
-----------------
Job transition recommendations carry real-world consequences: they influence
which careers people pursue and can systematically steer demographic groups
toward or away from certain occupations. This module implements automated
bias detection to catch disparate impact BEFORE it reaches users.

What we audit:
  1. **Bucket distribution parity**: Do occupations associated with different
     demographic groups receive statistically similar bucket distributions?
     If nursing (85% female workforce) is disproportionately classified as
     LONG_RESKILL while engineering (75% male) is classified as TRAINABLE
     with similar skill profiles, that signals potential bias.

  2. **Skill profile staleness**: O*NET data is periodically updated, but
     some occupations have stale skill profiles. If stale profiles cluster
     in occupations associated with specific demographic groups, the scoring
     engine may produce systematically biased results.

  3. **Score symmetry**: If the scorer produces different match_scores for
     two occupations with identical skill overlap but different SOC codes,
     something in the scoring pipeline is not occupation-agnostic.

Data source:
  We use stub demographic profiles derived from BLS Current Population Survey
  data. In production, these would be loaded from a maintained dataset. The
  stubs are illustrative and should NOT be used for real policy decisions.

Methodology:
  All tests produce AuditFinding objects with a severity level (INFO, WARNING,
  CRITICAL). The caller can filter by severity and decide whether to block
  deployment or flag for human review.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.ml.scoring import BaselineScorer, OccupationScore, SkillGap


# ---------------------------------------------------------------------------
# Severity levels for audit findings
# ---------------------------------------------------------------------------

class AuditSeverity(str, Enum):
    """Severity level for bias audit findings.

    INFO:     Observation, no action required.
    WARNING:  Potential bias detected, warrants human review.
    CRITICAL: Statistically significant disparity, should block deployment
              until investigated.
    """
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Audit finding dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditFinding:
    """A single finding from a bias audit test.

    Attributes:
        test_name: Identifier for the test that produced this finding.
        severity: How serious the finding is.
        description: Human-readable description of what was found.
        affected_occupations: List of O*NET codes affected.
        metric_name: Name of the metric that triggered the finding.
        metric_value: The measured value of the metric.
        threshold: The threshold that was exceeded (if applicable).
        recommended_action: Suggested mitigation.
        details: Additional structured data for downstream processing.
    """
    test_name: str
    severity: AuditSeverity
    description: str
    affected_occupations: List[str]
    metric_name: str
    metric_value: float
    threshold: Optional[float] = None
    recommended_action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stub demographic profiles from BLS data
# ---------------------------------------------------------------------------
# These are simplified representations of BLS Current Population Survey data
# for illustrative purposes. Each entry maps an O*NET SOC code to a
# demographic profile. In production, this would be a database table
# refreshed from BLS quarterly releases.
#
# Fields:
#   pct_female: Percentage of workers who are female.
#   pct_minority: Percentage of workers who identify as non-white.
#   median_age: Median age of workers.
#   last_updated: When this profile was last refreshed.

@dataclass
class DemographicProfile:
    """Demographic composition of an occupation's workforce.

    Source: Simplified from BLS Current Population Survey.
    WARNING: These are stubs for testing. Do NOT use for policy decisions.

    Attributes:
        onet_code: O*NET SOC code.
        title: Occupation title.
        pct_female: Percentage of workers who are female (0-100).
        pct_minority: Percentage of workers who are non-white (0-100).
        median_age: Median worker age.
        last_updated: Date the profile was last refreshed.
    """
    onet_code: str
    title: str
    pct_female: float
    pct_minority: float
    median_age: float
    last_updated: str  # ISO date string


# Stub profiles for a representative set of occupations.
# These are intentionally diverse to exercise the audit tests.
STUB_DEMOGRAPHIC_PROFILES: List[DemographicProfile] = [
    DemographicProfile("29-1141.00", "Registered Nurses", 85.0, 30.0, 42.0, "2024-01-15"),
    DemographicProfile("15-1252.00", "Software Developers", 20.0, 35.0, 34.0, "2024-03-10"),
    DemographicProfile("11-9013.00", "Farmers, Ranchers, Ag Managers", 25.0, 10.0, 55.0, "2023-06-01"),
    DemographicProfile("53-3032.00", "Heavy Truck Drivers", 7.0, 42.0, 46.0, "2024-02-20"),
    DemographicProfile("25-2021.00", "Elementary School Teachers", 80.0, 20.0, 41.0, "2024-01-15"),
    DemographicProfile("47-2111.00", "Electricians", 3.0, 18.0, 40.0, "2024-04-01"),
    DemographicProfile("13-2011.00", "Accountants and Auditors", 60.0, 28.0, 43.0, "2024-03-15"),
    DemographicProfile("31-1014.00", "Nursing Assistants", 88.0, 52.0, 36.0, "2023-09-01"),
    DemographicProfile("17-2141.00", "Mechanical Engineers", 9.0, 22.0, 38.0, "2024-02-10"),
    DemographicProfile("21-1021.00", "Child/Family Social Workers", 82.0, 35.0, 40.0, "2023-11-20"),
]


def get_demographic_profiles() -> Dict[str, DemographicProfile]:
    """Return all stub demographic profiles keyed by onet_code.

    Returns:
        Dict mapping O*NET code to DemographicProfile.
    """
    return {p.onet_code: p for p in STUB_DEMOGRAPHIC_PROFILES}


# ---------------------------------------------------------------------------
# Audit thresholds
# ---------------------------------------------------------------------------
# These define when a metric crosses from INFO to WARNING to CRITICAL.

# Bucket distribution parity: max allowed difference in READY_NOW rate
# between demographic groups. E.g., if >15% more male-dominated occupations
# land in READY_NOW than female-dominated ones, that is a WARNING.
PARITY_WARNING_THRESHOLD = 0.15   # 15 percentage points
PARITY_CRITICAL_THRESHOLD = 0.25  # 25 percentage points

# Staleness: occupations with profiles older than this many days get flagged.
STALENESS_WARNING_DAYS = 365      # 1 year
STALENESS_CRITICAL_DAYS = 730     # 2 years

# Score symmetry: max allowed difference in match_score for occupations
# with identical skill overlap (for synthetic test pairs).
SYMMETRY_WARNING_THRESHOLD = 5.0   # 5 points on 0-100 scale
SYMMETRY_CRITICAL_THRESHOLD = 10.0  # 10 points


# ---------------------------------------------------------------------------
# Audit tests
# ---------------------------------------------------------------------------

class BiasAuditEngine:
    """Core bias audit engine.

    Runs a suite of automated tests against a set of scored occupations
    and their demographic profiles. Each test produces zero or more
    AuditFinding objects.

    Usage::

        engine = BiasAuditEngine()
        findings = engine.run_full_audit(scored_occupations, profiles)
    """

    def run_full_audit(
        self,
        scores: List[OccupationScore],
        profiles: Optional[Dict[str, DemographicProfile]] = None,
    ) -> List[AuditFinding]:
        """Run all bias audit tests and return consolidated findings.

        Args:
            scores: List of OccupationScore objects from the scorer.
            profiles: Demographic profiles keyed by onet_code. If None,
                uses the stub profiles.

        Returns:
            List of AuditFinding objects, sorted by severity (CRITICAL first).
        """
        if profiles is None:
            profiles = get_demographic_profiles()

        findings: List[AuditFinding] = []

        findings.extend(self.test_bucket_distribution_parity(scores, profiles))
        findings.extend(self.test_skill_profile_staleness(profiles))
        findings.extend(self.test_score_symmetry(scores))

        # Sort: CRITICAL > WARNING > INFO
        severity_order = {
            AuditSeverity.CRITICAL: 0,
            AuditSeverity.WARNING: 1,
            AuditSeverity.INFO: 2,
        }
        findings.sort(key=lambda f: severity_order.get(f.severity, 99))

        return findings

    # -------------------------------------------------------------------
    # Test 1: Bucket distribution parity
    # -------------------------------------------------------------------

    def test_bucket_distribution_parity(
        self,
        scores: List[OccupationScore],
        profiles: Dict[str, DemographicProfile],
    ) -> List[AuditFinding]:
        """Test whether bucket assignments are demographically balanced.

        Methodology:
          1. Split occupations into two groups based on a demographic axis
             (e.g., female-majority vs. male-majority).
          2. Compute the READY_NOW rate for each group.
          3. If the difference exceeds thresholds, flag it.

        We test along two axes: gender composition and racial/ethnic composition.

        Args:
            scores: Scored occupations.
            profiles: Demographic profiles.

        Returns:
            List of AuditFinding objects.
        """
        findings: List[AuditFinding] = []

        # Only analyse occupations that have demographic profiles
        scored_with_profile = [
            (s, profiles[s.onet_code])
            for s in scores
            if s.onet_code in profiles
        ]

        if len(scored_with_profile) < 4:
            findings.append(AuditFinding(
                test_name="bucket_distribution_parity",
                severity=AuditSeverity.INFO,
                description=(
                    f"Only {len(scored_with_profile)} occupations have "
                    f"demographic profiles. Insufficient data for parity test."
                ),
                affected_occupations=[s.onet_code for s, _ in scored_with_profile],
                metric_name="sample_size",
                metric_value=float(len(scored_with_profile)),
            ))
            return findings

        # --- Gender axis ---
        findings.extend(self._test_parity_on_axis(
            scored_with_profile,
            axis_name="gender",
            split_fn=lambda profile: profile.pct_female >= 50.0,
            group_a_label="female_majority",
            group_b_label="male_majority",
        ))

        # --- Racial/ethnic composition axis ---
        findings.extend(self._test_parity_on_axis(
            scored_with_profile,
            axis_name="race_ethnicity",
            split_fn=lambda profile: profile.pct_minority >= 40.0,
            group_a_label="high_minority",
            group_b_label="low_minority",
        ))

        return findings

    def _test_parity_on_axis(
        self,
        scored_with_profile: List[Tuple[OccupationScore, DemographicProfile]],
        axis_name: str,
        split_fn,
        group_a_label: str,
        group_b_label: str,
    ) -> List[AuditFinding]:
        """Test bucket parity along a single demographic axis.

        Args:
            scored_with_profile: List of (score, profile) tuples.
            axis_name: Human-readable axis name.
            split_fn: Function that takes a DemographicProfile and returns True
                for group A, False for group B.
            group_a_label: Label for the True group.
            group_b_label: Label for the False group.

        Returns:
            List of AuditFinding objects.
        """
        findings: List[AuditFinding] = []

        group_a = [(s, p) for s, p in scored_with_profile if split_fn(p)]
        group_b = [(s, p) for s, p in scored_with_profile if not split_fn(p)]

        if not group_a or not group_b:
            findings.append(AuditFinding(
                test_name="bucket_distribution_parity",
                severity=AuditSeverity.INFO,
                description=(
                    f"Cannot test {axis_name} parity: one group is empty "
                    f"({group_a_label}: {len(group_a)}, "
                    f"{group_b_label}: {len(group_b)})."
                ),
                affected_occupations=[],
                metric_name=f"{axis_name}_group_balance",
                metric_value=0.0,
            ))
            return findings

        # Compute READY_NOW rate per group
        rate_a = sum(1 for s, _ in group_a if s.bucket == "ready_now") / len(group_a)
        rate_b = sum(1 for s, _ in group_b if s.bucket == "ready_now") / len(group_b)
        disparity = abs(rate_a - rate_b)

        # Also check LONG_RESKILL rate (inverse concern)
        lr_rate_a = sum(1 for s, _ in group_a if s.bucket == "long_reskill") / len(group_a)
        lr_rate_b = sum(1 for s, _ in group_b if s.bucket == "long_reskill") / len(group_b)
        lr_disparity = abs(lr_rate_a - lr_rate_b)

        # Determine severity for READY_NOW disparity
        if disparity >= PARITY_CRITICAL_THRESHOLD:
            severity = AuditSeverity.CRITICAL
        elif disparity >= PARITY_WARNING_THRESHOLD:
            severity = AuditSeverity.WARNING
        else:
            severity = AuditSeverity.INFO

        findings.append(AuditFinding(
            test_name="bucket_distribution_parity",
            severity=severity,
            description=(
                f"READY_NOW rate disparity on {axis_name} axis: "
                f"{group_a_label}={rate_a:.1%} vs {group_b_label}={rate_b:.1%} "
                f"(difference: {disparity:.1%}). "
                f"LONG_RESKILL rate: {group_a_label}={lr_rate_a:.1%} vs "
                f"{group_b_label}={lr_rate_b:.1%}."
            ),
            affected_occupations=(
                [s.onet_code for s, _ in group_a] +
                [s.onet_code for s, _ in group_b]
            ),
            metric_name=f"ready_now_disparity_{axis_name}",
            metric_value=round(disparity, 4),
            threshold=PARITY_WARNING_THRESHOLD,
            recommended_action=(
                "Review scoring thresholds and skill domain weights for "
                f"occupations split on the {axis_name} axis. Consider "
                "applying SKILL_REWEIGHTING mitigation."
                if severity != AuditSeverity.INFO
                else "No action required."
            ),
            details={
                "axis": axis_name,
                f"{group_a_label}_ready_now_rate": round(rate_a, 4),
                f"{group_b_label}_ready_now_rate": round(rate_b, 4),
                f"{group_a_label}_long_reskill_rate": round(lr_rate_a, 4),
                f"{group_b_label}_long_reskill_rate": round(lr_rate_b, 4),
                f"{group_a_label}_count": len(group_a),
                f"{group_b_label}_count": len(group_b),
            },
        ))

        return findings

    # -------------------------------------------------------------------
    # Test 2: Skill profile staleness
    # -------------------------------------------------------------------

    def test_skill_profile_staleness(
        self,
        profiles: Dict[str, DemographicProfile],
    ) -> List[AuditFinding]:
        """Test whether skill profiles are disproportionately stale.

        Stale O*NET data means skill requirements may have shifted since
        the last update. If stale profiles cluster in occupations with
        specific demographic compositions, the scoring engine produces
        biased results (e.g., overestimating gaps for occupations whose
        actual skill requirements have decreased).

        We flag individual stale profiles and check whether staleness
        correlates with demographic axes.

        Args:
            profiles: Demographic profiles with last_updated dates.

        Returns:
            List of AuditFinding objects.
        """
        findings: List[AuditFinding] = []
        now = datetime.utcnow()

        stale_warning: List[str] = []
        stale_critical: List[str] = []
        staleness_days: Dict[str, float] = {}

        for code, profile in profiles.items():
            try:
                last_updated = datetime.fromisoformat(profile.last_updated)
            except (ValueError, TypeError):
                # Cannot parse date; flag as critical staleness
                stale_critical.append(code)
                staleness_days[code] = float("inf")
                continue

            age_days = (now - last_updated).days
            staleness_days[code] = float(age_days)

            if age_days >= STALENESS_CRITICAL_DAYS:
                stale_critical.append(code)
            elif age_days >= STALENESS_WARNING_DAYS:
                stale_warning.append(code)

        # Report individual staleness findings
        if stale_critical:
            findings.append(AuditFinding(
                test_name="skill_profile_staleness",
                severity=AuditSeverity.CRITICAL,
                description=(
                    f"{len(stale_critical)} occupation(s) have skill profiles "
                    f"older than {STALENESS_CRITICAL_DAYS} days: "
                    f"{', '.join(stale_critical)}. Scores for these occupations "
                    f"may be unreliable."
                ),
                affected_occupations=stale_critical,
                metric_name="critical_staleness_count",
                metric_value=float(len(stale_critical)),
                threshold=float(STALENESS_CRITICAL_DAYS),
                recommended_action=(
                    "Refresh O*NET data for flagged occupations. Apply "
                    "STALENESS_PENALTY mitigation to reduce confidence in "
                    "scores derived from stale data."
                ),
                details={"stale_codes": stale_critical},
            ))

        if stale_warning:
            findings.append(AuditFinding(
                test_name="skill_profile_staleness",
                severity=AuditSeverity.WARNING,
                description=(
                    f"{len(stale_warning)} occupation(s) have skill profiles "
                    f"older than {STALENESS_WARNING_DAYS} days: "
                    f"{', '.join(stale_warning)}."
                ),
                affected_occupations=stale_warning,
                metric_name="warning_staleness_count",
                metric_value=float(len(stale_warning)),
                threshold=float(STALENESS_WARNING_DAYS),
                recommended_action=(
                    "Schedule O*NET data refresh for these occupations. "
                    "Consider applying STALENESS_PENALTY mitigation."
                ),
                details={"stale_codes": stale_warning},
            ))

        # Check demographic correlation of staleness
        if staleness_days:
            self._check_staleness_demographic_correlation(
                profiles, staleness_days, findings
            )

        return findings

    def _check_staleness_demographic_correlation(
        self,
        profiles: Dict[str, DemographicProfile],
        staleness_days: Dict[str, float],
        findings: List[AuditFinding],
    ) -> None:
        """Check if staleness correlates with demographic composition.

        If high-minority or high-female occupations have systematically
        older profiles, the scoring engine may be less accurate for those
        groups.

        Args:
            profiles: Demographic profiles.
            staleness_days: Days since last update per occupation.
            findings: List to append findings to (mutated in place).
        """
        female_majority_staleness = []
        male_majority_staleness = []

        for code, days in staleness_days.items():
            if code not in profiles:
                continue
            profile = profiles[code]
            if days == float("inf"):
                days = STALENESS_CRITICAL_DAYS * 2  # Treat unparseable as very stale
            if profile.pct_female >= 50.0:
                female_majority_staleness.append(days)
            else:
                male_majority_staleness.append(days)

        if female_majority_staleness and male_majority_staleness:
            mean_f = statistics.mean(female_majority_staleness)
            mean_m = statistics.mean(male_majority_staleness)
            diff = abs(mean_f - mean_m)

            if diff > STALENESS_WARNING_DAYS * 0.5:
                staler_group = (
                    "female-majority" if mean_f > mean_m else "male-majority"
                )
                findings.append(AuditFinding(
                    test_name="skill_profile_staleness",
                    severity=AuditSeverity.WARNING,
                    description=(
                        f"Staleness is unevenly distributed by gender: "
                        f"{staler_group} occupations average "
                        f"{max(mean_f, mean_m):.0f} days old vs "
                        f"{min(mean_f, mean_m):.0f} days for the other group. "
                        f"This may introduce systematic scoring bias."
                    ),
                    affected_occupations=list(staleness_days.keys()),
                    metric_name="staleness_gender_gap_days",
                    metric_value=round(diff, 1),
                    threshold=STALENESS_WARNING_DAYS * 0.5,
                    recommended_action=(
                        f"Prioritise O*NET data refresh for {staler_group} "
                        f"occupations to reduce staleness asymmetry."
                    ),
                    details={
                        "female_majority_mean_days": round(mean_f, 1),
                        "male_majority_mean_days": round(mean_m, 1),
                    },
                ))

    # -------------------------------------------------------------------
    # Test 3: Score symmetry
    # -------------------------------------------------------------------

    def test_score_symmetry(
        self,
        scores: List[OccupationScore],
    ) -> List[AuditFinding]:
        """Test whether the scorer treats occupations symmetrically.

        If two occupations have very similar skill profiles (overlap > 80%
        of their gaps), their match_scores should be similar for the same
        user. Large discrepancies suggest that something beyond skill
        overlap is influencing the score (e.g., hard-coded occupation-
        specific adjustments, or data quality issues).

        We check all pairs and flag the largest asymmetries.

        Args:
            scores: List of scored occupations.

        Returns:
            List of AuditFinding objects.
        """
        findings: List[AuditFinding] = []

        if len(scores) < 2:
            return findings

        # Build gap fingerprints: set of element_ids per occupation
        fingerprints: Dict[str, set] = {}
        for score in scores:
            fingerprints[score.onet_code] = {
                gap.element_id for gap in score.top_gaps
            }

        # Check all pairs
        score_lookup = {s.onet_code: s for s in scores}
        checked = set()
        asymmetric_pairs: List[Dict[str, Any]] = []

        for code_a in fingerprints:
            for code_b in fingerprints:
                if code_a >= code_b:
                    continue
                pair_key = (code_a, code_b)
                if pair_key in checked:
                    continue
                checked.add(pair_key)

                # Compute Jaccard similarity of gap fingerprints
                gaps_a = fingerprints[code_a]
                gaps_b = fingerprints[code_b]
                if not gaps_a and not gaps_b:
                    continue  # Both have no gaps; skip

                union = gaps_a | gaps_b
                intersection = gaps_a & gaps_b
                jaccard = len(intersection) / len(union) if union else 0.0

                # Only test pairs with high skill overlap (Jaccard > 0.6)
                if jaccard < 0.6:
                    continue

                score_a = score_lookup[code_a]
                score_b = score_lookup[code_b]
                score_diff = abs(score_a.match_score - score_b.match_score)

                if score_diff >= SYMMETRY_WARNING_THRESHOLD:
                    asymmetric_pairs.append({
                        "code_a": code_a,
                        "code_b": code_b,
                        "jaccard_similarity": round(jaccard, 3),
                        "match_score_a": score_a.match_score,
                        "match_score_b": score_b.match_score,
                        "score_difference": round(score_diff, 2),
                    })

        if asymmetric_pairs:
            # Sort by score_difference descending
            asymmetric_pairs.sort(
                key=lambda p: p["score_difference"], reverse=True
            )

            worst = asymmetric_pairs[0]
            severity = (
                AuditSeverity.CRITICAL
                if worst["score_difference"] >= SYMMETRY_CRITICAL_THRESHOLD
                else AuditSeverity.WARNING
            )

            findings.append(AuditFinding(
                test_name="score_symmetry",
                severity=severity,
                description=(
                    f"{len(asymmetric_pairs)} occupation pair(s) have high "
                    f"skill overlap but asymmetric scores. Worst pair: "
                    f"{worst['code_a']} vs {worst['code_b']} "
                    f"(Jaccard={worst['jaccard_similarity']}, "
                    f"score diff={worst['score_difference']}). "
                    f"This may indicate occupation-specific scoring bias."
                ),
                affected_occupations=[
                    p["code_a"] for p in asymmetric_pairs
                ] + [
                    p["code_b"] for p in asymmetric_pairs
                ],
                metric_name="max_asymmetric_score_diff",
                metric_value=worst["score_difference"],
                threshold=SYMMETRY_WARNING_THRESHOLD,
                recommended_action=(
                    "Investigate why these occupations with similar gap "
                    "profiles receive different scores. Check for data "
                    "quality issues or occupation-specific adjustments."
                ),
                details={"asymmetric_pairs": asymmetric_pairs},
            ))
        else:
            findings.append(AuditFinding(
                test_name="score_symmetry",
                severity=AuditSeverity.INFO,
                description=(
                    "No significant score asymmetry detected among "
                    "occupations with overlapping skill profiles."
                ),
                affected_occupations=[],
                metric_name="max_asymmetric_score_diff",
                metric_value=0.0,
                recommended_action="No action required.",
            ))

        return findings


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_bias_audit(
    scores: List[OccupationScore],
    profiles: Optional[Dict[str, DemographicProfile]] = None,
) -> List[AuditFinding]:
    """Run a full bias audit on a set of scored occupations.

    This is the primary entry point for external callers.

    Args:
        scores: Scored occupations from the baseline scorer.
        profiles: Optional demographic profiles. Defaults to stubs.

    Returns:
        List of AuditFinding objects sorted by severity.
    """
    engine = BiasAuditEngine()
    return engine.run_full_audit(scores, profiles)
