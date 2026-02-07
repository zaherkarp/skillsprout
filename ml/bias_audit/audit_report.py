"""Generates a Markdown bias audit report from audit findings.

Design rationale:
-----------------
Automated bias detection is only useful if the results are communicated
clearly to stakeholders (engineers, product managers, compliance officers).
This module generates a structured Markdown report (``docs/bias-audit-report.md``)
that includes:

1. **Executive summary**: High-level pass/fail with counts by severity.
2. **Detailed findings**: Each finding with description, affected occupations,
   metrics, and recommended mitigations.
3. **Demographic coverage**: Which occupations had demographic profiles and
   which were excluded from analysis.
4. **Methodology notes**: How each test works and what the thresholds mean.

The report is designed to be:
  - Checked into version control alongside the code, creating an audit trail.
  - Readable by non-technical stakeholders without requiring code access.
  - Parseable by CI/CD systems (e.g., grep for "CRITICAL" to fail a pipeline).

Output path:
  Default: ``docs/bias-audit-report.md`` (relative to project root).
  Override via the ``output_path`` parameter.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ml.bias_audit.audit_framework import (
    AuditFinding,
    AuditSeverity,
    DemographicProfile,
    get_demographic_profiles,
)


# Default output path relative to project root
DEFAULT_REPORT_PATH = os.path.join("docs", "bias-audit-report.md")


def generate_audit_report(
    findings: List[AuditFinding],
    profiles: Optional[Dict[str, DemographicProfile]] = None,
    output_path: Optional[str] = None,
    scored_occupation_count: int = 0,
    model_version: str = "unknown",
) -> str:
    """Generate a Markdown bias audit report and write it to disk.

    Args:
        findings: List of AuditFinding objects from the audit engine.
        profiles: Demographic profiles used in the audit. If None, uses stubs.
        output_path: File path for the report. Defaults to docs/bias-audit-report.md.
        scored_occupation_count: Number of occupations that were scored.
        model_version: The model version string that was audited.

    Returns:
        The full Markdown content of the report (also written to disk).
    """
    if profiles is None:
        profiles = get_demographic_profiles()

    if output_path is None:
        output_path = DEFAULT_REPORT_PATH

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Build report sections
    sections: List[str] = []

    sections.append(_build_header(model_version))
    sections.append(_build_executive_summary(findings, scored_occupation_count))
    sections.append(_build_findings_detail(findings))
    sections.append(_build_demographic_coverage(profiles))
    sections.append(_build_methodology_notes())
    sections.append(_build_recommended_mitigations(findings))
    sections.append(_build_footer())

    report = "\n\n".join(sections)

    # Write to disk
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------

def _build_header(model_version: str) -> str:
    """Build the report header with title and metadata.

    Args:
        model_version: The model version being audited.

    Returns:
        Markdown string for the header section.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return (
        "# SkillSprout Bias Audit Report\n\n"
        f"**Generated**: {now}  \n"
        f"**Model Version**: {model_version}  \n"
        f"**Audit Framework**: ml.bias_audit.audit_framework v1.0  \n"
        "**Status**: Automated audit - requires human review for any "
        "WARNING or CRITICAL findings."
    )


def _build_executive_summary(
    findings: List[AuditFinding],
    scored_count: int,
) -> str:
    """Build the executive summary section.

    Args:
        findings: All audit findings.
        scored_count: Number of occupations scored.

    Returns:
        Markdown string for the executive summary.
    """
    critical = sum(1 for f in findings if f.severity == AuditSeverity.CRITICAL)
    warning = sum(1 for f in findings if f.severity == AuditSeverity.WARNING)
    info = sum(1 for f in findings if f.severity == AuditSeverity.INFO)

    if critical > 0:
        overall = "FAIL - Critical issues detected"
    elif warning > 0:
        overall = "REVIEW - Warnings detected"
    else:
        overall = "PASS - No significant bias detected"

    lines = [
        "## Executive Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Overall Status | **{overall}** |",
        f"| Occupations Scored | {scored_count} |",
        f"| Total Findings | {len(findings)} |",
        f"| Critical | {critical} |",
        f"| Warning | {warning} |",
        f"| Info | {info} |",
    ]

    return "\n".join(lines)


def _build_findings_detail(findings: List[AuditFinding]) -> str:
    """Build the detailed findings section.

    Each finding gets its own subsection with full details.

    Args:
        findings: All audit findings.

    Returns:
        Markdown string for the findings detail section.
    """
    if not findings:
        return "## Detailed Findings\n\nNo findings to report."

    lines = ["## Detailed Findings\n"]

    for i, finding in enumerate(findings, 1):
        severity_badge = _severity_badge(finding.severity)
        lines.append(f"### Finding {i}: {finding.test_name} {severity_badge}\n")
        lines.append(f"**Severity**: {finding.severity.value.upper()}  ")
        lines.append(f"**Metric**: {finding.metric_name} = {finding.metric_value}")
        if finding.threshold is not None:
            lines.append(f"  (threshold: {finding.threshold})")
        lines.append(f"\n**Description**: {finding.description}\n")

        if finding.affected_occupations:
            occ_list = ", ".join(finding.affected_occupations[:10])
            if len(finding.affected_occupations) > 10:
                occ_list += f" ... and {len(finding.affected_occupations) - 10} more"
            lines.append(f"**Affected Occupations**: {occ_list}\n")

        if finding.recommended_action:
            lines.append(f"**Recommended Action**: {finding.recommended_action}\n")

        if finding.details:
            lines.append("**Details**:\n")
            lines.append("```")
            for key, value in finding.details.items():
                lines.append(f"  {key}: {value}")
            lines.append("```\n")

        lines.append("---\n")

    return "\n".join(lines)


def _build_demographic_coverage(
    profiles: Dict[str, DemographicProfile],
) -> str:
    """Build the demographic coverage section.

    Shows which occupations had profiles and their basic demographics.

    Args:
        profiles: Demographic profiles used in the audit.

    Returns:
        Markdown string for the demographic coverage section.
    """
    lines = [
        "## Demographic Coverage\n",
        "The following occupations had demographic profiles available for "
        "bias analysis:\n",
        "| O*NET Code | Title | % Female | % Minority | Last Updated |",
        "|------------|-------|----------|------------|--------------|",
    ]

    for code in sorted(profiles.keys()):
        p = profiles[code]
        lines.append(
            f"| {p.onet_code} | {p.title} | {p.pct_female:.0f}% "
            f"| {p.pct_minority:.0f}% | {p.last_updated} |"
        )

    lines.append(
        f"\n**Total profiles**: {len(profiles)}  \n"
        "**Note**: These are stub profiles for development. Production "
        "audits should use current BLS data."
    )

    return "\n".join(lines)


def _build_methodology_notes() -> str:
    """Build the methodology section explaining each test.

    Returns:
        Markdown string for the methodology section.
    """
    return (
        "## Methodology\n\n"
        "### Test 1: Bucket Distribution Parity\n\n"
        "Occupations are split into demographic groups (e.g., female-majority "
        "vs. male-majority) based on BLS workforce composition data. The "
        "READY_NOW classification rate is computed for each group. A "
        "disparity exceeding 15 percentage points triggers a WARNING; "
        "exceeding 25 points triggers a CRITICAL finding.\n\n"
        "**Rationale**: If the scoring engine systematically favours "
        "occupations associated with one demographic group, it creates a "
        "feedback loop that narrows career exploration for other groups.\n\n"
        "### Test 2: Skill Profile Staleness\n\n"
        "O*NET skill profiles are checked for recency. Profiles older than "
        "365 days trigger a WARNING; older than 730 days trigger CRITICAL. "
        "We also check whether staleness correlates with demographic axes.\n\n"
        "**Rationale**: Stale data can over- or under-estimate skill gaps. "
        "If staleness clusters in occupations dominated by a specific group, "
        "the bias is systematic.\n\n"
        "### Test 3: Score Symmetry\n\n"
        "Pairs of occupations with high skill overlap (Jaccard > 0.6) are "
        "checked for match_score agreement. Differences exceeding 5 points "
        "trigger a WARNING; exceeding 10 points trigger CRITICAL.\n\n"
        "**Rationale**: The scorer should be occupation-agnostic. If two "
        "occupations with the same skill profile receive different scores, "
        "something beyond skills is influencing the result."
    )


def _build_recommended_mitigations(findings: List[AuditFinding]) -> str:
    """Build the consolidated mitigations section.

    Aggregates unique recommended actions from all non-INFO findings.

    Args:
        findings: All audit findings.

    Returns:
        Markdown string for the mitigations section.
    """
    actionable = [
        f for f in findings
        if f.severity in (AuditSeverity.WARNING, AuditSeverity.CRITICAL)
        and f.recommended_action
    ]

    if not actionable:
        return (
            "## Recommended Mitigations\n\n"
            "No mitigations required. All tests passed within acceptable "
            "thresholds."
        )

    # Deduplicate actions
    seen = set()
    unique_actions: List[str] = []
    for f in actionable:
        if f.recommended_action not in seen:
            seen.add(f.recommended_action)
            unique_actions.append(f.recommended_action)

    lines = ["## Recommended Mitigations\n"]
    for i, action in enumerate(unique_actions, 1):
        lines.append(f"{i}. {action}")

    lines.append(
        "\nSee `ml.bias_audit.mitigation_strategies` for implementation "
        "details of SKILL_REWEIGHTING and STALENESS_PENALTY mitigations."
    )

    return "\n".join(lines)


def _build_footer() -> str:
    """Build the report footer.

    Returns:
        Markdown string for the footer.
    """
    return (
        "---\n\n"
        "*This report was generated automatically by SkillSprout's bias "
        "audit framework. All findings require human review before "
        "deployment decisions are made. The demographic data used is "
        "illustrative and should be replaced with current BLS data for "
        "production audits.*"
    )


def _severity_badge(severity: AuditSeverity) -> str:
    """Return a text badge for a severity level.

    Args:
        severity: The severity level.

    Returns:
        A bracketed text badge like [CRITICAL].
    """
    badges = {
        AuditSeverity.CRITICAL: "[CRITICAL]",
        AuditSeverity.WARNING: "[WARNING]",
        AuditSeverity.INFO: "[INFO]",
    }
    return badges.get(severity, "[UNKNOWN]")
