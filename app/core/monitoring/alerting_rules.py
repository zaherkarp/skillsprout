"""Alerting rule definitions for SkillSprout.

Encodes alerting thresholds as Python data structures and provides a helper
to render them as a Prometheus-compatible ``prometheus_rules.yml`` file.

Priority levels:
  - P1 (critical):  Health endpoint fails >60 s, scoring p99 >2 s
  - P2 (warning):   Cold-start fallback >50 %, calibration not retrained >7 d
  - P3 (info):      O*NET cache older than 30 days
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AlertRule:
    """A single Prometheus alerting rule."""

    alert: str
    expr: str
    for_duration: str
    severity: str  # P1 | P2 | P3
    summary: str
    description: str
    labels: dict = field(default_factory=dict)
    annotations: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

ALERTING_RULES: List[AlertRule] = [
    # ---- P1 (critical) ----
    AlertRule(
        alert="HealthEndpointDown",
        expr='up{job="skillsprout"} == 0 or probe_success{job="skillsprout-health"} == 0',
        for_duration="60s",
        severity="P1",
        summary="SkillSprout health endpoint is unreachable",
        description=(
            "The /health endpoint has been failing for more than 60 seconds. "
            "Immediate investigation required."
        ),
    ),
    AlertRule(
        alert="ScoringLatencyP99TooHigh",
        expr=(
            "histogram_quantile(0.99, "
            'rate(scoring_duration_seconds_bucket{job="skillsprout"}[5m])'
            ") > 2"
        ),
        for_duration="60s",
        severity="P1",
        summary="Scoring p99 latency exceeds 2 seconds",
        description=(
            "The 99th-percentile scoring latency has exceeded the 2-second "
            "SLO for at least 60 seconds."
        ),
    ),

    # ---- P2 (warning) ----
    AlertRule(
        alert="ColdStartFallbackRateHigh",
        expr=(
            "rate(cold_start_fallbacks_total[5m]) / "
            "clamp_min(rate(requests_total[5m]), 1) > 0.5"
        ),
        for_duration="5m",
        severity="P2",
        summary="Cold-start fallback rate exceeds 50 %",
        description=(
            "More than half of scoring requests are falling back to the "
            "cold-start baseline.  Consider warming the cache or training "
            "the calibration model."
        ),
    ),
    AlertRule(
        alert="CalibrationModelStale",
        expr=(
            '(time() - model_version_info{is_calibrated="true"}) > 604800'
        ),
        for_duration="1h",
        severity="P2",
        summary="Calibration model has not been retrained in 7 days",
        description=(
            "The calibrated model has not been retrained in over 7 days "
            "(604 800 s).  Verify that the periodic Celery beat task is "
            "running and that sufficient feedback data is available."
        ),
    ),

    # ---- P3 (info) ----
    AlertRule(
        alert="ONetCacheStale",
        expr=(
            "(time() - max(onet_cache_last_updated_timestamp)) > 2592000"
        ),
        for_duration="1h",
        severity="P3",
        summary="O*NET cache is older than 30 days",
        description=(
            "The O*NET occupation cache has not been refreshed for more "
            "than 30 days (2 592 000 s).  Run the cache-warming task."
        ),
    ),
]


# ---------------------------------------------------------------------------
# YAML renderer
# ---------------------------------------------------------------------------

def _rule_to_dict(rule: AlertRule) -> dict:
    """Convert an ``AlertRule`` to the dict structure expected by Prometheus."""
    labels = {"severity": rule.severity}
    labels.update(rule.labels)

    annotations = {
        "summary": rule.summary,
        "description": rule.description,
    }
    annotations.update(rule.annotations)

    entry: dict = {
        "alert": rule.alert,
        "expr": rule.expr,
        "for": rule.for_duration,
        "labels": labels,
        "annotations": annotations,
    }
    return entry


def generate_prometheus_rules_yaml(
    group_name: str = "skillsprout_alerts",
    rules: Optional[List[AlertRule]] = None,
) -> str:
    """Render alerting rules as a Prometheus-compatible YAML string.

    Args:
        group_name: The ``groups[].name`` field in the output.
        rules: List of rules to render.  Defaults to ``ALERTING_RULES``.

    Returns:
        A YAML string suitable for writing to ``prometheus_rules.yml``.
    """
    if rules is None:
        rules = ALERTING_RULES

    doc = {
        "groups": [
            {
                "name": group_name,
                "rules": [_rule_to_dict(r) for r in rules],
            }
        ]
    }
    return yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True)


def write_prometheus_rules_file(
    path: str = "prometheus_rules.yml",
    group_name: str = "skillsprout_alerts",
) -> str:
    """Write the alerting rules YAML to disk.

    Args:
        path: Destination file path.
        group_name: Group name for the YAML document.

    Returns:
        The absolute path of the written file.
    """
    import os

    content = generate_prometheus_rules_yaml(group_name=group_name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)

    abs_path = os.path.abspath(path)
    logger.info("Wrote Prometheus alerting rules to %s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# Threshold constants (importable for use in application logic)
# ---------------------------------------------------------------------------

# P1 thresholds
HEALTH_FAIL_MAX_SECONDS = 60
SCORING_P99_MAX_SECONDS = 2.0

# P2 thresholds
COLD_START_FALLBACK_MAX_RATIO = 0.50
CALIBRATION_STALE_MAX_SECONDS = 7 * 24 * 3600  # 7 days

# P3 thresholds
ONET_CACHE_STALE_MAX_SECONDS = 30 * 24 * 3600  # 30 days
