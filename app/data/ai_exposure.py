"""AI exposure data from Eloundou et al. (2023) and the Anthropic Economic Index.

Theoretical exposure is the share of tasks an LLM could theoretically speed up
by at least 2x (Eloundou et al.).  Observed exposure is the share actually being
performed by AI in professional settings (Anthropic Economic Index, March 2026).
"""

from typing import Optional

EXPOSURE_DATA: dict[str, dict] = {
    "15-1251.00": {
        "theoretical_exposure": 0.94,
        "observed_exposure": 0.75,
        "exposure_rank": "high",
    },
    "43-4051.00": {
        "theoretical_exposure": 0.90,
        "observed_exposure": 0.71,
        "exposure_rank": "high",
    },
    "43-9021.00": {
        "theoretical_exposure": 0.88,
        "observed_exposure": 0.67,
        "exposure_rank": "high",
    },
    "13-2051.00": {
        "theoretical_exposure": 0.90,
        "observed_exposure": 0.62,
        "exposure_rank": "high",
    },
    "43-6014.00": {
        "theoretical_exposure": 0.90,
        "observed_exposure": 0.58,
        "exposure_rank": "high",
    },
    "15-1299.08": {
        "theoretical_exposure": 0.94,
        "observed_exposure": 0.55,
        "exposure_rank": "high",
    },
    "27-3042.00": {
        "theoretical_exposure": 0.85,
        "observed_exposure": 0.53,
        "exposure_rank": "high",
    },
    "13-2011.00": {
        "theoretical_exposure": 0.86,
        "observed_exposure": 0.50,
        "exposure_rank": "high",
    },
    "15-2051.00": {
        "theoretical_exposure": 0.90,
        "observed_exposure": 0.48,
        "exposure_rank": "high",
    },
    "23-1011.00": {
        "theoretical_exposure": 0.78,
        "observed_exposure": 0.42,
        "exposure_rank": "moderate",
    },
    "29-1141.00": {
        "theoretical_exposure": 0.40,
        "observed_exposure": 0.08,
        "exposure_rank": "low",
    },
    "41-1011.00": {
        "theoretical_exposure": 0.55,
        "observed_exposure": 0.12,
        "exposure_rank": "low",
    },
    "11-9111.00": {
        "theoretical_exposure": 0.62,
        "observed_exposure": 0.18,
        "exposure_rank": "moderate",
    },
    "21-1094.00": {
        "theoretical_exposure": 0.45,
        "observed_exposure": 0.06,
        "exposure_rank": "minimal",
    },
    "29-1171.00": {
        "theoretical_exposure": 0.42,
        "observed_exposure": 0.09,
        "exposure_rank": "low",
    },
    "29-9092.00": {
        "theoretical_exposure": 0.38,
        "observed_exposure": 0.05,
        "exposure_rank": "minimal",
    },
    "11-2022.00": {
        "theoretical_exposure": 0.60,
        "observed_exposure": 0.15,
        "exposure_rank": "low",
    },
    "13-1161.00": {
        "theoretical_exposure": 0.82,
        "observed_exposure": 0.38,
        "exposure_rank": "moderate",
    },
    "13-1199.06": {
        "theoretical_exposure": 0.65,
        "observed_exposure": 0.20,
        "exposure_rank": "moderate",
    },
    "35-2014.00": {
        "theoretical_exposure": 0.10,
        "observed_exposure": 0.00,
        "exposure_rank": "minimal",
    },
    "33-9092.00": {
        "theoretical_exposure": 0.05,
        "observed_exposure": 0.00,
        "exposure_rank": "minimal",
    },
    # --- Anthropic Labor Market Report (March 2026) additions ---
    "15-1251.00": {
        "theoretical_exposure": 0.94,
        "observed_exposure": 0.75,
        "exposure_rank": "high",
    },
    "29-2072.00": {
        "theoretical_exposure": 0.88,
        "observed_exposure": 0.67,
        "exposure_rank": "high",
    },
    "13-1161.01": {
        "theoretical_exposure": 0.94,
        "observed_exposure": 0.65,
        "exposure_rank": "high",
    },
    "41-4012.00": {
        "theoretical_exposure": 0.80,
        "observed_exposure": 0.63,
        "exposure_rank": "high",
    },
    "15-1253.00": {
        "theoretical_exposure": 0.94,
        "observed_exposure": 0.52,
        "exposure_rank": "high",
    },
    "15-1212.00": {
        "theoretical_exposure": 0.94,
        "observed_exposure": 0.49,
        "exposure_rank": "high",
    },
    "15-1232.00": {
        "theoretical_exposure": 0.94,
        "observed_exposure": 0.47,
        "exposure_rank": "high",
    },
    "47-2111.00": {
        "theoretical_exposure": 0.17,
        "observed_exposure": 0.00,
        "exposure_rank": "minimal",
    },
    "37-3011.00": {
        "theoretical_exposure": 0.04,
        "observed_exposure": 0.00,
        "exposure_rank": "minimal",
    },
    "35-3023.00": {
        "theoretical_exposure": 0.17,
        "observed_exposure": 0.00,
        "exposure_rank": "minimal",
    },
}

DEMOGRAPHIC_CONTEXT: dict[str, float | int] = {
    "high_vs_low_female_gap_pp": 16,
    "high_vs_low_earnings_premium_pct": 47,
    "high_grad_degree_pct": 17.4,
    "low_grad_degree_pct": 4.5,
    "pct_workers_zero_coverage": 30,
}


def get_exposure(onet_code: str) -> Optional[dict]:
    """Return the exposure dict for *onet_code*, or None if not found."""
    return EXPOSURE_DATA.get(onet_code)
