"""BLS Employment Projections (2024-2034) for occupations tracked by SkillSprout."""

from typing import Optional

BLS_PROJECTIONS: dict[str, dict] = {
    "15-1251.00": {
        "projected_growth_pct": -10.6,
        "projected_openings_annual": 9600,
        "current_employment": 147400,
        "outlook": "declining",
    },
    "43-4051.00": {
        "projected_growth_pct": -4.5,
        "projected_openings_annual": 389400,
        "current_employment": 2890600,
        "outlook": "declining",
    },
    "43-9021.00": {
        "projected_growth_pct": -32.4,
        "projected_openings_annual": 10300,
        "current_employment": 152900,
        "outlook": "declining",
    },
    "13-2051.00": {
        "projected_growth_pct": 8.2,
        "projected_openings_annual": 37100,
        "current_employment": 348500,
        "outlook": "strong growth",
    },
    "29-1141.00": {
        "projected_growth_pct": 5.6,
        "projected_openings_annual": 193100,
        "current_employment": 3175400,
        "outlook": "moderate growth",
    },
    "41-1011.00": {
        "projected_growth_pct": 0.5,
        "projected_openings_annual": 118600,
        "current_employment": 1178500,
        "outlook": "stable",
    },
    "11-9111.00": {
        "projected_growth_pct": 28.4,
        "projected_openings_annual": 61400,
        "current_employment": 571700,
        "outlook": "strong growth",
    },
    "21-1094.00": {
        "projected_growth_pct": 14.0,
        "projected_openings_annual": 12200,
        "current_employment": 68500,
        "outlook": "strong growth",
    },
    "29-1171.00": {
        "projected_growth_pct": 40.2,
        "projected_openings_annual": 30200,
        "current_employment": 264800,
        "outlook": "strong growth",
    },
    "29-9092.00": {
        "projected_growth_pct": 18.5,
        "projected_openings_annual": 1100,
        "current_employment": 5200,
        "outlook": "strong growth",
    },
    "11-2022.00": {
        "projected_growth_pct": 4.2,
        "projected_openings_annual": 45200,
        "current_employment": 469800,
        "outlook": "moderate growth",
    },
    "13-1161.00": {
        "projected_growth_pct": 13.4,
        "projected_openings_annual": 99800,
        "current_employment": 965600,
        "outlook": "strong growth",
    },
    "13-1199.06": {
        "projected_growth_pct": 9.8,
        "projected_openings_annual": 5600,
        "current_employment": 42300,
        "outlook": "strong growth",
    },
    "15-2051.00": {
        "projected_growth_pct": 35.2,
        "projected_openings_annual": 20800,
        "current_employment": 192300,
        "outlook": "strong growth",
    },
    "23-1011.00": {
        "projected_growth_pct": 5.1,
        "projected_openings_annual": 39500,
        "current_employment": 813900,
        "outlook": "moderate growth",
    },
    "27-3042.00": {
        "projected_growth_pct": -2.3,
        "projected_openings_annual": 4200,
        "current_employment": 48500,
        "outlook": "declining",
    },
    "43-6014.00": {
        "projected_growth_pct": -10.2,
        "projected_openings_annual": 136600,
        "current_employment": 2005000,
        "outlook": "declining",
    },
    "13-2011.00": {
        "projected_growth_pct": 5.8,
        "projected_openings_annual": 126500,
        "current_employment": 1538400,
        "outlook": "moderate growth",
    },
    "15-1299.08": {
        "projected_growth_pct": 10.0,
        "projected_openings_annual": 5300,
        "current_employment": 48200,
        "outlook": "strong growth",
    },
    "35-2014.00": {
        "projected_growth_pct": 5.6,
        "projected_openings_annual": 169400,
        "current_employment": 1547700,
        "outlook": "moderate growth",
    },
    "33-9092.00": {
        "projected_growth_pct": 5.3,
        "projected_openings_annual": 23100,
        "current_employment": 131700,
        "outlook": "moderate growth",
    },
}


def get_projections(onet_code: str) -> Optional[dict]:
    """Return the BLS projection dict for *onet_code*, or None if not found."""
    return BLS_PROJECTIONS.get(onet_code)
