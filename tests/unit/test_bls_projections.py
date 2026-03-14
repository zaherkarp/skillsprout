"""Unit tests for app.data.bls_projections module."""
import pytest

from app.data.bls_projections import BLS_PROJECTIONS, get_projections


REQUIRED_KEYS = {"projected_growth_pct", "projected_openings_annual", "current_employment", "outlook"}
VALID_OUTLOOKS = {"strong growth", "moderate growth", "stable", "declining"}


def _expected_outlook(growth: float) -> str:
    if growth > 8:
        return "strong growth"
    elif growth >= 3:
        return "moderate growth"
    elif growth >= -2:
        return "stable"
    else:
        return "declining"


class TestProjectionsDataCompleteness:
    def test_projections_data_completeness(self):
        """Every code in BLS_PROJECTIONS has all required keys."""
        for code, entry in BLS_PROJECTIONS.items():
            missing = REQUIRED_KEYS - set(entry.keys())
            assert not missing, f"{code} missing keys: {missing}"

    def test_outlook_labels(self):
        """Every outlook is one of the four valid strings."""
        for code, entry in BLS_PROJECTIONS.items():
            assert entry["outlook"] in VALID_OUTLOOKS, (
                f"{code}: invalid outlook '{entry['outlook']}'"
            )

    def test_outlook_consistency(self):
        """Outlook string matches the projected_growth_pct thresholds."""
        for code, entry in BLS_PROJECTIONS.items():
            expected = _expected_outlook(entry["projected_growth_pct"])
            assert entry["outlook"] == expected, (
                f"{code}: growth={entry['projected_growth_pct']}% "
                f"should be '{expected}', got '{entry['outlook']}'"
            )


class TestGetProjections:
    def test_get_projections_found(self):
        """get_projections returns data for a known code."""
        result = get_projections("29-1141.00")
        assert result is not None
        assert result["projected_growth_pct"] == 5.6

    def test_get_projections_not_found(self):
        """get_projections returns None for an unknown code."""
        assert get_projections("99-9999.00") is None
