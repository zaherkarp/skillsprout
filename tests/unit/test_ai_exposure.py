"""Unit tests for app.data.ai_exposure module."""
import pytest

from app.data.ai_exposure import EXPOSURE_DATA, get_exposure, DEMOGRAPHIC_CONTEXT


REQUIRED_KEYS = {"theoretical_exposure", "observed_exposure", "exposure_rank"}
VALID_RANKS = {"high", "moderate", "low", "minimal"}


class TestExposureDataCompleteness:
    def test_exposure_data_completeness(self):
        """Every code in EXPOSURE_DATA has all required keys."""
        for code, entry in EXPOSURE_DATA.items():
            missing = REQUIRED_KEYS - set(entry.keys())
            assert not missing, f"{code} missing keys: {missing}"

    def test_observed_lte_theoretical(self):
        """Observed exposure never exceeds theoretical exposure."""
        for code, entry in EXPOSURE_DATA.items():
            assert entry["observed_exposure"] <= entry["theoretical_exposure"], (
                f"{code}: observed ({entry['observed_exposure']}) > "
                f"theoretical ({entry['theoretical_exposure']})"
            )

    def test_exposure_rank_values(self):
        """Every exposure_rank is one of the four valid strings."""
        for code, entry in EXPOSURE_DATA.items():
            assert entry["exposure_rank"] in VALID_RANKS, (
                f"{code}: invalid rank '{entry['exposure_rank']}'"
            )


class TestGetExposure:
    def test_get_exposure_found(self):
        """get_exposure returns data for a known code."""
        result = get_exposure("15-1251.00")
        assert result is not None
        assert result["observed_exposure"] == 0.75

    def test_get_exposure_not_found(self):
        """get_exposure returns None for an unknown code."""
        assert get_exposure("99-9999.00") is None


class TestDemographicContext:
    def test_demographic_context_keys(self):
        """DEMOGRAPHIC_CONTEXT has all expected keys."""
        expected = {
            "high_vs_low_female_gap_pp",
            "high_vs_low_earnings_premium_pct",
            "high_grad_degree_pct",
            "low_grad_degree_pct",
            "pct_workers_zero_coverage",
        }
        assert set(DEMOGRAPHIC_CONTEXT.keys()) == expected
