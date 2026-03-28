"""
tests/test_features/test_quality_composites.py

Tests for src/features/quality_composites.py (FE-008 to FE-012).

Coverage:
  - Output columns appended, no rows dropped for all five functions
  - _missing flags are binary integers
  - log_market_cap: non-positive cur_mkt_cap flagged missing
  - earnings_persistence_score: requires window quarters of history
  - roe_change_1q: first quarter per ticker is missing (no lag)
  - firm_risk_composite: NaN inputs flagged missing
  - yield_sustainability_score: requires FE-001 and FE-002 pre-computed;
    any component NaN propagates to missing
  - Cross-sectional z-scores are zero-mean per date
"""

import numpy as np
import pandas as pd
import pytest

from src.features.dividend_metrics import earnings_payout_ratio, fcf_payout_ratio
from src.features.quality_composites import (
    earnings_persistence_score,
    firm_risk_composite,
    log_market_cap,
    roe_change_1q,
    yield_sustainability_score,
)


class TestLogMarketCap:
    def test_output_columns_appended(self, base_panel):
        result = log_market_cap(base_panel)
        assert "log_market_cap_usd" in result.columns
        assert "log_market_cap_usd_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        assert len(log_market_cap(base_panel)) == len(base_panel)

    def test_positive_values_not_missing(self, base_panel):
        result = log_market_cap(base_panel)
        assert result["log_market_cap_usd_missing"].sum() == 0  # all positive in fixture

    def test_non_positive_mktcap_flagged(self):
        idx = pd.MultiIndex.from_tuples(
            [("T1", pd.Timestamp("2020-03-31")),
             ("T1", pd.Timestamp("2020-06-30"))],
            names=["ticker", "date"],
        )
        panel = pd.DataFrame({"cur_mkt_cap": [0.0, -1e9]}, index=idx)
        result = log_market_cap(panel)
        assert result["log_market_cap_usd_missing"].sum() == 2

    def test_log_values_finite(self, base_panel):
        result = log_market_cap(base_panel)
        valid = result[result["log_market_cap_usd_missing"] == 0]["log_market_cap_usd"]
        assert np.isfinite(valid).all()


class TestEarningsPersistenceScore:
    def test_output_columns_appended(self, base_panel):
        result = earnings_persistence_score(base_panel)
        assert "earnings_persistence_score" in result.columns
        assert "earnings_persistence_score_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        assert len(earnings_persistence_score(base_panel)) == len(base_panel)

    def test_first_window_quarters_missing(self, base_panel):
        result = earnings_persistence_score(base_panel, window=8)
        dates = result.index.get_level_values("date").unique().sort_values()
        for ticker in ["AAPL_US", "JNJ_US", "KO_US"]:
            for d in dates[:8]:
                assert result.loc[(ticker, d), "earnings_persistence_score_missing"] == 1

    def test_valid_values_non_negative(self, base_panel):
        result = earnings_persistence_score(base_panel)
        valid = result[result["earnings_persistence_score_missing"] == 0]
        assert (valid["earnings_persistence_score"] >= 0).all()

    def test_custom_window(self, base_panel):
        r4 = earnings_persistence_score(base_panel, window=4)
        r8 = earnings_persistence_score(base_panel, window=8)
        assert (r4["earnings_persistence_score_missing"] == 0).sum() > \
               (r8["earnings_persistence_score_missing"] == 0).sum()


class TestRoeChange1q:
    def test_output_columns_appended(self, base_panel):
        result = roe_change_1q(base_panel)
        assert "roe_change_1q_pct" in result.columns
        assert "roe_change_1q_pct_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        assert len(roe_change_1q(base_panel)) == len(base_panel)

    def test_first_quarter_per_ticker_missing(self, base_panel):
        result = roe_change_1q(base_panel)
        dates = result.index.get_level_values("date").unique().sort_values()
        first_date = dates[0]
        for ticker in ["AAPL_US", "JNJ_US", "KO_US"]:
            assert result.loc[(ticker, first_date), "roe_change_1q_pct_missing"] == 1

    def test_no_cross_ticker_leakage(self):
        """Lag must stay within each ticker."""
        dates = pd.date_range("2020-03-31", periods=4, freq="QE")
        idx = pd.MultiIndex.from_product(
            [["T1", "T2"], dates], names=["ticker", "date"]
        )
        # T1 ROE: 0.10, 0.15, 0.20, 0.25 → changes: NaN, +0.05, +0.05, +0.05
        # T2 ROE: 0.30, 0.30, 0.30, 0.30 → changes: NaN, 0.0,  0.0,  0.0
        roe = [0.10, 0.15, 0.20, 0.25, 0.30, 0.30, 0.30, 0.30]
        panel = pd.DataFrame({"return_on_eq": roe}, index=idx)
        result = roe_change_1q(panel)

        t2_changes = result.loc["T2", "roe_change_1q_pct"].dropna()
        np.testing.assert_allclose(t2_changes.values, 0.0, atol=1e-10)


class TestFirmRiskComposite:
    def test_output_columns_appended(self, base_panel):
        result = firm_risk_composite(base_panel)
        assert "firm_risk_composite" in result.columns
        assert "firm_risk_composite_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        assert len(firm_risk_composite(base_panel)) == len(base_panel)

    def test_missing_flag_binary(self, base_panel):
        result = firm_risk_composite(base_panel)
        assert set(result["firm_risk_composite_missing"].unique()).issubset({0, 1})

    def test_nan_beta_flagged_missing(self):
        dates = pd.date_range("2020-03-31", periods=2, freq="QE")
        idx = pd.MultiIndex.from_product(
            [["T1", "T2"], dates], names=["ticker", "date"]
        )
        panel = pd.DataFrame({
            "beta_raw_1yr": [np.nan, 1.0, 1.2, 0.8],
            "short_and_long_term_debt": [1e9, 2e9, 1.5e9, 2.5e9],
            "cur_mkt_cap": [5e10, 6e10, 4e10, 7e10],
        }, index=idx)
        result = firm_risk_composite(panel)
        assert result.loc[("T1", dates[0]), "firm_risk_composite_missing"] == 1

    def test_cross_sectional_zscore_approximately_zero_mean(self, base_panel):
        """Within each date, composite should be approximately zero-mean."""
        result = firm_risk_composite(base_panel)
        valid = result[result["firm_risk_composite_missing"] == 0]
        date_means = valid.groupby(level="date")["firm_risk_composite"].mean()
        np.testing.assert_allclose(date_means.values, 0.0, atol=0.5)


class TestYieldSustainabilityScore:
    @pytest.fixture
    def panel_with_payout(self, base_panel):
        p = fcf_payout_ratio(base_panel)
        p = earnings_payout_ratio(p)
        return p

    def test_output_columns_appended(self, panel_with_payout):
        result = yield_sustainability_score(panel_with_payout)
        assert "yield_sustainability_score" in result.columns
        assert "yield_sustainability_score_missing" in result.columns

    def test_no_rows_dropped(self, panel_with_payout):
        result = yield_sustainability_score(panel_with_payout)
        assert len(result) == len(panel_with_payout)

    def test_requires_payout_columns(self, base_panel):
        with pytest.raises(KeyError):
            yield_sustainability_score(base_panel)  # missing payout columns

    def test_missing_flag_binary(self, panel_with_payout):
        result = yield_sustainability_score(panel_with_payout)
        assert set(result["yield_sustainability_score_missing"].unique()).issubset({0, 1})

    def test_nan_component_propagates_to_missing(self, panel_with_payout):
        # Row with NaN fcf_payout_ratio should produce missing score
        dates = panel_with_payout.index.get_level_values("date").unique().sort_values()
        row = ("AAPL_US", dates[0])
        # fcf_payout_ratio is NaN here (conftest injects NaN cf_free_cash_flow)
        assert pd.isna(panel_with_payout.loc[row, "fcf_payout_ratio"])
        result = yield_sustainability_score(panel_with_payout)
        assert result.loc[row, "yield_sustainability_score_missing"] == 1

    def test_cross_sectional_zscore_approximately_zero_mean(self, panel_with_payout):
        result = yield_sustainability_score(panel_with_payout)
        valid = result[result["yield_sustainability_score_missing"] == 0]
        date_means = valid.groupby(level="date")["yield_sustainability_score"].mean()
        np.testing.assert_allclose(date_means.values, 0.0, atol=0.5)
