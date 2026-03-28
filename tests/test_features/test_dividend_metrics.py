"""
tests/test_features/test_dividend_metrics.py

Tests for src/features/dividend_metrics.py (FE-001 to FE-005).

Coverage:
  - Output columns are appended without dropping existing columns
  - _missing flags are binary integers (0/1)
  - Missing flags are set where denominators are zero or NaN
  - Negative net income rows are flagged missing in earnings_payout_ratio
  - DGR values are NaN for the first lag-N quarters (insufficient history)
  - dgr() raises ValueError for unsupported years argument
  - No original rows are dropped (panel length preserved)
"""

import numpy as np
import pandas as pd
import pytest

from src.features.dividend_metrics import (
    dgr,
    earnings_payout_ratio,
    fcf_payout_ratio,
)


class TestFcfPayoutRatio:
    def test_output_columns_appended(self, base_panel):
        result = fcf_payout_ratio(base_panel)
        assert "fcf_payout_ratio" in result.columns
        assert "fcf_payout_ratio_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        result = fcf_payout_ratio(base_panel)
        assert len(result) == len(base_panel)

    def test_missing_flag_is_binary_int(self, base_panel):
        result = fcf_payout_ratio(base_panel)
        vals = result["fcf_payout_ratio_missing"].unique()
        assert set(vals).issubset({0, 1})

    def test_nan_fcf_flagged_missing(self, base_panel):
        # conftest injects NaN cf_free_cash_flow for ("AAPL_US", DATES[0])
        result = fcf_payout_ratio(base_panel)
        flag = result.loc[("AAPL_US", result.index.get_level_values("date")[0]),
                          "fcf_payout_ratio_missing"]
        assert flag == 1

    def test_nan_fcf_produces_nan_value(self, base_panel):
        result = fcf_payout_ratio(base_panel)
        val = result.loc[("AAPL_US", result.index.get_level_values("date")[0]),
                         "fcf_payout_ratio"]
        assert pd.isna(val)

    def test_zero_fcf_flagged_missing(self):
        idx = pd.MultiIndex.from_tuples(
            [("T1", pd.Timestamp("2020-03-31"))], names=["ticker", "date"]
        )
        panel = pd.DataFrame(
            {"dvd_sh_12m": [1.0], "sh_out": [1e9], "cf_free_cash_flow": [0.0]},
            index=idx,
        )
        result = fcf_payout_ratio(panel)
        assert result.loc[("T1", pd.Timestamp("2020-03-31")), "fcf_payout_ratio_missing"] == 1

    def test_valid_rows_not_flagged(self, base_panel):
        result = fcf_payout_ratio(base_panel)
        # Most rows should be valid — non-missing count should be large
        assert result["fcf_payout_ratio_missing"].sum() < len(result) * 0.1

    def test_missing_column_raises(self, base_panel):
        with pytest.raises(KeyError):
            fcf_payout_ratio(base_panel.drop(columns=["cf_free_cash_flow"]))


class TestEarningsPayoutRatio:
    def test_output_columns_appended(self, base_panel):
        result = earnings_payout_ratio(base_panel)
        assert "earnings_payout_ratio" in result.columns
        assert "earnings_payout_ratio_missing" in result.columns

    def test_negative_net_income_flagged_missing(self, base_panel):
        # conftest injects net_income = -1e8 for ("JNJ_US", DATES[1])
        result = earnings_payout_ratio(base_panel)
        dates = result.index.get_level_values("date").unique().sort_values()
        flag = result.loc[("JNJ_US", dates[1]), "earnings_payout_ratio_missing"]
        assert flag == 1

    def test_no_rows_dropped(self, base_panel):
        result = earnings_payout_ratio(base_panel)
        assert len(result) == len(base_panel)

    def test_valid_ratio_positive(self, base_panel):
        result = earnings_payout_ratio(base_panel)
        valid = result[result["earnings_payout_ratio_missing"] == 0]
        assert (valid["earnings_payout_ratio"] >= 0).all()


class TestDgr:
    @pytest.mark.parametrize("years", [1, 3, 5])
    def test_output_columns_appended(self, base_panel, years):
        result = dgr(base_panel, years=years)
        assert f"dgr_{years}y_pct" in result.columns
        assert f"dgr_{years}y_missing" in result.columns

    @pytest.mark.parametrize("years", [1, 3, 5])
    def test_no_rows_dropped(self, base_panel, years):
        result = dgr(base_panel, years=years)
        assert len(result) == len(base_panel)

    def test_dgr_1y_first_four_quarters_missing_per_ticker(self, base_panel):
        result = dgr(base_panel, years=1)
        dates = result.index.get_level_values("date").unique().sort_values()
        for ticker in ["AAPL_US", "JNJ_US", "KO_US"]:
            for d in dates[:4]:
                assert result.loc[(ticker, d), "dgr_1y_pct_missing"] == 1

    def test_dgr_1y_computed_after_lag(self, base_panel):
        result = dgr(base_panel, years=1)
        dates = result.index.get_level_values("date").unique().sort_values()
        # From Q5 onward most rows should be non-missing
        later_rows = result.xs(slice(dates[4], None), level="date", drop_level=False)
        assert later_rows["dgr_1y_pct_missing"].mean() < 0.1

    def test_dgr_invalid_years_raises(self, base_panel):
        with pytest.raises(ValueError, match="years must be 1, 3, or 5"):
            dgr(base_panel, years=2)

    def test_dgr_missing_column_raises(self, base_panel):
        with pytest.raises(KeyError):
            dgr(base_panel.drop(columns=["dvd_sh_12m"]), years=1)

    def test_no_cross_ticker_leakage(self):
        """Shift within ticker must not bleed across tickers on the same date."""
        dates = pd.date_range("2020-03-31", periods=8, freq="QE")
        idx = pd.MultiIndex.from_product(
            [["T1", "T2"], dates], names=["ticker", "date"]
        )
        # T1 grows; T2 is flat
        dvd = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,  # T1
               2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # T2
        panel = pd.DataFrame({"dvd_sh_12m": dvd}, index=idx)
        result = dgr(panel, years=1)

        # T2 dgr_1y should be 0.0, not influenced by T1 values
        t2_dgr = result.loc["T2", "dgr_1y_pct"].dropna()
        np.testing.assert_allclose(t2_dgr.values, 0.0, atol=1e-10)
