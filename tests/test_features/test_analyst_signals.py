"""
tests/test_features/test_analyst_signals.py

Tests for src/features/analyst_signals.py (FE-013 to FE-015).

Coverage:
  - Output columns appended, no rows dropped
  - _missing flags are binary integers
  - NaN best_eps rows are flagged missing for FE-013 and FE-014
  - FE-014 first quarter per ticker is missing (no lag available)
  - No cross-ticker leakage in FE-014 revision calculation
  - FE-015 NaN best_ltg_eps rows are flagged missing
"""

import numpy as np
import pandas as pd
import pytest

from src.features.analyst_signals import (
    analyst_eps_consensus,
    eps_forecast_revision_1q,
    ltg_analyst_estimate,
)


class TestAnalystEpsConsensus:
    def test_output_columns_appended(self, base_panel):
        result = analyst_eps_consensus(base_panel)
        assert "analyst_eps_consensus_usd" in result.columns
        assert "analyst_eps_consensus_usd_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        assert len(analyst_eps_consensus(base_panel)) == len(base_panel)

    def test_missing_flag_binary(self, base_panel):
        result = analyst_eps_consensus(base_panel)
        assert set(result["analyst_eps_consensus_usd_missing"].unique()).issubset({0, 1})

    def test_nan_best_eps_flagged_missing(self, base_panel):
        # conftest injects best_eps = NaN for ("AAPL_US", DATES[3])
        result = analyst_eps_consensus(base_panel)
        dates = result.index.get_level_values("date").unique().sort_values()
        flag = result.loc[("AAPL_US", dates[3]), "analyst_eps_consensus_usd_missing"]
        assert flag == 1

    def test_valid_rows_pass_through_unchanged(self, base_panel):
        result = analyst_eps_consensus(base_panel)
        valid = result[result["analyst_eps_consensus_usd_missing"] == 0]
        # Should match source column exactly for non-NaN rows
        pd.testing.assert_series_equal(
            valid["analyst_eps_consensus_usd"],
            valid["best_eps"].rename("analyst_eps_consensus_usd"),
        )

    def test_missing_column_raises(self, base_panel):
        with pytest.raises(KeyError):
            analyst_eps_consensus(base_panel.drop(columns=["best_eps"]))


class TestEpsForecastRevision1q:
    def test_output_columns_appended(self, base_panel):
        result = eps_forecast_revision_1q(base_panel)
        assert "eps_forecast_revision_1q_usd" in result.columns
        assert "eps_forecast_revision_1q_usd_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        assert len(eps_forecast_revision_1q(base_panel)) == len(base_panel)

    def test_missing_flag_binary(self, base_panel):
        result = eps_forecast_revision_1q(base_panel)
        assert set(result["eps_forecast_revision_1q_usd_missing"].unique()).issubset({0, 1})

    def test_first_quarter_per_ticker_missing(self, base_panel):
        result = eps_forecast_revision_1q(base_panel)
        dates = result.index.get_level_values("date").unique().sort_values()
        first_date = dates[0]
        for ticker in ["AAPL_US", "JNJ_US", "KO_US"]:
            assert result.loc[(ticker, first_date), "eps_forecast_revision_1q_usd_missing"] == 1

    def test_no_cross_ticker_leakage(self):
        """Revision for T2 must not pick up T1 lagged values."""
        dates = pd.date_range("2020-03-31", periods=4, freq="QE")
        idx = pd.MultiIndex.from_product(
            [["T1", "T2"], dates], names=["ticker", "date"]
        )
        # T1 EPS growing; T2 EPS flat at 5.0
        eps = [1.0, 2.0, 3.0, 4.0,   # T1
               5.0, 5.0, 5.0, 5.0]   # T2
        panel = pd.DataFrame({"best_eps": eps}, index=idx)
        result = eps_forecast_revision_1q(panel)

        t2_revisions = result.loc["T2", "eps_forecast_revision_1q_usd"].dropna()
        np.testing.assert_allclose(t2_revisions.values, 0.0, atol=1e-10)

    def test_nan_current_eps_flagged_missing(self, base_panel):
        # conftest injects best_eps = NaN for ("AAPL_US", DATES[3])
        result = eps_forecast_revision_1q(base_panel)
        dates = result.index.get_level_values("date").unique().sort_values()
        assert result.loc[("AAPL_US", dates[3]), "eps_forecast_revision_1q_usd_missing"] == 1


class TestLtgAnalystEstimate:
    def test_output_columns_appended(self, base_panel):
        result = ltg_analyst_estimate(base_panel)
        assert "ltg_analyst_estimate_pct" in result.columns
        assert "ltg_analyst_estimate_pct_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        assert len(ltg_analyst_estimate(base_panel)) == len(base_panel)

    def test_missing_flag_binary(self, base_panel):
        result = ltg_analyst_estimate(base_panel)
        assert set(result["ltg_analyst_estimate_pct_missing"].unique()).issubset({0, 1})

    def test_no_missing_when_all_populated(self, base_panel):
        # base_panel has no NaN best_ltg_eps
        result = ltg_analyst_estimate(base_panel)
        assert result["ltg_analyst_estimate_pct_missing"].sum() == 0

    def test_nan_ltg_flagged_missing(self):
        idx = pd.MultiIndex.from_tuples(
            [("T1", pd.Timestamp("2020-03-31"))], names=["ticker", "date"]
        )
        panel = pd.DataFrame({"best_ltg_eps": [np.nan]}, index=idx)
        result = ltg_analyst_estimate(panel)
        assert result.loc[("T1", pd.Timestamp("2020-03-31")), "ltg_analyst_estimate_pct_missing"] == 1

    def test_valid_values_pass_through(self, base_panel):
        result = ltg_analyst_estimate(base_panel)
        valid = result[result["ltg_analyst_estimate_pct_missing"] == 0]
        pd.testing.assert_series_equal(
            valid["ltg_analyst_estimate_pct"],
            valid["best_ltg_eps"].rename("ltg_analyst_estimate_pct"),
        )

    def test_missing_column_raises(self, base_panel):
        with pytest.raises(KeyError):
            ltg_analyst_estimate(base_panel.drop(columns=["best_ltg_eps"]))
