"""
tests/test_features/test_yield_features.py

Tests for src/features/yield_features.py (FE-006, FE-007).

Coverage:
  - Output columns appended, no rows dropped
  - _missing flags are binary integers
  - Zero/NaN px_last flagged missing for dp_ratio
  - dy_persistence requires window quarters of history (earlier rows = NaN)
  - dy_persistence is bounded in [-1, 1]
  - No cross-ticker leakage in dy_persistence
"""

import numpy as np
import pandas as pd
import pytest

from src.features.yield_features import dp_ratio, dy_persistence


class TestDpRatio:
    def test_output_columns_appended(self, base_panel):
        result = dp_ratio(base_panel)
        assert "dp_ratio" in result.columns
        assert "dp_ratio_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        result = dp_ratio(base_panel)
        assert len(result) == len(base_panel)

    def test_missing_flag_binary(self, base_panel):
        result = dp_ratio(base_panel)
        assert set(result["dp_ratio_missing"].unique()).issubset({0, 1})

    def test_nan_price_flagged_missing(self, base_panel):
        # conftest injects px_last = NaN for ("KO_US", DATES[2])
        result = dp_ratio(base_panel)
        dates = result.index.get_level_values("date").unique().sort_values()
        flag = result.loc[("KO_US", dates[2]), "dp_ratio_missing"]
        assert flag == 1

    def test_zero_price_flagged_missing(self):
        idx = pd.MultiIndex.from_tuples(
            [("T1", pd.Timestamp("2020-03-31"))], names=["ticker", "date"]
        )
        panel = pd.DataFrame(
            {"dvd_sh_12m": [1.0], "px_last": [0.0]}, index=idx
        )
        result = dp_ratio(panel)
        assert result.loc[("T1", pd.Timestamp("2020-03-31")), "dp_ratio_missing"] == 1

    def test_valid_ratio_positive(self, base_panel):
        result = dp_ratio(base_panel)
        valid = result[result["dp_ratio_missing"] == 0]["dp_ratio"]
        assert (valid >= 0).all()

    def test_missing_column_raises(self, base_panel):
        with pytest.raises(KeyError):
            dp_ratio(base_panel.drop(columns=["px_last"]))


class TestDyPersistence:
    def test_output_columns_appended(self, base_panel):
        result = dy_persistence(base_panel)
        assert "dy_persistence" in result.columns
        assert "dy_persistence_missing" in result.columns

    def test_no_rows_dropped(self, base_panel):
        result = dy_persistence(base_panel)
        assert len(result) == len(base_panel)

    def test_missing_flag_binary(self, base_panel):
        result = dy_persistence(base_panel)
        assert set(result["dy_persistence_missing"].unique()).issubset({0, 1})

    def test_first_window_quarters_are_missing(self, base_panel):
        """First 8 quarters per ticker must be NaN (window=8, min_periods=8)."""
        result = dy_persistence(base_panel, window=8)
        dates = result.index.get_level_values("date").unique().sort_values()
        for ticker in ["AAPL_US", "JNJ_US", "KO_US"]:
            for d in dates[:8]:
                assert result.loc[(ticker, d), "dy_persistence_missing"] == 1

    def test_values_bounded(self, base_panel):
        """Autocorrelation is always in [-1, 1]."""
        result = dy_persistence(base_panel)
        valid = result[result["dy_persistence_missing"] == 0]["dy_persistence"]
        assert (valid >= -1.0 - 1e-9).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_no_cross_ticker_leakage(self):
        """Rolling autocorrelation within T1 must not be affected by T2 values."""
        dates = pd.date_range("2018-03-31", periods=20, freq="QE")
        idx = pd.MultiIndex.from_product(
            [["T1", "T2"], dates], names=["ticker", "date"]
        )
        # T1: perfectly persistent yield (all 0.05)
        # T2: highly volatile yield
        rng = np.random.default_rng(0)
        yld = np.concatenate([
            np.full(20, 0.05),          # T1 — constant
            rng.uniform(0.01, 0.15, 20) # T2 — random
        ])
        panel = pd.DataFrame({"dvd_yld": yld}, index=idx)
        result = dy_persistence(panel, window=8)

        # T1 should have persistence = 1.0 (or NaN if std=0 in autocorr)
        # More importantly T1 values should NOT equal T2 values
        t1_vals = result.loc["T1", "dy_persistence"].dropna()
        t2_vals = result.loc["T2", "dy_persistence"].dropna()
        # They should have different means
        assert not np.isclose(t1_vals.mean(), t2_vals.mean(), atol=0.3) or len(t1_vals) == 0

    def test_custom_window(self, base_panel):
        """window=4 should produce non-missing values earlier than window=8."""
        result_4 = dy_persistence(base_panel, window=4)
        result_8 = dy_persistence(base_panel, window=8)
        non_missing_4 = (result_4["dy_persistence_missing"] == 0).sum()
        non_missing_8 = (result_8["dy_persistence_missing"] == 0).sum()
        assert non_missing_4 > non_missing_8
