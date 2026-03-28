"""
src/features/dividend_metrics.py
Feature IDs: FE-001 through FE-005

Core dividend-based features: FCF payout ratio, earnings payout ratio,
and dividend growth rates at 1Y, 3Y, and 5Y horizons.

All features are derived from trailing/historical data only — leakage-safe
by construction. All functions accept and return a panel DataFrame with a
(ticker, date) MultiIndex where date is the quarter-end.

Bloomberg source fields: DVD_SH_12M, CF_FREE_CASH_FLOW, SH_OUT, NET_INCOME.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FE-001
# ---------------------------------------------------------------------------

def fcf_payout_ratio(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-001: FCF payout ratio — total dividends paid / free cash flow.

    Derivation: (dvd_sh_12m * sh_out) / cf_free_cash_flow

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required columns:
        dvd_sh_12m, sh_out, cf_free_cash_flow.

    Returns
    -------
    pd.DataFrame
        Input panel with fcf_payout_ratio and fcf_payout_ratio_missing appended.
    """
    _check_columns(panel, {"dvd_sh_12m", "sh_out", "cf_free_cash_flow"})
    out = panel.copy()

    # LEAKAGE CHECK: dvd_sh_12m is trailing 12M dividends per share reported in
    # prior quarter filing. cf_free_cash_flow is trailing 12M FCF from prior
    # quarter filing. Both are available at time t without forward information.
    total_divs = out["dvd_sh_12m"] * out["sh_out"]
    raw = total_divs / out["cf_free_cash_flow"]

    invalid = (
        raw.isna()
        | out["cf_free_cash_flow"].isna()
        | out["cf_free_cash_flow"].eq(0)
    )
    out["fcf_payout_ratio_missing"] = invalid.astype(int)
    out["fcf_payout_ratio"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-002
# ---------------------------------------------------------------------------

def earnings_payout_ratio(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-002: Earnings payout ratio — total dividends paid / net income.

    Derivation: (dvd_sh_12m * sh_out) / net_income

    Negative net income rows are flagged missing — payout on a loss is
    not economically meaningful and will contaminate the feature.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required columns:
        dvd_sh_12m, sh_out, net_income.

    Returns
    -------
    pd.DataFrame
        Input panel with earnings_payout_ratio and earnings_payout_ratio_missing
        appended.
    """
    _check_columns(panel, {"dvd_sh_12m", "sh_out", "net_income"})
    out = panel.copy()

    # LEAKAGE CHECK: net_income is from prior quarter filing. dvd_sh_12m is
    # trailing 12M. Both available at time t.
    total_divs = out["dvd_sh_12m"] * out["sh_out"]
    raw = total_divs / out["net_income"]

    invalid = (
        raw.isna()
        | out["net_income"].isna()
        | out["net_income"].le(0)  # negative or zero net income — flag missing
    )
    out["earnings_payout_ratio_missing"] = invalid.astype(int)
    out["earnings_payout_ratio"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-003 / FE-004 / FE-005
# ---------------------------------------------------------------------------

def dgr(panel: pd.DataFrame, years: int) -> pd.DataFrame:
    """FE-003/FE-004/FE-005: Dividend growth rate (CAGR) over N years.

    For years=1: simple growth rate — (dvd_t / dvd_{t-4}) - 1.
    For years>1: CAGR — (dvd_t / dvd_{t - years*4})^(1/years) - 1.

    The panel must be sorted by date within each ticker before calling this
    function. Shift is applied within-ticker to prevent cross-ticker leakage.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: dvd_sh_12m.
    years : int
        Lookback horizon in years. Must be 1, 3, or 5.

    Returns
    -------
    pd.DataFrame
        Input panel with dgr_{years}y and dgr_{years}y_missing appended.

    Raises
    ------
    ValueError
        If years is not 1, 3, or 5.
    """
    if years not in (1, 3, 5):
        raise ValueError(f"years must be 1, 3, or 5; got {years}")

    _check_columns(panel, {"dvd_sh_12m"})
    out = panel.copy()

    lag = years * 4
    col = f"dgr_{years}y_pct"

    # LEAKAGE CHECK: dvd_sh_12m is trailing 12M as reported in prior quarter
    # filing. shift(lag) is applied within-ticker — no cross-ticker contamination,
    # no forward-looking data.
    dvd = out["dvd_sh_12m"]
    dvd_lagged = dvd.groupby(level="ticker").shift(lag)

    if years == 1:
        raw = (dvd / dvd_lagged) - 1.0
    else:
        raw = (dvd / dvd_lagged) ** (1.0 / years) - 1.0

    invalid = (
        dvd_lagged.isna()
        | dvd.isna()
        | dvd_lagged.le(0)
        | dvd.le(0)
    )
    out[f"{col}_missing"] = invalid.astype(int)
    out[col] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_columns(panel: pd.DataFrame, required: set) -> None:
    """Raise KeyError if any required columns are absent from panel."""
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Panel missing required columns: {sorted(missing)}")
