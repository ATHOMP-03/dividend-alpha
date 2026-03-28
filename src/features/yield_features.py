"""
src/features/yield_features.py
Feature IDs: FE-006, FE-007

D/P ratio and dividend yield persistence.

IMPORTANT — Goyal-Welch warning (RB-005): raw D/P ratio has demonstrated
out-of-sample predictive failure at the aggregate level. Both features here
are included for benchmarking and diagnostic purposes. Do not treat dp_ratio
as a primary predictor without strong out-of-sample validation.

Bloomberg source fields: DVD_SH_12M, PX_LAST, DVD_YLD.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# FE-006
# ---------------------------------------------------------------------------

def dp_ratio(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-006: Dividend-price ratio (trailing 12M dividends / price).

    Derivation: dvd_sh_12m / px_last

    NOTE: Per RB-005 (Goyal & Welch 2003), this feature has demonstrated
    out-of-sample predictive failure at the aggregate level. Include for
    benchmarking only — do not weight heavily in the composite score without
    strong out-of-sample validation evidence.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required columns:
        dvd_sh_12m, px_last.

    Returns
    -------
    pd.DataFrame
        Input panel with dp_ratio and dp_ratio_missing appended.
    """
    _check_columns(panel, {"dvd_sh_12m", "px_last"})
    out = panel.copy()

    # LEAKAGE CHECK: dvd_sh_12m is trailing 12M dividends per share, available
    # at time t. px_last is the end-of-quarter closing price, available at time t.
    # No forward-looking data used.
    raw = out["dvd_sh_12m"] / out["px_last"]

    invalid = (
        raw.isna()
        | out["px_last"].isna()
        | out["px_last"].le(0)
    )
    out["dp_ratio_missing"] = invalid.astype(int)
    out["dp_ratio"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-007
# ---------------------------------------------------------------------------

def dy_persistence(panel: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """FE-007: Dividend yield persistence — rolling lag-1 autocorrelation.

    Computes the lag-1 autocorrelation of dvd_yld over a trailing `window`-
    quarter rolling window within each ticker.

    High persistence (near 1.0) is a Goyal-Welch warning signal (RB-005):
    a stock whose yield autocorrelation is near 1 is more susceptible to the
    D/P predictability failure mode. Can be used as a feature quality filter
    or weight dampener in the composite score.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: dvd_yld.
    window : int
        Rolling window in quarters (default 8 = 2 years). Minimum periods
        is set equal to window — partial windows produce NaN.

    Returns
    -------
    pd.DataFrame
        Input panel with dy_persistence and dy_persistence_missing appended.
    """
    _check_columns(panel, {"dvd_yld"})
    out = panel.copy()

    # LEAKAGE CHECK: rolling autocorrelation is computed over a strictly trailing
    # window within each ticker. No forward-looking data used. Requires window
    # quarters of history — earlier rows are NaN and flagged missing.
    def _rolling_autocorr(s: pd.Series) -> pd.Series:
        return s.rolling(window=window, min_periods=window).apply(
            lambda x: x.autocorr(lag=1), raw=False
        )

    raw = (
        out["dvd_yld"]
        .groupby(level="ticker")
        .transform(_rolling_autocorr)
    )

    invalid = raw.isna()
    out["dy_persistence_missing"] = invalid.astype(int)
    out["dy_persistence"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_columns(panel: pd.DataFrame, required: set) -> None:
    """Raise KeyError if any required columns are absent from panel."""
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Panel missing required columns: {sorted(missing)}")
