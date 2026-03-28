"""
src/features/analyst_signals.py
Feature IDs: FE-013 through FE-015

Forward-looking analyst consensus features from Bloomberg BEST_ fields.

Per RB-004 (Bordalo et al. 2024): Expectation-Based Returns (EBRs) — derived
from analyst EPS forecasts — account for most cross-sectional return
predictability. These features are among the most forward-informative signals
in the pipeline.

CRITICAL LEAKAGE NOTE for all three features:
  Bloomberg BEST_ fields must be sourced as-of the quarter-end date using
  historical estimates (BDH with PERIODICITY_OVERRIDE=Q, or equivalent).
  A live pull of BEST_EPS for a historical row would embed analyst revisions
  made after that date — confirmed forward-looking leak. This is a data
  sourcing requirement, not enforced in code.

Bloomberg source fields: BEST_EPS, BEST_LTG_EPS.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# FE-013
# ---------------------------------------------------------------------------

def analyst_eps_consensus(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-013: Analyst consensus EPS forecast (next 4 quarters).

    Per RB-004 (Bordalo et al. 2024): analyst EPS forecasts are the primary
    input for Expectation-Based Returns. EBRs from these forecasts predict
    most cross-sectional return variation.

    Bloomberg field: BEST_EPS (as-of quarter-end from historical estimates).
    Coverage begins ~2002 on Bloomberg — earlier rows will be NaN.
    Missing coverage is informative (smaller/less-covered firms) — do not impute.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: best_eps.

    Returns
    -------
    pd.DataFrame
        Input panel with analyst_eps_consensus and analyst_eps_consensus_missing
        appended.
    """
    _check_columns(panel, {"best_eps"})
    out = panel.copy()

    # LEAKAGE CHECK: best_eps must be sourced as-of quarter-end from Bloomberg
    # historical estimates — not a live pull. See module docstring.
    invalid = out["best_eps"].isna()
    out["analyst_eps_consensus_usd_missing"] = invalid.astype(int)
    out["analyst_eps_consensus_usd"] = out["best_eps"].where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-014
# ---------------------------------------------------------------------------

def eps_forecast_revision_1q(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-014: Quarter-over-quarter change in consensus EPS forecast.

    First difference of analyst consensus EPS across consecutive quarter-end
    vintages. Upward revisions are associated with future outperformance
    per RB-004 (Bordalo et al. 2024).

    Requires two consecutive quarter-end BEST_EPS vintages, each sourced
    as-of their respective quarter-end (see module leakage note).

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: best_eps.

    Returns
    -------
    pd.DataFrame
        Input panel with eps_forecast_revision_1q and
        eps_forecast_revision_1q_missing appended.
    """
    _check_columns(panel, {"best_eps"})
    out = panel.copy()

    # LEAKAGE CHECK: shift(1) within each ticker lags by one quarter.
    # Both the current and lagged best_eps must be sourced as-of their
    # respective quarter-end dates. Never use forward-revised estimates.
    lagged = out["best_eps"].groupby(level="ticker").shift(1)
    raw = out["best_eps"] - lagged

    invalid = raw.isna() | lagged.isna()
    out["eps_forecast_revision_1q_usd_missing"] = invalid.astype(int)
    out["eps_forecast_revision_1q_usd"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-015
# ---------------------------------------------------------------------------

def ltg_analyst_estimate(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-015: Long-term EPS growth estimate from analyst consensus (%).

    Bloomberg field: BEST_LTG_EPS (as-of quarter-end from historical estimates).
    Cross-sectional variation in LTG predicts return spreads per RB-004.

    Coverage may be limited before ~2000. Missing coverage is informative —
    do not impute.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: best_ltg_eps.

    Returns
    -------
    pd.DataFrame
        Input panel with ltg_analyst_estimate_pct and
        ltg_analyst_estimate_pct_missing appended.
    """
    _check_columns(panel, {"best_ltg_eps"})
    out = panel.copy()

    # LEAKAGE CHECK: best_ltg_eps must be sourced as-of quarter-end from
    # Bloomberg historical estimates — not a live pull. See module docstring.
    invalid = out["best_ltg_eps"].isna()
    out["ltg_analyst_estimate_pct_missing"] = invalid.astype(int)
    out["ltg_analyst_estimate_pct"] = out["best_ltg_eps"].where(~invalid)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_columns(panel: pd.DataFrame, required: set) -> None:
    """Raise KeyError if any required columns are absent from panel."""
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Panel missing required columns: {sorted(missing)}")
