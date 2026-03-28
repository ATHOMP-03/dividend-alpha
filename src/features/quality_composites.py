"""
src/features/quality_composites.py
Feature IDs: FE-008 through FE-012

Firm-quality and risk composite features derived from Bloomberg fundamentals.

  FE-008  log_market_cap_usd        Log market cap — RB-001 key predictor
  FE-009  earnings_persistence_score Rolling CoV of net income — RB-002
  FE-010  roe_change_1q_pct          Q-o-Q ROE momentum — RB-002
  FE-011  firm_risk_composite        Cross-sectional beta + leverage z-score — RB-001
  FE-012  yield_sustainability_score FCF payout + earnings payout + debt composite

Cross-sectional z-scores are computed within each quarter-date to avoid any
temporal forward look. All rolling operations use strictly trailing windows.

Bloomberg source fields: CUR_MKT_CAP, NET_INCOME, RETURN_ON_EQ, BETA_RAW_1YR,
SHORT_AND_LONG_TERM_DEBT, CF_FREE_CASH_FLOW, DVD_SH_12M, SH_OUT.

Depends on: FE-001 (fcf_payout_ratio) and FE-002 (earnings_payout_ratio) must
be present in panel before calling yield_sustainability_score.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FE-008
# ---------------------------------------------------------------------------

def log_market_cap(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-008: Natural log of market capitalization (USD millions).

    Per RB-001 (Ivascu 2023), log market cap is the strongest single predictor
    of dividend initiation and continuation. Log-transform compresses the
    scale and reduces skew without additional winsorization.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: cur_mkt_cap.

    Returns
    -------
    pd.DataFrame
        Input panel with log_market_cap_usd and log_market_cap_usd_missing
        appended.
    """
    _check_columns(panel, {"cur_mkt_cap"})
    out = panel.copy()

    # LEAKAGE CHECK: cur_mkt_cap is end-of-quarter market capitalization.
    # Available at time t — no forward-looking data used.
    invalid = out["cur_mkt_cap"].isna() | out["cur_mkt_cap"].le(0)
    out["log_market_cap_usd_missing"] = invalid.astype(int)
    out["log_market_cap_usd"] = np.log(out["cur_mkt_cap"]).where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-009
# ---------------------------------------------------------------------------

def earnings_persistence_score(
    panel: pd.DataFrame, window: int = 8
) -> pd.DataFrame:
    """FE-009: Earnings persistence — rolling coefficient of variation of net income.

    CoV = rolling_std(net_income) / |rolling_mean(net_income)|

    Lower CoV → more persistent earnings → higher dividend sustainability.
    Per RB-002 (Jones 2023): earnings persistence is a strong predictor of
    forward payout capacity.

    Rows where the rolling mean is near zero are flagged missing to avoid
    division instability. Partial windows (fewer than `window` quarters of
    history) are also flagged missing.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: net_income.
    window : int
        Rolling window in quarters (default 8 = 2 years).

    Returns
    -------
    pd.DataFrame
        Input panel with earnings_persistence_score and
        earnings_persistence_score_missing appended.
    """
    _check_columns(panel, {"net_income"})
    out = panel.copy()

    # LEAKAGE CHECK: net_income is from prior quarter filing. Trailing rolling
    # window within each ticker only — no forward-looking data used.
    def _rolling_cov(s: pd.Series) -> pd.Series:
        roll = s.rolling(window=window, min_periods=window)
        std_vals = roll.std()
        mean_abs = roll.mean().abs()
        # Mask near-zero mean to avoid division instability
        safe_mean = mean_abs.where(mean_abs > 1e-6, other=np.nan)
        return std_vals / safe_mean

    raw = (
        out["net_income"]
        .groupby(level="ticker")
        .transform(_rolling_cov)
    )

    invalid = raw.isna()
    out["earnings_persistence_score_missing"] = invalid.astype(int)
    out["earnings_persistence_score"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-010
# ---------------------------------------------------------------------------

def roe_change_1q(panel: pd.DataFrame) -> pd.DataFrame:
    """FE-010: Quarter-over-quarter change in return on equity (first difference).

    Profitability momentum signal. Per RB-002 (Jones 2023), upward ROE momentum
    predicts forward payout capacity.

    The panel must be sorted by date within each ticker before calling.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required column: return_on_eq.

    Returns
    -------
    pd.DataFrame
        Input panel with roe_change_1q_pct and roe_change_1q_pct_missing
        appended.
    """
    _check_columns(panel, {"return_on_eq"})
    out = panel.copy()

    # LEAKAGE CHECK: return_on_eq is from prior quarter filing. shift(1) within
    # each ticker computes the trailing Q-o-Q change — no forward data used.
    lagged = out["return_on_eq"].groupby(level="ticker").shift(1)
    raw = out["return_on_eq"] - lagged

    invalid = raw.isna() | lagged.isna()
    out["roe_change_1q_pct_missing"] = invalid.astype(int)
    out["roe_change_1q_pct"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-011
# ---------------------------------------------------------------------------

def firm_risk_composite(
    panel: pd.DataFrame,
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """FE-011: Firm risk composite — equal-weighted cross-sectional z-score
    of market beta and debt-to-market ratio.

    Higher score = riskier firm. Per RB-001 (Ivascu 2023): lower-risk firms
    are systematically more likely to sustain dividends.

    Construction:
      1. Compute leverage = short_and_long_term_debt / cur_mkt_cap
      2. Winsorize both beta and leverage at [winsorize_pct, 1-winsorize_pct]
         cross-sectionally within each quarter-date
      3. Z-score both cross-sectionally within each quarter-date
      4. firm_risk_composite = (z_beta + z_leverage) / 2

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required columns:
        beta_raw_1yr, short_and_long_term_debt, cur_mkt_cap.
    winsorize_pct : float
        Percentile to winsorize at (default 0.01 = 1st/99th).

    Returns
    -------
    pd.DataFrame
        Input panel with firm_risk_composite and firm_risk_composite_missing
        appended.
    """
    _check_columns(panel, {"beta_raw_1yr", "short_and_long_term_debt", "cur_mkt_cap"})
    out = panel.copy()

    # LEAKAGE CHECK: beta_raw_1yr is trailing 1Y; short_and_long_term_debt and
    # cur_mkt_cap are end-of-quarter values. Z-scores computed cross-sectionally
    # within each quarter-date — no temporal forward look.

    leverage = out["short_and_long_term_debt"] / out["cur_mkt_cap"].where(
        out["cur_mkt_cap"].gt(0), other=np.nan
    )

    def _winsorize_zscore(series: pd.Series) -> pd.Series:
        def _per_date(x: pd.Series) -> pd.Series:
            lo = x.quantile(winsorize_pct)
            hi = x.quantile(1.0 - winsorize_pct)
            clipped = x.clip(lower=lo, upper=hi)
            std = clipped.std(ddof=1)
            if std > 0:
                return (clipped - clipped.mean()) / std
            return pd.Series(0.0, index=x.index)
        return series.groupby(level="date").transform(_per_date)

    z_beta = _winsorize_zscore(out["beta_raw_1yr"])
    z_leverage = _winsorize_zscore(leverage)

    raw = (z_beta + z_leverage) / 2.0

    invalid = (
        out["beta_raw_1yr"].isna()
        | leverage.isna()
        | raw.isna()
    )
    out["firm_risk_composite_missing"] = invalid.astype(int)
    out["firm_risk_composite"] = raw.where(~invalid)

    return out


# ---------------------------------------------------------------------------
# FE-012
# ---------------------------------------------------------------------------

def yield_sustainability_score(
    panel: pd.DataFrame,
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """FE-012: Yield sustainability composite score.

    Equal-weighted composite of three winsorized, cross-sectionally z-scored
    payout/leverage measures. Sign-flipped so that higher score = more
    sustainable yield.

    Components (all sign-flipped — lower is better before flip):
      - fcf_payout_ratio   (FE-001 — must be pre-computed)
      - earnings_payout_ratio  (FE-002 — must be pre-computed)
      - debt_to_mktcap     (short_and_long_term_debt / cur_mkt_cap)

    yield_sustainability_score = -(z_fcf + z_earn + z_debt) / 3

    Requires FE-001 and FE-002 to be present in the panel before calling.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel with (ticker, date) MultiIndex. Required columns:
        fcf_payout_ratio, earnings_payout_ratio,
        short_and_long_term_debt, cur_mkt_cap.
    winsorize_pct : float
        Percentile to winsorize each component at (default 0.01 = 1st/99th).

    Returns
    -------
    pd.DataFrame
        Input panel with yield_sustainability_score and
        yield_sustainability_score_missing appended.
    """
    _check_columns(
        panel,
        {"fcf_payout_ratio", "earnings_payout_ratio",
         "short_and_long_term_debt", "cur_mkt_cap"},
    )
    out = panel.copy()

    # LEAKAGE CHECK: all component features (FE-001, FE-002) are leakage-safe
    # by construction. Cross-sectional z-scoring within each quarter-date — no
    # temporal forward look.

    debt_to_mktcap = out["short_and_long_term_debt"] / out["cur_mkt_cap"].where(
        out["cur_mkt_cap"].gt(0), other=np.nan
    )

    def _winsorize_zscore(series: pd.Series) -> pd.Series:
        def _per_date(x: pd.Series) -> pd.Series:
            lo = x.quantile(winsorize_pct)
            hi = x.quantile(1.0 - winsorize_pct)
            clipped = x.clip(lower=lo, upper=hi)
            std = clipped.std(ddof=1)
            if std > 0:
                return (clipped - clipped.mean()) / std
            return pd.Series(0.0, index=x.index)
        return series.groupby(level="date").transform(_per_date)

    z_fcf = _winsorize_zscore(out["fcf_payout_ratio"])
    z_earn = _winsorize_zscore(out["earnings_payout_ratio"])
    z_debt = _winsorize_zscore(debt_to_mktcap)

    # Sign-flip: lower payout/debt = higher sustainability
    raw = -(z_fcf + z_earn + z_debt) / 3.0

    any_missing = (
        out["fcf_payout_ratio"].isna()
        | out["earnings_payout_ratio"].isna()
        | debt_to_mktcap.isna()
    )
    out["yield_sustainability_score_missing"] = any_missing.astype(int)
    out["yield_sustainability_score"] = raw.where(~any_missing)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_columns(panel: pd.DataFrame, required: set) -> None:
    """Raise KeyError if any required columns are absent from panel."""
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"Panel missing required columns: {sorted(missing)}")
