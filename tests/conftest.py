"""
tests/conftest.py

Shared pytest fixtures for dividend-alpha feature tests.

Synthetic panel: 3 tickers × 24 quarters (2018-Q1 through 2023-Q4).
All Bloomberg-equivalent columns are populated with plausible values.
Missing values are injected deliberately to verify _missing flag logic.
"""

import numpy as np
import pandas as pd
import pytest

# ── panel dimensions ────────────────────────────────────────────────────────
TICKERS = ["AAPL_US", "JNJ_US", "KO_US"]
DATES = pd.date_range("2018-03-31", periods=24, freq="QE")


@pytest.fixture(scope="session")
def base_panel() -> pd.DataFrame:
    """Minimal panel with all Bloomberg source columns populated.

    Values are deterministic and chosen so that:
      - fcf_payout_ratio lands in [0, 1] for most rows
      - DGR calculations are computable from Q5 onward (for 1Y)
      - A small number of rows have NaN injected to test missing flags
    """
    rng = np.random.default_rng(42)
    n_tickers = len(TICKERS)
    n_dates = len(DATES)
    n = n_tickers * n_dates

    index = pd.MultiIndex.from_product(
        [TICKERS, DATES], names=["ticker", "date"]
    )

    # Dividend series — slowly growing, positive throughout
    dvd_base = np.tile(
        np.linspace(1.0, 1.5, n_dates), n_tickers
    ) + rng.normal(0, 0.02, n)
    dvd_base = np.clip(dvd_base, 0.5, 3.0)

    data = {
        "dvd_sh_12m": dvd_base,
        "sh_out": rng.uniform(1e9, 5e9, n),                  # shares outstanding
        "cf_free_cash_flow": rng.uniform(5e8, 5e9, n),        # FCF — all positive
        "net_income": rng.uniform(3e8, 4e9, n),               # NI — all positive
        "px_last": rng.uniform(50.0, 200.0, n),               # price
        "dvd_yld": dvd_base / rng.uniform(50.0, 200.0, n),    # ~yield
        "cur_mkt_cap": rng.uniform(5e10, 5e11, n),            # market cap
        "return_on_eq": rng.uniform(0.05, 0.35, n),           # ROE
        "beta_raw_1yr": rng.uniform(0.4, 1.6, n),             # beta
        "short_and_long_term_debt": rng.uniform(1e9, 3e10, n),# debt
        "best_eps": rng.uniform(1.0, 10.0, n),                # EPS consensus
        "best_ltg_eps": rng.uniform(0.03, 0.15, n),           # LTG estimate
    }

    panel = pd.DataFrame(data, index=index)

    # Inject a small number of NaNs to exercise missing flag logic
    panel.loc[("AAPL_US", DATES[0]), "cf_free_cash_flow"] = np.nan
    panel.loc[("JNJ_US", DATES[1]), "net_income"] = -1e8       # negative NI
    panel.loc[("KO_US", DATES[2]), "px_last"] = np.nan
    panel.loc[("AAPL_US", DATES[3]), "best_eps"] = np.nan

    return panel
