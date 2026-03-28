"""
src/features/__init__.py

Public API for the features package.

Feature IDs and their functions:
  FE-001  fcf_payout_ratio            dividend_metrics
  FE-002  earnings_payout_ratio       dividend_metrics
  FE-003  dgr (years=1)               dividend_metrics
  FE-004  dgr (years=3)               dividend_metrics
  FE-005  dgr (years=5)               dividend_metrics
  FE-006  dp_ratio                    yield_features
  FE-007  dy_persistence              yield_features
  FE-008  log_market_cap              quality_composites
  FE-009  earnings_persistence_score  quality_composites
  FE-010  roe_change_1q               quality_composites
  FE-011  firm_risk_composite         quality_composites
  FE-012  yield_sustainability_score  quality_composites
  FE-013  analyst_eps_consensus       analyst_signals
  FE-014  eps_forecast_revision_1q    analyst_signals
  FE-015  ltg_analyst_estimate        analyst_signals

All functions share the same contract:
  - Accept a panel DataFrame with (ticker, date) MultiIndex
  - Return the same panel with new feature column(s) appended
  - Never drop rows
  - Produce a paired _missing binary flag for every feature column
"""

from .analyst_signals import (
    analyst_eps_consensus,
    eps_forecast_revision_1q,
    ltg_analyst_estimate,
)
from .dividend_metrics import dgr, earnings_payout_ratio, fcf_payout_ratio
from .quality_composites import (
    earnings_persistence_score,
    firm_risk_composite,
    log_market_cap,
    roe_change_1q,
    yield_sustainability_score,
)
from .yield_features import dp_ratio, dy_persistence

__all__ = [
    # dividend_metrics
    "fcf_payout_ratio",
    "earnings_payout_ratio",
    "dgr",
    # yield_features
    "dp_ratio",
    "dy_persistence",
    # quality_composites
    "log_market_cap",
    "earnings_persistence_score",
    "roe_change_1q",
    "firm_risk_composite",
    "yield_sustainability_score",
    # analyst_signals
    "analyst_eps_consensus",
    "eps_forecast_revision_1q",
    "ltg_analyst_estimate",
]
