# CLAUDE.md — Dividend Alpha Research Project

## Project overview
Personal research project applying ML-based time-series forecasting to rank
US dividend-producing stocks by expected risk-adjusted return. The goal is to
build a replicable, predictive scoring pipeline — not a trading system.

## Repo structure
```
dividend-alpha/
├── CLAUDE.md
├── .gitignore
├── requirements.txt
├── data/                        # gitignored entirely
│   ├── raw/                     # Bloomberg exports (CSV/XLSX)
│   ├── processed/               # Cleaned, feature-engineered panel (.parquet)
│   ├── universe/                # Stock universe list (mkt cap ranked)
│   └── outputs/                 # Final scorecards, plots
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_scoring_ranking.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── bloomberg/               # Pull scripts and field mappings
│   ├── features/                # Feature construction functions
│   ├── models/                  # Chronos-2 wrapper, NeuralForecast
│   ├── scoring/                 # Composite score + ranking logic
│   ├── evaluation/              # Walk-forward splits, metrics
│   └── utils/                   # I/O, logging, panel helpers
├── tests/
│   ├── test_features/           # Leakage + range checks
│   └── test_scoring/
├── docs/
│   ├── agent_schema.md          # Inter-agent handoff contracts
│   ├── bloomberg_pull_spec.yaml # Field definitions and pull config
│   ├── feature_registry.yaml   # All FE-XXX feature specs
│   └── variable_codebook.md    # Bloomberg field reference
└── references/                  # gitignored entirely — local PDFs + notes
    ├── README.md
    ├── pdfs/
    └── notes/
```

## Primary research question
**Which dividend-producing stocks offer the best risk-adjusted total return
over a 1–3 year horizon, based on dividend growth sustainability, yield
quality, and historical payout resilience?**

## Methodology at a glance
1. Pull ~5–15 years of panel data from Bloomberg for the universe
2. Engineer features: yield, DGR, payout ratio, FCF coverage, quality composites
3. Forecast forward dividend trajectory with Chronos-2 (phase 1: zero-shot + fine-tune)
4. Train from scratch on Bloomberg panel with NeuralForecast (phase 2)
5. Stress-test against dividend cut risk using classification model
6. Combine into a composite rank scorecard

## Key design decisions
- **Unit of analysis**: individual stock × quarter (panel structure)
- **Target variable**: forward 12-month total return (price + dividends)
- **Phase 1 model**: Chronos-2 — zero-shot probabilistic TS, then fine-tuned
- **Phase 2 model**: NeuralForecast — trained from scratch on Bloomberg panel
- **Evaluation**: Walk-forward validation — no look-ahead leakage
- **No live trading**: scoring outputs are analytical only

## Agent system
This project uses four parallel Claude agents with structured JSON/YAML handoffs.
All agent system prompts live in docs/agents/. Handoff schema in docs/agent_schema.md.

| Agent | Role | Output |
|---|---|---|
| Research agent | Academic literature, IVs, methodology | RB-XXX JSON briefs |
| OSS agent | HuggingFace / GitHub model survey | TB-XXX YAML briefs |
| Coding agent | Feature engineering, notebooks → scripts | FE-XXX YAML specs |
| Architecture agent | System integrity, 7-check mandatory review | AR-XXX YAML reviews |

**Tool adoption decisions are made by the researcher only.**
TB-XXX briefs land as `researcher_decision: pending` until explicitly approved.

## Current tool decisions
| Tool | Decision | Phase |
|---|---|---|
| Chronos-2 (amazon/chronos-2) | adopted | Phase 1 — zero-shot + fine-tune |
| NeuralForecast (Nixtla) | adopted | Phase 2 — train from scratch |
| TimesFM 2.5 | deferred | High fit, computationally heavy |
| Moirai 2.0 | deferred | — |
| Prophet | deferred | Baseline reference only |

## What Claude Code should know
- Data is pulled manually from Bloomberg terminal; no live API connection
- All heavy computation runs in Colab Pro (A100/TPU v2 available)
- Local environment used for lightweight scripting and orchestration
- Preferred language: Python 3.11+
- Key libraries: pandas, polars, scikit-learn, statsmodels, chronos-forecasting,
  autogluon.timeseries, neuralforecast, prophet, matplotlib, seaborn,
  openpyxl, pyarrow

## Constraints and non-goals
- This is NOT a backtesting / trading strategy framework
- Do not build an order execution layer
- Prioritize interpretability alongside predictive performance
- Avoid data leakage at all costs — check every feature for forward-looking bias
- References folder is gitignored — never commit PDFs or personal annotations

## Terminology
- **DGR**: Dividend Growth Rate (1Y, 3Y, 5Y annualized)
- **FCF payout ratio**: dividends paid / free cash flow (preferred over earnings payout)
- **Yield sustainability score**: composite of payout ratio, FCF coverage, debt load
- **Cut risk**: probability dividend is reduced or suspended in next 4 quarters
- **Chronos-2**: Amazon's pretrained probabilistic TS foundation model (120M params)
- **NeuralForecast**: Nixtla library — 30+ neural architectures, sklearn-style API
- **Walk-forward validation**: rolling train/test splits with no look-ahead
- **Universe**: Bloomberg-ranked list of US dividend stocks by market cap
- **Panel**: (ticker, quarter_end_date) MultiIndex dataset

## Authorship
Personal project. MS Economics + MS Data Science.
Applying quant skills to dividend equity research as a learning exercise.
