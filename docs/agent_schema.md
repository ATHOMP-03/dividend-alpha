# Agent Handoff Schema — Dividend Alpha Project
# Version: 1.1
# All inter-agent communication uses one of these four document types.
# Researcher is final arbiter on all tool and methodology decisions.

# ─────────────────────────────────────────────────────────────
# 1. RESEARCH BRIEF  (Research Agent → All)
# ─────────────────────────────────────────────────────────────
research_brief:
  research_brief_id: "RB-XXX"        # sequential
  theme: "methodology | predictive_vars | instrumental_vars | ml_approaches"
  paper:
    title: string
    authors: [string]
    year: int
    source: "NBER | SSRN | Journal | Training"
    url_or_doi: "string | null"
    access: "open | paywalled | local | training_knowledge"
    local_path: "references/pdfs/RB-XXX_author_year.pdf | null"
  key_finding: string                 # 2–3 sentences
  operationalizable_variables:
    - variable_name: string
      definition: string
      bloomberg_field_candidate: "string | null"
      derivation_notes: string
  iv_candidates:
    - instrument: string
      mechanism: string
      relevance_to_project: string
  methodological_notes: string
  limitations: string
  priority: "high | medium | low"
  flag: "string | null"               # PAYWALLED, NEEDS VERIFICATION, SEMINAL, OPEN ACCESS

# ─────────────────────────────────────────────────────────────
# 2. TOOL BRIEF  (OSS Agent → Coding + Architecture Agents)
# ─────────────────────────────────────────────────────────────
tool_brief:
  tool_brief_id: "TB-XXX"
  name: string
  type: "forecasting_model | classification_model | library | dataset | eval_framework"
  source:
    platform: "HuggingFace | GitHub | PyPI | Other"
    url: string
    model_id: "string | null"
    last_updated: "YYYY-MM | unknown"
    stars: "int | unknown"
    license: string
  architecture: string
  pretraining_data: string
  probabilistic_output: bool
  horizon_fit:
    min_steps: int
    max_steps: int
    suitable_for_project: bool
    notes: string
  covariate_support: "none | exogenous_only | multivariate"
  fine_tuning_support: bool
  colab_compatibility:
    estimated_vram_gb: "float | unknown"
    a100_compatible: bool
    notes: string
  use_case_fit:
    dividend_series_forecasting: "strong | moderate | weak | unknown"
    cut_risk_classification: "strong | moderate | weak | unknown"
    macro_covariate_integration: "strong | moderate | weak | unknown"
  recommendation: "evaluate_first | worth_testing | low_priority | not_suitable"
  recommendation_rationale: string
  integration_notes: string
  flags: [string]
  informed_agents: [string]           # coding | architecture
  researcher_decision: "pending | adopted | rejected | deferred"

# ─────────────────────────────────────────────────────────────
# 3. FEATURE SPEC  (Coding Agent → Architecture Agent)
# ─────────────────────────────────────────────────────────────
feature_spec:
  feature_id: "FE-XXX"               # sequential
  name: string                        # snake_case
  description: string
  source_research_brief: "RB-XXX | null"
  source_fields:
    - bloomberg: string
  derivation: string                  # formula or plain English
  frequency: "daily | monthly | quarterly | annual"
  leakage_safe: bool
  leakage_notes: string
  expected_range: [float, float]
  outlier_treatment: string
  missing_strategy: string
  notebook: string                    # path e.g. notebooks/02_feature_engineering.ipynb
  script: string                      # path e.g. src/features/payout_ratios.py
  validated: bool

# ─────────────────────────────────────────────────────────────
# 4. ARCH REVIEW  (Architecture Agent → Researcher)
# ─────────────────────────────────────────────────────────────

# --- Feature review ---
arch_review_feature:
  arch_review_id: "AR-XXX"
  type: feature_review
  target_feature: "FE-XXX"
  verdict: "approved | approved_with_notes | needs_revision | blocked"
  checks_completed:                   # ALL seven required — no exceptions
    leakage_audit: "complete | incomplete — [reason]"
    bloomberg_consistency: "complete | incomplete — [reason]"
    schema_conventions: "complete | incomplete — [reason]"
    redundancy_check: "complete | incomplete — [reason]"
    outlier_missing_strategy: "complete | incomplete — [reason]"
    unit_test_coverage: "complete | incomplete — [reason]"
    traceability: "complete | incomplete — [reason]"
  issues:
    - severity: "warning | error"
      check: string                   # which of the 7 checks
      description: string
      suggested_fix: string
  leakage_assessment: "clean | suspect | confirmed_leak"
  cross_feature_notes: string
  technical_debt_items:
    - id: "TD-XXX"
      description: string
      introduced_at: string
      priority: "low | medium | high"
      resolution_plan: string

# --- Milestone review ---
arch_review_milestone:
  arch_review_id: "AR-MXX"
  type: milestone_review
  milestone: string
  checks_completed:                   # ALL twelve required — no exceptions
    leakage_audit: "complete | incomplete — [reason]"
    bloomberg_consistency: "complete | incomplete — [reason]"
    schema_conventions: "complete | incomplete — [reason]"
    redundancy_check: "complete | incomplete — [reason]"
    outlier_missing_strategy: "complete | incomplete — [reason]"
    unit_test_coverage: "complete | incomplete — [reason]"
    traceability: "complete | incomplete — [reason]"
    target_lag_verified: "complete | incomplete — [reason]"
    walk_forward_boundaries: "complete | incomplete — [reason]"
    panel_balance_documented: "complete | incomplete — [reason]"
    feature_registry_complete: "complete | incomplete — [reason]"
    technical_debt_current: "complete | incomplete — [reason]"
  integration_checklist:
    panel_balanced: "true | false | partial"
    leakage_safe: "true | false | partial"
    target_lagged_correctly: bool
    walk_forward_boundaries_defined: bool
    unit_tests_present: bool
    bloomberg_coverage_verified: bool
  blocking_issues: [string]
  warnings: [string]
  approved_to_proceed: bool           # false if ANY blocking issue exists
  technical_debt_log:
    - id: "TD-XXX"
      description: string
      introduced_at: string
      priority: "low | medium | high"
      resolution_plan: string

# ─────────────────────────────────────────────────────────────
# VERDICT RULES (architecture agent — non-negotiable)
# ─────────────────────────────────────────────────────────────
# confirmed_leak          → blocked
# suspect leakage         → needs_revision (cannot proceed)
# any incomplete check    → needs_revision
# any unresolved blocker  → blocked
# all checks pass, notes  → approved_with_notes
# all checks pass, clean  → approved
#
# approved_with_notes does NOT mean proceed-and-fix-later
# for leakage or schema issues. Notes must be resolved
# before model training begins.
