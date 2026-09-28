#!/usr/bin/env bash
# CPU steps of the existing-benchmark comparison and the per-user baselines,
# in dependency order. Requires the per-option GPU outputs (per_option_*.sh),
# the coverage-table inputs (coverage_inputs.sh), the audit results, and
# <ANALYSIS>/gains.json (attribution_trials.analysis.gains).
set -euo pipefail

# Existing-benchmark comparison (tab:comparison-upstream) and the Olmo counts
python -m attribution_trials.comparison.accuracy
python -m attribution_trials.comparison.total_variation
python -m attribution_trials.comparison.population_js
python -m attribution_trials.comparison.comparison_rows
python -m attribution_trials.comparison.coverage_table
python -m attribution_trials.tables.comparison_upstream

# Chess prompt-cap check
python -m attribution_trials.comparison.chess_prompt_cap

# Per-user marginal baseline (tab:marginal)
python -m attribution_trials.comparison.marginal_timing
python -m attribution_trials.comparison.marginal_discrete
python -m attribution_trials.comparison.marginal_controls
python -m attribution_trials.tables.marginal_share

# Stronger baselines (tab:strongbase)
python -m attribution_trials.comparison.stronger_baselines
python -m attribution_trials.tables.stronger_baselines
