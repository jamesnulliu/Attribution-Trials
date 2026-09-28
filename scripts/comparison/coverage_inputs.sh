#!/usr/bin/env bash
# Inputs of the population-realism coverage table (comparison.coverage_table):
#   GPU  OPeRA per-option rescoring, Qwen3-8B (2 channels x none/static/dynamic/combined)
#        and Osim-8B (2 channels x none/combined)
#   CPU  chess timing distributions, conditions a0/static/evolving x seeds 0 1 2
# Finished cells are skipped.
#
#   scripts/comparison/coverage_inputs.sh
set -euo pipefail

python -m attribution_trials.comparison.rescore_opera 8b \
  action:none action:static action:dynamic action:combined \
  timing:none timing:static timing:dynamic timing:combined
python -m attribution_trials.comparison.rescore_opera osim-8b \
  action:combined timing:combined action:none timing:none

for seed in 0 1 2; do
  for arm in a0 static evolving; do
    python -m attribution_trials.comparison.rescore_chess "$arm" "$seed"
  done
done
