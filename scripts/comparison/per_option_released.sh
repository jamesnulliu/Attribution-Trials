#!/usr/bin/env bash
# Per-option rescoring, released simulators: Osim (4 checkpoints) and the
# Sim2Real trio (2 checkpoints), six OPeRA configurations each. Each
# checkpoint is loaded once for all six configurations.
set -euo pipefail

SUBS="opera-action-combined opera-action-static opera-action-dynamic
      opera-timing-combined opera-timing-static opera-timing-dynamic"

for tag in osim-8b osim-8b-mid coser-8b humanlike-7b osim-4b osim-4b-mid; do
  # shellcheck disable=SC2086
  python -m attribution_trials.comparison.per_option_released "$tag" $SUBS --mode fast
done
