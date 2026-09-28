#!/usr/bin/env bash
# Per-option rescoring, frozen-prompt family: the 16 frozen-prompt
# combinations, plus the two chess combinations rescored at a prompt cap
# that truncates nothing (chess_prompt_cap).
set -euo pipefail

SUBS="kt chess
      opera-action-combined opera-action-static opera-action-dynamic
      opera-timing-combined opera-timing-static opera-timing-dynamic"

for size in 8b 1.7b; do
  # shellcheck disable=SC2086
  python -m attribution_trials.comparison.per_option_frozen "$size" $SUBS --mode fast
done

for size in 8b 1.7b; do
  python -m attribution_trials.comparison.per_option_frozen "$size" chess \
    --cap 1024 --suffix _cap1024 --mode fast
done
