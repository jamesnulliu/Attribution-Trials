#!/usr/bin/env bash
# Per-option rescoring, frozen-prompt Olmo-3.1-32B on the three OPeRA timing
# configurations (outside the 84 combinations).
set -euo pipefail

python -m attribution_trials.comparison.per_option_frozen olmo-32b \
  opera-timing-combined opera-timing-static opera-timing-dynamic --mode fast
