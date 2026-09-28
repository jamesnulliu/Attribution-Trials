#!/usr/bin/env bash
# Dense cross-user donor matrices for the static-embedding family (random
# unrelated profile row). Needs the static-embedding ladder's A4 checkpoints.
set -euo pipefail

python -m attribution_trials.comparison.per_option_cross kt
python -m attribution_trials.comparison.per_option_cross kt-rt
python -m attribution_trials.comparison.per_option_cross chess-pooled
