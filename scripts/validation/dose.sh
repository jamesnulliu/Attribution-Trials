#!/usr/bin/env bash
# Target-information dose ladder, frozen prompt Qwen3-8B:
#   substrates kt chess x (2 endpoint passes + alpha 0.25 0.5 0.75 x seeds 0 1 2).
# The model (~16 GB) is loaded once per substrate; finished mixtures are skipped.
# Then the readout.
#
#   scripts/validation/dose.sh
set -euo pipefail
OUT=$(python -c 'from attribution_trials import paths; print(paths.VALIDATION / "dose")')
mkdir -p "$OUT/logs"

for sub in kt chess; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start dose $sub" >&2
  python -m attribution_trials.validation.dose "$sub" >> "$OUT/logs/dose_${sub}.log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) done  dose $sub" >&2
done
python -m attribution_trials.validation.dose_readout
