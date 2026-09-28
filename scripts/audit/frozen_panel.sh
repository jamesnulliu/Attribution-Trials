#!/usr/bin/env bash
# Frozen prompt panel on chess and KT: substrates kt chess x sizes 1.7b 8b.
# Skips a run whose output already exists.
#
#   scripts/audit/frozen_panel.sh
set -euo pipefail
OUT=$(python -c 'from attribution_trials import paths; print(paths.AUDIT / "chess_kt")')

for sub in kt chess; do
  for size in 1.7b 8b; do
    [[ -f "$OUT/frozen_${sub}_${size}.json" ]] && continue
    python -m attribution_trials.audit.frozen_panel "$sub" "$size"
  done
done
