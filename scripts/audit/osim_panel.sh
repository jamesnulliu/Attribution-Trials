#!/usr/bin/env bash
# Osim released simulators on the OPeRA frozen prompt panel:
# 4 checkpoints x 2 channels x 3 configs.
# Skips a run whose output already exists.
#
#   scripts/audit/osim_panel.sh
set -euo pipefail
OUT=$(python -c 'from attribution_trials import paths; print(paths.AUDIT / "released")')

for tag in osim-4b osim-4b-mid osim-8b osim-8b-mid; do
  for ch in action timing; do
    for cfg in combined static dynamic; do
      [[ -f "$OUT/osim-panel_${ch}-${cfg}_${tag}.json" ]] && continue
      python -m attribution_trials.audit.osim_panel "opera-$ch-$cfg" "$tag"
    done
  done
done
