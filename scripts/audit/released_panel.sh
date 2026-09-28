#!/usr/bin/env bash
# Released simulators (CoSER-8B, HumanLike-7B) on the OPeRA frozen prompt
# panel, chat format: 2 models x 2 channels x 3 configs.
# Download the pinned weights first: scripts/download/released_simulators.sh
# Skips a run whose output already exists.
#
#   scripts/audit/released_panel.sh
set -euo pipefail
OUT=$(python -c 'from attribution_trials import paths; print(paths.AUDIT / "released")')

for tag in coser-8b humanlike-7b; do
  for ch in action timing; do
    for cfg in combined static dynamic; do
      [[ -f "$OUT/trio-panel_${ch}-${cfg}_${tag}_chat.json" ]] && continue
      python -m attribution_trials.audit.released_panel "opera-$ch-$cfg" "$tag"
    done
  done
done
