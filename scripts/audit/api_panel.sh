#!/usr/bin/env bash
# OPeRA API panel: 3 models x 2 channels x 3 configs (18 files).
# N=300 eval decisions per cell, except the GPT-4.1 / GPT-4.1-mini timing
# cells, which use N=1500.
# Needs OPENAI_API_KEY (and OPENAI_BASE_URL for a non-OpenAI endpoint).
# Sequential; skips a cell whose output already exists.
#
#   scripts/audit/api_panel.sh
set -euo pipefail
OUT=$(python -c 'from attribution_trials import paths; print(paths.AUDIT / "opera")')
tag() { echo "$1" | sed 's#/#-#g; s#\.#-#g'; }

for ch in timing action; do
  for cfg in static dynamic combined; do
    for m in gpt-4.1-mini gpt-4.1 deepseek-v3.2; do
      n=300
      workers=12
      [[ $m == deepseek-v3.2 ]] && workers=8
      [[ $ch == timing && $m != deepseek-v3.2 ]] && n=1500
      out="$OUT/frozen_opera-${ch}-${cfg}_$(tag "$m").json"
      [[ -f "$out" ]] && { echo "skip $out"; continue; }
      API_WORKERS=$workers python -m attribution_trials.audit.api_panel \
        "$ch" "$cfg" "$m" "$n"
    done
  done
done
