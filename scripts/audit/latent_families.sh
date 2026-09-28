#!/usr/bin/env bash
# Latent families on chess and KT, seeds 0 1 2:
#   static embedding    chess, kt, kt-rt
#   recurrent embedding chess, kt-rt
#   structured memory   kt (all seeds in one run)
# Skips a run whose output already exists.
#
#   scripts/audit/latent_families.sh
set -euo pipefail
OUT=$(python -c 'from attribution_trials import paths; print(paths.AUDIT / "chess_kt")')

for seed in 0 1 2; do
  [[ -f "$OUT/full_ladder_chess_s$seed.json" ]] \
    || python -m attribution_trials.audit.static_embedding chess "$seed"
  for sub in kt kt-rt; do
    [[ -f "$OUT/small_ladder_${sub}_s$seed.json" ]] \
      || python -m attribution_trials.audit.static_embedding "$sub" "$seed"
  done
  [[ -f "$OUT/full_evolving_chess_s$seed.json" ]] \
    || python -m attribution_trials.audit.recurrent_embedding chess "$seed"
  [[ -f "$OUT/evolving_kt-rt_s$seed.json" ]] \
    || python -m attribution_trials.audit.recurrent_embedding kt-rt "$seed"
done
[[ -f "$OUT/memory_pilot_kt_s2.json" ]] || python -m attribution_trials.audit.structured_memory
