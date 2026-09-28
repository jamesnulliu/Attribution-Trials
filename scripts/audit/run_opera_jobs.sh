#!/usr/bin/env bash
# OPeRA user-profile LoRA and frozen prompt panels: runs the job list from
# attribution_trials.audit.opera_jobs, one job per GPU at a time.
# Skips a job whose output already exists.
#
#   scripts/audit/run_opera_jobs.sh [n_gpus]
set -u
NGPU="${1:-1}"
OUT=$(python -c 'from attribution_trials import paths; print(paths.AUDIT / "opera")')
mkdir -p "$OUT/logs"
mapfile -t JOBS < <(python -m attribution_trials.audit.opera_jobs)

run_slot() {
  local gpu=$1 i kind a b c d key
  local -a cmd
  for ((i = gpu; i < ${#JOBS[@]}; i += NGPU)); do
    read -r kind a b c d <<< "${JOBS[$i]}"
    if [[ $kind == SFT ]]; then
      key="sft_${a}_${b}_${d}_s${c}"
      cmd=(attribution_trials.audit.sft_matrix "$a" "$b" "$c" "$d")
    else
      key="frozen_${a}_${b}"
      cmd=(attribution_trials.audit.frozen_panel "$a" "$b")
    fi
    [[ -f "$OUT/$key.json" ]] && continue
    echo "[gpu$gpu] $key"
    CUDA_VISIBLE_DEVICES=$gpu python -m "${cmd[@]}" > "$OUT/logs/$key.log" 2>&1 \
      || echo "[gpu$gpu] FAILED $key (see $OUT/logs/$key.log)"
  done
}

for ((g = 0; g < NGPU; g++)); do run_slot "$g" & done
wait
