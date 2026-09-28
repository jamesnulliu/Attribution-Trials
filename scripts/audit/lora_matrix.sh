#!/usr/bin/env bash
# User-profile LoRA on chess and KT:
#   substrates kt chess x conditions none placebo group self x seeds 0 1 2
#   x sizes 1.7b 8b (48 runs), one run per GPU at a time.
# Skips a run whose output already exists.
#
#   scripts/audit/lora_matrix.sh [n_gpus]
set -u
NGPU="${1:-1}"
OUT=$(python -c 'from attribution_trials import paths; print(paths.AUDIT / "chess_kt")')
mkdir -p "$OUT/logs"

JOBS=()
for sub in kt chess; do
  for size in 1.7b 8b; do
    for cond in none placebo group self; do
      for seed in 0 1 2; do
        JOBS+=("$sub $cond $seed $size")
      done
    done
  done
done

run_slot() {
  local gpu=$1 i sub cond seed size key
  for ((i = gpu; i < ${#JOBS[@]}; i += NGPU)); do
    read -r sub cond seed size <<< "${JOBS[$i]}"
    key="sft_${sub}_${cond}_${size}_s${seed}"
    [[ -f "$OUT/$key.json" ]] && continue
    echo "[gpu$gpu] $key"
    CUDA_VISIBLE_DEVICES=$gpu python -m attribution_trials.audit.sft_matrix \
      "$sub" "$cond" "$seed" "$size" > "$OUT/logs/$key.log" 2>&1 \
      || echo "[gpu$gpu] FAILED $key (see $OUT/logs/$key.log)"
  done
}

for ((g = 0; g < NGPU; g++)); do run_slot "$g" & done
wait
