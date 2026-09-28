#!/usr/bin/env bash
# Generate and score A3/A4 samples for the seven WildChat simulators, one process per GPU slot
# in $GPUS; simulators already complete are skipped.
# GENERATION_BATCH changes padding and therefore the sampling RNG stream; the reported run used 4.
#   GPUS=0,1,2,3,4,5,6 GENERATION_BATCH=4 bash scripts/wildchat/generate.sh
set -u -o pipefail

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2,3,4,5,6}"
export GENERATION_BATCH="${GENERATION_BATCH:-4}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="$("$PYTHON" -c 'from attribution_trials import paths; print(paths.ensure(paths.WILDCHAT))')"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

candidates=(1.7b 8b dense humanlike coser osim humanlm)

candidate_complete() {
  local c="$1"
  [[ -s "$OUT/content_${c}.json" &&
     -s "$OUT/form_${c}.json" &&
     -s "$OUT/samples/samples_${c}.jsonl.gz" ]]
}

IFS=',' read -r -a gpus <<<"$GPUS"
offset=0
overall=0
while (( offset < ${#candidates[@]} )); do
  pids=()
  for ((slot=0; slot<${#gpus[@]} && offset+slot<${#candidates[@]}; slot++)); do
    c="${candidates[offset+slot]}"
    if candidate_complete "$c"; then
      echo "existing outputs; skipping $c"
      continue
    fi
    gpu="${gpus[$slot]}"
    echo "start $c on GPU $gpu (batch $GENERATION_BATCH)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m attribution_trials.wildchat.generate "$c" \
      >"$LOGS/generate_${c}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then overall=1; fi
  done
  ((offset += ${#gpus[@]}))
done

if (( overall )); then
  echo "one or more generation jobs failed; see $LOGS" >&2
  exit 1
fi
echo "generation complete"
