#!/usr/bin/env bash
# Re-run the generation at a lower temperature for Qwen3-8B, HumanLike-7B and Osim-8B,
# one process per simulator over the GPUs in $GPUS (round robin), then the temperature readout.
# GENERATION_BATCH changes padding and therefore the sampling RNG stream; the reported run
# used 32 for 8b and humanlike and 16 for osim (e.g. CANDIDATES=osim GENERATION_BATCH=16).
#   GPUS=0,1 TEMPERATURE=0.7 GENERATION_BATCH=32 bash scripts/wildchat/temperature.sh
set -u -o pipefail

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1}"
TEMPERATURE="${TEMPERATURE:-0.7}"
CANDIDATES="${CANDIDATES:-8b humanlike osim}"
export GENERATION_BATCH="${GENERATION_BATCH:-32}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="$("$PYTHON" -c 'from attribution_trials import paths; print(paths.ensure(paths.WILDCHAT))')"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

IFS=',' read -r -a gpus <<<"$GPUS"
read -r -a candidates <<<"$CANDIDATES"
status=0
pids=()
for i in "${!candidates[@]}"; do
  c="${candidates[$i]}"
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  echo "start $c on GPU $gpu (T=$TEMPERATURE, batch $GENERATION_BATCH)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m attribution_trials.wildchat.temperature "$c" \
    --temperature "$TEMPERATURE" >"$LOGS/temperature_${c}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status )); then
  echo "one or more temperature jobs failed; see $LOGS" >&2
  exit 1
fi

if [[ -s "$OUT/temperature_$TEMPERATURE/content_8b.json" &&
      -s "$OUT/temperature_$TEMPERATURE/content_humanlike.json" &&
      -s "$OUT/temperature_$TEMPERATURE/content_osim.json" ]]; then
  "$PYTHON" -u -m attribution_trials.wildchat.temperature_readout --dir "$OUT/temperature_$TEMPERATURE"
fi
