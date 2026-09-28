#!/usr/bin/env bash
# Attribution-trial scores for the seven WildChat simulators, one process per simulator
# over the GPUs in $GPUS (round robin), then the per-simulator trial table.
# Existing scale_<size>.json files are kept.
#   GPUS=0,1,2,3,4,5,6 bash scripts/wildchat/score_trial.sh
set -u -o pipefail

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2,3,4,5,6}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT="$("$PYTHON" -c 'from attribution_trials import paths; print(paths.ensure(paths.WILDCHAT))')"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

IFS=',' read -r -a gpus <<<"$GPUS"
candidates=(1.7b 8b dense humanlike coser osim humanlm)
status=0
pids=()
for i in "${!candidates[@]}"; do
  c="${candidates[$i]}"
  output="$OUT/scale_${c/dense/32b}.json"
  if [[ -s "$output" ]]; then
    echo "existing $output; skipping $c"
    continue
  fi
  case "$c" in
    1.7b|8b) module=score_trial_frozen ;;
    *) module=score_trial ;;
  esac
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  echo "start $c on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m "attribution_trials.wildchat.$module" "$c" \
    >"$LOGS/score_${c}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status )); then
  echo "one or more scoring jobs failed; see $LOGS" >&2
  exit 1
fi

"$PYTHON" -u -m attribution_trials.wildchat.trial_table
