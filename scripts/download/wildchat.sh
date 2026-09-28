#!/usr/bin/env bash
# Download the 14 WildChat-1M parquet shards at the pinned revision into $AT_DATA/wildchat.
# If the dataset asks for authentication, accept its terms on the Hub and export HF_TOKEN.
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA="$("$PYTHON" -c 'from attribution_trials import paths; print(paths.DATA)')"
OUT="$DATA/wildchat"
REV="7d6490e462285cf85d91eabea0f9a954fbddcd1f"
BASE="https://huggingface.co/datasets/allenai/WildChat-1M/resolve/$REV/data"
AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH=(-H "Authorization: Bearer $HF_TOKEN")
fi

mkdir -p "$OUT"
for n in $(seq 0 13); do
  shard=$(printf '%05d' "$n")
  dest="$OUT/train-${shard}-of-00014.parquet"
  part="${dest}.part"
  if [ -s "$dest" ]; then
    echo "present $dest"
    continue
  fi
  echo "downloading train-${shard}-of-00014.parquet"
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    ${AUTH[@]+"${AUTH[@]}"} --output "$part" \
    "$BASE/train-${shard}-of-00014.parquet?download=true"
  mv -f "$part" "$dest"
done
echo "WildChat download complete: $OUT"
