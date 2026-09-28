#!/usr/bin/env bash
# Build the three Lichess blitz cohorts of the pooled chess panel into $AT_DATA/chess/<cohort>.
# Each cohort is selected from the first N games of one monthly archive of the Lichess
# open database (N below), with identical ingest settings.
# Needs the `chess` extra (python-chess, zstandard). The archives are large.
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA="$("$PYTHON" -c 'from attribution_trials import paths; print(paths.DATA)')"
RAW="$DATA/chess/raw"
BASE="https://database.lichess.org/standard"
mkdir -p "$RAW"

#        cohort  month    games read
COHORTS=("ec2017 2017-04 1209418"
         "ec2019 2019-07 1257490"
         "ec2021 2021-06 1158576")

for row in "${COHORTS[@]}"; do
  read -r cohort month n_games <<< "$row"
  archive="$RAW/lichess_db_standard_rated_${month}.pgn.zst"
  if [ ! -s "$archive" ]; then
    curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
      --output "${archive}.part" "$BASE/lichess_db_standard_rated_${month}.pgn.zst"
    mv -f "${archive}.part" "$archive"
  fi
  [ -s "$DATA/chess/$cohort/dataset.jsonl.gz" ] && continue
  "$PYTHON" -m attribution_trials.cli ingest "$archive" \
    --out "$DATA/chess/$cohort" \
    --speed blitz \
    --min-games 30 \
    --min-sessions 3 \
    --max-players 100 \
    --max-games-per-player 20 \
    --gap-threshold 1800 \
    --max-games "$n_games" \
    --batch-size 512 \
    --workers 8
done
