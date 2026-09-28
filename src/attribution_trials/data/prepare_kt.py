"""Preprocess raw ASSISTments 2009 ("skill builder") data into the 6-column
TSV ``attribution_trials.data.kt_csv.load_kt_csv`` reads: ``user_id, item_id,
timestamp, correct, skill_id, ms_first_response``.

Mirrors the standard ``theophilee/learner-performance-prediction`` recipe for
"assistments09" (filter non-binary outcomes, drop untagged-skill rows,
dedupe multi-skill rows to one row per interaction, sort by ``order_id`` then
group by student preserving order) and additionally carries through
``ms_first_response`` (dropped by that recipe) so the real response-time
channel is available.

The raw file is not redistributed; fetch it yourself, e.g. from the USTC
mirror EduData points at, into ``<DATA>/kt/raw/``:
  curl -O http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip
  unzip 2009_skill_builder_data_corrected.zip

Usage:
  python -m attribution_trials.data.prepare_kt [raw.csv] [out.tsv]
Defaults: <DATA>/kt/raw/skill_builder_data_corrected.csv ->
          <DATA>/kt/prepared/assist09.tsv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from attribution_trials import paths

RAW = paths.DATA / "kt/raw/skill_builder_data_corrected.csv"
OUT = paths.DATA / "kt/prepared/assist09.tsv"


def prepare_assistments09(raw_path: str, out_path: str) -> None:
    seen_order_ids: set[str] = set()
    rows: list[tuple[int, str, str, str, str, str]] = []
    # (order_id, user_id, item_id, correct, skill_id, ms)

    with open(raw_path, encoding="ISO-8859-1", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            order_id = r["order_id"]
            if order_id in seen_order_ids:
                continue  # keep first skill tag for multi-skill items
            correct = r["correct"]
            if correct not in ("0", "1"):
                continue  # drop continuous/partial-credit outcomes
            skill_id = r["skill_id"]
            if not skill_id:
                continue  # drop untagged-skill rows
            seen_order_ids.add(order_id)
            rows.append(
                (
                    int(order_id),
                    r["user_id"],
                    r["problem_id"],
                    correct,
                    skill_id,
                    r["ms_first_response"],
                )
            )

    rows.sort(key=lambda x: x[0])  # temporal proxy, per theophilee's recipe

    by_user: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        by_user[row[1]].append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["user_id", "item_id", "timestamp", "correct", "skill_id", "ms"])
        n = 0
        for user_id, seq in by_user.items():
            # No real wall-clock timestamp in this dataset (per theophilee's
            # recipe); order_id already gives the temporal order used above.
            for _oid, _u, item_id, correct, skill_id, ms in seq:
                writer.writerow([user_id, item_id, 0, correct, skill_id, ms])
                n += 1

    print(
        f"{out_path}: {len(by_user)} students, {len(rows)} responses, "
        f"{len({r[4] for r in rows})} skills"
    )


if __name__ == "__main__":
    if len(sys.argv) not in (1, 3):
        print(__doc__)
        raise SystemExit(1)
    raw, out = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else (RAW, OUT)
    prepare_assistments09(str(raw), str(out))
