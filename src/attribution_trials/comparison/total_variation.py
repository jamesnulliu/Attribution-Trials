"""Group-level total variation per arm (the benchmark comparison's TVD row).

SimBench and Meister score a simulator by how close the group's answer
distribution is to the real group's, not by per-row accuracy:

    mean total variation = TVD(P, Q)

with P the real (observed-label) distribution and Q the method-implied
distribution (the mean of the per-row softmax).  The group is the whole cell:
one distribution over all rows of the cell.

    python -m attribution_trials.comparison.total_variation
Output: <COMPARISON>/total_variation.json
"""

from __future__ import annotations

import json

import numpy as np

from attribution_trials import paths

ARMS = ("A0", "A1", "A2", "A3", "A4f")
MIN_GROUP_ROWS = 5  # a distribution over fewer rows is mostly sampling noise


def softmax(logp: np.ndarray) -> np.ndarray:
    shifted = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def group_scores(probs: np.ndarray, labels: np.ndarray, keys: np.ndarray, n_labels: int) -> dict:
    """Mean TVD over groups defined by `keys`."""
    tv_pq, sizes = [], []
    for key in np.unique(keys):
        mask = keys == key
        if mask.sum() < MIN_GROUP_ROWS:
            continue
        real = np.bincount(labels[mask], minlength=n_labels).astype(float)
        real /= real.sum()
        model = probs[mask].mean(axis=0)
        tv_pq.append(tvd(real, model))
        sizes.append(int(mask.sum()))
    return {
        "groups_scored": len(tv_pq),
        "rows_covered": int(sum(sizes)),
        "mean_tvd": float(np.mean(tv_pq)) if tv_pq else None,
    }


def analyse_cell(path) -> dict:
    data = np.load(path, allow_pickle=True)
    labels = data["labels"]
    n_labels = int(len(data["label_set"]))
    whole = np.zeros(len(labels), dtype=np.int64)
    row = {
        "cell": path.stem,
        "n_rows": int(len(labels)),
        "n_labels": n_labels,
        "whole": {},
    }
    for arm in ARMS:
        key = f"logp_{arm}"
        if key not in data.files:
            row["whole"][arm] = None
            continue
        row["whole"][arm] = group_scores(softmax(data[key]), labels, whole, n_labels)
    return row


def main() -> None:
    files = sorted((paths.COMPARISON / "per_option").glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz under {paths.COMPARISON / 'per_option'}")
    rows = [analyse_cell(p) for p in files]
    out = paths.ensure(paths.COMPARISON) / "total_variation.json"
    out.write_text(json.dumps({"cells": rows}, indent=2))
    print(f"wrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
