"""Next-decision accuracy per arm (the benchmark comparison's accuracy row).

Reads the per-option rescoring cells and reports, per cell and arm, how often
the argmax over the closed option set is the observed label.

    A0   no card                    A1   placebo (population) card
    A2   group (cell) card          A3   same-cell imposter's frozen card
    A4f  own frozen card

A4f is the target column and A3 the imposter column.

    python -m attribution_trials.comparison.accuracy
Output: <COMPARISON>/accuracy.json
"""

from __future__ import annotations

import json

import numpy as np

from attribution_trials import paths

ARMS = ("A0", "A1", "A2", "A3", "A4f")


def accuracy(logp: np.ndarray, labels: np.ndarray) -> float:
    """Share of rows whose argmax over the option vector is the observed label.

    The stored vectors are unnormalised total log-probabilities; softmax is
    monotone, so the argmax is the same as over the normalised distribution.
    """
    return float((logp.argmax(axis=1) == labels).mean())


def analyse_cell(path) -> dict:
    data = np.load(path, allow_pickle=True)
    labels = data["labels"]
    row = {
        "cell": path.stem,
        "n_rows": int(len(labels)),
        "n_labels": int(len(data["label_set"])),
        "accuracy": {},
    }
    for arm in ARMS:
        key = f"logp_{arm}"
        row["accuracy"][arm] = accuracy(data[key], labels) if key in data.files else None
    return row


def main() -> None:
    files = sorted((paths.COMPARISON / "per_option").glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz under {paths.COMPARISON / 'per_option'}")
    rows = [analyse_cell(p) for p in files]
    out = paths.ensure(paths.COMPARISON) / "accuracy.json"
    out.write_text(json.dumps({"cells": rows}, indent=2))
    print(f"wrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
