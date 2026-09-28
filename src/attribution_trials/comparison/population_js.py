"""Population realism per arm: Jensen-Shannon divergence (nats) between the
real label distribution of a cell and the method-implied one (the mean of the
per-row softmax), for every arm.  The benchmark comparison's population
realism row reads the score kept, JS(A3) / JS(A4f).

    python -m attribution_trials.comparison.population_js
Output: <COMPARISON>/population_js.json
"""

from __future__ import annotations

import json

import numpy as np

from attribution_trials import paths

ARMS = ("A0", "A1", "A2", "A3", "A4f")


def softmax(logp: np.ndarray) -> np.ndarray:
    shifted = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))


def analyse_cell(path) -> dict:
    data = np.load(path, allow_pickle=True)
    labels = data["labels"]
    n_labels = int(len(data["label_set"]))
    real = np.bincount(labels, minlength=n_labels).astype(float)
    real /= real.sum()

    row = {
        "cell": path.stem,
        "n_rows": int(len(labels)),
        "n_labels": n_labels,
        "arms": {},
    }
    for arm in ARMS:
        key = f"logp_{arm}"
        if key not in data.files:
            row["arms"][arm] = None
            continue
        model = softmax(data[key]).mean(axis=0)
        row["arms"][arm] = {"js": js_divergence(real, model)}
    return row


def main() -> None:
    files = sorted((paths.COMPARISON / "per_option").glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz under {paths.COMPARISON / 'per_option'}")
    rows = [analyse_cell(p) for p in files]
    out = paths.ensure(paths.COMPARISON) / "population_js.json"
    out.write_text(json.dumps({"cells": rows}, indent=2))
    print(f"wrote {out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
