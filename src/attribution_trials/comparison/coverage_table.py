"""Population-realism table: the coverage basket over every method-implied
population, against the matching real per-user profiles.

Populations (inputs under <COMPARISON>/coverage_inputs, written by
comparison.rescore_opera and comparison.rescore_chess):
- OPeRA per-option rescores `gpu/rescore_opera-*.json` (Qwen3-8B and Osim-8B,
  action/timing channels).  R = per-user label histograms (min 3 decisions),
  G = per-user mean predictive distributions.
- Chess timing `cpu/chess_*.npz`, same construction over timing buckets.

Two reference rows per domain are added: `::trap` (point mass at the
population mean) and `::boot` (a bootstrap resample of R).  Rows without `::`
are the method populations the benchmark comparison's population-realism row
counts.  Positive aggregate_z = worse than a resample of the real population.

    python -m attribution_trials.comparison.coverage_table
Output: <COMPARISON>/coverage_table.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from attribution_trials import paths
from attribution_trials.comparison.coverage_basket import basket

INPUTS = paths.COMPARISON / "coverage_inputs"
SEED = 0
MIN_COUNT = 3


def opera_profiles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    options = data["options"]
    idx = {o: i for i, o in enumerate(options)}
    players = np.asarray(data["players"])
    labels = np.asarray([idx[l] for l in data["labels"]])
    logp = np.asarray(data["option_logprobs"], dtype=float)
    probs = np.exp(logp - logp.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    R, G = [], []
    for u in sorted(set(players)):
        mask = players == u
        if mask.sum() < MIN_COUNT:
            continue
        hist = np.bincount(labels[mask], minlength=len(options)).astype(float)
        R.append(hist / hist.sum())
        G.append(probs[mask].mean(axis=0))
    return np.asarray(R), np.asarray(G)


def npz_timing_profiles(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    pidx = z["dec_player_idx"]
    true_b = z["timing_true_bucket"]
    probs = z["timing_bucket_probs"]
    n_buckets = probs.shape[1]
    R, G = [], []
    for u in range(int(pidx.max()) + 1):
        mask = pidx == u
        if mask.sum() < MIN_COUNT:
            continue
        hist = np.bincount(true_b[mask], minlength=n_buckets).astype(float)
        R.append(hist / hist.sum())
        G.append(probs[mask].mean(axis=0))
    return np.asarray(R), np.asarray(G)


def references(R: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    trap = np.tile(R.mean(axis=0), (len(R), 1))
    boot = R[rng.integers(0, len(R), size=len(R))]
    return {"trap": trap, "boot": boot}


def row(name: str, R: np.ndarray, G: np.ndarray) -> dict:
    res = basket(R, G, seed=SEED)
    return {
        "cell": name,
        "n": int(len(R)),
        "d": int(R.shape[1]),
        "aggregate_z": float(res.aggregate_z),
    }


def main() -> None:
    rows = []
    domains_seen = {}
    for path in sorted((INPUTS / "gpu").glob("rescore_opera-*.json")):
        cell = path.stem.replace("rescore_", "")
        R, G = opera_profiles(path)
        rows.append(row(cell, R, G))
        domains_seen.setdefault(("opera-" + cell.split("-")[1]).split("_")[0], R)
        print("done", cell, flush=True)
    for path in sorted((INPUTS / "cpu").glob("chess_*.npz")):
        R, G = npz_timing_profiles(path)
        cell = path.stem + "_timing"
        rows.append(row(cell, R, G))
        domains_seen.setdefault(path.stem.split("_")[0] + "-timing", R)
        print("done", cell, flush=True)
    if not rows:
        raise SystemExit(f"No populations under {INPUTS}")
    for domain, R in sorted(domains_seen.items()):
        for ref_name, G in references(R, SEED).items():
            rows.append(row(f"{domain}::{ref_name}", R, G))
            print("done", domain, ref_name, flush=True)
    out = paths.ensure(paths.COMPARISON) / "coverage_table.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    n_methods = sum(1 for r in rows if "::" not in r["cell"])
    print(f"wrote {out} ({len(rows)} rows, {n_methods} method populations)")


if __name__ == "__main__":
    main()
