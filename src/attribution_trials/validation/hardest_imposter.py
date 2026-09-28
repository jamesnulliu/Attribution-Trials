"""The hardest-imposter bound for the static-embedding combinations.

From the full within-cell donor cross-matrices (static_cross_matrices), the
imposter for each user is the best-fitting donor in the user's cell (the one
with the lowest held-out NLL), averaged over seeds.  This selects on the
evaluation outcome, so it is an adversarial bound, not an estimand.  The value
is target minus imposter NLL (negative favours the target), with a user
bootstrap interval.

  python -m attribution_trials.validation.hardest_imposter
Output: <VALIDATION>/hardest_imposter.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from attribution_trials import paths

MATRICES = paths.VALIDATION / "static_matrices"

SEED = 0
BOOT_N = 10000

# file stem -> combination key
CELLS = {
    "M_static_chess-pooled_move": "static-embedding|chess-pooled|move",
    "M_static_chess-pooled_timing": "static-embedding|chess-pooled|timing",
    "M_static_kt_move": "static-embedding|kt|move",
    "M_static_kt-rt_move": "static-embedding|kt-rt|move",
    "M_static_kt-rt_timing": "static-embedding|kt-rt|timing",
}


def boot_mean(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), (BOOT_N, len(values)))].mean(1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def analyse(stem: str) -> dict:
    blob = np.load(MATRICES / f"{stem}.npz", allow_pickle=True)
    matrix = blob["M"]  # (seeds, players, players)
    a4_seed = blob["a4"]  # (seeds, players)
    players = [str(p) for p in blob["player_ids"]]
    cell_of = {p: str(c) for p, c in zip(players, blob["cell_of"])}
    keep = [str(u) for u in blob["keep_ids"]]
    position = {p: i for i, p in enumerate(players)}

    members = defaultdict(list)
    for player in players:
        members[cell_of[player]].append(player)

    a4 = np.array([a4_seed[:, position[u]].mean() for u in keep])
    donors = []
    for user in keep:
        values = [
            np.nanmean(matrix[:, position[user], position[d]])
            for d in members[cell_of[user]]
            if d != user
        ]
        donors.append(np.array([v for v in values if np.isfinite(v)]))

    # the best-fitting (hardest) donor
    a3 = np.array([np.quantile(d, 0.0) for d in donors])
    point, low, high = boot_mean(a4 - a3)
    return {
        "cell": CELLS[stem],
        "n_users": len(keep),
        "seeds": int(matrix.shape[0]),
        "hardest": {"point": point, "low": low, "high": high},
    }


def main() -> None:
    rows = [analyse(stem) for stem in CELLS]
    payload = {
        "ci": f"user bootstrap, {BOOT_N} resamples, seed {SEED}",
        "cells": rows,
    }
    out = paths.ensure(paths.VALIDATION) / "hardest_imposter.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        h = row["hardest"]
        print(
            f"  {row['cell']:<40} n={row['n_users']:<4} "
            f"hardest={h['point']:+.4f} [{h['low']:+.4f}, {h['high']:+.4f}]",
            flush=True,
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
