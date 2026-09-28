"""Structured memory family on ASSISTments 2009 (response channel).

Per-user statistics read by a pooled readout; no per-user training. The seed
sets only the within-cell imposter derangement and the permutation draws.
Runs seeds 0, 1 and 2.

  python -m attribution_trials.audit.structured_memory

Output: <AUDIT>/chess_kt/memory_pilot_kt_s<seed>.json
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial

from attribution_trials import paths
from attribution_trials.audit.static_embedding import TRAIN_FRAC, components_json
from attribution_trials.bench.cells import assign_tertile_cells, kt_prior_accuracy_scalar
from attribution_trials.bench.memory_family import run_memory_family_ladder
from attribution_trials.data.kt_csv import load_kt_csv


def main() -> int:
    data = load_kt_csv(
        str(paths.DATA / "kt/prepared/assist09.tsv"),
        n_students=500,
        min_responses=50,
    )
    cells = assign_tertile_cells(
        data,
        partial(kt_prior_accuracy_scalar, train_frac=TRAIN_FRAC),
        prefix="pacc",
    )
    out_dir = paths.ensure(paths.AUDIT / "chess_kt")
    for seed in (0, 1, 2):
        t0 = time.time()
        run = run_memory_family_ladder(
            data,
            cell_of=cells,
            train_frac=TRAIN_FRAC,
            seed=seed,
            bootstrap_n=2000,
            permutations=10000,
        )
        payload = {
            "substrate": "kt-assist09",
            "family": "structured-memory",
            "seed": seed,
            "n_players": len(run["player_ids"]),
            "singletons": run["singletons"],
            "wall_seconds": time.time() - t0,
            "grades": {},
            "drift": {},
        }
        for g, chans in run["grades"].items():
            comp = chans["response"]
            print(f"== structured memory seed={seed} grade={g}")
            print(comp.summary(), flush=True)
            payload["grades"][g] = {"response": components_json(comp)}
        for g, d in run["drift"].items():
            payload["drift"][g] = {"per_player": d}
        out = out_dir / f"memory_pilot_kt_s{seed}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"structured memory seed={seed} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
