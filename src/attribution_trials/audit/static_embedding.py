"""Static embedding family: a learned vector per user, one model retrained per condition.

Substrates:
  chess  pooled Lichess panel, groups = cohort x Elo tertile
  kt     ASSISTments 2009, response channel
  kt-rt  ASSISTments 2009 with the response-time column (col 5) as the timing
         channel; without it the KT timing target is constant

  python -m attribution_trials.audit.static_embedding <chess|kt|kt-rt> <seed>

Output: <AUDIT>/chess_kt/full_ladder_chess_s<seed>.json (chess) or
        <AUDIT>/chess_kt/small_ladder_<kt|kt-rt>_s<seed>.json.
The A4 checkpoints are saved under <AUDIT>/chess_kt/runs/<label>-a4/.
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial

from attribution_trials import paths
from attribution_trials.bench.cells import assign_tertile_cells, kt_prior_accuracy_scalar
from attribution_trials.bench.ladder import run_static_family_ladder
from attribution_trials.data.kt_csv import load_kt_csv

TRAIN_FRAC = 0.7


def chess_setup():
    from attribution_trials.data.pooled_chess import backbone_factory, build_pooled_chess

    data, cells, _, _ = build_pooled_chess()
    return data, cells, backbone_factory


def kt_setup(response_time_col: int | None = None):
    data = load_kt_csv(
        str(paths.DATA / "kt/prepared/assist09.tsv"),
        n_students=500,
        min_responses=50,
        response_time_col=response_time_col,
    )
    cells = assign_tertile_cells(
        data,
        partial(kt_prior_accuracy_scalar, train_frac=TRAIN_FRAC),
        prefix="pacc",
    )
    from attribution_trials.policy.kt_backbone import KTBackbone

    factory = lambda seed: KTBackbone(latent_dim=16, hidden_dim=64, seed=seed)
    return data, cells, factory


def components_json(comp) -> dict:
    """Serialize one channel's :class:`LadderComponents`."""
    return {
        "identity_max_abs_err": comp.identity_max_abs_err,
        "components": {
            k: {
                "point": comp.cis[k].point,
                "low": comp.cis[k].low,
                "high": comp.cis[k].high,
                "p_flip": comp.pvalues[k],
            }
            for k in comp.cis
        },
        "player_ids": comp.player_ids,
        "per_player_nll": comp.per_player,
    }


def main() -> int:
    substrate, seed = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    if substrate == "chess":
        data, cells, factory = chess_setup()
        label = f"bench-full-chess-cohort-tertile-s{seed}"
        out_name = f"full_ladder_chess_s{seed}.json"
    elif substrate in ("kt", "kt-rt"):
        data, cells, factory = kt_setup(response_time_col=5 if substrate == "kt-rt" else None)
        label = f"bench-small-{substrate}-s{seed}"
        out_name = f"small_ladder_{substrate}_s{seed}.json"
    else:
        raise ValueError(substrate)

    run = run_static_family_ladder(
        data,
        backbone_factory=factory,
        cell_of=cells,
        train_frac=TRAIN_FRAC,
        latent_dim=16,
        epochs=15,
        lr=1e-2,
        batch_size=16,
        seed=seed,
        timing_lambda=0.5,
        bootstrap_n=2000,
        permutations=10000,
        label=label,
    )
    print(run.report(), flush=True)

    payload = {
        "substrate": "chess-pooled" if substrate == "chess" else substrate,
        "seed": seed,
        "n_players": len(run.player_ids),
        "singletons": sorted(run.singleton_players),
        "cells": run.cell_of,
        "wall_seconds": time.time() - t0,
        "grades": {
            g: {ch: components_json(comp) for ch, comp in chans.items()}
            for g, chans in run.grades.items()
        },
    }
    out = paths.ensure(paths.AUDIT / "chess_kt") / out_name
    out.write_text(json.dumps(payload, indent=2))
    print(f"static embedding {substrate} seed={seed} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
