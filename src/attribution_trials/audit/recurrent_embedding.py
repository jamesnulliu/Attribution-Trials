"""Recurrent embedding family: a per-user state updated along the history.

The A4 model is trained once; the conditions swap the state fed to it at
evaluation. The fixed target (A4frozen) freezes the user's state at the end
of the training period; the updated target (A4) keeps updating through the
held-out period.

Substrates:
  chess  pooled Lichess panel, groups = cohort x Elo tertile
  kt-rt  ASSISTments 2009 with the response-time column as the timing channel

  python -m attribution_trials.audit.recurrent_embedding <chess|kt-rt> <seed>

Output: <AUDIT>/chess_kt/full_evolving_chess_s<seed>.json (chess) or
        <AUDIT>/chess_kt/evolving_kt-rt_s<seed>.json.
The A4 checkpoints are saved under <AUDIT>/chess_kt/runs/<label>-a4/.
"""

from __future__ import annotations

import json
import sys
import time

from attribution_trials import paths
from attribution_trials.audit.static_embedding import TRAIN_FRAC, components_json, kt_setup
from attribution_trials.bench.evolving_family import run_evolving_family_ladder


def main() -> int:
    substrate, seed = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    if substrate == "chess":
        from attribution_trials.data.pooled_chess import backbone_factory, build_pooled_chess

        data, cells, _, _ = build_pooled_chess()
        factory = backbone_factory
        label = f"bench-full-evolving-chess-s{seed}"
        out_name = f"full_evolving_chess_s{seed}.json"
    elif substrate == "kt-rt":
        data, cells, factory = kt_setup(response_time_col=5)
        label = f"bench-evolving-kt-rt-s{seed}"
        out_name = f"evolving_kt-rt_s{seed}.json"
    else:
        raise ValueError(substrate)

    run = run_evolving_family_ladder(
        data,
        backbone_factory=factory,
        cell_of=cells,
        train_frac=TRAIN_FRAC,
        latent_dim=16,
        epochs=15,
        lr=1e-2,
        batch_size=16,
        seed=seed,
        label=label,
    )
    for ch, comp in run["grades"]["eval-swap"].items():
        print(f"== recurrent embedding {substrate} s{seed} {ch}")
        print(comp.summary(), flush=True)

    payload = {
        "substrate": "chess-pooled" if substrate == "chess" else substrate,
        "family": "evolving-latent",
        "seed": seed,
        "n_players": len(run["player_ids"]),
        "singletons": run["singletons"],
        "cells": cells,
        "wall_seconds": time.time() - t0,
        "drift_per_player": {ch: run["drift"][ch] for ch in run["drift"]},
        "grades": {
            "eval-swap": {
                ch: components_json(comp) for ch, comp in run["grades"]["eval-swap"].items()
            }
        },
    }
    out = paths.ensure(paths.AUDIT / "chess_kt") / out_name
    out.write_text(json.dumps(payload, indent=2))
    print(f"recurrent embedding {substrate} seed={seed} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
