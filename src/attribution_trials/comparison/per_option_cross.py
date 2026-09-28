"""Dense cross-user donor matrices for the static-embedding family.

The random-unrelated-profile control substitutes a donor drawn from all other
users, not only from the user's own cell.  For every ordered pair
(user p, donor d) this scores p's held-out decisions with d's learned static
embedding, using the trained A4 checkpoints of the static-embedding ladder.

The donor ring is global: offsets 1..N-1 over all players sorted, so after
N-1 passes every ordered pair has been scored exactly once and M is dense.
The diagonal is the own-embedding arm A4; the registered within-cell
derangement gives A3.

  python -m attribution_trials.comparison.per_option_cross [substrate ...]
Substrates: chess-pooled | kt | kt-rt
Output: <COMPARISON>/cross_matrices/M_cross_<substrate>_<channel>.npz
"""

from __future__ import annotations

import sys
import time
from functools import partial

import numpy as np
import torch

from attribution_trials import paths
from attribution_trials.bench.cells import (
    assign_tertile_cells,
    kt_prior_accuracy_scalar,
)
from attribution_trials.bench.swap import IdRemapInjector, within_cell_derangement
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import _per_player_nlls, session_split_indices
from attribution_trials.latent.static_individual import StaticIndividualInjector

# A4 checkpoints written by the static-embedding ladder runs:
#   <RUNS>/<run name>/<run id>/artifacts/checkpoint.pt
RUNS = paths.AUDIT / "chess_kt" / "runs"
KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
OUT = paths.COMPARISON / "cross_matrices"

LATENT_DIM = 16
BATCH = 16
SEEDS = (0, 1, 2)
TRAIN_FRAC = 0.7


def chess_pooled():
    from attribution_trials.data.pooled_chess import (
        backbone_factory,
        build_pooled_chess,
    )

    data, cells, _, dropped = build_pooled_chess("cohort-tertile")
    return {
        "data": data,
        "cells": cells,
        "factory": backbone_factory,
        "run_glob": "bench-full-chess-cohort-tertile-s{seed}-a4",
        "channels": ("move", "timing"),
        "note": f"dropped duplicate players: {dropped}",
    }


def kt_leg(response_time_col):
    from attribution_trials.policy.kt_backbone import KTBackbone

    data = load_kt_csv(
        str(KT_DATA),
        n_students=500,
        min_responses=50,
        response_time_col=response_time_col,
    )
    cells = assign_tertile_cells(
        data,
        partial(kt_prior_accuracy_scalar, train_frac=TRAIN_FRAC),
        prefix="pacc",
    )
    leg = "kt-rt" if response_time_col is not None else "kt"
    return {
        "data": data,
        "cells": cells,
        "factory": lambda seed: KTBackbone(latent_dim=LATENT_DIM, hidden_dim=64, seed=seed),
        "run_glob": f"bench-small-{leg}-s{{seed}}-a4",
        # the kt leg has a constant time_spent, so only move is interpretable
        "channels": ("move", "timing") if response_time_col else ("move",),
        "note": f"kt leg={leg}",
    }


SUBSTRATES = {
    "chess-pooled": chess_pooled,
    "kt": lambda: kt_leg(None),
    "kt-rt": lambda: kt_leg(5),
}


def load_a4(spec, player_ids, seed):
    pattern = spec["run_glob"].format(seed=seed)
    checkpoints = sorted((RUNS / pattern).glob("*/artifacts/checkpoint.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no A4 checkpoint: {RUNS / pattern}")
    blob = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    injector = StaticIndividualInjector(player_ids, latent_dim=LATENT_DIM, seed=seed)
    injector._build().load_state_dict(blob["injector"])
    backbone = spec["factory"](seed)
    backbone._build().load_state_dict(blob["backbone"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    injector.to(device)
    backbone.to(device)
    return injector, backbone


def run_substrate(name: str) -> None:
    spec = SUBSTRATES[name]()
    data, cells = spec["data"], spec["cells"]
    player_ids = [t.player_id for t in data.trajectories]
    index_of = {p: i for i, p in enumerate(player_ids)}
    splits = session_split_indices(data.trajectories, TRAIN_FRAC)
    channels = spec["channels"]
    print(
        f"[{name}] {len(player_ids)} players, "
        f"{len(set(cells.values()))} cells, channels={channels} "
        f"({spec['note']})",
        flush=True,
    )

    per_seed = {c: {"M": [], "a4": [], "a3": []} for c in channels}
    for seed in SEEDS:
        started = time.time()
        injector, backbone = load_a4(spec, player_ids, seed)
        move4, time4 = _per_player_nlls(injector, backbone, data, splits, batch_size=BATCH)
        arm4 = {"move": np.asarray(move4, float), "timing": np.asarray(time4, float)}

        derangement = within_cell_derangement(cells, seed=seed)
        move3, time3 = _per_player_nlls(
            IdRemapInjector(injector, derangement),
            backbone,
            data,
            splits,
            batch_size=BATCH,
        )
        arm3 = {"move": np.asarray(move3, float), "timing": np.asarray(time3, float)}

        matrices = {c: np.full((len(player_ids), len(player_ids)), np.nan) for c in channels}
        # One global ring: after N-1 offsets every ordered pair
        # (player, donor) has been scored exactly once.
        ring = sorted(player_ids)
        n_ring = len(ring)
        for offset in range(1, n_ring):
            id_map = {p: ring[(i + offset) % n_ring] for i, p in enumerate(ring)}
            if offset % 50 == 0:
                print(
                    f"    [{name}] seed={seed} offset {offset}/{n_ring - 1} "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )
            move_o, time_o = _per_player_nlls(
                IdRemapInjector(injector, id_map),
                backbone,
                data,
                splits,
                batch_size=BATCH,
            )
            scored = {"move": move_o, "timing": time_o}
            for channel in channels:
                for player, donor in id_map.items():
                    matrices[channel][index_of[player], index_of[donor]] = scored[channel][
                        index_of[player]
                    ]
        for channel in channels:
            for i in range(len(player_ids)):
                matrices[channel][i, i] = arm4[channel][i]
            per_seed[channel]["M"].append(matrices[channel])
            per_seed[channel]["a4"].append(arm4[channel])
            per_seed[channel]["a3"].append(arm3[channel])
        print(
            f"  seed {seed} done, {n_ring - 1} offsets (global ring), "
            f"wall={time.time() - started:.0f}s",
            flush=True,
        )

    paths.ensure(OUT)
    for channel in channels:
        target = OUT / f"M_cross_{name}_{channel}.npz"
        np.savez_compressed(
            target,
            M=np.stack(per_seed[channel]["M"]),
            a4=np.stack(per_seed[channel]["a4"]),
            a3_reg=np.stack(per_seed[channel]["a3"]),
            player_ids=np.array(player_ids),
            cell_of=np.array([cells[p] for p in player_ids]),
            seeds=np.array(SEEDS),
        )
        print(f"-> {target}", flush=True)


def main() -> None:
    wanted = sys.argv[1:] or list(SUBSTRATES)
    for name in wanted:
        if name not in SUBSTRATES:
            raise SystemExit(f"unknown substrate {name}")
        run_substrate(name)


if __name__ == "__main__":
    main()
