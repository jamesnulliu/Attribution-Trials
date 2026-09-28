"""Full within-cell donor cross-matrices for the static-embedding family.

For every static-embedding combination, M[s, p, d] is the seed-s NLL of
player p's held-out rows scored under donor d's trained embedding, over every
within-cell donor; the diagonal is the target (A4).  Every closer-imposter
statistic (hardest donor, nearest neighbour) is then a lookup.  Both channels
come out of the same forward passes, so five combinations come from three
training substrates:

  chess-pooled : move, timing
  kt           : move
  kt-rt        : move, timing

Inputs: the seed-0/1/2 target (A4) checkpoints the static-embedding audit runs
save under <AUDIT>/chess_kt/runs/<run name>/<run id>/artifacts/checkpoint.pt
(run names below), the pooled chess panel, and <DATA>/kt/prepared/assist09.tsv.

  python -m attribution_trials.validation.static_cross_matrices [substrate ...]
Output: <VALIDATION>/static_matrices/M_static_<substrate>_<channel>.npz
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from functools import partial

import numpy as np
import torch

from attribution_trials import paths
from attribution_trials.bench.cells import assign_tertile_cells, kt_prior_accuracy_scalar
from attribution_trials.bench.swap import IdRemapInjector, within_cell_derangement
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import _per_player_nlls, session_split_indices
from attribution_trials.latent.static_individual import StaticIndividualInjector

RUNS = paths.AUDIT / "chess_kt" / "runs"
KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
OUT = paths.VALIDATION / "static_matrices"

LATENT_DIM = 16
BATCH = 16
SEEDS = (0, 1, 2)
TRAIN_FRAC = 0.7


def chess_pooled():
    from attribution_trials.data.pooled_chess import backbone_factory, build_pooled_chess

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
        raise AssertionError(f"no A4 checkpoint: {RUNS / pattern}")
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

    members = defaultdict(list)
    for player in player_ids:
        members[cells[player]].append(player)
    members = {c: sorted(m) for c, m in members.items()}
    widest = max(len(m) for m in members.values())

    per_seed = {c: {"M": [], "a4": []} for c in channels}
    for seed in SEEDS:
        started = time.time()
        injector, backbone = load_a4(spec, player_ids, seed)
        move4, time4 = _per_player_nlls(injector, backbone, data, splits, batch_size=BATCH)
        arm4 = {"move": np.asarray(move4, float), "timing": np.asarray(time4, float)}

        matrices = {c: np.full((len(player_ids), len(player_ids)), np.nan) for c in channels}
        for offset in range(1, widest):
            id_map = {}
            for group in members.values():
                size = len(group)
                if offset >= size:
                    continue
                for position, player in enumerate(group):
                    id_map[player] = group[(position + offset) % size]
            if not id_map:
                continue
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
        print(
            f"  seed {seed} done, {widest - 1} offsets, wall={time.time() - started:.0f}s",
            flush=True,
        )

    paths.ensure(OUT)
    keep = [p for p in player_ids if within_cell_derangement(cells, seed=0)[p] != p]
    for channel in channels:
        target = OUT / f"M_static_{name}_{channel}.npz"
        np.savez_compressed(
            target,
            M=np.stack(per_seed[channel]["M"]),
            a4=np.stack(per_seed[channel]["a4"]),
            player_ids=np.array(player_ids),
            cell_of=np.array([cells[p] for p in player_ids]),
            keep_ids=np.array(keep),
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
