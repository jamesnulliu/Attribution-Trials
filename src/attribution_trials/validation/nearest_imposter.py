"""The gain under a nearest-neighbour imposter, static-embedding combinations.

The imposter is the nearest other user inside the registered cell, in
z-scored Euclidean distance over a per-user feature vector built from the
training split alone (the evaluation split is never touched):

  chess   log activity, mean and sd of log seconds per move, entropy of the
          (from,to) move distribution, and its top-move share
  kt      log activity, train accuracy
  kt-rt   the same plus mean and sd of log time read with
          response_time_col=2 (the timestamp column of assist09.tsv, which
          is constant, so these two features are constant after z-scoring)

The gain is then a lookup in the static cross-matrices: a4[u] - M[u, donor(u)]
(target minus imposter NLL, negative favours the target), averaged over seeds
and users, with a user bootstrap interval.

  python -m attribution_trials.validation.nearest_imposter
Output: <VALIDATION>/nearest_imposter.json
"""

from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np

from attribution_trials import paths
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import session_split_indices
from attribution_trials.policy.board_native import _square_index

MATRICES = paths.VALIDATION / "static_matrices"
KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"

SEED = 0
BOOT_N = 10000
TRAIN_FRAC = 0.7

# file stem -> (combination key, feature space)
CELLS = {
    "M_static_chess-pooled_move": ("static-embedding|chess-pooled|move", "chess"),
    "M_static_chess-pooled_timing": ("static-embedding|chess-pooled|timing", "chess"),
    "M_static_kt_move": ("static-embedding|kt|move", "kt"),
    "M_static_kt-rt_move": ("static-embedding|kt-rt|move", "kt-rt"),
    "M_static_kt-rt_timing": ("static-embedding|kt-rt|timing", "kt-rt"),
}


def boot_mean(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(BOOT_N)]
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


# ------------------------------------------------------ train features --


def chess_features() -> dict[str, np.ndarray]:
    from attribution_trials.data.pooled_chess import build_pooled_chess

    data, _, _, _ = build_pooled_chess()
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    out = {}
    for traj, cut in zip(trajs, splits):
        pairs: Counter = Counter()
        logt = []
        for i, (decision, obs) in enumerate(zip(traj.decisions, traj.observations)):
            if i >= cut:
                break
            legal = decision.legal_actions
            move = obs.move if obs.move in legal else legal[0]
            pairs[(_square_index(move[:2]), _square_index(move[2:4]))] += 1
            spent = obs.time_spent
            logt.append(math.log(max(float(spent) if spent is not None else 1e-3, 1e-3)))
        if not logt or not pairs:
            continue
        times = np.array(logt)
        counts = np.array(list(pairs.values()), dtype=float)
        share = counts / counts.sum()
        out[traj.player_id] = np.array(
            [
                math.log(len(times)),
                float(times.mean()),
                float(times.std()),
                float(-(share * np.log(share)).sum()),  # move-choice entropy
                float(share.max()),  # top-move share
            ]
        )
    return out


def kt_features(rt_col: int | None) -> dict[str, np.ndarray]:
    data = load_kt_csv(
        str(KT_DATA),
        n_students=500,
        min_responses=50,
        response_time_col=rt_col,
    )
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    out = {}
    for traj, cut in zip(trajs, splits):
        correct = [1.0 if o.move == "correct" else 0.0 for o in traj.observations[:cut]]
        if not correct:
            continue
        row = [math.log(len(correct)), float(np.mean(correct))]
        if rt_col is not None:
            logt = np.array(
                [math.log(max(float(o.time_spent), 1e-3)) for o in traj.observations[:cut]]
            )
            row += [float(logt.mean()), float(logt.std())]
        out[traj.player_id] = np.array(row)
    return out


def nearest_within_cell(features, players, cell_of):
    """Nearest other user in z-scored feature space, inside the same cell."""
    matrix = np.array([features[p] for p in players], dtype=float)
    centre = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (matrix - centre) / scale

    donor = {}
    by_cell = {}
    for index, player in enumerate(players):
        by_cell.setdefault(cell_of[player], []).append(index)
    for cell, members in by_cell.items():
        block = z[members]
        distance = np.sqrt(((block[:, None, :] - block[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(distance, np.inf)
        for position, index in enumerate(members):
            donor[players[index]] = players[members[int(distance[position].argmin())]]
    return donor


def analyse(stem, features) -> dict:
    key, _ = CELLS[stem]
    blob = np.load(MATRICES / f"{stem}.npz", allow_pickle=True)
    matrix = blob["M"]
    a4_seed = blob["a4"]
    players = [str(p) for p in blob["player_ids"]]
    cell_of = {p: str(c) for p, c in zip(players, blob["cell_of"])}
    keep = [str(u) for u in blob["keep_ids"]]
    position = {p: i for i, p in enumerate(players)}

    missing = [p for p in players if p not in features]
    if missing:
        raise SystemExit(
            f"{stem}: {len(missing)} matrix players absent from the rebuilt "
            f"training split (e.g. {missing[:3]}); the cohort build does not "
            "match the matrices"
        )

    donor = nearest_within_cell(features, players, cell_of)
    a4 = np.array([a4_seed[:, position[u]].mean() for u in keep])
    a3 = np.array([float(np.nanmean(matrix[:, position[u], position[donor[u]]])) for u in keep])
    if not np.all(np.isfinite(a3)):
        raise SystemExit(f"{stem}: nearest donor outside the scored cell")

    point, low, high = boot_mean(a4 - a3)
    return {
        "cell": key,
        "n_users": len(keep),
        "n_features": int(len(next(iter(features.values())))),
        "dindiv_nn": {"point": point, "low": low, "high": high},
    }


def main() -> int:
    features = {
        "chess": chess_features(),
        "kt": kt_features(None),
        "kt-rt": kt_features(2),
    }
    rows = []
    for stem, (key, domain) in CELLS.items():
        print(f"[nearest imposter] {key}", flush=True)
        rows.append(analyse(stem, features[domain]))

    out = paths.ensure(paths.VALIDATION) / "nearest_imposter.json"
    out.write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_bootstrap": BOOT_N,
                "train_frac": TRAIN_FRAC,
                "candidate_pool": "registered cell",
                "cells": rows,
            },
            indent=1,
        )
    )
    for row in rows:
        print(
            f"  {row['cell']:40s} {row['dindiv_nn']['point']:+.4f} "
            f"[{row['dindiv_nn']['low']:+.4f}, {row['dindiv_nn']['high']:+.4f}]"
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
