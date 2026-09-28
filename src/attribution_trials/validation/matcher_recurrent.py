"""Full within-cell donor cross-matrix for the recurrent embedding on KT-RT.

Every user's evaluation rows are scored under every same-cell donor's
boundary-frozen latent stream.  The diagonal is the target (A4frozen), so the
gain under any matcher M is a lookup:
    like(M)[u] = M[u, u] - M[u, M(u)].
Latents are extracted once, and a rotation map per offset scores one full
pass, so a cell of size m is covered by m-1 passes.  The registered imposter
(A3) is scored in the same process.

Inputs: the seed's target (A4) checkpoint the recurrent-embedding KT-RT audit
run saves under <AUDIT>/chess_kt/runs/bench-evolving-kt-rt-s<seed>-a4/<run id>/
artifacts/checkpoint.pt, and <DATA>/kt/prepared/assist09.tsv.

  python -m attribution_trials.validation.matcher_recurrent <seed>
Output: <VALIDATION>/matchers/cross_evolving_kt-rt_s<seed>.npz
"""

from __future__ import annotations

import sys
import time
from functools import partial

import numpy as np
import torch

from attribution_trials import paths
from attribution_trials.bench.cells import assign_tertile_cells, kt_prior_accuracy_scalar
from attribution_trials.bench.evolving_family import (
    PrecomputedLatentInjector,
    build_arm_latents,
    extract_latents,
)
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import _per_player_nlls, session_split_indices
from attribution_trials.latent.base import InjectionKind
from attribution_trials.latent.neural import NeuralInjector
from attribution_trials.policy.kt_backbone import KTBackbone

RUNS = paths.AUDIT / "chess_kt" / "runs"
KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
OUT = paths.VALIDATION / "matchers"

TRAIN_FRAC = 0.7
LATENT_DIM = 16
HIDDEN_DIM = 64
BATCH = 16
CHANNELS = {"response": 0, "timing": 1}


def load_arm(seed: int):
    pattern = f"bench-evolving-kt-rt-s{seed}-a4"
    checkpoints = sorted((RUNS / pattern).glob("*/artifacts/checkpoint.pt"))
    if not checkpoints:
        raise SystemExit(f"no checkpoint for {RUNS / pattern}")
    blob = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    injector = NeuralInjector(
        kind=InjectionKind.HIDDEN, latent_dim=LATENT_DIM, seed=seed, persist=True
    )
    injector._build().load_state_dict(blob["injector"])
    backbone = KTBackbone(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, seed=seed)
    backbone._build().load_state_dict(blob["backbone"])
    return injector, backbone


def main() -> int:
    seed = int(sys.argv[1])
    t0 = time.time()
    data = load_kt_csv(str(KT_DATA), n_students=500, min_responses=50, response_time_col=5)
    cells = assign_tertile_cells(
        data, partial(kt_prior_accuracy_scalar, train_frac=TRAIN_FRAC), prefix="pacc"
    )
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    split_of = {t.player_id: k for t, k in zip(trajs, splits)}
    players = [t.player_id for t in trajs]
    index_of = {p: i for i, p in enumerate(players)}

    injector, backbone = load_arm(seed)

    latents = extract_latents(injector, backbone, data, batch_size=BATCH)
    arms, singletons = build_arm_latents(latents, split_of, cells, seed)
    frozen = arms["A4frozen"]

    def score(latent_map):
        served = PrecomputedLatentInjector(injector, latent_map)
        return _per_player_nlls(served, backbone, data, splits, batch_size=BATCH)

    scored_a3 = score(arms["A3"])
    print(f"[s{seed}] registered imposter scored ({time.time() - t0:.0f}s)", flush=True)

    members = {}
    for p, c in cells.items():
        members.setdefault(c, []).append(p)
    for c in members:
        members[c].sort(key=lambda p: index_of[p])
    max_m = max(len(v) for v in members.values())

    n = len(players)
    M = {ch: np.full((n, n), np.nan) for ch in CHANNELS}
    # diagonal = A4frozen
    scored_own = score(frozen)
    for ch, ci in CHANNELS.items():
        vals = np.array(scored_own[ci])
        M[ch][np.arange(n), np.arange(n)] = vals

    for k in range(1, max_m):
        latent_map, donor_of = {}, {}
        for c, mem in members.items():
            m = len(mem)
            for i, p in enumerate(mem):
                d = mem[(i + k) % m]
                donor_of[p] = d
                latent_map[p] = frozen[d]
        scored = score(latent_map)
        for ch, ci in CHANNELS.items():
            vals = np.array(scored[ci])
            for p in players:
                M[ch][index_of[p], index_of[donor_of[p]]] = vals[index_of[p]]
        if k % 25 == 0 or k == max_m - 1:
            print(f"[s{seed}] pass {k}/{max_m - 1} ({time.time() - t0:.0f}s)", flush=True)

    out = paths.ensure(OUT) / f"cross_evolving_kt-rt_s{seed}.npz"
    np.savez_compressed(
        out,
        M_response=M["response"],
        M_timing=M["timing"],
        players=np.array(players),
        cell_of=np.array([cells[p] for p in players]),
        singletons=np.array(sorted(singletons)),
        a3_reg_response=np.array(scored_a3[0]),
        a3_reg_timing=np.array(scored_a3[1]),
    )
    print(f"[s{seed}] wrote {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
