"""Per-option rescoring of the chess latent families (input to the coverage table).

Trains one condition on the pooled chess panel with the latent-family training
configuration (15 epochs, lr 1e-2, batch 16, latent 16, hidden 64, timing
lambda 0.5, session split 0.7, same-seed backbone initialization) and stores,
for every held-out decision, the timing head's log-normal distribution
discretized over the think-time buckets (edges 1, 3, 8, 20 s) together with the
observed bucket. Runs on CPU.

Conditions: a0 (no user information), static (per-user embedding),
evolving (recurrent per-user state).

  python -m attribution_trials.comparison.rescore_chess <a0|static|evolving> <seed>

Output: <COMPARISON>/coverage_inputs/cpu/chess_<condition>_s<seed>.npz
Skips a cell whose output exists.
"""

from __future__ import annotations

import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

from attribution_trials import paths  # noqa: E402
from attribution_trials.data.pooled_chess import TRAIN_FRAC, build_pooled_chess  # noqa: E402
from attribution_trials.experiments.ec import session_split_indices  # noqa: E402
from attribution_trials.latent.base import InjectionKind  # noqa: E402
from attribution_trials.latent.neural import NeuralInjector  # noqa: E402
from attribution_trials.latent.static_individual import StaticIndividualInjector  # noqa: E402
from attribution_trials.policy.board_native import BoardNativeBackbone  # noqa: E402
from attribution_trials.train.base import TrainConfig  # noqa: E402
from attribution_trials.train.sft import EvalSpec, SFTTrainer  # noqa: E402

OUT = paths.COMPARISON / "coverage_inputs" / "cpu"
CHESS_EDGES = [1.0, 3.0, 8.0, 20.0]
EPOCHS, LR, BATCH, LATENT, HIDDEN, TLAMBDA = 15, 1e-2, 16, 16, 64, 0.5


def make_injector(arm: str, player_ids, seed: int):
    if arm == "a0":
        return NeuralInjector(
            kind=InjectionKind.HIDDEN, latent_dim=LATENT, seed=seed, persist=False
        )
    if arm == "static":
        return StaticIndividualInjector(player_ids, latent_dim=LATENT, seed=seed)
    if arm == "evolving":
        return NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=LATENT, seed=seed, persist=True)
    raise ValueError(arm)


def bucket_probs_lognormal(mu, sigma, edges):
    """[..., n_edges+1] bucket probabilities of a log-normal (mu on ln seconds)."""
    z = torch.stack([(math.log(e) - mu) / sigma for e in edges], dim=-1)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    lo = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    hi = torch.cat([cdf, torch.ones_like(cdf[..., :1])], dim=-1)
    return (hi - lo).clamp_min(0.0)


def bucket_of(seconds: float) -> int:
    for i, e in enumerate(CHESS_EDGES):
        if seconds < e:
            return i
    return len(CHESS_EDGES)


def main() -> int:
    arm, seed = sys.argv[1], int(sys.argv[2])
    out = paths.ensure(OUT) / f"chess_{arm}_s{seed}.npz"
    if out.exists():
        print(f"[skip] {out.name} exists", flush=True)
        return 0
    t0 = time.time()

    data, _, _, _ = build_pooled_chess()
    trajs = data.trajectories
    player_ids = [t.player_id for t in trajs]
    splits = session_split_indices(trajs, TRAIN_FRAC)

    backbone = BoardNativeBackbone(latent_dim=LATENT, hidden_dim=HIDDEN, seed=seed)
    injector = make_injector(arm, player_ids, seed)
    cfg = TrainConfig(
        epochs=EPOCHS,
        lr=LR,
        seed=seed,
        batch_size=BATCH,
        experiment=f"rescore-chess-{arm}-s{seed}",
        extra={"timing_lambda": TLAMBDA, "arm": arm},
    )
    SFTTrainer(injector, backbone, cfg).fit(data, eval_spec=EvalSpec(dataset=data, splits=splits))
    print(f"[chess/{arm}/s{seed}] trained ({time.time() - t0:.0f}s)", flush=True)

    backbone._build().eval()
    injector._build().eval()
    dec_player, dec_t, tm_probs, tm_true, tm_secs = [], [], [], [], []
    indexed = sorted(
        enumerate(zip(trajs, splits)), key=lambda kv: len(kv[1][0].decisions), reverse=True
    )
    for i in range(0, len(indexed), BATCH):
        chunk = indexed[i : i + BATCH]
        batch = backbone.encode_batch([t for _, (t, _) in chunk])
        _, eval_mask = backbone.train_eval_masks(batch, [s for _, (_, s) in chunk])
        with torch.no_grad():
            latent = injector.latent_trajectory(batch.feats, player_ids=batch.player_ids)
            mu, sigma = backbone.timing_mu_sigma(latent)  # [T,B], scalar
            bp = bucket_probs_lognormal(mu, sigma, CHESS_EDGES)  # [T,B,5]
        for b, (_, (traj, _)) in enumerate(chunk):
            for st in np.nonzero(eval_mask[:, b].numpy().astype(bool))[0]:
                tp = bp[st, b].numpy().astype(np.float32)
                assert abs(float(tp.sum()) - 1.0) < 1e-3
                sec = float(batch.times[st, b])
                dec_player.append(traj.player_id)
                dec_t.append(int(st))
                tm_probs.append(tp)
                tm_secs.append(sec)
                tm_true.append(bucket_of(sec))

    uniq = sorted(set(dec_player))
    pidx = {p: j for j, p in enumerate(uniq)}
    np.savez_compressed(
        out,
        players=np.array(uniq),
        dec_player_idx=np.array([pidx[p] for p in dec_player], np.int16),
        dec_t=np.array(dec_t, np.int32),
        timing_bucket_probs=np.stack(tm_probs),
        timing_true_bucket=np.array(tm_true, np.int8),
        timing_seconds=np.array(tm_secs, np.float32),
    )
    print(
        f"chess/{arm}/s{seed}: {len(dec_player)} decisions ({time.time() - t0:.0f}s) -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
