"""Recurrent-embedding family: the five conditions over a learned latent.

The recurrent injector is a shared GRU with NO per-user parameters -- its
conditioning is the history it is run over. Conditions therefore replace the
latent-state TRAJECTORY at evaluation time:

    A0        none: memoryless twin (persist=False), its own training run
    A1        population: mean state trajectory (TRAIN-split states only)
    A2        group: group-mean state trajectory (same synthesis)
    A3        imposter: a same-group donor's TRAIN states, index-aligned,
              last state held beyond their split
    A4        target, updated: own state, updating through the held-out rows
    A4frozen  target: own state held at the split boundary

A1..A4frozen all score through the SAME trained A4 model; only the latent
stream differs. Condition payloads are built from pre-split data only,
except updated A4, which updates through the held-out rows.
"""

from __future__ import annotations

import numpy as np

from attribution_trials.bench.swap import within_cell_derangement
from attribution_trials.bench.telescope import telescope_components
from attribution_trials.experiments.ec import _per_player_nlls, session_split_indices
from attribution_trials.latent.base import InjectionKind
from attribution_trials.latent.neural import NeuralInjector
from attribution_trials.train.base import TrainConfig
from attribution_trials.train.sft import EvalSpec, SFTTrainer


class PrecomputedLatentInjector:
    """Serve stored per-player latent trajectories (evaluation-time).

    `latent_of`: player_id -> np.ndarray [T_i, latent_dim]. For steps
    beyond the stored length the last stored state is held (imposter and
    population/group streams are shorter than the scored trajectory by
    construction). Delegates `_build` to the trained base injector so
    the scoring loop's `.eval()` call reaches a real module.
    """

    def __init__(self, base, latent_of: dict[str, np.ndarray]):
        self.base = base
        self.latent_of = latent_of
        self.kind = base.kind
        self.produces = base.produces
        self.latent_dim = base.latent_dim

    def latent_trajectory(self, feats_seq, player_ids=None):
        import torch

        steps, batch, _ = feats_seq.shape
        if player_ids is None:
            raise ValueError("PrecomputedLatentInjector needs player_ids")
        out = feats_seq.new_zeros(steps, batch, self.latent_dim)
        for b, pid in enumerate(player_ids):
            z = self.latent_of[pid]
            t_have = min(steps, len(z))
            zt = torch.as_tensor(z, dtype=feats_seq.dtype, device=feats_seq.device)
            out[:t_have, b] = zt[:t_have]
            if steps > t_have:
                out[t_have:, b] = zt[t_have - 1]
        return out

    def parameters(self):
        return self.base.parameters()

    def to(self, device):
        self.base.to(device)
        return self

    def _build(self):
        return self.base._build()

    @property
    def name(self) -> str:
        return "PrecomputedLatent"


def extract_latents(injector, backbone, dataset, batch_size: int = 16) -> dict[str, np.ndarray]:
    """Per-player latent trajectories [T_i, L] from a trained injector."""
    import torch

    device = next(backbone.parameters()).device
    out: dict[str, np.ndarray] = {}
    trajs = dataset.trajectories
    order = sorted(range(len(trajs)), key=lambda i: len(trajs[i].decisions), reverse=True)
    for i in range(0, len(order), batch_size):
        idxs = order[i : i + batch_size]
        chunk = [trajs[k] for k in idxs]
        batch = backbone.encode_batch(chunk).to(device)
        with torch.no_grad():
            lat = injector.latent_trajectory(batch.feats, player_ids=batch.player_ids)
        for b, k in enumerate(idxs):
            t = len(trajs[k].decisions)
            out[trajs[k].player_id] = lat[:t, b].detach().cpu().numpy().copy()
    return out


def mean_state_trajectory(latents: dict[str, np.ndarray], split_of: dict[str, int]) -> np.ndarray:
    """Index-aligned mean over members' TRAIN-split states, last held."""
    horizon = max(split_of.values())
    dim = next(iter(latents.values())).shape[1]
    out = np.zeros((horizon, dim))
    last = np.zeros(dim)
    for t in range(horizon):
        rows = [z[t] for pid, z in latents.items() if t < min(split_of[pid], len(z))]
        if rows:
            last = np.mean(rows, axis=0)
        out[t] = last
    return out


def build_arm_latents(
    latents: dict[str, np.ndarray],
    split_of: dict[str, int],
    cell_of: dict[str, str],
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], set[str]]:
    """Arm -> player -> served latent stream; plus singleton players."""
    ids = list(latents)
    derang = within_cell_derangement(cell_of, seed=seed)
    singletons = {p for p, q in derang.items() if p == q}

    cell_members: dict[str, list[str]] = {}
    for p, c in cell_of.items():
        cell_members.setdefault(c, []).append(p)
    pop = mean_state_trajectory(latents, split_of)
    cell_mean = {
        c: mean_state_trajectory(
            {p: latents[p] for p in members},
            {p: split_of[p] for p in members},
        )
        for c, members in cell_members.items()
    }

    arms: dict[str, dict[str, np.ndarray]] = {}
    arms["A1"] = {p: pop for p in ids}
    arms["A2"] = {p: cell_mean[cell_of[p]] for p in ids}
    arms["A3"] = {p: latents[derang[p]][: max(1, split_of[derang[p]])] for p in ids}
    arms["A4"] = {p: latents[p] for p in ids}
    arms["A4frozen"] = {p: latents[p][: max(1, split_of[p])] for p in ids}
    return arms, singletons


def run_evolving_family_ladder(
    full,
    *,
    backbone_factory,
    cell_of: dict[str, str],
    train_frac: float = 0.7,
    latent_dim: int = 16,
    epochs: int = 15,
    lr: float = 1e-2,
    batch_size: int = 16,
    seed: int = 0,
    timing_lambda: float = 0.5,
    bootstrap_n: int = 2000,
    permutations: int = 10000,
    label: str = "bench-evolving",
) -> dict:
    """Train the A0 twin + the A4 recurrent model; score every condition."""
    trajs = full.trajectories
    ids = [t.player_id for t in trajs]
    splits = session_split_indices(trajs, train_frac)
    split_of = {t.player_id: k for t, k in zip(trajs, splits)}

    def cfg(arm: str) -> TrainConfig:
        return TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=batch_size,
            experiment=f"{label}-{arm}",
            extra={"timing_lambda": timing_lambda, "arm": arm},
        )

    def train_arm(injector, arm):
        backbone = backbone_factory(seed)
        trainer = SFTTrainer(injector, backbone, cfg(arm))
        summary = trainer.fit(full, eval_spec=EvalSpec(dataset=full, splits=splits))
        return injector, backbone, summary

    a0 = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=seed,
        persist=False,
    )
    a0, bb0, s0 = train_arm(a0, "A0")
    mv0, tm0 = _per_player_nlls(a0, bb0, full, splits, batch_size=batch_size)

    a4 = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=seed,
        persist=True,
    )
    a4, bb4, s4 = train_arm(a4, "A4")

    latents = extract_latents(a4, bb4, full, batch_size=batch_size)
    arm_lat, singletons = build_arm_latents(latents, split_of, cell_of, seed)

    per_arm: dict[str, tuple[list[float], list[float]]] = {"A0": (mv0, tm0)}
    for arm, lat_map in arm_lat.items():
        inj = PrecomputedLatentInjector(a4, lat_map)
        per_arm[arm] = _per_player_nlls(inj, bb4, full, splits, batch_size=batch_size)

    out: dict = {
        "family": "evolving-latent",
        "seed": seed,
        "player_ids": ids,
        "singletons": sorted(singletons),
        "summaries": {"A0": s0, "A4": s4},
        "grades": {"eval-swap": {}},
        "drift": {},
        "per_arm": per_arm,
        "latents": latents,
        "split_of": split_of,
        "trained": (a4, bb4),
    }
    for ch, k in (("move", 0), ("timing", 1)):
        chain = {a: per_arm[a][k] for a in ("A0", "A1", "A2", "A3", "A4")}
        out["grades"]["eval-swap"][ch] = telescope_components(
            chain,
            ids,
            ch,
            bootstrap_n=bootstrap_n,
            permutations=permutations,
            seed=seed,
            exclude=singletons,
        )
        out["drift"][ch] = [f - l for f, l in zip(per_arm["A4frozen"][k], per_arm["A4"][k])]
    return out
