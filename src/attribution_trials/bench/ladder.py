"""Static-embedding family: build, train, and score the five conditions.

A4 (target) is a per-user static embedding. A1 (population) and A2 (group)
come in two constructions: "averaged" (row means of the trained A4
embedding, scored on the A4 model) and "retrained" (an embedding trained
under the remapped identity: one shared row, or one row per group). A3
(imposter) is a within-group derangement of the trained A4 identities; A0
(none) is the memoryless twin.

Every condition is scored with `ec._per_player_nlls` on identical splits;
all backbones share the same seed (identical init).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from attribution_trials.bench.swap import (
    IdRemapInjector,
    MeanLatentInjector,
    within_cell_derangement,
)
from attribution_trials.bench.telescope import LadderComponents, telescope_components
from attribution_trials.experiments.ec import _per_player_nlls, session_split_indices
from attribution_trials.latent.base import InjectionKind
from attribution_trials.latent.neural import NeuralInjector
from attribution_trials.latent.static_individual import StaticIndividualInjector
from attribution_trials.train.base import TrainConfig, TrajectoryDataset
from attribution_trials.train.sft import EvalSpec, SFTTrainer

POP_ID = "__population__"


def _train_arm_with(backbone_factory, injector, full, splits, cfg):
    backbone = backbone_factory(cfg.seed)
    trainer = SFTTrainer(injector, backbone, cfg)
    summary = trainer.fit(full, eval_spec=EvalSpec(dataset=full, splits=splits))
    move_pp, timing_pp = _per_player_nlls(
        injector, backbone, full, splits, batch_size=cfg.batch_size
    )
    return injector, backbone, summary, move_pp, timing_pp


@dataclass
class LadderRun:
    """One cohort x one family x one seed: per-arm scores + components."""

    family: str
    seed: int
    player_ids: list[str]
    cell_of: dict[str, str]
    singleton_players: set[str]
    # grade -> channel -> LadderComponents
    grades: dict[str, dict[str, LadderComponents]] = field(default_factory=dict)
    summaries: dict[str, dict] = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"== LadderRun family={self.family} seed={self.seed} "
            f"n={len(self.player_ids)} "
            f"singletons_excluded={sorted(self.singleton_players)}"
        ]
        for grade, chans in self.grades.items():
            for comp in chans.values():
                lines.append(f"-- grade={grade}")
                lines.append(comp.summary())
        return "\n".join(lines)


def run_static_family_ladder(
    full: TrajectoryDataset,
    *,
    backbone_factory,
    cell_of: dict[str, str],
    train_frac: float = 0.7,
    latent_dim: int = 8,
    epochs: int = 2,
    lr: float = 1e-2,
    batch_size: int = 8,
    seed: int = 0,
    timing_lambda: float = 0.5,
    bootstrap_n: int = 2000,
    permutations: int = 10000,
    constructions: tuple[str, ...] = ("averaged", "retrained"),
    label: str = "bench-static",
) -> LadderRun:
    player_ids = [t.player_id for t in full.trajectories]
    splits = session_split_indices(full.trajectories, train_frac)

    def cfg(arm: str) -> TrainConfig:
        return TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=batch_size,
            experiment=f"{label}-{arm}",
            extra={"timing_lambda": timing_lambda, "arm": arm},
        )

    # A0 (none) -- memoryless twin.
    a0 = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=seed,
        persist=False,
    )
    _, _, s0, mv0, tm0 = _train_arm_with(backbone_factory, a0, full, splits, cfg("A0"))

    # A4 (target) -- per-user static embedding (the trained base for the swaps).
    a4_base = StaticIndividualInjector(player_ids, latent_dim=latent_dim, seed=seed)
    a4_base, bb4, s4, mv4, tm4 = _train_arm_with(backbone_factory, a4_base, full, splits, cfg("A4"))

    # A3 (imposter) -- within-group derangement of the trained A4 model.
    derang = within_cell_derangement(cell_of, seed=seed)
    singletons = {p for p, q in derang.items() if p == q}
    a3 = IdRemapInjector(a4_base, derang)
    mv3, tm3 = _per_player_nlls(a3, bb4, full, splits, batch_size=batch_size)

    run = LadderRun(
        family="static-embedding",
        seed=seed,
        player_ids=player_ids,
        cell_of=dict(cell_of),
        singleton_players=singletons,
        summaries={"A0": s0, "A4": s4},
    )

    all_members = {p: player_ids for p in player_ids}
    cell_members: dict[str, list[str]] = {}
    for p, c in cell_of.items():
        cell_members.setdefault(c, []).append(p)
    per_cell = {p: cell_members[cell_of[p]] for p in player_ids}

    grade_arms: dict[str, dict[str, tuple[list[float], list[float]]]] = {}

    if "averaged" in constructions:
        a1 = MeanLatentInjector(a4_base, all_members)
        a2 = MeanLatentInjector(a4_base, per_cell)
        mv1, tm1 = _per_player_nlls(a1, bb4, full, splits, batch_size=batch_size)
        mv2, tm2 = _per_player_nlls(a2, bb4, full, splits, batch_size=batch_size)
        grade_arms["averaged"] = {
            "A0": (mv0, tm0),
            "A1": (mv1, tm1),
            "A2": (mv2, tm2),
            "A3": (mv3, tm3),
            "A4": (mv4, tm4),
        }

    if "retrained" in constructions:
        a1r = IdRemapInjector(
            StaticIndividualInjector([POP_ID], latent_dim=latent_dim, seed=seed),
            {p: POP_ID for p in player_ids},
        )
        _, _, s1r, mv1r, tm1r = _train_arm_with(backbone_factory, a1r, full, splits, cfg("A1r"))
        a2r = IdRemapInjector(
            StaticIndividualInjector(
                sorted(set(cell_of.values())), latent_dim=latent_dim, seed=seed
            ),
            dict(cell_of),
        )
        _, _, s2r, mv2r, tm2r = _train_arm_with(backbone_factory, a2r, full, splits, cfg("A2r"))
        run.summaries.update({"A1_retrained": s1r, "A2_retrained": s2r})
        grade_arms["retrained"] = {
            "A0": (mv0, tm0),
            "A1": (mv1r, tm1r),
            "A2": (mv2r, tm2r),
            "A3": (mv3, tm3),
            "A4": (mv4, tm4),
        }

    for grade, arms in grade_arms.items():
        run.grades[grade] = {}
        for ch, k in (("move", 0), ("timing", 1)):
            per_player = {a: arms[a][k] for a in arms}
            run.grades[grade][ch] = telescope_components(
                per_player,
                player_ids,
                ch,
                bootstrap_n=bootstrap_n,
                permutations=permutations,
                seed=seed,
                exclude=singletons,
            )
    return run
