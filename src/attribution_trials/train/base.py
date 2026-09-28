"""Trainer interface + dataset abstraction.

A trainer fits the parameters of a
:class:`~attribution_trials.latent.base.LatentStateInjector` together with
its backbone so that they reproduce each user's observed actions and timing.
The dataset is a sequence of per-user, chronologically ordered trajectories;
the trainer must never train on a user's held-out sessions.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from attribution_trials.interface import DecisionPoint
from attribution_trials.latent.base import LatentStateInjector, Observation
from attribution_trials.policy.base import PolicyBackbone
from attribution_trials.results import DEFAULT_ROOT


@dataclass
class Trajectory:
    """One user's ordered decisions + ground-truth observations."""

    player_id: str
    decisions: list[DecisionPoint]
    observations: list[Observation]


@dataclass
class TrajectoryDataset:
    """A collection of per-user trajectories in time order."""

    trajectories: list[Trajectory] = field(default_factory=list)

    def players(self) -> set[str]:
        return {t.player_id for t in self.trajectories}

    def __len__(self) -> int:
        return len(self.trajectories)


@dataclass
class TrainConfig:
    """Common training hyperparameters."""

    epochs: int = 3
    lr: float = 1e-4
    batch_size: int = 8
    grad_clip: float = 1.0
    seed: int = 0
    # Logical experiment name; becomes the result-store sub-directory and the
    # W&B group. Defaults to the trainer class name.
    experiment: str | None = None
    # Root of the local per-run store (see attribution_trials.results).
    results_root: str = DEFAULT_ROOT
    # Optional W&B project/entity overrides (else read from env / default).
    wandb_project: str | None = None
    wandb_entity: str | None = None
    extra: dict = field(default_factory=dict)


class Trainer(abc.ABC):
    """Fits an injector and its backbone to trajectories.

    Every concrete ``fit`` calls :meth:`begin_run` first, which opens a local
    run directory (:mod:`attribution_trials.results`) and, when
    ``WANDB_API_KEY`` is set, a W&B run (:mod:`attribution_trials.tracking`).
    """

    def __init__(
        self,
        injector: LatentStateInjector,
        backbone: PolicyBackbone,
        config: TrainConfig | None = None,
    ) -> None:
        self.injector = injector
        self.backbone = backbone
        self.config = config or TrainConfig()

    @property
    def experiment_name(self) -> str:
        return self.config.experiment or type(self).__name__

    def run_config(self) -> dict:
        """The resolved config dict recorded to the result store + W&B."""
        from dataclasses import asdict

        cfg = asdict(self.config)
        cfg.update(
            {
                "trainer": type(self).__name__,
                "injector": type(self.injector).__name__,
                "backbone": self.backbone.name,
            }
        )
        return cfg

    def begin_run(self, dataset: TrajectoryDataset):
        """Open a local run directory and (optionally) a W&B run.

        Returns ``(run_handle, wandb_run)``. Concrete ``fit`` implementations
        stream metrics via ``wandb_run.log(...)`` / ``wandb_run.summary(...)``
        and end with ``wandb_run.finish(...)`` + ``run_handle.finalize(...)``.
        """
        from attribution_trials.results import ResultStore
        from attribution_trials.tracking import start_run

        config = self.run_config()
        handle = ResultStore(self.config.results_root).create(self.experiment_name, config)
        wandb_run = start_run(
            experiment=self.experiment_name,
            config=config,
            handle=handle,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
        )
        return handle, wandb_run

    @abc.abstractmethod
    def fit(self, dataset: TrajectoryDataset) -> dict:
        """Train; return a metrics/summary dict. Must not touch test data."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist trained parameters."""
