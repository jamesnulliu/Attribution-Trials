"""Group assignment for the group and imposter conditions.

Users are split into tertile groups ("cells") of a per-user scalar computed
on the train split only. The group condition uses the user's group; the
imposter is drawn from the same group.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

from attribution_trials.experiments.ec import session_split_indices
from attribution_trials.train.base import Trajectory, TrajectoryDataset


def assign_tertile_cells(
    dataset: TrajectoryDataset,
    scalar_of: Callable[[object], float],
    n_bins: int = 3,
    prefix: str = "cell",
) -> dict[str, str]:
    """Tertile (n-tile) cells from a deterministic per-player scalar.

    `scalar_of(trajectory) -> float` defines the axis (Elo for chess,
    train-split prior accuracy for KT). Ties broken by user id for
    determinism.
    """
    scored = sorted(
        ((scalar_of(t), t.player_id) for t in dataset.trajectories),
    )
    n = len(scored)
    cells: dict[str, str] = {}
    for rank, (_, pid) in enumerate(scored):
        b = min(n_bins - 1, rank * n_bins // max(1, n))
        cells[pid] = f"{prefix}{b}"
    return cells


def _train_split_of(traj: Trajectory, train_frac: float) -> int:
    return session_split_indices([traj], train_frac)[0]


def elo_scalar(traj: Trajectory, train_frac: float = 0.7) -> float:
    """Median ``player_elo`` over the player's TRAIN-split decisions.

    ``train_frac`` must match the held-out split so cell assignment never
    reads held-out decisions. A player with no rated train-split decision
    raises.
    """
    k = _train_split_of(traj, train_frac)
    elos = [
        d.context.get("player_elo")
        for d in traj.decisions[:k]
        if d.context.get("player_elo") is not None
    ]
    if not elos:
        raise ValueError(f"no train-split Elo for player {traj.player_id}")
    return float(statistics.median(elos))


def kt_prior_accuracy_scalar(traj: Trajectory, train_frac: float = 0.7) -> float:
    """Mean correctness over the student's TRAIN-split responses."""
    k = _train_split_of(traj, train_frac)
    obs = traj.observations[:k]
    if not obs:
        raise ValueError(f"no train-split responses for {traj.player_id}")
    return sum(o.move == "correct" for o in obs) / len(obs)
