"""Pooled chess panel: three Lichess blitz cohorts (2017-04, 2019-07, 2021-06).

Each cohort is 100 players ingested with identical settings
(``scripts/data/lichess.sh``). A player present in more than one cohort is
kept only in the earliest, so every trajectory is a distinct user.

Groups are cohort x Elo tertile (9 cells), so the group and imposter
conditions never cross cohorts.
"""

from __future__ import annotations

from functools import partial

from attribution_trials import paths
from attribution_trials.bench.cells import assign_tertile_cells, elo_scalar
from attribution_trials.data.store import load_dataset
from attribution_trials.train.base import TrajectoryDataset

COHORTS = ("ec2017", "ec2019", "ec2021")
TRAIN_FRAC = 0.7


def build_pooled_chess():
    """Returns ``(data, cells, cohort_of, dropped_duplicates)``."""
    kept: dict[str, list] = {}
    seen: set[str] = set()
    dropped: list[tuple[str, str]] = []
    for c in COHORTS:  # chronological order, so dedup keeps the earliest
        traj = load_dataset(str(paths.DATA / "chess" / c / "dataset.jsonl.gz")).trajectories
        keep = []
        for t in traj:
            if t.player_id in seen:
                dropped.append((c, t.player_id))
                continue
            seen.add(t.player_id)
            keep.append(t)
        kept[c] = keep

    cells: dict[str, str] = {}
    cohort_of: dict[str, str] = {}
    for c in COHORTS:
        sub = TrajectoryDataset(trajectories=kept[c])
        cells.update(
            assign_tertile_cells(
                sub,
                partial(elo_scalar, train_frac=TRAIN_FRAC),
                prefix=f"{c}-elo",
            )
        )
        for t in kept[c]:
            cohort_of[t.player_id] = c

    data = TrajectoryDataset(trajectories=[t for c in COHORTS for t in kept[c]])
    return data, cells, cohort_of, dropped


def backbone_factory(seed: int):
    from attribution_trials.policy.board_native import BoardNativeBackbone

    return BoardNativeBackbone(latent_dim=16, hidden_dim=64, seed=seed)
