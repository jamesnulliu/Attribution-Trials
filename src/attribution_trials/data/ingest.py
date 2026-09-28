"""Ingest driver: Lichess archive -> persisted chess trajectory dataset.

Two passes over one ``.pgn`` / ``.pgn.zst`` archive:

1. **Header-only pass** (:func:`iter_game_summaries`): bucket every player,
   count games + sessions, and pick the cohort with enough volume *and*
   multi-session history (the later-sessions split needs both). Restricted
   to a single speed class so think-time stays comparable.
2. **Full parse pass** (sharded, :func:`iter_game_records_parallel`): parse
   only the cohort's games into trajectories and persist them.

Trajectories carry ``fen_before`` + ``legal_actions`` per move, which is all
move and think-time scoring needs; no engine evaluation is attached.

Output (``out_dir``)::

    dataset.jsonl.gz   # the TrajectoryDataset (attribution_trials.data.store format)
    manifest.json      # archive + params + per-player counts
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from attribution_trials.data.lichess import (
    bucket_by_player,
    build_dataset,
    iter_game_records_parallel,
    iter_game_summaries,
    open_pgn,
    player_stats,
    select_players,
    speed_class,
)
from attribution_trials.data.sessions import DEFAULT_GAP_THRESHOLD_SECONDS
from attribution_trials.data.store import save_dataset


def _select_cohort(
    archive_path: str,
    *,
    speed: str | None,
    min_games: int,
    min_sessions: int,
    max_players: int | None,
    gap_threshold_seconds: float,
    max_games: int | None,
) -> tuple[list[str], dict, int]:
    """Pass 1: header-only stats -> selected cohort.

    Returns ``(cohort, stats_by_player, n_games_scanned)``. ``stats_by_player``
    is keyed by player id (only the selected cohort is retained, to keep the
    manifest small).
    """
    scanned = [0]

    def _counted(itr):
        for s in itr:
            scanned[0] += 1
            yield s

    with open_pgn(archive_path) as stream:
        summaries = _counted(iter_game_summaries(stream, max_games=max_games))
        if speed is not None:
            summaries = (s for s in summaries if speed_class(s.time_control) == speed)
        buckets = bucket_by_player(summaries)

    stats = player_stats(buckets, gap_threshold_seconds=gap_threshold_seconds)
    cohort = select_players(stats, min_games=min_games, min_sessions=min_sessions)
    if max_players is not None:
        cohort = cohort[:max_players]
    cohort_stats = {p: asdict(stats[p]) for p in cohort}
    return cohort, cohort_stats, scanned[0]


def run_ingest(
    archive_path: str,
    out_dir: str,
    *,
    speed: str | None = "blitz",
    min_games: int = 50,
    min_sessions: int = 3,
    max_players: int | None = None,
    gap_threshold_seconds: float = DEFAULT_GAP_THRESHOLD_SECONDS,
    workers: int = 1,
    batch_size: int = 512,
    max_games: int | None = None,
    max_games_per_player: int | None = None,
    dataset_name: str = "dataset.jsonl.gz",
    verbose: bool = True,
) -> dict:
    """Run the two-pass ingest; write the dataset + manifest; return manifest.

    Parameters
    ----------
    archive_path:
        ``.pgn`` or ``.pgn.zst`` Lichess monthly archive.
    out_dir:
        Directory to create and write ``dataset.jsonl.gz`` + ``manifest.json``.
    speed:
        Single Lichess speed class to keep (``blitz`` default; ``None`` keeps
        all -- discouraged, mixes time scales). See
        :func:`attribution_trials.data.lichess.speed_class`.
    min_games / min_sessions:
        Cohort gates (volume + multi-session history).
    max_players:
        Optional cap; keep the top-N players by game count.
    workers:
        Processes for the pass-2 parse (the bottleneck). ``1`` = serial.
    max_games:
        Optional cap on games *read* per pass.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print(f"[ingest] pass 1 (cohort selection) over {archive_path} ...")
    cohort, cohort_stats, n_scanned = _select_cohort(
        archive_path,
        speed=speed,
        min_games=min_games,
        min_sessions=min_sessions,
        max_players=max_players,
        gap_threshold_seconds=gap_threshold_seconds,
        max_games=max_games,
    )
    t_pass1 = time.time() - t0
    if verbose:
        print(
            f"[ingest] pass 1 done: {n_scanned} games scanned, "
            f"{len(cohort)} players selected ({t_pass1:.1f}s)."
        )
    if not cohort:
        raise SystemExit(
            "No players cleared the cohort gates "
            f"(min_games={min_games}, min_sessions={min_sessions}, "
            f"speed={speed}). Loosen the gates or scan more games."
        )

    if verbose:
        print(f"[ingest] pass 2 (full parse, workers={workers}) for {len(cohort)} players ...")
    t1 = time.time()
    cohort_set = set(cohort)
    records = iter_game_records_parallel(
        archive_path,
        players=cohort_set,
        speed=speed,
        workers=workers,
        batch_size=batch_size,
        max_games=max_games,
    )
    buckets = bucket_by_player(records, players=cohort_set)
    dataset = build_dataset(
        buckets,
        players=cohort,
        gap_threshold_seconds=gap_threshold_seconds,
        max_games_per_player=max_games_per_player,
    )
    t_pass2 = time.time() - t1

    dataset_path = os.path.join(out_dir, dataset_name)
    save_dataset(dataset, dataset_path)

    # Per-player decision counts (the unit experiments actually consume).
    per_player = {}
    total_decisions = 0
    for traj in dataset.trajectories:
        n = len(traj.decisions)
        total_decisions += n
        entry = dict(cohort_stats.get(traj.player_id, {}))
        entry["n_decisions"] = n
        per_player[traj.player_id] = entry

    manifest = {
        "archive": os.path.abspath(archive_path),
        "dataset": os.path.basename(dataset_path),
        "params": {
            "speed": speed,
            "min_games": min_games,
            "min_sessions": min_sessions,
            "max_players": max_players,
            "gap_threshold_seconds": gap_threshold_seconds,
            "workers": workers,
            "batch_size": batch_size,
            "max_games": max_games,
            "max_games_per_player": max_games_per_player,
        },
        "n_games_scanned_pass1": n_scanned,
        "n_players_selected": len(cohort),
        "n_trajectories": len(dataset.trajectories),
        "n_decisions_total": total_decisions,
        "seconds_pass1": round(t_pass1, 2),
        "seconds_pass2": round(t_pass2, 2),
        "players": per_player,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    if verbose:
        print(
            f"[ingest] pass 2 done: {len(dataset.trajectories)} trajectories, "
            f"{total_decisions} decisions ({t_pass2:.1f}s)."
        )
        print(f"[ingest] wrote {dataset_path}")
        print(f"[ingest] wrote {manifest_path}")
    return manifest
