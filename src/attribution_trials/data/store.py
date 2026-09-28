"""Persist + reload a :class:`~attribution_trials.train.base.TrajectoryDataset`.

The ingest driver (:mod:`attribution_trials.data.ingest`) parses a Lichess
archive into per-player trajectories once (python-chess legal-move generation
is the slow step) and writes the cohort here; every later step reads it back.

Format
------
Newline-delimited JSON (JSONL): **one trajectory per line**, so the file
streams (write/read without holding the whole corpus in memory) and a crashed
ingest still leaves a readable prefix. The path extension drives compression
-- ``.jsonl`` is plain, ``.jsonl.gz`` is gzip -- mirroring
:func:`attribution_trials.data.lichess.open_pgn`'s ``.zst`` handling.

Why a normalized line (not just ``dataclasses.asdict``)
-------------------------------------------------------
``build_trajectory`` shares **one** :class:`OutcomeStream` object across every
decision within a game (the per-game history snapshot). A naive per-decision
dump re-serializes that growing stream for every move -- O(moves x games),
tens of MB for a single active player. Instead each unique stream is written
once into ``streams`` and decisions reference it by index. On load the object
identity is restored (all decisions of a game share one stream again), so the
reloaded trajectory is structurally identical to a freshly built one. The
dedup keys on ``id()`` during a single pass; if a trajectory was built without
that sharing it still round-trips correctly, just larger.

Pure stdlib (``json`` + ``gzip``): no chess/torch import, so a persisted
dataset is inspectable and loadable on any box.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from typing import IO

from attribution_trials.interface import (
    DecisionPoint,
    EngineReference,
    Game,
    Outcome,
    OutcomeStream,
    TimeSignal,
)
from attribution_trials.latent.base import Observation
from attribution_trials.train.base import Trajectory, TrajectoryDataset

#: Bump when the on-disk trajectory format changes incompatibly.
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Leaf (de)serializers
# --------------------------------------------------------------------------- #


def _engine_ref_to_dict(ref: EngineReference | None) -> dict | None:
    if ref is None:
        return None
    return {
        "candidate_values": dict(ref.candidate_values),
        "best_move": ref.best_move,
        "best_value": ref.best_value,
        "unit": ref.unit,
        "depth": ref.depth,
    }


def _engine_ref_from_dict(d: dict | None) -> EngineReference | None:
    if d is None:
        return None
    return EngineReference(
        candidate_values={k: float(v) for k, v in d["candidate_values"].items()},
        best_move=d.get("best_move"),
        best_value=d.get("best_value"),
        unit=d.get("unit", "centipawn"),
        depth=d.get("depth"),
    )


def _time_signal_to_dict(ts: TimeSignal) -> dict:
    return {
        "time_remaining": ts.time_remaining,
        "increment": ts.increment,
        "time_spent": ts.time_spent,
        "move_number": ts.move_number,
        "byo_yomi_periods_left": ts.byo_yomi_periods_left,
        "byo_yomi_period_length": ts.byo_yomi_period_length,
        "phase": ts.phase,
    }


def _time_signal_from_dict(d: dict) -> TimeSignal:
    return TimeSignal(
        time_remaining=d.get("time_remaining"),
        increment=d.get("increment"),
        time_spent=d.get("time_spent"),
        move_number=d.get("move_number", 0),
        byo_yomi_periods_left=d.get("byo_yomi_periods_left"),
        byo_yomi_period_length=d.get("byo_yomi_period_length"),
        phase=d.get("phase"),
    )


def _outcome_to_dict(o: Outcome) -> dict:
    return {
        "won": o.won,
        "margin": o.margin,
        "engine_swing": o.engine_swing,
        "blunders": o.blunders,
        "time_scramble": o.time_scramble,
        "gap_to_next_seconds": o.gap_to_next_seconds,
    }


def _outcome_from_dict(d: dict) -> Outcome:
    return Outcome(
        won=d.get("won"),
        margin=d.get("margin"),
        engine_swing=d.get("engine_swing"),
        blunders=d.get("blunders", 0),
        time_scramble=d.get("time_scramble", False),
        gap_to_next_seconds=d.get("gap_to_next_seconds"),
    )


def _stream_to_dict(s: OutcomeStream) -> dict:
    return {
        "recent": [_outcome_to_dict(o) for o in s.recent],
        "session_position": s.session_position,
    }


def _stream_from_dict(d: dict) -> OutcomeStream:
    return OutcomeStream(
        recent=[_outcome_from_dict(o) for o in d.get("recent", [])],
        session_position=d.get("session_position", 0),
    )


# --------------------------------------------------------------------------- #
# Trajectory <-> dict
# --------------------------------------------------------------------------- #


def trajectory_to_dict(traj: Trajectory) -> dict:
    """Serialize one trajectory, deduping shared outcome streams by index."""
    streams: list[dict] = []
    stream_index: dict[int, int] = {}  # id(stream) -> position in `streams`

    decisions: list[dict] = []
    for dp in traj.decisions:
        key = id(dp.recent_outcomes)
        idx = stream_index.get(key)
        if idx is None:
            idx = len(streams)
            stream_index[key] = idx
            streams.append(_stream_to_dict(dp.recent_outcomes))
        decisions.append(
            {
                "game": dp.game.value,
                "state": dp.state,
                "legal_actions": list(dp.legal_actions),
                "engine_reference": _engine_ref_to_dict(dp.engine_reference),
                "time_signal": _time_signal_to_dict(dp.time_signal),
                "recent_outcomes": idx,
                "context": dp.context,
            }
        )

    observations = [{"move": o.move, "time_spent": o.time_spent} for o in traj.observations]
    return {
        "player_id": traj.player_id,
        "streams": streams,
        "decisions": decisions,
        "observations": observations,
    }


def trajectory_from_dict(d: dict) -> Trajectory:
    """Inverse of :func:`trajectory_to_dict`; restores stream sharing."""
    player_id = d["player_id"]
    streams = [_stream_from_dict(s) for s in d["streams"]]

    decisions: list[DecisionPoint] = []
    for dd in d["decisions"]:
        decisions.append(
            DecisionPoint(
                game=Game(dd["game"]),
                player_id=player_id,
                state=dd["state"],
                legal_actions=tuple(dd["legal_actions"]),
                engine_reference=_engine_ref_from_dict(dd.get("engine_reference")),
                time_signal=_time_signal_from_dict(dd["time_signal"]),
                # Re-alias: every decision of a game shares one stream object,
                # exactly as build_trajectory produced it.
                recent_outcomes=streams[dd["recent_outcomes"]],
                context=dd.get("context", {}),
            )
        )

    observations = [
        Observation(move=o["move"], time_spent=o.get("time_spent")) for o in d["observations"]
    ]
    return Trajectory(player_id=player_id, decisions=decisions, observations=observations)


# --------------------------------------------------------------------------- #
# Dataset file IO (gz-aware, streaming)
# --------------------------------------------------------------------------- #


def _open_text(path: str, mode: str) -> IO[str]:
    """Open ``path`` as text; gzip transparently when it ends in ``.gz``."""
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def save_dataset(dataset: TrajectoryDataset, path: str) -> None:
    """Write a dataset as one-trajectory-per-line JSONL (gz by extension).

    A schema/header line is written first so a reader can detect format drift;
    it carries ``{"schema_version", "n_trajectories"}`` and no trajectory data.
    """
    with _open_text(path, "w") as fh:
        header = {
            "schema_version": SCHEMA_VERSION,
            "n_trajectories": len(dataset.trajectories),
        }
        fh.write(json.dumps(header) + "\n")
        for traj in dataset.trajectories:
            fh.write(json.dumps(trajectory_to_dict(traj)) + "\n")


def iter_trajectories(path: str) -> Iterator[Trajectory]:
    """Stream trajectories from a persisted dataset without loading all."""
    with _open_text(path, "r") as fh:
        first = fh.readline()
        if not first:
            return
        head = json.loads(first)
        if "schema_version" not in head:
            # No header (e.g. hand-written file): treat the first line as data.
            yield trajectory_from_dict(head)
        elif head["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"trajectory store schema {head['schema_version']} != "
                f"expected {SCHEMA_VERSION}; re-run the ingest "
                "(python -m attribution_trials.cli ingest)."
            )
        for line in fh:
            line = line.strip()
            if line:
                yield trajectory_from_dict(json.loads(line))


def load_dataset(path: str) -> TrajectoryDataset:
    """Read a persisted dataset fully into a :class:`TrajectoryDataset`."""
    return TrajectoryDataset(trajectories=list(iter_trajectories(path)))


def save_trajectories(trajectories: Iterable[Trajectory], path: str) -> int:
    """Stream-write trajectories from an iterable; return the count.

    Convenience for an ingest that produces trajectories lazily and does not
    want to materialize a whole :class:`TrajectoryDataset` first.
    """
    trajectories = list(trajectories)
    save_dataset(TrajectoryDataset(trajectories=trajectories), path)
    return len(trajectories)
