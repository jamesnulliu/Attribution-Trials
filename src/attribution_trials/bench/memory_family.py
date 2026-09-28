"""Structured-memory family: synthesized memories for the five conditions.

The model is a pooled logistic readout over strictly causal running
statistics of a student's own answers (the memory block) plus decision
context. Conditions replace only the MEMORY block of the feature row --
bias, item difficulty, and the instantaneous history features are decision
context and stay the student's own in every condition:

    A0  none: neutral "no memory yet" values at every step
    A1  population: population-mean memory trajectory, synthesized from
        TRAIN-SPLIT rows only (no held-out answers of anyone)
    A2  group: same synthesis within the student's group
    A3  imposter: a same-group donor's own TRAIN memory rows
        (index-aligned; held at the donor's last train row)
    A4  target: the student's own memory -- "live" updates through the
        held-out rows; "frozen" stops at the split boundary

The readout is ONE pooled logistic over all students' train rows, so
per-user information enters only through the memory features, never
through per-user weights. Two grades: "refit" retrains the pooled readout
under each condition's features; "frozen-swap" fits it once under live A4
and only swaps features at evaluation. Feature dimension and readout are
identical across conditions.
"""

from __future__ import annotations

import math

import numpy as np

from attribution_trials.bench.swap import within_cell_derangement
from attribution_trials.bench.telescope import telescope_components
from attribution_trials.experiments.ec import session_split_indices
from attribution_trials.latent.structured import DIMENSIONS, history_features

N_MEM = 10  # memory-block features
NEUTRAL_MEM = [0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def memory_rows(traj) -> list[list[float]]:
    """Strictly causal memory-block rows (row t uses answers before t)."""
    rows = []
    n = 0
    sum_c = ewma_c = lag1_c = 0.0
    recent: list[float] = []
    streak = 0
    sum_r = ewma_r = lag1_r = 0.0
    for dp, obs in zip(traj.decisions, traj.observations):
        rows.append(
            [
                math.log1p(n) / 6.0,
                (sum_c / n) if n else 0.5,
                ewma_c if n else 0.5,
                (sum(recent) / len(recent)) if recent else 0.5,
                lag1_c,
                math.tanh(streak / 5.0),
                (sum_r / n) if n else 0.0,
                ewma_r,
                lag1_r,
                1.0 if n else 0.0,
            ]
        )
        correct = 1.0 if obs.move == "correct" else 0.0
        p_item = 1.0 - float(dp.state[0])
        resid = correct - p_item
        n += 1
        sum_c += correct
        ewma_c = correct if n == 1 else 0.9 * ewma_c + 0.1 * correct
        recent.append(correct)
        if len(recent) > 10:
            recent.pop(0)
        lag1_c = correct
        streak = streak + 1 if correct else min(streak, 0) - 1
        if correct and streak < 0:
            streak = 1
        sum_r += resid
        ewma_r = resid if n == 1 else 0.9 * ewma_r + 0.1 * resid
        lag1_r = resid
    return rows


def context_rows(traj) -> list[list[float]]:
    """Non-swapped feature block: bias, item difficulty, instant dims."""
    out = []
    for dp in traj.decisions:
        feats = history_features(dp)
        out.append([1.0, float(dp.state[0])] + [feats[d] for d in DIMENSIONS])
    return out


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact IRLS with a tiny ridge; deterministic."""
    w = np.zeros(x.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (y - p) - 1e-6 * w
        hess = (x * (p * (1.0 - p) + 1e-9)[:, None]).T @ x
        hess[np.diag_indices_from(hess)] += 1e-6
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def _mean_rows(
    member_mem: dict[str, list[list[float]]],
    member_split: dict[str, int],
) -> list[list[float]]:
    """Population/cell-mean memory trajectory from TRAIN rows only.

    Index t averages members' rows at t where t is still in that
    member's train split; indices beyond every member's split hold the
    last synthesized row (a frozen generic memory -- eval-period answers
    of no one are read).
    """
    horizon = max(member_split.values())
    out = []
    last = list(NEUTRAL_MEM)
    for t in range(horizon):
        rows = [mem[t] for pid, mem in member_mem.items() if t < member_split[pid]]
        if rows:
            last = [float(np.mean(col)) for col in zip(*rows)]
        out.append(list(last))
    return out


def _at(rows: list[list[float]], t: int) -> list[float]:
    return rows[t] if t < len(rows) else rows[-1] if rows else NEUTRAL_MEM


def pooled_event_rows(
    trajs: list,
    split_of: dict[str, int],
    seed: int = 0,
) -> list[list[float]]:
    """Alternative A1/A2 construction: run the memory UPDATE RULE over a
    pooled event stream (one seeded member event per step, TRAIN events
    only) instead of averaging feature rows. Same update code as
    memory_rows, so the constructions differ only in synthesis.
    """
    import random as _random

    rng = _random.Random(seed)
    horizon = max(split_of.values())
    rows = []
    n = 0
    sum_c = ewma_c = lag1_c = 0.0
    recent: list[float] = []
    streak = 0
    sum_r = ewma_r = lag1_r = 0.0
    for t in range(horizon):
        rows.append(
            [
                math.log1p(n) / 6.0,
                (sum_c / n) if n else 0.5,
                ewma_c if n else 0.5,
                (sum(recent) / len(recent)) if recent else 0.5,
                lag1_c,
                math.tanh(streak / 5.0),
                (sum_r / n) if n else 0.0,
                ewma_r,
                lag1_r,
                1.0 if n else 0.0,
            ]
        )
        donors = [tr for tr in trajs if t < split_of[tr.player_id]]
        if not donors:
            continue
        tr = donors[rng.randrange(len(donors))]
        dp, obs = tr.decisions[t], tr.observations[t]
        correct = 1.0 if obs.move == "correct" else 0.0
        p_item = 1.0 - float(dp.state[0])
        resid = correct - p_item
        n += 1
        sum_c += correct
        ewma_c = correct if n == 1 else 0.9 * ewma_c + 0.1 * correct
        recent.append(correct)
        if len(recent) > 10:
            recent.pop(0)
        lag1_c = correct
        streak = streak + 1 if correct else min(streak, 0) - 1
        if correct and streak < 0:
            streak = 1
        sum_r += resid
        ewma_r = resid if n == 1 else 0.9 * ewma_r + 0.1 * resid
        lag1_r = resid
    return rows


def run_memory_family_ladder(
    dataset,
    *,
    cell_of: dict[str, str],
    train_frac: float = 0.7,
    seed: int = 0,
    bootstrap_n: int = 2000,
    permutations: int = 10000,
    construction: str = "mean-rows",
) -> dict:
    """Score the memory-family conditions on the KT response channel.

    Returns {"grades": {grade: {"response": LadderComponents}},
    "drift": per-grade Δdrift columns (A4frozen - A4live per player),
    "singletons": [...]}.
    """
    trajs = dataset.trajectories
    splits = session_split_indices(trajs, train_frac)
    split_of = {t.player_id: k for t, k in zip(trajs, splits)}
    mem = {t.player_id: memory_rows(t) for t in trajs}
    ctx = {t.player_id: context_rows(t) for t in trajs}
    y = {t.player_id: [1.0 if o.move == "correct" else 0.0 for o in t.observations] for t in trajs}
    ids = [t.player_id for t in trajs]

    derang = within_cell_derangement(cell_of, seed=seed)
    singletons = {p for p, q in derang.items() if p == q}

    cell_members: dict[str, list[str]] = {}
    for p, c in cell_of.items():
        cell_members.setdefault(c, []).append(p)
    if construction == "pooled-events":
        pop_mean = pooled_event_rows(trajs, split_of, seed=seed)
        by_pid = {t.player_id: t for t in trajs}
        cell_mean = {
            c: pooled_event_rows(
                [by_pid[p] for p in members],
                {p: split_of[p] for p in members},
                seed=seed,
            )
            for c, members in cell_members.items()
        }
    else:
        pop_mean = _mean_rows(mem, split_of)
        cell_mean = {
            c: _mean_rows(
                {p: mem[p] for p in members},
                {p: split_of[p] for p in members},
            )
            for c, members in cell_members.items()
        }

    def arm_mem(pid: str, arm: str) -> list[list[float]]:
        n = len(mem[pid])
        k = split_of[pid]
        if arm == "A0":
            return [list(NEUTRAL_MEM) for _ in range(n)]
        if arm == "A1":
            return [_at(pop_mean, t) for t in range(n)]
        if arm == "A2":
            rows = cell_mean[cell_of[pid]]
            return [_at(rows, t) for t in range(n)]
        if arm == "A3":
            imp = derang[pid]
            imp_train = mem[imp][: split_of[imp]]
            return [_at(imp_train, t) for t in range(n)]
        if arm == "A4frozen":
            frozen = _at(mem[pid], max(0, k - 1))
            return [mem[pid][t] if t < k else list(frozen) for t in range(n)]
        return [list(r) for r in mem[pid]]  # A4 (live)

    def feats(pid: str, arm: str) -> np.ndarray:
        m = arm_mem(pid, arm)
        return np.array([c + r for c, r in zip(ctx[pid], m)], dtype=float)

    def eval_nll(w: np.ndarray, x: np.ndarray, yy: np.ndarray, k: int):
        # Numerically stable binary NLL in nats.
        z = x[k:] @ w
        nll = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0) - yy[k:] * z
        return float(np.mean(nll))

    arms = ("A0", "A1", "A2", "A3", "A4", "A4frozen")
    x_of = {a: {pid: feats(pid, a) for pid in ids} for a in arms}

    def pooled_fit(arm: str) -> np.ndarray:
        xs = np.concatenate([x_of[arm][pid][: split_of[pid]] for pid in ids])
        ys = np.concatenate([np.array(y[pid][: split_of[pid]], dtype=float) for pid in ids])
        return _fit_logistic(xs, ys)

    w_own = pooled_fit("A4")
    per_arm: dict[str, dict[str, list[float]]] = {
        g: {a: [] for a in arms} for g in ("refit", "frozen-swap")
    }
    for a in arms:
        w_a = pooled_fit(a)
        for pid in ids:
            yy = np.array(y[pid], dtype=float)
            k = split_of[pid]
            per_arm["refit"][a].append(eval_nll(w_a, x_of[a][pid], yy, k))
            per_arm["frozen-swap"][a].append(eval_nll(w_own, x_of[a][pid], yy, k))

    out: dict = {"grades": {}, "drift": {}, "singletons": sorted(singletons)}
    for g in per_arm:
        chain = {a: per_arm[g][a] for a in ("A0", "A1", "A2", "A3", "A4")}
        out["grades"][g] = {
            "response": telescope_components(
                chain,
                ids,
                "response",
                bootstrap_n=bootstrap_n,
                permutations=permutations,
                seed=seed,
                exclude=singletons,
            )
        }
        out["drift"][g] = [f - l for f, l in zip(per_arm[g]["A4frozen"], per_arm[g]["A4"])]
    out["per_arm"] = per_arm
    out["player_ids"] = ids
    return out
