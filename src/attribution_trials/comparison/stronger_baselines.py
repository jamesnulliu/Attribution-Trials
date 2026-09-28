"""Stronger train-only baselines for the KT response combinations.

The per-user marginal knows one number per student.  Two stronger train-only
competitors, both fit on the training split alone and swapped under the same
within-cell derangements the ladder draws:

  IRT (1PL)     p(correct) = sigmoid(theta_u - b_item), theta per student and
                b per item, fit jointly on all train rows. The item key is the
                skill's empirical-difficulty feature (state[0]) -- the item
                information the audited families themselves receive.
  fields        a logistic reader of the profile card's own ten numeric fields
                (bench.memory_family.memory_rows: overall/recent/EWMA
                accuracy, last answer, streak, skill-adjusted trend, ...),
                fit causally on train rows, then FROZEN at each user's train
                boundary for scoring, so it holds the same information budget
                as the frozen card of A4f/A3.

Both are recency-clean by construction, so their target-versus-imposter
contrast compares directly against the method's A4f - A3.  Capture fraction =
baseline contrast / method contrast (ratio of means, paired bootstrap over
users), residual = per-user method - fields baseline with its own interval.
Imposters: per-training-seed within-cell derangements on the ladder's own
cells, averaged over seeds; the LoRA / frozen-prompt users are paired by
student ID.

    python -m attribution_trials.comparison.stronger_baselines
Output: <COMPARISON>/stronger_baselines.json
"""

from __future__ import annotations

import json

import numpy as np

from attribution_trials import paths
from attribution_trials.analysis.cells import load_cells
from attribution_trials.bench.memory_family import memory_rows
from attribution_trials.bench.swap import within_cell_derangement
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import session_split_indices

KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
LADDERS = paths.AUDIT / "chess_kt"
SEEDS = (0, 1, 2)
SEED = 0
NBOOT = 10000
TRAIN_FRAC = 0.7
L2 = 1e-4

CELLS = {
    "static-embedding|kt|move": "kt",
    "static-embedding|kt-rt|move": "kt-rt",
    "evolving-latent|kt-rt|move": "kt-rt",
    "sft-lora|kt (1.7b)|response": "kt",
    "sft-lora|kt (8b)|response": "kt",
    "persona-frozen|kt (1.7b)|response": "kt",
    "persona-frozen|kt (8b)|response": "kt",
}
# Ladder combinations keep the ladder's own user order (the bootstrap
# resamples positions); the LoRA / frozen-prompt ones use sorted student IDs.
LADDER_ORDER = {
    "static-embedding|kt|move": ("small_ladder_kt_s0.json", "retrained"),
    "static-embedding|kt-rt|move": ("small_ladder_kt-rt_s0.json", "retrained"),
    "evolving-latent|kt-rt|move": ("evolving_kt-rt_s0.json", "eval-swap"),
}


def boot_mean(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(NBOOT)]
    return {
        "point": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def boot_ratio(numer, denom):
    """Ratio of means with a paired bootstrap over users."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(NBOOT):
        index = rng.integers(0, len(numer), len(numer))
        d = denom[index].mean()
        if abs(d) > 1e-12:
            draws.append(float(numer[index].mean() / d))
    return {
        "point": float(numer.mean() / denom.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def fit_logistic(x, y, l2=L2, iters=200):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x @ w)))
        grad = x.T @ (y - p) - l2 * w
        hess = (x * (p * (1.0 - p) + 1e-9)[:, None]).T @ x
        hess[np.diag_indices_from(hess)] += l2
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def bce(y, p):
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def build_leg(leg: str) -> dict:
    """All train-only per-user objects for one KT leg."""
    data = load_kt_csv(
        str(KT_DATA),
        n_students=500,
        min_responses=50,
        response_time_col=5 if leg == "kt-rt" else None,
    )
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    users = [t.player_id for t in trajs]
    cells = json.loads((LADDERS / f"small_ladder_{leg}_s0.json").read_text())["cells"]
    cells = {u: cells[u] for u in users}

    y_of, item_of, cut_of, feat_frozen = {}, {}, {}, {}
    feat_train_x, feat_train_y = [], []
    irt_rows = []  # (user index, item key, y) on the train split
    item_index: dict[float, int] = {}
    for ui, (traj, cut) in enumerate(zip(trajs, splits)):
        y = np.array([1.0 if o.move == "correct" else 0.0 for o in traj.observations])
        items = [round(float(dp.state[0]), 12) for dp in traj.decisions]
        rows = memory_rows(traj)
        boundary = min(cut, len(rows) - 1)
        u = traj.player_id
        y_of[u], item_of[u], cut_of[u] = y, items, cut
        feat_frozen[u] = np.array([1.0] + rows[boundary])
        for t in range(cut):
            feat_train_x.append([1.0] + rows[t])
            feat_train_y.append(y[t])
            key = items[t]
            if key not in item_index:
                item_index[key] = len(item_index)
            irt_rows.append((ui, item_index[key], y[t]))

    # ---- fields logistic (population weights, causal features) ----
    w_fields = fit_logistic(np.array(feat_train_x), np.array(feat_train_y))

    # ---- 1PL IRT: alternating Newton on theta and b --------------------
    ui_arr = np.array([r[0] for r in irt_rows])
    it_arr = np.array([r[1] for r in irt_rows])
    y_arr = np.array([r[2] for r in irt_rows])
    theta = np.zeros(len(users))
    b = np.zeros(len(item_index))
    for _ in range(50):
        z = theta[ui_arr] - b[it_arr]
        p = 1.0 / (1.0 + np.exp(-z))
        r = y_arr - p
        wgt = p * (1.0 - p) + 1e-9
        g_t = np.bincount(ui_arr, r, len(theta)) - L2 * theta
        h_t = np.bincount(ui_arr, wgt, len(theta)) + L2
        theta_new = theta + g_t / h_t
        z = theta_new[ui_arr] - b[it_arr]
        p = 1.0 / (1.0 + np.exp(-z))
        r = y_arr - p
        wgt = p * (1.0 - p) + 1e-9
        g_b = -np.bincount(it_arr, r, len(b)) - L2 * b
        h_b = np.bincount(it_arr, wgt, len(b)) + L2
        b_new = b + g_b / h_b
        moved = max(float(np.max(np.abs(theta_new - theta))), float(np.max(np.abs(b_new - b))))
        theta, b = theta_new, b_new
        if moved < 1e-8:
            break
    b_fallback = float(b.mean())
    theta_of = dict(zip(users, theta))

    unseen = 0
    eval_b = {}
    for u in users:
        vals = []
        for t in range(cut_of[u], len(y_of[u])):
            key = item_of[u][t]
            if key in item_index:
                vals.append(b[item_index[key]])
            else:
                vals.append(b_fallback)
                unseen += 1
        eval_b[u] = np.array(vals)

    def nll_irt(user, donor):
        z = theta_of[donor] - eval_b[user]
        p = 1.0 / (1.0 + np.exp(-z))
        return float(bce(y_of[user][cut_of[user] :], p).mean())

    def nll_fields(user, donor):
        p = 1.0 / (1.0 + np.exp(-float(feat_frozen[donor] @ w_fields)))
        return float(
            bce(y_of[user][cut_of[user] :], np.full(len(y_of[user]) - cut_of[user], p)).mean()
        )

    deltas = {"irt": np.zeros(len(users)), "fields": np.zeros(len(users))}
    for seed in SEEDS:
        derangement = within_cell_derangement(cells, seed=seed)
        for name, nll in (("irt", nll_irt), ("fields", nll_fields)):
            deltas[name] += np.array([nll(u, u) - nll(u, derangement[u]) for u in users])
    for name in deltas:
        deltas[name] /= len(SEEDS)
    print(
        f"  [{leg}] {len(users)} users, {len(item_index)} items, {unseen} unseen eval item hits",
        flush=True,
    )
    return {
        "users": users,
        "irt": dict(zip(users, deltas["irt"])),
        "fields": dict(zip(users, deltas["fields"])),
        "n_items": len(item_index),
        "unseen_eval_items": unseen,
    }


def method_users(cell: dict) -> list:
    """The combination's users, in the order its per-user vector is resampled."""
    if cell["key"] not in LADDER_ORDER:
        return list(cell["users"])
    fname, grade = LADDER_ORDER[cell["key"]]
    ids = json.loads((LADDERS / fname).read_text())["grades"][grade]["move"]["player_ids"]
    have = set(cell["users"])
    return [u for u in ids if u in have]


def main() -> int:
    legs = {leg: build_leg(leg) for leg in ("kt", "kt-rt")}
    by_key = {c["key"]: c for c in load_cells()}

    rows = []
    for key, leg in CELLS.items():
        c = by_key[key]
        pos = {u: i for i, u in enumerate(c["users"])}
        method = c["arms"]["A4f"] - c["arms"]["A3"]
        table = legs[leg]
        users = [u for u in method_users(c) if str(u) in {str(x) for x in table["irt"]}]
        key_of = {str(x): x for x in table["irt"]}
        m_id = np.array([method[pos[u]] for u in users])
        row = {"cell": key, "n_users": len(users), "method_identity": boot_mean(m_id)}
        for name in ("irt", "fields"):
            base = np.array([table[name][key_of[str(u)]] for u in users])
            row[name] = {
                "delta": boot_mean(base),
                "capture_of_identity": boot_ratio(base, m_id),
            }
            if name == "fields":
                row[name]["residual_vs_identity"] = boot_mean(m_id - base)
        rows.append(row)
        print(
            f"  {key:<38} id {row['method_identity']['point']:+.4f}  "
            f"irt {row['irt']['delta']['point']:+.4f}  "
            f"fields {row['fields']['delta']['point']:+.4f}",
            flush=True,
        )

    blob = {
        "legs": {leg: {k: legs[leg][k] for k in ("n_items", "unseen_eval_items")} for leg in legs},
        "cells": rows,
    }
    out = paths.ensure(paths.COMPARISON) / "stronger_baselines.json"
    out.write_text(json.dumps(blob, indent=1, default=float))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
