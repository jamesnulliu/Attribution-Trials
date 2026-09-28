"""Train-only per-user label marginals for the discrete channels.

For every user, the held-out decisions are scored under

  own           the user's own training-split empirical behavior model
  random_other  the within-cell derangement partner's model (seed 0), the
                registered imposter construction

in the same metric convention as the audit rows (mean NLL in nats per
decision over the user's held-out steps).

Empirical model per channel (training split ONLY, per user):
  chess move   Laplace-smoothed (alpha=0.5) categorical over (from-square,
               to-square) pairs, restricted to the decision's legal actions
               and renormalized (mirrors the factored from/to head's action
               space; promotions sharing a from/to get equal mass, as in the
               backbone).
  kt response  Laplace-smoothed Bernoulli (train accuracy, alpha=1).

Splits are the audit's own: session_split_indices(train_frac=0.7) on the
pooled chess panel and on load_kt_csv with the ladder's arguments.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from attribution_trials import paths
from attribution_trials.bench.swap import within_cell_derangement
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import session_split_indices
from attribution_trials.policy.board_native import _square_index

KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
SEED = 0
ALPHA_CHESS = 0.5  # Laplace for the 4096-way (from,to) space
ALPHA_CAT = 1.0  # Laplace for small categorical spaces (kt)
TRAIN_FRAC = 0.7


def assemble_channel(
    users: list[str],
    per_user_nll: dict[str, dict[str, float]],  # row -> {user: nll}
    rand_map: dict[str, str],
) -> dict:
    return {
        "users": list(users),
        "n_users": len(users),
        "per_user_nll": {
            row: [float(vals[u]) for u in users] for row, vals in per_user_nll.items()
        },
        "random_pair": {u: rand_map[u] for u in users},
    }


# -------------------------------------------------------------- chess --


def chess_move() -> dict:
    from attribution_trials.data.pooled_chess import build_pooled_chess

    data, cells, cohort_of, dropped = build_pooled_chess()
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    users = [t.player_id for t in trajs]
    print(f"[chess] {len(users)} players (dropped dups: {len(dropped)})", flush=True)

    train_pairs: dict[str, Counter] = {}
    eval_moves: dict[str, list] = {}  # (legal_pairs tuple, chosen_pair)
    for t, k in zip(trajs, splits):
        u = t.player_id
        cnt, ev = Counter(), []
        for i, (dp, obs) in enumerate(zip(t.decisions, t.observations)):
            la = dp.legal_actions
            mv = obs.move if obs.move in la else la[0]  # mirrors encode_batch
            pair = (_square_index(mv[:2]), _square_index(mv[2:4]))
            legal = tuple((_square_index(m[:2]), _square_index(m[2:4])) for m in la)
            if i < k:
                cnt[pair] += 1
            else:
                ev.append((legal, pair))
        train_pairs[u] = cnt
        eval_moves[u] = ev

    rand_map = within_cell_derangement(cells, seed=SEED)

    def score_move(u: str, cnt: Counter) -> float:
        """Mean held-out NLL for user u under counts cnt."""
        nlls = []
        for legal, chosen in eval_moves[u]:
            w = np.array([cnt.get(p, 0) + ALPHA_CHESS for p in legal])
            wc = cnt.get(chosen, 0) + ALPHA_CHESS
            nlls.append(-math.log(wc / w.sum()))
        return float(np.mean(nlls))

    nll = {"own": {}, "random_other": {}}
    for u in users:
        nll["own"][u] = score_move(u, train_pairs[u])
        nll["random_other"][u] = score_move(u, train_pairs[rand_map[u]])
    return assemble_channel(users, nll, rand_map)


# ----------------------------------------------------------------- kt --


def kt_response(leg: str, rt_col: int | None) -> dict:
    data = load_kt_csv(
        str(KT_DATA),
        n_students=500,
        min_responses=50,
        response_time_col=rt_col,
    )
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    users = [t.player_id for t in trajs]
    print(f"[kt:{leg}] {len(users)} students", flush=True)

    tr_c, tr_n, ev_resp = {}, {}, {}
    for t, k in zip(trajs, splits):
        u = t.player_id
        resp = [1.0 if o.move == "correct" else 0.0 for o in t.observations]
        tr_c[u] = sum(resp[:k])
        tr_n[u] = k
        ev_resp[u] = np.array(resp[k:])

    p_own = {u: (tr_c[u] + ALPHA_CAT) / (tr_n[u] + 2 * ALPHA_CAT) for u in users}
    acc_tr = {u: tr_c[u] / tr_n[u] for u in users}
    # cells: prior-accuracy tertiles (pacc prefix)
    order = sorted(users, key=lambda u: acc_tr[u])
    cells = {u: f"pacc_t{min(2, i * 3 // len(order))}" for i, u in enumerate(order)}
    rand_map = within_cell_derangement(cells, seed=SEED)

    def bern_nll(u, p):
        r = ev_resp[u]
        return float(-(r * math.log(p) + (1 - r) * math.log(1 - p)).mean())

    nll = {
        "own": {u: bern_nll(u, p_own[u]) for u in users},
        "random_other": {u: bern_nll(u, p_own[rand_map[u]]) for u in users},
    }
    return assemble_channel(users, nll, rand_map)
