"""Train-only per-user timing marginal for chess and KT (joint) timing.

Each user's held-out timings are scored under log-normal models fitted only to
training-split log-times.  The target marginal uses that user's own
(mu, sigma), the imposter marginal the registered within-cell random
imposter's (mu, sigma).  Neither model observes position, item, action,
sequence, or test information.

    python -m attribution_trials.comparison.marginal_timing
Output: <COMPARISON>/marginal/marginal_timing.json, per domain the users and
their per-user own NLL, imposter NLL and marginal_dindiv = own - imposter
(negative favours the target).
"""

from __future__ import annotations

import json
import math

import numpy as np

from attribution_trials import paths
from attribution_trials.bench.swap import within_cell_derangement
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.data.pooled_chess import build_pooled_chess
from attribution_trials.experiments.ec import session_split_indices

KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
SEED = 0
TRAIN_FRAC = 0.7
SIGMA_FLOOR = 0.05
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def lognormal_nll(log_times: np.ndarray, mu: float, sigma: float) -> float:
    z = (log_times - mu) / sigma
    return float((0.5 * z * z + math.log(sigma) + HALF_LOG_2PI + log_times).mean())


def fit_and_score(users, train, heldout, cells) -> dict:
    mu = {user: float(train[user].mean()) for user in users}
    raw_sigma = {user: float(train[user].std()) for user in users}
    sigma = {user: max(raw_sigma[user], SIGMA_FLOOR) for user in users}
    imposter = within_cell_derangement(cells, seed=SEED)

    own = np.array([lognormal_nll(heldout[user], mu[user], sigma[user]) for user in users])
    other = np.array(
        [lognormal_nll(heldout[user], mu[imposter[user]], sigma[imposter[user]]) for user in users]
    )
    return {
        "users": list(users),
        "imposter": imposter,
        "own_nll": own,
        "imposter_nll": other,
        "marginal_dindiv": own - other,
    }


def chess_data() -> dict:
    data, cells, _, dropped = build_pooled_chess("cohort-tertile")
    trajectories = data.trajectories
    splits = session_split_indices(trajectories, TRAIN_FRAC)
    users = [trajectory.player_id for trajectory in trajectories]
    train, heldout = {}, {}
    for trajectory, split in zip(trajectories, splits):
        times = np.array(
            [
                math.log(max(float(observation.time_spent or 1e-3), 1e-3))
                for observation in trajectory.observations
            ]
        )
        train[trajectory.player_id] = times[:split]
        heldout[trajectory.player_id] = times[split:]
    result = fit_and_score(users, train, heldout, cells)
    result["dropped_duplicate_players"] = dropped
    return result


def kt_rt_data() -> dict:
    data = load_kt_csv(
        str(KT_DATA),
        n_students=500,
        min_responses=50,
        response_time_col=5,
    )
    trajectories = data.trajectories
    splits = session_split_indices(trajectories, TRAIN_FRAC)
    users = [trajectory.player_id for trajectory in trajectories]
    train, heldout, accuracy = {}, {}, {}
    for trajectory, split in zip(trajectories, splits):
        times = np.array(
            [
                math.log(max(float(observation.time_spent), 1e-3))
                for observation in trajectory.observations
            ]
        )
        responses = np.array(
            [
                1.0 if observation.move == "correct" else 0.0
                for observation in trajectory.observations
            ]
        )
        user = trajectory.player_id
        train[user] = times[:split]
        heldout[user] = times[split:]
        accuracy[user] = float(responses[:split].mean())
    ordered = sorted(users, key=lambda user: accuracy[user])
    cells = {
        user: f"pacc_t{min(2, index * 3 // len(ordered))}" for index, user in enumerate(ordered)
    }
    return fit_and_score(users, train, heldout, cells)


def main() -> None:
    domains = {
        "chess-pooled": chess_data(),
        "kt-rt": kt_rt_data(),
    }
    payload = {}
    for domain, result in domains.items():
        payload[domain] = {
            "n_users": len(result["users"]),
            "users": result["users"],
            "imposter": result["imposter"],
            "own_nll": result["own_nll"].tolist(),
            "imposter_nll": result["imposter_nll"].tolist(),
            "marginal_dindiv": result["marginal_dindiv"].tolist(),
        }
        print(
            f"  {domain:14s} n={len(result['users']):4d} "
            f"marginal dindiv mean {result['marginal_dindiv'].mean():+.4f}",
            flush=True,
        )
    out = paths.ensure(paths.COMPARISON / "marginal") / "marginal_timing.json"
    out.write_text(json.dumps(payload) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
