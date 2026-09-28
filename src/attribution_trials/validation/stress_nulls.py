"""Synthetic stress populations for the trial, 200 replicates each.

The audited method is the per-user Laplace-smoothed Bernoulli rate; cells are
training-accuracy tertiles and the imposter is the within-cell derangement.
Each contrast is target minus imposter held-out NLL per user, so negative
favours the target (the paper's gain is its negation); a replicate counts
when its paired bootstrap interval lies below zero.

  drift      Everyone shares one time-varying rate, so the identity signal is
             zero, but the rate moves across the train boundary.  The updated
             variant re-estimates each user's rate over the most recent
             LIVE_WINDOW responses (crossing the boundary); the fixed variant
             stops at the boundary.

  coarse     A fully individual, fully stable population (lambda = 1).  The
             gain is computed under a 2-cell, a tertile, and a
             nearest-neighbour (training accuracy) matcher.

No language model is involved.

  python -m attribution_trials.validation.stress_nulls
Output: <VALIDATION>/stress_nulls.json
"""

from __future__ import annotations

import json

import numpy as np

from attribution_trials import paths
from attribution_trials.bench.swap import within_cell_derangement
from attribution_trials.eval.bootstrap import bootstrap_ci

N_USERS = 500
N_TOTAL = 100
TRAIN_FRAC = 0.7
N_TRAIN = int(N_TOTAL * TRAIN_FRAC)
ALPHA = 1.0
ALPHA_BETA = (4.0, 2.0)
REPLICATES = 200
BOOT_N = 2000
BASE_SEED = 20260818
LIVE_WINDOW = 40
DRIFT_STEPS = (0.0, 0.15, 0.3, 0.45)  # total shift of the shared rate


def tertile_cells(train_accuracy: np.ndarray, users: list) -> dict:
    order = np.argsort(train_accuracy, kind="stable")
    cells = {}
    for rank, index in enumerate(order):
        cells[users[index]] = f"pacc_t{min(2, rank * 3 // len(users))}"
    return cells


def halves_cells(train_accuracy: np.ndarray, users: list) -> dict:
    order = np.argsort(train_accuracy, kind="stable")
    cells = {}
    for rank, index in enumerate(order):
        cells[users[index]] = f"half_{min(1, rank * 2 // len(users))}"
    return cells


def nn_map(train_accuracy: np.ndarray, users: list, cells: dict) -> dict:
    """Nearest other user by training accuracy, inside the registered cell."""
    donor = {}
    by_cell: dict[str, list[int]] = {}
    for i, u in enumerate(users):
        by_cell.setdefault(cells[u], []).append(i)
    for members in by_cell.values():
        block = train_accuracy[members]
        dist = np.abs(block[:, None] - block[None, :]).astype(float)
        np.fill_diagonal(dist, np.inf)
        for pos, i in enumerate(members):
            donor[users[i]] = users[members[int(dist[pos].argmin())]]
    return donor


def bernoulli_nll(heldout: np.ndarray, rate) -> np.ndarray:
    rate = np.clip(np.asarray(rate, dtype=float), 1e-9, 1 - 1e-9)
    if rate.ndim == 1:
        rate = rate[:, None]
    return -(heldout * np.log(rate) + (1 - heldout) * np.log(1 - rate)).mean(axis=1)


def screen(delta: np.ndarray) -> dict:
    boot = bootstrap_ci(delta.tolist(), n_resamples=BOOT_N, seed=0)
    return {
        "point": float(boot.point),
        "low": float(boot.low),
        "high": float(boot.high),
        "favorable": bool(boot.high < 0),
    }


def run_replicates(build, contrasts, tag):
    """REPLICATES replicates; per contrast, the favourable-interval rate."""
    hits = {c: 0 for c in contrasts}
    points = {c: [] for c in contrasts}
    for replicate in range(REPLICATES):
        deltas = build(BASE_SEED + replicate)
        for c in contrasts:
            s = screen(deltas[c])
            hits[c] += s["favorable"]
            points[c].append(s["point"])
    out = {}
    for c in contrasts:
        p = np.array(points[c])
        out[c] = {
            "favorable_rate": hits[c] / REPLICATES,
            "mean": float(p.mean()),
            "sd": float(p.std(ddof=1)),
        }
        print(
            f"  [{tag}] {c:<24} rate {hits[c]}/{REPLICATES} = "
            f"{hits[c] / REPLICATES:.3f}  mean {p.mean():+.5f}",
            flush=True,
        )
    return out


def smooth_rows(train: np.ndarray) -> np.ndarray:
    return (train.sum(axis=1) + ALPHA) / (train.shape[1] + 2 * ALPHA)


def build_drift(seed: int, step: float) -> dict:
    """Shared drifting rate, zero identity signal, updated vs fixed estimates."""
    rng = np.random.default_rng(seed)
    qbar = ALPHA_BETA[0] / sum(ALPHA_BETA)
    t = np.arange(N_TOTAL) / (N_TOTAL - 1)
    shared = np.clip(qbar + step * (t - t[:N_TRAIN].mean()), 0.02, 0.98)
    draws = rng.random((N_USERS, N_TOTAL)) < shared[None, :]
    train, heldout = draws[:, :N_TRAIN], draws[:, N_TRAIN:]
    users = [f"u{i:04d}" for i in range(N_USERS)]
    train_accuracy = train.mean(axis=1)
    cells = tertile_cells(train_accuracy, users)
    donor = within_cell_derangement(cells, seed=0)
    donor_index = np.array([users.index(donor[u]) for u in users])

    frozen = smooth_rows(train)
    # the updated estimate carried into each held-out response: the last
    # LIVE_WINDOW responses before it, crossing the train boundary
    live_rates = np.empty_like(heldout, dtype=float)
    for j in range(heldout.shape[1]):
        upto = N_TRAIN + j
        window = draws[:, max(0, upto - LIVE_WINDOW) : upto]
        live_rates[:, j] = (window.sum(axis=1) + ALPHA) / (window.shape[1] + 2 * ALPHA)
    live_nll = -(
        heldout * np.log(np.clip(live_rates, 1e-9, 1 - 1e-9))
        + (1 - heldout) * np.log(np.clip(1 - live_rates, 1e-9, 1 - 1e-9))
    ).mean(axis=1)
    frozen_nll = bernoulli_nll(heldout, frozen)
    donor_nll = bernoulli_nll(heldout, frozen[donor_index])
    return {
        "updated (A4live-A3)": live_nll - donor_nll,
        "fixed (A4frozen-A3)": frozen_nll - donor_nll,
    }


def build_coarse(seed: int) -> dict:
    """Fully individual stable population under three matcher resolutions."""
    rng = np.random.default_rng(seed)
    q = rng.beta(*ALPHA_BETA, size=N_USERS)
    draws = rng.random((N_USERS, N_TOTAL)) < q[:, None]
    train, heldout = draws[:, :N_TRAIN], draws[:, N_TRAIN:]
    users = [f"u{i:04d}" for i in range(N_USERS)]
    train_accuracy = train.mean(axis=1)
    own = smooth_rows(train)
    own_nll = bernoulli_nll(heldout, own)
    index = {u: i for i, u in enumerate(users)}
    out = {}
    tert = tertile_cells(train_accuracy, users)
    for name, donor in (
        ("2-cell matcher", within_cell_derangement(halves_cells(train_accuracy, users), seed=0)),
        ("tertile matcher", within_cell_derangement(tert, seed=0)),
        ("NN-accuracy matcher", nn_map(train_accuracy, users, tert)),
    ):
        d = own_nll - bernoulli_nll(heldout, own[[index[donor[u]] for u in users]])
        out[name] = d
    return out


def main() -> int:
    results = {
        **{
            f"drift step={step}": run_replicates(
                lambda seed, step=step: build_drift(seed, step),
                ("updated (A4live-A3)", "fixed (A4frozen-A3)"),
                f"drift {step}",
            )
            for step in DRIFT_STEPS
        },
        "coarse": run_replicates(
            build_coarse, ("2-cell matcher", "tertile matcher", "NN-accuracy matcher"), "coarse"
        ),
    }
    blob = {
        "n_users": N_USERS,
        "n_total": N_TOTAL,
        "replicates": REPLICATES,
        "drift_steps": list(DRIFT_STEPS),
        "live_window": LIVE_WINDOW,
        "results": results,
    }
    out = paths.ensure(paths.VALIDATION) / "stress_nulls.json"
    out.write_text(json.dumps(blob, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
