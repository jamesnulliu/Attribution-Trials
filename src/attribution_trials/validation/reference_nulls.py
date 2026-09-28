# -*- coding: utf-8 -*-
"""Every reference condition on synthetic populations with zero individual signal.

Target versus no user / population / group / imposter on the same populations,
so every reference is read against a known truth (500 users, 100 Bernoulli
responses, 70/30 split, the per-user Laplace rate as the audited method,
200 replicates, 2,000 bootstrap resamples).  A contrast is target minus
reference held-out NLL; a replicate counts when its bootstrap interval lies
below zero.

Arms: A0 pooled rate, A1 an out-of-cell donor's rate, A2 the cell's pooled
rate, A3 a within-cell donor's rate, A4 the user's own rate.

Nulls:
  population     one shared rate
  drift          one shared rate that moves across the train boundary
  group, fine    6 latent groups, cells = training-accuracy tertiles, so a
                 within-cell donor can come from another group
  group, matched 6 groups, cells = the group label, as when the grouping
                 variable is observed and matched on

  python -m attribution_trials.validation.reference_nulls
Output: <VALIDATION>/reference_nulls.json
"""

import json

import numpy as np

from attribution_trials import paths

N_USERS, N_TOTAL, N_TRAIN, ALPHA = 500, 100, 70, 1.0
ALPHA_BETA, REPLICATES, BOOT_N, BASE_SEED = (4.0, 2.0), 200, 2000, 20260818
LIVE_WINDOW, DRIFT_STEPS, N_GROUPS, GROUP_SD = 40, (0.0, 0.15, 0.3, 0.45), 6, 0.12
QBAR = ALPHA_BETA[0] / sum(ALPHA_BETA)
IDX = np.random.default_rng(0).integers(
    0, N_USERS, size=(BOOT_N, N_USERS)
)  # one resample set, seed 0, for every interval


def favourable(delta):
    return bool(np.percentile(delta[IDX].mean(axis=1), 97.5) < 0)


def nll(heldout, rate):
    rate = np.clip(np.asarray(rate, float), 1e-9, 1 - 1e-9)
    rate = rate[:, None] if rate.ndim == 1 else rate
    return -(heldout * np.log(rate) + (1 - heldout) * np.log(1 - rate)).mean(axis=1)


def tertiles(acc):
    cell = np.empty(N_USERS, int)
    cell[np.argsort(acc, kind='stable')] = np.minimum(2, np.arange(N_USERS) * 3 // N_USERS)
    return cell


def derangement(cell, rng):
    donor = np.empty(N_USERS, int)
    for c in np.unique(cell):
        m = rng.permutation(np.flatnonzero(cell == c))
        donor[m] = np.roll(m, -1)
    return donor


def nearest(acc, cell):
    donor = np.empty(N_USERS, int)
    for c in np.unique(cell):
        m = np.flatnonzero(cell == c)
        d = np.abs(acc[m][:, None] - acc[m][None, :]).astype(float)
        np.fill_diagonal(d, np.inf)
        donor[m] = m[d.argmin(axis=1)]
    return donor


def contrasts(draws, cell, rng):
    train, heldout = draws[:, :N_TRAIN], draws[:, N_TRAIN:]
    own = (train.sum(axis=1) + ALPHA) / (N_TRAIN + 2 * ALPHA)
    pooled = (train.sum() + ALPHA) / (N_USERS * N_TRAIN + 2 * ALPHA)
    group = np.empty(N_USERS)
    for c in np.unique(cell):
        m = cell == c
        group[m] = (train[m].sum() + ALPHA) / (m.sum() * N_TRAIN + 2 * ALPHA)
    placebo = np.array([rng.choice(np.flatnonzero(cell != cell[i])) for i in range(N_USERS)])
    a4, a3 = nll(heldout, own), nll(heldout, own[derangement(cell, rng)])
    out = {
        'target vs no user': a4 - nll(heldout, np.full(N_USERS, pooled)),
        'target vs population': a4 - nll(heldout, own[placebo]),
        'target vs group': a4 - nll(heldout, group),
        'AT: target vs imposter': a4 - a3,
    }
    return out, own, a3, heldout


def build(kind, seed, step=0.0):
    rng = np.random.default_rng(seed)
    if kind in ('population', 'drift'):
        t = np.arange(N_TOTAL) / (N_TOTAL - 1)
        shared = np.clip(QBAR + step * (t - t[:N_TRAIN].mean()), 0.02, 0.98)
        draws = rng.random((N_USERS, N_TOTAL)) < shared[None, :]
        cell = tertiles(draws[:, :N_TRAIN].mean(axis=1))
    else:
        mu = np.clip(QBAR + GROUP_SD * rng.standard_normal(N_GROUPS), 0.05, 0.95)
        membership = rng.integers(0, N_GROUPS, N_USERS)
        draws = rng.random((N_USERS, N_TOTAL)) < mu[membership][:, None]
        cell = membership if kind == 'group, matched' else tertiles(draws[:, :N_TRAIN].mean(axis=1))
    out, own, a3, heldout = contrasts(draws, cell, rng)
    if kind in ('population', 'drift'):
        live = np.empty_like(heldout, dtype=float)
        for j in range(heldout.shape[1]):
            w = draws[:, max(0, N_TRAIN + j - LIVE_WINDOW) : N_TRAIN + j]
            live[:, j] = (w.sum(axis=1) + ALPHA) / (w.shape[1] + 2 * ALPHA)
        out['updated information vs imposter'] = nll(heldout, live) - a3
    if kind == 'group, fine':
        acc = draws[:, :N_TRAIN].mean(axis=1)
        out['AT, nearest-neighbour imposter'] = nll(heldout, own) - nll(
            heldout, own[nearest(acc, cell)]
        )
    return out


def run(kind, step=0.0):
    hits, points = {}, {}
    for r in range(REPLICATES):
        for name, d in build(kind, BASE_SEED + r, step).items():
            hits[name] = hits.get(name, 0) + favourable(d)
            points.setdefault(name, []).append(float(d.mean()))
    return {
        name: {'favourable_rate': hits[name] / REPLICATES, 'mean': float(np.mean(points[name]))}
        for name in hits
    }


def main():
    results = {'population': run('population')}
    for s in DRIFT_STEPS[1:]:
        results[f'drift {s}'] = run('drift', s)
    results['group, fine'] = run('group, fine')
    results['group, matched'] = run('group, matched')
    out = paths.ensure(paths.VALIDATION) / 'reference_nulls.json'
    out.write_text(
        json.dumps(
            {
                'settings': {
                    'n_users': N_USERS,
                    'replicates': REPLICATES,
                    'boot_n': BOOT_N,
                    'n_groups': N_GROUPS,
                    'group_sd': GROUP_SD,
                },
                'results': results,
            },
            indent=1,
        )
    )
    for k, v in results.items():
        print(k, {name: node['favourable_rate'] for name, node in v.items()})
    print(f'-> {out}')


if __name__ == '__main__':
    main()
