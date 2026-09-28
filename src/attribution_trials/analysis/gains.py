# -*- coding: utf-8 -*-
"""The four target gains of every combination, and the same contrasts with the
imposter's information in place of the target's.

For each combination and reference r in {none A0, population A1, group A2}:
  target    A4f - r   (with dind = A4f - A3)
  stranger  A3 - r    (the matched imposter's information in place of the target's)
Values are stored as target-minus-reference surprisal (negative favors the
target); the paper prints the negation.  Intervals are paired-user bootstrap
95% intervals with B = 10,000; Holm is pooled over the 84 combinations on
two-sided sign-flip p-values (9,999 flips).  The perplexity ratio
exp(mean A3 - mean A4f) is stored per combination.

Writes paths.ANALYSIS / "gains.json".
Run:  python -m attribution_trials.analysis.gains
"""

import json
from collections import defaultdict

import numpy as np

from attribution_trials import paths
from attribution_trials.analysis.cells import load_cells

OUT = paths.ANALYSIS / 'gains.json'
B, N_FLIP = 10000, 9999
REFS = (('dtot', 'A0'), ('dpop', 'A1'), ('dgrp', 'A2'))
DATASETS = ('chess', 'KT', 'OPeRA')


def boot(delta, rng):
    n = len(delta)
    means = delta[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return {
        'point': float(delta.mean()),
        'low': float(np.percentile(means, 2.5)),
        'high': float(np.percentile(means, 97.5)),
    }


def p_flip(delta, rng):
    signs = rng.choice((-1.0, 1.0), size=(N_FLIP, len(delta)))
    return float(
        (1 + np.sum(np.abs((signs * delta).mean(axis=1)) >= abs(delta.mean()))) / (N_FLIP + 1)
    )


def holm(pvals, alpha=0.05):
    order, m, keep = np.argsort(pvals), len(pvals), [False] * len(pvals)
    for rank, idx in enumerate(order):
        if pvals[idx] > alpha / (m - rank):
            break
        keep[idx] = True
    return keep


def fav(x):
    return x['high'] < 0


def adv(x):
    return x['low'] > 0


def dataset(c):
    return 'OPeRA' if c['panel'] != 'full tier' else ('chess' if 'chess' in c['key'] else 'KT')


def main():
    cells = load_cells()
    assert len(cells) == 84, len(cells)

    # pass 1: target-side intervals, one stream in load_cells order
    rng = np.random.default_rng(0)
    for c in cells:
        a = c['arms']
        c['target'] = {k: boot(a['A4f'] - a[r], rng) for k, r in REFS}
        c['target']['dind'] = boot(a['A4f'] - a['A3'], rng)

    # pass 2: stranger-side intervals and every sign-flip p-value share a second stream
    rng = np.random.default_rng(20260920)
    for c in cells:
        a = c['arms']
        c['stranger'] = {k: boot(a['A3'] - a[r], rng) for k, r in REFS}
        c['p'] = {
            'target': {k: p_flip(a['A4f'] - a[r], rng) for k, r in REFS},
            'stranger': {k: p_flip(a['A3'] - a[r], rng) for k, r in REFS},
            'dind': p_flip(a['A4f'] - a['A3'], rng),
        }
        # perplexity = exp(mean per-user surprisal); ratio stranger / target = exp(-dind)
        c['ppl_ratio'] = float(np.exp(a['A3'].mean() - a['A4f'].mean()))

    keep = holm([c['p']['dind'] for c in cells])
    holm_at = [c['key'] for c, k in zip(cells, keep) if k and c['target']['dind']['point'] < 0]
    holm_at_stranger = [
        c['key'] for c, k in zip(cells, keep) if k and c['target']['dind']['point'] > 0
    ]

    rows = {}
    for k, _ in REFS:
        tk = holm([c['p']['target'][k] for c in cells])
        sk = holm([c['p']['stranger'][k] for c in cells])
        r = {
            'favourable': defaultdict(int),
            'stranger_favourable': defaultdict(int),
            'still_favourable': defaultdict(int),
        }
        kept = []
        for c, th, sh in zip(cells, tk, sk):
            d, t, s = dataset(c), c['target'][k], c['stranger'][k]
            r['favourable'][d] += fav(t)
            r['stranger_favourable'][d] += fav(s)
            r['still_favourable'][d] += fav(t) and fav(s)
            if fav(t):
                kept.append(100 * s['point'] / t['point'])
            c.setdefault('holm', {})[k] = {
                'target': bool(th and t['point'] < 0),
                'stranger': bool(sh and s['point'] < 0),
            }
        rows[k] = {
            m: {**{d: int(v[d]) for d in DATASETS}, 'all': int(sum(v.values()))}
            for m, v in r.items()
        }
        rows[k]['gain_kept_pct'] = {
            'median': float(np.median(kept)),
            'min': float(min(kept)),
            'max': float(max(kept)),
        }
        rows[k]['holm_pooled'] = {
            'target': sum(c['holm'][k]['target'] for c in cells),
            'stranger': sum(c['holm'][k]['stranger'] for c in cells),
            'both': sum(c['holm'][k]['target'] and c['holm'][k]['stranger'] for c in cells),
        }

    at = {'favourable': defaultdict(int), 'stranger_favourable': defaultdict(int)}
    for c in cells:
        at['favourable'][dataset(c)] += fav(c['target']['dind'])
        at['stranger_favourable'][dataset(c)] += adv(c['target']['dind'])
    rows['dind'] = {
        m: {**{d: int(v[d]) for d in DATASETS}, 'all': int(sum(v.values()))} for m, v in at.items()
    }
    rows['dind']['still_favourable'] = {'all': 0}  # an interval cannot lie on both sides of zero
    rows['dind']['holm_pooled'] = {
        'target': len(holm_at),
        'stranger': len(holm_at_stranger),
        'target_keys': holm_at,
        'stranger_keys': holm_at_stranger,
    }

    ppl = {d: [100 * (c['ppl_ratio'] - 1) for c in cells if dataset(c) == d] for d in DATASETS}
    rows['perplexity'] = {
        d: {
            'n': len(v),
            'median_pct_change': float(np.median(v)),
            'min': float(min(v)),
            'max': float(max(v)),
            'abs_change_under_1pct': int(sum(abs(x) < 1 for x in v)),
        }
        for d, v in ppl.items()
    }
    n_cells = {d: sum(dataset(c) == d for c in cells) for d in DATASETS}

    out = {
        'n_cells': n_cells,
        'rows': rows,
        'cells': [
            {
                'key': c['key'],
                'panel': c['panel'],
                'dataset': dataset(c),
                'n': c['n'],
                'target': c['target'],
                'stranger': c['stranger'],
                'p_flip': c['p'],
                'holm_pooled': c['holm'],
                'ppl_ratio': c['ppl_ratio'],
            }
            for c in cells
        ],
    }
    paths.ensure(OUT.parent)
    json.dump(out, OUT.open('w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
