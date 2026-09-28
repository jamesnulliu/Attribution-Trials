"""Existing-benchmark rows on the 84 combinations, and the Olmo block.

Joins the per-cell readouts (accuracy, total_variation, population_js) and the
static-embedding cross matrices to load_cells()'s 84 keys:

  join             the two `_cap1024` cells are the chess prompt-cap control
                   and stay out; the three Olmo cells are outside the 84 and
                   get their own block
  accuracy         favourable = accuracy gain A4f-A0 > 0, point estimate; the
                   same with A3 for the imposter side
  total_variation  one distribution per cell; favourable = mean TVD under A4f
                   below A0; the same with A3 for the imposter side
  population       score kept = JS(A3) / JS(A4f)
  random_profile   static-embedding family only; the random unrelated donor is
                   the mean over every other user of the cross matrix;
                   favourable = paired bootstrap interval of A4f - A_rand
                   below zero, imposter side = the same for A3 - A_rand
                   (B = 10,000, its own RNG stream)
  olmo             per-row NLL = -logp[label] / label tokens (frozen_panel's
                   loss), per-user mean, the four contrasts with intervals,
                   plus accuracy and total variation

Join check: every per-option cell maps to exactly one of {84-key, cap1024,
olmo}, and the 84 keys hit are exactly the frozen-prompt / Osim / Sim2Real
trio combinations.

    python -m attribution_trials.comparison.comparison_rows
Output: <COMPARISON>/comparison_rows.json
"""

import json
from collections import defaultdict

import numpy as np

from attribution_trials import paths
from attribution_trials.analysis.cells import load_cells

B = 10000
DATASETS = ('chess', 'KT', 'OPeRA')
ARMS = ('A0', 'A1', 'A2', 'A3', 'A4f')
REFS = (('dtot', 'A0'), ('dpop', 'A1'), ('dgrp', 'A2'), ('dind', 'A3'))
STATIC_FILES = {
    'static-embedding|chess-pooled|move': 'M_cross_chess-pooled_move.npz',
    'static-embedding|chess-pooled|timing': 'M_cross_chess-pooled_timing.npz',
    'static-embedding|kt|move': 'M_cross_kt_move.npz',
    'static-embedding|kt-rt|move': 'M_cross_kt-rt_move.npz',
    'static-embedding|kt-rt|timing': 'M_cross_kt-rt_timing.npz',
}
PER_OPTION = paths.COMPARISON / 'per_option'
CROSS = paths.COMPARISON / 'cross_matrices'


def boot(delta, rng):
    n = len(delta)
    means = delta[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return {
        'point': float(delta.mean()),
        'low': float(np.percentile(means, 2.5)),
        'high': float(np.percentile(means, 97.5)),
    }


def fav(x):
    return x['high'] < 0


def dataset(c):
    return 'OPeRA' if c['panel'] != 'full tier' else ('chess' if 'chess' in c['key'] else 'KT')


def key_of(cell):
    """per_option basename -> 84-key, or 'cap1024' / 'olmo'."""
    if cell.endswith('_cap1024'):
        return 'cap1024'
    fam, rest = cell.split('_', 1)
    sub, model = rest.rsplit('_', 1)
    if model == 'olmo-32b':
        return 'olmo'
    if fam == 'persona-frozen' and sub in ('chess', 'kt'):
        return f'persona-frozen|{sub} ({model})|' + ('timing' if sub == 'chess' else 'response')
    _, ch, cfg = sub.split('-')
    return f'{fam}|{ch}|{cfg}|{model}'


def per_user_nll(z, arm):
    lp, lab, tl = z[f'logp_{arm}'], z['labels'], z['option_token_lens']
    row = -lp[np.arange(len(lab)), lab] / tl[lab]
    acc = defaultdict(list)
    for p, x in zip(z['players'], row):
        acc[str(p)].append(x)
    return {p: float(np.mean(v)) for p, v in acc.items()}


def row_counts(flags, kept):
    """flags: {key: (dataset, favourable, still)}; kept: list of % over favourable."""
    r = {
        'n': defaultdict(int),
        'favourable': defaultdict(int),
        'still_favourable': defaultdict(int),
    }
    for d, f, s in flags.values():
        r['n'][d] += 1
        r['favourable'][d] += bool(f)
        r['still_favourable'][d] += bool(f and s)
    out = {
        m: {**{d: int(v[d]) for d in DATASETS}, 'all': int(sum(v.values()))} for m, v in r.items()
    }
    out['gain_kept_pct'] = (
        {
            'median': float(np.median(kept)),
            'min': float(min(kept)),
            'max': float(max(kept)),
            'n': len(kept),
        }
        if kept
        else None
    )
    return out


def main() -> None:
    cells = load_cells()
    assert len(cells) == 84, len(cells)
    by_key = {c['key']: c for c in cells}
    u4 = {
        r['cell']: r for r in json.loads((paths.COMPARISON / 'accuracy.json').read_text())['cells']
    }
    u5 = {
        r['cell']: r['whole']
        for r in json.loads((paths.COMPARISON / 'total_variation.json').read_text())['cells']
    }
    u6 = {
        r['cell']: r
        for r in json.loads((paths.COMPARISON / 'population_js.json').read_text())['cells']
    }
    names = sorted(u4)
    assert set(names) == set(u5) == set(u6) == {p.stem for p in PER_OPTION.glob('*.npz')}, (
        'the three readouts and per_option disagree'
    )

    # ------------------------------------------------------------ the join
    groups = defaultdict(list)
    for n in names:
        groups[key_of(n)].append(n)
    cap1024, olmo = groups.pop('cap1024'), groups.pop('olmo')
    assert all(k in by_key for k in groups), [k for k in groups if k not in by_key]
    expected = {
        c['key']
        for c in cells
        if c['key'].split('|')[0] in ('persona-frozen', 'trained-sim', 'sim2real-trio')
    }
    assert set(groups) == expected, set(groups) ^ expected
    assert len(groups) == 52 and len(cap1024) == 2 and len(olmo) == 3, (
        len(groups),
        len(cap1024),
        len(olmo),
    )
    assert all(len(v) == 1 for v in groups.values())
    cell_of = {k: v[0] for k, v in groups.items()}
    join = {
        'matched_84_keys': len(groups),
        'per_dataset': {d: sum(dataset(by_key[k]) == d for k in groups) for d in DATASETS},
        'excluded_cap1024': cap1024,
        'outside_84_olmo': olmo,
    }

    rows = {}
    # ------------------------------------------------------------ accuracy (point rule)
    flags, kept = {}, []
    for k in groups:
        acc = u4[cell_of[k]]['accuracy']
        g, s = acc['A4f'] - acc['A0'], acc['A3'] - acc['A0']
        flags[k] = (dataset(by_key[k]), g > 0, s > 0)
        if g > 0:
            kept.append(100 * s / g)
    rows['accuracy'] = row_counts(flags, kept)
    rows['accuracy']['rule'] = 'favourable = accuracy(A4f) > accuracy(A0), point estimate'
    rows['accuracy']['cells'] = {
        k: {'dataset': v[0], 'favourable': bool(v[1]), 'stranger_favourable': bool(v[2])}
        for k, v in flags.items()
    }

    # ------------------------------------------------------------ total variation, one distribution per cell
    flags, kept = {}, []
    for k in groups:
        tvd = {a: u5[cell_of[k]][a]['mean_tvd'] for a in ARMS}
        g, s = (
            tvd['A0'] - tvd['A4f'],
            tvd['A0'] - tvd['A3'],
        )  # reduction in TVD, positive = closer to real
        flags[k] = (dataset(by_key[k]), g > 0, s > 0)
        if g > 0:
            kept.append(100 * s / g)
    rows['total_variation'] = row_counts(flags, kept)
    rows['total_variation']['rule'] = (
        'favourable = mean TVD(A4f) < mean TVD(A0), one distribution per cell'
    )
    rows['total_variation']['cells'] = {
        k: {'dataset': v[0], 'favourable': bool(v[1]), 'stranger_favourable': bool(v[2])}
        for k, v in flags.items()
    }

    # ------------------------------------------------------------ population realism, imposter side
    js_kept = []
    for k in groups:
        js = {a: u6[cell_of[k]]['arms'][a]['js'] for a in ARMS}
        js_kept.append(100 * js['A3'] / js['A4f'])
    rows['population'] = {
        'n': len(js_kept),
        'js_kept_pct': {
            'median': float(np.median(js_kept)),
            'min': float(min(js_kept)),
            'max': float(max(js_kept)),
        },
    }

    # ------------------------------------------------------------ random unrelated profile (static embedding)
    rng = np.random.default_rng(20260922)
    flags, kept, rp_cells = {}, [], {}
    for k, fname in STATIC_FILES.items():
        c = by_key[k]
        z = np.load(CROSS / fname, allow_pickle=True)
        idx = {str(p): i for i, p in enumerate(z['player_ids'])}
        sel = np.array([idx[u] for u in c['users']])
        M = z['M'].mean(axis=0)
        rand = (M[sel].sum(axis=1) - M[sel, sel]) / (
            M.shape[0] - 1
        )  # mean over every other user as donor
        t, s = boot(c['arms']['A4f'] - rand, rng), boot(c['arms']['A3'] - rand, rng)
        flags[k] = (dataset(c), fav(t), fav(s))
        if fav(t):
            kept.append(100 * s['point'] / t['point'])
        rp_cells[k] = {
            'dataset': dataset(c),
            'n': int(len(sel)),
            'target_minus_random': t,
            'stranger_minus_random': s,
            'mean_nll': {
                'A4f': float(c['arms']['A4f'].mean()),
                'A3': float(c['arms']['A3'].mean()),
                'random_donor': float(rand.mean()),
            },
        }
    rows['random_profile'] = row_counts(flags, kept)
    rows['random_profile']['rule'] = (
        'favourable = 95% paired-user bootstrap interval of A4f - A_random below zero; A_random = mean NLL over every other user as donor'
    )
    rows['random_profile']['cells'] = rp_cells

    # ------------------------------------------------------------ Olmo block (outside the 84)
    rng = np.random.default_rng(20260922)
    olmo_out = {}
    for n in olmo:
        z = np.load(PER_OPTION / f'{n}.npz', allow_pickle=True)
        arms = {a: per_user_nll(z, a) for a in ARMS}
        users = sorted(set.intersection(*(set(v) for v in arms.values())))
        A = {a: np.array([arms[a][u] for u in users]) for a in ARMS}
        olmo_out[n] = {
            'n_users': len(users),
            'target': {k: boot(A['A4f'] - A[r], rng) for k, r in REFS},
            'stranger': {k: boot(A['A3'] - A[r], rng) for k, r in REFS if k != 'dind'},
            'mean_nll': {a: float(A[a].mean()) for a in ARMS},
            'accuracy': u4[n]['accuracy'],
            'mean_tvd': {a: u5[n][a]['mean_tvd'] for a in ARMS},
        }

    out = paths.ensure(paths.COMPARISON) / 'comparison_rows.json'
    out.write_text(json.dumps({'join': join, 'rows': rows, 'olmo': olmo_out}, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
