"""Chess frozen-prompt combinations at a prompt cap that never truncates.

`frozen_panel` front-truncates the prompt to `CAP - len(label_tokens)`.  At
CAP 640 the chess prompts are truncated, by an amount that varies with the
label's token count and with card length.  This compares the audit's CAP-640
cells against `per_option_frozen ... --cap 1024 --suffix _cap1024` reruns, at
which no row truncates, on the four gains:

    dtot = A4f - A0   dpop = A4f - A1   dgrp = A4f - A2   dind = A4f - A3

and reports, per cell and gain: both point estimates with 95% bootstrap
intervals, the paired per-user shift, and each interval's status (favourable
= interval strictly below zero, adverse = strictly above, else none) and
whether it changes.

    python -m attribution_trials.comparison.chess_prompt_cap
Output: <COMPARISON>/chess_prompt_cap.json
"""

import json
import sys

import numpy as np

from attribution_trials import paths
from attribution_trials.analysis.cells import by_player

B = 10000
ARMS = ('A0', 'A1', 'A2', 'A3', 'A4f')
ART_ARM = {'A0': 'A0', 'A1': 'A1', 'A2': 'A2', 'A3': 'A3', 'A4f': 'A4frozen'}
REFS = (('dtot', 'A0'), ('dpop', 'A1'), ('dgrp', 'A2'), ('dind', 'A3'))


def boot(delta, rng):
    n = len(delta)
    means = delta[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return {
        'point': float(delta.mean()),
        'low': float(np.percentile(means, 2.5)),
        'high': float(np.percentile(means, 97.5)),
    }


def status(x):
    if x['high'] < 0:
        return 'favourable'
    if x['low'] > 0:
        return 'adverse'
    return 'none'


def cap640_arms(size):
    d = json.loads((paths.AUDIT / 'chess_kt' / f'frozen_chess_{size}.json').read_text())
    return {a: by_player(d['nll'][ART_ARM[a]], d['players']) for a in ARMS}


def cap1024_arms(size):
    f = paths.COMPARISON / 'per_option' / f'persona-frozen_chess_{size}_cap1024.npz'
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    players = [str(p) for p in z['players']]
    ntok = z['option_token_lens']
    true_idx = z['labels']
    rows = np.arange(len(true_idx))
    out = {}
    for a in ARMS:
        nll = -z[f'logp_{a}'][rows, true_idx] / ntok[true_idx]
        out[a] = by_player(list(nll), players)
    return out


def main() -> int:
    rng = np.random.default_rng(20260921)
    report = {}
    for size in ('1.7b', '8b'):
        new = cap1024_arms(size)
        if new is None:
            print(f'[missing] chess {size}: no _cap1024 cell', flush=True)
            continue
        old = cap640_arms(size)
        users = sorted(set.intersection(*[set(old[a]) for a in ARMS], *[set(new[a]) for a in ARMS]))
        oa = {a: np.array([old[a][p] for p in users]) for a in ARMS}
        na = {a: np.array([new[a][p] for p in users]) for a in ARMS}
        cell = {'users': len(users), 'contrasts': {}}
        for name, ref in REFS:
            do, dn = oa['A4f'] - oa[ref], na['A4f'] - na[ref]
            bo, bn = boot(do, rng), boot(dn, rng)
            shift = boot(dn - do, rng)
            cell['contrasts'][name] = {
                'cap640': bo,
                'cap1024': bn,
                'paired_shift': shift,
                'status_cap640': status(bo),
                'status_cap1024': status(bn),
                'sign_changed': (bo['point'] > 0) != (bn['point'] > 0),
                'status_changed': status(bo) != status(bn),
            }
        cell['mean_nll'] = {
            a: {'cap640': float(oa[a].mean()), 'cap1024': float(na[a].mean())} for a in ARMS
        }
        report[f'persona-frozen|chess ({size})|timing'] = cell
    if not report:
        print('nothing to report', flush=True)
        return 0

    out = paths.ensure(paths.COMPARISON) / 'chess_prompt_cap.json'
    out.write_text(json.dumps(report, indent=2))
    for key, cell in report.items():
        print(f'{key} ({cell["users"]} users)')
        for name, c in cell['contrasts'].items():
            b, n = c['cap640'], c['cap1024']
            print(
                f'  {name}: CAP 640 {b["point"]:+.4f} [{b["low"]:+.4f}, {b["high"]:+.4f}] '
                f'-> CAP 1024 {n["point"]:+.4f} [{n["low"]:+.4f}, {n["high"]:+.4f}]  '
                f'{c["status_cap640"]} -> {c["status_cap1024"]}'
                + ('  CHANGED' if c['status_changed'] or c['sign_changed'] else '')
            )
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
