"""Other rules for choosing a WildChat simulator, scored with the R2 rule.

R2 asks whether a selector's ordering agrees with the content-gain ordering on
the simulator pairs whose content-gain difference is resolvable (paired-user
bootstrap, seed 0, 10,000 resamples).  Selectors: AT (-dind), profile versus
no profile (-dtot) and absolute similarity of the target-conditioned samples
to the real turn (mean D_A4), each at avg@8 and best@8.  Writes
``paths.WILDCHAT / "selector_comparison.json"``.

  python -m attribution_trials.wildchat.selectors
"""

import itertools
import json
from collections import defaultdict

import numpy as np

from attribution_trials import paths

RES = paths.WILDCHAT
SEED, NBOOT, METRIC = 0, 10000, 'M1_minilm_max_card_bullet'
CANDS = ('1.7b', '8b', 'dense', 'humanlike', 'coser', 'osim', 'humanlm')
SCALE = {
    '1.7b': 'scale_1.7b',
    '8b': 'scale_8b',
    'dense': 'scale_32b',
    'humanlike': 'scale_humanlike',
    'coser': 'scale_coser',
    'osim': 'scale_osim',
    'humanlm': 'scale_humanlm',
}
ENDPOINTS = {'avg@8': ('D_A3', 'D_A4'), 'best@8': ('Dmin_A3', 'Dmin_A4')}


def bootstrap(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def kendall_tau(x, y):
    c = d = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
        c += s > 0
        d += s < 0
    return (c - d) / (len(x) * (len(x) - 1) / 2)


def at_rows(cand):
    s = json.loads((RES / f'{SCALE[cand]}.json').read_text())
    if 'rows' in s:
        return {
            r['key']: (
                r['user_id'],
                r['nll']['A4'] - r['nll']['A3'],
                r['nll']['A4'] - r['nll']['A0'],
            )
            for r in s['rows']
        }
    n = s['nll']
    # the columnar schema keys rows as [conv_id, turn_index]; the frame uses "user|conv|turn"
    return {
        f'{p}|{k[0]}|{int(k[1])}': (p, a4 - a3, a4 - a0)
        for k, p, a0, a3, a4 in zip(s['keys'], s['players'], n['A0'], n['A3'], n['A4'])
    }


def user_means(users, pairs):
    acc = defaultdict(list)
    for u, v in pairs:
        acc[u].append(v)
    return np.array([np.mean(acc[u]) for u in users])


def main():
    frame_users, sel, gain = None, defaultdict(dict), defaultdict(dict)
    for c in CANDS:
        rows = at_rows(c)
        e = json.loads((RES / f'content_{c}.json').read_text())
        frame_users = frame_users or e['users']
        assert e['users'] == frame_users
        keys = [r['key'] for r in e['rows']]
        sel['AT (-dind)'][c] = user_means(frame_users, [(rows[k][0], rows[k][1]) for k in keys])
        sel['profile vs no profile (-dtot)'][c] = user_means(
            frame_users, [(rows[k][0], rows[k][2]) for k in keys]
        )
        for ep, (a3, a4) in ENDPOINTS.items():
            m = [(r['user_id'], r['metrics'][METRIC]) for r in e['rows']]
            gain[ep][c] = user_means(
                frame_users, [(u, x[a4] - x[a3]) for u, x in m]
            )  # negative = target closer
            sel[f'absolute similarity, {ep}'][c] = user_means(
                frame_users, [(u, x[a4]) for u, x in m]
            )

    out = {}
    for ep in ENDPOINTS:
        pairs = []
        for l, r in itertools.combinations(CANDS, 2):
            point, lo, hi = bootstrap(gain[ep][l] - gain[ep][r])
            pairs.append((l, r, point, lo > 0 or hi < 0))
        best_gain = min(CANDS, key=lambda c: gain[ep][c].mean())
        for name, score in sel.items():
            res = [(l, r, p) for l, r, p, ok in pairs if ok]
            agree = sum(np.sign(score[l].mean() - score[r].mean()) == np.sign(p) for l, r, p in res)
            pick = min(CANDS, key=lambda c: score[c].mean())
            order = sorted(CANDS, key=lambda c: gain[ep][c].mean())
            out[f'{name} | {ep}'] = {
                'selector': name,
                'endpoint': ep,
                'n_resolvable': len(res),
                'n_agree': int(agree),
                'kendall_tau': kendall_tau(
                    [score[c].mean() for c in CANDS], [gain[ep][c].mean() for c in CANDS]
                ),
                'top_pick': pick,
                'top_pick_gain_rank': order.index(pick) + 1,
                'top_pick_content_gain': float(-gain[ep][pick].mean()),
                'best_content_gain': float(-gain[ep][best_gain].mean()),
            }

    paths.ensure(RES)
    json.dump(
        {'n_users': len(frame_users), 'metric': METRIC, 'rows': list(out.values())},
        (RES / 'selector_comparison.json').open('w'),
        indent=1,
    )
    for r in out.values():
        print(
            f'{r["selector"]} | {r["endpoint"]}: {r["n_agree"]} / {r["n_resolvable"]}, tau {r["kendall_tau"]:+.2f}, '
            f'top pick {r["top_pick"]} (gain rank {r["top_pick_gain_rank"]})'
        )


if __name__ == '__main__':
    main()
