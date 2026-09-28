"""Target-versus-imposter gain for the 18 API rows outside the 84 combinations.

Frozen-prompt GPT-4.1, GPT-4.1-mini and DeepSeek-V3.2 on OPeRA (two channels x
three profile configurations), scored with the letter-choice protocol.  Per
user: fixed-profile target minus imposter surprisal, A4frozen - A3, over the
users scored under every condition; paired per-user bootstrap 95% interval,
B = 10,000, seed 0.  Stored as target-minus-imposter (negative favors the
target), like gains.json.

Writes paths.ANALYSIS / "api_rows.json".
Run:  python -m attribution_trials.analysis.api_rows
"""

import json

import numpy as np

from attribution_trials import paths
from attribution_trials.analysis.cells import OPERA, by_player

OUT = paths.ANALYSIS / 'api_rows.json'
MODELS = ('gpt-4-1', 'gpt-4-1-mini', 'deepseek-v3-2')
B = 10000


def boot(delta, rng):
    n = len(delta)
    means = delta[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return {
        'point': float(delta.mean()),
        'low': float(np.percentile(means, 2.5)),
        'high': float(np.percentile(means, 97.5)),
    }


def main():
    rows = []
    for ch in ('action', 'timing'):
        for cfg in ('combined', 'static', 'dynamic'):
            for model in MODELS:
                d = json.loads((OPERA / f'frozen_opera-{ch}-{cfg}_{model}.json').read_text())
                arms = {a: by_player(d['nll'][a], d['players']) for a in d['nll']}
                common = sorted(set.intersection(*[set(v) for v in arms.values()]))
                delta = np.array([arms['A4frozen'][p] - arms['A3'][p] for p in common])
                dind = boot(delta, np.random.default_rng(0))
                rows.append(
                    {
                        'key': f'persona-frozen-api|{ch}|{cfg}|{model}',
                        'model': d['model'],
                        'n_users': len(common),
                        'n_decisions': d['n_eval'],
                        'dind': dind,
                        'favorable': dind['high'] < 0,
                        'adverse': dind['low'] > 0,
                    }
                )
    out = {
        'n': len(rows),
        'favorable': sum(r['favorable'] for r in rows),
        'adverse': sum(r['adverse'] for r in rows),
        'rows': rows,
    }
    paths.ensure(OUT.parent)
    OUT.write_text(json.dumps(out, indent=1))
    print('wrote', OUT, {k: out[k] for k in ('n', 'favorable', 'adverse')})


if __name__ == '__main__':
    main()
