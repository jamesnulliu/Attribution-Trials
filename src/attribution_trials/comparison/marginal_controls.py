"""Per-user marginal baseline on the 17 complete five-condition combinations.

For each combination: the method's target-versus-imposter gain with the
profile fixed at the training boundary (dind = A4f - A3, 95% paired bootstrap
interval), and the residual after subtracting a train-only per-user marginal
model's own target-versus-imposter gain.  Stored sign: target minus imposter
surprisal, negative favours the target; a negative residual means the method
separates the target from the imposter by more than the marginal does.

  discrete channels  residual = mean over users of
                     (A4f - A3) - (own - random_other)
                     with the label marginals of `marginal_discrete`
  timing channels    residual = mean over users of
                     (A4 - A3) - (own - imposter) + drift,
                     drift = mean(A4f - A4), with the log-normal marginals of
                     `marginal_timing`

The dind intervals use one RNG stream (seed 0) over all 84 combinations in
load_cells() order, four gains per combination; the pooled-Holm flag is read
from <ANALYSIS>/gains.json.

    python -m attribution_trials.comparison.marginal_controls
Output: <COMPARISON>/marginal/marginal_controls.json
"""

import json

import numpy as np

from attribution_trials import paths
from attribution_trials.analysis.cells import load_cells

B = 10000
MARGINAL = paths.COMPARISON / 'marginal'

DISPLAY = {
    "static-embedding|chess-pooled|move": "static embedding $\\cdot$ chess $\\cdot$ move",
    "static-embedding|chess-pooled|timing": "static embedding $\\cdot$ chess $\\cdot$ timing",
    "static-embedding|kt|move": "static embedding $\\cdot$ KT $\\cdot$ response",
    "static-embedding|kt-rt|move": "static embedding $\\cdot$ KT (joint) $\\cdot$ response",
    "static-embedding|kt-rt|timing": "static embedding $\\cdot$ KT $\\cdot$ timing",
    "evolving-latent|chess-pooled|move": "recurrent embedding $\\cdot$ chess $\\cdot$ move",
    "evolving-latent|chess-pooled|timing": "recurrent embedding $\\cdot$ chess $\\cdot$ timing",
    "evolving-latent|kt-rt|move": "recurrent embedding $\\cdot$ KT $\\cdot$ response",
    "evolving-latent|kt-rt|timing": "recurrent embedding $\\cdot$ KT $\\cdot$ timing",
    "sft-lora|kt (1.7b)|response": "user-profile LoRA 1.7B $\\cdot$ KT $\\cdot$ response",
    "sft-lora|kt (8b)|response": "user-profile LoRA 8B $\\cdot$ KT $\\cdot$ response",
    "sft-lora|chess (1.7b)|timing": "user-profile LoRA 1.7B $\\cdot$ chess $\\cdot$ timing",
    "sft-lora|chess (8b)|timing": "user-profile LoRA 8B $\\cdot$ chess $\\cdot$ timing",
    "persona-frozen|kt (1.7b)|response": "frozen prompt 1.7B $\\cdot$ KT $\\cdot$ response",
    "persona-frozen|kt (8b)|response": "frozen prompt 8B $\\cdot$ KT $\\cdot$ response",
    "persona-frozen|chess (1.7b)|timing": "frozen prompt 1.7B $\\cdot$ chess $\\cdot$ timing",
    "persona-frozen|chess (8b)|timing": "frozen prompt 8B $\\cdot$ chess $\\cdot$ timing",
}
# discrete combination -> (domain, channel) of marginal_per_user.json
DISCRETE = {
    "static-embedding|chess-pooled|move": ("chess-pooled", "move"),
    "evolving-latent|chess-pooled|move": ("chess-pooled", "move"),
    "static-embedding|kt|move": ("kt", "response"),
    "static-embedding|kt-rt|move": ("kt-rt", "response"),
    "evolving-latent|kt-rt|move": ("kt-rt", "response"),
    "sft-lora|kt (1.7b)|response": ("kt", "response"),
    "sft-lora|kt (8b)|response": ("kt", "response"),
    "persona-frozen|kt (1.7b)|response": ("kt", "response"),
    "persona-frozen|kt (8b)|response": ("kt", "response"),
}
# timing combination -> domain of marginal_timing.json
TIMING = {
    "static-embedding|chess-pooled|timing": "chess-pooled",
    "evolving-latent|chess-pooled|timing": "chess-pooled",
    "static-embedding|kt-rt|timing": "kt-rt",
    "evolving-latent|kt-rt|timing": "kt-rt",
    "sft-lora|chess (1.7b)|timing": "chess-pooled",
    "sft-lora|chess (8b)|timing": "chess-pooled",
    "persona-frozen|chess (1.7b)|timing": "chess-pooled",
    "persona-frozen|chess (8b)|timing": "chess-pooled",
}
assert set(DISCRETE) | set(TIMING) == set(DISPLAY) and not set(DISCRETE) & set(TIMING)


def main() -> None:
    rng = np.random.default_rng(0)

    def boot(delta):
        n = len(delta)
        idx = rng.integers(0, n, size=(B, n))
        means = delta[idx].mean(axis=1)
        return {
            'point': float(delta.mean()),
            'low': float(np.percentile(means, 2.5)),
            'high': float(np.percentile(means, 97.5)),
        }

    cells = load_cells()
    assert len(cells) == 84, len(cells)
    for c in cells:
        a = c['arms']
        c['fixed'] = {
            'dtot': boot(a['A4f'] - a['A0']),
            'dpop': boot(a['A4f'] - a['A1']),
            'dgrp': boot(a['A4f'] - a['A2']),
            'dind': boot(a['A4f'] - a['A3']),
        }
        c['drift'] = float((a['A4f'] - a['A4']).mean())

    gains = json.loads((paths.ANALYSIS / 'gains.json').read_text())
    holm = set(gains['rows']['dind']['holm_pooled']['target_keys'])
    discrete = json.loads((MARGINAL / 'marginal_per_user.json').read_text())
    timing = json.loads((MARGINAL / 'marginal_timing.json').read_text())

    CM = {c['key']: c for c in cells}
    ctrl = []
    for key, disp in DISPLAY.items():
        c = CM[key]
        a = c['arms']
        pos = {u: i for i, u in enumerate(c['users'])}
        dind = c['fixed']['dind']
        if key in DISCRETE:
            node = discrete[DISCRETE[key][0]][DISCRETE[key][1]]
            own = dict(zip(node['users'], node['per_user_nll']['own']))
            other = dict(zip(node['users'], node['per_user_nll']['random_other']))
            common = [u for u in c['users'] if u in own]
            idx = [pos[u] for u in common]
            marg = np.array([own[u] - other[u] for u in common])
            final = float(((a['A4f'] - a['A3'])[idx] - marg).mean())
            live = None
            source = 'marginal_discrete'
        else:
            node = timing[TIMING[key]]
            by_user = dict(zip(node['users'], node['marginal_dindiv']))
            common = [u for u in c['users'] if u in by_user]
            idx = [pos[u] for u in common]
            marg = np.array([by_user[u] for u in common])
            live = float(((a['A4'] - a['A3'])[idx] - marg).mean())
            final = live + c['drift']  # (A4f-A3)-marg = (A4-A3)-marg + (A4f-A4)
            source = 'marginal_timing'
        nd = not (dind['high'] < 0 or dind['low'] > 0)  # ratio has no stable reading
        ctrl.append(
            {
                'key': key,
                'display': disp,
                'dind': dind,
                'n_common': len(common),
                'marginal_point': float(marg.mean()),
                'residual_live': live,
                'drift': c['drift'],
                'residual_point_final': final,
                'residual_source': source,
                'pooled_holm': key in holm,
                'dind_nd': nd,
                'static': key.startswith('static-embedding'),
            }
        )

    n_int = [x for x in ctrl if not x['dind_nd']]
    summary = {
        'n_cells': len(ctrl),
        'interpretable': len(n_int),
        'marginal_matches_or_exceeds_point': sum(x['residual_point_final'] >= 0 for x in n_int),
        'residual_favorable_point': sum(x['residual_point_final'] < 0 for x in n_int),
        'pooled_holm_passes': [x['display'] for x in ctrl if x['pooled_holm']],
        'pooled_holm_matched_by_marginal': [
            x['display'] for x in ctrl if x['pooled_holm'] and x['residual_point_final'] >= 0
        ],
        'residual_favorable_cells': [x['display'] for x in n_int if x['residual_point_final'] < 0],
        'nd_cells': [x['display'] for x in ctrl if x['dind_nd']],
    }
    out = paths.ensure(MARGINAL) / 'marginal_controls.json'
    out.write_text(json.dumps({'summary': summary, 'cells': ctrl}, indent=1))
    print('summary:', json.dumps(summary, indent=1))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
