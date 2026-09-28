"""Counts and shares quoted in the Results section (Analysing AT audits).

Inputs store target-minus-reference surprisal, so a gain (reference minus
target) is significant when the stored interval lies below zero, and positive
when the stored point is below zero.  Shares are point estimates.

Reads paths.ANALYSIS / {"gains.json", "identity.json"} and
paths.COMPARISON / "marginal" / "marginal_controls.json".
Writes paths.ANALYSIS / "shares.json".
Run:  python -m attribution_trials.analysis.shares
"""

import json

import numpy as np

from attribution_trials import paths

GAINS = paths.ANALYSIS / 'gains.json'
IDENTITY = paths.ANALYSIS / 'identity.json'
MARGINAL = paths.COMPARISON / 'marginal' / 'marginal_controls.json'
OUT = paths.ANALYSIS / 'shares.json'

NAMES = ('dtot', 'dpop', 'dgrp', 'dind')
EMBEDDING = ('static-embedding', 'evolving-latent', 'structured-memory')
FROZEN_PROMPT = ('persona-frozen', 'strong-open')


def significant(x):
    return x['high'] < 0


def adverse(x):
    return x['low'] > 0


def positive(x):
    return x['point'] < 0


def family(c):
    return c['key'].split('|')[0]


def main():
    cells = json.loads(GAINS.read_text())['cells']
    holm_ind = {
        c['key']: c['identity']['holm_pooled'] for c in json.loads(IDENTITY.read_text())['cells']
    }
    assert len(cells) == len(holm_ind) == 84
    for c in cells:
        c['holm'] = {k: c['holm_pooled'][k]['target'] for k in NAMES[:3]}
        c['holm']['dind'] = holm_ind[c['key']]
    t = lambda c, k: c['target'][k]

    passes = [c for c in cells if c['holm']['dind']]
    all_positive = [c for c in cells if all(positive(t(c, k)) for k in NAMES)]
    n_holm = [sum(c['holm'].values()) for c in cells]

    # combinations where the target's information helps against no information
    helped = [c for c in cells if significant(t(c, 'dtot'))]
    # share of that gain population information already gives: (A0 - A1) / (A0 - A4f)
    pop_share = [
        (t(c, 'dtot')['point'] - t(c, 'dpop')['point']) / t(c, 'dtot')['point'] for c in helped
    ]
    beats_pop = [c for c in helped if significant(t(c, 'dpop'))]

    frozen = [c for c in cells if family(c) in FROZEN_PROMPT]
    frozen_helped = [c for c in frozen if significant(t(c, 'dtot'))]
    frozen_adverse = [c for c in frozen if adverse(t(c, 'dtot'))]

    # per-user marginal baseline on the combinations significant against the imposter
    marg = [m for m in json.loads(MARGINAL.read_text())['cells'] if significant(m['dind'])]
    marg_beats = [m for m in marg if m['residual_point_final'] < 0]

    out = {
        'significant': {k: sum(significant(t(c, k)) for c in cells) for k in NAMES},
        'holm_significant_vs_imposter': len(passes),
        'holm_passes': [c['key'] for c in passes],
        'holm_passes_language_model': sum(family(c) not in EMBEDDING for c in passes),
        'n_language_model': sum(family(c) not in EMBEDDING for c in cells),
        'holm_passes_worse_than_none': sum(not positive(t(c, 'dtot')) for c in passes),
        'all_four_positive': len(all_positive),
        'max_holm_significant_gains': max(n_holm),
        'max_holm_significant_gains_all_four_positive': max(
            sum(c['holm'].values()) for c in all_positive
        )
        if all_positive
        else 0,
        'significant_vs_none': len(helped),
        'population_share_of_none_gain_median_pct': float(100 * np.median(pop_share)),
        'not_better_than_population': len(helped) - len(beats_pop),
        'better_than_population': len(beats_pop),
        'imposter_as_good': sum(not significant(t(c, 'dind')) for c in beats_pop),
        'frozen_prompt': {
            'n': len(frozen),
            'significant_vs_none': len(frozen_helped),
            'imposter_significant_vs_none': sum(
                significant(c['stranger']['dtot']) for c in frozen_helped
            ),
            'significant_vs_imposter': sum(significant(t(c, 'dind')) for c in frozen),
            'holm_significant_vs_imposter': sum(c['holm']['dind'] for c in frozen),
            'adverse_vs_none': len(frozen_adverse),
            'adverse_vs_none_timing': sum('timing' in c['key'].split('|') for c in frozen_adverse),
        },
        'marginal': {
            'significant_vs_imposter': len(marg),
            'marginal_matches_or_exceeds': len(marg) - len(marg_beats),
            'method_beyond_marginal': {m['key']: -m['residual_point_final'] for m in marg_beats},
        },
    }
    paths.ensure(OUT.parent)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
