# -*- coding: utf-8 -*-
"""Per-combination ladder CSV: the fixed-profile target-versus-imposter gain of
all 84 combinations with its interval and Holm flags, plus the row metadata the
appendix tables print.

Source: paths.ANALYSIS / "identity.json" (the `identity` endpoint, A4frozen - A3,
stored as target-minus-imposter surprisal; negative favors the target).
Writes paths.ANALYSIS / "master_ladder.csv".
Run:  python -m attribution_trials.analysis.master_ladder
"""

import csv
import json

from attribution_trials import paths

SRC = paths.ANALYSIS / 'identity.json'
OUT = paths.ANALYSIS / 'master_ladder.csv'

FIELDS = [
    'source',
    'cell',
    'family',
    'channel',
    'config',
    'model',
    'N',
    'unit',
    'dindiv_point',
    'dindiv_low',
    'dindiv_high',
    'dindiv_p_flip',
    'sign_favorable',
    'ci_excludes_0',
    'holm_reject_within_source',
    'holm_reject_pooled_evidence',
]


def metadata(source, key):
    """family / channel / config / model from the combination key.

    Chess and KT keys are family|substrate|channel; OPeRA keys are
    family|channel|config|model.
    """
    parts = key.split('|')
    if source == 'full-tier':
        family, config, channel = parts
        model = ''
    else:
        family, channel, config, model = parts
    return {'family': family, 'channel': channel, 'config': config, 'model': model}


def main():
    cells = json.loads(SRC.read_text())['cells']
    assert len(cells) == 84, len(cells)
    rows = []
    for c in cells:
        ident = c['identity']
        rows.append(
            {
                'source': c['source'],
                'cell': c['key'],
                **metadata(c['source'], c['key']),
                'N': c['n_users'],
                'unit': 'nats/decision',
                'dindiv_point': ident['point'],
                'dindiv_low': ident['low'],
                'dindiv_high': ident['high'],
                'dindiv_p_flip': ident['p_flip'],
                'sign_favorable': str(ident['point'] < 0),
                'ci_excludes_0': str(ident['high'] < 0 or ident['low'] > 0),
                'holm_reject_within_source': str(bool(ident['holm_within'])),
                'holm_reject_pooled_evidence': str(bool(ident['holm_pooled'])),
            }
        )
    rows.sort(key=lambda r: (r['source'], r['cell']))
    paths.ensure(OUT.parent)
    with OUT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows), 'rows')


if __name__ == '__main__':
    main()
