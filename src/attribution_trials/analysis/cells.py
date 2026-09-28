# -*- coding: utf-8 -*-
"""Per-user arm loaders for the 84 combinations.

Arms: A0 none, A1 population, A2 group, A3 imposter, A4 target with a profile
updated on held-out rows, A4f target with the profile fixed at the training
boundary.  The fixed target arm per family:
  static embedding            A4f = A4 (no updated variant)
  recurrent embedding         A4f = A4 + drift_per_player[ch]
  structured memory           A4f = A4 + drift.refit.per_player
  user-profile LoRA           nll.self_frozen
  frozen prompt               nll.A4frozen
  Osim, CoSER, HumanLike      nll.A4frozen
Trained families average each user's score over the three training seeds.
Every cell record carries the combination key used by every table.
"""

import json
from collections import defaultdict

import numpy as np

from attribution_trials import paths

SEEDS = (0, 1, 2)
CHESS_KT = paths.AUDIT / 'chess_kt'
OPERA = paths.AUDIT / 'opera'
RELEASED = paths.AUDIT / 'released'
ARMS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A4f')


def by_player(nll, players):
    d = defaultdict(list)
    for p, x in zip(players, nll):
        if x is not None:
            d[p].append(x)
    return {p: float(np.mean(v)) for p, v in d.items() if v}


def _finish(panel, key, dataset, family, channel, tag, arms):
    common = None
    for a in ARMS:
        if a not in arms:
            return None
        common = set(arms[a]) if common is None else common & set(arms[a])
    common = sorted(common)
    if len(common) < 5:
        return None
    return {
        'panel': panel,
        'key': key,
        'dataset': dataset,
        'family': family,
        'channel': channel,
        'tag': tag,
        'n': len(common),
        'arms': {a: np.array([arms[a][p] for p in common]) for a in ARMS},
        'users': common,
    }


def _ladder(files, grade, ch, drift_getter=None):
    acc, ids, drifts = defaultdict(list), None, []
    for f in files:
        blob = json.loads(f.read_text())
        g = blob['grades']
        g = g[grade] if grade in g else g['retrained']
        blk = g[ch]
        ids = blk['player_ids'] if ids is None else ids
        assert blk['player_ids'] == ids
        for a, v in blk['per_player_nll'].items():
            acc[a].append(v)
        if drift_getter is not None:
            drifts.append(drift_getter(blob, ch))
    arms = {a: dict(zip(ids, np.mean(v, axis=0))) for a, v in acc.items()}
    if drift_getter is None:
        arms['A4f'] = dict(arms['A4'])
    else:
        dr = dict(zip(ids, np.mean(drifts, axis=0)))
        arms['A4f'] = {p: arms['A4'][p] + dr[p] for p in ids}
    return arms


def load_cells():
    cells = []
    # ---- chess: static embedding / recurrent embedding
    for key_fam, pat, grade, drift in (
        ('static-embedding', 'full_ladder_chess_s%d.json', 'retrained', None),
        (
            'evolving-latent',
            'full_evolving_chess_s%d.json',
            'eval-swap',
            lambda b, ch: b['drift_per_player'][ch],
        ),
    ):
        files = [CHESS_KT / (pat % s) for s in SEEDS]
        for ch in ('move', 'timing'):
            arms = _ladder(files, grade, ch, drift)
            cells.append(
                _finish('full tier', f'{key_fam}|chess-pooled|{ch}', 'chess', key_fam, ch, '', arms)
            )
    # ---- knowledge tracing: response-only (kt) and response + timing (kt-rt) models
    for key_fam, pat, grade, sub, chans, drift in (
        ('static-embedding', 'small_ladder_kt_s%d.json', 'retrained', 'kt', ('move',), None),
        (
            'static-embedding',
            'small_ladder_kt-rt_s%d.json',
            'retrained',
            'kt-rt',
            ('move', 'timing'),
            None,
        ),
        (
            'evolving-latent',
            'evolving_kt-rt_s%d.json',
            'eval-swap',
            'kt-rt',
            ('move', 'timing'),
            lambda b, ch: b['drift_per_player'][ch],
        ),
        (
            'structured-memory',
            'memory_pilot_kt_s%d.json',
            'refit',
            'kt',
            ('response',),
            lambda b, ch: b['drift']['refit']['per_player'],
        ),
    ):
        files = [CHESS_KT / (pat % s) for s in SEEDS]
        for ch in chans:
            arms = _ladder(files, grade, ch, drift)
            disp = 'response' if ch in ('move', 'response') else 'timing'
            cells.append(
                _finish('full tier', f'{key_fam}|{sub}|{ch}', sub, key_fam, disp, '', arms)
            )
    # ---- chess / KT: user-profile LoRA (sft) and frozen prompt
    for sub, ch in (('chess', 'timing'), ('kt', 'response')):
        for size in ('1.7b', '8b'):
            per = defaultdict(lambda: defaultdict(list))
            for s in SEEDS:
                b = {
                    c: json.loads((CHESS_KT / f'sft_{sub}_{c}_{size}_s{s}.json').read_text())
                    for c in ('none', 'placebo', 'group', 'self')
                }
                pm = {
                    'A0': by_player(b['none']['nll']['eval'], b['none']['players']),
                    'A1': by_player(b['placebo']['nll']['eval'], b['placebo']['players']),
                    'A2': by_player(b['group']['nll']['eval'], b['group']['players']),
                    'A3': by_player(b['self']['nll']['imposter'], b['self']['players']),
                    'A4': by_player(b['self']['nll']['self_live'], b['self']['players']),
                    'A4f': by_player(b['self']['nll']['self_frozen'], b['self']['players']),
                }
                for a, m in pm.items():
                    for p, v in m.items():
                        per[a][p].append(v)
            arms = {a: {p: float(np.mean(v)) for p, v in per[a].items()} for a in per}
            cells.append(
                _finish(
                    'full tier', f'sft-lora|{sub} ({size})|{ch}', sub, 'sft-lora', ch, size, arms
                )
            )
            d = json.loads((CHESS_KT / f'frozen_{sub}_{size}.json').read_text())
            arms = {a: by_player(d['nll'][a], d['players']) for a in d['nll']}
            arms['A4f'] = arms.pop('A4frozen')
            cells.append(
                _finish(
                    'full tier',
                    f'persona-frozen|{sub} ({size})|{ch}',
                    sub,
                    'persona-frozen',
                    ch,
                    size,
                    arms,
                )
            )
    # ---- OPeRA: frozen prompt panel (1.7b, 8b, 30b) and user-profile LoRA
    for f in sorted(OPERA.glob('frozen_opera-*.json')):
        stem = f.stem
        body, model = stem.split('_')[1], stem.split('_', 2)[2]
        _, ch, cfg = body.split('-')
        if model not in ('1.7b', '8b', '30b'):
            continue  # API models are outside the 84 (see analysis.api_rows)
        d = json.loads(f.read_text())
        arms = {a: by_player(d['nll'][a], d['players']) for a in d['nll']}
        arms['A4f'] = arms.pop('A4frozen')
        fam = 'strong-open' if model == '30b' else 'persona-frozen'
        cells.append(
            _finish(
                'OPeRA GPU', f'{fam}|{ch}|{cfg}|{model}', 'opera', fam, ch, f'{model}/{cfg}', arms
            )
        )
    for ch in ('action', 'timing'):
        for cfg in ('combined', 'static', 'dynamic'):
            for size in ('1.7b', '8b'):
                per = defaultdict(lambda: defaultdict(list))
                n_seeds = 0
                for s in SEEDS:
                    # the no-information run does not depend on the profile config
                    none_f = OPERA / f'sft_opera-{ch}-combined_none_{size}_s{s}.json'
                    need = {
                        c: OPERA / f'sft_opera-{ch}-{cfg}_{c}_{size}_s{s}.json'
                        for c in ('placebo', 'group', 'self')
                    }
                    if not none_f.exists() or not all(x.exists() for x in need.values()):
                        continue
                    n_seeds += 1
                    bn = json.loads(none_f.read_text())
                    b = {c: json.loads(x.read_text()) for c, x in need.items()}
                    pm = {
                        'A0': by_player(bn['nll']['eval'], bn['players']),
                        'A1': by_player(b['placebo']['nll']['eval'], b['placebo']['players']),
                        'A2': by_player(b['group']['nll']['eval'], b['group']['players']),
                        'A3': by_player(b['self']['nll']['imposter'], b['self']['players']),
                        'A4': by_player(b['self']['nll']['self_live'], b['self']['players']),
                        'A4f': by_player(b['self']['nll']['self_frozen'], b['self']['players']),
                    }
                    for a, m in pm.items():
                        for p, v in m.items():
                            per[a][p].append(v)
                if n_seeds:
                    arms = {a: {p: float(np.mean(v)) for p, v in per[a].items()} for a in per}
                    cells.append(
                        _finish(
                            'OPeRA GPU',
                            f'sft-lora|{ch}|{cfg}|{size}',
                            'opera',
                            'sft-lora',
                            ch,
                            f'{size}/{cfg}',
                            arms,
                        )
                    )
    # ---- released simulators on OPeRA under their native interface: Osim, CoSER / HumanLike
    for pat, panel, keyfam in (
        ('osim-panel_*.json', 'Osim', 'trained-sim'),
        ('trio-panel_*_chat.json', 'Sim2Real trio', 'sim2real-trio'),
    ):
        for f in sorted(RELEASED.glob(pat)):
            d = json.loads(f.read_text())
            _, ch, cfg = d['substrate'].split('-')
            arms = {a: by_player(d['nll'][a], d['players']) for a in d['nll']}
            arms['A4f'] = arms.pop('A4frozen')
            cells.append(
                _finish(
                    panel,
                    f"{keyfam}|{ch}|{cfg}|{d['size']}",
                    'opera',
                    'released simulator',
                    ch,
                    f"{d['size']}/{cfg}",
                    arms,
                )
            )
    cells = [c for c in cells if c is not None]
    return cells
