"""Data for the headline figure's panel (a): OPeRA timing populations in a PCA basis.

Each real user's profile is their normalized histogram of held-out timing
labels (users with at least 3 decisions).  A two-component PCA is fitted on
the real users; each simulator's profile is its per-user mean predictive
distribution under the target's information (A4frozen), projected into the
same basis.

Inputs: <AUDIT>/opera/frozen_opera-timing-combined_8b.json (real players and
labels) and four <COMPARISON>/per_option/*.npz files.

  python -m attribution_trials.figures.headline_data
Output: <FIGURES>/opera_timing_pca.json
"""

import collections
import json
from pathlib import Path

import numpy as np

from attribution_trials import paths

REFERENCE = paths.AUDIT / 'opera' / 'frozen_opera-timing-combined_8b.json'
PER_OPTION = paths.COMPARISON / 'per_option'
OUTPUT = paths.FIGURES / 'opera_timing_pca.json'
METHODS = {
    'Qwen3-8B': 'persona-frozen_opera-timing-combined_8b',
    'Osim-8B': 'trained-sim_opera-timing-combined_osim-8b',
    'CoSER-8B': 'sim2real-trio_opera-timing-combined_coser-8b',
    'HumanLike-7B': 'sim2real-trio_opera-timing-combined_humanlike-7b',
}


def load_real(path: Path, min_count: int = 3):
    data = json.loads(path.read_text())
    players, labels = data["players"], data["labels"]
    vocab = sorted(set(labels))
    if "timing" in path.name:
        order = ["under 1.5s", "1.5 to 5s", "5 to 20s", "20 to 60s", "over 60s"]
        vocab = [v for v in order if v in vocab] + [v for v in vocab if v not in order]
    index = {value: i for i, value in enumerate(vocab)}
    per_user = collections.defaultdict(lambda: np.zeros(len(vocab)))
    for player, label in zip(players, labels):
        per_user[player][index[label]] += 1
    users = [u for u in sorted(per_user) if per_user[u].sum() >= min_count]
    profiles = np.stack([per_user[u] / per_user[u].sum() for u in users])
    return profiles, users, vocab, data


def pca(real: np.ndarray):
    mean = real.mean(axis=0)
    centered = real - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance_ratio = singular**2 / (singular**2).sum()
    return mean, vt[:2], variance_ratio[:2]


def main():
    real, users, vocab, ref = load_real(REFERENCE)
    mean, components, variance = pca(real)
    project = lambda a: ((a - mean) @ components.T).tolist()
    profiles = {'Real users': project(real)}
    for name, stem in METHODS.items():
        with np.load(PER_OPTION / f'{stem}.npz', allow_pickle=False) as values:
            players = values['players']
            labels = values['label_set'][values['labels']]
            # every method is scored on the reference's held-out rows
            assert players.tolist() == ref['players']
            assert labels.tolist() == ref['labels']
            logp = values['logp_A4f']
            prob = np.exp(logp - logp.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            cols = [values['label_set'].tolist().index(v) for v in vocab]
            implied = np.stack([prob[players == user][:, cols].mean(axis=0) for user in users])
            profiles[name] = project(implied)
    output = {'n_users': len(users), 'variance_ratio': variance.tolist(), 'profiles': profiles}
    paths.ensure(OUTPUT.parent)
    OUTPUT.write_text(json.dumps(output, indent=2) + '\n')
    print(f'Wrote {len(profiles)} populations × {len(users)} users to {OUTPUT}.')


if __name__ == '__main__':
    main()
