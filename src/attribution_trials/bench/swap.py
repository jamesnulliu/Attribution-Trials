"""Condition wrappers around trained injectors.

Conditions are realised either by remapping the identity a trained injector
sees or by replacing the latent with a row average of a trained embedding.
Arm labels: A0 = none, A1 = population, A2 = group, A3 = imposter,
A4 = target. Wrappers satisfy the duck-typed contract the trainer and
`ec._per_player_nlls` rely on: `latent_trajectory(feats, player_ids)`,
`parameters()`, `to(device)`, `_build()`.
"""

from __future__ import annotations

import random
from collections.abc import Mapping


class IdRemapInjector:
    """Delegate injector that remaps player ids before the base sees them.

    Uses: the imposter condition (A3, a within-group derangement applied
    to the trained target model at evaluation) and the retrained
    population/group conditions (A1/A2: train the base embedding with all
    users mapped to one population id or to their group id, and evaluate
    under the same map).
    """

    def __init__(self, base, id_map: Mapping[str, str]):
        self.base = base
        self.id_map = dict(id_map)
        self.kind = base.kind
        self.produces = base.produces
        self.latent_dim = base.latent_dim

    def _remap(self, player_ids):
        if player_ids is None:
            return None
        return [self.id_map.get(p, p) for p in player_ids]

    def latent_trajectory(self, feats_seq, player_ids=None):
        return self.base.latent_trajectory(feats_seq, player_ids=self._remap(player_ids))

    def parameters(self):
        return self.base.parameters()

    def to(self, device):
        self.base.to(device)
        return self

    def _build(self):
        return self.base._build()

    def param_report(self):
        rep = dict(self.base.param_report())
        rep["wrapper"] = "IdRemapInjector"
        return rep

    @property
    def name(self) -> str:
        return f"IdRemap({self.base.name})"


class MeanLatentInjector:
    """Replace the per-player latent with a mean over trained rows.

    The "averaged" construction of A1/A2: given a trained
    `StaticIndividualInjector`, emit for every user the mean embedding
    over the rows of their group (A2) or over all rows (A1, pass a single
    shared group). Evaluation-time only; carries no trainable state.
    """

    def __init__(self, base, group_members: Mapping[str, list[str]]):
        # group_members: player_id -> list of player_ids to average over.
        self.base = base
        self.group_members = {k: list(v) for k, v in group_members.items()}
        self.kind = base.kind
        self.produces = base.produces
        self.latent_dim = base.latent_dim

    def _mean_vec(self, player_id):
        import torch

        net = self.base._build()
        members = self.group_members[player_id]
        rows = [self.base.player_to_idx[m] for m in members]
        idx = torch.tensor(rows, dtype=torch.long, device=net.weight.device)
        return net(idx).mean(dim=0)

    def latent_trajectory(self, feats_seq, player_ids=None):
        import torch

        steps, batch, _ = feats_seq.shape
        if player_ids is None:
            raise ValueError("MeanLatentInjector needs player_ids")
        vecs = torch.stack([self._mean_vec(p) for p in player_ids])  # [B,L]
        return vecs.unsqueeze(0).expand(steps, batch, self.latent_dim)

    def parameters(self):
        return self.base.parameters()

    def to(self, device):
        self.base.to(device)
        return self

    def _build(self):
        return self.base._build()

    @property
    def name(self) -> str:
        return f"MeanLatent({self.base.name})"


def within_cell_derangement(cell_of: Mapping[str, str], seed: int = 0) -> dict[str, str]:
    """Map each user to a *different* user in the same group (the imposter).

    Cyclic shift within each group after a seeded shuffle: no fixed point
    for groups of size >= 2. Singleton groups map to themselves; the caller
    excludes those users (their imposter condition equals the target).
    """
    rng = random.Random(seed)
    by_cell: dict[str, list[str]] = {}
    for p, c in cell_of.items():
        by_cell.setdefault(c, []).append(p)
    mapping: dict[str, str] = {}
    for members in by_cell.values():
        members = sorted(members)
        rng.shuffle(members)
        n = len(members)
        for i, p in enumerate(members):
            mapping[p] = members[(i + 1) % n]
    return mapping
