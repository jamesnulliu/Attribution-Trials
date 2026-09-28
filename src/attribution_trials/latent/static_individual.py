"""Static per-user embedding: the static-embedding family.

One free vector per user (an ``nn.Embedding``), returned unchanged at every
step, so the latent carries identity but no temporal evolution. Conditions
are built by changing which row a user reads
(:class:`~attribution_trials.bench.swap.IdRemapInjector`): the population and
group conditions train a single shared row or one row per group (or average
the trained rows), the imposter reads a matched donor's row, and the target
reads the user's own.

``latent_trajectory`` has the same ``[T, B, latent_dim]`` contract as
:class:`~attribution_trials.latent.neural.NeuralInjector`, so the trainer
calls both the same way. torch is imported lazily; initialisation is seeded
through a forked RNG, independent of global RNG state.
"""

from __future__ import annotations

from collections.abc import Iterable

from attribution_trials.interface import DecisionPoint
from attribution_trials.latent.base import (
    Injection,
    InjectionKind,
    LatentState,
    LatentStateInjector,
    Observation,
)


class StaticIndividualInjector(LatentStateInjector):
    """Per-user learned embedding, constant over the trajectory."""

    def __init__(
        self,
        player_ids: Iterable[str],
        kind: InjectionKind = InjectionKind.HIDDEN,
        latent_dim: int = 16,
        seed: int = 0,
    ) -> None:
        self.kind = kind
        self.produces = (kind,)
        self.latent_dim = latent_dim
        self.seed = seed
        # Stable id->row map (sorted for determinism). An unknown id falls
        # back to row 0.
        self.player_to_idx = {p: i for i, p in enumerate(sorted(set(player_ids)))}
        self._net = None

    def _build(self):
        if self._net is not None:
            return self._net
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for StaticIndividualInjector; install the "
                "'train' extra: pip install '.[train]'."
            ) from e

        n = max(1, len(self.player_to_idx))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self._net = nn.Embedding(n, self.latent_dim)
        return self._net

    def parameters(self):
        return self._build().parameters()

    def to(self, device):
        self._build().to(device)
        return self

    def _idx_tensor(self, player_ids):
        import torch

        net = self._build()
        device = net.weight.device
        rows = [self.player_to_idx.get(p, 0) for p in player_ids]
        return torch.tensor(rows, dtype=torch.long, device=device)

    def latent_trajectory(self, feats_seq, player_ids=None):
        """Per-player embedding broadcast over time: ``[T, B, latent_dim]``.

        ``feats_seq`` is used only for its ``[T, B, *]`` shape; the latent is a
        function of identity (``player_ids``), not the history features.
        """
        net = self._build()
        steps, batch, _ = feats_seq.shape
        if player_ids is None:
            # No identity available -> a single shared row (population-style).
            import torch

            idx = torch.zeros(batch, dtype=torch.long, device=net.weight.device)
        else:
            idx = self._idx_tensor(player_ids)
        emb = net(idx)  # [B, latent_dim]
        return emb.unsqueeze(0).expand(steps, batch, self.latent_dim)

    # --- per-step lifecycle ---------------------------------------------
    def initial_state(self, player_id: str) -> LatentState:
        idx = self._idx_tensor([player_id])
        vec = self._build()(idx).squeeze(0)
        return LatentState(
            payload=vec,
            probe_vector=[float(x) for x in vec.detach().tolist()],
            meta={"player_id": player_id},
        )

    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        vec = [float(x) for x in state.payload.detach().tolist()]
        return Injection(kind=self.kind, vector=vec)

    def update(
        self,
        state: LatentState,
        dp: DecisionPoint,
        observed: Observation | None = None,
    ) -> LatentState:
        return state  # static: the per-player vector never changes

    def param_report(self) -> dict[str, object]:
        net = self._build()
        return {
            "injector": type(self).__name__,
            "n_players": len(self.player_to_idx),
            "latent_dim": self.latent_dim,
            "n_parameters": int(sum(p.numel() for p in net.parameters())),
        }

    @property
    def name(self) -> str:
        return (
            f"StaticIndividualInjector(n_players={len(self.player_to_idx)}, "
            f"latent_dim={self.latent_dim})"
        )
