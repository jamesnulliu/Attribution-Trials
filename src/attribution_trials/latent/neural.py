"""Trainable recurrent latent: the recurrent-embedding family.

A GRU accumulates the per-step
:func:`~attribution_trials.latent.structured.history_features` into an
evolving hidden state ``h_t`` that conditions the backbone. The memoryless
twin (``persist=False``) has identical parameters and inputs but resets the
state every step; it is the no-information condition. The population,
group, imposter and target conditions replay stored latent trajectories
through the trained recurrent model
(:mod:`attribution_trials.bench.evolving_family`).

torch is imported lazily; the network is built on first use.
"""

from __future__ import annotations

from attribution_trials.interface import DecisionPoint
from attribution_trials.latent.base import (
    Injection,
    InjectionKind,
    LatentState,
    LatentStateInjector,
    Observation,
)
from attribution_trials.latent.structured import (
    _Z,
    DIMENSIONS,
    StructuredInjector,
    history_features,
)


class NeuralInjector(LatentStateInjector):
    """A GRU recurrence over the shared history features.

    Parameters
    ----------
    kind:
        Injection channel advertised via :attr:`produces` (the backbones
        consume ``HIDDEN``).
    latent_dim:
        Width of the recurrent hidden state ``h_t``.
    seed:
        Seeds parameter initialisation so construction is reproducible without
        perturbing the global torch RNG (uses a forked RNG).
    persist:
        Whether the hidden state carries across steps. ``True`` is the
        recurrent latent (the target condition's model). ``False`` zeroes the
        state before every step, giving a **memoryless** twin with identical
        parameters and inputs (the no-information condition). With
        ``persist=False`` the GRU's hidden-to-hidden weights receive no
        gradient, since their input is always zero.
    """

    def __init__(
        self,
        kind: InjectionKind = InjectionKind.VERBAL,
        latent_dim: int = 8,
        seed: int = 0,
        persist: bool = True,
    ) -> None:
        self.kind = kind
        self.latent_dim = latent_dim
        self.seed = seed
        self.persist = persist
        self.produces = (kind,)
        self.input_dim = len(DIMENSIONS)
        # The structured injector renders the per-step Injection.
        self._renderer = StructuredInjector(kind=kind)
        self._net = None  # built lazily on the first call needing torch

    # --- network construction ------------------------------------------
    def _build(self):
        if self._net is not None:
            return self._net
        try:
            import torch
            from torch import nn
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "torch required for NeuralInjector; install the 'train' "
                "extra: pip install '.[train]'."
            ) from e

        class _Net(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.cell = nn.GRUCell(input_dim, hidden_dim)
                # Readout to the anchored dimensions (per-step injection view).
                self.readout = nn.Linear(hidden_dim, len(DIMENSIONS))

            def step(self, x, h):
                return self.cell(x, h)

            def anchored(self, h):
                # First three anchored dims are severities in [0, 1]
                # (sigmoid); momentum is signed in [-1, 1] (tanh).
                raw = self.readout(h)
                sev = torch.sigmoid(raw[..., :3])
                mom = torch.tanh(raw[..., 3:4])
                return torch.cat([sev, mom], dim=-1)

        # Build with a forked RNG so seeding is reproducible and side-effect
        # free w.r.t. any surrounding training run.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            net = _Net(self.input_dim, self.latent_dim)
        self._net = net
        return net

    def parameters(self):
        """torch parameters (GRU + readout) for :class:`SFTTrainer` to fit."""
        return self._build().parameters()

    def to(self, device):
        """Move ``f_phi`` to a torch device (returns self, for chaining)."""
        self._build().to(device)
        return self

    def latent_trajectory(self, feats_seq, player_ids=None):
        """Differentiable recurrence over a whole trajectory.

        ``feats_seq`` is a ``[T, B, input_dim]`` tensor of per-step
        :func:`history_features`. Returns the hidden-state sequence
        ``[T, B, latent_dim]`` -- the latent the backbone consumes during
        training and scoring.

        ``player_ids`` is accepted (and ignored) so this injector is
        interchangeable with identity-keyed injectors like
        :class:`~attribution_trials.latent.static_individual.StaticIndividualInjector`.

        When :attr:`persist` is ``False`` the hidden state is reset to zero
        before each step, so the output depends only on the current features.
        """
        net = self._build()
        steps, batch, _ = feats_seq.shape
        h = feats_seq.new_zeros(batch, self.latent_dim)
        outs = []
        for t in range(steps):
            h_in = h if self.persist else feats_seq.new_zeros(batch, self.latent_dim)
            h = net.step(feats_seq[t], h_in)
            outs.append(h)
        import torch

        return torch.stack(outs, dim=0)

    # --- helpers --------------------------------------------------------
    def _features_tensor(self, dp: DecisionPoint):
        import torch

        feats = history_features(dp)
        return torch.tensor([feats[d] for d in DIMENSIONS], dtype=torch.float32)

    def _anchored_list(self, h) -> list[float]:
        net = self._build()
        vec = net.anchored(h).detach().reshape(-1).tolist()
        return [float(x) for x in vec]

    # --- lifecycle ------------------------------------------------------
    def initial_state(self, player_id: str) -> LatentState:
        import torch

        self._build()
        h0 = torch.zeros(self.latent_dim, dtype=torch.float32)
        return LatentState(
            payload=h0,
            probe_vector=self._anchored_list(h0),
            meta={"player_id": player_id},
        )

    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        # Read the learned state out to the anchored dims, then render through
        # the structured renderer.
        anchored = self._anchored_list(state.payload)
        z = _Z(values=dict(zip(DIMENSIONS, anchored)))
        return self._renderer.render(LatentState(payload=z, probe_vector=z.as_vector()), dp)

    def update(
        self,
        state: LatentState,
        dp: DecisionPoint,
        observed: Observation | None = None,
    ) -> LatentState:
        # ``observed`` is unused: the recurrence reads only history_features,
        # the same input the memoryless twin sees.
        net = self._build()
        x = self._features_tensor(dp)
        h_new = net.step(x.unsqueeze(0), state.payload.unsqueeze(0)).squeeze(0)
        return LatentState(
            payload=h_new,
            probe_vector=self._anchored_list(h_new),
            meta=state.meta,
        )

    # --- capacity bookkeeping ------------------------------------------
    def param_report(self) -> dict[str, object]:
        """Parameter count of the injector."""
        net = self._build()
        total = sum(p.numel() for p in net.parameters())
        return {
            "injector": type(self).__name__,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "n_parameters": int(total),
        }

    @property
    def name(self) -> str:
        return f"NeuralInjector(latent_dim={self.latent_dim}, kind={self.kind.value})"
