"""History features and the structured renderer shared by the latent injectors.

:func:`history_features` is the per-step input of the recurrent latent
(:class:`~attribution_trials.latent.neural.NeuralInjector`) and of both
backbones: four anchored dimensions describing how the session has gone so
far. :class:`StructuredInjector` renders a state over those dimensions as an
:class:`~attribution_trials.latent.base.Injection`.
"""

from __future__ import annotations

from dataclasses import dataclass

from attribution_trials.interface import DecisionPoint
from attribution_trials.latent.base import (
    Injection,
    InjectionKind,
    LatentState,
    LatentStateInjector,
    Observation,
)

# Anchored dimensions, in fixed order. The probe_vector follows this order.
DIMENSIONS = ("time_pressure", "post_loss", "fatigue", "momentum")


def history_features(dp: DecisionPoint) -> dict[str, float]:
    """Engineered, *instantaneous* history features for one decision point.

    The single definition of what a model sees about how the session has
    gone so far; the recurrent latent and its memoryless twin both read it.

    Returns one value per anchored dimension (``DIMENSIONS`` order): the first
    three in ``[0, 1]`` (severity), ``momentum`` signed in ``[-1, 1]``.
    """
    ts = dp.time_signal
    stream = dp.recent_outcomes

    # Time pressure: 1 when out of time, 0 when comfortable.
    tp = 0.0
    if ts.time_remaining is not None:
        tp = max(0.0, min(1.0, 1.0 - ts.time_remaining / 30.0))

    # Post-loss: 1 right after a loss, decaying with games since.
    pl = 0.0
    for i, o in enumerate(reversed(stream.recent)):
        if o.won is False:
            pl = max(0.0, 1.0 - i / 3.0)
            break

    # Fatigue: ramps with session position.
    fat = max(0.0, min(1.0, stream.session_position / 20.0))

    # Momentum: signed recent win rate centred at 0.
    wr = stream.recent_win_rate(k=5)
    mom = 0.0 if wr is None else (wr - 0.5) * 2.0

    return {
        "time_pressure": tp,
        "post_loss": pl,
        "fatigue": fat,
        "momentum": mom,
    }


@dataclass
class _Z:
    """Payload for the structured latent: one float per anchored dimension."""

    values: dict[str, float]

    def as_vector(self) -> list[float]:
        return [self.values[d] for d in DIMENSIONS]


class StructuredInjector(LatentStateInjector):
    """EMA over the history features; renders a hidden vector or a text note.

    Parameters
    ----------
    kind:
        Which injection channel to produce. ``HIDDEN`` returns the vector;
        ``VERBAL`` returns a short templated note.
    alpha:
        EMA smoothing for within/cross-game updates (0..1, higher = faster).
    """

    def __init__(self, kind: InjectionKind = InjectionKind.VERBAL, alpha: float = 0.4) -> None:
        self.kind = kind
        self.alpha = alpha
        self.produces = (kind,)

    # --- lifecycle ------------------------------------------------------
    def initial_state(self, player_id: str) -> LatentState:
        z = _Z(values={d: 0.0 for d in DIMENSIONS})
        return LatentState(
            payload=z,
            probe_vector=z.as_vector(),
            meta={"player_id": player_id},
        )

    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        z: _Z = state.payload
        if self.kind is InjectionKind.HIDDEN:
            return Injection(kind=self.kind, vector=z.as_vector())
        return Injection(kind=self.kind, text=self._verbalize(z))

    def update(
        self,
        state: LatentState,
        dp: DecisionPoint,
        observed: Observation | None = None,
    ) -> LatentState:
        z: _Z = state.payload
        ind = self._indicators(dp, observed)
        new_vals = {d: (1 - self.alpha) * z.values[d] + self.alpha * ind[d] for d in DIMENSIONS}
        new_z = _Z(values=new_vals)
        return LatentState(
            payload=new_z,
            probe_vector=new_z.as_vector(),
            meta=state.meta,
        )

    # --- internals -----------------------------------------------------
    def _indicators(self, dp: DecisionPoint, observed: Observation | None) -> dict[str, float]:
        """Engineered indicators in [0,1] (or signed for momentum)."""
        return history_features(dp)

    def _verbalize(self, z: _Z) -> str:
        """Render the latent as a compact templated note."""
        parts = []
        v = z.values
        if v["time_pressure"] > 0.5:
            parts.append("under time pressure (rushing, more errors likely)")
        if v["post_loss"] > 0.5:
            parts.append("just lost (may be tilted / over-aggressive)")
        if v["fatigue"] > 0.5:
            parts.append("deep into the session (fatigued, slower decline)")
        if v["momentum"] > 0.4:
            parts.append("on a winning streak (confident)")
        elif v["momentum"] < -0.4:
            parts.append("on a losing streak (shaken)")
        if not parts:
            parts.append("composed, near baseline form")
        return "Current player state: " + "; ".join(parts) + "."
