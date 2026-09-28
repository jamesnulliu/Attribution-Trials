"""Latent-state injector interface.

A latent ``z_t`` conditions a backbone on who the user is (static
embedding) or on how the user's history has evolved (recurrent latent). It
reaches the backbone as an :class:`Injection`; the backbones here consume the
*hidden* kind (a vector concatenated to the backbone input).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum

from attribution_trials.interface import DecisionPoint
from attribution_trials.prediction import Prediction


class InjectionKind(str, Enum):
    """How a latent state is delivered to a policy backbone."""

    VERBAL = "verbal"  # natural-language memory spliced into the prompt
    HIDDEN = "hidden"  # vector -> soft prompt / prefix tokens


@dataclass
class Injection:
    """A latent state rendered for one specific backbone.

    Exactly one of ``text`` / ``vector`` is set, matching ``kind``. The
    backbone reads only the field for the kind it supports.
    """

    kind: InjectionKind
    text: str | None = None
    vector: list[float] | None = None


@dataclass
class LatentState:
    """Opaque carrier for z_t, threaded through a trajectory.

    Subclasses stash whatever they need in ``payload``. ``probe_vector`` is
    a fixed-width numeric view of the state.
    """

    payload: object = None
    probe_vector: list[float] | None = None
    # Bookkeeping (e.g. the user id).
    meta: dict[str, object] = field(default_factory=dict)


class LatentStateInjector(abc.ABC):
    """Maintains and injects a per-user latent state.

    Lifecycle, per player trajectory::

        z = injector.initial_state(player_id)
        for dp in trajectory:
            inj = injector.render(z, dp)          # -> Injection for policy
            pred = policy.predict(dp, inj)        # policy consumes injection
            z = injector.update(z, dp, observed)  # advance z_t -> z_{t+1}

    Implementations must be deterministic given their parameters and inputs.
    Training and scoring use the batched ``latent_trajectory`` of the
    concrete injectors; this per-step lifecycle is the single-decision view.
    """

    #: Injection kinds this injector can produce.
    produces: tuple[InjectionKind, ...] = ()

    @abc.abstractmethod
    def initial_state(self, player_id: str) -> LatentState:
        """z_0 for a player, before any moves are observed."""

    @abc.abstractmethod
    def render(self, state: LatentState, dp: DecisionPoint) -> Injection:
        """Render z_t as an :class:`Injection` for the current decision."""

    @abc.abstractmethod
    def update(
        self,
        state: LatentState,
        dp: DecisionPoint,
        observed: Observation | None = None,
    ) -> LatentState:
        """Advance z_t -> z_{t+1} after observing what actually happened.

        ``observed`` is ``None`` at inference time when the true move/timing
        is not yet known; in that case the injector advances on its own
        prediction or on the engine reference, as the subclass sees fit.
        """


@dataclass
class Observation:
    """Ground truth at a decision point, used to advance the latent state.

    Mirrors the prediction targets: which move was actually played and how
    long it took. Carried separately from :class:`DecisionPoint` so the same
    decision point can be used for both prediction (no peeking) and update.
    """

    move: str
    time_spent: float | None = None
    # The policy's own prediction at this step, if we want to advance the
    # latent on prediction error rather than ground truth.
    prediction: Prediction | None = None
