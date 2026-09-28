"""Shared decision-point schema.

Chess moves and knowledge-tracing responses both reduce to the same
:class:`DecisionPoint` record, so injectors, backbones and the trainer never
see domain-specific types.

* Pure stdlib (``dataclasses`` + ``typing``).
* ``state`` is an opaque ``object``: a FEN string for chess, a tuple of item
  features for knowledge tracing.
* Actions are plain ``str`` (UCI for chess, ``correct``/``incorrect`` for
  KT). The legal-action set is carried explicitly so prediction heads can
  normalise over exactly the legal actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Game(str, Enum):
    """Which domain a decision point belongs to.

    The schema is domain-agnostic: knowledge tracing reuses the same
    :class:`DecisionPoint` as chess, swapping only the encoder and head.
    The stored value is part of the persisted trajectory format.
    """

    CHESS = "chess"
    KNOWLEDGE_TRACING = "knowledge_tracing"


@dataclass(frozen=True)
class EngineReference:
    """Optional engine view of a position (part of the persisted format).

    The pipeline leaves ``engine_reference=None``; the type is kept so stored
    trajectories round-trip. ``candidate_values`` maps each legal move to the
    engine's value for the mover.
    """

    candidate_values: dict[str, float]
    best_move: str | None = None
    best_value: float | None = None
    unit: str = "centipawn"
    depth: int | None = None


@dataclass(frozen=True)
class TimeSignal:
    """Per-decision timing context.

    All times are in seconds. The byo-yomi fields are part of the persisted
    format and are left ``None``.
    """

    time_remaining: float | None = None
    increment: float | None = None
    time_spent: float | None = None
    move_number: int = 0
    # Byo-yomi periods left + length of each period (seconds).
    byo_yomi_periods_left: int | None = None
    byo_yomi_period_length: float | None = None
    # Coarse game phase tag ("opening" / "middlegame" / "endgame" / ...),
    # filled by the game encoder when cheap to compute.
    phase: str | None = None

    @property
    def in_time_trouble(self) -> bool:
        """True when remaining time is known and at most 10 seconds."""
        return self.time_remaining is not None and self.time_remaining <= 10.0


@dataclass(frozen=True)
class Outcome:
    """One completed game in a player's recent stream."""

    won: bool | None
    # Result margin in game units (centipawns / score). Sign is from the
    # tracked player's perspective; ``None`` if unknown.
    margin: float | None = None
    # Engine-scored swing magnitude over the game (volatility proxy).
    engine_swing: float | None = None
    blunders: int = 0
    time_scramble: bool = False
    # Seconds between the end of this game and the start of the next, used
    # to segment sessions. ``None`` for the most recent game.
    gap_to_next_seconds: float | None = None


@dataclass
class OutcomeStream:
    """Running stream of recent results, within and across a session.

    Ordered oldest -> newest. The history features
    (:func:`~attribution_trials.latent.structured.history_features`) read it.
    """

    recent: list[Outcome] = field(default_factory=list)
    # Index of the current game within its session (0 == first game). This
    # is a *derived* quantity; see ``attribution_trials.data.sessions`` for segmentation.
    session_position: int = 0

    def last(self) -> Outcome | None:
        return self.recent[-1] if self.recent else None

    def recent_win_rate(self, k: int = 5) -> float | None:
        """Win rate over the last ``k`` decided games."""
        decided = [o.won for o in self.recent[-k:] if o.won is not None]
        if not decided:
            return None
        return sum(decided) / len(decided)


@dataclass(frozen=True)
class DecisionPoint:
    """One decision (a chess move or a KT response), domain-agnostic.

    Given this user, this state, the timing context, and how the session has
    gone so far -- what does *this* user do, and how long do they take?
    """

    game: Game
    player_id: str
    state: object  # opaque board encoding; game-specific (see module doc)
    legal_actions: tuple[str, ...]
    engine_reference: EngineReference | None
    time_signal: TimeSignal
    recent_outcomes: OutcomeStream
    # Free-form game context: time_control, rating_gap, color, etc. Kept as
    # a dict so games can attach extras without widening the schema.
    context: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.legal_actions:
            raise ValueError("DecisionPoint must have at least one legal action")
