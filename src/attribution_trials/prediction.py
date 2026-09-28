"""Prediction objects: what a backbone emits at a decision point.

A backbone returns a full distribution over the legal actions plus a
distribution over think-time or response time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MoveDistribution:
    """A calibrated distribution over the legal moves.

    Stored as a plain dict so it is backend-agnostic. Probabilities are
    normalised on construction over whatever keys are present.
    """

    probs: dict[str, float]

    def __post_init__(self) -> None:
        total = sum(self.probs.values())
        if total <= 0:
            raise ValueError("move distribution has non-positive mass")
        if abs(total - 1.0) > 1e-6:
            self.probs = {m: p / total for m, p in self.probs.items()}

    def prob_of(self, move: str, floor: float = 1e-9) -> float:
        """Probability of ``move``, floored so NLL stays finite."""
        return max(self.probs.get(move, 0.0), floor)

    def logprob_of(self, move: str, floor: float = 1e-9) -> float:
        return math.log(self.prob_of(move, floor))

    def top_k(self, k: int) -> list[tuple[str, float]]:
        return sorted(self.probs.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def argmax(self) -> str:
        return max(self.probs.items(), key=lambda kv: kv[1])[0]


@dataclass
class TimingPrediction:
    """A distribution over think-time for the move (seconds).

    Parameterised as a log-normal (think-time is positive and heavy-tailed).
    ``mu`` and ``sigma`` are the parameters of the underlying normal on
    ``log(seconds)``.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("timing sigma must be positive")

    def logpdf(self, seconds: float) -> float:
        """Log density of an observed think-time (seconds > 0)."""
        if seconds <= 0:
            # Treat non-positive observed times as a tiny epsilon: some
            # servers log 0 for premoves / byo-yomi flooring.
            seconds = 1e-3
        x = math.log(seconds)
        z = (x - self.mu) / self.sigma
        # log N(x; mu, sigma) - log(seconds)  [Jacobian of the log map]
        return -0.5 * z * z - math.log(self.sigma) - 0.5 * math.log(2 * math.pi) - x

    @property
    def median_seconds(self) -> float:
        return math.exp(self.mu)


@dataclass
class Prediction:
    """A policy's full output at one decision point."""

    moves: MoveDistribution
    timing: TimingPrediction | None = None
    # Optional latent vector that produced this prediction.
    latent: list[float] | None = None
    meta: dict[str, object] = field(default_factory=dict)
