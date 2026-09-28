"""Policy backbone interface.

A backbone turns ``(decision point, latent injection)`` into a
:class:`~attribution_trials.prediction.Prediction` and declares which
:class:`~attribution_trials.latent.base.InjectionKind` s it can consume.
"""

from __future__ import annotations

import abc

from attribution_trials.interface import DecisionPoint
from attribution_trials.latent.base import Injection, InjectionKind
from attribution_trials.prediction import Prediction


class PolicyBackbone(abc.ABC):
    """Emits a move+timing prediction, optionally conditioned on a latent.

    Subclasses must set :attr:`accepts` to the injection kinds they can
    consume.
    """

    #: Latent injection kinds this backbone can consume.
    accepts: tuple[InjectionKind, ...] = ()

    @abc.abstractmethod
    def predict(
        self,
        dp: DecisionPoint,
        injection: Injection | None = None,
    ) -> Prediction:
        """Predict the move (and timing) distribution at ``dp``.

        Implementations must normalise the move distribution over
        ``dp.legal_actions`` only. ``injection`` is ``None`` when no latent
        is in use or when this backbone advertises no accepted kinds.
        """

    def accepts_kind(self, kind: InjectionKind) -> bool:
        return kind in self.accepts

    @property
    def name(self) -> str:
        return type(self).__name__

    def close(self) -> None:  # noqa: B027  (intentional optional no-op hook)
        """Release resources. No-op by default."""
