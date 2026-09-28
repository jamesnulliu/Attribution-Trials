"""Bootstrap confidence intervals over independent units (users).

Significance is assessed by **bootstrapping over users, not over
decisions** -- decisions within a user are correlated. Give it one value per
user (e.g. a per-user gain) and it returns a percentile confidence interval
on the mean plus the fraction of resamples below zero.

numpy-only; deterministic given ``seed``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class BootstrapCI:
    """A bootstrap estimate of a statistic over independent units."""

    point: float  # statistic on the observed sample
    low: float  # lower percentile bound
    high: float  # upper percentile bound
    p_below_zero: float  # fraction of bootstrap statistics < 0
    n_units: int  # number of independent units resampled
    confidence: float  # e.g. 0.95

    @property
    def excludes_zero(self) -> bool:
        """True iff the whole CI is on one side of 0 (a 'significant' sign)."""
        return self.high < 0.0 or self.low > 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the *mean* of ``values``.

    ``values`` is one number per independent unit (e.g. per player). Resamples
    the units with replacement ``n_resamples`` times, takes the mean of each
    resample, and reports the central ``confidence`` percentile interval. Also
    reports ``p_below_zero`` -- the share of resample means below 0.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    # [n_resamples, n] indices, then mean over the unit axis.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        point=point,
        low=float(low),
        high=float(high),
        p_below_zero=float((means < 0).mean()),
        n_units=int(n),
        confidence=confidence,
    )
