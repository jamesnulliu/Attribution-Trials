"""Telescoping gains + paired statistics.

Per user i, with s = held-out per-decision NLL and conditions A0 = none,
A1 = population, A2 = group, A3 = imposter, A4 = target:

    dpop_i      = s_i(A1) - s_i(A0)   population over none
    dgroup_i    = s_i(A2) - s_i(A1)   group over population
    ddistract_i = s_i(A3) - s_i(A2)   imposter over group
    dindiv_i    = s_i(A4) - s_i(A3)   target over imposter
    total_i     = s_i(A4) - s_i(A0)   target over none

Negative values favor the richer condition; the paper reports each contrast
with the opposite sign (reference minus target, positive favors the target).
Identity (exact arithmetic): the four steps sum to the total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from attribution_trials.eval.bootstrap import BootstrapCI, bootstrap_ci

ARMS = ("A0", "A1", "A2", "A3", "A4")
COMPONENTS = {
    "dpop": ("A1", "A0"),
    "dgroup": ("A2", "A1"),
    "ddistract": ("A3", "A2"),
    "dindiv": ("A4", "A3"),
    "total": ("A4", "A0"),
}


def signflip_pvalue(diffs: list[float], n_permutations: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(diffs).

    The unit is the user (each flip negates one user's paired delta).
    Add-one correction.
    """
    import numpy as np

    x = np.asarray(diffs, dtype=float)
    if x.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = abs(x.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_permutations, x.size))
    null = np.abs((signs * x).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (1 + n_permutations))


@dataclass
class LadderComponents:
    """Per-channel component estimates over one cohort."""

    channel: str
    per_player: dict[str, list[float]]  # arm -> [n_players] NLL
    player_ids: list[str]
    components: dict[str, list[float]] = field(default_factory=dict)
    cis: dict[str, BootstrapCI] = field(default_factory=dict)
    pvalues: dict[str, float] = field(default_factory=dict)
    identity_max_abs_err: float = 0.0

    def summary(self) -> str:
        lines = [f"[ladder:{self.channel}] n={len(self.player_ids)}"]
        for name in COMPONENTS:
            ci = self.cis[name]
            lines.append(
                f"  {name:9s} {ci.point:+.4f} "
                f"[{ci.low:+.4f},{ci.high:+.4f}] "
                f"p_flip={self.pvalues[name]:.4f}"
            )
        lines.append(f"  identity max|err|={self.identity_max_abs_err:.2e}")
        return "\n".join(lines)


def telescope_components(
    per_player: dict[str, list[float]],
    player_ids: list[str],
    channel: str,
    *,
    bootstrap_n: int = 2000,
    permutations: int = 10000,
    seed: int = 0,
    exclude: set[str] | None = None,
) -> LadderComponents:
    """Components + paired CIs + sign-flip p from per-arm NLL lists.

    `exclude`: user ids dropped from every contrast (singleton groups,
    whose imposter condition equals the target).
    """
    keep = [i for i, p in enumerate(player_ids) if not exclude or p not in exclude]
    ids = [player_ids[i] for i in keep]
    pp = {a: [per_player[a][i] for i in keep] for a in ARMS}
    out = LadderComponents(channel=channel, per_player=pp, player_ids=ids)
    for name, (hi, lo) in COMPONENTS.items():
        diffs = [h - l for h, l in zip(pp[hi], pp[lo])]
        out.components[name] = diffs
        out.cis[name] = bootstrap_ci(diffs, n_resamples=bootstrap_n, seed=seed)
        out.pvalues[name] = signflip_pvalue(diffs, n_permutations=permutations, seed=seed)
    sums = [
        out.components["dpop"][i]
        + out.components["dgroup"][i]
        + out.components["ddistract"][i]
        + out.components["dindiv"][i]
        - out.components["total"][i]
        for i in range(len(ids))
    ]
    out.identity_max_abs_err = max((abs(s) for s in sums), default=0.0)
    return out
