"""Population-coverage metric basket.

Kynkaanniemi precision/recall, per-dimension W1 and JS, plus:

(a) an out-of-support penalty          -> :func:`in_support`
(b) gap detection beyond recall        -> :func:`dispersion` (largest-empty-ball)
(c) a calibrated radius so coverage numbers compare across domains and
    dimensionalities                   -> :func:`coverage_sobol_calibrated`
(d) smoothing hygiene for histogram KL -> :func:`kl_laplace`
plus a convex-hull / generalized-variance spread ratio -> :func:`hull_or_spread`
and a joint report with an aggregate null-standardized z -> :func:`basket`.

All functions operate on two point sets, ``R`` (real per-user profile vectors)
and ``G`` (generated per-user profile vectors), shape ``(n, d)`` (1-D inputs
are treated as ``(n, 1)``).  Everything is k-NN / quasi-Monte-Carlo based,
dimension-agnostic, and sane for d <= ~64 after the optional PCA projection
(``pca_dim=`` in :func:`basket`; auto-applied above 64 dims).

Each metric alone is gameable (hull by outliers, recall by spraying wide,
in-support/precision by a point mass at the population mean, average-distance
scores by holes), so the basket is reported jointly.

Dependencies: numpy + scipy.  CPU-only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Small linear-algebra / sampling helpers
# ----------------------------------------------------------------------------


def _as2d(x) -> np.ndarray:
    arr = np.asarray(x, float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.reshape(len(arr), -1)


def _cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix [len(a), len(b)] (fine for n <= few 1000)."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def _knn_radii(points: np.ndarray, k: int) -> np.ndarray:
    """k-th nearest-neighbour distance of each point within its own set."""
    n = len(points)
    d = _cdist(points, points)
    d[np.arange(n), np.arange(n)] = np.inf
    kk = min(k, n - 1)
    return np.sort(d, axis=1)[:, kk - 1]


def _kth_cross_dist(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point of `a` to its k-th nearest point in `b`."""
    kk = min(k, len(b))
    d = _cdist(a, b)
    return np.sort(d, axis=1)[:, kk - 1]


# ----------------------------------------------------------------------------
# Precision/recall, W1, JS
# ----------------------------------------------------------------------------


def wasserstein_1d(a, b):
    """1-D W1 via a common quantile grid."""
    sa = np.sort(np.asarray(a, float))
    sb = np.sort(np.asarray(b, float))
    n = max(len(sa), len(sb))
    q = (np.arange(n) + 0.5) / n
    ia = np.interp(q, (np.arange(len(sa)) + 0.5) / len(sa), sa)
    ib = np.interp(q, (np.arange(len(sb)) + 0.5) / len(sb), sb)
    return float(np.abs(ia - ib).mean())


def js_divergence(a, b, *, bins=12):
    """JS divergence (log base 2) of two 1-D samples, histogram estimate."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


@dataclass
class _PR:
    precision: float
    recall: float
    k: int


def precision_recall(real, generated, *, k=3):
    """Kynkaanniemi 2019 precision/recall, any dimension."""
    r = _as2d(real)
    g = _as2d(generated)
    if len(r) < 2 or len(g) < 2:
        return _PR(0.0, 0.0, k)
    r_rad = _knn_radii(r, k)
    g_rad = _knn_radii(g, k)
    cross = _cdist(g, r)
    precision = float((cross <= r_rad[None, :]).any(axis=1).mean())
    recall = float((cross.T <= g_rad[None, :]).any(axis=1).mean())
    return _PR(precision, recall, k)


# ----------------------------------------------------------------------------
# PCA and Sobol helpers
# ----------------------------------------------------------------------------


def pca_fit(R: np.ndarray, d_max: int | None = None, tol: float = 1e-9):
    """PCA fitted on R only. Returns (mean, components[d_kept, d]).

    Components with eigenvalue <= tol * max_eigenvalue are dropped, so
    rank-deficient profiles (e.g. simplex-valued bucket distributions)
    project to a full-dimensional subspace where ConvexHull is well-posed.
    """
    R = _as2d(R)
    mu = R.mean(axis=0)
    cov = np.cov((R - mu).T) if R.shape[1] > 1 else np.array([[np.var(R)]])
    cov = np.atleast_2d(cov)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    keep = evals > tol * max(evals[0], 1e-30)
    comps = evecs[:, keep].T
    if d_max is not None:
        comps = comps[:d_max]
    return mu, comps


def pca_project(R: np.ndarray, G: np.ndarray, d_max: int = 64, tol: float = 1e-9):
    """Project R and G onto R's PCA basis (fit on R only). Returns (Rp, Gp)."""
    mu, comps = pca_fit(R, d_max=d_max, tol=tol)
    return (_as2d(R) - mu) @ comps.T, (_as2d(G) - mu) @ comps.T


def _make_sobol(d: int, seed):
    """scipy version-tolerant Sobol constructor (rng= vs seed= keyword)."""
    from scipy.stats import qmc

    try:
        return qmc.Sobol(d, scramble=True, rng=seed)
    except TypeError:  # older scipy
        return qmc.Sobol(d, scramble=True, seed=seed)


def sobol_box_sample(R: np.ndarray, m: int, seed) -> np.ndarray:
    """m scrambled-Sobol points quasi-uniform over R's axis-aligned bbox."""
    R = _as2d(R)
    lo, hi = R.min(axis=0), R.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    sob = _make_sobol(R.shape[1], seed)
    with warnings.catch_warnings():
        # Sobol balance warning for non-power-of-2 m is irrelevant here: only
        # a quasi-uniform reference cloud is needed, not integration balance.
        warnings.simplefilter("ignore")
        u = sob.random(m)
    return lo + u * span


# ----------------------------------------------------------------------------
# 1. In-support rate (out-of-support penalty)
# ----------------------------------------------------------------------------


def in_support(G, R, k: int = 3) -> float:
    """Fraction of generated points inside the k-NN support of the real set.

    A generated point g is in-support if it lies within the k-NN radius of at
    least one real point (radii computed within R).  ``1 - in_support`` is the
    out-of-support rate.  Gameable alone: a point mass at the population mean
    scores 1.0.
    """
    Ga, Ra = _as2d(G), _as2d(R)
    if len(Ra) < 2 or len(Ga) < 1:
        return float("nan")
    radii = _knn_radii(Ra, k)
    cross = _cdist(Ga, Ra)
    return float((cross <= radii[None, :]).any(axis=1).mean())


# ----------------------------------------------------------------------------
# 2. Sobol-calibrated coverage (comparable-across-domains recall)
# ----------------------------------------------------------------------------


def calibrate_sobol_radius(
    R,
    m: int,
    k: int = 3,
    n_reps: int = 1000,
    target: float = 0.99,
    seed: int = 0,
) -> float:
    """Radius rho such that an m-point quasi-uniform reference covers R.

    For each of ``n_reps`` scrambled-Sobol draws S of size m over R's bounding
    box, compute each real point's distance to its k-th nearest S point and
    take the ``target`` quantile; rho is the mean over draws.
    """
    Ra = _as2d(R)
    rhos = np.empty(n_reps)
    for i in range(n_reps):
        S = sobol_box_sample(Ra, m, seed=seed + i)
        dk = _kth_cross_dist(Ra, S, k)
        rhos[i] = np.quantile(dk, target)
    return float(rhos.mean())


def coverage_sobol_calibrated(
    R,
    G,
    k: int = 3,
    n_reps: int = 1000,
    target: float = 0.99,
    seed: int = 0,
    rho: float | None = None,
) -> float:
    """Kynkaanniemi-style recall at a Sobol-calibrated global radius.

    A real point is covered if its k-th nearest *generated* point lies within
    rho, where rho comes from :func:`calibrate_sobol_radius`.  Pass a
    precomputed ``rho`` to amortize calibration (it depends only on R, len(G),
    k, target -- not on G).
    """
    Ra, Ga = _as2d(R), _as2d(G)
    if rho is None:
        rho = calibrate_sobol_radius(Ra, len(Ga), k=k, n_reps=n_reps, target=target, seed=seed)
    dk = _kth_cross_dist(Ra, Ga, k)
    return float((dk <= rho).mean())


# ----------------------------------------------------------------------------
# 3. Dispersion (largest-empty-ball proxy; hole detector)
# ----------------------------------------------------------------------------


def dispersion(
    R,
    G,
    n_probe: int = 2048,
    seed: int = 0,
    probes: np.ndarray | None = None,
    disp_R: float | None = None,
) -> float:
    """Largest-empty-ball ratio: how much bigger are G's holes than R's own.

    Over a Sobol probe set spanning R's bounding box, take the maximum
    distance-to-nearest-G and normalize by the identical statistic for R.
    ``probes`` / ``disp_R`` can be precomputed and passed in.
    """
    Ra, Ga = _as2d(R), _as2d(G)
    if probes is None:
        probes = sobol_box_sample(Ra, n_probe, seed=seed)
    d_g = _cdist(probes, Ga).min(axis=1)
    if disp_R is None:
        disp_R = float(_cdist(probes, Ra).min(axis=1).max())
    return float(d_g.max() / max(disp_R, 1e-12))


# ----------------------------------------------------------------------------
# 4. Hull-volume or generalized-variance spread ratio
# ----------------------------------------------------------------------------


def hull_or_spread(G, R, hull_max_dim: int = 8, tol: float = 1e-9) -> dict:
    """Spread ratio G/R: convex-hull volume when feasible, else gen. variance.

    Both sets are projected onto R's PCA basis (rank-truncated).  If the
    projected dimension is <= ``hull_max_dim`` and both sets have enough
    points, returns Vol(hull(G)) / Vol(hull(R)) via scipy.spatial.ConvexHull
    (QJ joggle).  Otherwise falls back to the generalized-variance ratio
    sqrt(det cov(G)) / sqrt(det cov(R)) in the projected space.
    """
    Rp, Gp = pca_project(R, G, d_max=hull_max_dim, tol=tol)
    d = Rp.shape[1]
    use_hull = d <= hull_max_dim and len(Rp) > d + 1 and len(Gp) > d and d >= 1

    if use_hull and d >= 2:
        from scipy.spatial import ConvexHull, QhullError

        try:
            vol_r = ConvexHull(Rp, qhull_options="QJ").volume
        except QhullError:
            use_hull = False
        if use_hull:
            try:
                vol_g = ConvexHull(Gp, qhull_options="QJ").volume
            except QhullError:  # degenerate G (e.g. point mass)
                vol_g = 0.0
            return {
                "ratio": float(vol_g / max(vol_r, 1e-30)),
                "method": "convex_hull",
            }
    if use_hull and d == 1:  # 1-D "hull" = range
        vol_r = float(Rp.max() - Rp.min())
        vol_g = float(Gp.max() - Gp.min())
        return {"ratio": vol_g / max(vol_r, 1e-30), "method": "convex_hull"}

    def _sqrt_det_cov(X):
        c = np.atleast_2d(np.cov(X.T)) if X.shape[1] > 1 else np.array([[np.var(X)]])
        det = float(np.linalg.det(c))
        return np.sqrt(max(det, 0.0))

    sr = _sqrt_det_cov(Rp)
    sg = _sqrt_det_cov(Gp)
    return {
        "ratio": float(sg / max(sr, 1e-30)),
        "method": "generalized_variance",
    }


# ----------------------------------------------------------------------------
# 5. Laplace-smoothed histogram KL
# ----------------------------------------------------------------------------


def kl_laplace(P, Q, bins: int = 10, alpha: float | None = None) -> float:
    """Histogram KL(P||Q) with Laplace smoothing.

    Shared bin edges span the pooled range per dimension; ``alpha`` (default
    ``1/bins``) is added to every cell of BOTH histograms before normalizing.
    Multivariate inputs (n, d): returns the MEAN of the d per-dimension
    marginal KLs (nats).  Direction: KL(real || generated).
    """
    Pa, Qa = _as2d(P), _as2d(Q)
    if alpha is None:
        alpha = 1.0 / bins
    kls = []
    for j in range(Pa.shape[1]):
        p, q = Pa[:, j], Qa[:, j]
        lo = min(p.min(), q.min())
        hi = max(p.max(), q.max())
        if hi <= lo:
            kls.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        cp, _ = np.histogram(p, bins=edges)
        cq, _ = np.histogram(q, bins=edges)
        pp = (cp + alpha) / (cp.sum() + alpha * bins)
        qq = (cq + alpha) / (cq.sum() + alpha * bins)
        kls.append(float(np.sum(pp * np.log(pp / qq))))
    return float(np.mean(kls))


# ----------------------------------------------------------------------------
# 6. The basket
# ----------------------------------------------------------------------------

#: metric name -> orientation for null-standardized z
#:   "low_bad"  : smaller than the null = worse   (coverage-like)
#:   "high_bad" : larger than the null = worse    (divergence-like)
#:   "two_sided": any deviation from the null = worse (spread ratio)
_ORIENTATION = {
    "in_support": "low_bad",
    "coverage_sobol": "low_bad",
    "precision": "low_bad",
    "recall": "low_bad",
    "dispersion_ratio": "high_bad",
    "hull_ratio": "two_sided",
    "kl_rg": "high_bad",
    "w1_mean": "high_bad",
    "js_mean": "high_bad",
}

_Z_STD_FLOOR = 1e-3  # bounded metrics can have zero null variance
_Z_CAP = 10.0  # keep the aggregate a bounded ranking score


@dataclass
class BasketResult:
    metrics: dict
    z: dict
    aggregate_z: float
    null_mean: dict
    null_std: dict
    hull_method: str
    params: dict = field(default_factory=dict)


def _metric_row(Ra, Ga, k, rho, probes, disp_R, hull_max_dim, kl_bins) -> tuple[dict, str]:
    """All basket metrics for one (R, G) pair with shared precomputes."""
    pr = precision_recall(Ra, Ga, k=k)
    hs = hull_or_spread(Ga, Ra, hull_max_dim=hull_max_dim)
    w1s = [wasserstein_1d(Ra[:, j], Ga[:, j]) for j in range(Ra.shape[1])]
    jss = [js_divergence(Ra[:, j], Ga[:, j]) for j in range(Ra.shape[1])]
    row = {
        "in_support": in_support(Ga, Ra, k=k),
        "coverage_sobol": coverage_sobol_calibrated(Ra, Ga, k=k, rho=rho),
        "precision": pr.precision,
        "recall": pr.recall,
        "dispersion_ratio": dispersion(Ra, Ga, probes=probes, disp_R=disp_R),
        "hull_ratio": hs["ratio"],
        "kl_rg": kl_laplace(Ra, Ga, bins=kl_bins),
        "w1_mean": float(np.mean(w1s)),
        "js_mean": float(np.mean(jss)),
    }
    row["out_of_support"] = 1.0 - row["in_support"]
    return row, hs["method"]


def basket(
    R,
    G,
    k: int = 3,
    n_null: int = 200,
    n_sobol_reps: int = 1000,
    target: float = 0.99,
    n_probe: int = 2048,
    hull_max_dim: int = 8,
    kl_bins: int = 10,
    pca_dim: int | None = None,
    seed: int = 0,
) -> BasketResult:
    """Full coverage basket for a generated population G against real R.

    Every metric above, each with a z-score standardized against a null of
    G = bootstrap-resample-of-R (``n_null`` draws, size len(G)), oriented so
    POSITIVE z = worse than a resample of the real population, and their mean
    as ``aggregate_z`` (std floored at 1e-3, |z| capped at 10, because bounded
    metrics have degenerate bootstrap nulls).

    W1/JS/KL are per-dimension marginal means; joint structure is carried by
    the k-NN metrics.  If d > 64 (or ``pca_dim`` is given) both sets are first
    projected onto R's PCA basis.

    ``params["coverage_saturated"]`` is True when rho / median pairwise
    distance within R >= 1: a |G|-point quasi-uniform reference is then so
    sparse that coverage_sobol saturates for this (n, d).
    """
    Ra, Ga = _as2d(R), _as2d(G)
    if Ra.shape[1] != Ga.shape[1]:
        raise ValueError("R and G must share the feature dimension")
    if pca_dim is not None or Ra.shape[1] > 64:
        Ra, Ga = pca_project(Ra, Ga, d_max=pca_dim or 64)

    rng = np.random.default_rng(seed)

    # Shared precomputes (depend on R / len(G) only -> valid for null draws).
    rho = calibrate_sobol_radius(Ra, len(Ga), k=k, n_reps=n_sobol_reps, target=target, seed=seed)
    iu = np.triu_indices(len(Ra), 1)
    med_pair = float(np.median(_cdist(Ra, Ra)[iu])) if len(Ra) > 1 else 0.0
    rho_rel = rho / max(med_pair, 1e-12)
    probes = sobol_box_sample(Ra, n_probe, seed=seed + 10_000)
    disp_R = float(_cdist(probes, Ra).min(axis=1).max())

    row, hull_method = _metric_row(Ra, Ga, k, rho, probes, disp_R, hull_max_dim, kl_bins)

    # Null: G = bootstrap resample of R.
    null_rows = {m: np.empty(n_null) for m in _ORIENTATION}
    for b in range(n_null):
        idx = rng.integers(0, len(Ra), size=len(Ga))
        nrow, _ = _metric_row(Ra, Ra[idx], k, rho, probes, disp_R, hull_max_dim, kl_bins)
        for m in _ORIENTATION:
            null_rows[m][b] = nrow[m]

    z, null_mean, null_std = {}, {}, {}
    for m, orient in _ORIENTATION.items():
        mu = float(null_rows[m].mean())
        sd = float(null_rows[m].std())
        null_mean[m], null_std[m] = mu, sd
        sd_eff = max(sd, _Z_STD_FLOOR)
        x = row[m]
        if orient == "low_bad":
            zz = (mu - x) / sd_eff
        elif orient == "high_bad":
            zz = (x - mu) / sd_eff
        else:  # two_sided
            zz = abs(x - mu) / sd_eff
        z[m] = float(np.clip(zz, -_Z_CAP, _Z_CAP))
    aggregate_z = float(np.mean(list(z.values())))

    return BasketResult(
        metrics=row,
        z=z,
        aggregate_z=aggregate_z,
        null_mean=null_mean,
        null_std=null_std,
        hull_method=hull_method,
        params={
            "k": k,
            "n_null": n_null,
            "n_sobol_reps": n_sobol_reps,
            "target": target,
            "n_probe": n_probe,
            "hull_max_dim": hull_max_dim,
            "kl_bins": kl_bins,
            "pca_dim": pca_dim,
            "seed": seed,
            "rho_calibrated": rho,
            "rho_rel_spread": rho_rel,
            "coverage_saturated": bool(rho_rel >= 1.0),
            "n_real": int(len(Ra)),
            "n_gen": int(len(Ga)),
            "dim_used": int(Ra.shape[1]),
        },
    )
