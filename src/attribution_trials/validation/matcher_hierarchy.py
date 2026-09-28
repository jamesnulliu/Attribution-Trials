"""Every combination under three imposter matchers.

Matchers:
  registered   the within-cell tertile derangement of the audit
  NN           nearest other user in the registered cell, z-scored Euclidean
               distance over training-split features
  hardest      the best-fitting donor per user: an adversarial bound, not an
               estimand

The gain holds the profile fixed at the training boundary (A4frozen - A3_M;
for static-embedding combinations the two coincide).  Values are target minus
imposter NLL, so negative favours the target.  Combinations: the six
significant after pooled Holm correction plus the recurrent-embedding KT-RT
timing combination and the two strongest prompting combinations (user-profile
LoRA 8B and frozen prompt 8B on KT response).

Inputs: <ANALYSIS>/identity.json, <VALIDATION>/nearest_imposter.json,
<VALIDATION>/hardest_imposter.json, <VALIDATION>/matchers/*,
<AUDIT>/chess_kt/frozen_kt_8b.json, <DATA>/kt/prepared/assist09.tsv.

  python -m attribution_trials.validation.matcher_hierarchy
Output: <VALIDATION>/matcher_hierarchy.json
"""

from __future__ import annotations

import json
import math

import numpy as np

from attribution_trials import paths
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import session_split_indices

IDENTITY = paths.ANALYSIS / "identity.json"
NEAREST = paths.VALIDATION / "nearest_imposter.json"
HARDEST = paths.VALIDATION / "hardest_imposter.json"
MATCHERS = paths.VALIDATION / "matchers"
FROZEN_PANEL = paths.AUDIT / "chess_kt" / "frozen_kt_8b.json"
KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"

SEEDS = (0, 1, 2)
SEED = 0
NBOOT = 10000
TRAIN_FRAC = 0.7

STATIC = (
    "static-embedding|chess-pooled|move",
    "static-embedding|chess-pooled|timing",
    "static-embedding|kt|move",
    "static-embedding|kt-rt|move",
    "static-embedding|kt-rt|timing",
)


def boot(values) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(NBOOT)]
    return {
        "point": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def ktrt_features() -> dict:
    """KT-RT feature space: [log n, train acc, mean/sd log RT]."""
    data = load_kt_csv(str(KT_DATA), n_students=500, min_responses=50, response_time_col=5)
    splits = session_split_indices(data.trajectories, TRAIN_FRAC)
    out = {}
    for traj, cut in zip(data.trajectories, splits):
        correct = [1.0 if o.move == "correct" else 0.0 for o in traj.observations[:cut]]
        if not correct:
            continue
        logt = np.array([math.log(max(float(o.time_spent), 1e-3)) for o in traj.observations[:cut]])
        out[traj.player_id] = np.array(
            [math.log(len(correct)), float(np.mean(correct)), float(logt.mean()), float(logt.std())]
        )
    return out


def nn_from_features(features, players, cell_of) -> dict:
    matrix = np.array([features[p] for p in players], dtype=float)
    centre, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (matrix - centre) / scale
    by_cell: dict[str, list[int]] = {}
    for i, p in enumerate(players):
        by_cell.setdefault(cell_of[p], []).append(i)
    donor = {}
    for members in by_cell.values():
        block = z[members]
        dist = np.sqrt(((block[:, None, :] - block[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        for pos, i in enumerate(members):
            donor[players[i]] = players[members[int(dist[pos].argmin())]]
    return donor


def per_player(values, players) -> dict:
    acc: dict[str, list] = {}
    for p, x in zip(players, values):
        acc.setdefault(str(p), []).append(float(x))
    return {p: float(np.mean(v)) for p, v in acc.items()}


def main() -> int:
    identity = json.loads(IDENTITY.read_text())
    cells = {r["key"]: r for r in identity["cells"]}
    certified = set(identity["survivors"]["identity"])
    nearest = {r["cell"]: r for r in json.loads(NEAREST.read_text())["cells"]}
    hardest = {r["cell"]: r for r in json.loads(HARDEST.read_text())["cells"]}
    rows = []

    # ---- static embedding: lookups in the cross-matrices ------------------
    for key in STATIC:
        rows.append(
            {
                "cell": key,
                "certified_identity": key in certified,
                "registered": cells[key]["identity"],
                "nn": nearest[key]["dindiv_nn"],
                "hardest": hardest[key]["hardest"],
            }
        )

    # ---- recurrent embedding, KT-RT: the cross-matrix ---------------------
    feats = ktrt_features()
    for channel, mkey in (("response", "M_response"), ("timing", "M_timing")):
        key = (
            "evolving-latent|kt-rt|move"
            if channel == "response"
            else "evolving-latent|kt-rt|timing"
        )
        like = {"registered": [], "nn": [], "hardest": []}
        for s in SEEDS:
            z = np.load(MATCHERS / f"cross_evolving_kt-rt_s{s}.npz", allow_pickle=True)
            players = [str(p) for p in z["players"]]
            cell_of = {p: str(c) for p, c in zip(players, z["cell_of"])}
            singles = {str(p) for p in z["singletons"]}
            keep = [i for i, p in enumerate(players) if p not in singles]
            M = z[mkey]
            diag = M[np.arange(len(players)), np.arange(len(players))]
            a3_reg = z[f"a3_reg_{channel}"]
            donor_nn = nn_from_features(feats, players, cell_of)
            nn_idx = np.array([players.index(donor_nn[p]) for p in players])
            hardest_m = np.nanmin(np.where(np.eye(len(players), dtype=bool), np.nan, M), axis=1)
            like["registered"].append((diag - a3_reg)[keep])
            like["nn"].append((diag - M[np.arange(len(players)), nn_idx])[keep])
            like["hardest"].append((diag - hardest_m)[keep])
        stats = {k: boot(np.mean(v, axis=0)) for k, v in like.items()}
        rows.append(
            {
                "cell": key,
                "certified_identity": key in certified,
                "registered": stats["registered"],
                "nn": stats["nn"],
                "hardest": stats["hardest"],
            }
        )

    # ---- user-profile LoRA 8B, KT: NN pass on the saved adapters ----------
    per = {"registered": {}, "nn": {}}
    for s in SEEDS:
        blob = json.loads((MATCHERS / f"lora_nn_s{s}.json").read_text())
        players = blob["row_players"]
        a4f = per_player(blob["nll"]["a4frozen"], players)
        a3r = per_player(blob["nll"]["a3_registered"], players)
        a3n = per_player(blob["nll"]["a3_nn"], players)
        for p in a4f:
            per["registered"].setdefault(p, []).append(a4f[p] - a3r[p])
            per["nn"].setdefault(p, []).append(a4f[p] - a3n[p])
    common = sorted(per["registered"])
    reg = boot([np.mean(per["registered"][p]) for p in common])
    nn = boot([np.mean(per["nn"][p]) for p in common])
    rows.append(
        {
            "cell": "sft-lora|kt (8b)|response",
            "certified_identity": False,
            "registered": reg,
            "nn": nn,
            "hardest": None,
        }
    )

    # ---- frozen prompt 8B, KT: NN pass ------------------------------------
    blob = json.loads((MATCHERS / "frozen_nn_8b.json").read_text())
    panel = json.loads(FROZEN_PANEL.read_text())
    players = [str(p) for p in panel["players"]]
    a4f = per_player(panel["nll"]["A4frozen"], players)
    a3r = per_player(panel["nll"]["A3"], players)
    a3n = per_player(blob["nll"]["a3_nn"], blob["row_players"])
    common = sorted(set(a4f) & set(a3n))
    reg = boot([a4f[p] - a3r[p] for p in common])
    nn = boot([a4f[p] - a3n[p] for p in common])
    rows.append(
        {
            "cell": "persona-frozen|kt (8b)|response",
            "certified_identity": False,
            "registered": reg,
            "nn": nn,
            "hardest": None,
        }
    )

    favorable = lambda s: bool(s and s["high"] < 0)  # noqa: E731
    for r in rows:
        r["k_of_matchers"] = sum(
            favorable(r[m]) for m in ("registered", "nn", "hardest") if r[m] is not None
        )
        r["matchers_reported"] = sum(r[m] is not None for m in ("registered", "nn", "hardest"))

    out = paths.ensure(paths.VALIDATION) / "matcher_hierarchy.json"
    out.write_text(
        json.dumps({"endpoint": "fixed profile (A4frozen - A3_M)", "rows": rows}, indent=1)
    )

    def fmt(s):
        if s is None:
            return "---"
        return f"{s['point']:+.4f} [{s['low']:+.4f},{s['high']:+.4f}]"

    for r in rows:
        print(
            f"{r['cell']:40s} {fmt(r['registered'])}  {fmt(r['nn'])}  "
            f"{fmt(r['hardest'])}  {r['k_of_matchers']}/{r['matchers_reported']}"
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
