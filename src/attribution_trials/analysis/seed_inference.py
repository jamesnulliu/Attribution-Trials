"""Training-seed variability in the target-versus-imposter intervals.

The estimator is
    Delta_hat = (1/U) sum_u (1/S) sum_s d_{u,s}
-- each user's paired delta is averaged over the S = 3 training seeds first,
then the bootstrap resamples users. That interval conditions on the realized
training runs. For every combination with at least two seeds this adds:

  user-only     resample users over the seed-averaged deltas.
  hierarchical  resample seeds with replacement AND users with replacement
                in each replicate.
  LOSO          leave-one-seed-out point estimates: the spread across the
                three 2-seed estimators.

Both endpoints of analysis.identity are analysed: `registered` (A4 - A3,
profile updated with held-out rows) and `identity` (A4frozen - A3, fixed
profile). With S = 3 the seed resample is coarse, so the hierarchical interval
is an inflation bound beside the user-only one, not a replacement.

Combinations: the 14 trained chess/KT combinations (static embedding,
structured memory, recurrent embedding, user-profile LoRA) and the OPeRA
user-profile LoRA combinations with three seeds; single-seed OPeRA
combinations are listed as out of scope.

Needs paths.ANALYSIS / "identity.json" (pooled-Holm survivors).
Writes paths.ANALYSIS / "seed_inference.json".
Run:  python -m attribution_trials.analysis.seed_inference
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from attribution_trials import paths

IDENTITY = paths.ANALYSIS / "identity.json"
OUT = paths.ANALYSIS / "seed_inference.json"
BFULL = paths.AUDIT / "chess_kt"
SMALL = paths.AUDIT / "chess_kt"
OPERA_SFT = paths.AUDIT / "opera"

SEEDS = (0, 1, 2)
NBOOT = 10000
RNG_SEED = 0


# ------------------------------------------------- per-seed delta loaders


def ladder_per_seed(files, grade, channel, drift_key=None):
    """-> dict endpoint -> (U, S) matrix of per-user deltas."""
    dindiv, like = [], []
    ids = None
    for f in files:
        blob = json.loads(Path(f).read_text())
        node = blob["grades"][grade][channel]
        if ids is None:
            ids = node["player_ids"]
        assert node["player_ids"] == ids
        d = np.asarray(node["per_player_nll"]["A4"]) - np.asarray(node["per_player_nll"]["A3"])
        dindiv.append(d)
        if drift_key == "drift_per_player":
            like.append(d + np.asarray(blob["drift_per_player"][channel]))
        elif drift_key == "memory":
            like.append(d + np.asarray(blob["drift"]["refit"]["per_player"]))
        else:
            like.append(d)
    return {"registered": np.column_stack(dindiv), "identity": np.column_stack(like)}


def sft_per_seed(files):
    """`self`-condition score files, one per seed -> per-user delta matrix."""
    dindiv, like = [], []
    ids = None
    for f in files:
        blob = json.loads(Path(f).read_text())
        per = defaultdict(lambda: defaultdict(list))
        for key in ("imposter", "self_live", "self_frozen"):
            for p, x in zip(blob["players"], blob["nll"][key]):
                per[key][p].append(x)
        players = sorted(per["imposter"])
        if ids is None:
            ids = players
        assert players == ids
        a3 = np.array([np.mean(per["imposter"][p]) for p in ids])
        a4 = np.array([np.mean(per["self_live"][p]) for p in ids])
        a4f = np.array([np.mean(per["self_frozen"][p]) for p in ids])
        dindiv.append(a4 - a3)
        like.append(a4f - a3)
    return {"registered": np.column_stack(dindiv), "identity": np.column_stack(like)}


def collect() -> tuple[list[dict], list[str]]:
    cells = []

    def add(key, mats):
        cells.append({"key": key, "mats": mats})

    cfiles = [BFULL / f"full_ladder_chess_s{s}.json" for s in SEEDS]
    for ch in ("move", "timing"):
        add(f"static-embedding|chess-pooled|{ch}", ladder_per_seed(cfiles, "retrained", ch))
    for sub, fname, chans in (
        ("kt", "small_ladder_kt_s{}.json", ("move",)),
        ("kt-rt", "small_ladder_kt-rt_s{}.json", ("move", "timing")),
    ):
        files = [SMALL / fname.format(s) for s in SEEDS]
        for ch in chans:
            add(f"static-embedding|{sub}|{ch}", ladder_per_seed(files, "retrained", ch))
    add(
        "structured-memory|kt|response",
        ladder_per_seed(
            [SMALL / f"memory_pilot_kt_s{s}.json" for s in SEEDS],
            "refit",
            "response",
            drift_key="memory",
        ),
    )
    for sub, template in (
        ("chess-pooled", BFULL / "full_evolving_chess_s{}.json"),
        ("kt-rt", SMALL / "evolving_kt-rt_s{}.json"),
    ):
        files = [Path(str(template).format(s)) for s in SEEDS]
        for ch in ("move", "timing"):
            add(
                f"evolving-latent|{sub}|{ch}",
                ladder_per_seed(files, "eval-swap", ch, drift_key="drift_per_player"),
            )
    for sub, ch in (("kt", "response"), ("chess", "timing")):
        for size in ("1.7b", "8b"):
            add(
                f"sft-lora|{sub} ({size})|{ch}",
                sft_per_seed([BFULL / f"sft_{sub}_self_{size}_s{s}.json" for s in SEEDS]),
            )

    skipped = []
    for size in ("1.7b", "8b"):
        for ch in ("action", "timing"):
            for cfg in ("combined", "static", "dynamic"):
                files = []
                for s in SEEDS:
                    f = OPERA_SFT / f"sft_opera-{ch}-{cfg}_self_{size}_s{s}.json"
                    if f.exists():
                        files.append(f)
                key = f"sft-lora|{ch}|{cfg}|{size}"
                if len(files) < 2:
                    skipped.append(key)
                    continue
                add(key, sft_per_seed(files))
    return cells, skipped


# ------------------------------------------------------------- inference


def analyse(mat: np.ndarray) -> dict:
    """(U, S) per-user per-seed deltas -> user-only, hierarchical and LOSO."""
    U, S = mat.shape
    user_means = mat.mean(axis=1)
    point = float(user_means.mean())

    rng = np.random.default_rng(RNG_SEED)
    user_only = np.empty(NBOOT)
    joint = np.empty(NBOOT)
    for b in range(NBOOT):
        u = rng.integers(0, U, U)
        s = rng.integers(0, S, S)
        user_only[b] = user_means[u].mean()
        joint[b] = mat[np.ix_(u, s)].mean()

    def ci(draws):
        return {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
            "width": float(np.quantile(draws, 0.975) - np.quantile(draws, 0.025)),
        }

    out = {
        "point": point,
        "n_users": U,
        "n_seeds": S,
        "user_only": ci(user_only),
        "hierarchical": ci(joint),
    }
    out["width_ratio"] = out["hierarchical"]["width"] / out["user_only"]["width"]
    out["loso_points"] = [float(np.delete(mat, s, axis=1).mean()) for s in range(S)]
    out["loso_range"] = float(max(out["loso_points"]) - min(out["loso_points"]))
    out["hier_excludes_0"] = out["hierarchical"]["high"] < 0 or out["hierarchical"]["low"] > 0
    return out


def main() -> int:
    identity = json.loads(IDENTITY.read_text())
    survivors = {tag: set(identity["survivors"][tag]) for tag in ("registered", "identity")}

    cells, skipped = collect()
    rows = []
    for cell in cells:
        row = {"key": cell["key"]}
        for tag in ("registered", "identity"):
            res = analyse(cell["mats"][tag])
            res["pooled_holm_survivor"] = cell["key"] in survivors[tag]
            row[tag] = res
        rows.append(row)

    flips = {
        tag: [
            r["key"]
            for r in rows
            if r[tag]["pooled_holm_survivor"] and not r[tag]["hier_excludes_0"]
        ]
        for tag in ("registered", "identity")
    }
    surv_ratios = {
        tag: [r[tag]["width_ratio"] for r in rows if r[tag]["pooled_holm_survivor"]]
        for tag in ("registered", "identity")
    }

    blob = {
        "estimator": (
            "Delta_hat = (1/U) sum_u (1/S) sum_s d_us; the "
            "user-only interval resamples users over the "
            "seed-averaged d_u; the hierarchical variant resamples "
            "seeds with replacement and users with replacement in "
            "each replicate"
        ),
        "n_boot": NBOOT,
        "rng_seed": RNG_SEED,
        "cells_out_of_scope_single_seed": skipped,
        "survivor_width_ratio_range": {
            tag: ([float(min(v)), float(max(v))] if v else None) for tag, v in surv_ratios.items()
        },
        "survivors_losing_hierarchical_resolution": flips,
        "cells": rows,
    }
    paths.ensure(OUT.parent)
    OUT.write_text(json.dumps(blob, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
