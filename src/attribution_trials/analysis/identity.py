"""Target-versus-imposter gain for all 84 combinations, with Holm correction.

Two endpoints per combination, both per-user paired deltas (negative favors
the target):
  identity    A4frozen - A3: the target's profile fixed at the training
              boundary against the imposter's (the gain the paper reports);
  registered  A4 - A3: the target's profile updated with held-out rows
              (the updated-profile ablation).
Static embedding has no updated variant, so both endpoints coincide there.

Each endpoint gets a paired per-user bootstrap 95% interval (2,000 resamples,
seed 0) and a two-sided sign-flip p-value (10,000 permutations, seed 0).
Holm runs over the pooled 84 combinations and within each of the four groups
(chess/KT, OPeRA LoRA and frozen prompt, CoSER/HumanLike, Osim); a Holm
rejection counts only when the point estimate favors the target.

Writes paths.ANALYSIS / "identity.json".
Run:  python -m attribution_trials.analysis.identity
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from attribution_trials import paths
from attribution_trials.bench.telescope import signflip_pvalue
from attribution_trials.eval.bootstrap import bootstrap_ci

BFULL = paths.AUDIT / "chess_kt"
SMALL = paths.AUDIT / "chess_kt"
OPERA_FROZEN = paths.AUDIT / "opera"
OPERA_SFT = paths.AUDIT / "opera"
TRIO = paths.AUDIT / "released"
OSIM = paths.AUDIT / "released"
OUT = paths.ANALYSIS / "identity.json"

SEEDS = (0, 1, 2)
DINDIV = {"dindiv": ("A4", "A3")}


# ---------------------------------------------------------------- loaders


def _by_player(nll, players):
    d = defaultdict(list)
    for p, x in zip(players, nll):
        d[p].append(x)
    return {p: float(np.mean(v)) for p, v in d.items()}


def pool_ladder(files, grade, channel):
    acc, ids = defaultdict(list), None
    for f in files:
        node = json.loads(Path(f).read_text())["grades"][grade][channel]
        if ids is None:
            ids = node["player_ids"]
        assert node["player_ids"] == ids
        for arm, v in node["per_player_nll"].items():
            acc[arm].append(v)
    return {a: np.mean(v, axis=0) for a, v in acc.items()}, ids


def deltas(arms, comps):
    return {n: list(np.asarray(arms[hi]) - np.asarray(arms[lo])) for n, (hi, lo) in comps.items()}


def sft_full_arms(substrate, size):
    arms_seed = defaultdict(lambda: defaultdict(list))
    for s in SEEDS:
        need = {
            c: BFULL / f"sft_{substrate}_{c}_{size}_s{s}.json"
            for c in ("none", "placebo", "group", "self")
        }
        assert all(f.exists() for f in need.values()), (substrate, size, s)
        b = {c: json.loads(f.read_text()) for c, f in need.items()}
        pmap = {
            "A0": _by_player(b["none"]["nll"]["eval"], b["none"]["players"]),
            "A1": _by_player(b["placebo"]["nll"]["eval"], b["placebo"]["players"]),
            "A2": _by_player(b["group"]["nll"]["eval"], b["group"]["players"]),
            "A3": _by_player(b["self"]["nll"]["imposter"], b["self"]["players"]),
            "A4": _by_player(b["self"]["nll"]["self_live"], b["self"]["players"]),
            "A4frozen": _by_player(b["self"]["nll"]["self_frozen"], b["self"]["players"]),
        }
        for arm, m in pmap.items():
            for p, v in m.items():
                arms_seed[arm][p].append(v)
    common = sorted(set.intersection(*[set(arms_seed[a]) for a in arms_seed]))
    return {a: np.array([np.mean(arms_seed[a][p]) for p in common]) for a in arms_seed}


def frozen_file_arms(path):
    d = json.loads(Path(path).read_text())
    pmap = {a: _by_player(d["nll"][a], d["players"]) for a in d["nll"]}
    common = sorted(set.intersection(*[set(pmap[a]) for a in pmap]))
    return {a: np.array([pmap[a][p] for p in common]) for a in pmap}


def opera_sft_arms(channel, config, size):
    # the no-information run does not depend on the profile config
    none_sub = f"opera-{channel}-combined"
    sub = f"opera-{channel}-{config}"
    arms_seed = defaultdict(lambda: defaultdict(list))
    n_seeds = 0
    for s in SEEDS:
        none_f = OPERA_SFT / f"sft_{none_sub}_none_{size}_s{s}.json"
        need = {
            c: OPERA_SFT / f"sft_{sub}_{c}_{size}_s{s}.json" for c in ("placebo", "group", "self")
        }
        if not none_f.exists() or not all(f.exists() for f in need.values()):
            continue
        n_seeds += 1
        bn = json.loads(none_f.read_text())
        b = {c: json.loads(f.read_text()) for c, f in need.items()}
        pmap = {
            "A0": _by_player(bn["nll"]["eval"], bn["players"]),
            "A1": _by_player(b["placebo"]["nll"]["eval"], b["placebo"]["players"]),
            "A2": _by_player(b["group"]["nll"]["eval"], b["group"]["players"]),
            "A3": _by_player(b["self"]["nll"]["imposter"], b["self"]["players"]),
            "A4": _by_player(b["self"]["nll"]["self_live"], b["self"]["players"]),
            "A4frozen": _by_player(b["self"]["nll"]["self_frozen"], b["self"]["players"]),
        }
        for arm, m in pmap.items():
            for p, v in m.items():
                arms_seed[arm][p].append(v)
    assert n_seeds > 0, (channel, config, size)
    common = sorted(set.intersection(*[set(arms_seed[a]) for a in arms_seed]))
    return {a: np.array([np.mean(arms_seed[a][p]) for p in common]) for a in arms_seed}


# --------------------------------------------------------------- assembly


def collect_cells() -> list[dict]:
    """One record per combination.

    `components["dindiv"]` is the updated-profile delta A4 - A3, `like` the
    fixed-profile delta A4frozen - A3 (None where no updated variant exists),
    `drift` the per-user A4frozen - A4.
    """
    cells: list[dict] = []

    def add(source, key, comps, drift, like):
        cells.append(
            {
                "source": source,
                "key": key,
                "components": comps,
                "drift": drift,
                "like": like,
                "identity_only_by_construction": like is None,
            }
        )

    # ---- chess and knowledge tracing ----
    cfiles = [BFULL / f"full_ladder_chess_s{s}.json" for s in SEEDS]
    for ch in ("move", "timing"):
        pooled, _ = pool_ladder(cfiles, "retrained", ch)
        add("full-tier", f"static-embedding|chess-pooled|{ch}", deltas(pooled, DINDIV), None, None)
    for sub, fname, chans in (
        ("kt", "small_ladder_kt_s{}.json", ("move",)),
        ("kt-rt", "small_ladder_kt-rt_s{}.json", ("move", "timing")),
    ):
        files = [SMALL / fname.format(s) for s in SEEDS]
        for ch in chans:
            pooled, _ = pool_ladder(files, "retrained", ch)
            add("full-tier", f"static-embedding|{sub}|{ch}", deltas(pooled, DINDIV), None, None)

    mfiles = [SMALL / f"memory_pilot_kt_s{s}.json" for s in SEEDS]
    pooled, _ = pool_ladder(mfiles, "refit", "response")
    comps = deltas(pooled, DINDIV)
    drift = list(
        np.mean([json.loads(f.read_text())["drift"]["refit"]["per_player"] for f in mfiles], axis=0)
    )
    like = list(np.asarray(comps["dindiv"]) + np.asarray(drift))
    add("full-tier", "structured-memory|kt|response", comps, drift, like)

    for sub, template in (
        ("chess-pooled", BFULL / "full_evolving_chess_s{}.json"),
        ("kt-rt", SMALL / "evolving_kt-rt_s{}.json"),
    ):
        files = [Path(str(template).format(s)) for s in SEEDS]
        blobs = [json.loads(f.read_text()) for f in files]
        for ch in ("move", "timing"):
            pooled, _ = pool_ladder(files, "eval-swap", ch)
            comps = deltas(pooled, DINDIV)
            drift = list(np.mean([b["drift_per_player"][ch] for b in blobs], axis=0))
            like = list(np.asarray(comps["dindiv"]) + np.asarray(drift))
            add("full-tier", f"evolving-latent|{sub}|{ch}", comps, drift, like)

    for sub, ch in (("kt", "response"), ("chess", "timing")):
        for size in ("1.7b", "8b"):
            arms = sft_full_arms(sub, size)
            add(
                "full-tier",
                f"sft-lora|{sub} ({size})|{ch}",
                deltas(arms, DINDIV),
                list(arms["A4frozen"] - arms["A4"]),
                list(arms["A4frozen"] - arms["A3"]),
            )
    for sub, ch in (("kt", "response"), ("chess", "timing")):
        for size in ("1.7b", "8b"):
            arms = frozen_file_arms(BFULL / f"frozen_{sub}_{size}.json")
            add(
                "full-tier",
                f"persona-frozen|{sub} ({size})|{ch}",
                deltas(arms, DINDIV),
                list(arms["A4frozen"] - arms["A4"]),
                list(arms["A4frozen"] - arms["A3"]),
            )

    # ---- OPeRA: user-profile LoRA and frozen prompt ----
    for size in ("1.7b", "8b"):
        for ch in ("action", "timing"):
            for cfg in ("combined", "static", "dynamic"):
                arms = opera_sft_arms(ch, cfg, size)
                add(
                    "opera-gpu",
                    f"sft-lora|{ch}|{cfg}|{size}",
                    deltas(arms, DINDIV),
                    list(arms["A4frozen"] - arms["A4"]),
                    list(arms["A4frozen"] - arms["A3"]),
                )
    for size in ("1.7b", "8b", "30b"):
        fam = "strong-open" if size == "30b" else "persona-frozen"
        for ch in ("action", "timing"):
            for cfg in ("combined", "static", "dynamic"):
                arms = frozen_file_arms(OPERA_FROZEN / f"frozen_opera-{ch}-{cfg}_{size}.json")
                add(
                    "opera-gpu",
                    f"{fam}|{ch}|{cfg}|{size}",
                    deltas(arms, DINDIV),
                    list(arms["A4frozen"] - arms["A4"]),
                    list(arms["A4frozen"] - arms["A3"]),
                )

    # ---- OPeRA: CoSER and HumanLike ----
    for tag, fmt in (("coser-8b", "chat"), ("humanlike-7b", "chat")):
        for ch in ("action", "timing"):
            for cfg in ("combined", "static", "dynamic"):
                arms = frozen_file_arms(TRIO / f"trio-panel_{ch}-{cfg}_{tag}_{fmt}.json")
                add(
                    "sim2real-trio",
                    f"sim2real-trio|{ch}|{cfg}|{tag}",
                    deltas(arms, DINDIV),
                    list(arms["A4frozen"] - arms["A4"]),
                    list(arms["A4frozen"] - arms["A3"]),
                )

    # ---- OPeRA: Osim ----
    for tag in ("osim-4b", "osim-4b-mid", "osim-8b", "osim-8b-mid"):
        for ch in ("action", "timing"):
            for cfg in ("combined", "static", "dynamic"):
                arms = frozen_file_arms(OSIM / f"osim-panel_{ch}-{cfg}_{tag}.json")
                add(
                    "osim-panel",
                    f"trained-sim|{ch}|{cfg}|{tag}",
                    deltas(arms, DINDIV),
                    list(arms["A4frozen"] - arms["A4"]),
                    list(arms["A4frozen"] - arms["A3"]),
                )

    return cells


# ------------------------------------------------------------- inference


def interval(diffs, seed=0):
    """Paired per-user bootstrap 95% interval and sign-flip p-value."""
    ci = bootstrap_ci(list(diffs), n_resamples=2000, seed=seed)
    return {
        "point": float(ci.point),
        "low": float(ci.low),
        "high": float(ci.high),
        "p_flip": float(signflip_pvalue(list(diffs), n_permutations=10000, seed=seed)),
    }


def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    keep = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            keep[idx] = True
        else:
            break
    return keep


def certify(rows, tag):
    """Attach pooled-84 and within-group Holm flags for endpoint `tag` in place."""
    pooled_keep = holm([r[tag]["p_flip"] for r in rows])
    for r, k in zip(rows, pooled_keep):
        r[tag]["holm_pooled"] = bool(k and r[tag]["point"] < 0)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for source, rs in by_source.items():
        keep = holm([r[tag]["p_flip"] for r in rs])
        for r, k in zip(rs, keep):
            r[tag]["holm_within"] = bool(k and r[tag]["point"] < 0)


def counts(rows, tag):
    out = {}
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    for source in ("full-tier", "opera-gpu", "sim2real-trio", "osim-panel"):
        rs = by_source[source]
        out[source] = {
            "cells": len(rs),
            "sign_favorable": sum(r[tag]["point"] < 0 for r in rs),
            "ci_excludes_0_favorable": sum(r[tag]["high"] < 0 for r in rs),
            "holm_pooled_survivors": sum(r[tag]["holm_pooled"] for r in rs),
            "holm_within_survivors": sum(r[tag]["holm_within"] for r in rs),
        }
    out["pooled-84"] = {
        "cells": len(rows),
        "sign_favorable": sum(r[tag]["point"] < 0 for r in rows),
        "ci_excludes_0_favorable": sum(r[tag]["high"] < 0 for r in rows),
        "holm_pooled_survivors": sum(r[tag]["holm_pooled"] for r in rows),
    }
    return out


def main() -> int:
    cells = collect_cells()
    assert len(cells) == 84, len(cells)

    rows = []
    for cell in cells:
        reg = interval(cell["components"]["dindiv"])
        ident = reg if cell["like"] is None else interval(cell["like"])
        row = {
            "source": cell["source"],
            "key": cell["key"],
            "n_users": len(cell["components"]["dindiv"]),
            "identity_only_by_construction": cell["identity_only_by_construction"],
            "registered": dict(reg),
            "identity": dict(ident),
        }
        if cell["drift"] is not None:
            row["drift"] = interval(cell["drift"])
        rows.append(row)

    for tag in ("registered", "identity"):
        certify(rows, tag)

    blob = {
        "definition": (
            "identity = A4frozen - A3 per user (fixed profile); "
            "registered = A4 - A3 (profile updated with held-out "
            "rows); equal for static embedding, which has no "
            "updated variant. Negative favors the target."
        ),
        "counts": {tag: counts(rows, tag) for tag in ("registered", "identity")},
        "survivors": {
            tag: [r["key"] for r in rows if r[tag]["holm_pooled"]]
            for tag in ("registered", "identity")
        },
        "cells": rows,
    }
    paths.ensure(OUT.parent)
    OUT.write_text(json.dumps(blob, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
