"""Judged fidelity as a simulator selector, and the judge's score under the imposter's card.

Reads ``judge_scores.json`` (one record per candidate/arm/key with the fidelity
index and the alignment score) and applies the R2 rule the other selectors
use: a pair of simulators is resolvable when the paired-user bootstrap
interval of its content-gain difference excludes zero; a selector agrees when
it orders the pair the same way.  Selector: judged fidelity, the per-user mean
fidelity index under the target card (A4), scored as -value so lower = better.

Imposter column: for both judge scores, the per-user mean under the
imposter's card (A3) against A4, the paired-user bootstrap of the difference,
and the share kept.  Writes ``paths.WILDCHAT / "judge_readout.json"``.

  python -m attribution_trials.wildchat.judge_readout
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict

import numpy as np

from attribution_trials import paths

RES = paths.WILDCHAT
SEED, NBOOT, METRIC = 0, 10000, "M1_minilm_max_card_bullet"
CANDS = ("1.7b", "8b", "dense", "humanlike", "coser", "osim", "humanlm")
ENDPOINTS = {"avg@8": ("D_A3", "D_A4"), "best@8": ("Dmin_A3", "Dmin_A4")}
FIDELITY = "judged fidelity (RealUserSim PT3 index)"
JUDGE = {FIDELITY: "fidelity_index", "judged alignment (HumanLM key points)": "alignment"}


def bootstrap(values, seed=SEED):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def kendall_tau(x, y):
    c = d = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
        c += s > 0
        d += s < 0
    return (c - d) / (len(x) * (len(x) - 1) / 2)


def user_means(users, pairs):
    acc = defaultdict(list)
    for u, v in pairs:
        acc[u].append(v)
    return np.array([np.mean(acc[u]) for u in users])


def load_gain():
    """Content gain per endpoint and candidate from the content files."""
    users, gain, keys, user_of = None, defaultdict(dict), {}, {}
    for c in CANDS:
        e = json.loads((RES / f"content_{c}.json").read_text())
        users = users or e["users"]
        assert e["users"] == users
        keys[c] = [r["key"] for r in e["rows"]]
        user_of.update({r["key"]: r["user_id"] for r in e["rows"]})
        m = [(r["user_id"], r["metrics"][METRIC]) for r in e["rows"]]
        for ep, (a3, a4) in ENDPOINTS.items():
            gain[ep][c] = user_means(users, [(u, x[a4] - x[a3]) for u, x in m])
    return users, gain, keys, user_of


def selector_rows(users, gain, selectors):
    out, pairs_by_ep = {}, {}
    for ep in ENDPOINTS:
        pairs = []
        for l, r in itertools.combinations(CANDS, 2):
            point, lo, hi = bootstrap(gain[ep][l] - gain[ep][r])
            pairs.append((l, r, point, lo > 0 or hi < 0))
        pairs_by_ep[ep] = pairs
        best_gain = min(CANDS, key=lambda c: gain[ep][c].mean())
        order = sorted(CANDS, key=lambda c: gain[ep][c].mean())
        for name, score in selectors.items():
            res = [(l, r, p) for l, r, p, ok in pairs if ok]
            agree = sum(np.sign(score[l].mean() - score[r].mean()) == np.sign(p) for l, r, p in res)
            pick = min(CANDS, key=lambda c: score[c].mean())
            out[f"{name} | {ep}"] = {
                "selector": name,
                "endpoint": ep,
                "n_resolvable": len(res),
                "n_agree": int(agree),
                "kendall_tau": kendall_tau(
                    [score[c].mean() for c in CANDS], [gain[ep][c].mean() for c in CANDS]
                ),
                "top_pick": pick,
                "top_pick_gain_rank": order.index(pick) + 1,
                "top_pick_content_gain": float(-gain[ep][pick].mean()),
                "best_content_gain": float(-gain[ep][best_gain].mean()),
            }
    return out, pairs_by_ep


def main() -> None:
    users, gain, keys, user_of = load_gain()
    scores = json.loads((RES / "judge_scores.json").read_text())
    by = {(r["candidate"], r["arm"], r["key"]): r for r in scores["records"]}
    assert len(by) == len(scores["records"]), "duplicate (candidate, arm, key)"
    missing = [
        (c, a, k) for c in CANDS for a in ("A3", "A4") for k in keys[c] if (c, a, k) not in by
    ]
    coverage = {
        "n_records": len(by),
        "n_missing": len(missing),
        "missing_examples": missing[:5],
        "n_failed_in_judge": scores.get("n_failed"),
    }

    selectors, stranger = {}, {}
    for name, field in JUDGE.items():
        selectors[name] = {}
        stranger[name] = {}
        for c in CANDS:
            present = [k for k in keys[c] if (c, "A4", k) in by and (c, "A3", k) in by]
            a4 = user_means(users, [(user_of[k], by[(c, "A4", k)][field]) for k in present])
            a3 = user_means(users, [(user_of[k], by[(c, "A3", k)][field]) for k in present])
            selectors[name][c] = -a4  # lower = better, as the other selectors
            point, lo, hi = bootstrap(a3 - a4)
            stranger[name][c] = {
                "target_A4": float(a4.mean()),
                "stranger_A3": float(a3.mean()),
                "A3_minus_A4": {"point": point, "low": lo, "high": hi},
                "share_kept": float(a3.mean() / a4.mean()) if a4.mean() else None,
                "n_rows": len(present),
            }
    rows, _ = selector_rows(users, gain, {FIDELITY: selectors[FIDELITY]})

    out = {
        "model": scores.get("model"),
        "n_users": len(users),
        "coverage": coverage,
        "rows": rows,
        "stranger": stranger,
    }
    paths.ensure(RES)
    (RES / "judge_readout.json").write_text(json.dumps(out, indent=1))
    for r in rows.values():
        print(
            f"{r['selector']} | {r['endpoint']}: {r['n_agree']} / {r['n_resolvable']}, tau {r['kendall_tau']:+.2f}, "
            f"top pick {r['top_pick']} (gain rank {r['top_pick_gain_rank']})"
        )
    for name, per in stranger.items():
        for c, v in per.items():
            print(
                f"{name} | {c}: target {v['target_A4']:.3f}, imposter {v['stranger_A3']:.3f}, kept {v['share_kept']:.3f}"
            )


if __name__ == "__main__":
    main()
