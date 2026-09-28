"""Content gain at a lower decoding temperature against temperature 1, three simulators.

For each temperature and simulator: content gain (behavioral delta_indiv,
avg@8 and best@8, user bootstrap 10k seed 0); the three-way content-gain
ordering at each temperature; and the paired per-user change in content gain
between temperatures.  Writes ``paths.WILDCHAT / "temperature_readout.json"``.

  python -m attribution_trials.wildchat.temperature_readout [--dir <paths.WILDCHAT>/temperature_0.7]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from attribution_trials import paths

RES = paths.WILDCHAT
SEED, NBOOT, METRIC = 0, 10_000, "M1_minilm_max_card_bullet"
CANDS = ("8b", "humanlike", "osim")
ENDPOINTS = {"avg@8": ("D_A3", "D_A4"), "best@8": ("Dmin_A3", "Dmin_A4")}


def boot(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return {
        "point": float(values.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def user_means(users, pairs):
    acc = defaultdict(list)
    for u, v in pairs:
        acc[u].append(v)
    return np.asarray([np.mean(acc[u]) for u in users])


def readout(dirpath: Path, users, keys):
    out = {"per_candidate": {}, "gain": {}}
    for c in CANDS:
        e = json.loads((dirpath / f"content_{c}.json").read_text())
        assert e["users"] == users and [r["key"] for r in e["rows"]] == keys, c
        m = [(r["user_id"], r["metrics"][METRIC]) for r in e["rows"]]
        entry = {
            "temperature": e["temperature"],
            "model": e["model"],
            "backend": {k: e["runtime"].get(k) for k in ("gpu", "transformers", "torch")},
        }
        for ep, (a3, a4) in ENDPOINTS.items():
            gain = user_means(users, [(u, x[a4] - x[a3]) for u, x in m])
            out["gain"].setdefault(ep, {})[c] = gain
            entry[ep] = {"behavioral_delta_indiv": boot(gain)}
        out["per_candidate"][c] = entry
    for ep in ENDPOINTS:
        g = out["gain"][ep]
        out[f"order {ep}"] = sorted(CANDS, key=lambda c: g[c].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(RES / "temperature_0.7"))
    args = ap.parse_args()
    ref = json.loads((RES / "content_8b.json").read_text())
    users, keys = ref["users"], [r["key"] for r in ref["rows"]]
    base = readout(RES, users, keys)
    new = readout(Path(args.dir), users, keys)

    change = {}
    for ep in ENDPOINTS:
        change[ep] = {c: boot(new["gain"][ep][c] - base["gain"][ep][c]) for c in CANDS}
    ordering_holds = {ep: new[f"order {ep}"] == base[f"order {ep}"] for ep in ENDPOINTS}

    def strip(o):
        return {k: v for k, v in o.items() if k != "gain"}

    result = {
        "candidates": CANDS,
        "metric": METRIC,
        "baseline_temperature": base["per_candidate"]["8b"]["temperature"],
        "rerun_temperature": new["per_candidate"]["8b"]["temperature"],
        "baseline": strip(base),
        "rerun": strip(new),
        "paired_change_in_content_gain_rerun_minus_baseline": change,
        "three_way_ordering_holds": ordering_holds,
    }
    paths.ensure(RES)
    (RES / "temperature_readout.json").write_text(json.dumps(result, indent=1) + "\n")
    for c in CANDS:
        for ep in ENDPOINTS:
            b, n, ch = (
                base["per_candidate"][c][ep]["behavioral_delta_indiv"],
                new["per_candidate"][c][ep]["behavioral_delta_indiv"],
                change[ep][c],
            )
            print(
                f"{c} {ep}: content gain {-b['point']:+.4f} -> {-n['point']:+.4f}; "
                f"change {-ch['point']:+.4f} [{-ch['ci_high']:+.4f}, {-ch['ci_low']:+.4f}]"
            )
    print(f"three-way ordering holds: {ordering_holds}")


if __name__ == "__main__":
    main()
