"""Readout of the target-information dose ladder.

alpha 0 is the matched imposter's frozen card (the A3 endpoint), alpha 1 the
target's own (the A4frozen endpoint); interior points are the mean over three
seeds.  Reports the end-to-end swing, monotonicity, and the largest
across-seed standard deviation.

  python -m attribution_trials.validation.dose_readout
Output: <VALIDATION>/dose_readout.json
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from attribution_trials import paths

LADDER = paths.VALIDATION / "dose"
ALPHAS = ("25", "50", "75")
SEEDS = (0, 1, 2)


def substrates(ladder: str) -> list[str]:
    found = set()
    for path in glob.glob(os.path.join(ladder, "mix_fixed_*_8b_a*_s*.json")):
        match = re.match(r"mix_fixed_(.+)_8b_a\d+_s\d+\.json$", os.path.basename(path))
        if match:
            found.add(match.group(1))
    return sorted(found)


def analyse(ladder: str, substrate: str) -> dict:
    a3 = np.load(os.path.join(ladder, f"endpoint_{substrate}_8b_A3.npy"))
    a4 = np.load(os.path.join(ladder, f"endpoint_{substrate}_8b_A4frozen.npy"))
    points = [
        {
            "alpha": 0.0,
            "realized": 0.0,
            "mean_nll": float(a3.mean()),
            "sd_over_seeds": None,
            "seeds": None,
            "source": "A3 endpoint",
        }
    ]
    for alpha in ALPHAS:
        nlls, realized = [], []
        for seed in SEEDS:
            path = os.path.join(ladder, f"mix_fixed_{substrate}_8b_a{alpha}_s{seed}.json")
            with open(path) as handle:
                record = json.load(handle)
            nlls.append(record["mean_nll"])
            realized.append(record["mean_realized_target_fraction"])
        points.append(
            {
                "alpha": int(alpha) / 100,
                "realized": float(np.mean(realized)),
                "mean_nll": float(np.mean(nlls)),
                "sd_over_seeds": float(np.std(nlls, ddof=1)),
                "seeds": [float(v) for v in nlls],
                "source": "mixed",
            }
        )
    points.append(
        {
            "alpha": 1.0,
            "realized": 1.0,
            "mean_nll": float(a4.mean()),
            "sd_over_seeds": None,
            "seeds": None,
            "source": "A4frozen endpoint",
        }
    )

    means = [p["mean_nll"] for p in points]
    swing = means[-1] - means[0]
    # monotone decreasing: every added share of the target's card helps
    monotone = all(b <= a + 1e-12 for a, b in zip(means, means[1:]))
    interior_sd = [p["sd_over_seeds"] for p in points if p["sd_over_seeds"] is not None]
    return {
        "substrate": substrate,
        "n_rows": int(len(a3)),
        "points": points,
        "swing_nats": float(swing),
        "swing_pct": float(100.0 * swing / means[0]),
        "monotone_decreasing": bool(monotone),
        "max_seed_sd": float(max(interior_sd)) if interior_sd else None,
        "swing_over_seed_sd": float(abs(swing) / max(interior_sd)) if interior_sd else None,
    }


def main() -> None:
    ladder = str(LADDER)
    results = [analyse(ladder, s) for s in substrates(ladder)]
    if not results:
        raise SystemExit(f"No ladder json under {ladder}")

    for r in results:
        print(f"=== {r['substrate']} (n={r['n_rows']}, 8b) ===")
        print(f"  {'alpha':>6s} {'realized':>9s} {'mean NLL':>10s} {'sd(seed)':>9s}  source")
        for p in r["points"]:
            sd = f"{p['sd_over_seeds']:9.5f}" if p["sd_over_seeds"] is not None else "        -"
            print(
                f"  {p['alpha']:6.2f} {p['realized']:9.3f} {p['mean_nll']:10.5f} {sd}  {p['source']}"
            )
        print(
            f"  swing A3->A4f: {r['swing_nats']:+.5f} nats ({r['swing_pct']:+.2f}%), "
            f"monotone={r['monotone_decreasing']}, "
            f"swing/seed-sd={r['swing_over_seed_sd']:.2f}"
        )

    out = paths.ensure(paths.VALIDATION) / "dose_readout.json"
    with open(out, "w") as handle:
        json.dump({"substrates": results}, handle, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
