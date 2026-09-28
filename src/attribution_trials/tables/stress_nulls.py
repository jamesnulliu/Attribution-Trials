"""Generate tables/stress_nulls.tex: the drift sweep, with group and matcher results in the caption.

Both inputs store target minus imposter NLL; means are negated here to the
paper's gain (reference minus target).  The drift rows and the coarse
population's magnitudes come from stress_nulls.json; the group-structure
rates in the caption come from reference_nulls.json (`group, fine`).

Inputs: <VALIDATION>/stress_nulls.json, <VALIDATION>/reference_nulls.json

    python -m attribution_trials.tables.stress_nulls
Output: <TABLES>/stress_nulls.tex
"""

from __future__ import annotations

import json

from attribution_trials import paths


def main() -> int:
    blob = json.loads((paths.VALIDATION / "stress_nulls.json").read_text())
    ref = json.loads((paths.VALIDATION / "reference_nulls.json").read_text())["results"]
    res = blob["results"]
    drift_rows = []
    for step in blob["drift_steps"]:
        node = res[f"drift step={step}"]
        a = node["updated (A4live-A3)"]
        b = node["fixed (A4frozen-A3)"]
        drift_rows.append(
            f"{step:.2f} & {a['favorable_rate']:.3f} & {-a['mean']:+.4f} & "
            f"{b['favorable_rate']:.3f} & {-b['mean']:+.4f} \\\\"
        )
    g = ref["group, fine"]
    c = res["coarse"]
    text = f"""\\begin{{table}}[htbp]
\\caption{{Stress nulls, {blob['replicates']} replicates each, rate of significant intervals against a nominal $5\\%$.
Left: a shared drifting rate with identity signal exactly zero---the updated-profile gain fires once drift is moderate, the fixed-profile $\\dind$ never does.
Right: under pure group structure (zero within-group individuality) the tertile rule fires at {g['AT: target vs imposter']['favourable_rate']:.3f} and the accuracy nearest-neighbour at {g['AT, nearest-neighbour imposter']['favourable_rate']:.3f}; on a fully individual, fully stable population the gain falls from {-c['2-cell matcher']['mean']:+.4f} (2-cell) to {-c['tertile matcher']['mean']:+.4f} (tertile) to {-c['NN-accuracy matcher']['mean']:+.4f} (NN).}}
\\label{{tab:stress}}
\\begin{{center}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}crrrr@{{}}}}
\\toprule
& \\multicolumn{{2}}{{c}}{{updated profile}} &
\\multicolumn{{2}}{{c}}{{$\\dind$}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
drift & FP rate & mean & FP rate & mean \\\\
\\midrule
{chr(10).join(drift_rows)}
\\bottomrule
\\end{{tabular*}}
\\end{{center}}
\\end{{table}}
"""
    out = paths.ensure(paths.TABLES) / "stress_nulls.tex"
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
