"""Generate tables/stronger_baselines.tex (tab:strongbase).

Source: <COMPARISON>/stronger_baselines.json, which stores target-minus-imposter
surprisal; the table prints reference minus target (imposter minus target), so
every contrast and residual is negated and its interval ends swapped.  Capture
ratios are sign-free.

    python -m attribution_trials.tables.stronger_baselines
Output: <TABLES>/stronger_baselines.tex
"""

import json

from attribution_trials import paths

LABEL = {
    "static-embedding|kt|move": "static $\\cdot$ KT $\\cdot$ resp.",
    "static-embedding|kt-rt|move": "static $\\cdot$ KT (joint) $\\cdot$ resp.",
    "evolving-latent|kt-rt|move": "recurrent embedding $\\cdot$ KT $\\cdot$ resp.",
    "sft-lora|kt (1.7b)|response": "user-profile LoRA 1.7B $\\cdot$ KT $\\cdot$ resp.",
    "sft-lora|kt (8b)|response": "user-profile LoRA 8B $\\cdot$ KT $\\cdot$ resp.",
    "persona-frozen|kt (1.7b)|response": "frozen prompt 1.7B $\\cdot$ KT $\\cdot$ resp.",
    "persona-frozen|kt (8b)|response": "frozen prompt 8B $\\cdot$ KT $\\cdot$ resp.",
}


def num(x, digits=4):
    return f"${-x:+.{digits}f}$"


def ci(node, digits=4):
    return (
        f"${-node['point']:+.{digits}f}$ [${-node['high']:+.{digits}f}$,"
        f"${-node['low']:+.{digits}f}$]"
    )


def main() -> None:
    blob = json.loads((paths.COMPARISON / "stronger_baselines.json").read_text())
    rows = []
    for r in blob["cells"]:
        rows.append(
            f"{LABEL[r['cell']]} & "
            f"{ci(r['method_identity'])} & "
            f"{num(r['irt']['delta']['point'])} & "
            f"{r['irt']['capture_of_identity']['point']:.2f} & "
            f"{num(r['fields']['delta']['point'])} & "
            f"{r['fields']['capture_of_identity']['point']:.2f} & "
            f"{ci(r['fields']['residual_vs_identity'])} \\\\"
        )
    text = f"""\\begin{{table}}[htbp]
\\caption{{Stronger train-only baselines on the KT response combinations.
IRT is a 1PL per-student ability with per-item difficulty; `fields' is a population logistic over the profile's own ten numeric fields, frozen at each user's boundary.
Both are recency-clean.
Capture is baseline over method (ratio of means); the residual is per-user method minus the fields baseline.
No residual interval excludes zero favorably.}}
\\label{{tab:strongbase}}
\\begin{{center}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}p{{88pt}}rrrrrr@{{}}}}
\\toprule
combination & method $\\dind$ [CI] & IRT & capt. & fields & capt. & residual [CI] \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular*}}
\\end{{center}}
\\end{{table}}
"""
    out = paths.ensure(paths.TABLES) / "stronger_baselines.tex"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
