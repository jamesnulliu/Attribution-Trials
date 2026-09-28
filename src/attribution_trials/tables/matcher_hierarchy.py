"""Generate tables/matcher_hierarchy.tex: each combination under three imposter matchers.

matcher_hierarchy.json stores target minus imposter NLL; every value is
negated here to the paper's gain (reference minus target, positive favors the
target), so an interval [low, high] prints as [-high, -low].

Input: <VALIDATION>/matcher_hierarchy.json

    python -m attribution_trials.tables.matcher_hierarchy
Output: <TABLES>/matcher_hierarchy.tex
"""

from __future__ import annotations

import json

from attribution_trials import paths

SHORT = {
    "static-embedding|chess-pooled|move": "static $\\cdot$ chess $\\cdot$ move",
    "static-embedding|chess-pooled|timing": "static $\\cdot$ chess $\\cdot$ timing",
    "static-embedding|kt|move": "static $\\cdot$ KT $\\cdot$ resp.",
    "static-embedding|kt-rt|move": "static $\\cdot$ KT $\\cdot$ resp.",
    "static-embedding|kt-rt|timing": "static $\\cdot$ KT $\\cdot$ timing",
    "evolving-latent|kt-rt|move": "recurrent embedding $\\cdot$ KT $\\cdot$ resp.",
    "evolving-latent|kt-rt|timing": "recurrent embedding $\\cdot$ KT $\\cdot$ timing",
    "sft-lora|kt (8b)|response": "user-profile LoRA 8B $\\cdot$ KT $\\cdot$ resp.",
    "persona-frozen|kt (8b)|response": "frozen prompt 8B $\\cdot$ KT $\\cdot$ resp.",
}


def ci(node, digits=4):
    """Paper sign: the gain and its interval, negated."""
    if not node:
        return "---"
    return (
        f"${-node['point']:+.{digits}f}$ [${-node['high']:+.{digits}f}$,"
        f"${-node['low']:+.{digits}f}$]"
    )


def main() -> int:
    blob = json.loads((paths.VALIDATION / "matcher_hierarchy.json").read_text())
    rows = []
    for r in blob["rows"]:
        cert = "\\yesmark" if r["certified_identity"] else ""
        rows.append(
            f"{SHORT.get(r['cell'], r['cell'])} & {cert} & "
            f"{ci(r['registered'])} & {ci(r['nn'])} & {ci(r['hardest'])} & "
            f"{r['k_of_matchers']}/{r['matchers_reported']} \\\\"
        )
    text = f"""\\begin{{table}}[htbp]
\\caption{{The matcher hierarchy: the registered tertile derangement, the training-only nearest-neighbour matcher, and the hardest donor in the recorded set (an adversarial bound, not an estimand).
`k/m' counts matchers under which the gain remains significant, of those reported; the two prompting combinations have no hardest-donor bound recorded and carry their Section~\\ref{{sec:fulltier}} bound.}}
\\label{{tab:matcherhier}}
\\begin{{center}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{1.8pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}p{{73pt}}crrrc@{{}}}}
\\toprule
combination & cert. & registered [CI] & nearest-neighbor [CI] & hardest [CI] & k/m \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular*}}
\\end{{center}}
\\end{{table}}
"""
    out = paths.ensure(paths.TABLES) / "matcher_hierarchy.tex"
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
