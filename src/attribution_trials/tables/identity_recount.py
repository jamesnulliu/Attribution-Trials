"""Generate tables/identity_recount.tex: significance counts, updated versus fixed profile.

Per group of combinations: how many are significant (interval above zero in
the paper's sign) and how many pass pooled Holm, when the target profile is
updated with held-out rows (`registered` in identity.json) versus fixed at the
training boundary (`identity`, the paper's gain).

Input: <ANALYSIS>/identity.json

    python -m attribution_trials.tables.identity_recount
Output: <TABLES>/identity_recount.tex
"""

from __future__ import annotations

import json

from attribution_trials import paths

PANELS = (
    ("full-tier", "chess and knowledge tracing"),
    ("opera-gpu", "OPeRA, LoRA and frozen prompt"),
    ("sim2real-trio", "OPeRA, CoSER and HumanLike"),
    ("osim-panel", "OPeRA, Osim"),
    ("pooled-84", "all"),
)


def main() -> int:
    counts = json.loads((paths.ANALYSIS / "identity.json").read_text())["counts"]
    rows = []
    for panel, label in PANELS:
        a = counts["registered"][panel]
        b = counts["identity"][panel]
        rows.append(
            f"{label} & {a['cells']} & "
            f"{a['ci_excludes_0_favorable']} $\\to$ "
            f"{b['ci_excludes_0_favorable']} & "
            f"{a['holm_pooled_survivors']} $\\to$ "
            f"{b['holm_pooled_survivors']} \\\\"
        )
    text = f"""\\begin{{table}}[htbp]
\\caption{{Significance counts when the target profile is updated with held-out rows (`updated') versus fixed at the training boundary ($\\dind$, the gain used in the paper).
`Significant' counts combinations whose interval lies above zero; the final column counts pooled-Holm passes.}}
\\label{{tab:recount}}
\\begin{{center}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}lrcc@{{}}}}
\\toprule
combinations & number & \\shortstack{{significant\\\\updated / $\\dind$}} & \\shortstack{{pooled Holm pass\\\\updated / $\\dind$}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular*}}
\\end{{center}}
\\end{{table}}
"""
    out = paths.ensure(paths.TABLES) / "identity_recount.tex"
    out.write_text(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
