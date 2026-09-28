"""Appendix table (tab:seedinf): training-seed variability in the target-versus-imposter intervals.

Rows: every combination with three seeds that passes pooled Holm on either the
fixed-profile or the updated-profile gain.  Values are the fixed-profile gain,
printed as imposter minus target (positive favors the target).

Reads paths.ANALYSIS / "seed_inference.json".  Writes paths.TABLES / "seed_inference.tex".
Run:  python -m attribution_trials.tables.seed_inference
"""

from __future__ import annotations

import json

from attribution_trials import paths

SRC = paths.ANALYSIS / "seed_inference.json"
OUT = paths.TABLES / "seed_inference.tex"

LABEL = {
    "static-embedding|chess-pooled|move": "static $\\cdot$ chess $\\cdot$ move",
    "static-embedding|chess-pooled|timing": "static $\\cdot$ chess $\\cdot$ timing",
    "static-embedding|kt|move": "static $\\cdot$ KT $\\cdot$ resp.",
    "static-embedding|kt-rt|move": "static $\\cdot$ KT $\\cdot$ resp.",
    "static-embedding|kt-rt|timing": "static $\\cdot$ KT $\\cdot$ timing",
    "structured-memory|kt|response": "structured memory $\\cdot$ KT $\\cdot$ resp.",
    "evolving-latent|chess-pooled|move": "recurrent embedding $\\cdot$ chess $\\cdot$ move",
    "evolving-latent|chess-pooled|timing": "recurrent embedding $\\cdot$ chess $\\cdot$ timing",
    "evolving-latent|kt-rt|move": "recurrent embedding $\\cdot$ KT $\\cdot$ resp.",
    "evolving-latent|kt-rt|timing": "recurrent embedding $\\cdot$ KT $\\cdot$ timing",
    "sft-lora|kt (1.7b)|response": "user-profile LoRA 1.7B $\\cdot$ KT $\\cdot$ resp.",
    "sft-lora|kt (8b)|response": "user-profile LoRA 8B $\\cdot$ KT $\\cdot$ resp.",
    "sft-lora|chess (1.7b)|timing": "user-profile LoRA 1.7B $\\cdot$ chess $\\cdot$ timing",
    "sft-lora|chess (8b)|timing": "user-profile LoRA 8B $\\cdot$ chess $\\cdot$ timing",
}


def gain_ci(node):
    """Stored target-minus-imposter interval, printed as imposter minus target."""
    return f"[${-node['high']:+.4f}$, ${-node['low']:+.4f}$]"


def main() -> None:
    blob = json.loads(SRC.read_text())
    rows = []
    for r in blob["cells"]:
        a = r["identity"]
        if not (a["pooled_holm_survivor"] or r["registered"]["pooled_holm_survivor"]):
            continue
        rows.append(
            f"{LABEL.get(r['key'], r['key'])} & ${-a['point']:+.4f}$ & "
            f"{gain_ci(a['user_only'])} & {gain_ci(a['hierarchical'])} & "
            f"{a['width_ratio']:.2f} & {a['loso_range']:.4f} \\\\"
        )
    reg = blob["survivor_width_ratio_range"]["registered"]
    text = f"""\\begin{{table}}[htbp]
\\caption{{Training-seed variability in the intervals, for every combination that passes pooled Holm on either the fixed or the updated-profile gain and has three seeds.
The hierarchical bootstrap resamples users and seeds jointly; LOSO is the spread of the three leave-one-seed-out points.
Across these combinations the width ratio runs {reg[0]:.2f} to {reg[1]:.2f}; none loses resolution under the hierarchical interval on either gain.}}
\\label{{tab:seedinf}}
\\begin{{center}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}lrrrrr@{{}}}}
\\toprule
combination & $\\dind$ & user-only CI & hierarchical CI & ratio & LOSO \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular*}}
\\end{{center}}
\\end{{table}}
"""
    paths.ensure(OUT.parent)
    OUT.write_text(text)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
