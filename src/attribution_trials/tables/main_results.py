"""Appendix per-combination table (tab:main_results): one row per combination, four gain columns.

All 84 audit combinations and both gains for all seven WildChat simulators
are retained; the source files store target-minus-reference surprisal, so each
value is negated to give the paper's gain (reference minus target; positive
favors the target).  Bold shows significance, a double dagger pooled Holm passes.

Reads paths.ANALYSIS / {"master_ladder.csv", "gains.json"} and
paths.WILDCHAT / "trial_table.json".  Writes paths.TABLES / "main_results.tex".
Run:  python -m attribution_trials.tables.main_results
"""

import csv
import json

from attribution_trials import paths

AUDIT = paths.ANALYSIS / "master_ladder.csv"
WILDCHAT = paths.WILDCHAT / "trial_table.json"
CONTRASTS = paths.ANALYSIS / "gains.json"  # dtot/dpop/dgrp per combination
OUT = paths.TABLES / "main_results.tex"
GAINS = ("dtot", "dpop", "dgrp", "dind")


def styled_number(point, *, scale=1, decimals=3, bold=False, dagger=False):
    value = f"{point * scale:+.{decimals}f}"
    if bold:
        value = r"\mathbf{" + value + "}"
    value = "$" + value + "$"
    if dagger:
        value += r"$^{\ddagger}$"
    return value


def main():
    rows = list(csv.DictReader(AUDIT.open()))
    full = {row["cell"]: row for row in rows if row["source"] == "full-tier"}
    opera = {
        (row["family"], row["model"], row["config"], row["channel"]): row
        for row in rows
        if row["source"] != "full-tier"
    }
    contrasts = {c["key"]: c for c in json.loads(CONTRASTS.read_text())["cells"]}
    absent = r"\textcolor{ATtablemuted}{--}"
    used = []
    display = []

    def gains(row, key):
        """Four gain cells for one combination: bold = significant, double dagger = Holm pass."""
        used.append((row["source"], row["cell"]))
        target = contrasts[key]["target"]
        holm = contrasts[key]["holm_pooled"]
        assert abs(-target["dind"]["point"] - -float(row["dindiv_point"])) < 1e-9, key
        cells = []
        for name in GAINS:
            d = target[name]
            cells.append(
                styled_number(
                    -d["point"],
                    scale=1000,
                    decimals=1,
                    bold=d["high"] < 0,
                    dagger=(
                        row["holm_reject_pooled_evidence"] == "True"
                        if name == "dind"
                        else holm[name]["target"]
                    ),
                )
            )
        return cells

    def add_audit(dataset, method, channel, row, key):
        display.append(
            {"dataset": dataset, "label": f"{method} / {channel}", "values": gains(row, key)}
        )

    for dataset, domain in [("Chess", "chess-pooled"), ("KT", "kt-rt")]:
        behavior = "move" if dataset == "Chess" else "response"
        if dataset == "Chess":
            add_audit(
                dataset,
                "static embedding",
                "move",
                full["static-embedding|chess-pooled|move"],
                "static-embedding|chess-pooled|move",
            )
            add_audit(
                dataset,
                "static embedding",
                "timing",
                full["static-embedding|chess-pooled|timing"],
                "static-embedding|chess-pooled|timing",
            )
        else:
            add_audit(
                dataset,
                "static embedding (response-only)",
                "response",
                full["static-embedding|kt|move"],
                "static-embedding|kt|move",
            )
            add_audit(
                dataset,
                "static embedding (joint)",
                "response",
                full["static-embedding|kt-rt|move"],
                "static-embedding|kt-rt|move",
            )
            add_audit(
                dataset,
                "static embedding (joint)",
                "timing",
                full["static-embedding|kt-rt|timing"],
                "static-embedding|kt-rt|timing",
            )
        add_audit(
            dataset,
            "recurrent embedding",
            behavior,
            full[f"evolving-latent|{domain}|move"],
            f"evolving-latent|{domain}|move",
        )
        add_audit(
            dataset,
            "recurrent embedding",
            "timing",
            full[f"evolving-latent|{domain}|timing"],
            f"evolving-latent|{domain}|timing",
        )
        if dataset == "KT":
            add_audit(
                dataset,
                "structured memory",
                "response",
                full["structured-memory|kt|response"],
                "structured-memory|kt|response",
            )
        for family, method in [
            ("sft-lora", "user-profile LoRA"),
            ("persona-frozen", "frozen prompt"),
        ]:
            for size in ["1.7b", "8b"]:
                if dataset == "Chess":
                    key = f"{family}|chess ({size})|timing"
                    add_audit(dataset, f"{method} {size.upper()}", "timing", full[key], key)
                else:
                    key = f"{family}|kt ({size})|response"
                    add_audit(dataset, f"{method} {size.upper()}", "response", full[key], key)

    opera_specs = [
        ("sft-lora", "1.7b", "user-profile LoRA", "1.7B"),
        ("sft-lora", "8b", "user-profile LoRA", "8B"),
        ("persona-frozen", "1.7b", "frozen prompt", "1.7B"),
        ("persona-frozen", "8b", "frozen prompt", "8B"),
        ("strong-open", "30b", "frozen prompt", "30B-A3B"),
        ("trained-sim", "osim-4b", "Osim", "4B"),
        ("trained-sim", "osim-4b-mid", "Osim", "4B-mid"),
        ("trained-sim", "osim-8b", "Osim", "8B"),
        ("trained-sim", "osim-8b-mid", "Osim", "8B-mid"),
        ("sim2real-trio", "coser-8b", "CoSER", "8B"),
        ("sim2real-trio", "humanlike-7b", "HumanLike", "7B"),
    ]
    profiles = {"static": "S", "dynamic": "D", "combined": "C"}
    for family, model, method, variant in opera_specs:
        for profile in ["static", "dynamic", "combined"]:
            for channel in ["action", "timing"]:
                row = opera[(family, model, profile, channel)]
                add_audit(
                    "OPeRA",
                    f"{method} {variant} / {profiles[profile]}",
                    channel,
                    row,
                    f"{family}|{channel}|{profile}|{model}",
                )

    wildchat = json.loads(WILDCHAT.read_text())
    wild_specs = [
        ("1.7b", "Qwen3", "1.7B"),
        ("8b", "Qwen3", "8B"),
        ("dense", "Qwen3", "32B (dense)"),
        ("humanlike", "HumanLike", "7B"),
        ("coser", "CoSER", "8B"),
        ("osim", "Osim", "8B"),
        ("humanlm", "HumanLM", "opinion"),
    ]
    for key, method, model in wild_specs:
        row = wildchat["candidate_summary"][key]
        display.append(
            {
                "dataset": "WildChat",
                "label": f"{method} {model} / next turn",
                "values": [
                    styled_number(-row["delta_total"]["point"], scale=1, decimals=3),
                    absent,
                    absent,
                    styled_number(-row["delta_indiv"]["point"], scale=1, decimals=3),
                ],
            }
        )

    out = [
        "% AUTO-GENERATED by attribution_trials.tables.main_results; do not edit by hand.",
        r"\definecolor{ATtablemuted}{HTML}{9AA1A8}",
        r"\begin{center}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.02}",
        r"\setlength{\LTcapwidth}{\linewidth}\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}\begin{longtable}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\caption{\textbf{All four gains for every combination.} "
        r"Units: $10^{-3}$ nats for the 84 audit combinations, nats for the seven WildChat simulators. "
        r"Each entry is a gain of Section~\ref{sec:ladder}; positive favors the target user's information. "
        r"\textbf{Bold}: significant (95\% interval above zero); $^{\ddagger}$: significant after Holm correction over the 84 combinations. "
        r"No combination passes (Section~\ref{sec:stats}): no row carries $^{\ddagger}$ on all four gains. "
        r"OPeRA profiles: S = static, D = dynamic, C = combined; Osim `mid' variants are midtrained checkpoints; "
        r"KT (joint) is the static-embedding model that predicts response and timing together. "
        r"--: gain not defined for that panel. "
        r"Intervals for $\dind$ are in Tables~\ref{tab:master-ladder} and~\ref{tab:selection-intervals}.}"
        r"\label{tab:main_results}\\",
        r"\toprule",
        r"Method / channel & $\dtot$ & $\dpop$ & $\dgrp$ & $\dind$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{@{}l}{\textit{Table~\ref{tab:main_results}, continued}}\\",
        r"\toprule",
        r"Method / channel & $\dtot$ & $\dpop$ & $\dgrp$ & $\dind$ \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        r"\endlastfoot",
    ]
    previous = None
    for row in display:
        if row["dataset"] != previous:
            if previous is not None:
                out.append(r"\addlinespace[3pt]")
            out.append(r"\multicolumn{5}{@{}l}{\textbf{" + row["dataset"] + r"}}\\")
            previous = row["dataset"]
        out.append(row["label"] + " & " + " & ".join(row["values"]) + r" \\")
    out.extend([r"\end{longtable}", r"\end{center}", ""])

    expected = {(row["source"], row["cell"]) for row in rows}
    assert len(rows) == len(expected) == len(used) == len(set(used)) == 84
    assert set(used) == expected
    assert {key for key, _, _ in wild_specs} == set(wildchat["candidates"])
    assert wildchat["n_users"] == 240
    paths.ensure(OUT.parent)
    OUT.write_text("\n".join(out))
    print(f"Wrote {OUT}: {len(display)} rows, all 84 audit combinations and 7 WildChat simulators.")


if __name__ == "__main__":
    main()
