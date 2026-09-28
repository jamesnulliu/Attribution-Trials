#!/usr/bin/env python3
"""Dose figure: surprisal against the share of the target's information.

One facet per substrate. x is the nominal share alpha of the target's fixed
information mixed into the imposter's (0 = the A3 row, 1 = the A4frozen row);
y is the mean surprisal per decision. Interior points are the mean over three
seeds with the across-seed sd as the error bar and the seeds as small dots;
the endpoints carry no bar.

Data : <VALIDATION>/dose_readout.json (validation.dose_readout).
Run  : python -m attribution_trials.figures.dose_ladder
Output: <FIGURES>/dose_ladder.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from attribution_trials import paths

# --- style knobs: one variable per thing it controls ------------------------
FONT_TITLE = 10  # facet titles (axes.titlesize)
FONT_LABEL = 9.5  # axis labels
FONT_TICK = 8.5  # tick labels, and the base font size
FONT_LEGEND = 8.5  # legend entries
LINEWIDTH_AXIS = 0.6  # axis spines
LINEWIDTH_TICK = 0.6  # tick marks

# The paper's figures are set in Times New Roman; fail rather than fall back.
_TNR_PATH = font_manager.findfont(
    font_manager.FontProperties(family="Times New Roman"),
    fallback_to_default=False,
)
if "Times New Roman" not in font_manager.FontProperties(fname=_TNR_PATH).get_name():
    raise RuntimeError(
        "Times New Roman not installed; refusing to fall back. "
        "Install the font and clear matplotlib's font cache."
    )

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": FONT_TICK,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": LINEWIDTH_AXIS,
        "xtick.major.width": LINEWIDTH_TICK,
        "ytick.major.width": LINEWIDTH_TICK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

# --- figure knobs: one variable per thing it controls -----------------------
PANEL_WIDTH = 2.75  # figure width per facet; total width = this x n_facets
PANEL_HEIGHT = 2.25  # figure height
LINE_COLOR = "0.25"  # the mixture curve
LINE_KW = dict(linewidth=1.3, zorder=3)
MARKER_INTERIOR = dict(marker="o", markersize=5, mfc="white", mec=LINE_COLOR, mew=1.1, zorder=4)
MARKER_ENDPOINT = dict(
    marker="s", markersize=5.5, mfc=LINE_COLOR, mec=LINE_COLOR, mew=0.8, zorder=5
)
SEED_KW = dict(marker=".", markersize=4, color="tab:blue", alpha=0.7, linestyle="none", zorder=2)
ERRORBAR_KW = dict(ecolor=LINE_COLOR, elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)
Y_PAD = 0.18  # fraction of the y span left above and below the data
X_TICKS = [0, 0.25, 0.5, 0.75, 1.0]
X_LABEL = r"Share $\alpha$ of the target's information"
Y_LABEL = "Surprisal (nats per decision)"
LEGEND_Y = -0.04  # figure-level legend anchor
LAYOUT_RECT = (0, 0.10, 1, 1.0)  # room reserved below the axes for the legend

# Vocabulary: order and display text. Membership comes from the data.
SUBSTRATES = ["kt", "chess"]
SUBSTRATE_DISPLAY = {"kt": "KT response, Qwen3-8B", "chess": "Chess timing, Qwen3-8B"}

DATA_FILE = paths.VALIDATION / "dose_readout.json"
OUT_FILE = paths.FIGURES / "dose_ladder.pdf"


def load(path: Path) -> dict:
    """{substrate: points without their `source` field}, from the readout JSON."""
    readout = json.loads(path.read_text())
    return {
        s["substrate"]: [{k: v for k, v in p.items() if k != "source"} for p in s["points"]]
        for s in readout["substrates"]
    }


def ordered(present, declared: list) -> list:
    """Declared order first, then whatever else the data contains."""
    return [k for k in declared if k in present] + [k for k in present if k not in declared]


def plot(data: dict, out_pdf: Path) -> None:
    substrates = ordered(data, SUBSTRATES)
    n = len(substrates)
    fig, axes_arr = plt.subplots(1, n, figsize=(PANEL_WIDTH * n, PANEL_HEIGHT), squeeze=False)
    axes = list(axes_arr[0])

    for panel_idx, (sub, ax) in enumerate(zip(substrates, axes)):
        pts = sorted(data[sub], key=lambda p: p["alpha"])
        xs = np.array([p["alpha"] for p in pts], float)
        ys = np.array([p["mean_nll"] for p in pts], float)
        sd = np.array([p["sd_over_seeds"] or 0.0 for p in pts], float)
        endpoint = np.array([p["sd_over_seeds"] is None for p in pts])

        ax.plot(xs, ys, color=LINE_COLOR, **LINE_KW)
        for p in pts:
            if p["seeds"]:
                ax.plot([p["alpha"]] * len(p["seeds"]), p["seeds"], **SEED_KW)
        ax.errorbar(
            xs[~endpoint], ys[~endpoint], yerr=sd[~endpoint], linestyle="none", **ERRORBAR_KW
        )
        ax.plot(xs[~endpoint], ys[~endpoint], linestyle="none", **MARKER_INTERIOR)
        ax.plot(xs[endpoint], ys[endpoint], linestyle="none", **MARKER_ENDPOINT)

        lo = min(ys.min(), min(min(p["seeds"]) for p in pts if p["seeds"]))
        hi = max(ys.max(), max(max(p["seeds"]) for p in pts if p["seeds"]))
        span = hi - lo
        ax.set_ylim(lo - Y_PAD * span, hi + Y_PAD * span)
        ax.set_xlim(-0.06, 1.06)
        ax.set_xticks(X_TICKS)
        ax.set_title(SUBSTRATE_DISPLAY.get(sub, sub))
        ax.set_xlabel(X_LABEL)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=0)
        if panel_idx == 0:
            ax.set_ylabel(Y_LABEL)

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=LINE_COLOR,
            linestyle="none",
            **MARKER_ENDPOINT,
            label=r"banked endpoint ($\alpha=0$: imposter's, $\alpha=1$: target's)",
        ),
        Line2D(
            [0],
            [0],
            color=LINE_COLOR,
            linestyle="none",
            **MARKER_INTERIOR,
            label="mixture, mean of 3 seeds $\\pm$ sd",
        ),
        Line2D([0], [0], **SEED_KW, label="one seed"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=3,
        frameon=False,
        fontsize=FONT_LEGEND,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=LAYOUT_RECT)

    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA_FILE)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(load(args.data), args.out)


if __name__ == "__main__":
    main()
