"""Attribution map: the four gains of every one of the 84 combinations.

Three panels, one point per combination.  Vertical axis (all panels): gain
against the matched imposter's information, Delta_indiv.  Horizontal axis: gain
against no user information, generic population information, or matched-group
information.  Points to the right of zero but not above it are combinations
where user information helps without identifying the individual.

Data: paths.ANALYSIS / "gains.json" (target contrasts stored as target minus
reference, with paired-user bootstrap 95% intervals) and "identity.json"
(pooled Holm pass on the identity gain).  Gains are the negation, so positive
favors the target user's information; a gain is significant when its interval
lies above zero.

Writes paths.FIGURES / "attribution_map.{json,pdf}" (and .png with --png).
Run:  python -m attribution_trials.figures.attribution_map [--png]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from attribution_trials import paths  # noqa: E402

GAINS = paths.ANALYSIS / "gains.json"
IDENTITY = paths.ANALYSIS / "identity.json"
NAME = "attribution_map"
DATA_FILE = paths.FIGURES / f"{NAME}.json"
OUT_STEM = paths.FIGURES / NAME

CONTRASTS = {"tot": "dtot", "pop": "dpop", "grp": "dgrp", "ind": "dind"}
FAMILY = {
    "static-embedding": "static embedding",
    "evolving-latent": "recurrent embedding",
    "structured-memory": "structured memory",
    "sft-lora": "user-profile LoRA",
    "persona-frozen": "frozen prompt",
    "strong-open": "frozen prompt",
    "trained-sim": "Osim",
    "sim2real-trio": "released simulator",
}

# Font sizes for major text elements.
FONT_TITLE = 8.0
FONT_LABEL = 7.6
FONT_TICK = 7.0
FONT_LEGEND = 7.0
FONT_NOTE = 6.8

# Stroke widths for axes and marks.
LINEWIDTH_AXIS = 0.8
LINEWIDTH_TICK = 0.7
LINEWIDTH_MARK = 0.9
LINEWIDTH_ZERO = 0.7
LINEWIDTH_CI = 0.45

# Figure geometry (inches) and panel spacing.
FIGSIZE = (5.5, 2.15)
WSPACE = 0.10
LAYOUT = dict(left=0.125, right=0.99, top=0.92, bottom=0.30)
LEGEND_Y = -0.01
LEGEND_NCOL = 6
YLIM = (-0.08, 0.08)
X_PAD = 0.06  # fraction of the x range added on both sides
MARKER_SIZE = 3.8
MARKER_SIZE_PASS = 4.6
CI_ALPHA = 0.45
SHADE_COLOR = "#000000"
SHADE_ALPHA = 0.035

# Okabe-Ito, matching the other paper figures.
DATASET_COLOR = {"chess": "#CC79A7", "KT": "#0072B2", "OPeRA": "#E69F00"}
DATASET_LABEL = {"chess": "Chess", "KT": "Knowledge tracing", "OPeRA": "OPeRA (shopping)"}
PANELS = [
    ("tot", "vs. no user information", r"$\Delta_{\mathrm{total}}$ (nats)"),
    ("pop", "vs. population information", r"$\Delta_{\mathrm{pop}}$ (nats)"),
    ("grp", "vs. matched-group information", r"$\Delta_{\mathrm{group}}$ (nats)"),
]
YLABEL = r"vs. matched imposter, $\Delta_{\mathrm{indiv}}$ (nats)"
SHADE_NOTE = "helps, but so does a lookalike's"
SHADE_NOTE_XY = (0.985, 0.31)


def build_data() -> dict:
    gains = json.loads(GAINS.read_text())
    holm = {
        c["key"]: bool(c["identity"]["holm_pooled"])
        for c in json.loads(IDENTITY.read_text())["cells"]
    }
    points = []
    for cell in gains["cells"]:
        g = {}
        for short, name in CONTRASTS.items():
            d = cell["target"][name]
            g[short] = {
                "point": -d["point"],
                "low": -d["high"],
                "high": -d["low"],
                "favorable": d["high"] < 0,
            }
        points.append(
            {
                "key": cell["key"],
                "dataset": cell["dataset"],
                "family": FAMILY[cell["key"].split("|")[0]],
                "gains": g,
                "pass": holm[cell["key"]],
            }
        )
    assert len(points) == 84
    counts = {
        short: {
            ds: sum(p["gains"][short]["favorable"] for p in points if p["dataset"] == ds)
            for ds in ("chess", "KT", "OPeRA")
        }
        for short in CONTRASTS
    }
    for short in counts:
        counts[short]["all"] = sum(counts[short][ds] for ds in ("chess", "KT", "OPeRA"))
    data = {
        "unit": "nats",
        "n": len(points),
        "favorable": counts,
        "passes": sum(p["pass"] for p in points),
        "points": points,
    }
    paths.ensure(DATA_FILE.parent)
    DATA_FILE.write_text(json.dumps(data, indent=1))
    print("wrote", DATA_FILE, counts)
    return data


def require_times_new_roman() -> None:
    names = {f.name for f in fm.fontManager.ttflist}
    if "Times New Roman" not in names:
        raise RuntimeError("Times New Roman is required for paper figures")


def plot(data: dict, png: bool) -> None:
    require_times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": FONT_LABEL,
            "axes.linewidth": LINEWIDTH_AXIS,
            "xtick.major.width": LINEWIDTH_TICK,
            "ytick.major.width": LINEWIDTH_TICK,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    points = data["points"]
    fav = data["favorable"]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(wspace=WSPACE, **LAYOUT)
    for ax, (short, title, xlabel) in zip(axes, PANELS):
        xs = [p["gains"][short]["point"] for p in points]
        lo, hi = min(xs), max(xs)
        pad = (hi - lo) * X_PAD
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(*YLIM)
        # Lower-right quadrant: helps against the reference, not against the imposter.
        ax.axvspan(
            0,
            hi + pad,
            ymin=0,
            ymax=(0 - YLIM[0]) / (YLIM[1] - YLIM[0]),
            color=SHADE_COLOR,
            alpha=SHADE_ALPHA,
            lw=0,
            zorder=0,
        )
        ax.axhline(0, color="#555555", lw=LINEWIDTH_ZERO, zorder=1)
        ax.axvline(0, color="#555555", lw=LINEWIDTH_ZERO, zorder=1)
        for p in points:
            g, y = p["gains"][short], p["gains"]["ind"]
            color = DATASET_COLOR[p["dataset"]]
            ax.plot(
                [g["low"], g["high"]],
                [y["point"], y["point"]],
                color=color,
                lw=LINEWIDTH_CI,
                alpha=CI_ALPHA,
                zorder=2,
                solid_capstyle="butt",
            )
            ax.plot(
                [g["point"], g["point"]],
                [y["low"], y["high"]],
                color=color,
                lw=LINEWIDTH_CI,
                alpha=CI_ALPHA,
                zorder=2,
                solid_capstyle="butt",
            )
        for p in points:
            g, y = p["gains"][short], p["gains"]["ind"]
            color = DATASET_COLOR[p["dataset"]]
            filled = y["favorable"]
            ax.plot(
                g["point"],
                y["point"],
                marker="o",
                ls="none",
                ms=MARKER_SIZE_PASS if p["pass"] else MARKER_SIZE,
                mfc=color if filled else "white",
                mec="black" if p["pass"] else color,
                mew=LINEWIDTH_MARK * (1.4 if p["pass"] else 1.0),
                zorder=4 if filled else 3,
            )
        ax.set_title(title, fontsize=FONT_TITLE, pad=3)
        ax.set_xlabel(xlabel + f", significant {fav[short]['all']} of 84", fontsize=FONT_LABEL)
        ax.tick_params(length=2.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(
        YLABEL + f"\nsignificant {fav['ind']['all']} of 84, {data['passes']} after correction",
        fontsize=FONT_LABEL,
    )
    axes[0].text(
        *SHADE_NOTE_XY,
        SHADE_NOTE,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=FONT_NOTE,
        color="#444444",
        zorder=5,
    )
    handles = [
        Line2D([], [], marker="o", ls="none", ms=MARKER_SIZE, mfc=c, mec=c, label=DATASET_LABEL[d])
        for d, c in DATASET_COLOR.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=MARKER_SIZE,
            mfc="white",
            mec="#666666",
            label="not significant",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=MARKER_SIZE,
            mfc="#666666",
            mec="#666666",
            label="significant",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=MARKER_SIZE_PASS,
            mfc="#666666",
            mec="black",
            mew=LINEWIDTH_MARK * 1.4,
            label="significant after correction",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=LEGEND_NCOL,
        frameon=False,
        fontsize=FONT_LEGEND,
        handletextpad=0.1,
        columnspacing=0.6,
    )
    paths.ensure(OUT_STEM.parent)
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    if png:
        fig.savefig(OUT_STEM.with_suffix(".png"), dpi=170)
    print("wrote", OUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true")
    plot(build_data(), parser.parse_args().png)
