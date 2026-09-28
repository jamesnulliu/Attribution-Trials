#!/usr/bin/env python3
"""Headline figure.

(a) OPeRA timing: real users and four simulators in a PCA basis fitted on the
    real users, with 2-SD population ellipses and the real-user mean.
(b) The trial: the five user-information conditions, the gain definition, the
    four gains, and the datasets.

Data  : <FIGURES>/opera_timing_pca.json (figures.headline_data) and the PNG
        icons in assets/ next to this module.
Run   : python -m attribution_trials.figures.headline
Output: <FIGURES>/headline.pdf
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from attribution_trials import paths

FONT_TITLE = 19  # panel headings
FONT_LABEL = 17  # workflow body and PCA labels
FONT_TICK = 14  # PCA numeric ticks
FONT_LEGEND = 16  # one-line population-source legend
LINEWIDTH_AXIS = 0.65  # PCA axes
LINEWIDTH_TICK = 0.65  # PCA ticks
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
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'image.composite_image': False,
        'font.size': FONT_LABEL,
        'axes.titlesize': FONT_TITLE,
        'axes.labelsize': FONT_LABEL,
        'xtick.labelsize': FONT_TICK,
        'ytick.labelsize': FONT_TICK,
        'legend.fontsize': FONT_LEGEND,
        'axes.linewidth': LINEWIDTH_AXIS,
        'xtick.major.width': LINEWIDTH_TICK,
        'ytick.major.width': LINEWIDTH_TICK,
    }
)


ASSETS = Path(__file__).resolve().parent / 'assets'
DATA_FILE = paths.FIGURES / 'opera_timing_pca.json'
OUT_FILE = paths.FIGURES / 'headline.pdf'
METHOD_NAME = 'Attribution Trials'
WIDTH = 14
HEIGHTS = dict(final=3.62)
DPI = 240
INK = '#24323B'
MUTED = '#637179'
RULE = '#CCD5D9'
PAPER = '#FFFFFF'
WORKFLOW_STROKE = '#000000'
WORKFLOW_LINEWIDTH = 1.4
PALE = '#F0F2F3'
AT_COLOR = '#236A70'
AT_FILL = '#EFF6F5'
AT_BORDER = '#174C51'
NO_USER_TEXT = '#4E5961'
CONDITIONS = ['No user', 'Population', 'Group', 'Imposter', 'Target']
CONDITION_COLORS = ['#7B8790', '#507DAC', '#8A73AE', '#C5853F', '#27837B']
CONDITION_TEXT = [NO_USER_TEXT, '#507DAC', '#8A73AE', '#C5853F', '#27837B']
CONDITION_BORDERS = ['#4E5961', '#2D5279', '#5A4377', '#88531F', '#1C615A']
CONDITION_FILLS = ['#F0F2F3', '#EDF3FA', '#F2EDF8', '#FCF2E7', '#E8F5F0']
SYMBOLS = ['no-user', 'pop', 'group', 'imposter', 'target']
GAINS = ['total', 'pop', 'group', 'indiv']
PCA_ORDER = ['Real users', 'Qwen3-8B', 'Osim-8B', 'CoSER-8B', 'HumanLike-7B']
PCA_COLORS = {
    'Real users': '#596572',
    'Qwen3-8B': '#155F96',
    'Osim-8B': '#A75A18',
    'CoSER-8B': '#147A7A',
    'HumanLike-7B': '#744395',
}
PCA_LABELS = {
    'Real users': 'Real',
    'Qwen3-8B': 'Qwen3',
    'Osim-8B': 'Osim',
    'CoSER-8B': 'CoSER',
    'HumanLike-7B': 'HumanLike',
}
PCA_XLIM = (-0.42, 0.62)
PCA_YLIM = (-0.34, 0.30)
PCA_LEFT = 0.48
PCA_TOP = 0.43
PCA_WIDTH = 4.08
PCA_BOTTOM_SPACE = 0.72
PCA_MARKER_REAL = 0.073
PCA_MARKER_MODEL = 0.115
PCA_MARKER_LEGEND = 0.074
PCA_TRIANGLE_LEGEND = 0.10
PCA_LEGEND_X = [0.18, 0.91, 1.85, 2.63, 3.51]
# One real-user point is not drawn (it lies outside the axis range); all real
# users still define the basis, the ellipse and the mean.
HIDDEN_REAL_INDEX = 7
PCA_ELLIPSE_SD = 2
PCA_ELLIPSE_REAL_ALPHA = 0.14
PCA_ELLIPSE_MODEL_ALPHA = 0.24
PCA_GRID_COLOR = '#D5DDE2'
PCA_GRID_LINEWIDTH = 0.65
PCA_GRID_X = [-0.2, 0, 0.2, 0.4]
PCA_GRID_Y = [-0.2, 0, 0.2]
PCA_MEAN_SIZE = 0.17
PCA_MEAN_OUTLINE = 0.85
# A single filled, symmetric multiplication cross with fine white separation.
PCA_MEAN_VERTICES = [
    (0, 0.18),
    (0.18, 0),
    (0.50, 0.32),
    (0.82, 0),
    (1, 0.18),
    (0.68, 0.50),
    (1, 0.82),
    (0.82, 1),
    (0.50, 0.68),
    (0.18, 1),
    (0, 0.82),
    (0.32, 0.50),
]
FONT_AVERAGE = 16
FONT_DEFINITION = 19
FONT_GAIN_EXPLANATION = 18
RIGHT_X = 4.95
RIGHT_EDGE = 13.83
FONT_COLUMN = 18
FONT_CONDITION = 17
FONT_FORMULA = 18
FONT_METHOD = 20
FONT_DELTA = 21
FONT_DELTA_COMPACT = 20
FONT_GAIN_LABEL = 17
FONT_DATASET_TITLE = 17
FONT_DATASET_CHANNEL = 15
FONT_DATASET_GROUP = 15
GAIN_REFERENCES = ['no user', 'population', 'matched group', 'matched imposter']
DATASETS = [
    ('lichess.png', 'Lichess', 'Moves\nThinking time', 4.99, 2.11),
    ('assistments.png', 'ASSISTments', 'Correctness\nResponse time', 7.10, 2.25),
    ('opera.png', 'OPeRA', 'Actions\nTiming', 9.35, 2.01),
    ('wildchat.png', 'WildChat', 'Free-text\ndialogue', 11.53, 2.30),
]


class Scene:
    def __init__(self, name):
        self.name, self.height, self.items = name, HEIGHTS[name], []

    def text(
        self,
        x,
        y,
        w,
        h,
        text,
        size=FONT_LABEL,
        color=INK,
        bold=False,
        align='left',
        math=None,
        rotation=0,
    ):
        self.items.append(
            dict(
                kind='text',
                x=x,
                y=y,
                w=w,
                h=h,
                text=text,
                size=size,
                color=color,
                bold=bold,
                align=align,
                math=math,
                rotation=rotation,
            )
        )

    def box(self, x, y, w, h, fill=PALE, edge=RULE, radius=0.06, lw=0.8):
        self.items.append(
            dict(kind='box', x=x, y=y, w=w, h=h, fill=fill, edge=edge, radius=radius, lw=lw)
        )

    def line(self, a, b, color=RULE, lw=0.9, arrow=False, dashed=False):
        self.items.append(
            dict(kind='line', a=a, b=b, color=color, lw=lw, arrow=arrow, dashed=dashed)
        )

    def grid(self, a, b):
        self.items.append(dict(kind='grid', a=a, b=b, color=PCA_GRID_COLOR, lw=PCA_GRID_LINEWIDTH))

    def marker(self, x, y, size, shape, color, alpha=1):
        self.items.append(
            dict(kind='marker', x=x, y=y, size=size, shape=shape, color=color, alpha=alpha)
        )

    def ellipse(self, x, y, w, h, angle, color, alpha, edge_alpha=0.25, lw=0.7):
        self.items.append(
            dict(
                kind='ellipse',
                x=x,
                y=y,
                w=w,
                h=h,
                angle=angle,
                color=color,
                alpha=alpha,
                edge_alpha=edge_alpha,
                lw=lw,
            )
        )

    def mean(self, x, y, color):
        self.items.append(dict(kind='mean', x=x, y=y, size=PCA_MEAN_SIZE, color=color))

    def image(self, x, y, w, h, asset='imposter_red.png'):
        self.items.append(dict(kind='image', x=x, y=y, w=w, h=h, asset=asset))


def base(s, data, horizontal=False):
    s.text(0.08, 0.025, 4.6, 0.32, '(a) OPeRA (Timing)', FONT_TITLE, INK, True)
    s.text(
        RIGHT_X,
        0.025,
        4.8,
        0.32,
        '(b) User information' if horizontal else '(b)',
        FONT_TITLE,
        INK,
        True,
    )
    x, y, w = PCA_LEFT, PCA_TOP, PCA_WIDTH
    h = s.height - PCA_BOTTOM_SPACE - y
    sx = w / (PCA_XLIM[1] - PCA_XLIM[0])
    sy = -h / (PCA_YLIM[1] - PCA_YLIM[0])

    def xy(px, py):
        return x + (px - PCA_XLIM[0]) * sx, y + (py - PCA_YLIM[1]) * sy

    # Background grid has no tick marks or numeric labels.
    for value in PCA_GRID_X:
        gx, _ = xy(value, 0)
        s.grid((gx, y), (gx, y + h))
    for value in PCA_GRID_Y:
        _, gy = xy(0, value)
        s.grid((x, gy), (x + w, gy))
    transform = np.diag([sx, sy])
    means = {}
    ellipses = []
    for key in PCA_ORDER:
        points = np.asarray(data['profiles'][key])
        values, vectors = np.linalg.eigh(transform @ np.cov(points, rowvar=False) @ transform.T)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        ew, eh = 2 * PCA_ELLIPSE_SD * np.sqrt(values)
        angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
        cx, cy = xy(*points.mean(axis=0))
        means[key] = (cx, cy)
        ellipses.append(
            (
                cx,
                cy,
                ew,
                eh,
                angle,
                PCA_COLORS[key],
                PCA_ELLIPSE_REAL_ALPHA if key == 'Real users' else PCA_ELLIPSE_MODEL_ALPHA,
                0.25 if key == 'Real users' else 0.68,
                0.7 if key == 'Real users' else 1.05,
            )
        )
    for key in PCA_ORDER:
        for index, point in enumerate(data['profiles'][key]):
            if key == 'Real users' and index == HIDDEN_REAL_INDEX:
                continue
            cx, cy = xy(*point)
            s.marker(
                cx,
                cy,
                PCA_MARKER_REAL if key == 'Real users' else PCA_MARKER_MODEL,
                'circle' if key == 'Real users' else 'triangle',
                PCA_COLORS[key],
                0.66 if key == 'Real users' else 0.88,
            )
    # Translucent ellipses are drawn over the points.
    for args in ellipses:
        s.ellipse(*args)
    # Only the real-user mean is marked, using every real profile.
    mx, my = means['Real users']
    s.mean(mx, my, PCA_COLORS['Real users'])
    s.text(0.78, 1.15, 0.88, 0.24, 'average', FONT_AVERAGE, INK)
    s.line((1.61, 1.37), (mx - 0.11, my - 0.045), INK, 1.0, arrow=True)
    s.line((x, y), (x, y + h), INK, LINEWIDTH_AXIS)
    s.line((x, y + h), (x + w, y + h), INK, LINEWIDTH_AXIS)
    s.text(x, y + h + 0.11, w, 0.23, 'PC1', 16, bold=True, align='center')
    s.text(0.19, y + h / 2 - 0.14, 0.22, 0.28, 'PC2', 16, bold=True, rotation=90)
    ly = s.height - 0.21
    for key, lx in zip(PCA_ORDER, PCA_LEGEND_X):
        s.marker(
            lx,
            ly,
            PCA_MARKER_LEGEND if key == 'Real users' else PCA_TRIANGLE_LEGEND,
            'circle' if key == 'Real users' else 'triangle',
            PCA_COLORS[key],
        )
        s.text(lx + 0.085, ly - 0.115, 1.1, 0.23, PCA_LABELS[key], FONT_LEGEND)


def arrow(s, x1, x2, y, color=WORKFLOW_STROKE):
    s.line((x1, y), (x2, y), WORKFLOW_STROKE, WORKFLOW_LINEWIDTH, arrow=True)


def condition(s, i, x, y, w, h=0.34):
    s.box(x, y, w, h, CONDITION_FILLS[i], CONDITION_BORDERS[i], radius=0.025, lw=WORKFLOW_LINEWIDTH)
    s.text(x, y, w, h, CONDITIONS[i], FONT_CONDITION, CONDITION_TEXT[i], True, 'center')


def audit(s, x, y, w, h):
    s.box(x, y, w, h, AT_FILL, AT_BORDER, radius=0.035, lw=WORKFLOW_LINEWIDTH)
    s.image(x + 0.10, y + 0.08, 0.51, 0.51)
    s.text(x + 0.69, y + 0.18, w - 0.77, 0.27, METHOD_NAME, FONT_METHOD, AT_COLOR, True)
    s.line((x + 0.12, y + 0.64), (x + w - 0.12, y + 0.64), AT_BORDER, WORKFLOW_LINEWIDTH)
    s.text(
        x + 0.10,
        y + 0.80,
        w - 0.20,
        0.45,
        'δ measures the advantage\nof target-user information',
        FONT_GAIN_EXPLANATION,
        AT_COLOR,
        align='center',
    )
    s.text(
        x + 0.10,
        y + 1.36,
        w - 0.20,
        0.38,
        'δ(I) = 𝕊(I) − 𝕊(Itarget)',
        FONT_DEFINITION,
        AT_COLOR,
        align='center',
        math=r'$\delta(I)=\mathbb{S}(I)-\mathbb{S}(I_{\mathrm{target}})$',
    )


def gain(s, i, x, y, w, h=0.405, matrix=False):
    s.box(x, y, w, h, CONDITION_FILLS[i], CONDITION_BORDERS[i], radius=0.025, lw=WORKFLOW_LINEWIDTH)
    if matrix:
        s.text(
            x + 0.03,
            y + 0.05,
            w - 0.06,
            0.30,
            'Δ' + GAINS[i],
            FONT_DELTA,
            CONDITION_TEXT[i],
            align='center',
            math=rf'$\Delta_{{\mathrm{{{GAINS[i]}}}}}$',
        )
        s.text(x + 0.03, y + 0.355, w - 0.06, 0.19, 'vs.', 15, CONDITION_TEXT[i], align='center')
        s.text(
            x + 0.03,
            y + 0.565,
            w - 0.06,
            0.23,
            GAIN_REFERENCES[i],
            16,
            CONDITION_TEXT[i],
            align='center',
        )
    else:
        s.text(
            x + 0.045,
            y,
            0.97,
            h,
            'Δ' + GAINS[i],
            FONT_DELTA_COMPACT if h < 0.35 else FONT_DELTA,
            CONDITION_TEXT[i],
            align='center',
            math=rf'$\Delta_{{\mathrm{{{GAINS[i]}}}}}$',
        )
        s.line(
            (x + 1.065, y + 0.07),
            (x + 1.065, y + h - 0.07),
            CONDITION_BORDERS[i],
            WORKFLOW_LINEWIDTH,
        )
        s.text(
            x + 1.20,
            y,
            w - 1.29,
            h,
            'vs. ' + GAIN_REFERENCES[i],
            FONT_GAIN_LABEL,
            CONDITION_TEXT[i],
        )


def datasets(s, y, h, at_bottom, at_x, gain_bottom, separate=False):
    # This entire band belongs to panel (b), including both role labels.
    s.line(
        (at_x, y - 0.025),
        (at_x, at_bottom + 0.035),
        WORKFLOW_STROKE,
        WORKFLOW_LINEWIDTH,
        arrow=True,
    )
    s.line(
        (13.71, gain_bottom + 0.025),
        (13.71, y - 0.025),
        WORKFLOW_STROKE,
        WORKFLOW_LINEWIDTH,
        arrow=True,
        dashed=True,
    )
    s.text(RIGHT_X, y - 0.25, 2.1, 0.22, 'Main trials', FONT_DATASET_GROUP, WORKFLOW_STROKE)
    s.text(11.59, y - 0.25, 2.08, 0.22, 'Simulator selection', FONT_DATASET_GROUP, WORKFLOW_STROKE)
    if not separate:
        s.box(RIGHT_X, y, 6.41, h, PAPER, WORKFLOW_STROKE, radius=0.025, lw=WORKFLOW_LINEWIDTH)
        s.box(11.53, y, 2.30, h, PAPER, WORKFLOW_STROKE, radius=0.025, lw=WORKFLOW_LINEWIDTH)
        for bx in [7.10, 9.35]:
            s.line((bx, y + 0.12), (bx, y + h - 0.12), WORKFLOW_STROKE, WORKFLOW_LINEWIDTH)
    for asset, title, channel, x, w in DATASETS:
        if separate:
            s.box(x, y, w - 0.055, h, PAPER, WORKFLOW_STROKE, radius=0.025, lw=WORKFLOW_LINEWIDTH)
        s.image(x + 0.045, y + (h - 0.53) / 2, 0.53, 0.53, asset)
        s.text(x + 0.64, y + 0.04, w - 0.70, 0.24, title, FONT_DATASET_TITLE, INK, True)
        s.text(x + 0.64, y + 0.30, w - 0.70, 0.40, channel, FONT_DATASET_CHANNEL, MUTED)


def columns(data):
    s = Scene('final')
    base(s, data)
    s.text(RIGHT_X, 0.335, 1.90, 0.25, 'User information', FONT_COLUMN, INK, True, 'left')
    for i in range(5):
        condition(s, i, RIGHT_X, 0.62 + i * 0.375, 1.60)
    audit(s, 6.87, 0.62, 3.05, 1.84)
    arrow(s, 6.61, 6.81, 1.54)
    arrow(s, 9.99, 10.22, 1.54)
    for i in range(4):
        gain(s, i, 10.28, 0.62 + i * 0.477, 3.55)
    datasets(s, 2.83, 0.70, 2.46, 8.395, 2.456)
    return s


def render(s, out_pdf):
    fig = plt.figure(figsize=(WIDTH, s.height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(s.height, 0))
    ax.set_axis_off()
    for q in s.items:
        k = q['kind']
        if k == 'text':
            xpos = q['x'] + {'left': 0, 'center': 0.5, 'right': 1}[q['align']] * q['w']
            if q['rotation']:
                xpos = q['x'] + q['w'] / 2
            ax.text(
                xpos,
                q['y'] + q['h'] / 2,
                q['math'] or q['text'],
                fontsize=q['size'],
                color=q['color'],
                fontweight='bold' if q['bold'] else 'normal',
                ha=q['align'] if not q['rotation'] else 'center',
                va='center',
                rotation=q['rotation'],
                linespacing=1.05,
                zorder=5,
            )
        elif k == 'box':
            ax.add_patch(
                FancyBboxPatch(
                    (q['x'], q['y']),
                    q['w'],
                    q['h'],
                    boxstyle=f'round,pad=0,rounding_size={q["radius"]}',
                    facecolor=q['fill'],
                    edgecolor=q['edge'] or 'none',
                    lw=q['lw'],
                    zorder=1,
                )
            )
        elif k in ('line', 'grid'):
            ax.add_patch(
                FancyArrowPatch(
                    q['a'],
                    q['b'],
                    arrowstyle='-|>' if q.get('arrow') else '-',
                    mutation_scale=10,
                    color=q['color'],
                    lw=q['lw'],
                    shrinkA=0,
                    shrinkB=0,
                    linestyle='--' if q.get('dashed') else '-',
                    zorder=0 if k == 'grid' else (6 if q['arrow'] else 2),
                )
            )
        elif k == 'ellipse':
            ax.add_patch(
                Ellipse(
                    (q['x'], q['y']),
                    q['w'],
                    q['h'],
                    angle=q['angle'],
                    facecolor=matplotlib.colors.to_rgba(q['color'], q['alpha']),
                    edgecolor=matplotlib.colors.to_rgba(q['color'], q['edge_alpha']),
                    lw=q['lw'],
                    zorder=4,
                )
            )
        elif k == 'mean':
            vertices = [
                (q['x'] + (vx - 0.5) * q['size'], q['y'] + (vy - 0.5) * q['size'])
                for vx, vy in PCA_MEAN_VERTICES
            ]
            ax.add_patch(
                Polygon(
                    vertices,
                    facecolor=q['color'],
                    edgecolor=PAPER,
                    lw=PCA_MEAN_OUTLINE,
                    joinstyle='round',
                    zorder=7,
                )
            )
        elif k == 'marker':
            x, y, z = q['x'], q['y'], q['size']
            kwargs = dict(
                facecolor=q['color'], edgecolor='white', lw=0.3, alpha=q['alpha'], zorder=3
            )
            if q['shape'] == 'circle':
                shape = Ellipse((x, y), z, z, **kwargs)
            elif q['shape'] == 'square':
                shape = Rectangle((x - z / 2, y - z / 2), z, z, **kwargs)
            else:
                shape = Polygon(
                    [(x, y - z / 2), (x - z / 2, y + z / 2), (x + z / 2, y + z / 2)], **kwargs
                )
            ax.add_patch(shape)
        elif k == 'image':
            ax.imshow(
                plt.imread(ASSETS / q['asset']),
                extent=(q['x'], q['x'] + q['w'], q['y'] + q['h'], q['y']),
                aspect='auto',
                interpolation='antialiased',
                zorder=3,
            )
    fig.savefig(out_pdf, dpi=DPI, facecolor='white')
    plt.close(fig)


def main():
    data = json.loads(DATA_FILE.read_text())
    paths.ensure(OUT_FILE.parent)
    render(columns(data), OUT_FILE)
    print(f'wrote {OUT_FILE}')


if __name__ == '__main__':
    main()
