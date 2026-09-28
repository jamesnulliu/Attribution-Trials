"""Generate tables/marginal_share.tex (tab:marginal).

Source: <COMPARISON>/marginal/marginal_controls.json.  Per row: dind = the
target-versus-imposter gain with the profile fixed at the training boundary;
residual = residual_point_final; marginal-only = dind - residual (the per-user
marginal's own gain); capture = marginal-only / dind, n.d. where the dind
interval covers zero.  Point estimates only.

    python -m attribution_trials.tables.marginal_share
Output: <TABLES>/marginal_share.tex
"""

import json

from attribution_trials import paths


def fmt(x):
    return f"${x:+.4f}$"


def main() -> None:
    cells = json.loads((paths.COMPARISON / 'marginal' / 'marginal_controls.json').read_text())[
        'cells'
    ]

    rows = []
    for c in cells:
        # marginal_controls.json stores target-minus-imposter surprisal; the paper prints imposter minus target
        d = -c['dind']['point']
        r = -c['residual_point_final']
        m = d - r
        cap = None if c['dind_nd'] else m / d
        rows.append((c['display'], d, m, r, cap))
    rows.sort(key=lambda t: (t[4] is None, -(t[4] or 0)))

    lines = [
        r"\begin{table}[t]",
        r"\caption{Marginal decomposition of all 17 complete five-condition combinations, point estimates in nats with profiles fixed at the training boundary.",
        r"`marginal-only' is the target-versus-imposter gain of a per-user marginal model fitted on the training split alone; `residual' is the method's $\dind$ minus it, and negative means the marginal model matches or exceeds the method.",
        r"\emph{Capture} is marginal-only over method: $1.00$ means the marginal reproduces the whole of $\dind$, and above $1.00$ it beats the method.",
        r"Combinations whose $\dind$ interval covers zero have no stable ratio and are marked n.d.; positive $\dind$ favors the target.}",
        r"\label{tab:marginal}",
        r"\begin{center}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r"combination & $\dind$ & marginal-only & residual & capture \\",
        r"\midrule",
    ]
    for disp, d, m, r, cap in rows:
        capt = 'n.d.' if cap is None else f"${cap:.2f}$"
        lines.append(f"{disp} & {fmt(d)} & {fmt(m)} & {fmt(r)} & {capt} \\\\")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{center}", r"\end{table}", ""]
    out = paths.ensure(paths.TABLES) / 'marginal_share.tex'
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
