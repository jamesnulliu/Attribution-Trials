"""Dump the train-only per-user discrete marginals behind the residual.

For chess move and KT response (both KT legs), every user's held-out NLL under
their own training-split marginal (`own`) and under their within-cell
imposter's (`random_other`), from `floor_anchors`.

    python -m attribution_trials.comparison.marginal_discrete
Output: <COMPARISON>/marginal/marginal_per_user.json,
  {domain: {channel: {"users", "n_users", "per_user_nll": {"own", "random_other"},
                      "random_pair"}}}
"""

from __future__ import annotations

import json

from attribution_trials import paths
from attribution_trials.comparison import floor_anchors


def rebuild_marginals() -> dict:
    return {
        "chess-pooled": {"move": floor_anchors.chess_move()},
        "kt": {"response": floor_anchors.kt_response("kt", None)},
        "kt-rt": {"response": floor_anchors.kt_response("kt-rt", 5)},
    }


def main() -> None:
    out = rebuild_marginals()
    for domain, channels in out.items():
        for channel, node in channels.items():
            for row, values in node["per_user_nll"].items():
                assert len(values) == len(node["users"]), f"{domain}/{channel}/{row}"
    path = paths.ensure(paths.COMPARISON / "marginal") / "marginal_per_user.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path}")
    for domain, channels in out.items():
        for channel, node in channels.items():
            print(f"  {domain:14s} {channel:9s} n={node['n_users']:4d}")


if __name__ == "__main__":
    main()
