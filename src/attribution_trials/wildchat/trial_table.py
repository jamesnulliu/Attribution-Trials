"""Per-simulator attribution-trial gains on WildChat with user-bootstrap intervals.

For each of the seven simulators: dind (A4 - A3) and dtot (A4 - A0) per user,
then the mean over the 240 users with a 10,000-resample user bootstrap
(seed 0).  All seven score files must cover identical rows.  Writes
``paths.WILDCHAT / "trial_table.json"``.

  python -m attribution_trials.wildchat.trial_table
"""

from __future__ import annotations

import json

import numpy as np

from attribution_trials import paths

NBOOT = 10_000
SEED = 0
MAIN = ("1.7b", "8b", "dense", "humanlike", "coser", "osim", "humanlm")
FILES = {
    "1.7b": "scale_1.7b.json",
    "8b": "scale_8b.json",
    "dense": "scale_32b.json",
    "humanlike": "scale_humanlike.json",
    "coser": "scale_coser.json",
    "osim": "scale_osim.json",
    "humanlm": "scale_humanlm.json",
}


def bootstrap(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return {
        "n_users": int(len(values)),
        "point": float(values.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def per_user(data: dict, arm_left: str, arm_right: str) -> dict[str, float]:
    if arm_left == "A4" and arm_right == "A3" and data.get("per_user_delta_indiv"):
        return {u: float(v) for u, v in data["per_user_delta_indiv"].items()}
    if arm_left == "A4" and arm_right == "A0" and data.get("per_user_delta_total"):
        return {u: float(v) for u, v in data["per_user_delta_total"].items()}
    users = data.get("users") or sorted(set(data["players"]))
    by_user = {u: {"A0": [], "A3": [], "A4": []} for u in users}
    if "rows" in data:
        for row in data["rows"]:
            by_user[row["user_id"]][arm_left].append(float(row["nll"][arm_left]))
            by_user[row["user_id"]][arm_right].append(float(row["nll"][arm_right]))
    else:
        # Columnar schema written by score_trial_frozen.
        for index, user in enumerate(data["players"]):
            by_user[user][arm_left].append(float(data["nll"][arm_left][index]))
            by_user[user][arm_right].append(float(data["nll"][arm_right][index]))
    return {u: float(np.mean(by_user[u][arm_left]) - np.mean(by_user[u][arm_right])) for u in users}


def main() -> None:
    data = {}
    expected = json.loads((paths.WILDCHAT / FILES["8b"]).read_text())
    expected_keys = [
        f"{uid}|{key[0]}|{int(key[1])}" for uid, key in zip(expected["players"], expected["keys"])
    ]
    expected_digest = expected["row_digest"]
    for candidate in MAIN:
        path = paths.WILDCHAT / FILES[candidate]
        if not path.exists():
            raise FileNotFoundError(path)
        node = json.loads(path.read_text())
        if "rows" in node:
            keys = [row["key"] for row in node["rows"]]
        else:
            keys = [
                f"{uid}|{key[0]}|{int(key[1])}" for uid, key in zip(node["players"], node["keys"])
            ]
        if keys != expected_keys or node.get("row_digest") != expected_digest:
            raise AssertionError(f"{candidate}: scored rows differ from the Qwen3-8B file")
        data[candidate] = node

    d_indiv = {c: per_user(data[c], "A4", "A3") for c in MAIN}
    d_total = {c: per_user(data[c], "A4", "A0") for c in MAIN}
    users = expected.get("users") or sorted(set(expected["players"]))
    summary = {}
    for candidate in MAIN:
        summary[candidate] = {
            "delta_indiv": bootstrap(np.asarray([d_indiv[candidate][u] for u in users])),
            "delta_total": bootstrap(np.asarray([d_total[candidate][u] for u in users])),
            "model": data[candidate].get("model"),
            "revision": data[candidate].get("model_revision"),
            "interface": data[candidate].get("interface", "frozen_qwen3"),
        }

    out = {
        "schema": "wildchat-trial-table-v1",
        "n_bootstrap": NBOOT,
        "seed": SEED,
        "n_users": len(users),
        "n_rows": len(expected_keys),
        "row_digest": expected_digest,
        "candidates": list(MAIN),
        "candidate_summary": summary,
    }
    paths.ensure(paths.WILDCHAT)
    path = paths.WILDCHAT / "trial_table.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(expected_keys),
                "users": len(users),
                "candidates": list(MAIN),
                "output": str(path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
