"""Attribution-trial scores for one simulator on the WildChat panel.

Teacher-forced next-turn NLL under no card (A0), the imposter's card (A3) and
the target's card (A4) on the primary out-of-distribution turns, through the
candidate's own prompt interface (``common.build_context``).  Used for
Qwen3-32B, HumanLike-7B, CoSER-8B, Osim-8B and HumanLM-opinion; writes
``paths.WILDCHAT / "scale_<size>.json"``.

  python -m attribution_trials.wildchat.score_trial humanlike
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
import transformers

from attribution_trials import paths
from attribution_trials.wildchat.common import (
    ARMS,
    CANDIDATES,
    PANEL_RUNTIME,
    TRIAL_CANDIDATES,
    build_context,
    card_for,
    load_local_model,
    row_key,
    score_candidate_target,
)

NBOOT = 10_000
SEED = 0


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


def output_path(candidate: str):
    size = "32b" if candidate == "dense" else candidate
    return paths.WILDCHAT / f"scale_{size}.json"


def metadata_digest(rows: list[dict]) -> str:
    """Same digest definition as score_trial_frozen.row_digest."""
    metadata = [
        {
            "user_id": row["user_id"],
            "conv_id": row["conv_id"],
            "turn_index": int(row["turn_index"]),
            "split": row["split"],
            "cluster_id": row["cluster_id"],
        }
        for row in rows
    ]
    blob = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=TRIAL_CANDIDATES)
    args = parser.parse_args()
    candidate = args.candidate
    panel = json.loads(PANEL_RUNTIME.read_text())
    users = panel["users"]
    selected = sorted(users)

    rows = []
    per_user = defaultdict(lambda: defaultdict(list))
    primary_decisions = {
        uid: [d for d in users[uid]["decisions"] if d.get("split") == "primary"] for uid in selected
    }
    total = sum(len(primary_decisions[uid]) for uid in selected)

    started = time.time()
    tok, model = load_local_model(candidate)
    done = 0
    for uid in selected:
        imposter = panel["imposter"][uid]
        if users[imposter]["profile_card"] == users[uid]["profile_card"]:
            raise AssertionError(f"A3/A4 cards collide for {uid}")
        cards = {arm: card_for(panel, uid, arm) for arm in ARMS}
        for decision in primary_decisions[uid]:
            key = row_key(uid, decision)
            scores = {}
            target_tokens = None
            for arm in ARMS:
                prompt, interface_meta = build_context(
                    tok, candidate, cards[arm], decision["prefix"]
                )
                value, nt = score_candidate_target(
                    model, tok, candidate, prompt, decision["target"]
                )
                if not math.isfinite(value):
                    raise FloatingPointError(f"non-finite NLL at {key}/{arm}")
                scores[arm] = value
                target_tokens = nt if target_tokens is None else target_tokens
            rows.append(
                {
                    "key": key,
                    "user_id": uid,
                    "conv_id": decision["conv_id"],
                    "turn_index": int(decision["turn_index"]),
                    "split": decision["split"],
                    "cluster_id": decision["cluster_id"],
                    "target_tokens": int(target_tokens),
                    "nll": scores,
                }
            )
            for arm in ARMS:
                per_user[uid][arm].append(scores[arm])
            done += 1
            if done % 100 == 0 or done == total:
                print(f"[{candidate}] {done}/{total} decisions", flush=True)

    user_ids = selected
    d_indiv = np.asarray(
        [np.mean(per_user[uid]["A4"]) - np.mean(per_user[uid]["A3"]) for uid in user_ids]
    )
    d_total = np.asarray(
        [np.mean(per_user[uid]["A4"]) - np.mean(per_user[uid]["A0"]) for uid in user_ids]
    )
    spec = CANDIDATES[candidate]
    out = {
        "schema": "wildchat-trial-nll-v1",
        "substrate": "wildchat-ood",
        "candidate": candidate,
        "size": "32b" if candidate == "dense" else candidate,
        "model": spec["model"],
        "model_revision": spec["revision"],
        "interface": spec["interface"],
        "panel": "240-user WildChat panel",
        "panel_revision": panel["meta"]["revision"],
        "n_users": len(user_ids),
        "n_decisions": len(rows),
        "arms": list(ARMS),
        "row_digest": metadata_digest(rows),
        "users": user_ids,
        "per_user_delta_indiv": {uid: float(d_indiv[i]) for i, uid in enumerate(user_ids)},
        "per_user_delta_total": {uid: float(d_total[i]) for i, uid in enumerate(user_ids)},
        "delta_indiv": bootstrap(d_indiv),
        "delta_total": bootstrap(d_total),
        "mean_nll": {arm: float(np.mean([row["nll"][arm] for row in rows])) for arm in ARMS},
        "runtime": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "gpu": torch.cuda.get_device_name(0),
            "interface_note": (
                "Qwen3/Osim use the Qwen3 text serialization; HumanLike/CoSER "
                "use native chat persona system slot; HumanLM uses native "
                "persona/task prompt with an empty think block and response "
                "wrapper for teacher-forced target likelihood."
            ),
            "wall_seconds": time.time() - started,
        },
        "rows": rows,
    }
    paths.ensure(paths.WILDCHAT)
    path = output_path(candidate)
    tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(out) + "\n")
    tmp.replace(path)
    print(f"[{candidate}] wrote {path}", flush=True)
    print(
        json.dumps(
            {
                "candidate": candidate,
                "delta_indiv": out["delta_indiv"],
                "delta_total": out["delta_total"],
                "rows": len(rows),
                "wall_seconds": out["runtime"]["wall_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
