"""Frozen-prompt Qwen3-8B on KT response under the nearest-neighbour imposter.

The imposter is the nearest other user in the registered prior-accuracy
tertile, z-scored train-split [log activity, train accuracy].  The frozen
prompt panel is a single deterministic evaluation of the untrained base model,
so the extra imposter arm is one pass.  Protocol (few-shot prefix, evaluation
subset, token cap, raw-completion scoring) is the frozen panel's
(audit.frozen_panel); the target and registered-imposter rows are read from
that panel's <AUDIT>/chess_kt/frozen_kt_8b.json downstream.

  python -m attribution_trials.validation.matcher_frozen
Output: <VALIDATION>/matchers/frozen_nn_8b.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from attribution_trials import paths  # noqa: E402
from attribution_trials.audit.sft_matrix import (  # noqa: E402
    MODELS,
    N_EVAL,
    build_for,
)
from attribution_trials.data.kt_csv import load_kt_csv  # noqa: E402
from attribution_trials.experiments.ec import session_split_indices  # noqa: E402

KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
OUT = paths.VALIDATION / "matchers"

SIZE = "8b"
CAP = 640
TRAIN_FRAC_KT = 0.7


def nn_map(cells: dict) -> dict:
    data = load_kt_csv(str(KT_DATA), n_students=500, min_responses=50)
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC_KT)
    feats = {}
    for traj, cut in zip(trajs, splits):
        correct = [1.0 if o.move == "correct" else 0.0 for o in traj.observations[:cut]]
        if correct:
            feats[traj.player_id] = np.array([math.log(len(correct)), float(np.mean(correct))])
    players = [p for p in cells if p in feats]
    matrix = np.array([feats[p] for p in players])
    centre, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (matrix - centre) / scale
    by_cell = {}
    for i, p in enumerate(players):
        by_cell.setdefault(cells[p], []).append(i)
    donor = {}
    for members in by_cell.values():
        block = z[members]
        distance = np.sqrt(((block[:, None, :] - block[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(distance, np.inf)
        for pos, i in enumerate(members):
            donor[players[i]] = players[members[int(distance[pos].argmin())]]
    return donor


def main() -> int:
    t0 = time.time()
    train, evals, cells, frozen_card_of = build_for("kt")
    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:N_EVAL]
    donor_nn = nn_map(cells)

    # few-shot prefix exactly as the frozen panel
    rng_fs = random.Random(7)
    fs_pool = rng_fs.sample(train, min(len(train), 8))
    K = 4
    fewshot = "".join(f"{ex['base']}\nAnswer: {ex['label']}\n\n" for ex in fs_pool[:K])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[SIZE])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[SIZE], torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    print(f"model loaded ({time.time() - t0:.0f}s)", flush=True)

    @torch.no_grad()
    def _nll(base, card, label):
        prompt = fewshot + base + ("\n" + card if card else "") + "\nAnswer:"
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        cid = tok(" " + label, add_special_tokens=False)["input_ids"]
        if len(pid) + len(cid) > CAP:
            pid = pid[-(CAP - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        return float(model(input_ids=ids, labels=labels).loss)

    def arm(card_of_row):
        out = []
        for r in evals:
            out.append(_nll(r["base"], card_of_row(r), r["label"]))
        return np.array(out, dtype=np.float64)

    a3_nn = arm(lambda r: frozen_card_of[donor_nn[r["player"]]])
    print(f"A3 NN scored ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "cell": f"persona-frozen|kt ({SIZE})|response",
        "matcher": "nearest-neighbour, z-scored train-split "
        "[log activity, train accuracy], within registered cell",
        "row_players": [str(r["player"]) for r in evals],
        "nll": {"a3_nn": a3_nn.tolist()},
        "donor_nn": {str(p): str(d) for p, d in donor_nn.items()},
    }
    path = paths.ensure(OUT) / f"frozen_nn_{SIZE}.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
