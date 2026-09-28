"""User-profile LoRA 8B on KT response under the nearest-neighbour imposter.

The imposter is the nearest other user inside the registered prior-accuracy
tertile, in z-scored Euclidean distance over training-split features
(log activity, train accuracy).  Nothing about the evaluation split enters the
choice.  The contrast holds the profile fixed at the training boundary: the
user's own frozen card (A4frozen) against the donor's.  Everything is
inference on the saved `self` adapter (train_self_adapters); the registered
imposter (A3) is scored in the same process.

  python -m attribution_trials.validation.matcher_lora <seed>
Output: <VALIDATION>/matchers/lora_nn_s<seed>.json
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from attribution_trials import paths  # noqa: E402
from attribution_trials.audit.sft_matrix import (  # noqa: E402
    MODELS,
    N_EVAL,
    N_TRAIN,
    build_for,
    imposter_map,
)
from attribution_trials.data.kt_csv import load_kt_csv  # noqa: E402
from attribution_trials.experiments.ec import session_split_indices  # noqa: E402

ADAPTERS = paths.VALIDATION / "self_adapters"
KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"
OUT = paths.VALIDATION / "matchers"

SIZE = "8b"
MAX_LEN = 320
TRAIN_FRAC_KT = 0.7


def nn_map(cells: dict) -> dict:
    """Nearest other user in the registered cell, z-scored train features."""
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
    seed = int(sys.argv[1])
    t0 = time.time()

    train, evals, cells, frozen_card_of = build_for("kt")
    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(evals)
    train, evals = train[:N_TRAIN], evals[:N_EVAL]
    row_players = [str(r["player"]) for r in evals]

    donor_reg = imposter_map(cells, seed)
    donor_nn = nn_map(cells)
    frozen = {str(p): c for p, c in frozen_card_of.items()}
    same = sum(1 for p in donor_nn if str(donor_nn[p]) == str(donor_reg.get(p)))
    print(
        f"[s{seed}] NN map built; agrees with registered map on {same}/{len(donor_nn)} users",
        flush=True,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[SIZE])

    def prompt_ids(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    adapter = ADAPTERS / f"self_kt_{SIZE}_s{seed}"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"no saved adapter at {adapter}")
    base = AutoModelForCausalLM.from_pretrained(
        MODELS[SIZE], torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    print(f"[s{seed}] adapter loaded ({time.time() - t0:.0f}s)", flush=True)

    cache: dict[tuple[int, str], float] = {}

    @torch.no_grad()
    def score(index, card):
        key = (index, card)
        hit = cache.get(key)
        if hit is not None:
            return hit
        record = evals[index]
        prompt = record["base"] + ("\n" + card if card else "") + "\nAnswer:"
        pid = prompt_ids(prompt)
        cid = tok(" " + record["label"], add_special_tokens=False)["input_ids"]
        if len(pid) + len(cid) > MAX_LEN:
            pid = pid[-(MAX_LEN - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        value = float(model(input_ids=ids, labels=labels).loss)
        cache[key] = value
        return value

    def arm(card_of_player):
        return np.array(
            [score(i, card_of_player(row_players[i])) for i in range(len(evals))], dtype=np.float64
        )

    a4f = arm(lambda p: frozen[p])
    print(f"[s{seed}] A4frozen scored ({time.time() - t0:.0f}s)", flush=True)
    a3_reg = arm(lambda p: frozen[str(donor_reg[p])])
    print(f"[s{seed}] A3 registered scored ({time.time() - t0:.0f}s)", flush=True)
    a3_nn = arm(lambda p: frozen[str(donor_nn[p])])
    print(f"[s{seed}] A3 NN scored ({time.time() - t0:.0f}s)", flush=True)

    out = {
        "cell": f"sft-lora|kt ({SIZE})|response",
        "seed": seed,
        "matcher": "nearest-neighbour, z-scored train-split "
        "[log activity, train accuracy], within registered cell",
        "nn_agrees_with_registered": same,
        "row_players": row_players,
        "nll": {
            "a4frozen": a4f.tolist(),
            "a3_registered": a3_reg.tolist(),
            "a3_nn": a3_nn.tolist(),
        },
        "donor_nn": {str(p): str(d) for p, d in donor_nn.items()},
    }
    path = paths.ensure(OUT) / f"lora_nn_s{seed}.json"
    path.write_text(json.dumps(out))
    print(f"[s{seed}] wrote {path} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
