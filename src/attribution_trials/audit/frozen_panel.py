"""Frozen prompt panel (no training).

Same profile cards (none/placebo/group/self + imposter + frozen), same eval
set and per-user pairing as the user-profile LoRA arms, but scored on the
untrained base model, so one model load covers every arm (only the card
in the prompt changes). Eval-only.

Arms:
  A0 = no card; A1 = placebo card; A2 = group card;
  A3 = same-cell imposter's frozen card; A4 = own live card;
  A4frozen = own frozen card (the target).

  python -m attribution_trials.audit.frozen_panel <substrate> <1.7b|8b|30b>
Output: <AUDIT>/{chess_kt,opera}/frozen_<substrate>_<size>.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials.audit.sft_matrix import (
    MODELS,
    N_EVAL,
    build_for,
    imposter_map,
    out_dir,
)


def main() -> int:
    substrate, size = sys.argv[1], sys.argv[2]
    t0 = time.time()
    # OPeRA prompts (persona + running-state card + page context) are
    # longer than the KT/chess ones -> larger token cap.
    CAP = 1200 if substrate.startswith("opera") else 640

    train, evals, cells, frozen_card_of = build_for(substrate)
    import random

    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:N_EVAL]
    imp = imposter_map(cells, 0)

    # Few-shot format calibration: a fixed K-shot prefix of POPULATION train
    # examples (no cards -> no per-user leakage; identical for every arm and
    # eval item) establishes the label space, so the card's marginal effect
    # is what is measured.
    rng_fs = random.Random(7)
    fs_pool = rng_fs.sample(train, min(len(train), 8))
    K = 4
    fewshot = "".join(f"{ex['base']}\nAnswer: {ex['label']}\n\n" for ex in fs_pool[:K])

    tok = AutoTokenizer.from_pretrained(MODELS[size])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[size], torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    def _prompt_ids(prompt):
        # Raw few-shot completion, not the chat template.
        return tok(prompt, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def _nll(base, card, label):
        prompt = fewshot + base + ("\n" + card if card else "") + "\nAnswer:"
        pid = _prompt_ids(prompt)
        cid = tok(" " + label, add_special_tokens=False)["input_ids"]
        # Longer inputs are truncated from the front to CAP tokens.
        if len(pid) + len(cid) > CAP:
            pid = pid[-(CAP - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        return float(model(input_ids=ids, labels=labels).loss)

    arm_card = {
        "A0": lambda r: None,
        "A1": lambda r: r["placebo_card"],
        "A2": lambda r: r["group_card"],
        "A3": lambda r: frozen_card_of[imp[r["player"]]],
        "A4": lambda r: r["self_card"],
        "A4frozen": lambda r: r["frozen_card"],
    }
    nll = {a: [] for a in arm_card}
    for a, cf in arm_card.items():
        for r in evals:
            nll[a].append(_nll(r["base"], cf(r), r["label"]))
        print(f"  [{substrate}/{size}] arm {a} scored ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "family": "persona-prompt-frozen",
        "substrate": substrate,
        "size": size,
        "model": MODELS[size],
        "n_eval": len(evals),
        "players": [r["player"] for r in evals],
        "labels": [r["label"] for r in evals],
        "nll": nll,
        "mean_nll": {a: sum(v) / len(v) for a, v in nll.items()},
        "wall_seconds": time.time() - t0,
    }
    out = out_dir(substrate) / f"frozen_{substrate}_{size}.json"
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(
        f"FROZEN PANEL {substrate}/{size} mean_nll={payload['mean_nll']} "
        f"({payload['wall_seconds']:.0f}s) -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
