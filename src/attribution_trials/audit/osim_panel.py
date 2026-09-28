"""Osim released simulators on the OPeRA frozen prompt panel.

The frozen prompt panel protocol (``audit.frozen_panel``) with the model
replaced by an Osim checkpoint: same eval rows and shuffle (Random(0), capped
at N_EVAL), same 4-shot population prefix (Random(7)), same imposter map
(seed 0), same 1200-token cap, same six conditions, NLL of the label tokens.
The prompt is the raw few-shot completion (no chat template).

  python -m attribution_trials.audit.osim_panel <opera-substrate> <model-tag>
  model-tag in {osim-4b, osim-4b-mid, osim-8b, osim-8b-mid}

Output: <AUDIT>/released/osim-panel_<channel>-<config>_<model-tag>.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths
from attribution_trials.audit.sft_matrix import N_EVAL, build_for, imposter_map

OSIM = {
    "osim-8b": "cmu-lti/osim-8b",
    "osim-8b-mid": "cmu-lti/osim-8b-mid",
    "osim-4b": "cmu-lti/osim-4b",
    "osim-4b-mid": "cmu-lti/osim-4b-mid",
}
CAP = 1200


def main() -> int:
    substrate, tag = sys.argv[1], sys.argv[2]
    assert substrate.startswith("opera"), substrate
    model_name = OSIM[tag]
    t0 = time.time()

    train, evals, cells, frozen_card_of = build_for(substrate)
    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:N_EVAL]
    imp = imposter_map(cells, 0)

    rng_fs = random.Random(7)
    fs_pool = rng_fs.sample(train, min(len(train), 8))
    K = 4
    fewshot = "".join(f"{ex['base']}\nAnswer: {ex['label']}\n\n" for ex in fs_pool[:K])

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    @torch.no_grad()
    def _score(pid, cid):
        if len(pid) + len(cid) > CAP:
            pid = pid[-(CAP - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        return float(model(input_ids=ids, labels=labels).loss)

    def _nll(base, card, label):
        prompt = fewshot + base + ("\n" + card if card else "") + "\nAnswer:"
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        cid = tok(" " + label, add_special_tokens=False)["input_ids"]
        return _score(pid, cid)

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
        print(f"  [{substrate}/{tag}] arm {a} scored ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "family": "trained-simulator-frozen",
        "substrate": substrate,
        "size": tag,
        "model": model_name,
        "n_eval": len(evals),
        "players": [r["player"] for r in evals],
        "labels": [r["label"] for r in evals],
        "nll": nll,
        "mean_nll": {a: sum(v) / len(v) for a, v in nll.items()},
        "wall_seconds": time.time() - t0,
    }
    chcfg = substrate[len("opera-") :]
    out = paths.ensure(paths.AUDIT / "released") / f"osim-panel_{chcfg}_{tag}.json"
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(f"osim panel {substrate}/{tag} ({payload['wall_seconds']:.0f}s) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
