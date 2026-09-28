"""Released simulators on the OPeRA frozen prompt panel.

  coser-8b      Neph0s/CoSER-Llama-3.1-8B                (role-play SFT, Llama-3.1)
  humanlike-7b  HumanLLMs/Human-Like-Qwen2.5-7B-Instruct (DPO, Qwen2.5)

These are cross-family rows: the gains are within-model paired deltas, so
they share an axis with the other rows; absolute NLL levels are not
comparable across tokenizers.

Protocol constants come from ``attribution_trials.audit.sft_matrix`` and
match ``frozen_panel``: same few-shot prefix (Random(7).sample(train,8)[:4]),
same eval shuffle (Random(0), cap N_EVAL=2000), same imposter derangement
(seed 0), same CAP=1200, same arms A0/A1/A2/A3/A4/A4frozen, same
NLL-on-label scoring, single deterministic run (no seeds).

Chat format: card in the system prompt, few-shot + question in the user
turn, label scored as the assistant turn.

Usage:
  python -m attribution_trials.audit.released_panel <opera-substrate> <model-tag>
Output: <AUDIT>/released/trio-panel_<channel>-<config>_<model-tag>_chat.json
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths
from attribution_trials.audit.sft_matrix import N_EVAL, build_for, imposter_map

SIMULATORS = {
    "coser-8b": "Neph0s/CoSER-Llama-3.1-8B",
    "humanlike-7b": "HumanLLMs/Human-Like-Qwen2.5-7B-Instruct",
}
REVISIONS = {
    "coser-8b": "80d5c88072599026bd0b0e2eb4369a87e6a5c960",
    "humanlike-7b": "7cab6062ab32fa984f51d4bf0254472b2b320362",
}
FORMAT = "chat"


def main() -> int:
    substrate, tag = sys.argv[1], sys.argv[2]
    assert substrate.startswith("opera"), substrate
    model_name = SIMULATORS[tag]
    revision = REVISIONS[tag]
    t0 = time.time()
    CAP = 1200  # opera cap, as in frozen_panel.py

    train, evals, cells, frozen_card_of = build_for(substrate)
    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:N_EVAL]
    imp = imposter_map(cells, 0)

    rng_fs = random.Random(7)
    fs_pool = rng_fs.sample(train, min(len(train), 8))
    K = 4
    fewshot = "".join(f"{ex['base']}\nAnswer: {ex['label']}\n\n" for ex in fs_pool[:K])

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    template_sha256 = hashlib.sha256((tok.chat_template or "").encode("utf-8")).hexdigest()
    tokenizer_sha256 = hashlib.sha256(tok.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    truncation_count = 0

    @torch.no_grad()
    def _score(pid, cid):
        nonlocal truncation_count
        if len(pid) + len(cid) > CAP:
            truncation_count += 1
            pid = pid[-(CAP - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        return float(model(input_ids=ids, labels=labels).loss)

    def _nll(base, card, label):
        sys_p = ("You are this shopper: " + card) if card else "You are a shopper browsing Amazon."
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": fewshot + base + "\nAnswer:"},
        ]
        text = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pid = tok(text, add_special_tokens=False)["input_ids"]
        cid = tok(label, add_special_tokens=False)["input_ids"]
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
        print(
            f"  [{substrate}/{tag}/{FORMAT}] arm {a} scored ({time.time() - t0:.0f}s)", flush=True
        )

    payload = {
        "family": "sim2real-trio",
        "substrate": substrate,
        "size": tag,
        "model": model_name,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "tokenizer_sha256": tokenizer_sha256,
        "chat_template_sha256": template_sha256,
        "format": FORMAT,
        "study_status": "new-dataset-port",
        "dtype": "bfloat16",
        "cap_tokens": CAP,
        "truncation_count": truncation_count,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "n_eval": len(evals),
        "players": [r["player"] for r in evals],
        "labels": [r["label"] for r in evals],
        "nll": nll,
        "mean_nll": {a: sum(v) / len(v) for a, v in nll.items()},
        "wall_seconds": time.time() - t0,
    }
    chcfg = substrate[len("opera-") :]  # "<channel>-<config>"
    out = paths.ensure(paths.AUDIT / "released") / f"trio-panel_{chcfg}_{tag}_{FORMAT}.json"
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(
        f"RELEASED PANEL {substrate}/{tag}/{FORMAT} "
        f"mean_nll={payload['mean_nll']} "
        f"({payload['wall_seconds']:.0f}s) -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
