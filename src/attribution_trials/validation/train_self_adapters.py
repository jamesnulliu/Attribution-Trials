"""Train and save the user-profile LoRA `self` adapter for Qwen3-8B on KT.

Same recipe as the audit's `self` condition in sft_matrix (LoRA r=16 on every
projection, AdamW lr 1e-4, 2 epochs, gradient accumulation 16, the user's own
causal card), saved so the closer-imposter scoring (matcher_lora) can reuse it.
bf16 LoRA training is not bit-reproducible.

  python -m attribution_trials.validation.train_self_adapters <seed>
Output: <VALIDATION>/self_adapters/self_kt_8b_s<seed>/
"""

from __future__ import annotations

import os
import random
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from attribution_trials import paths  # noqa: E402
from attribution_trials.audit.sft_matrix import (  # noqa: E402
    MODELS,
    N_TRAIN,
    build_for,
    card_for,
)

ADAPTERS = paths.VALIDATION / "self_adapters"

SIZE = "8b"
MAX_LEN = 320


def main() -> int:
    seed = int(sys.argv[1])
    t0 = time.time()
    train, evals, cells, frozen_card_of = build_for("kt")
    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(evals)
    train = train[:N_TRAIN]

    adapter_dir = paths.ensure(ADAPTERS) / f"self_kt_{SIZE}_s{seed}"
    if (adapter_dir / "adapter_config.json").exists():
        print(f"[s{seed}] adapter exists: {adapter_dir}", flush=True)
        return 0

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELS[SIZE])

    def _prompt_ids(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    torch.manual_seed(seed)
    random.seed(seed)
    base = AutoModelForCausalLM.from_pretrained(
        MODELS[SIZE], torch_dtype=torch.bfloat16, device_map="cuda"
    )

    def _nll(model, base_txt, card, label):
        prompt = base_txt + ("\n" + card if card else "") + "\nAnswer:"
        pid = _prompt_ids(prompt)
        cid = tok(" " + label, add_special_tokens=False)["input_ids"]
        if len(pid) + len(cid) > MAX_LEN:
            pid = pid[-(MAX_LEN - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        return model(input_ids=ids, labels=labels).loss

    model = get_peft_model(
        base,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    accum = 16
    for ep in range(2):
        random.shuffle(train)
        opt.zero_grad()
        for i, rec in enumerate(train):
            loss = _nll(model, rec["base"], card_for("self", rec), rec["label"])
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            if (i + 1) % 1000 == 0:
                print(
                    f"  [s{seed}] ep{ep} {i + 1}/{len(train)} ({time.time() - t0:.0f}s)", flush=True
                )
        opt.step()
        opt.zero_grad()
    model.eval()
    model.save_pretrained(str(adapter_dir))
    print(f"[s{seed}] adapter saved -> {adapter_dir} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
