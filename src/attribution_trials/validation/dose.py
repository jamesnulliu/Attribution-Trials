"""Target-information dose: mix the target's fixed card into the imposter's.

Frozen-prompt Qwen3-8B.  For each evaluation row the target's boundary-frozen
card and the matched imposter's frozen card are split into fields, and each
field is taken from the target with probability alpha (a keyed draw per
substrate, player, seed and field).  alpha = 0 rebuilds the imposter's card
and alpha = 1 the target's, byte for byte, so the endpoints are the frozen
panel's A3 and A4frozen rows.  The evaluation stream, 4-shot prefix, token
cap 640 and raw-completion NLL are the frozen panel's.

One process per substrate loads the model once and runs the two endpoint
passes plus alpha in {0.25, 0.5, 0.75} x 3 seeds.

  python -m attribution_trials.validation.dose <kt|chess>
Output: <VALIDATION>/dose/{endpoint_<substrate>_8b_<A3|A4frozen>.npy,
        mix_fixed_<substrate>_8b_a<25|50|75>_s<seed>.json}
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths
from attribution_trials.audit.sft_matrix import MODELS, N_EVAL, build_for, imposter_map

OUT = paths.VALIDATION / "dose"
ALPHAS = (0.25, 0.5, 0.75)
SEEDS = (0, 1, 2)
CAP = 640


def card_fields(card: str) -> tuple[list[str], str]:
    separator = "; " if "; " in card else ", "
    return card.split(separator), separator


def mix_card(target, imposter, alpha, substrate, player, seed):
    target_fields, target_sep = card_fields(target)
    imp_fields, imp_sep = card_fields(imposter)
    if len(target_fields) != len(imp_fields) or target_sep != imp_sep:
        raise AssertionError(
            "A3/A4 card schema mismatch: "
            f"target={len(target_fields)}{target_sep!r}, "
            f"imposter={len(imp_fields)}{imp_sep!r}"
        )
    if alpha == 0.0:
        return imposter, 0.0
    if alpha == 1.0:
        return target, 1.0
    chosen, target_count = [], 0
    for k, (tf, imf) in enumerate(zip(target_fields, imp_fields)):
        # keyed per-field draw; the key string is the seed
        take = random.Random(f"e34v2:{substrate}:{player}:{seed}:{k}").random() < alpha
        chosen.append(tf if take else imf)
        target_count += int(take)
    return target_sep.join(chosen), target_count / len(chosen)


def prompt_text(fewshot, base, card):
    return fewshot + base + ("\n" + card if card else "") + "\nAnswer:"


def main() -> int:
    substrate = sys.argv[1]
    out_dir = paths.ensure(OUT)
    t0 = time.time()

    train, evals, cells, frozen_card_of = build_for(substrate)
    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:N_EVAL]
    imp = imposter_map(cells, 0)
    rng_fs = random.Random(7)
    fs_pool = rng_fs.sample(train, min(len(train), 8))
    fewshot = "".join(f"{ex['base']}\nAnswer: {ex['label']}\n\n" for ex in fs_pool[:4])

    # the target side of the mix is the boundary-frozen card
    pairs = [(r, r["frozen_card"], frozen_card_of[imp[r["player"]]]) for r in evals]

    for r, target, imposter_card in pairs:  # endpoints rebuild exactly
        m0, f0 = mix_card(target, imposter_card, 0.0, substrate, r["player"], 0)
        m1, f1 = mix_card(target, imposter_card, 1.0, substrate, r["player"], 0)
        assert prompt_text(fewshot, r["base"], m0) == prompt_text(fewshot, r["base"], imposter_card)
        assert prompt_text(fewshot, r["base"], m1) == prompt_text(fewshot, r["base"], target)
        assert f0 == 0.0 and f1 == 1.0
    print(f"endpoint cards rebuild exactly: substrate={substrate} rows={len(pairs)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODELS["8b"])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS["8b"], dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    @torch.no_grad()
    def nll(base, card, label):
        pid = tok(prompt_text(fewshot, base, card), add_special_tokens=False)["input_ids"]
        cid = tok(" " + label, add_special_tokens=False)["input_ids"]
        if len(pid) + len(cid) > CAP:
            pid = pid[-(CAP - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        return float(model(input_ids=ids, labels=labels).loss)

    def score(alpha, seed):
        vals, fracs = [], []
        for r, target, imposter_card in pairs:
            card, frac = mix_card(target, imposter_card, alpha, substrate, r["player"], seed)
            vals.append(nll(r["base"], card, r["label"]))
            fracs.append(frac)
        return np.array(vals), float(np.mean(fracs))

    for alpha, arm in ((1.0, "A4frozen"), (0.0, "A3")):
        vals, _ = score(alpha, 0)
        np.save(out_dir / f"endpoint_{substrate}_8b_{arm}.npy", vals)
        print(
            f"  endpoint alpha={alpha} ({arm}) mean_nll={vals.mean():.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    for alpha in ALPHAS:
        for seed in SEEDS:
            tag = f"mix_fixed_{substrate}_8b_a{int(alpha * 100)}_s{seed}"
            out = out_dir / f"{tag}.json"
            if out.exists():
                print(f"[skip] {out.name}", flush=True)
                continue
            vals, frac = score(alpha, seed)
            out.write_text(
                json.dumps(
                    {
                        "family": "persona-prompt-frozen-content-ladder-fixed",
                        "substrate": substrate,
                        "size": "8b",
                        "model": MODELS["8b"],
                        "target_side": "frozen_card",
                        "alpha": alpha,
                        "seed": seed,
                        "n_eval": len(vals),
                        "mean_realized_target_fraction": frac,
                        "players": [r["player"] for r, _, _ in pairs],
                        "labels": [r["label"] for r, _, _ in pairs],
                        "nll": vals.tolist(),
                        "mean_nll": float(vals.mean()),
                        "cap": CAP,
                    }
                )
            )
            print(
                f"dose {tag}: mean_nll={vals.mean():.4f} frac={frac:.3f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    print(f"dose {substrate} done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
