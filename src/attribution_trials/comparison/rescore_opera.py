"""Per-option rescoring of the OPeRA frozen prompt panel (input to the coverage table).

Reruns the frozen prompt panel protocol (raw few-shot completion, 1200-token
cap, Random(7) 4-shot prefix, Random(0) eval shuffle, N_EVAL cap) but scores
every label option at each decision and stores the full option log-probability
vector. Conditioning is the user's own card under the config; config ``none``
is the no-card condition. The option set is the label vocabulary of the
matching frozen panel file (action d=15, timing d=5).

Per-option total log-probability = sum of token log-probabilities of
" {option}" after "<fewshot><base>[\\n<card>]\\nAnswer:". The fast path runs
the prompt once with a KV cache and then all options in one batch; it is
self-tested against full forwards at startup and replaced by them if the two
disagree.

  python -m attribution_trials.comparison.rescore_opera <8b|osim-8b> <channel>:<config> [...]

Output: <COMPARISON>/coverage_inputs/gpu/rescore_opera-<channel>-<config>_<tag>.json
Skips a cell whose output exists.
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
from attribution_trials.audit.sft_matrix import N_EVAL, build_for

OUT = paths.COMPARISON / "coverage_inputs" / "gpu"
MODELS = {"8b": "Qwen/Qwen3-8B", "osim-8b": "cmu-lti/osim-8b"}
CAP = 1200
TIMING_ORDER = ["under 1.5s", "1.5 to 5s", "5 to 20s", "20 to 60s", "over 60s"]


def load_vocab(channel: str) -> list[str]:
    d = json.loads((paths.AUDIT / "opera" / f"frozen_opera-{channel}-combined_8b.json").read_text())
    vocab = sorted(set(d["labels"]))
    if channel == "timing":
        vocab = [v for v in TIMING_ORDER if v in vocab] + [
            v for v in vocab if v not in TIMING_ORDER
        ]
    return vocab


def build_cell_inputs(channel: str, config: str):
    """Eval records and few-shot prefix, as in the frozen prompt panel."""
    substrate = f"opera-{channel}-{'combined' if config == 'none' else config}"
    train, evals, cells, _ = build_for(substrate)
    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:N_EVAL]
    rng_fs = random.Random(7)
    fs_pool = rng_fs.sample(train, min(len(train), 8))
    fewshot = "".join(f"{ex['base']}\nAnswer: {ex['label']}\n\n" for ex in fs_pool[:4])
    return evals, fewshot


# ------------------------------------------------------------ scoring --


def _legacy_pairs(past):
    if hasattr(past, "to_legacy_cache"):
        return list(past.to_legacy_cache())
    if hasattr(past, "layers"):
        return [(l.keys, l.values) for l in past.layers]
    if hasattr(past, "key_cache"):
        return list(zip(past.key_cache, past.value_cache))
    return list(past)


def _expanded_cache(past, k: int):
    from transformers import DynamicCache

    new = DynamicCache()
    for i, (kk, vv) in enumerate(_legacy_pairs(past)):
        new.update(
            kk.expand(k, *kk.shape[1:]).contiguous(), vv.expand(k, *vv.shape[1:]).contiguous(), i
        )
    return new


@torch.no_grad()
def score_options_fast(model, pid: list[int], cids: list[list[int]]):
    """Total log-probability per option via a shared-prompt KV cache."""
    dev = model.device
    ids = torch.tensor([pid], device=dev)
    out = model(input_ids=ids, use_cache=True)
    lp_last = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    k = len(cids)
    maxl = max(len(c) for c in cids)
    totals = [float(lp_last[c[0]]) for c in cids]
    if maxl > 1:
        opt = torch.zeros((k, maxl), dtype=torch.long, device=dev)
        omask = torch.zeros((k, maxl), dtype=torch.long, device=dev)
        for j, c in enumerate(cids):
            opt[j, : len(c)] = torch.tensor(c, device=dev)
            omask[j, : len(c)] = 1
        cache = _expanded_cache(out.past_key_values, k)
        amask = torch.cat([torch.ones((k, len(pid)), dtype=torch.long, device=dev), omask], dim=1)
        out2 = model(input_ids=opt, past_key_values=cache, attention_mask=amask)
        lp2 = torch.log_softmax(out2.logits.float(), dim=-1)
        for j, c in enumerate(cids):
            for pos in range(1, len(c)):
                totals[j] += float(lp2[j, pos - 1, c[pos]])
    return totals


@torch.no_grad()
def score_options_naive(model, pid: list[int], cids: list[list[int]]):
    """Same quantity via full prompt+option forwards, batched over options."""
    dev = model.device
    k = len(cids)
    plen = len(pid)
    maxt = plen + max(len(c) for c in cids)
    ids = torch.zeros((k, maxt), dtype=torch.long, device=dev)
    mask = torch.zeros((k, maxt), dtype=torch.long, device=dev)
    for j, c in enumerate(cids):
        seq = pid + c
        ids[j, : len(seq)] = torch.tensor(seq, device=dev)
        mask[j, : len(seq)] = 1
    logits = model(input_ids=ids, attention_mask=mask).logits
    # the option token at plen+pos is predicted from position plen+pos-1
    lp = torch.log_softmax(logits[:, plen - 1 : maxt - 1, :].float(), dim=-1)
    totals = []
    for j, c in enumerate(cids):
        t = 0.0
        for pos, tokid in enumerate(c):
            t += float(lp[j, pos, tokid])
        totals.append(t)
    return totals


def run_cell(model, tok, tag, channel, config, scorer):
    name = f"rescore_opera-{channel}-{config}_{tag}"
    out = OUT / f"{name}.json"
    t0 = time.time()
    evals, fewshot = build_cell_inputs(channel, config)
    options = load_vocab(channel)
    cids = [tok(" " + o, add_special_tokens=False)["input_ids"] for o in options]
    maxl = max(len(c) for c in cids)
    opt_idx = {o: i for i, o in enumerate(options)}

    players, ts, labels, lps, nll_true = [], [], [], [], []
    truncations = 0
    for i, r in enumerate(evals):
        card = None if config == "none" else r["self_card"]
        prompt = fewshot + r["base"] + ("\n" + card if card else "") + "\nAnswer:"
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        if len(pid) + maxl > CAP:
            pid = pid[-(CAP - maxl) :]
            truncations += 1
        totals = scorer(model, pid, cids)
        players.append(r["player"])
        ts.append(r["t"])
        labels.append(r["label"])
        lps.append([round(x, 4) for x in totals])
        ti = opt_idx[r["label"]]
        nll_true.append(-totals[ti] / len(cids[ti]))
        if (i + 1) % 200 == 0:
            print(f"  [{name}] {i + 1}/{len(evals)} ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "family": "per-option-rescore",
        "substrate": f"opera-{channel}-{config}",
        "channel": channel,
        "config": config,
        "conditioning": "none" if config == "none" else "self(live)",
        "size": tag,
        "model": MODELS[tag],
        "format": "raw",
        "n_eval": len(evals),
        "options": options,
        "option_token_lens": [len(c) for c in cids],
        "players": players,
        "t": ts,
        "labels": labels,
        "option_logprobs": lps,
        "nll_true": nll_true,
        "truncations": truncations,
        "scorer": scorer.__name__,
        "wall_seconds": time.time() - t0,
    }
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(
        f"{name}: n={len(evals)} truncated={truncations} ({time.time() - t0:.0f}s) -> {out}",
        flush=True,
    )


def pick_scorer(model, tok):
    """Self-test the fast path against full forwards on three real prompts."""
    evals, fewshot = build_cell_inputs("action", "combined")
    options = load_vocab("action")
    cids = [tok(" " + o, add_special_tokens=False)["input_ids"] for o in options]
    for r in evals[:3]:
        prompt = fewshot + r["base"] + "\n" + r["self_card"] + "\nAnswer:"
        pid = tok(prompt, add_special_tokens=False)["input_ids"]
        try:
            fast = score_options_fast(model, pid, cids)
        except Exception as e:  # cache API mismatch -> full forwards
            print(f"[scorer] fast path raised {type(e).__name__}: {e} -> naive", flush=True)
            return score_options_naive
        naive = score_options_naive(model, pid, cids)
        dmax = max(abs(a - b) for a, b in zip(fast, naive))
        print(f"[scorer] self-test max|fast-naive|={dmax:.4f}", flush=True)
        if dmax > 0.05:
            print("[scorer] disagreement > 0.05 -> naive", flush=True)
            return score_options_naive
    return score_options_fast


def main() -> int:
    tag = sys.argv[1]
    paths.ensure(OUT)
    todo = []
    for c in sys.argv[2:]:
        ch, cfg = c.split(":")
        if (OUT / f"rescore_opera-{ch}-{cfg}_{tag}.json").exists():
            print(f"[skip] {ch}:{cfg} exists", flush=True)
        else:
            todo.append((ch, cfg))
    if not todo:
        return 0

    # fp32: batched bf16 kernels add noise to the per-option distributions.
    tok = AutoTokenizer.from_pretrained(MODELS[tag])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[tag], torch_dtype=torch.float32, device_map="cuda"
    ).eval()
    scorer = pick_scorer(model, tok)
    for ch, cfg in todo:
        run_cell(model, tok, tag, ch, cfg, scorer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
