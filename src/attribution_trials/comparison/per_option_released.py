"""Per-option rescoring, released simulators (Osim panel, Sim2Real trio).

Same gap and same fix as `per_option_frozen`, for the cells scored on
published checkpoints rather than a Qwen base: the released panels store only
the observed label's mean-per-token NLL, so no distribution exists to compute
accuracy, total variation or a method-implied population from.

Both released panels are ports of frozen_panel's protocol and share its
constants: Random(0) eval shuffle, Random(7) 4-shot population prefix,
CAP 1200, `imposter_map(cells, 0)`, arms A0/A1/A2/A3/A4frozen.  The two
formats differ only in how the card and the label are placed:

  raw   card appended to the question, label scored as " <label>"
  chat  card in the system turn ("You are this shopper: ..."), few-shot and
        question in the user turn, label scored as the assistant turn with NO
        leading space

The released panel fixes the number of eval rows and the observed label
stream; both are read from it.  Option scoring is `per_option_frozen`'s.

Usage:
  python -m attribution_trials.comparison.per_option_released <tag> \\
      <substrate> [<substrate> ...] [--mode fast|exact] [--out DIR]
Tags: osim-4b osim-4b-mid osim-8b osim-8b-mid coser-8b humanlike-7b
Idempotent: skips a cell whose .npz already exists.
Output: <COMPARISON>/per_option/{trained-sim,sim2real-trio}_<substrate>_<tag>.npz
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths
from attribution_trials.comparison.per_option_frozen import (
    ARMS,
    build_cell,
    label_set,
    pick_scorer,
)

# tag -> (hub name, format, family key as it appears in the 84)
RELEASED = {
    "osim-4b": ("cmu-lti/osim-4b", "raw", "trained-sim"),
    "osim-4b-mid": ("cmu-lti/osim-4b-mid", "raw", "trained-sim"),
    "osim-8b": ("cmu-lti/osim-8b", "raw", "trained-sim"),
    "osim-8b-mid": ("cmu-lti/osim-8b-mid", "raw", "trained-sim"),
    "coser-8b": ("Neph0s/CoSER-Llama-3.1-8B", "chat", "sim2real-trio"),
    "humanlike-7b": ("HumanLLMs/Human-Like-Qwen2.5-7B-Instruct", "chat", "sim2real-trio"),
}
CAP = 1200  # opera cap, as in frozen_panel


def artifact_path(substrate: str, tag: str) -> Path:
    body = substrate.split("-", 1)[1]  # "action-combined"
    if RELEASED[tag][1] == "raw":
        return paths.AUDIT / "released" / f"osim-panel_{body}_{tag}.json"
    return paths.AUDIT / "released" / f"trio-panel_{body}_{tag}_chat.json"


def make_prompt_fn(tok, fewshot, fmt):
    """(card, base) -> prompt ids, and the option continuation strings."""
    if fmt == "raw":

        def prompt_ids(base, card):
            p = fewshot + base + ("\n" + card if card else "") + "\nAnswer:"
            return tok(p, add_special_tokens=False)["input_ids"]

        cont = lambda o: " " + o  # noqa: E731
    else:

        def prompt_ids(base, card):
            sys_p = (
                ("You are this shopper: " + card) if card else "You are a shopper browsing Amazon."
            )
            msgs = [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": fewshot + base + "\nAnswer:"},
            ]
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            return tok(text, add_special_tokens=False)["input_ids"]

        cont = lambda o: o  # noqa: E731
    return prompt_ids, cont


def run_cell(model, tok, tag, substrate, out_dir, mode):
    name, fmt, family = RELEASED[tag]
    stem = f"{family}_{substrate}_{tag}"
    out = out_dir / f"{stem}.npz"
    if out.exists():
        print(f"[skip] {out.name} exists", flush=True)
        return None
    t0 = time.time()
    evals, fewshot, arm_card = build_cell(substrate)
    art = json.loads(artifact_path(substrate, tag).read_text())
    options = label_set(substrate, art["labels"])
    prompt_ids, cont = make_prompt_fn(tok, fewshot, fmt)
    cids = [tok(cont(o), add_special_tokens=False)["input_ids"] for o in options]
    ntok = np.array([len(c) for c in cids])
    oi = {o: i for i, o in enumerate(options)}
    scorer, mode_used, selftest = pick_scorer(
        model, lambda r, card: prompt_ids(r["base"], card), evals, cids, CAP, mode
    )

    evals = evals[: art["n_eval"]]

    n, k = len(evals), len(options)
    logp = {a: np.zeros((n, k), dtype=np.float64) for a in ARMS}
    truncations = 0
    for a in ARMS:
        cf = arm_card[a]
        for i, r in enumerate(evals):
            pid = prompt_ids(r["base"], cf(r))
            totals, tr = scorer(model, pid, cids, CAP)
            logp[a][i] = totals
            truncations += tr
        print(f"  [{stem}] arm {a} scored ({time.time() - t0:.0f}s)", flush=True)

    true_idx = np.array([oi[r["label"]] for r in evals])
    payload = {
        "players": np.array([r["player"] for r in evals]),
        "t": np.array([r.get("t", i) for i, r in enumerate(evals)]),
        "labels": true_idx,
        "label_set": np.array(options),
        "option_token_lens": ntok,
    }
    for a in ARMS:
        payload[f"logp_{a}"] = logp[a]
    tmp = out.with_suffix(f".npz.tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out)

    meta = {
        "family": family,
        "substrate": substrate,
        "size": tag,
        "model": name,
        "format": fmt,
        "scorer": mode_used,
        "fast_vs_exact_max": selftest,
        "arms": list(ARMS),
        "n_eval": n,
        "options": options,
        "option_token_lens": ntok.tolist(),
        "cap": CAP,
        "truncated_prompt_forwards": truncations,
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"CELL {stem}: n={n} k={k} fmt={fmt} mode={mode_used} ({time.time() - t0:.0f}s) -> {out}",
        flush=True,
    )
    return meta


def main() -> int:
    argv = sys.argv[1:]
    flags, pos, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--out", "--mode"):
            flags[a] = argv[i + 1]
            i += 2
        else:
            pos.append(a)
            i += 1
    tag, subs = pos[0], pos[1:]
    mode = flags.get("--mode", "fast")
    out_dir = paths.ensure(Path(flags.get("--out", paths.COMPARISON / "per_option")))

    family = RELEASED[tag][2]
    todo = [s for s in subs if not (out_dir / f"{family}_{s}_{tag}.npz").exists()]
    if not todo:
        print("nothing to do", flush=True)
        return 0

    name = RELEASED[tag][0]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    if RELEASED[tag][1] == "chat" and tok.chat_template is None:
        raise SystemExit(f"{name} ships no chat template")
    for s in todo:
        run_cell(model, tok, tag, s, out_dir, mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
