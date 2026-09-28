"""Per-option rescoring, frozen-prompt family.

The frozen panel stores only the observed label's mean-per-token NLL per
decision, so accuracy, total variation and method-implied populations cannot
be computed from it.  This reruns the frozen-panel protocol and scores EVERY
label of the closed set, for the five arms the comparison tables need.

Protocol (as `audit.frozen_panel` and `data.opera`): raw few-shot completion,
Random(0) eval shuffle, N_EVAL cap, Random(7) 4-shot population prefix,
CAP 640 (kt/chess) or 1200 (opera), `imposter_map(cells, 0)` for A3.  Arms:

  A0  no card              A1  placebo (population) card
  A2  group (cell) card    A3  same-cell imposter's FROZEN card
  A4f own FROZEN card

Option scoring: one forward of prompt + option per option (`score_exact`, the
panel's own computation), or the `fast` path that shares one forward across
all single-token options, self-tested against `score_exact` at startup.
The run dtype is bf16, the panel's own.

Usage:
  python -m attribution_trials.comparison.per_option_frozen <size> \\
      <substrate> [<substrate> ...] [--mode fast|exact] [--cap N] \\
      [--suffix S] [--out DIR]
Sizes: 1.7b | 8b | olmo-32b
Substrates: kt | chess | opera-<action|timing>-<static|dynamic|combined>
Idempotent: skips a cell whose .npz already exists.
Output: <COMPARISON>/per_option/persona-frozen_<substrate>_<size><suffix>.npz
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths
from attribution_trials.audit.sft_matrix import (
    CHESS_LABELS,
    MODELS,
    N_EVAL,
    build_for,
    imposter_map,
)

# Checkpoints outside the audit's MODELS table; scored on the same protocol
# as new rows outside the 84 combinations.
MODELS_EXTRA = {
    "olmo-32b": "allenai/Olmo-3.1-32B-Think",
}


def model_name(size):
    return MODELS_EXTRA.get(size) or MODELS[size]


ARMS = ("A0", "A1", "A2", "A3", "A4f")
TIMING_ORDER = ["under 1.5s", "1.5 to 5s", "5 to 20s", "20 to 60s", "over 60s"]


def label_set(substrate: str, labels) -> list[str]:
    """Closed label set, in a stable order (ordinal where one exists)."""
    seen = sorted(set(labels))
    if substrate == "chess":
        order = CHESS_LABELS
    elif substrate.startswith("opera-timing"):
        order = TIMING_ORDER
    elif substrate == "kt":
        order = ["correct", "incorrect"]
    else:  # opera-action: no ordinal structure, alphabetical is the order
        return seen
    missing = [v for v in seen if v not in order]
    assert not missing, f"{substrate}: labels outside the closed set {missing}"
    return [v for v in order if v in seen]


def build_cell(substrate: str):
    """Eval stream + few-shot prefix + arm card functions, as frozen_panel."""
    train, evals, cells, frozen_card_of = build_for(substrate)
    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:N_EVAL]
    imp = imposter_map(cells, 0)
    rng_fs = random.Random(7)
    fs_pool = rng_fs.sample(train, min(len(train), 8))
    fewshot = "".join(f"{ex['base']}\nAnswer: {ex['label']}\n\n" for ex in fs_pool[:4])
    arm_card = {
        "A0": lambda r: None,
        "A1": lambda r: r["placebo_card"],
        "A2": lambda r: r["group_card"],
        "A3": lambda r: frozen_card_of[imp[r["player"]]],
        "A4f": lambda r: r["frozen_card"],
    }
    return evals, fewshot, arm_card


# ------------------------------------------------------------ scoring --
#
# `frozen_panel._nll` truncates the prompt to `CAP - len(cid)` using the TRUE
# label's own token count, so the prompt a label is scored against depends on
# that label's length.  Options are therefore grouped by token length: one
# truncated prompt per group, reproducing the panel's rule for every option.


def group_by_len(cids):
    """[(token_len, [option index, ...]), ...] — the truncation groups."""
    g = {}
    for j, c in enumerate(cids):
        g.setdefault(len(c), []).append(j)
    return sorted(g.items())


def _truncate(pid, cap, clen):
    """frozen_panel's rule, verbatim."""
    return pid[-(cap - clen) :] if len(pid) + clen > cap else pid


@torch.no_grad()
def score_exact(model, pid, cids, cap):
    """`frozen_panel._nll`'s own computation, per option: one batch-1 forward
    of prompt+label with no cache."""
    dev = model.device
    totals, trunc = [], 0
    for c in cids:
        p = _truncate(pid, cap, len(c))
        trunc += len(p) != len(pid)
        ids = torch.tensor([p + c], device=dev)
        lp = torch.log_softmax(model(input_ids=ids).logits[0, len(p) - 1 : -1].float(), dim=-1)
        totals.append(sum(float(lp[pos, t]) for pos, t in enumerate(c)))
    return totals, trunc // len(cids)


@torch.no_grad()
def score_fast(model, pid, cids, cap):
    """`exact`, with the single-token options sharing one forward.

    A prompt-only forward has length L while the panel's has L+1, and bf16
    kernels are not length-invariant, so a 1-token option's logprob is not
    read off a prompt-only forward.  Instead the panel's sequence length is
    kept and causality is used: under a causal mask the appended label token
    cannot affect the logits at any earlier position, so ONE forward of
    `prompt + <any single token>` yields position len(p)-1 for every 1-token
    option.  Multi-token options are scored as in `score_exact`.
    """
    dev = model.device
    totals = [0.0] * len(cids)
    trunc = 0
    for clen, idxs in group_by_len(cids):
        p = _truncate(pid, cap, clen)
        trunc += len(p) != len(pid)
        if clen == 1:
            ids = torch.tensor([p + cids[idxs[0]]], device=dev)
            lp = torch.log_softmax(model(input_ids=ids).logits[0, len(p) - 1].float(), dim=-1)
            for j in idxs:
                totals[j] = float(lp[cids[j][0]])
        else:
            for j in idxs:
                c = cids[j]
                ids = torch.tensor([p + c], device=dev)
                lp = torch.log_softmax(
                    model(input_ids=ids).logits[0, len(p) - 1 : -1].float(), dim=-1
                )
                totals[j] = sum(float(lp[pos, t]) for pos, t in enumerate(c))
    return totals, trunc


SCORERS = {"exact": score_exact, "fast": score_fast}


def pick_scorer(model, prompt_ids_of, rows, cids, cap, mode):
    """Return `mode`'s scorer, after proving it agrees with `score_exact`.

    `score_fast`'s shortcut rests on causality, which holds for a dense
    transformer but not for a Mixture-of-Experts one, where routing groups
    tokens across the sequence and the appended token can change the
    accumulation order at earlier positions.  So the property is checked per
    model on the first rows: any disagreement and the cell falls back to
    `score_exact`, which issues the panel's own forward per option.
    """
    scorer = SCORERS[mode]
    if scorer is not score_fast:
        return scorer, mode, 0.0
    worst = 0.0
    for r in rows[:3]:
        for card in (None, r["frozen_card"]):
            pid = prompt_ids_of(r, card)
            f, _ = score_fast(model, pid, cids, cap)
            e, _ = score_exact(model, pid, cids, cap)
            worst = max(worst, max(abs(a - b) for a, b in zip(f, e)))
    if worst > 0.0:
        print(f"[scorer] fast disagrees with exact by {worst:.6f} -> exact", flush=True)
        return score_exact, "exact", worst
    print(f"[scorer] fast == exact (max|d|={worst:.1e}) -> fast", flush=True)
    return score_fast, "fast", worst


# ---------------------------------------------------------------- cell --


def run_cell(model, tok, size, substrate, out_dir, mode, cap_override=None, suffix=""):
    stem = f"persona-frozen_{substrate}_{size}{suffix}"
    out = out_dir / f"{stem}.npz"
    if out.exists():
        print(f"[skip] {out.name} exists", flush=True)
        return None
    t0 = time.time()
    cap = cap_override or (1200 if substrate.startswith("opera") else 640)
    evals, fewshot, arm_card = build_cell(substrate)
    options = label_set(substrate, [r["label"] for r in evals])
    cids = [tok(" " + o, add_special_tokens=False)["input_ids"] for o in options]
    ntok = np.array([len(c) for c in cids])
    oi = {o: i for i, o in enumerate(options)}

    def _pids(r, card):
        return tok(
            fewshot + r["base"] + ("\n" + card if card else "") + "\nAnswer:",
            add_special_tokens=False,
        )["input_ids"]

    scorer, mode_used, selftest = pick_scorer(model, _pids, evals, cids, cap, mode)

    n, k = len(evals), len(options)
    logp = {a: np.zeros((n, k), dtype=np.float64) for a in ARMS}
    truncations = 0
    for a in ARMS:
        cf = arm_card[a]
        for i, r in enumerate(evals):
            card = cf(r)
            prompt = fewshot + r["base"] + ("\n" + card if card else "") + "\nAnswer:"
            pid = tok(prompt, add_special_tokens=False)["input_ids"]
            totals, tr = scorer(model, pid, cids, cap)
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
    with open(tmp, "wb") as fh:  # savez appends .npz to a path, not to a handle
        np.savez_compressed(fh, **payload)
    os.replace(tmp, out)

    meta = {
        "family": "persona-frozen",
        "substrate": substrate,
        "size": size,
        "model": model_name(size),
        "scorer": mode_used,
        "fast_vs_exact_max": selftest,
        "arms": list(ARMS),
        "n_eval": n,
        "options": options,
        "option_token_lens": ntok.tolist(),
        "cap": cap,
        "truncated_prompt_forwards": truncations,
    }
    (out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"CELL {stem}: n={n} k={k} mode={mode_used} ({time.time() - t0:.0f}s) -> {out}", flush=True
    )
    return meta


def main() -> int:
    argv = sys.argv[1:]
    flags = {}
    subs = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--out", "--mode", "--cap", "--suffix"):
            flags[a] = argv[i + 1]
            i += 2
        else:
            subs.append(a)
            i += 1
    size, subs = subs[0], subs[1:]
    mode = flags.get("--mode", "fast")
    cap_override = int(flags["--cap"]) if "--cap" in flags else None
    suffix = flags.get("--suffix", "")
    out_dir = paths.ensure(Path(flags.get("--out", paths.COMPARISON / "per_option")))

    todo = [s for s in subs if not (out_dir / f"persona-frozen_{s}_{size}{suffix}.npz").exists()]
    if not todo:
        print("nothing to do", flush=True)
        return 0

    tok = AutoTokenizer.from_pretrained(model_name(size))
    model = AutoModelForCausalLM.from_pretrained(
        model_name(size), dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for s in todo:
        run_cell(model, tok, size, s, out_dir, mode, cap_override, suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
