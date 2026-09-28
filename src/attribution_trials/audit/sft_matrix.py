"""User-profile LoRA: one peft LoRA retrained per condition.

Also the shared record builders (build_kt / build_chess / build_for) and the
imposter map that every LLM-scored family uses.

  conditions (each trained separately):
    none    -- A0: base prompt, no profile card
    placebo -- A1: population card (TRAIN rows only)
    group   -- A2: cell card (user's cell, TRAIN rows only)
    self    -- the user's own causal card

  eval-time variants on the `self`-trained model:
    self_live   -- A4 (causal card keeps updating through eval)
    self_frozen -- A4frozen, the target (card frozen at the train-split
                   boundary)
    imposter    -- A3 (same-cell imposter's card frozen at their split)

Substrates/channels: kt response (" correct"/" incorrect"); chess
think-time bucket (fixed edges 1/3/8/20 s, 5 labels) on the pooled
chess panel; OPeRA "opera-<action|timing>-<static|dynamic|combined>".
Same data subsample (data seed 0) and same eval examples across all
conditions/backbones/seeds -> paired per-user bootstrap downstream.
Imposter derangement seeded by run seed.

Usage:
  python -m attribution_trials.audit.sft_matrix <substrate> \\
      <none|placebo|group|self> <seed> <1.7b|8b>
Output: <AUDIT>/{chess_kt,opera}/sft_<substrate>_<cond>_<size>_s<seed>.json
"""

from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths
from attribution_trials.bench.cells import (
    assign_tertile_cells,
    kt_prior_accuracy_scalar,
)
from attribution_trials.bench.memory_family import memory_rows
from attribution_trials.data.kt_csv import load_kt_csv
from attribution_trials.experiments.ec import session_split_indices

MODELS = {
    "1.7b": "Qwen/Qwen3-1.7B",
    "8b": "Qwen/Qwen3-8B",
    "30b": "Qwen/Qwen3-30B-A3B",
}
N_TRAIN, N_EVAL = 5000, 2000
MAX_LEN = 320
TRAIN_FRAC = 0.7
CHESS_EDGES = [1.0, 3.0, 8.0, 20.0]
CHESS_LABELS = ["under 1s", "1 to 3s", "3 to 8s", "8 to 20s", "over 20s"]
KT_DATA = paths.DATA / "kt/prepared/assist09.tsv"


def out_dir(substrate: str) -> Path:
    """Result directory for a substrate (OPeRA vs chess/KT)."""
    sub = "opera" if substrate.startswith("opera") else "chess_kt"
    return paths.ensure(paths.AUDIT / sub)


def bucket_of(seconds: float) -> str:
    for e, lab in zip(CHESS_EDGES, CHESS_LABELS):
        if seconds < e:
            return lab
    return CHESS_LABELS[-1]


# ---------------------------------------------------------------- KT --


def kt_scorecard(m: list[float]) -> str:
    seen = "no answers yet" if m[9] < 0.5 else "answers so far"
    return (
        f"Student state ({seen}): overall accuracy {m[1]:.2f}, "
        f"recent accuracy {m[3]:.2f}, last answer "
        f"{'correct' if m[4] > 0.5 else 'incorrect'}, "
        f"streak {m[5]:+.2f}, skill-adjusted trend {m[7]:+.2f}."
    )


def _mean_traj(
    rows_of: dict[str, list], members: list[str], splits_of: dict[str, int]
) -> list[list[float]]:
    """Per-t mean of members' TRAIN rows; hold last when exhausted."""
    horizon = max(splits_of[p] for p in members)
    out, last = [], None
    for t in range(horizon):
        rs = [rows_of[p][t] for p in members if t < splits_of[p]]
        last = [sum(c) / len(c) for c in zip(*rs)] if rs else last
        out.append(list(last))
    return out


def build_kt():
    ds = load_kt_csv(
        str(KT_DATA),
        n_students=500,
        min_responses=50,
    )
    trajs = ds.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    splits_of = {t.player_id: k for t, k in zip(trajs, splits)}
    from functools import partial

    cells = assign_tertile_cells(
        ds,
        partial(kt_prior_accuracy_scalar, train_frac=TRAIN_FRAC),
        prefix="pacc",
    )
    rows_of = {t.player_id: memory_rows(t) for t in trajs}
    allp = [t.player_id for t in trajs]
    pop = _mean_traj(rows_of, allp, splits_of)
    cell_traj = {
        c: _mean_traj(rows_of, [p for p in allp if cells[p] == c], splits_of)
        for c in set(cells.values())
    }

    train, evals = [], []
    for traj, k in zip(trajs, splits):
        p = traj.player_id
        rows = rows_of[p]
        for t, (dp, obs) in enumerate(zip(traj.decisions, traj.observations)):
            base = (
                "Predict whether this student answers the next question "
                "correctly.\n"
                f"Question difficulty: {float(dp.state[0]):.2f} "
                f"(0 = easiest, 1 = hardest). Attempt {t + 1}."
            )
            ct = cell_traj[cells[p]]
            rec = {
                "player": p,
                "t": t,
                "base": base,
                "self_card": kt_scorecard(rows[t]),
                "frozen_card": kt_scorecard(rows[min(k, len(rows) - 1)]),
                "group_card": kt_scorecard(ct[min(t, len(ct) - 1)]),
                "placebo_card": kt_scorecard(pop[min(t, len(pop) - 1)]),
                "label": "correct" if obs.move == "correct" else "incorrect",
            }
            (train if t < k else evals).append(rec)
    frozen_card_of = {
        t.player_id: kt_scorecard(
            rows_of[t.player_id][min(splits_of[t.player_id], len(rows_of[t.player_id]) - 1)]
        )
        for t in trajs
    }
    return train, evals, cells, frozen_card_of


# ------------------------------------------------------------- chess --


def chess_rows(traj) -> list[list[float]]:
    """Causal per-decision features: [has_hist, elo, median_think,
    recent5_mean, last_think, frac_fast(<2s), pressure_think]."""
    rows, hist, press = [], [], []
    for t, (dp, obs) in enumerate(zip(traj.decisions, traj.observations)):
        elo = float(dp.context.get("player_elo") or 1500.0)
        if not hist:
            rows.append([0.0, elo, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            rows.append(
                [
                    1.0,
                    elo,
                    float(statistics.median(hist)),
                    float(sum(hist[-5:]) / len(hist[-5:])),
                    float(hist[-1]),
                    float(sum(h < 2.0 for h in hist) / len(hist)),
                    float(sum(press) / len(press)) if press else float(statistics.median(hist)),
                ]
            )
        ts = float(obs.time_spent or 0.0)
        hist.append(ts)
        rem = dp.time_signal.time_remaining if dp.time_signal else None
        if rem is not None and rem < 60.0:
            press.append(ts)
    return rows


def chess_card(m: list[float]) -> str:
    if m[0] < 0.5:
        return f"Player profile: rating about {m[1]:.0f}; no move history yet."
    return (
        f"Player profile: rating about {m[1]:.0f}; typical think time "
        f"{m[2]:.1f}s (median), last 5 moves average {m[3]:.1f}s, last "
        f"move {m[4]:.1f}s; {100 * m[5]:.0f}% of moves under 2s; under "
        f"time pressure thinks {m[6]:.1f}s."
    )


def build_chess():
    from attribution_trials.data.pooled_chess import build_pooled_chess

    data, cells, _, _ = build_pooled_chess()
    trajs = data.trajectories
    splits = session_split_indices(trajs, TRAIN_FRAC)
    splits_of = {t.player_id: k for t, k in zip(trajs, splits)}
    rows_of = {t.player_id: chess_rows(t) for t in trajs}
    allp = [t.player_id for t in trajs]
    pop = _mean_traj(rows_of, allp, splits_of)
    cell_traj = {
        c: _mean_traj(rows_of, [p for p in allp if cells[p] == c], splits_of)
        for c in set(cells.values())
    }

    train, evals = [], []
    for traj, k in zip(trajs, splits):
        p = traj.player_id
        rows = rows_of[p]
        for t, (dp, obs) in enumerate(zip(traj.decisions, traj.observations)):
            if obs.time_spent is None or dp.time_signal is None:
                continue
            sig = dp.time_signal
            base = (
                "Predict how long this chess player thinks before their "
                "next move.\n"
                f"Position (FEN): {dp.state}\n"
                f"Playing {dp.context.get('color')}, time control "
                f"{dp.context.get('time_control')}, move "
                f"{sig.move_number}, {sig.time_remaining:.0f}s left on "
                f"the clock, {sig.phase} phase.\n"
                "Buckets: under 1s, 1 to 3s, 3 to 8s, 8 to 20s, over 20s."
            )
            ct = cell_traj[cells[p]]
            rec = {
                "player": p,
                "t": t,
                "base": base,
                "self_card": chess_card(rows[t]),
                "frozen_card": chess_card(rows[k]),
                "group_card": chess_card(ct[min(t, len(ct) - 1)]),
                "placebo_card": chess_card(pop[min(t, len(pop) - 1)]),
                "label": bucket_of(float(obs.time_spent)),
            }
            (train if t < k else evals).append(rec)
    frozen_card_of = {
        t.player_id: chess_card(rows_of[t.player_id][splits_of[t.player_id]]) for t in trajs
    }
    return train, evals, cells, frozen_card_of


# ----------------------------------------------------------- harness --


def build_for(substrate: str):
    """Dispatch build() by substrate. OPeRA encodes channel+config in the
    name: 'opera-<action|timing>-<static|dynamic|combined>'."""
    if substrate.startswith("opera"):
        from attribution_trials.data.opera import build_opera

        _, channel, config = substrate.split("-")
        return build_opera(channel, config)
    return build_kt() if substrate == "kt" else build_chess()


def card_for(cond: str, rec: dict) -> str | None:
    return {
        "none": None,
        "placebo": rec["placebo_card"],
        "group": rec["group_card"],
        "self": rec["self_card"],
    }[cond]


def imposter_map(cells: dict[str, str], seed: int) -> dict[str, str]:
    """Seeded within-cell derangement (cycle)."""
    rng = random.Random(1000 + seed)
    out = {}
    members: dict[str, list[str]] = {}
    for p, c in cells.items():
        members.setdefault(c, []).append(p)
    for c, ids in members.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        for a, b in zip(ids, ids[1:] + ids[:1]):
            out[a] = b
    return out


def run(substrate, cond, seed, size):
    train, evals, cells, frozen_card_of = build_for(substrate)
    rng = random.Random(0)
    rng.shuffle(train)
    rng.shuffle(evals)
    train, evals = train[:N_TRAIN], evals[:N_EVAL]

    torch.manual_seed(seed)
    random.seed(seed)
    model_name = MODELS[size]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = get_peft_model(
        model,
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

    def _prompt_ids(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, add_special_tokens=False)["input_ids"]

    def _nll(base, card, label):
        prompt = base + ("\n" + card if card else "") + "\nAnswer:"
        pid = _prompt_ids(prompt)
        cid = tok(" " + label, add_special_tokens=False)["input_ids"]
        if len(pid) + len(cid) > MAX_LEN:
            pid = pid[-(MAX_LEN - len(cid)) :]
        ids = torch.tensor([pid + cid], device="cuda")
        labels = ids.clone()
        labels[0, : len(pid)] = -100
        return model(input_ids=ids, labels=labels).loss

    t0 = time.time()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    accum = 16
    for ep in range(2):
        random.shuffle(train)
        opt.zero_grad()
        for i, rec in enumerate(train):
            loss = _nll(rec["base"], card_for(cond, rec), rec["label"])
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
            if (i + 1) % 1000 == 0:
                print(
                    f"  [{substrate}/{cond}/{size} s{seed}] ep{ep} "
                    f"{i + 1}/{len(train)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        opt.step()
        opt.zero_grad()

    model.eval()
    variants = (
        {"eval": lambda r: card_for(cond, r)}
        if cond != "self"
        else {
            "self_live": lambda r: r["self_card"],
            "self_frozen": lambda r: r["frozen_card"],
            "imposter": lambda r: frozen_card_of[imp[r["player"]]],
        }
    )
    imp = imposter_map(cells, seed) if cond == "self" else None
    results = {}
    with torch.no_grad():
        for name, cf in variants.items():
            results[name] = [float(_nll(r["base"], cf(r), r["label"])) for r in evals]
    return {
        "substrate": substrate,
        "condition": cond,
        "seed": seed,
        "size": size,
        "model": model_name,
        "n_train": len(train),
        "n_eval": len(evals),
        "players": [r["player"] for r in evals],
        "labels": [r["label"] for r in evals],
        "nll": results,
        "mean_nll": {k: sum(v) / len(v) for k, v in results.items()},
        "wall_seconds": time.time() - t0,
    }


def main() -> int:
    substrate, cond, seed, size = (sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    payload = run(substrate, cond, seed, size)
    out = out_dir(substrate) / f"sft_{substrate}_{cond}_{size}_s{seed}.json"
    # Write atomically so an interrupted run never leaves a partial file.
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(
        f"SFT MATRIX {substrate}/{cond}/{size} s{seed} "
        f"mean_nll={payload['mean_nll']} "
        f"({payload['wall_seconds']:.0f}s) -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
