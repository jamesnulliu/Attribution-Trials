"""Attribution-trial scores for Qwen3-1.7B and Qwen3-8B on the WildChat panel.

Teacher-forced next-turn NLL of the real user turn under no card (A0), the
imposter's card (A3) and the target's card (A4), on the 1,444 primary
out-of-distribution turns.  Writes ``paths.WILDCHAT / "scale_<size>.json"``
in the columnar schema (``players``/``keys``/``nll``) that the readouts accept.

  python -m attribution_trials.wildchat.score_trial_frozen 8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from attribution_trials import paths
from attribution_trials.wildchat.common import CANDIDATES, PANEL_RUNTIME

SEED = 0
ROW_SCOPE = "primary"
EXPECTED_ROWS = 1444
SCALES = ("1.7b", "8b")
PREFIX_BUDGET = 2000
SYSTEM_BASE = (
    "You are simulating a specific user who is chatting with an AI assistant. "
    "Write the user's messages exactly as that user would."
)
SYSTEM_CARD = SYSTEM_BASE + (" Here are examples of messages this user has written before:\n{card}")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def seg(tok, role: str, text: str) -> list[int]:
    return tok.encode(
        f"<|im_start|>{role}\n{text}<|im_end|>\n",
        add_special_tokens=False,
    )


def build_ids(tok, card: str | None, prefix: list[dict], target: str):
    system = SYSTEM_CARD.format(card=card) if card else SYSTEM_BASE
    ids = seg(tok, "system", system)
    pieces = []
    for turn in prefix:
        role = turn["role"] if turn["role"] in ("user", "assistant") else "user"
        pieces.append(seg(tok, role, " ".join(turn["content"].split())[:6000]))
    while pieces and sum(len(x) for x in pieces) > PREFIX_BUDGET:
        pieces.pop(0)
    for piece in pieces:
        ids.extend(piece)
    head = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    tgt = tok.encode(target, add_special_tokens=False)
    return ids + head, tgt


@torch.inference_mode()
def score_turn(model, tok, card, prefix, target, device):
    ctx, tgt = build_ids(tok, card, prefix, target)
    if not tgt:
        return None
    ids = torch.tensor([ctx + tgt], dtype=torch.long, device=device)
    logits = model(ids).logits[0, len(ctx) - 1 : -1, :].float()
    target_t = torch.tensor(tgt, dtype=torch.long, device=device)
    nll = torch.nn.functional.cross_entropy(logits, target_t, reduction="mean")
    value = float(nll.cpu())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite NLL")
    return value, len(tgt)


def resolve_revision(model_dir) -> tuple[str | None, str]:
    """Return (revision, source) from a hub snapshot path or a git checkout, if any."""
    path = Path(model_dir)
    for candidate in (path, path / "config.json"):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        for part in resolved.parts:
            if SHA_RE.match(part):
                return part, "hub snapshot path"
    head = path / ".git" / "HEAD"
    if head.is_file():
        text = head.read_text().strip()
        if SHA_RE.match(text):
            return text, "git HEAD"
        if text.startswith("ref: "):
            ref = (path / ".git" / text[5:]).resolve()
            if ref.is_file():
                sha = ref.read_text().strip()
                if SHA_RE.match(sha):
                    return sha, "git HEAD"
    return None, "unresolvable: plain local directory with no hub or git provenance"


def check_revision(model_dir, expected: str, label: str) -> dict:
    """Raise when the on-disk revision is known and differs from the pin."""
    revision, source = resolve_revision(model_dir)
    if revision is not None and revision != expected:
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected}, "
            f"found {revision} via {source} at {model_dir}"
        )
    return {
        "expected": expected,
        "resolved": revision,
        "source": source,
        "verified": revision is not None,
    }


def row_digest(rows: list[dict]) -> str:
    blob = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", choices=SCALES)
    args = ap.parse_args()
    spec = CANDIDATES[args.scale]
    model_path = spec["model_path"]
    if not PANEL_RUNTIME.exists():
        raise FileNotFoundError(PANEL_RUNTIME)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("scoring requires a CUDA/ROCm device")

    panel = json.loads(PANEL_RUNTIME.read_text())
    users = panel["users"]
    selected = []
    for uid in sorted(users):
        for decision in users[uid]["decisions"]:
            if decision.get("split") == ROW_SCOPE:
                selected.append((uid, decision))
    if len(selected) != EXPECTED_ROWS:
        raise AssertionError(f"primary row count changed: {len(selected)} != {EXPECTED_ROWS}")

    row_meta = [
        {
            "user_id": uid,
            "conv_id": d["conv_id"],
            "turn_index": int(d["turn_index"]),
            "split": d["split"],
            "cluster_id": d["cluster_id"],
        }
        for uid, d in selected
    ]
    digest = row_digest(row_meta)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    revision = check_revision(model_path, spec["revision"], spec["model"])

    nll = {"A0": [], "A3": [], "A4": []}
    players, keys, target_tokens, cluster_ids = [], [], [], []
    profile_conditioning = []
    total = len(selected)
    for index, (uid, decision) in enumerate(selected, 1):
        user = users[uid]
        imposter_uid = panel["imposter"][uid]
        cards = {
            "A0": None,
            "A3": users[imposter_uid]["profile_card"],
            "A4": user["profile_card"],
        }
        if not cards["A3"] or not cards["A4"] or cards["A3"] == cards["A4"]:
            raise AssertionError(f"invalid A3/A4 cards for {uid}")
        profile_conditioning.append(
            {
                "user_id": uid,
                "imposter_id": imposter_uid,
                "cards_differ": True,
            }
        )
        row_scores = {}
        row_tokens = None
        for arm, card in cards.items():
            result = score_turn(
                model, tokenizer, card, decision["prefix"], decision["target"], "cuda"
            )
            if result is None:
                raise AssertionError(f"empty target at row {index}")
            value, ntokens = result
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite NLL at row {index}")
            row_scores[arm] = float(value)
            row_tokens = ntokens if row_tokens is None else row_tokens
        players.append(uid)
        keys.append([decision["conv_id"], int(decision["turn_index"])])
        cluster_ids.append(decision["cluster_id"])
        target_tokens.append(int(row_tokens))
        for arm in nll:
            nll[arm].append(row_scores[arm])
        if index % 50 == 0 or index == total:
            print(f"{args.scale} primary rows {index}/{total}", flush=True)

    by_user = defaultdict(lambda: defaultdict(list))
    for i, uid in enumerate(players):
        for arm in nll:
            by_user[uid][arm].append(nll[arm][i])
    per_user = {}
    for uid in sorted(by_user):
        means = {arm: float(np.mean(by_user[uid][arm])) for arm in nll}
        per_user[uid] = {
            "n_decisions": len(by_user[uid]["A4"]),
            "mean_nll": means,
            "delta_indiv_A4_minus_A3": means["A4"] - means["A3"],
        }

    out = {
        "scale": args.scale,
        "model": spec["model"],
        "model_path": str(model_path),
        "model_revision_expected": spec["revision"],
        "model_revision_check": revision,
        "panel_revision": panel["meta"]["revision"],
        "row_scope": ROW_SCOPE,
        "n_users": len(by_user),
        "n_decisions": len(players),
        "row_digest": digest,
        "players": players,
        "keys": keys,
        "cluster_ids": cluster_ids,
        "target_tokens": target_tokens,
        "nll": nll,
        "per_user": per_user,
        "profile_conditioning": profile_conditioning,
        "arms": ["A0", "A3", "A4"],
        "seed": SEED,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "rocm": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
        },
    }
    paths.ensure(paths.WILDCHAT)
    path = paths.WILDCHAT / f"scale_{args.scale}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scale": args.scale,
                "n_users": out["n_users"],
                "n_decisions": out["n_decisions"],
                "row_digest": digest,
                "output": str(path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
