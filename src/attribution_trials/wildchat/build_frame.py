"""Build the generation frame: 236 users with at least six train-window turns.

Rebuilds each panel user's full train and evaluation windows from the panel
database, keeps the users with at least MIN_PER_HALF train-window turns, joins
their primary out-of-distribution turns to realized form labels (length
bucket, question, code, terminal), and fixes the restricted imposter map, the
user folds A/B and the odd/even row splits.  Writes ``paths.WILDCHAT /
"frame.json"`` (contains conversation text).

  python -m attribution_trials.wildchat.build_frame
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from attribution_trials import paths
from attribution_trials.wildchat.build_panel import (
    DB_PATH,
    MANIFEST_PATH,
    MAX_DECISIONS_PER_CONV,
    RUNTIME_PATH,
    TARGET_TOKENS,
    build_user,
    dedup_and_cluster,
    embed_texts,
    representative,
    safe_id,
    ts_key,
)

TOKENIZER = paths.MODELS / "Qwen3-1.7B"
FRAME = paths.WILDCHAT / "frame.json"

SEED = 0
MIN_PER_HALF = 6
EXPECTED_USERS = 236


def load_selected_conversations(manifest: dict) -> dict[str, tuple[str, list[dict]]]:
    selected = set(manifest["users"])
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    raw_by_safe: dict[str, str] = {}
    for (raw_uid,) in con.execute("SELECT DISTINCT uid FROM conv"):
        sid = safe_id("u_", raw_uid)
        if sid in selected:
            raw_by_safe[sid] = raw_uid
    missing = sorted(selected - set(raw_by_safe))
    if missing:
        raise AssertionError(f"selected users absent from panel DB: {missing[:3]}")

    out: dict[str, tuple[str, list[dict]]] = {}
    for sid in sorted(selected):
        raw_uid = raw_by_safe[sid]
        convs = []
        for uid, ch, ts, model, country, turns_json in con.execute(
            "SELECT uid,conv_hash,ts,model,country,turns FROM conv WHERE uid=?",
            (raw_uid,),
        ):
            convs.append(
                {
                    "hash": ch,
                    "ts": ts,
                    "model": model,
                    "country": country,
                    "turns": json.loads(turns_json),
                }
            )
        convs.sort(key=lambda x: (ts_key(x["ts"]), x["hash"]))
        out[sid] = (raw_uid, convs)
    con.close()
    return out


def rebuild_users(manifest: dict, tokenizer) -> dict[str, dict]:
    selected = load_selected_conversations(manifest)
    items = []
    for sid, (_, convs) in selected.items():
        for conv in convs:
            conv["rep"] = representative(conv)
            items.append((sid, conv))
    vectors = embed_texts([conv["rep"] for _, conv in items])
    for index, (_, conv) in enumerate(items):
        conv["vec"] = vectors[index]
    del vectors

    rebuilt = {}
    for sid in sorted(selected):
        raw_uid, convs = selected[sid]
        deduped, _ = dedup_and_cluster(convs)
        x = build_user(raw_uid, deduped, tokenizer)
        if x is None:
            raise AssertionError(f"selected user became ineligible: {sid}")
        expected = manifest["users"][sid]
        checks = {
            "n_convs": len(x["convs"]),
            "n_train": len(x["train"]),
            "n_eval_convs": len(x["eval"]),
            "n_primary_ood_convs": len(x["primary"]),
            "profile_conv_ids": sorted(safe_id("c_", c["hash"], 20) for c in x["train"]),
            "primary_conv_ids": sorted(safe_id("c_", c["hash"], 20) for c in x["primary"]),
        }
        for key, actual in checks.items():
            if actual != expected[key]:
                raise AssertionError(f"{sid} manifest mismatch for {key}")
        rebuilt[sid] = x
    return rebuilt


def make_rows(users: dict[str, dict], tokenizer) -> tuple[list[dict], list[int], list[int]]:
    rows = []
    train_lengths, all_lengths = [], []
    for sid in sorted(users):
        x = users[sid]
        for split, convs in (("train", x["train"]), ("eval", x["eval"])):
            for conv in convs:
                used = 0
                for turn_index, turn in enumerate(conv["turns"]):
                    if turn["role"] != "user" or not turn["content"].strip():
                        continue
                    length = len(tokenizer.encode(turn["content"], add_special_tokens=False))
                    if length > TARGET_TOKENS:
                        continue
                    if used >= MAX_DECISIONS_PER_CONV:
                        break
                    record = {
                        "user_id": sid,
                        "conv_id": safe_id("c_", conv["hash"], 20),
                        "conv_hash": conv["hash"],
                        "ts": conv["ts"],
                        "turn_index": turn_index,
                        "split": split,
                        "length_tokens": length,
                        "target": turn["content"],
                        "terminal": not any(
                            later["role"] == "user" and later["content"].strip()
                            for later in conv["turns"][turn_index + 1 :]
                        ),
                    }
                    rows.append(record)
                    all_lengths.append(length)
                    if split == "train":
                        train_lengths.append(length)
                    used += 1
    rows.sort(key=lambda r: (r["user_id"], r["ts"], r["conv_hash"], r["turn_index"]))
    return rows, train_lengths, all_lengths


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def code_feature(text: str) -> int:
    if "```" in text:
        return 1
    matches = 0
    for line in text.splitlines():
        if re.search(r"^\s*(def|import|from|class|for|while|if)\b", line):
            matches += 1
        elif re.search(r"[{};]\s*$", line):
            matches += 1
    return int(matches >= 2)


def length_bucket(value: int, edges: list[float]) -> int:
    return max(0, min(4, int(np.digitize([value], edges, right=False)[0])))


def assign_folds(users: list[str], train_counts: dict[str, int]) -> dict[str, str]:
    ordered = sorted(users, key=lambda u: (train_counts[u], u))
    tertile = {user: min(2, 3 * i // len(ordered)) for i, user in enumerate(ordered)}
    result = {}
    rng = random.Random(SEED)
    for t in range(3):
        pool = [u for u in ordered if tertile[u] == t]
        rng.shuffle(pool)
        for i, user in enumerate(pool):
            result[user] = "A" if i % 2 == 0 else "B"
    return result


def filter_imposters(imposter: dict[str, str], eligible: list[str]) -> dict[str, str]:
    """Restrict the 240-user tertile rotation to the eligible users.

    Follows the rotation past any dropped user so the map stays a bijection
    over the eligible users without changing its activity-tertile ordering.
    """
    eligible_set = set(eligible)
    out = {}
    for user in eligible:
        candidate = imposter[user]
        seen = {user}
        while candidate not in eligible_set:
            if candidate in seen or candidate not in imposter:
                raise AssertionError(f"cannot restrict imposter cycle for {user}")
            seen.add(candidate)
            candidate = imposter[candidate]
        if candidate == user:
            raise AssertionError(f"self-imposter after filtering: {user}")
        out[user] = candidate
    if set(out) != eligible_set or set(out.values()) != eligible_set:
        raise AssertionError("filtered imposter map is not a bijection over eligible users")
    return out


def main() -> None:
    if not MANIFEST_PATH.exists() or not RUNTIME_PATH.exists():
        raise FileNotFoundError("panel manifest/runtime panel is missing; run build_panel first")
    manifest = json.loads(MANIFEST_PATH.read_text())
    runtime = json.loads(RUNTIME_PATH.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    rebuilt = rebuild_users(manifest, tokenizer)
    all_rows, train_lengths, _ = make_rows(rebuilt, tokenizer)

    train_counts_all = defaultdict(int)
    for row in all_rows:
        if row["split"] == "train":
            train_counts_all[row["user_id"]] += 1
    users = sorted(runtime["users"])
    eligible = [u for u in users if train_counts_all[u] >= MIN_PER_HALF]
    dropped = [
        {
            "user_id": u,
            "train_rows": int(train_counts_all[u]),
            "reason": "fewer than MIN_PER_HALF train-window labels",
        }
        for u in users
        if u not in set(eligible)
    ]
    if len(eligible) != EXPECTED_USERS:
        raise AssertionError(
            f"expected {EXPECTED_USERS} eligible users, rebuilt {len(eligible)}; dropped={dropped}"
        )

    eligible_set = set(eligible)
    train_eligible_lengths = [
        int(row["length_tokens"])
        for row in all_rows
        if row["user_id"] in eligible_set and row["split"] == "train"
    ]
    edges = [
        float(x)
        for x in np.quantile(np.asarray(train_eligible_lengths, dtype=float), [0.2, 0.4, 0.6, 0.8])
    ]
    rows = []
    for row in all_rows:
        if row["user_id"] not in eligible_set:
            continue
        rows.append(
            {
                "user_id": row["user_id"],
                "conv_id": row["conv_id"],
                "conv_hash": row["conv_hash"],
                "ts": row["ts"],
                "turn_index": int(row["turn_index"]),
                "split": row["split"],
                "length_tokens": int(row["length_tokens"]),
                "length": length_bucket(int(row["length_tokens"]), edges),
                "question": int("?" in row.get("target", "")),
                "code": code_feature(row.get("target", "")),
                "terminal": int(bool(row["terminal"])),
            }
        )

    train_counts = {
        u: int(sum(r["user_id"] == u and r["split"] == "train" for r in rows)) for u in eligible
    }
    if any(v < MIN_PER_HALF for v in train_counts.values()):
        raise AssertionError("eligible user has too few train rows after feature build")
    eligible_imposters = filter_imposters(runtime["imposter"], eligible)

    # Join the primary rows to their full-window labels.
    by_key = {(r["user_id"], r["conv_id"], r["turn_index"]): r for r in rows}
    primary = []
    for uid in users:
        if uid not in eligible_set:
            continue
        for decision in runtime["users"][uid]["decisions"]:
            if decision.get("split") != "primary":
                continue
            key = (uid, decision["conv_id"], int(decision["turn_index"]))
            source = by_key.get(key)
            if source is None:
                raise AssertionError(f"primary row missing from full window: {key}")
            primary.append(
                {
                    "user_id": uid,
                    "conv_id": decision["conv_id"],
                    "turn_index": int(decision["turn_index"]),
                    "split": "primary",
                    "cluster_id": decision["cluster_id"],
                    "prefix": decision["prefix"],
                    "target": decision["target"],
                    "target_tokens": int(source["length_tokens"]),
                    "length": int(source["length"]),
                    "question": int(source["question"]),
                    "code": int(source["code"]),
                    "terminal": int(source["terminal"]),
                }
            )
    if len(primary) == 0:
        raise AssertionError("no eligible primary rows")

    primary_by_user = {u: [i for i, r in enumerate(primary) if r["user_id"] == u] for u in eligible}
    row_splits = {}
    for u in eligible:
        indices = sorted(
            primary_by_user[u], key=lambda i: (primary[i]["conv_id"], primary[i]["turn_index"])
        )
        row_splits[u] = {
            "odd": indices[0::2],
            "even": indices[1::2],
        }
    folds = assign_folds(eligible, train_counts)
    row_meta = [
        {
            k: r[k]
            for k in (
                "user_id",
                "conv_id",
                "turn_index",
                "split",
                "cluster_id",
                "target_tokens",
                "length",
                "question",
                "code",
                "terminal",
            )
        }
        for r in primary
    ]
    all_meta = [
        {
            k: r[k]
            for k in (
                "user_id",
                "conv_id",
                "turn_index",
                "split",
                "length_tokens",
                "length",
                "question",
                "code",
                "terminal",
            )
        }
        for r in rows
    ]
    frame = {
        "schema": "wildchat-frame-v1",
        "seed": SEED,
        "panel_revision": runtime["meta"]["revision"],
        "n_panel_users": len(users),
        "n_users": len(eligible),
        "users": eligible,
        "dropped_users": dropped,
        "min_per_half_train_window": MIN_PER_HALF,
        "n_full_window_rows": len(rows),
        "full_window_row_digest": digest(all_meta),
        "n_primary_rows": len(primary),
        "primary_row_digest": digest(row_meta),
        "bucket_edges_from_train_tokens": edges,
        "train_counts": train_counts,
        "imposter_of": eligible_imposters,
        "fold_of_user": folds,
        "row_splits": row_splits,
        "primary_by_user": primary_by_user,
        "primary_rows": primary,
    }
    paths.ensure(paths.WILDCHAT)
    FRAME.write_text(json.dumps(frame, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "n_users": len(eligible),
                "dropped": dropped,
                "n_full_rows": len(rows),
                "n_primary_rows": len(primary),
                "edges": edges,
                "frame": str(FRAME),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
