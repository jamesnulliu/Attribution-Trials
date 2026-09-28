"""Build the 240-user WildChat panel with intent-cluster out-of-distribution turns.

The build is deterministic:

* WildChat-1M revision and filters are pinned.
* duplicate conversations are removed by conversation_hash; repeated prompts
  are removed exactly or by a high-cosine/high-token-overlap rule.
* ``all-MiniLM-L6-v2`` is loaded from the local model directory.
* per-user greedy cosine clustering is chronological and threshold-pinned.
* user sampling and imposter rotation use seed 0.

Writes under ``paths.WILDCHAT``: ``panel.json`` (manifest without conversation
or profile text), ``panel_runtime.json`` (cards, prefixes, targets, imposter
map) and ``panel_build.sqlite`` (all candidate-user conversations).

  python -m attribution_trials.wildchat.build_panel
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

from attribution_trials import paths

RAW = paths.DATA / "wildchat"
DB_PATH = paths.WILDCHAT / "panel_build.sqlite"
MANIFEST_PATH = paths.WILDCHAT / "panel.json"
RUNTIME_PATH = paths.WILDCHAT / "panel_runtime.json"

DATASET = "allenai/WildChat-1M"
REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
EMBEDDING_MODEL = paths.MODELS / "all-MiniLM-L6-v2"
TOKENIZER_MODEL = paths.MODELS / "Qwen3-1.7B"
SEED = 0

MIN_CONVS = 10
MIN_DAYS = 3
MIN_PRIMARY_CONVS = 2
TARGET_PANEL_USERS = 240
EMBED_BATCH = 64
EMBED_MAX_LENGTH = 256
CLUSTER_COSINE = 0.72
NEAR_DUP_COSINE = 0.985
NEAR_DUP_JACCARD = 0.75
PROFILE_TOKENS = 1200
TARGET_TOKENS = 512
MAX_DECISIONS_PER_CONV = 6
MAX_PRIMARY_CLUSTERS = 2


def safe_id(prefix: str, value: str, n: int = 16) -> str:
    # The salt fixes the user and conversation ids, and with them every sort order.
    return prefix + hashlib.sha256(("e36:" + str(value)).encode()).hexdigest()[:n]


def day_of(value) -> str:
    return str(value)[:10]


def ts_key(value: str):
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value


def norm_text(text: str) -> str:
    text = text.lower().replace("​", " ")
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", norm_text(text), flags=re.UNICODE))


def token_jaccard(a: str, b: str) -> float:
    aa, bb = token_set(a), token_set(b)
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(1, len(aa | bb))


def compact_turns(conversation) -> list[dict[str, str]]:
    out = []
    for turn in conversation or []:
        role = str(turn.get("role", "user"))
        content = turn.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        if content.strip():
            out.append({"role": role, "content": content})
    return out


def has_user_turn(turns: list[dict[str, str]]) -> bool:
    return any(t["role"] == "user" and t["content"].strip() for t in turns)


def row_ok(row: dict) -> bool:
    return bool(
        row.get("language") == "English"
        and not row.get("toxic")
        and not row.get("redacted")
        and row.get("hashed_ip")
        and int(row.get("turn") or 0) >= 1
    )


def parquet_paths() -> list[Path]:
    found = sorted(RAW.glob("train-*-of-00014.parquet"))
    if len(found) != 14:
        raise FileNotFoundError(
            f"expected 14 WildChat shards under {RAW}, found {len(found)}; "
            "run scripts/download/wildchat.sh first"
        )
    return found


def iter_rows(columns: list[str] | None = None) -> Iterable[dict]:
    for path in parquet_paths():
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            yield from batch.to_pylist()


def first_pass() -> tuple[set[str], dict[str, int], dict[str, set[str]]]:
    counts: Counter[str] = Counter()
    days: dict[str, set[str]] = defaultdict(set)
    seen_hash: set[str] = set()
    columns = [
        "conversation_hash",
        "timestamp",
        "turn",
        "language",
        "toxic",
        "redacted",
        "hashed_ip",
    ]
    n_rows = 0
    for row in iter_rows(columns):
        n_rows += 1
        if not row_ok(row):
            continue
        ch = str(row.get("conversation_hash") or "")
        uid = str(row["hashed_ip"])
        if not ch or ch in seen_hash:
            continue
        seen_hash.add(ch)
        counts[uid] += 1
        days[uid].add(day_of(row.get("timestamp")))
    candidates = {
        uid for uid, count in counts.items() if count >= MIN_CONVS and len(days[uid]) >= MIN_DAYS
    }
    print(
        f"metadata rows={n_rows}; unique qualifying conversations="
        f"{sum(counts.values())}; candidate users={len(candidates)}",
        flush=True,
    )
    return candidates, dict(counts), days


def materialize_candidates(candidates: set[str]) -> sqlite3.Connection:
    paths.ensure(paths.WILDCHAT)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE conv (uid TEXT NOT NULL, conv_hash TEXT PRIMARY KEY, "
        "ts TEXT NOT NULL, model TEXT, country TEXT, turns TEXT NOT NULL)"
    )
    con.execute("CREATE INDEX conv_uid_idx ON conv(uid)")
    columns = [
        "conversation_hash",
        "timestamp",
        "conversation",
        "model",
        "country",
        "turn",
        "language",
        "toxic",
        "redacted",
        "hashed_ip",
    ]
    inserted = 0
    for row in iter_rows(columns):
        if not row_ok(row):
            continue
        uid = str(row["hashed_ip"])
        if uid not in candidates:
            continue
        turns = compact_turns(row.get("conversation"))
        if not has_user_turn(turns):
            continue
        ch = str(row.get("conversation_hash") or "")
        if not ch:
            continue
        con.execute(
            "INSERT OR IGNORE INTO conv(uid,conv_hash,ts,model,country,turns) VALUES (?,?,?,?,?,?)",
            (
                uid,
                ch,
                str(row.get("timestamp")),
                str(row.get("model") or ""),
                str(row.get("country") or ""),
                json.dumps(turns, ensure_ascii=False),
            ),
        )
        inserted += 1
        if inserted % 5000 == 0:
            con.commit()
            print(f"materialized {inserted} candidate conversations", flush=True)
    con.commit()
    print(f"materialized {inserted} candidate conversations into {DB_PATH}", flush=True)
    return con


def load_users(con: sqlite3.Connection) -> dict[str, list[dict]]:
    users: dict[str, list[dict]] = defaultdict(list)
    for uid, ch, ts, model, country, turns_json in con.execute(
        "SELECT uid,conv_hash,ts,model,country,turns FROM conv"
    ):
        users[uid].append(
            {
                "hash": ch,
                "ts": ts,
                "model": model,
                "country": country,
                "turns": json.loads(turns_json),
            }
        )
    for convs in users.values():
        convs.sort(key=lambda x: (ts_key(x["ts"]), x["hash"]))
    return dict(users)


def representative(c: dict) -> str:
    user_turns = [t["content"] for t in c["turns"] if t["role"] == "user"]
    return "\n".join(user_turns[:2])[:6000]


def embed_texts(texts: list[str]) -> np.ndarray:
    if not EMBEDDING_MODEL.exists():
        raise FileNotFoundError(f"embedding model missing: {EMBEDDING_MODEL}")
    tok = AutoTokenizer.from_pretrained(str(EMBEDDING_MODEL), local_files_only=True)
    model = AutoModel.from_pretrained(str(EMBEDDING_MODEL), local_files_only=True)
    model.eval().to("cpu")
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), EMBED_BATCH):
            batch_text = texts[start : start + EMBED_BATCH]
            batch = tok(
                batch_text,
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LENGTH,
                return_tensors="pt",
            )
            out = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
            done = min(start + EMBED_BATCH, len(texts))
            if done % (EMBED_BATCH * 25) == 0 or done == len(texts):
                print(f"embedded {done}/{len(texts)} prompts", flush=True)
    del model
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 384), dtype=np.float32)


def assign_clusters(convs: list[dict]) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    centroids: list[np.ndarray] = []
    for c in convs:
        vec = c["vec"]
        if not centroids:
            best, score = None, -1.0
        else:
            scores = [float(np.dot(vec, center)) for center in centroids]
            best = int(np.argmax(scores))
            score = scores[best]
        if best is None or score < CLUSTER_COSINE:
            best = len(clusters)
            clusters.append([])
            centroids.append(vec.copy())
        clusters[best].append(c)
        total = sum(x["vec"] for x in clusters[best])
        centroids[best] = total / max(np.linalg.norm(total), 1e-12)
        c["cluster"] = f"c{best:02d}"
    return clusters


def dedup_and_cluster(convs: list[dict]) -> tuple[list[dict], dict[str, int]]:
    kept: list[dict] = []
    exact_dropped = 0
    near_dropped = 0
    seen_norm: set[str] = set()
    for c in convs:
        rep = c["rep"]
        nr = norm_text(rep)
        if not nr or nr in seen_norm:
            exact_dropped += 1
            continue
        duplicate = False
        for old in kept:
            cosine = float(np.dot(c["vec"], old["vec"]))
            if cosine >= NEAR_DUP_COSINE and token_jaccard(rep, old["rep"]) >= NEAR_DUP_JACCARD:
                duplicate = True
                break
        if duplicate:
            near_dropped += 1
            continue
        seen_norm.add(nr)
        kept.append(c)
    assign_clusters(kept)
    return kept, {"exact": exact_dropped, "near": near_dropped}


def filter_profile_collisions(train: list[dict], ev: list[dict]) -> tuple[list[dict], int]:
    """Remove whole eval conversations containing a train-window user turn.

    Conversation-level deduplication cannot catch a repeated later user turn
    when the conversation representatives differ, so the affected evaluation
    conversation is removed before the OOD clusters are selected.
    """
    train_norm = {
        norm_text(turn["content"])
        for conv in train
        for turn in conv["turns"]
        if turn["role"] == "user" and turn["content"].strip()
    }
    kept: list[dict] = []
    dropped = 0
    for conv in ev:
        collides = any(
            turn["role"] == "user" and norm_text(turn["content"]) in train_norm
            for turn in conv["turns"]
        )
        if collides:
            dropped += 1
        else:
            kept.append(conv)
    return kept, dropped


def make_card(convs: list[dict], qtok) -> tuple[str, int]:
    lines: list[str] = []
    budget = PROFILE_TOKENS
    for c in reversed(convs):
        for turn in c["turns"]:
            if turn["role"] != "user" or not turn["content"].strip():
                continue
            line = "- " + " ".join(turn["content"].split())[:2000]
            cost = len(qtok.encode(line, add_special_tokens=False)) + 1
            if cost > budget:
                continue
            lines.append(line)
            budget -= cost
        if budget < 40:
            break
    return "\n".join(lines), PROFILE_TOKENS - budget


def make_decisions(c: dict, split: str, qtok) -> list[dict]:
    out = []
    for index, turn in enumerate(c["turns"]):
        if turn["role"] != "user" or not turn["content"].strip():
            continue
        if len(qtok.encode(turn["content"], add_special_tokens=False)) > TARGET_TOKENS:
            continue
        out.append(
            {
                "conv_id": safe_id("c_", c["hash"], 20),
                "turn_index": index,
                "split": split,
                "cluster_id": c["cluster"],
                "prefix": c["turns"][:index],
                "target": turn["content"],
                "ts": c["ts"],
                "is_terminal": not any(
                    later["role"] == "user" and later["content"].strip()
                    for later in c["turns"][index + 1 :]
                ),
            }
        )
        if len(out) >= MAX_DECISIONS_PER_CONV:
            break
    return out


def choose_primary(
    convs: list[dict], train: list[dict], ev: list[dict]
) -> tuple[list[dict], list[str]]:
    train_clusters = {c["cluster"] for c in train}
    eval_only = [c for c in ev if c["cluster"] not in train_clusters]
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for c in eval_only:
        by_cluster[c["cluster"]].append(c)
    ranked = sorted(
        by_cluster.items(),
        key=lambda kv: (-len(kv[1]), min(ts_key(c["ts"]) for c in kv[1]), kv[0]),
    )
    chosen: list[str] = []
    n_convs = 0
    for cluster, members in ranked:
        chosen.append(cluster)
        n_convs += len(members)
        if n_convs >= MIN_PRIMARY_CONVS or len(chosen) >= MAX_PRIMARY_CLUSTERS:
            break
    primary = [c for c in ev if c["cluster"] in set(chosen)]
    return primary, chosen


def build_user(uid: str, convs: list[dict], qtok) -> dict | None:
    if len(convs) < MIN_CONVS:
        return None
    n_train = max(6, int(math.floor(0.6 * len(convs) + 0.5)))
    if len(convs) - n_train < 4:
        n_train = len(convs) - 4
    if n_train < 6:
        return None
    train, ev = convs[:n_train], convs[n_train:]
    ev, exact_ood_dropped = filter_profile_collisions(train, ev)
    primary, primary_clusters = choose_primary(convs, train, ev)
    if len(primary) < MIN_PRIMARY_CONVS:
        return None
    card, card_tokens = make_card(train, qtok)
    if not card:
        return None
    primary_decisions = []
    for c in primary:
        primary_decisions.extend(make_decisions(c, "primary", qtok))
    primary_with_decisions = {d["conv_id"] for d in primary_decisions}
    if len(primary_with_decisions) < MIN_PRIMARY_CONVS:
        return None
    return {
        "uid": uid,
        "safe_uid": safe_id("u_", uid),
        "convs": convs,
        "train": train,
        "eval": ev,
        "primary": primary,
        "primary_clusters": primary_clusters,
        "card": card,
        "card_tokens": card_tokens,
        "primary_decisions": primary_decisions,
        "exact_ood_dropped": exact_ood_dropped,
    }


def activity_tertiles(uids: list[str]) -> dict[str, int]:
    # Callers provide this list already sorted by activity.
    n = len(uids)
    return {u: min(2, 3 * i // max(n, 1)) for i, u in enumerate(uids)}


def select_users(eligible: dict[str, dict]) -> list[str]:
    ordered = sorted(eligible, key=lambda u: (len(eligible[u]["convs"]), u))
    tert = activity_tertiles(ordered)
    target = min(TARGET_PANEL_USERS, len(ordered))
    rng = random.Random(SEED)
    picked: list[str] = []
    for t in (0, 1, 2):
        pool = [u for u in ordered if tert[u] == t]
        take = target // 3 + (1 if t < target % 3 else 0)
        picked.extend(rng.sample(pool, min(take, len(pool))))
    # If a very uneven population caused a short stratum, fill deterministically.
    if len(picked) < target:
        rest = [u for u in ordered if u not in set(picked)]
        picked.extend(rng.sample(rest, min(target - len(picked), len(rest))))
    return sorted(picked)


def make_imposters(
    selected: list[str], info: dict[str, dict], tert: dict[str, int]
) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in (0, 1, 2):
        pool = sorted(u for u in selected if tert[u] == t)
        if len(pool) < 2:
            continue
        rotated = pool[1:] + pool[:1]
        for left, right in zip(pool, rotated):
            if info[left].get("language") != info[right].get("language"):
                raise RuntimeError("imposter language mismatch")
            out[safe_id("u_", left)] = safe_id("u_", right)
    return out


def write_outputs(
    selected: list[str],
    eligible: dict[str, dict],
    all_summaries: dict,
    tert_all: dict[str, int],
):
    info = {u: eligible[u] for u in selected}
    tert = {safe_id("u_", u): tert_all[u] for u in selected}
    imposters = make_imposters(selected, info, tert_all)

    runtime_users = {}
    manifest_users = {}
    for raw_uid in selected:
        x = eligible[raw_uid]
        sid = safe_id("u_", raw_uid)
        train = x["train"]
        primary = x["primary"]
        pdec = x["primary_decisions"]
        train_conv_ids = {safe_id("c_", c["hash"], 20) for c in train}
        primary_conv_ids = {safe_id("c_", c["hash"], 20) for c in primary}
        runtime_users[sid] = {
            "n_convs": len(x["convs"]),
            "n_train": len(train),
            "n_eval_convs": len(x["eval"]),
            "profile_card": x["card"],
            "profile_card_tokens": x["card_tokens"],
            "profile_conv_ids": sorted(train_conv_ids),
            "decisions": pdec,
        }
        clusters = []
        for cluster_id, members in sorted(
            ((k, list(v)) for k, v in _group_by_cluster(x["convs"]).items()),
        ):
            train_n = sum(c in train for c in members)
            primary_n = sum(c in primary for c in members)
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "n_convs": len(members),
                    "n_train": train_n,
                    "n_primary_ood": primary_n,
                }
            )
        manifest_users[sid] = {
            "n_convs": len(x["convs"]),
            "n_train": len(train),
            "n_eval_convs": len(x["eval"]),
            "n_primary_ood_convs": len(primary),
            "n_primary_ood_clusters": len(x["primary_clusters"]),
            "n_primary_decisions": len(pdec),
            "profile_card_tokens": x["card_tokens"],
            "profile_conv_ids": sorted(train_conv_ids),
            "primary_conv_ids": sorted(primary_conv_ids),
            "clusters": clusters,
        }

    meta = {
        "dataset": DATASET,
        "revision": REVISION,
        "filters": {
            "language": "English",
            "toxic": False,
            "redacted": False,
            "non_empty_user_turn": True,
        },
        "seed": SEED,
        "min_convs": MIN_CONVS,
        "min_days": MIN_DAYS,
        "split": {"train_fraction": 0.60, "minimum_train_convs": 6, "minimum_eval_convs": 4},
        "embedding_model": EMBEDDING_MODEL.name,
        "embedding_max_length": EMBED_MAX_LENGTH,
        "cluster": {
            "algorithm": "chronological greedy cosine centroid",
            "cosine_threshold": CLUSTER_COSINE,
        },
        "dedup": {
            "conversation_key": "conversation_hash",
            "near_cosine": NEAR_DUP_COSINE,
            "near_token_jaccard": NEAR_DUP_JACCARD,
        },
        "profile": {
            "source": "verbatim train-window user turns",
            "tokenizer": TOKENIZER_MODEL.name,
            "max_tokens": PROFILE_TOKENS,
        },
        "decision": {
            "target_max_tokens": TARGET_TOKENS,
            "max_decisions_per_conversation": MAX_DECISIONS_PER_CONV,
        },
        "sample": {
            "target_users": TARGET_PANEL_USERS,
            "selected_users": len(selected),
            "sampling": "activity-tertile stratified",
            "seed": SEED,
        },
        "n_candidate_users_before_cluster_gate": len(all_summaries),
        "n_eligible_users_after_ood_gate": len(eligible),
        "n_panel_users": len(selected),
    }
    panel = {
        "meta": meta,
        "tertile": tert,
        "imposter": imposters,
        "users": manifest_users,
    }
    runtime = {
        "meta": {**meta, "runtime_raw_text": True},
        "tertile": tert,
        "imposter": imposters,
        "users": runtime_users,
    }
    paths.ensure(paths.WILDCHAT)
    MANIFEST_PATH.write_text(json.dumps(panel, indent=2) + "\n")
    RUNTIME_PATH.write_text(json.dumps(runtime, ensure_ascii=False, separators=(",", ":")))
    print(
        "panel users",
        len(selected),
        "primary decisions",
        sum(x["n_primary_decisions"] for x in manifest_users.values()),
        flush=True,
    )


def _group_by_cluster(convs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for c in convs:
        out[c["cluster"]].append(c)
    return out


def main() -> None:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    shards = parquet_paths()
    print(f"using {len(shards)} pinned WildChat shards", flush=True)
    candidates, _, _ = first_pass()
    con = materialize_candidates(candidates)
    users = load_users(con)
    print(f"loaded {len(users)} candidate users for clustering", flush=True)

    items = []
    for uid, convs in users.items():
        for c in convs:
            c["rep"] = representative(c)
            items.append((uid, c["hash"], c["rep"]))
    vectors = embed_texts([x[2] for x in items])
    vec_by_hash = {ch: vectors[i] for i, (_, ch, _) in enumerate(items)}
    del vectors

    eligible: dict[str, dict] = {}
    all_summaries = {}
    qtok = AutoTokenizer.from_pretrained(str(TOKENIZER_MODEL), local_files_only=True)
    for index, (uid, convs) in enumerate(sorted(users.items())):
        for c in convs:
            c["vec"] = vec_by_hash[c["hash"]]
        convs, _ = dedup_and_cluster(convs)
        if len(convs) < MIN_CONVS:
            continue
        cluster_count = len(_group_by_cluster(convs))
        meaningful = sum(
            1
            for members in _group_by_cluster(convs).values()
            if any(len(c["rep"].split()) >= 4 and len(c["rep"]) >= 20 for c in members)
        )
        all_summaries[uid] = {"n_convs": len(convs), "clusters": cluster_count}
        if meaningful < 2:
            continue
        x = build_user(uid, convs, qtok)
        if x is not None:
            x["language"] = "English"
            eligible[uid] = x
        if (index + 1) % 250 == 0:
            print(f"clustered {index + 1}/{len(users)} users; eligible={len(eligible)}", flush=True)
    print(f"eligible users before panel sampling: {len(eligible)}", flush=True)
    if not eligible:
        raise RuntimeError("no eligible users after intent/OOD gates")
    selected = select_users(eligible)
    ordered_eligible = sorted(eligible, key=lambda u: (len(eligible[u]["convs"]), u))
    tert_all = activity_tertiles(ordered_eligible)
    write_outputs(selected, eligible, all_summaries, tert_all)
    con.close()


if __name__ == "__main__":
    main()
