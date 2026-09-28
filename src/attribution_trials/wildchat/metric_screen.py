"""Choose an external content metric on the real WildChat turns, before any generation.

For every primary turn, the metric's similarity to the target's card minus its
similarity to the imposter's card, averaged per user and bootstrapped over
users.  M1 is MiniLM cosine to the closest card line (mean pooling, L2
normalization, 256-token truncation); M2 is a sparse numpy implementation of
scikit-learn's default-smoothed ``char_wb`` TF-IDF on the whole card.  The
passing metric with the larger z is the primary metric.  Writes
``paths.WILDCHAT / "metric_screen.json"``; the module is also the embedding
and TF-IDF library for the generation readouts.

  python -m attribution_trials.wildchat.metric_screen
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from attribution_trials import paths

FRAME_PATH = paths.WILDCHAT / "frame.json"
PANEL_PATH = paths.WILDCHAT / "panel_runtime.json"
EMBED_MODEL = paths.MODELS / "all-MiniLM-L6-v2"
OUTPUT = paths.WILDCHAT / "metric_screen.json"

SEED = 0
NBOOT = 10_000
EMBED_BATCH = 64
MIN_USERS = 236
MIN_PRIMARY_ROWS = 1_427


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def split_card(card: str) -> list[str]:
    """Return non-empty newline-separated card messages without reformatting."""
    return [line.strip() for line in str(card).splitlines() if line.strip()]


def embed_many(texts: list[str]) -> tuple[np.ndarray, dict]:
    """MiniLM mean-pooled, L2-normalized embeddings, computed on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(str(EMBED_MODEL), local_files_only=True)
    model = AutoModel.from_pretrained(str(EMBED_MODEL), local_files_only=True).eval().to("cpu")
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), EMBED_BATCH):
            batch = tokenizer(
                texts[start : start + EMBED_BATCH],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
            done = min(start + EMBED_BATCH, len(texts))
            if done == len(texts) or done % (EMBED_BATCH * 25) == 0:
                print(f"MiniLM embedded {done}/{len(texts)}", flush=True)
    del model, tokenizer
    matrix = np.concatenate(vectors, axis=0) if vectors else np.empty((0, 384))
    return matrix, {
        "model_path": EMBED_MODEL.name,
        "pooling": "attention-mask mean pooling",
        "normalization": "L2",
        "max_length": 256,
        "device": "cpu",
        "n_texts": len(texts),
        "embedding_seconds": time.perf_counter() - started,
        "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def char_wb_ngrams(text: str) -> Counter[str]:
    """Generate the char_wb 3--5 grams for one lower-cased document."""
    counts: Counter[str] = Counter()
    for word in str(text).lower().split():
        padded = f" {word} "
        for n in range(3, 6):
            counts.update(padded[i : i + n] for i in range(len(padded) - n + 1))
    return counts


def build_tfidf(documents: list[str]) -> tuple[list[dict[int, float]], dict]:
    """Fit smoothed sublinear char_wb TF-IDF and return sparse unit vectors."""
    vocabulary: dict[str, int] = {}
    document_counts: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        counts = char_wb_ngrams(document)
        document_counts.append(counts)
        for gram in counts:
            if gram not in vocabulary:
                vocabulary[gram] = len(vocabulary)
            document_frequency[gram] += 1

    n_documents = len(documents)
    idf = np.ones(len(vocabulary), dtype=np.float64)
    for gram, index in vocabulary.items():
        idf[index] = math.log((1.0 + n_documents) / (1.0 + document_frequency[gram])) + 1.0

    vectors: list[dict[int, float]] = []
    for counts in document_counts:
        vector = {
            vocabulary[gram]: (1.0 + math.log(float(count))) * idf[vocabulary[gram]]
            for gram, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            vector = {index: value / norm for index, value in vector.items()}
        vectors.append(vector)
    return vectors, {
        "implementation": "numpy sparse equivalent of sklearn TfidfVectorizer",
        "analyzer": "char_wb",
        "ngram_range": [3, 5],
        "sublinear_tf": True,
        "normalization": "L2",
        "smooth_idf": True,
        "lowercase": True,
        "n_documents": n_documents,
        "n_features": len(vocabulary),
    }


def sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(value * b.get(index, 0.0) for index, value in a.items()))


def bootstrap(values: list[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if len(x) != MIN_USERS:
        raise AssertionError(f"expected {MIN_USERS} user values, got {len(x)}")
    rng = np.random.default_rng(SEED)
    draws = x[rng.integers(0, len(x), size=(NBOOT, len(x)))].mean(axis=1)
    point = float(x.mean())
    se = float(draws.std(ddof=1))
    return {
        "n_users": int(len(x)),
        "point": point,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "bootstrap_se": se,
        "z": float(point / se) if se > 0 else (float("inf") if point > 0 else 0.0),
        "pass": bool(np.percentile(draws, 2.5) > 0.0),
    }


def main() -> None:
    started = time.perf_counter()
    frame = json.loads(FRAME_PATH.read_text())
    panel = json.loads(PANEL_PATH.read_text())
    if frame.get("schema") != "wildchat-frame-v1":
        raise AssertionError(f"unexpected frame schema: {frame.get('schema')}")
    users = list(frame["users"])
    rows = list(frame["primary_rows"])
    if len(users) != MIN_USERS or frame.get("n_users") != MIN_USERS:
        raise AssertionError(f"expected {MIN_USERS} users, got {len(users)}")
    if len(rows) != MIN_PRIMARY_ROWS or frame.get("n_primary_rows") != MIN_PRIMARY_ROWS:
        raise AssertionError(f"expected {MIN_PRIMARY_ROWS} rows, got {len(rows)}")
    if any(row.get("split") != "primary" for row in rows):
        raise AssertionError("metric screen received a non-primary row")

    panel_users = panel["users"]
    cards = {uid: str(panel_users[uid]["profile_card"]) for uid in users}
    imposter = {uid: str(frame["imposter_of"][uid]) for uid in users}
    if any(uid not in panel_users or not cards[uid] for uid in users):
        raise AssertionError("missing non-empty own card in runtime panel")
    if any(imposter[uid] not in cards or not cards[imposter[uid]] for uid in users):
        raise AssertionError("missing non-empty imposter card in runtime panel")
    if any(cards[uid] == cards[imposter[uid]] for uid in users):
        raise AssertionError("identical own/imposter card")

    targets = [str(row["target"]) for row in rows]
    bullets = {uid: split_card(cards[uid]) for uid in users}
    if any(not bullets[uid] for uid in users):
        raise AssertionError("an own card has no non-empty newline bullet")

    print(f"frame users={len(users)} primary_rows={len(rows)}", flush=True)
    mini_texts = unique_in_order(targets + [line for uid in users for line in bullets[uid]])
    mini_matrix, mini_meta = embed_many(mini_texts)
    mini_vectors = {text: mini_matrix[i] for i, text in enumerate(mini_texts)}

    mini_row_diffs: list[float] = []
    tfidf_row_diffs: list[float] = []
    mini_by_user: dict[str, list[float]] = defaultdict(list)
    tfidf_by_user: dict[str, list[float]] = defaultdict(list)

    card_documents = [cards[uid] for uid in users]
    tfidf_documents = card_documents + targets
    tfidf_vectors, tfidf_meta = build_tfidf(tfidf_documents)
    tfidf_card_vectors = {uid: tfidf_vectors[i] for i, uid in enumerate(users)}
    tfidf_target_vectors = {
        index: tfidf_vectors[len(users) + index] for index in range(len(targets))
    }

    for index, row in enumerate(rows):
        uid = str(row["user_id"])
        target_vector = mini_vectors[targets[index]]
        own_mini = max(cosine(target_vector, mini_vectors[line]) for line in bullets[uid])
        imp_uid = imposter[uid]
        imp_bullets = split_card(cards[imp_uid])
        imp_mini = max(cosine(target_vector, mini_vectors[line]) for line in imp_bullets)
        mini_diff = own_mini - imp_mini
        mini_row_diffs.append(mini_diff)
        mini_by_user[uid].append(mini_diff)

        own_tfidf = sparse_cosine(tfidf_target_vectors[index], tfidf_card_vectors[uid])
        imp_tfidf = sparse_cosine(tfidf_target_vectors[index], tfidf_card_vectors[imp_uid])
        tfidf_diff = own_tfidf - imp_tfidf
        tfidf_row_diffs.append(tfidf_diff)
        tfidf_by_user[uid].append(tfidf_diff)

    mini_user_values = [float(np.mean(mini_by_user[uid])) for uid in users]
    tfidf_user_values = [float(np.mean(tfidf_by_user[uid])) for uid in users]
    metric_results = {
        "M1_minilm_max_card_bullet": {
            "description": "MiniLM cosine(real turn, max card-bullet cosine)",
            "screen": bootstrap(mini_user_values),
            "row_mean": float(np.mean(mini_row_diffs)),
            "n_card_bullets": int(sum(len(bullets[uid]) for uid in users)),
            "implementation": mini_meta,
        },
        "M2_char_wb_tfidf_whole_card": {
            "description": "char_wb TF-IDF cosine(real turn, whole card)",
            "screen": bootstrap(tfidf_user_values),
            "row_mean": float(np.mean(tfidf_row_diffs)),
            "implementation": tfidf_meta,
        },
    }
    passing = [name for name, value in metric_results.items() if value["screen"]["pass"]]
    primary = (
        max(passing, key=lambda name: metric_results[name]["screen"]["z"]) if passing else None
    )
    out = {
        "schema": "wildchat-metric-screen-v1",
        "frame_schema": frame["schema"],
        "frame_primary_digest": frame["primary_row_digest"],
        "frame_full_window_digest": frame["full_window_row_digest"],
        "panel_revision": frame["panel_revision"],
        "n_users": len(users),
        "n_primary_rows": len(rows),
        "users": users,
        "card_source": "runtime panel profile_card; own and frame imposter cards",
        "bootstrap": {"n_resamples": NBOOT, "seed": SEED, "unit": "user"},
        "metrics": metric_results,
        "passing_metrics": passing,
        "primary_metric": primary,
        "per_user": {
            uid: {
                "n_rows": len(mini_by_user[uid]),
                "M1_minilm_max_card_bullet": mini_user_values[i],
                "M2_char_wb_tfidf_whole_card": tfidf_user_values[i],
            }
            for i, uid in enumerate(users)
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cpu_only": True,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    paths.ensure(paths.WILDCHAT)
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passing_metrics": passing,
                "primary_metric": primary,
                "metrics": {name: value["screen"] for name, value in metric_results.items()},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
