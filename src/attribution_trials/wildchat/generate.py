"""Sample A3/A4 continuations for one simulator and score them in content space.

K=8 samples per primary turn and arm (imposter's card A3, target's card A4),
temperature 1, 96 new tokens, seed 0.  Each sample's distance to the real next
turn is scored with MiniLM (M1) and char_wb TF-IDF (M2); the same samples also
get form labels (length bucket, question, code, terminal).

Writes under ``RESULTS`` (default ``paths.WILDCHAT``): ``content_<c>.json``
(per-row distances), ``form_<c>.json`` (per-sample labels and per-row
histograms) and ``samples/samples_<c>.jsonl.gz`` (sample text).

  GENERATION_BATCH=4 python -m attribution_trials.wildchat.generate humanlike
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import platform
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from attribution_trials import paths
from attribution_trials.wildchat.common import (
    CANDIDATES,
    SIMULATORS,
    card_for,
    feature_code,
    generation_context,
    load_local_model,
    parse_generated,
    row_key,
)
from attribution_trials.wildchat.metric_screen import (
    char_wb_ngrams,
    cosine,
    embed_many,
    sparse_cosine,
    unique_in_order,
)

FRAME_PATH = paths.WILDCHAT / "frame.json"
PANEL_PATH = paths.WILDCHAT / "panel_runtime.json"
METRIC_SCREEN_PATH = paths.WILDCHAT / "metric_screen.json"
REFERENCE_TOKENIZER = paths.MODELS / "Qwen3-1.7B"
# Output locations; temperature.py points them at a separate directory.
RESULTS = paths.WILDCHAT
SAMPLES = paths.WILDCHAT / "samples"

SEED = 0
K_SAMPLES = 8
MAX_NEW_TOKENS = 96
TEMPERATURE = 1.0
ARMS = ("A3", "A4")
DIMENSIONS = ("length", "question", "code", "terminal")
METRIC_M1 = "M1_minilm_max_card_bullet"
METRIC_M2 = "M2_char_wb_tfidf_whole_card"


def trim_at_eos(ids: list[int], eos: int | None) -> list[int]:
    if eos is None or eos not in ids:
        return ids
    return ids[: ids.index(eos) + 1]


def sample_continuations(model, tok, prompts: list[list[int]]) -> list[list[int]]:
    pad = tok.pad_token_id
    eos = tok.eos_token_id
    if pad is None:
        pad = eos
    if pad is None:
        raise RuntimeError("tokenizer has no pad or EOS token")
    # Batch size changes padding and therefore the sampling RNG stream.
    batch_size = int(os.environ.get("GENERATION_BATCH", "8"))
    if batch_size < 1:
        raise ValueError("GENERATION_BATCH must be positive")
    outputs: list[list[int]] = []
    started = time.time()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        width = max(len(item) for item in batch)
        input_ids = torch.full((len(batch), width), pad, dtype=torch.long, device="cuda")
        attention = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        for i, item in enumerate(batch):
            input_ids[i, width - len(item) :] = torch.tensor(item, dtype=torch.long, device="cuda")
            attention[i, width - len(item) :] = 1
        with torch.inference_mode():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=True,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=eos,
                pad_token_id=pad,
                use_cache=True,
            )
        for sequence in sequences.detach().cpu().tolist():
            outputs.append(trim_at_eos(sequence[width:], eos))
        del sequences, input_ids, attention
        end = start + len(batch)
        if end == len(prompts) or end % max(batch_size * 16, 1) == 0:
            print(
                f"  generated {end}/{len(prompts)} "
                f"({end / max(time.time() - started, 1e-9):.2f} prompts/s)",
                flush=True,
            )
    return outputs


def length_bucket(value: int, edges: list[float]) -> int:
    return max(0, min(4, int(np.digitize([value], np.asarray(edges), right=False)[0])))


def write_gzip_samples(
    candidate: str, rows: list[dict], samples: dict[str, list[list[dict]]]
) -> tuple[Path, int]:
    path = SAMPLES / f"samples_{candidate}.jsonl.gz"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    SAMPLES.mkdir(parents=True, exist_ok=True)
    with gzip.open(tmp, "wt", encoding="utf-8", newline="\n") as handle:
        for row_index, row in enumerate(rows):
            key = row_key(row["user_id"], row)
            for arm in ARMS:
                for k, sample in enumerate(samples[arm][row_index]):
                    record = {
                        "key": key,
                        "arm": arm,
                        "k": k,
                        "text": sample["text"],
                        "generated_tokens": int(sample["generated_tokens"]),
                        "terminal": bool(sample["terminal"]),
                    }
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    count += 1
    os.replace(tmp, path)
    return path, count


def sample_arm(
    model, tokenizer, ref_tokenizer, candidate: str, panel: dict, frame: dict, arm: str
) -> tuple[list[list[dict]], list[list[dict]]]:
    rows = frame["primary_rows"]
    prompts: list[list[int]] = []
    for row in rows:
        # The card comes from the 240-user panel imposter map, as in the scoring runs.
        card = card_for(panel, row["user_id"], arm)
        prompt, _ = generation_context(tokenizer, candidate, card, row["prefix"])
        prompts.extend([prompt] * K_SAMPLES)
    generated = sample_continuations(model, tokenizer, prompts)
    if len(generated) != len(rows) * K_SAMPLES:
        raise AssertionError(f"{candidate}/{arm}: generation count mismatch")

    edges = frame["bucket_edges_from_train_tokens"]
    samples: list[list[dict]] = []
    labels: list[list[dict]] = []
    for row_index in range(len(rows)):
        row_samples: list[dict] = []
        row_labels: list[dict] = []
        for ids in generated[row_index * K_SAMPLES : (row_index + 1) * K_SAMPLES]:
            text, terminal = parse_generated(candidate, tokenizer, ids)
            ref_tokens = len(ref_tokenizer.encode(text, add_special_tokens=False))
            label = {
                "generated_tokens": len(ids),
                "realized_tokens_qwen3_1_7b": ref_tokens,
                "length": length_bucket(ref_tokens, edges),
                "question": int("?" in text),
                "code": int(feature_code(text)),
                "terminal": int(terminal),
            }
            row_samples.append(
                {
                    "text": text,
                    "generated_tokens": len(ids),
                    "terminal": bool(terminal),
                }
            )
            row_labels.append(label)
        samples.append(row_samples)
        labels.append(row_labels)
        if row_index == 0 or (row_index + 1) % 64 == 0 or row_index + 1 == len(rows):
            print(
                f"[{candidate}/{arm}] generated {(row_index + 1) * K_SAMPLES}/{len(rows) * K_SAMPLES}",
                flush=True,
            )
    return samples, labels


def form_histogram(samples: list[list[dict]], dimension: str, n_labels: int) -> list[list[float]]:
    output: list[list[float]] = []
    for row_samples in samples:
        counts = (
            np.bincount(
                np.asarray([sample[dimension] for sample in row_samples], dtype=int),
                minlength=n_labels,
            ).astype(float)
            + 0.5
        )
        output.append((counts / counts.sum()).tolist())
    return output


def fit_tfidf_transform(documents: list[str]):
    """Fit M2 on cards and real turns; return a frozen transform."""
    counts_by_document = [char_wb_ngrams(document) for document in documents]
    vocabulary: dict[str, int] = {}
    document_frequency: Counter[str] = Counter()
    for counts in counts_by_document:
        for gram in counts:
            if gram not in vocabulary:
                vocabulary[gram] = len(vocabulary)
            document_frequency[gram] += 1

    n_documents = len(documents)
    idf = np.ones(len(vocabulary), dtype=np.float64)
    for gram, index in vocabulary.items():
        idf[index] = math.log((1.0 + n_documents) / (1.0 + document_frequency[gram])) + 1.0

    def transform(text: str) -> dict[int, float]:
        counts = char_wb_ngrams(text)
        vector = {
            vocabulary[gram]: (1.0 + math.log(float(count))) * idf[vocabulary[gram]]
            for gram, count in counts.items()
            if gram in vocabulary
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {index: value / norm for index, value in vector.items()} if norm else {}

    return transform, {
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


def build_form_output(
    candidate: str,
    spec: dict,
    frame: dict,
    sampled_labels: dict[str, list[list[dict]]],
    histograms: dict[str, dict[str, list[list[float]]]],
) -> dict:
    rows = frame["primary_rows"]
    return {
        "schema": "wildchat-form-v1",
        "candidate": candidate,
        "model": spec["model"],
        "model_revision": spec["revision"],
        "interface": spec["interface"],
        "panel_revision": frame["panel_revision"],
        "frame_schema": frame["schema"],
        "frame_full_window_digest": frame["full_window_row_digest"],
        "frame_primary_digest": frame["primary_row_digest"],
        "n_users": frame["n_users"],
        "n_primary_rows": frame["n_primary_rows"],
        "users": frame["users"],
        "arms": list(ARMS),
        "dimensions": list(DIMENSIONS),
        "seed": SEED,
        "k_samples": K_SAMPLES,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "smooth": 0.5,
        "terminal_convention": "first generated EOS or empty decoded continuation",
        "length_feature_tokenizer": "Qwen/Qwen3-1.7B local reference tokenizer",
        "realized_labels": {
            dimension: [int(row[dimension]) for row in rows] for dimension in DIMENSIONS
        },
        "sampled_labels": sampled_labels,
        "histograms": histograms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=SIMULATORS)
    args = parser.parse_args()
    candidate = args.candidate
    frame = json.loads(FRAME_PATH.read_text())
    panel = json.loads(PANEL_PATH.read_text())
    if frame.get("schema") != "wildchat-frame-v1":
        raise AssertionError(f"unexpected frame schema: {frame.get('schema')}")
    if frame.get("n_users") != 236 or frame.get("n_primary_rows") != 1427:
        raise AssertionError("frame is not 236 users / 1,427 rows")
    rows = frame["primary_rows"]
    if any(row.get("split") != "primary" for row in rows):
        raise AssertionError("generation must run on primary rows only")
    users = frame["users"]
    if any(user not in panel["users"] for user in users):
        raise AssertionError("frame user missing from runtime panel")
    for user in users:
        imposter = frame["imposter_of"][user]
        if not panel["users"][user]["profile_card"] or not panel["users"][imposter]["profile_card"]:
            raise AssertionError(f"empty card for {user} or {imposter}")

    screen = json.loads(METRIC_SCREEN_PATH.read_text())
    if screen.get("primary_metric") != METRIC_M1:
        raise AssertionError("metric screen did not choose the MiniLM metric")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    started = time.perf_counter()
    spec = CANDIDATES[candidate]
    tokenizer, model = load_local_model(candidate)
    ref_tokenizer = AutoTokenizer.from_pretrained(str(REFERENCE_TOKENIZER), local_files_only=True)
    gpu_name = torch.cuda.get_device_name(0)
    generation_started = time.perf_counter()
    sample_texts: dict[str, list[list[dict]]] = {}
    sampled_labels: dict[str, list[list[dict]]] = {}
    for arm in ARMS:
        print(f"[{candidate}] generating {arm}", flush=True)
        sample_texts[arm], sampled_labels[arm] = sample_arm(
            model, tokenizer, ref_tokenizer, candidate, panel, frame, arm
        )
    generation_seconds = time.perf_counter() - generation_started
    sample_path, sample_count = write_gzip_samples(candidate, rows, sample_texts)
    del model, tokenizer, ref_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[{candidate}] generation complete; scoring content metrics on CPU", flush=True)
    scoring_started = time.perf_counter()
    targets = [str(row["target"]) for row in rows]
    cards = {user: str(panel["users"][user]["profile_card"]) for user in users}
    all_generated_texts = [
        sample["text"]
        for row_index in range(len(rows))
        for arm in ARMS
        for sample in sample_texts[arm][row_index]
    ]
    mini_texts = unique_in_order(targets + all_generated_texts)
    mini_matrix, mini_meta = embed_many(mini_texts)
    mini_vectors = {text: mini_matrix[i] for i, text in enumerate(mini_texts)}

    # Fit M2 once on the 236 whole cards plus the 1,427 real turns.
    # Generated text is transformed with that fixed vocabulary and IDF.
    tfidf_transform, tfidf_meta = fit_tfidf_transform([cards[user] for user in users] + targets)
    tfidf_target_vectors = {i: tfidf_transform(targets[i]) for i in range(len(targets))}
    generated_tfidf = [tfidf_transform(text) for text in all_generated_texts]
    row_outputs: list[dict] = []
    generated_index = 0
    for row_index, row in enumerate(rows):
        metric_values = {
            METRIC_M1: {},
            METRIC_M2: {},
        }
        for arm in ARMS:
            mini_values = [
                1.0
                - cosine(
                    mini_vectors[targets[row_index]],
                    mini_vectors[sample["text"]],
                )
                for sample in sample_texts[arm][row_index]
            ]
            tfidf_values = [
                1.0
                - sparse_cosine(
                    tfidf_target_vectors[row_index], generated_tfidf[generated_index + k]
                )
                for k in range(K_SAMPLES)
            ]
            generated_index += K_SAMPLES
            metric_values[METRIC_M1][arm] = {
                "D": float(np.mean(mini_values)),
                "Dmin": float(np.min(mini_values)),
            }
            metric_values[METRIC_M2][arm] = {
                "D": float(np.mean(tfidf_values)),
                "Dmin": float(np.min(tfidf_values)),
            }
        row_outputs.append(
            {
                "row_index": row_index,
                "key": row_key(row["user_id"], row),
                "user_id": row["user_id"],
                "conv_id": row["conv_id"],
                "turn_index": row["turn_index"],
                "split": row["split"],
                "cluster_id": row["cluster_id"],
                "metrics": {
                    metric: {
                        "D_A3": values["A3"]["D"],
                        "D_A4": values["A4"]["D"],
                        "Dmin_A3": values["A3"]["Dmin"],
                        "Dmin_A4": values["A4"]["Dmin"],
                    }
                    for metric, values in metric_values.items()
                },
            }
        )
    scoring_seconds = time.perf_counter() - scoring_started

    form_histograms = {
        dimension: {
            arm: form_histogram(sampled_labels[arm], dimension, 5 if dimension == "length" else 2)
            for arm in ARMS
        }
        for dimension in DIMENSIONS
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    form_output = build_form_output(candidate, spec, frame, sampled_labels, form_histograms)
    form_path = RESULTS / f"form_{candidate}.json"
    form_tmp = form_path.with_name(f".{form_path.name}.tmp.{os.getpid()}")
    form_tmp.write_text(json.dumps(form_output, ensure_ascii=False, separators=(",", ":")))
    os.replace(form_tmp, form_path)

    result = {
        "schema": "wildchat-content-v1",
        "candidate": candidate,
        "model": spec["model"],
        "model_revision": spec["revision"],
        "interface": spec["interface"],
        "panel_revision": frame["panel_revision"],
        "frame_schema": frame["schema"],
        "frame_full_window_digest": frame["full_window_row_digest"],
        "frame_primary_digest": frame["primary_row_digest"],
        "n_users": frame["n_users"],
        "n_primary_rows": frame["n_primary_rows"],
        "users": users,
        "arms": list(ARMS),
        "seed": SEED,
        "k_samples": K_SAMPLES,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "primary_metric": METRIC_M1,
        "secondary_metric": METRIC_M2,
        "metric_metadata": {
            METRIC_M1: mini_meta,
            METRIC_M2: {
                **tfidf_meta,
                "fit_documents": "236 whole cards + 1,427 real primary turns",
                "generated_texts_do_not_change_fit": True,
            },
        },
        "sample_file": str(sample_path.relative_to(paths.WILDCHAT)),
        "sample_count": sample_count,
        "rows": row_outputs,
        "runtime": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "gpu": gpu_name,
            "gpu_count_visible": torch.cuda.device_count(),
            "generation_batch": int(os.environ.get("GENERATION_BATCH", "8")),
            "generation_seconds": generation_seconds,
            "scoring_seconds": scoring_seconds,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    output_path = RESULTS / f"content_{candidate}.json"
    output_tmp = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    output_tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    os.replace(output_tmp, output_path)
    print(
        json.dumps(
            {
                "candidate": candidate,
                "users": len(users),
                "rows": len(rows),
                "samples": sample_count,
                "result": str(output_path),
                "form_result": str(form_path),
                "sample_file": str(sample_path),
                "wall_seconds": result["runtime"]["wall_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
