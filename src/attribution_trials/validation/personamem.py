"""PersonaMem known-signal control: no-history leakage screen, then target vs imposter.

Arms (native 4-option MCQ, chance 0.25):
  A0  the question alone, no conversation history (leakage screen)
  A3  imposter: another persona's shared context, matched on question type,
      topic and context length
  A4  target: the persona's own shared context up to the question

Stages (resumable):
  --stage screen   one A0 call per item over the 32k split; items answered
                   correctly without history are dropped
  --stage powered  A3 + A4 on the surviving items (capped at 220, round-robin
                   over personas by context length)

  python -m attribution_trials.validation.personamem --stage screen
  python -m attribution_trials.validation.personamem --stage powered
Output: <VALIDATION>/personamem/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from scipy.optimize import linear_sum_assignment

from attribution_trials import paths

OUT = paths.VALIDATION / "personamem"
REPO_ID = "bowen-upenn/PersonaMem-v1"
REVISION = "73dfd752d477d0c466cd441f1669397f5726d7ab"
MODEL = "gpt-5.6-luna"
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
POWERED_ITEM_CAP = 220
GO_MIN_SURVIVORS = 80
GO_MIN_PERSONAS = 15
GO_MIN_ITEMS_PER_PERSONA = 3
INSTRUCTION = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token <final_answer>."
)
PARAM_CANDIDATES = [
    {"temperature": 0.0, "max_tokens": 2048},
    {"temperature": 0.0, "max_completion_tokens": 2048},
    {"max_completion_tokens": 2048},
    {"max_tokens": 2048},
]


def data_paths() -> tuple[Path, Path]:
    kwargs = {
        "repo_id": REPO_ID,
        "repo_type": "dataset",
        "revision": REVISION,
    }
    return (
        Path(hf_hub_download(filename="questions_32k.csv", **kwargs)),
        Path(hf_hub_download(filename="shared_contexts_32k.jsonl", **kwargs)),
    )


def load_contexts(path: Path) -> dict[str, list[dict]]:
    out = {}
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            if len(item) != 1:
                raise ValueError("each context line must contain one ID")
            out.update(item)
    return out


def load_pool(questions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(questions_path).copy()
    df["row_id"] = df["question_id"].astype(str)
    if df.row_id.nunique() != len(df):
        raise ValueError("question_id is not unique in the 32k split")
    return df.reset_index(drop=True)


def prompt_messages(row: dict, context: list[dict] | None) -> list[dict]:
    content = row["user_question_or_message"] + "\n\n" + INSTRUCTION + "\n\n" + row["all_options"]
    return (context or []) + [{"role": "user", "content": content}]


def parse_answer(text: str) -> str | None:
    lowered = (text or "").lower()
    if "<final_answer>" in lowered:
        tail = lowered.rsplit("<final_answer>", 1)[-1]
        match = re.match(r"\s*\(?([a-d])\)?", tail)
        if match:
            hits = [match.group(1)]
        else:
            prefix = lowered.split("<final_answer>", 1)[0]
            match = re.match(r"\s*\(([a-d])\)(?:\s|$)", prefix)
            hits = [match.group(1)] if match else []
    else:
        hits = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([a-d])\)?",
            lowered,
        )
        if not hits:
            hits = re.findall(r"\(([a-d])\)", lowered)
    unique = sorted(set(hits))
    return f"({unique[0]})" if len(unique) == 1 else None


def resolve_params(key: str) -> dict:
    """Find a decoding-parameter set the provider accepts; persist to params.json."""
    params_path = OUT / "params.json"
    if params_path.exists():
        return json.loads(params_path.read_text())
    errors = []
    for cand in PARAM_CANDIDATES:
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word ok."}],
            **cand,
        }
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=120,
            )
        except requests.RequestException as exc:
            errors.append(f"{cand}: {type(exc).__name__}:{str(exc)[:120]}")
            continue
        if response.status_code == 200:
            data = response.json()
            resolved = {
                "params": cand,
                "provider_model": data.get("model"),
                "probe_finish_reason": data["choices"][0].get("finish_reason"),
            }
            params_path.write_text(json.dumps(resolved, indent=2) + "\n")
            return resolved
        errors.append(f"{cand}: http-{response.status_code}:{response.text[:200]}")
    raise RuntimeError("no accepted parameter set:\n" + "\n".join(errors))


def api_call(key: str, params: dict, messages: list[dict], tries: int = 6) -> dict:
    body = {"model": MODEL, "messages": messages, **params}
    last = "unknown"
    for attempt in range(tries):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=300,
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "finish_reason": data["choices"][0].get("finish_reason"),
                    "usage": data.get("usage", {}),
                    "provider_model": data.get("model"),
                }
            last = f"http-{response.status_code}:{response.text[:160]}"
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504, 529}:
                break
        except requests.RequestException as exc:
            last = f"{type(exc).__name__}:{str(exc)[:120]}"
        time.sleep(min(30, 2**attempt))
    return {"error": last}


def bootstrap(values: list[float], seed: int) -> dict:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(10_000, len(x)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": float(x.mean()), "ci95": [float(lo), float(hi)], "n_users": len(x)}


def build_context(contexts: dict, row: dict) -> list[dict]:
    return contexts[str(row["shared_context_id"])][: int(row["end_index_in_shared_context"])]


def run_jobs(jobs, key, params, contexts, row_lookup, output_path, workers) -> dict:
    completed = load_completed(output_path)
    lock = threading.Lock()

    def work(job: tuple[dict, str]) -> dict:
        row, arm = job
        if arm == "A0":
            context = None
        elif arm == "A4":
            context = build_context(contexts, row)
        else:
            context = build_context(contexts, row_lookup[str(row["donor_row_id"])])
        result = api_call(key, params, prompt_messages(row, context))
        text = result.get("text", "")
        return {
            "row_id": str(row["row_id"]),
            "persona_id": int(row["persona_id"]),
            "question_type": row["question_type"],
            "topic": row["topic"],
            "context_length_in_tokens": int(row["context_length_in_tokens"]),
            "arm": arm,
            "donor_persona_id": int(row["donor_persona_id"]) if arm == "A3" else None,
            "correct_answer": str(row["correct_answer"]).lower(),
            "predicted": parse_answer(text),
            "response": text,
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage", {}),
            "provider_model": result.get("provider_model"),
            "error": result.get("error"),
        }

    pending = [j for j in jobs if (str(j[0]["row_id"]), j[1]) not in completed]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, job) for job in pending]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            item = future.result()
            with lock, output_path.open("a") as f:
                f.write(json.dumps(item) + "\n")
            completed[(item["row_id"], item["arm"])] = item
            if i % 20 == 0 or i == len(futures):
                print(f"completed {i}/{len(futures)} new calls", flush=True)
    rows = sorted(completed.values(), key=lambda r: (r["row_id"], r["arm"]))
    output_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return completed


def load_completed(output_path: Path) -> dict:
    completed = {}
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                item = json.loads(line)
                item["predicted"] = parse_answer(item.get("response", ""))
                if item.get("error"):
                    continue
                completed[(item["row_id"], item["arm"])] = item
    return completed


def base_manifest(df: pd.DataFrame, resolved: dict) -> dict:
    return {
        "dataset": REPO_ID,
        "revision": REVISION,
        "model": MODEL,
        "decoding_params": resolved["params"],
        "provider_model_probe": resolved.get("provider_model"),
        "n_rows": len(df),
        "n_personas": int(df.persona_id.nunique()),
    }


def stage_screen(key: str, df: pd.DataFrame, contexts: dict, workers: int) -> None:
    resolved = resolve_params(key)
    manifest = base_manifest(df, resolved)
    (OUT / "screen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    out = OUT / "screen_rows.jsonl"
    jobs = [(row, "A0") for row in df.to_dict(orient="records")]
    completed = run_jobs(jobs, key, resolved["params"], contexts, {}, out, workers)
    items = [completed[k] for k in completed if k[1] == "A0"]
    if len(items) != len(df):
        raise RuntimeError(f"screen incomplete: {len(items)}/{len(df)} calls succeeded")
    parseable = [r for r in items if r["predicted"] is not None]
    survivors = [r for r in parseable if r["predicted"] != r["correct_answer"]]
    leaky = [r for r in parseable if r["predicted"] == r["correct_answer"]]
    excluded = [r for r in items if r["predicted"] is None]
    per_persona: dict[int, list[float]] = {}
    for r in parseable:
        per_persona.setdefault(int(r["persona_id"]), []).append(
            float(r["predicted"] == r["correct_answer"])
        )
    cluster_acc = bootstrap([float(np.mean(v)) for _, v in sorted(per_persona.items())], 101)
    survivor_counts: dict[int, int] = {}
    for r in survivors:
        survivor_counts[int(r["persona_id"])] = survivor_counts.get(int(r["persona_id"]), 0) + 1
    personas_ok = sum(1 for c in survivor_counts.values() if c >= GO_MIN_ITEMS_PER_PERSONA)
    gates = {
        "parse_ok": len(parseable) / len(items) >= 0.95,
        "go": (len(survivors) >= GO_MIN_SURVIVORS and personas_ok >= GO_MIN_PERSONAS),
    }
    result = {
        "n_items": len(items),
        "n_parseable": len(parseable),
        "n_leaky_dropped": len(leaky),
        "n_excluded_unparseable": len(excluded),
        "n_survivors": len(survivors),
        "personas_with_min_items": personas_ok,
        "a0_item_accuracy": len(leaky) / max(1, len(parseable)),
        "a0_persona_cluster_accuracy": cluster_acc,
        "survivor_counts_by_persona": {str(k): v for k, v in sorted(survivor_counts.items())},
        "gates": gates,
    }
    (OUT / "screen_result.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "survivors.json").write_text(
        json.dumps(
            {
                "go": gates["go"],
                "row_ids": sorted(r["row_id"] for r in survivors),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(gates, sort_keys=True))


def select_powered(survivor_df: pd.DataFrame, cap: int) -> pd.DataFrame:
    queues = {
        pid: list(group.sort_values("context_length_in_tokens").itertuples(index=False))
        for pid, group in survivor_df.groupby("persona_id")
    }
    order = sorted(queues)
    picked = []
    while len(picked) < cap and any(queues.values()):
        for pid in order:
            if queues[pid]:
                picked.append(queues[pid].pop(0)._asdict())
                if len(picked) >= cap:
                    break
    return pd.DataFrame(picked)


def add_derangement(selected: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    donor = {}
    for qtype, group in selected.groupby("question_type", sort=True):
        idx = list(group.index)
        while True:
            if len(idx) < 2 or selected.loc[idx].persona_id.nunique() < 2:
                dropped.extend(selected.loc[idx, "row_id"])
                idx = []
                break
            n = len(idx)
            cost = np.empty((n, n), dtype=float)
            for i, ti in enumerate(idx):
                t = selected.loc[ti]
                for j, dj in enumerate(idx):
                    d = selected.loc[dj]
                    if t.persona_id == d.persona_id:
                        cost[i, j] = 1e9
                    else:
                        topic_penalty = 1000.0 * float(t.topic != d.topic)
                        length_gap = abs(
                            math.log1p(float(t.context_length_in_tokens))
                            - math.log1p(float(d.context_length_in_tokens))
                        )
                        cost[i, j] = topic_penalty + length_gap
            rows, cols = linear_sum_assignment(cost)
            if not np.any(cost[rows, cols] >= 1e9):
                for i, j in zip(rows, cols):
                    donor[idx[i]] = idx[j]
                break
            counts = selected.loc[idx].persona_id.value_counts()
            victim_persona = counts.idxmax()
            victim = (
                selected.loc[idx][selected.loc[idx].persona_id == victim_persona]
                .sort_values("context_length_in_tokens")
                .index[-1]
            )
            dropped.append(selected.loc[victim, "row_id"])
            idx.remove(victim)
    kept = selected.loc[list(donor)].copy()
    kept["donor_row_id"] = [selected.loc[donor[i], "row_id"] for i in kept.index]
    kept["donor_persona_id"] = [int(selected.loc[donor[i], "persona_id"]) for i in kept.index]
    assert (kept.persona_id.astype(int) != kept.donor_persona_id).all()
    return kept.reset_index(drop=True), dropped


def stage_powered(key: str, df: pd.DataFrame, contexts: dict, workers: int) -> None:
    resolved = resolve_params(key)
    survivors_path = OUT / "survivors.json"
    if not survivors_path.exists():
        raise RuntimeError("survivors.json missing: run --stage screen first")
    survivors = json.loads(survivors_path.read_text())
    if not survivors["go"]:
        raise RuntimeError("the screen left too few items for the powered stage")
    survivor_df = df[df.row_id.isin(set(survivors["row_ids"]))].copy()
    selected = select_powered(survivor_df, POWERED_ITEM_CAP)
    selected, dropped = add_derangement(selected)
    manifest = base_manifest(selected, resolved)
    manifest["n_survivors_total"] = len(survivor_df)
    manifest["dropped_no_donor"] = dropped
    manifest["rows"] = selected[
        [
            "row_id",
            "persona_id",
            "question_type",
            "topic",
            "context_length_in_tokens",
            "shared_context_id",
            "end_index_in_shared_context",
            "donor_row_id",
            "donor_persona_id",
        ]
    ].to_dict(orient="records")
    (OUT / "powered_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    row_lookup = {str(r.row_id): r._asdict() for r in selected.itertuples(index=False)}
    out = OUT / "powered_rows.jsonl"
    jobs = []
    for row in selected.to_dict(orient="records"):
        jobs.append((row, "A3"))
        jobs.append((row, "A4"))
    completed = run_jobs(jobs, key, resolved["params"], contexts, row_lookup, out, workers)
    rows = list(completed.values())
    by_arm = {}
    per_user_arm: dict[str, dict[int, list[float]]] = {"A3": {}, "A4": {}}
    for arm in ("A3", "A4"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        valid = [r for r in arm_rows if r["predicted"] is not None]
        for r in valid:
            per_user_arm[arm].setdefault(int(r["persona_id"]), []).append(
                float(r["predicted"] == r["correct_answer"])
            )
        user_values = [float(np.mean(v)) for _, v in sorted(per_user_arm[arm].items())]
        by_arm[arm] = {
            "n_rows": len(arm_rows),
            "valid_rate": len(valid) / max(1, len(arm_rows)),
            "accuracy": bootstrap(user_values, 100 + len(by_arm)),
        }
    common = sorted(set(per_user_arm["A3"]) & set(per_user_arm["A4"]))
    diffs = [float(np.mean(per_user_arm["A4"][u]) - np.mean(per_user_arm["A3"][u])) for u in common]
    effect = bootstrap(diffs, 909)
    gates = {
        "valid_parsing": all(by_arm[a]["valid_rate"] >= 0.95 for a in ("A3", "A4")),
        "a4_minus_a3_positive": effect["ci95"][0] > 0.0,
    }
    result = {
        "arms": by_arm,
        "a4_minus_a3": effect,
        "gates": gates,
        "n_items": len(selected),
        "dropped_no_donor": dropped,
    }
    (OUT / "powered_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(gates, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["screen", "powered"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required at runtime")
    paths.ensure(OUT)
    questions_path, contexts_path = data_paths()
    contexts = load_contexts(contexts_path)
    df = load_pool(questions_path)
    if args.stage == "screen":
        stage_screen(key, df, contexts, args.workers)
    else:
        stage_powered(key, df, contexts, args.workers)


if __name__ == "__main__":
    main()
