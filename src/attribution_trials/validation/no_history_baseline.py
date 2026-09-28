"""PersonaMem no-history solvability baseline for gpt-4.1-mini.

Each PersonaMem-32k item is asked with no conversation history (A0), one item
per persona and question type (the shortest context). The persona-cluster
accuracy says how much of the benchmark a model answers without any user
information.

  python -m attribution_trials.validation.no_history_baseline
Output: <VALIDATION>/no_history_baseline/{rows.jsonl,result.json}
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download

from attribution_trials import paths

OUT = paths.VALIDATION / "no_history_baseline"
REPO_ID = "bowen-upenn/PersonaMem-v1"
REVISION = "73dfd752d477d0c466cd441f1669397f5726d7ab"
MODEL = "gpt-4.1-mini"
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
INSTRUCTION = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token <final_answer>."
)


def questions_path() -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename="questions_32k.csv",
        )
    )


def select_rows(df: pd.DataFrame) -> pd.DataFrame:
    selected = (
        df.sort_values(["context_length_in_tokens", "question_id"])
        .groupby(["persona_id", "question_type"], as_index=False)
        .head(1)
        .copy()
        .reset_index(drop=True)
    )
    selected["row_id"] = selected["question_id"].astype(str)
    return selected


def prompt_messages(row: dict) -> list[dict]:
    content = row["user_question_or_message"] + "\n\n" + INSTRUCTION + "\n\n" + row["all_options"]
    return [{"role": "user", "content": content}]


def parse_answer(text: str) -> str | None:
    lowered = text.lower()
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


def api_call(key: str, messages: list[dict], tries: int = 6) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    last = "unknown"
    for attempt in range(tries):
        try:
            response = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=180,
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


def analyze(rows: list[dict]) -> dict:
    per_user: dict[int, list[float]] = {}
    valid = [r for r in rows if r.get("predicted") is not None]
    for r in valid:
        per_user.setdefault(int(r["persona_id"]), []).append(
            float(r["predicted"] == r["correct_answer"])
        )
    user_values = [float(np.mean(v)) for _, v in sorted(per_user.items())]
    return {
        "model": MODEL,
        "n_rows": len(rows),
        "valid_rate": len(valid) / len(rows),
        "accuracy": bootstrap(user_values, 100),
        "chance": 0.25,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required at runtime")
    paths.ensure(OUT)
    df = select_rows(pd.read_csv(questions_path()))
    output_path = OUT / "rows.jsonl"
    completed = {}
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                item = json.loads(line)
                item["predicted"] = parse_answer(item.get("response", ""))
                if item.get("error") or item["predicted"] is None:
                    continue
                completed[item["row_id"]] = item

    def work(row: dict) -> dict:
        result = api_call(key, prompt_messages(row))
        text = result.get("text", "")
        return {
            "row_id": str(row["row_id"]),
            "persona_id": int(row["persona_id"]),
            "question_type": row["question_type"],
            "topic": row["topic"],
            "arm": "A0",
            "correct_answer": str(row["correct_answer"]).lower(),
            "predicted": parse_answer(text),
            "response": text,
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage", {}),
            "provider_model": result.get("provider_model"),
            "error": result.get("error"),
        }

    jobs = [row for row in df.to_dict(orient="records") if str(row["row_id"]) not in completed]
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, row) for row in jobs]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            item = future.result()
            with lock, output_path.open("a") as f:
                f.write(json.dumps(item) + "\n")
            completed[item["row_id"]] = item
            if i % 20 == 0 or i == len(futures):
                print(f"completed {i}/{len(futures)} new calls", flush=True)
    rows = sorted(completed.values(), key=lambda item: item["row_id"])
    output_path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    result = analyze(rows)
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["accuracy"]))


if __name__ == "__main__":
    main()
