"""HorizonBench known-signal control: no-history leakage screen, then target vs imposter.

Native contract: 5-option MCQ (A-E), pick the assistant response matching the
user's current preferences; chance = 0.20.

Arms:
  A0  final episode only (scenario + last user turns), no prior history
  A3  imposter history: another user's history (same generator, length-matched)
      spliced before the target's final episode
  A4  full original conversation

Stages (resumable):
  --stage screen   the shortest item of each of the 140 users with the shortest
                   histories; items answered correctly under A0 are dropped
  --stage powered  A3 + A4 on the 110 shortest surviving items

  python -m attribution_trials.validation.horizonbench --stage screen
  python -m attribution_trials.validation.horizonbench --stage powered
Output: <VALIDATION>/horizonbench/
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

OUT = paths.VALIDATION / "horizonbench"
REPO_ID = "stellalisy/HorizonBench"
REVISION = "50941f00f90c03a5a60219d76393869b757b835a"
MODEL = "gpt-5.6-luna"
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
N_SHARDS = 6
POWERED_ITEM_CAP = 110
GO_MIN_SURVIVORS = 80
MAX_CONV_CHARS = 300000
SCREEN_POOL_USERS = 140
CHANCE = 0.20
INSTRUCTION = (
    "Select the response that best aligns with the user's current "
    "preferences. Give your final answer (A), (B), (C), (D), or (E) after "
    "the special token <final_answer>."
)
PARAM_CANDIDATES = [
    {"temperature": 0.0, "max_tokens": 2048},
    {"temperature": 0.0, "max_completion_tokens": 2048},
    {"max_completion_tokens": 2048},
    {"max_tokens": 2048},
]


def load_benchmark() -> pd.DataFrame:
    frames = []
    for i in range(N_SHARDS):
        p = hf_hub_download(
            REPO_ID,
            f"benchmark/test-0000{i}-of-0000{N_SHARDS}.parquet",
            repo_type="dataset",
            revision=REVISION,
        )
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df["conv_chars"] = df["conversation"].str.len()
    df["row_id"] = df["id"].astype(str)
    if df.row_id.nunique() != len(df):
        raise ValueError("benchmark id is not unique")
    return df


def split_final_episode(conversation: str) -> tuple[str, str]:
    """Return (history, final_episode); final episode starts at the last Date: line."""
    matches = list(re.finditer(r"\nDate: ", conversation))
    if not matches:
        return "", conversation
    cut = matches[-1].start()
    return conversation[:cut], conversation[cut:]


def select_slice(df: pd.DataFrame) -> pd.DataFrame:
    pool = df[df.conv_chars <= MAX_CONV_CHARS].copy()
    histories = pool["conversation"].map(lambda c: split_final_episode(c)[0])
    pool = pool[histories.str.len() > 0]
    shortest = pool.sort_values(["conv_chars", "row_id"]).groupby("user_id").head(1)
    return (
        shortest.nsmallest(SCREEN_POOL_USERS, "conv_chars")
        .sort_values("row_id")
        .reset_index(drop=True)
    )


def format_options(options_json: str) -> str:
    options = json.loads(options_json)
    return "\n\n".join(f"({o['letter']}) {o['option']}" for o in options)


def prompt_messages(row: dict, history: str | None) -> list[dict]:
    _, final_episode = split_final_episode(row["conversation"])
    context = (history + final_episode) if history else final_episode.lstrip("\n")
    content = context + "\n\n" + INSTRUCTION + "\n\n" + format_options(row["options"])
    return [{"role": "user", "content": content}]


def parse_answer(text: str) -> str | None:
    lowered = (text or "").lower()
    if "<final_answer>" in lowered:
        tail = lowered.rsplit("<final_answer>", 1)[-1]
        match = re.match(r"\s*\(?([a-e])\)?", tail)
        if match:
            hits = [match.group(1)]
        else:
            prefix = lowered.split("<final_answer>", 1)[0]
            match = re.match(r"\s*\(([a-e])\)(?:\s|$)", prefix)
            hits = [match.group(1)] if match else []
    else:
        hits = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([a-e])\)?",
            lowered,
        )
        if not hits:
            hits = re.findall(r"\(([a-e])\)", lowered)
    unique = sorted(set(hits))
    return unique[0] if len(unique) == 1 else None


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
                timeout=600,
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


def run_jobs(jobs, key, params, donor_history, output_path, workers) -> dict:
    completed = load_completed(output_path)
    lock = threading.Lock()

    def work(job: tuple[dict, str]) -> dict:
        row, arm = job
        if arm == "A0":
            history = None
        elif arm == "A4":
            history = split_final_episode(row["conversation"])[0]
        else:
            history = donor_history[str(row["row_id"])]
        result = api_call(key, params, prompt_messages(row, history))
        text = result.get("text", "")
        return {
            "row_id": str(row["row_id"]),
            "user_id": row["user_id"],
            "generator": row["generator"],
            "has_evolved": bool(row["has_evolved"]),
            "conv_chars": int(row["conv_chars"]),
            "arm": arm,
            "donor_user_id": row.get("donor_user_id") if arm == "A3" else None,
            "correct_answer": str(row["correct_letter"]).lower(),
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
            if i % 10 == 0 or i == len(futures):
                print(f"completed {i}/{len(futures)} new calls", flush=True)
    rows = sorted(completed.values(), key=lambda r: (r["row_id"], r["arm"]))
    output_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return completed


def stage_screen(key: str, sel: pd.DataFrame, workers: int) -> None:
    resolved = resolve_params(key)
    manifest = {
        "dataset": REPO_ID,
        "revision": REVISION,
        "model": MODEL,
        "decoding_params": resolved["params"],
        "screen_pool_users": len(sel),
        "max_conv_chars": MAX_CONV_CHARS,
        "rows": sel[["row_id", "user_id", "generator", "conv_chars", "has_evolved"]].to_dict(
            orient="records"
        ),
    }
    (OUT / "screen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [(row, "A0") for row in sel.to_dict(orient="records")]
    completed = run_jobs(jobs, key, resolved["params"], {}, OUT / "screen_rows.jsonl", workers)
    items = [completed[k] for k in completed if k[1] == "A0"]
    if len(items) != len(sel):
        raise RuntimeError(f"screen incomplete: {len(items)}/{len(sel)}")
    parseable = [r for r in items if r["predicted"] is not None]
    survivors = [r for r in parseable if r["predicted"] != r["correct_answer"]]
    leaky = [r for r in parseable if r["predicted"] == r["correct_answer"]]
    gates = {
        "parse_ok": len(parseable) / len(items) >= 0.95,
        "go": len(survivors) >= GO_MIN_SURVIVORS,
    }
    result = {
        "n_items": len(items),
        "n_parseable": len(parseable),
        "n_leaky_dropped": len(leaky),
        "n_excluded_unparseable": len(items) - len(parseable),
        "n_survivors": len(survivors),
        "a0_item_accuracy": len(leaky) / max(1, len(parseable)),
        "chance": CHANCE,
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


def add_derangement(selected: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped: list[str] = []
    donor = {}
    for gen, group in selected.groupby("generator", sort=True):
        idx = list(group.index)
        while True:
            if len(idx) < 2:
                dropped.extend(selected.loc[idx, "row_id"])
                idx = []
                break
            n = len(idx)
            cost = np.empty((n, n), dtype=float)
            for i, ti in enumerate(idx):
                t = selected.loc[ti]
                for j, dj in enumerate(idx):
                    d = selected.loc[dj]
                    if t.user_id == d.user_id:
                        cost[i, j] = 1e9
                    else:
                        cost[i, j] = abs(
                            math.log1p(float(t.conv_chars)) - math.log1p(float(d.conv_chars))
                        )
            rows, cols = linear_sum_assignment(cost)
            if not np.any(cost[rows, cols] >= 1e9):
                for i, j in zip(rows, cols):
                    donor[idx[i]] = idx[j]
                break
            counts = selected.loc[idx].user_id.value_counts()
            victim_user = counts.idxmax()
            victim = (
                selected.loc[idx][selected.loc[idx].user_id == victim_user]
                .sort_values("conv_chars")
                .index[-1]
            )
            dropped.append(selected.loc[victim, "row_id"])
            idx.remove(victim)
    kept = selected.loc[list(donor)].copy()
    kept["donor_row_id"] = [selected.loc[donor[i], "row_id"] for i in kept.index]
    kept["donor_user_id"] = [selected.loc[donor[i], "user_id"] for i in kept.index]
    assert (kept.user_id != kept.donor_user_id).all()
    return kept.reset_index(drop=True), dropped


def stage_powered(key: str, sel: pd.DataFrame, workers: int) -> None:
    resolved = resolve_params(key)
    survivors_path = OUT / "survivors.json"
    if not survivors_path.exists():
        raise RuntimeError("survivors.json missing: run --stage screen first")
    survivors = json.loads(survivors_path.read_text())
    if not survivors["go"]:
        raise RuntimeError("the screen left too few items for the powered stage")
    pool = sel[sel.row_id.isin(set(survivors["row_ids"]))].copy()
    selected = pool.nsmallest(POWERED_ITEM_CAP, "conv_chars").reset_index(drop=True)
    selected, dropped = add_derangement(selected)
    donor_history = {}
    conv_by_row = {str(r.row_id): r.conversation for r in selected.itertuples(index=False)}
    for r in selected.itertuples(index=False):
        donor_history[str(r.row_id)] = split_final_episode(conv_by_row[str(r.donor_row_id)])[0]
    manifest = {
        "dataset": REPO_ID,
        "revision": REVISION,
        "model": MODEL,
        "decoding_params": resolved["params"],
        "n_survivors_total": len(pool),
        "n_selected": len(selected),
        "dropped_no_donor": dropped,
        "rows": selected[
            [
                "row_id",
                "user_id",
                "generator",
                "conv_chars",
                "has_evolved",
                "donor_row_id",
                "donor_user_id",
            ]
        ].to_dict(orient="records"),
    }
    (OUT / "powered_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for row in selected.to_dict(orient="records"):
        jobs.append((row, "A3"))
        jobs.append((row, "A4"))
    completed = run_jobs(
        jobs, key, resolved["params"], donor_history, OUT / "powered_rows.jsonl", workers
    )
    rows = list(completed.values())
    by_arm = {}
    per_user_arm: dict[str, dict[str, list[float]]] = {"A3": {}, "A4": {}}
    for arm in ("A3", "A4"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        valid = [r for r in arm_rows if r["predicted"] is not None]
        for r in valid:
            per_user_arm[arm].setdefault(r["user_id"], []).append(
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
    evolved = {}
    for flag in (True, False):
        vals = [
            r
            for r in rows
            if r["arm"] == "A4" and r["predicted"] is not None and r["has_evolved"] == flag
        ]
        evolved[str(flag)] = {
            "n": len(vals),
            "accuracy": float(np.mean([v["predicted"] == v["correct_answer"] for v in vals]))
            if vals
            else None,
        }
    gates = {
        "valid_parsing": all(by_arm[a]["valid_rate"] >= 0.95 for a in ("A3", "A4")),
        "a4_minus_a3_positive": effect["ci95"][0] > 0.0,
    }
    result = {
        "arms": by_arm,
        "a4_minus_a3": effect,
        "gates": gates,
        "chance": CHANCE,
        "n_items": len(selected),
        "a4_by_evolution_status": evolved,
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
    df = load_benchmark()
    sel = select_slice(df)
    if args.stage == "screen":
        stage_screen(key, sel, args.workers)
    else:
        stage_powered(key, sel, args.workers)


if __name__ == "__main__":
    main()
