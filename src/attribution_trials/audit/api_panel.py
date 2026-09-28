"""OPeRA API panel: frontier API models as frozen-prompt simulators.

Metric = per-decision NLL over the label set, measured through the API by
letter-mapped multiple choice + token logprobs (top_logprobs): each label
-> a letter; the model answers with a letter; we read the first-token
logprob of every option letter, renormalize over the options,
NLL = -log p(true letter). Same attribution arms and per-user pairing as
frozen_panel.

Any OpenAI-compatible endpoint: key and base URL from the environment
(OPENAI_API_KEY, optional OPENAI_BASE_URL); never hardcoded.

  python -m attribution_trials.audit.api_panel \\
      <action|timing> <static|dynamic|combined> <model> [N]
Output: <AUDIT>/opera/frozen_opera-<channel>-<config>_<modeltag>.json
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from attribution_trials import paths
from attribution_trials.data.opera import build_opera

KEY = os.environ["OPENAI_API_KEY"]
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LETTERS = "ABCDEFGHIJKLMNOPQRST"
N_EVAL_DEFAULT = 400
MAX_WORKERS = int(os.environ.get("API_WORKERS", "12"))
FLOOR = -15.0  # logprob floor for option letters absent from top_logprobs


def _labelset(evals):
    return sorted({r["label"] for r in evals})


def _prompt(base, card, labels):
    opts = "\n".join(f"{LETTERS[i]}) {lab}" for i, lab in enumerate(labels))
    persona = f"\nAbout this person: {card}" if card else ""
    return (
        "You are simulating one specific person's behavior on a prediction "
        "task. Choose the single most likely option.\n\n"
        f"{base}{persona}\n\nOptions:\n{opts}\n\n"
        "Answer with ONLY the capital letter of the most likely option."
    )


def _call(model, prompt, tries=6):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    last = "?"
    for a in range(tries):
        try:
            r = requests.post(
                f"{BASE}/chat/completions",
                headers={"Authorization": f"Bearer {KEY}"},
                json=body,
                timeout=60,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 529):
                last = f"http {r.status_code}"
                time.sleep(min(20, 3 * (a + 1)))
                continue
            return {"error": f"{r.status_code}:{r.text[:120]}"}
        except requests.RequestException as e:
            last = str(e)[:80]
            time.sleep(min(20, 3 * (a + 1)))
    return {"error": f"exhausted:{last}"}


def _parse(resp, true_idx, k):
    """Returns (nll_or_None, pred_or_None, err).

    Reads the first-token top_logprobs; option letters absent from them get
    FLOOR. pred = the option with the highest logprob."""
    try:
        top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        # sorted descending; keep FIRST (highest) per normalized token so a
        # low-prob ' B' can't clobber the real 'B'.
        lp = {}
        for t in top:
            key = t["token"].strip().upper()
            if key not in lp:
                lp[key] = t["logprob"]
        opt = [lp.get(LETTERS[i], FLOOR) for i in range(k)]
        m = max(opt)
        z = m + math.log(sum(math.exp(x - m) for x in opt))
        pred = max(range(k), key=lambda i: opt[i])
        return -(opt[true_idx] - z), pred, None
    except (KeyError, IndexError, TypeError):
        return None, None, resp.get("error", "no_output")


ARMS = ("A0", "A1", "A2", "A3", "A4", "A4frozen")


def main():
    channel, config, model = sys.argv[1], sys.argv[2], sys.argv[3]
    n = N_EVAL_DEFAULT
    for a in sys.argv[4:]:
        if a.isdigit():
            n = int(a)

    def imposter_map(cells, seed):  # seeded within-cell derangement (cycle)
        rng2 = random.Random(1000 + seed)
        members = {}
        for p, c in cells.items():
            members.setdefault(c, []).append(p)
        out = {}
        for c, ids in members.items():
            ids = sorted(ids)
            rng2.shuffle(ids)
            for a, b in zip(ids, ids[1:] + ids[:1]):
                out[a] = b
        return out

    train, evals, cells, frozen_card_of = build_opera(channel, config)
    rng = random.Random(0)
    rng.shuffle(evals)
    evals = evals[:n]
    imp = imposter_map(cells, 0)
    labels = _labelset(evals)
    k = len(labels)
    lab_idx = {lab: i for i, lab in enumerate(labels)}

    arm_card = {
        "A0": lambda r: None,
        "A1": lambda r: r["placebo_card"],
        "A2": lambda r: r["group_card"],
        "A3": lambda r: frozen_card_of[imp[r["player"]]],
        "A4": lambda r: r["self_card"],
        "A4frozen": lambda r: r["frozen_card"],
    }

    t0 = time.time()
    tasks = [(arm, i, r) for arm in ARMS for i, r in enumerate(evals)]
    nll = {a: [None] * len(evals) for a in ARMS}
    acc = {a: [None] * len(evals) for a in ARMS}
    errs = 0
    ptok = [0]

    def work(job):
        arm, i, r = job
        prompt = _prompt(r["base"], arm_card[arm](r), labels)
        resp = _call(model, prompt)
        val, pred, err = _parse(resp, lab_idx[r["label"]], k)
        try:
            ptok[0] += resp["usage"]["prompt_tokens"]
        except (KeyError, TypeError):
            pass
        return arm, i, val, pred, err

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for arm, i, val, pred, err in (
            f.result() for f in as_completed([ex.submit(work, j) for j in tasks])
        ):
            if pred is None:  # genuine failure
                errs += 1
                acc[arm][i] = 0.0
                nll[arm][i] = None
            else:
                acc[arm][i] = 1.0 if pred == lab_idx[evals[i]["label"]] else 0.0
                nll[arm][i] = val
    print(
        f"  [{model} {channel}/{config}] prompt_tokens={ptok[0]:,} (~{ptok[0] / 1e6:.2f}M)",
        flush=True,
    )

    payload = {
        "family": f"api:{model}",
        "substrate": f"opera-{channel}-{config}",
        "size": model,
        "model": model,
        "channel": channel,
        "config": config,
        "n_eval": len(evals),
        "n_labels": k,
        "errors": errs,
        "players": [r["player"] for r in evals],
        "labels": [r["label"] for r in evals],
        "nll": nll,
        "acc": acc,
        "mean_nll": {
            a: (sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v)))
            if any(x is not None for x in v)
            else None
            for a, v in nll.items()
        },
        "mean_acc": {a: sum(v) / len(v) for a, v in acc.items()},
        "wall_seconds": time.time() - t0,
    }
    tag = model.replace("/", "-").replace(".", "-")
    name = f"frozen_opera-{channel}-{config}_{tag}.json"
    out = paths.ensure(paths.AUDIT / "opera") / name
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out)
    print(
        f"API {model} opera-{channel}-{config}: "
        f"acc={ {a: round(v, 3) for a, v in payload['mean_acc'].items()} } "
        f"errs={errs}/{len(tasks)} ({payload['wall_seconds']:.0f}s) -> {name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
