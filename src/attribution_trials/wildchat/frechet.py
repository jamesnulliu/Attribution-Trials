"""Population-distance selector: Fréchet distance between generated and real turns.

Re-embeds every A3/A4 sample of the seven simulators and the 1,427 real turns
with MiniLM, then
  * computes each simulator's content gain per user from the per-sample
    distances (avg@8 and best@8),
  * computes the Fréchet distance between the embedding clouds of the samples
    and the real turns, under the target's card (A4) and the imposter's (A3),
    and the share of the A4 value kept under A3,
  * scores "population distance, Frechet (A4)" as a selector with the R2 rule
    (agree / resolvable, Kendall tau, top pick).
Writes ``paths.WILDCHAT / "frechet.json"``.

  python -m attribution_trials.wildchat.frechet
"""

from __future__ import annotations

import gzip
import itertools
import json
import time
from collections import defaultdict

import numpy as np

from attribution_trials import paths
from attribution_trials.wildchat.metric_screen import embed_many, unique_in_order

RES = paths.WILDCHAT
SEED, NBOOT, METRIC = 0, 10_000, "M1_minilm_max_card_bullet"
CANDS = ("1.7b", "8b", "dense", "humanlike", "coser", "osim", "humanlm")
ARMS = ("A3", "A4")
K = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_samples(path, n_rows: int) -> dict[tuple[str, str], list[str]]:
    texts: dict[tuple[str, str], list[str]] = defaultdict(lambda: [None] * K)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            texts[(rec["key"], rec["arm"])][rec["k"]] = rec["text"]
    assert len(texts) == n_rows * 2 and all(None not in v for v in texts.values()), path
    return texts


def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.einsum("ij,ij->i", a, b)


def boot_ci(values: np.ndarray, nboot: int = NBOOT) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(nboot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def tau_a(x, y) -> float:
    c = d = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
        c += s > 0
        d += s < 0
    return float((c - d) / (len(x) * (len(x) - 1) / 2))


def r2(
    gain: dict[str, np.ndarray], score: dict[str, np.ndarray] | dict[str, float], cands=CANDS
) -> dict:
    """Pairs resolvable under the content-gain bootstrap; the selector supplies a sign."""
    smean = {c: float(np.mean(score[c])) for c in cands}
    gmean = {c: float(gain[c].mean()) for c in cands}
    pairs, agree = [], 0
    for l, r in itertools.combinations(cands, 2):
        point, lo, hi = boot_ci(gain[l] - gain[r])
        ok = lo > 0 or hi < 0
        pairs.append(
            {
                "left": l,
                "right": r,
                "gain_diff": point,
                "ci": [lo, hi],
                "resolvable": ok,
                "agree": bool(ok and np.sign(smean[l] - smean[r]) == np.sign(point)),
            }
        )
        agree += pairs[-1]["agree"]
    n_res = sum(p["resolvable"] for p in pairs)
    order = sorted(cands, key=lambda c: gmean[c])
    pick = min(cands, key=lambda c: smean[c])
    return {
        "n_resolvable": n_res,
        "n_agree": int(agree),
        "support_fraction": agree / n_res if n_res else None,
        "kendall_tau": tau_a([smean[c] for c in cands], [gmean[c] for c in cands]),
        "top_pick": pick,
        "top_pick_gain_rank": order.index(pick) + 1,
        "top_pick_content_gain": -gmean[pick],
        "best_content_gain": -gmean[order[0]],
        "selector_means": smean,
        "gain_means": gmean,
        "pairs": pairs,
    }


def frechet(x: np.ndarray, y: np.ndarray) -> float:
    mu, nu = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    ev = np.linalg.eigvals(cx @ cy)
    tr = float(np.sqrt(np.clip(ev.real, 0, None)).sum())
    return float(((mu - nu) ** 2).sum() + np.trace(cx) + np.trace(cy) - 2 * tr)


def main() -> None:
    frame = json.loads((RES / "frame.json").read_text())
    users = frame["users"]
    rows = frame["primary_rows"]
    keys = [f"{r['user_id']}|{r['conv_id']}|{int(r['turn_index'])}" for r in rows]
    by_user = [np.asarray(frame["primary_by_user"][u], dtype=int) for u in users]
    targets = [str(r["target"]) for r in rows]
    n_rows, n_users = len(rows), len(users)
    assert (n_users, n_rows) == (236, 1427)

    def user_means(values: np.ndarray) -> np.ndarray:
        return np.asarray([values[idx].mean() for idx in by_user])

    content = {c: json.loads((RES / f"content_{c}.json").read_text()) for c in CANDS}
    for c in CANDS:
        assert content[c]["users"] == users and [r["key"] for r in content[c]["rows"]] == keys, c

    # per-sample MiniLM distances to the real next turn
    per_sample: dict[str, dict[str, np.ndarray]] = {}  # cand -> arm -> (rows, K)
    emb_gen: dict[str, dict[str, np.ndarray]] = {}  # cand -> arm -> (rows*K, 384)
    t0 = time.time()
    emb_real, _ = embed_many(targets)
    for c in CANDS:
        texts = load_samples(RES / content[c]["sample_file"], n_rows)
        flat = [texts[(k, arm)][j] for arm in ARMS for k in keys for j in range(K)]
        uniq = unique_in_order(flat)
        mat, _ = embed_many(uniq)
        index = {t: i for i, t in enumerate(uniq)}
        per_sample[c], emb_gen[c] = {}, {}
        for ai, arm in enumerate(ARMS):
            block = flat[ai * n_rows * K : (ai + 1) * n_rows * K]
            e = mat[[index[t] for t in block]]
            emb_gen[c][arm] = e
            per_sample[c][arm] = cosine_dist(e, np.repeat(emb_real, K, axis=0)).reshape(n_rows, K)
        log(f"{c}: {len(uniq)} unique texts embedded")
    log(f"embedding done in {time.time() - t0:.0f}s")

    gain8 = {
        "avg@8": {
            c: user_means(per_sample[c]["A4"].mean(1) - per_sample[c]["A3"].mean(1)) for c in CANDS
        },
        "best@8": {
            c: user_means(per_sample[c]["A4"].min(1) - per_sample[c]["A3"].min(1)) for c in CANDS
        },
    }

    pop = {c: {} for c in CANDS}
    for c in CANDS:
        for arm in ARMS:
            pop[c][f"frechet_{arm}"] = frechet(emb_gen[c][arm], emb_real)
        pop[c]["frechet_stranger_share_kept"] = pop[c]["frechet_A3"] / pop[c]["frechet_A4"]
        log(f"{c}: Frechet A4 {pop[c]['frechet_A4']:.4f} A3 {pop[c]['frechet_A3']:.4f}")

    name = "population distance, Frechet (A4)"
    score = {c: pop[c]["frechet_A4"] for c in CANDS}
    selector_rows = {}
    for ep in gain8:
        r = r2(gain8[ep], score)
        selector_rows[f"{name} | {ep}"] = {kk: v for kk, v in r.items() if kk != "pairs"}
        log(f"{name} | {ep}: agree {r['n_agree']}/{r['n_resolvable']} tau {r['kendall_tau']:+.2f}")

    out = {
        "n_users": n_users,
        "n_rows": n_rows,
        "metric": METRIC,
        "k": K,
        "seed": SEED,
        "nboot": NBOOT,
        "rows": selector_rows,
        "population": pop,
    }
    paths.ensure(RES)
    (RES / "frechet.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    log("wrote frechet.json")


if __name__ == "__main__":
    main()
