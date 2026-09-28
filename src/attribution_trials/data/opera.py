"""OPeRA data layer: per-decision records for the shopping domain.

Emits the same per-decision record contract as
``attribution_trials.audit.sft_matrix.build_kt`` / ``build_chess``, so the
user-profile LoRA, the frozen prompt panel, the released-simulator panel and
the API panel all consume it unchanged:

    build_opera(channel, config) -> (train, evals, cells, frozen_card_of)

Each record: {player, t, base, self_card, frozen_card, group_card,
placebo_card, label}. Cards are rendered from a numeric causal feature
row (persona[static] + running-state[dynamic]) so the population/cell
means flow through the same _mean_traj machinery, giving placebo
(A1) = population-mean persona/state and group (A2) = cell-mean,
parallel to chess/KT.

Two channels:
  action  -- discrete next-action label (click_type else action_type)
  timing  -- inter-action dwell bucket (edges 1.5/5/20/60 s)

Three configs:
  static   -- card = persona only
  dynamic  -- card = running-state only
  combined -- card = persona + running-state

The substrate string encodes both: "opera-<channel>-<config>", e.g.
"opera-action-combined". Data: HF dataset NEU-HAI/OPeRA at a pinned
revision, filtered tier.
"""

from __future__ import annotations

import glob
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from attribution_trials import paths

REPO_ID = "NEU-HAI/OPeRA"
REVISION = "6f26a2c5cc69084f1714e39db9776e61791344d6"
TIER = "OPeRA_filtered"
TRAIN_FRAC = 0.7
MIN_DEC = 20  # users below this are dropped
MIN_EVAL = 3  # need >= this many held-out decisions
DWELL_EDGES = [1.5, 5.0, 20.0, 60.0]
DWELL_LABELS = ["under 1.5s", "1.5 to 5s", "5 to 20s", "20 to 60s", "over 60s"]


# --------------------------------------------------------------- load --


@lru_cache(maxsize=1)
def _snapshot() -> str:
    """Local copy of the pinned dataset revision (the filtered tier only)."""
    return snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=f"{TIER}/*",
        local_dir=paths.DATA / "opera",
    )


def _load(part: str) -> pd.DataFrame:
    snap = _snapshot()
    fs = sorted(glob.glob(f"{snap}/{TIER}/{part}/**/*.parquet", recursive=True))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


# ------------------------------------------------------------ persona --


def _num(x, default=0.5):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _age_norm(s: str) -> float:
    # "25-34 years old" -> midpoint / 80
    ds = [int(t) for t in "".join(c if c.isdigit() else " " for c in str(s)).split()]
    return (sum(ds) / len(ds) / 80.0) if ds else 0.5


NP = 13  # persona feature count (row = [has_hist] + persona[NP] + state[NS])
NS = 6

_LEVEL = {
    "extremely low": 0.0,
    "very low": 0.12,
    "low": 0.25,
    "medium": 0.5,
    "moderate": 0.5,
    "average": 0.5,
    "high": 0.75,
    "very high": 0.9,
    "extremely high": 1.0,
}
_AGREE = {
    "strongly disagree": 0.0,
    "disagree": 0.2,
    "somewhat disagree": 0.35,
    "neutral": 0.5,
    "neither": 0.5,
    "somewhat agree": 0.65,
    "agree": 0.8,
    "strongly agree": 1.0,
}


def _lookup(table: dict, s, default=0.5) -> float:
    s = str(s).lower().strip()
    if s in table:
        return table[s]
    for k, v in table.items():  # substring fallback
        if k in s:
            return v
    return default


def _freq_norm(s: str) -> float:
    s = str(s).lower()
    if "day" in s or "daily" in s:
        return 1.0 if ("multiple" in s or "several" in s) else 0.85
    if "week" in s:
        if "couple" in s or "every other" in s or "two" in s:
            return 0.4
        return 0.75 if ("twice" in s or "several" in s or "few" in s) else 0.6
    if "month" in s:
        return 0.2 if ("less" in s or "once" in s) else 0.3
    if "rare" in s or "seldom" in s:
        return 0.1
    return 0.5


def _likert(shop: dict, needle: str) -> float:
    block = shop.get("To what extent do you agree with the following statements", {}) or {}
    if isinstance(block, dict):
        for k, v in block.items():
            if needle.lower() in str(k).lower():
                return _lookup(_AGREE, v)
    return 0.5


def parse_persona(survey_json: str) -> list[float]:
    """NP=13 static persona features in [0,1]:
    [age, female, O, C, E, A, N, freq, spend, prime,
     research, ad_attn, holiday]."""
    try:
        sv = json.loads(survey_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        sv = {}
    demo = sv.get("Demographic Information", {}) or {}
    shop = sv.get("Shopping Preference", {}) or {}
    pers = sv.get("Personality", {}) or {}
    age = _age_norm(demo.get("Age", ""))
    female = 1.0 if "female" in str(demo.get("Gender", "")).lower() else 0.0
    b5 = pers.get("Big Five Scores", {}) or {}
    b5 = b5 if isinstance(b5, dict) else {}

    def trait(*names):
        for k, v in b5.items():
            if any(nm in str(k).lower() for nm in names):
                return _lookup(_LEVEL, v)
        return 0.5

    O = trait("intellect", "openness")
    C = trait("conscien")
    E = trait("extra")
    A = trait("agree")
    N = 1.0 - trait("emotional", "stability", "neurotic")  # -> neuroticism
    freq = _freq_norm(shop.get("Online shopping frequency", ""))
    spend = min(1.0, _num(shop.get("Monthly online shopping spend $", 100), 100.0) / 500.0)
    prime = 1.0 if "yes" in str(shop.get("Amazon Prime membership", "")).lower() else 0.0
    research = _likert(shop, "research")
    ad_attn = _likert(shop, "ads")
    holiday = _likert(shop, "holiday")
    return [age, female, O, C, E, A, N, freq, spend, prime, research, ad_attn, holiday]


def _persona_text(pf: list[float]) -> str:
    age, female = pf[0], pf[1]
    o, c, e, a, n = pf[2:7]
    freq, spend, prime = pf[7:10]
    research, ad_attn, holiday = pf[10:13]
    g = "female" if female > 0.5 else "male"
    return (
        f"Shopper: age ~{age * 80:.0f}, {g}; shops online (freq {freq:.2f}), "
        f"~${spend * 500:.0f}/mo, Prime {'yes' if prime > 0.5 else 'no'}; "
        f"Big-Five O/C/E/A/N {o:.2f}/{c:.2f}/{e:.2f}/{a:.2f}/{n:.2f}; "
        f"researches-before-buying {research:.2f}, ad-attentive {ad_attn:.2f}, "
        f"holiday-shopper {holiday:.2f}."
    )


# ------------------------------------------------------------- pages --


def _page_type(url: str) -> str:
    u = str(url)
    if "/s?" in u or "field-keywords" in u or "/s/" in u:
        return "search results"
    if "/dp/" in u or "/gp/product" in u:
        return "product page"
    if "cart" in u:
        return "cart"
    if u.rstrip("/").endswith(".com") or "/ref=nav" in u:
        return "home"
    return "other page"


def _loads_dict(s) -> dict:
    """Robust to double-encoded JSON (some page_meta are JSON-in-JSON)."""
    for _ in range(2):
        if isinstance(s, dict):
            return s
        try:
            s = json.loads(s)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return s if isinstance(s, dict) else {}


def _cart_size(page_meta) -> float:
    m = _loads_dict(page_meta)
    return float(len(m.get("cart_items", []) or []))


def _search_term(page_meta) -> str:
    m = _loads_dict(page_meta)
    st = m.get("search_term", [])
    if st and isinstance(st, list) and isinstance(st[0], dict):
        return str(st[0].get("term", "") or "")
    return ""


def _action_label(action_type: str, click_type: str) -> str:
    if str(action_type) == "click":
        return str(click_type).replace("_", " ")
    return str(action_type)


# --------------------------------------------------------- assemble ---


@lru_cache(maxsize=1)
def _raw():
    u = _load("user").drop_duplicates("user_id")
    a = _load("action")
    a["user_id"] = a.session_id.str.split("_").str[0]
    a["ts"] = pd.to_datetime(a.timestamp, format="ISO8601", utc=True)
    a = a.sort_values(["user_id", "session_id", "ts"]).reset_index(drop=True)
    a["dwell"] = a.groupby("session_id").ts.diff().dt.total_seconds()
    persona = {r.user_id: parse_persona(r.survey) for r in u.itertuples()}
    return a, persona


def _bucket(seconds: float) -> str:
    for e, lab in zip(DWELL_EDGES, DWELL_LABELS):
        if seconds < e:
            return lab
    return DWELL_LABELS[-1]


def _user_rows(g: pd.DataFrame, pf: list[float]):
    """Causal feature rows + per-decision (base_ctx, action_label, dwell).

    Row = [has_hist] + persona(NP) + state(NS):
      state = [sess_idx_n, frac_click, prod_views_n, cart_n,
               frac_search, recent_dwell_n]
    """
    rows, ctx = [], []
    sess_idx = {}
    nclick = nsearch = nprod = 0
    recent = []
    prev_sess = None
    for i, r in enumerate(g.itertuples()):
        if r.session_id != prev_sess:
            prev_sess = r.session_id
        sess_idx[r.session_id] = sess_idx.get(r.session_id, 0)
        si = sess_idx[r.session_id]
        has = 1.0 if i > 0 else 0.0
        tot = max(i, 1)
        state = [
            min(1.0, si / 30.0),
            nclick / tot,
            min(1.0, nprod / 50.0),
            min(1.0, _cart_size(r.page_meta) / 10.0),
            nsearch / tot,
            min(1.0, (sum(recent[-5:]) / len(recent[-5:]) / 60.0) if recent else 0.0),
        ]
        rows.append([has] + list(pf) + state)
        lab = _action_label(r.action_type, r.click_type)
        ctx.append(
            {
                "page": _page_type(r.url),
                "term": _search_term(r.page_meta)[:40],
                "cart": int(_cart_size(r.page_meta)),
                "recent": [c["label"] for c in ctx[-3:]],
                "label": lab,
                "dwell": float(r.dwell) if pd.notna(r.dwell) else None,
            }
        )
        # advance causal counters AFTER emitting row for decision i
        sess_idx[r.session_id] += 1
        if str(r.action_type) == "click":
            nclick += 1
        if "search" in str(r.click_type):
            nsearch += 1
        if _page_type(r.url) == "product page":
            nprod += 1
        if pd.notna(r.dwell):
            recent.append(float(r.dwell))
    return rows, ctx


def _render(row: list[float], config: str) -> str | None:
    persona = _persona_text(row[1 : 1 + NP])
    s = row[1 + NP : 1 + NP + NS]
    if row[0] < 0.5:
        state = "Session just started."
    else:
        state = (
            f"Session so far: {s[1] * 100:.0f}% clicks, "
            f"~{s[2] * 50:.0f} product views, cart {s[3] * 10:.0f}; "
            f"recent pause ~{s[5] * 60:.0f}s."
        )
    if config == "static":
        return persona
    if config == "dynamic":
        return state
    return persona + " " + state


def _mean_traj(rows_of, members, splits_of):
    horizon = max(splits_of[p] for p in members)
    out, last = [], None
    for t in range(horizon):
        rs = [rows_of[p][t] for p in members if t < splits_of[p]]
        last = [sum(c) / len(c) for c in zip(*rs)] if rs else last
        out.append(list(last))
    return out


def _base(ctx: dict, channel: str) -> str:
    recent = ", ".join(ctx["recent"]) if ctx["recent"] else "none"
    head = (
        f"On Amazon: {ctx['page']}"
        + (f", searching '{ctx['term']}'" if ctx["term"] else "")
        + f", {ctx['cart']} items in cart.\nRecent actions: {recent}."
    )
    if channel == "timing":
        return (
            "Predict how long the shopper pauses before their next action.\n"
            + head
            + "\nBuckets: "
            + ", ".join(DWELL_LABELS)
            + "."
        )
    return (
        "Predict the shopper's next action.\n"
        + head
        + "\nActions: click a product link, review, product option, search, "
        "suggested term, filter, nav bar, add quantity, cart, purchase, "
        "input, terminate."
    )


def build_opera(channel: str = "action", config: str = "combined"):
    a, persona = _raw()
    users = [uid for uid, _ in a.groupby("user_id")]
    rows_of, ctx_of, splits_of, keep = {}, {}, {}, []
    for uid in users:
        g = a[a.user_id == uid]
        pf = persona.get(uid, [0.5] * 10)
        rows, ctx = _user_rows(g, pf)
        # per-user session-level temporal split at TRAIN_FRAC
        sess_order, seen = [], set()
        for c_sid in g.session_id.tolist():
            if c_sid not in seen:
                seen.add(c_sid)
                sess_order.append(c_sid)
        n_train_sess = max(1, int(round(TRAIN_FRAC * len(sess_order))))
        if len(sess_order) >= 2:
            train_sids = set(sess_order[:n_train_sess])
            k = sum(1 for sid in g.session_id.tolist() if sid in train_sids)
        else:  # single-session user -> split within session
            k = max(1, int(round(TRAIN_FRAC * len(rows))))
        if len(rows) < MIN_DEC or (len(rows) - k) < MIN_EVAL or k < 1:
            continue
        rows_of[uid] = rows
        ctx_of[uid] = ctx
        splits_of[uid] = k
        keep.append(uid)

    # cells: tertile by a STATIC persona scalar (monthly spend) -> channel-
    # independent, parallel to chess Elo / KT prior-accuracy tertiles.
    scalar = {u: persona.get(u, [0.5] * 10)[8] for u in keep}
    order = sorted(keep, key=lambda u: scalar[u])
    cells = {}
    for i, u in enumerate(order):
        cells[u] = f"spend_t{min(2, i * 3 // max(1, len(order)))}"

    pop = _mean_traj(rows_of, keep, splits_of)
    cell_traj = {
        c: _mean_traj(rows_of, [u for u in keep if cells[u] == c], splits_of)
        for c in set(cells.values())
    }

    train, evals = [], []
    for uid in keep:
        rows, ctx, k = rows_of[uid], ctx_of[uid], splits_of[uid]
        ct = cell_traj[cells[uid]]
        for t, c in enumerate(ctx):
            if channel == "timing" and c["dwell"] is None:
                continue
            label = _bucket(c["dwell"]) if channel == "timing" else c["label"]
            rec = {
                "player": uid,
                "t": t,
                "base": _base(c, channel),
                "self_card": _render(rows[t], config),
                "frozen_card": _render(rows[min(k, len(rows) - 1)], config),
                "group_card": _render(ct[min(t, len(ct) - 1)], config),
                "placebo_card": _render(pop[min(t, len(pop) - 1)], config),
                "label": label,
            }
            (train if t < k else evals).append(rec)
    frozen_card_of = {
        u: _render(rows_of[u][min(splits_of[u], len(rows_of[u]) - 1)], config) for u in keep
    }
    return train, evals, cells, frozen_card_of
