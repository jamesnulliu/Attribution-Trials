"""Content-space readouts R1-R5 for the seven WildChat simulators.

R1  content gain (A4 minus A3 distance to the real turn, per user) and the
    target win rate, avg@8 and best@8;
R2  agreement of the AT ordering (dind) with the content-gain ordering on the
    resolvable simulator pairs, Kendall tau, and user-fold cross-fit tau;
R3  the same for absolute A4 distance;
R4  user-level Spearman between AT (odd rows) and content gain (even rows);
R5  form labels of the same samples scored against the realized label.

Writes ``readouts.json``, ``readout_pairs.md`` (per-pair R2/R3 tables and the
subset without CoSER-8B) and ``empty_rate.json`` under ``paths.WILDCHAT``.

  python -m attribution_trials.wildchat.readouts
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np

from attribution_trials import paths
from attribution_trials.wildchat.common import SIMULATORS, row_key

RESULTS = paths.WILDCHAT
FRAME_PATH = paths.WILDCHAT / "frame.json"
METRIC_M1 = "M1_minilm_max_card_bullet"
METRIC_M2 = "M2_char_wb_tfidf_whole_card"
METRICS = (METRIC_M1, METRIC_M2)
CANDIDATES_ORDER = tuple(SIMULATORS)
DIMENSIONS = ("length", "question", "code", "terminal")
ARMS = ("A0", "A3", "A4")
NBOOT = 10_000
SEED = 0
SUPPORT_THRESHOLD = 0.70
REFUTE_THRESHOLD = 0.50
ENDPOINT_LABELS = {"D": "avg@8", "Dmin": "best@8"}
SCALE_FILES = {
    "1.7b": RESULTS / "scale_1.7b.json",
    "8b": RESULTS / "scale_8b.json",
    "dense": RESULTS / "scale_32b.json",
    "humanlike": RESULTS / "scale_humanlike.json",
    "coser": RESULTS / "scale_coser.json",
    "osim": RESULTS / "scale_osim.json",
    "humanlm": RESULTS / "scale_humanlm.json",
}


# ---------------------------------------------------------------- statistics


def bootstrap_mean(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty vector")
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return {
        "n_users": int(len(values)),
        "point": float(values.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0:
                ties_x += 1
            if dy == 0:
                ties_y += 1
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n0 = len(x) * (len(x) - 1) // 2
    denominator = math.sqrt(max(n0 - ties_x, 0) * max(n0 - ties_y, 0))
    if denominator == 0:
        return 0.0
    return float((concordant - discordant) / denominator)


def bootstrap_tau(
    at: np.ndarray, downstream: np.ndarray, at_indices: list[int], downstream_indices: list[int]
) -> dict:
    at = np.asarray(at, dtype=float)
    downstream = np.asarray(downstream, dtype=float)
    rng = np.random.default_rng(SEED)
    at_indices = np.asarray(at_indices, dtype=int)
    downstream_indices = np.asarray(downstream_indices, dtype=int)
    values = []
    for _ in range(NBOOT):
        ai = at_indices[rng.integers(0, len(at_indices), size=len(at_indices))]
        di = downstream_indices[
            rng.integers(0, len(downstream_indices), size=len(downstream_indices))
        ]
        values.append(kendall_tau(at[ai].mean(axis=0), downstream[di].mean(axis=0)))
    values = np.asarray(values, dtype=float)
    return {
        "n_at_users": int(len(at_indices)),
        "n_downstream_users": int(len(downstream_indices)),
        "point": kendall_tau(
            at[at_indices].mean(axis=0), downstream[downstream_indices].mean(axis=0)
        ),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def bootstrap_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return {"n_users": int(len(x)), "point": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(NBOOT):
        index = rng.integers(0, len(x), size=len(x))
        values.append(spearman(x[index], y[index]))
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    point = spearman(x, y)
    # A constant vector leaves every resample undefined; report n/a.
    if len(values) == 0:
        return {
            "n_users": int(len(x)),
            "point": float(point) if np.isfinite(point) else None,
            "ci_low": None,
            "ci_high": None,
        }
    return {
        "n_users": int(len(x)),
        "point": float(point) if np.isfinite(point) else None,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


# ---------------------------------------------------------------- AT scores


def key_for(uid: str, key: list | tuple) -> str:
    return f"{uid}|{key[0]}|{int(key[1])}"


def scale_rows(node: dict) -> tuple[list[str], dict[str, np.ndarray]]:
    if "rows" in node:
        keys = [str(row["key"]) for row in node["rows"]]
        nll = {
            arm: np.asarray([row["nll"][arm] for row in node["rows"]], dtype=float) for arm in ARMS
        }
    else:
        keys = [key_for(uid, key) for uid, key in zip(node["players"], node["keys"])]
        nll = {arm: np.asarray(node["nll"][arm], dtype=float) for arm in ARMS}
    if any(len(values) != len(keys) for values in nll.values()):
        raise AssertionError("NLL/key length mismatch")
    if not all(np.isfinite(values).all() for values in nll.values()):
        raise AssertionError("non-finite NLL")
    return keys, nll


def load_at(frame: dict) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    reference = json.loads(SCALE_FILES["8b"].read_text())
    ref_keys, _ = scale_rows(reference)
    ref_digest = reference["row_digest"]
    eligible = set(frame["users"])
    at_indiv, at_total, row_deltas = {}, {}, {}
    for candidate in CANDIDATES_ORDER:
        node = json.loads(SCALE_FILES[candidate].read_text())
        keys, nll = scale_rows(node)
        if keys != ref_keys:
            raise AssertionError(f"{candidate}: scored rows differ from the Qwen3-8B file")
        if node.get("row_digest") not in (None, ref_digest):
            raise AssertionError(f"{candidate}: row digest differs")
        if "rows" in node:
            row_users = [row["user_id"] for row in node["rows"]]
        else:
            row_users = list(node["players"])
        d_indiv = nll["A4"] - nll["A3"]
        d_total = nll["A4"] - nll["A0"]
        by_user_indiv = {
            u: float(d_indiv[np.asarray([i for i, x in enumerate(row_users) if x == u])].mean())
            for u in sorted(eligible)
        }
        by_user_total = {
            u: float(d_total[np.asarray([i for i, x in enumerate(row_users) if x == u])].mean())
            for u in sorted(eligible)
        }
        at_indiv[candidate] = by_user_indiv
        at_total[candidate] = by_user_total
        row_deltas[candidate] = {
            "keys": keys,
            "users": row_users,
            "indiv": d_indiv,
            "total": d_total,
        }
    return {"indiv": at_indiv, "total": at_total}, row_deltas


# ---------------------------------------------------------------- readouts


def sign(value: float) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


def preference(sign_value: int, left: str, right: str) -> str | None:
    if sign_value < 0:
        return left
    if sign_value > 0:
        return right
    return None


def validate_frame(frame: dict) -> None:
    if frame.get("schema") != "wildchat-frame-v1":
        raise AssertionError(f"unexpected frame schema: {frame.get('schema')}")
    if frame.get("n_users") != 236 or frame.get("n_primary_rows") != 1427:
        raise AssertionError("readouts require the 236-user / 1,427-row frame")


def load_content_outputs(frame: dict) -> dict[str, dict]:
    expected_keys = [row_key(row["user_id"], row) for row in frame["primary_rows"]]
    outputs: dict[str, dict] = {}
    for candidate in CANDIDATES_ORDER:
        path = RESULTS / f"content_{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        node = json.loads(path.read_text())
        for key, expected in (
            ("schema", "wildchat-content-v1"),
            ("frame_schema", frame["schema"]),
            ("frame_primary_digest", frame["primary_row_digest"]),
            ("n_users", 236),
            ("n_primary_rows", 1427),
            ("arms", ["A3", "A4"]),
            ("primary_metric", METRIC_M1),
            ("secondary_metric", METRIC_M2),
        ):
            if node.get(key) != expected:
                raise AssertionError(f"{candidate}: {key} contract mismatch")
        if node.get("users") != frame["users"]:
            raise AssertionError(f"{candidate}: user order mismatch")
        rows = node.get("rows")
        if not isinstance(rows, list) or len(rows) != len(expected_keys):
            raise AssertionError(f"{candidate}: row count mismatch")
        if [row.get("key") for row in rows] != expected_keys:
            raise AssertionError(f"{candidate}: row-key order mismatch")
        for row in rows:
            metrics = row.get("metrics", {})
            if set(metrics) != set(METRICS):
                raise AssertionError(f"{candidate}: missing content metric")
            for metric in METRICS:
                values = metrics[metric]
                if set(values) != {"D_A3", "D_A4", "Dmin_A3", "Dmin_A4"}:
                    raise AssertionError(f"{candidate}/{metric}: distance schema mismatch")
                if not all(np.isfinite(float(value)) for value in values.values()):
                    raise AssertionError(f"{candidate}/{metric}: non-finite distance")
        sample_path = paths.WILDCHAT / node["sample_file"]
        if node.get("sample_count") != 1427 * 2 * 8:
            raise AssertionError(f"{candidate}: sample count mismatch")
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)
        outputs[candidate] = node
    return outputs


def load_form_outputs(frame: dict) -> dict[str, dict]:
    forms: dict[str, dict] = {}
    for candidate in CANDIDATES_ORDER:
        path = RESULTS / f"form_{candidate}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        node = json.loads(path.read_text())
        for key, expected in (
            ("schema", "wildchat-form-v1"),
            ("frame_primary_digest", frame["primary_row_digest"]),
            ("n_users", 236),
            ("n_primary_rows", 1427),
            ("arms", ["A3", "A4"]),
        ):
            if node.get(key) != expected:
                raise AssertionError(f"{candidate} form: {key} contract mismatch")
        if node.get("users") != frame["users"]:
            raise AssertionError(f"{candidate} form: user order mismatch")
        for arm in ("A3", "A4"):
            labels = node.get("sampled_labels", {}).get(arm)
            if not isinstance(labels, list) or len(labels) != 1427:
                raise AssertionError(f"{candidate} form: {arm} row count mismatch")
            if any(len(row) != 8 for row in labels):
                raise AssertionError(f"{candidate} form: {arm} K mismatch")
        for dimension in DIMENSIONS:
            hist = node.get("histograms", {}).get(dimension, {})
            for arm in ("A3", "A4"):
                if len(hist.get(arm, [])) != 1427:
                    raise AssertionError(f"{candidate} form: {dimension}/{arm} histogram mismatch")
        forms[candidate] = node
    return forms


def row_arrays(outputs: dict[str, dict], metric: str, field: str) -> dict[str, np.ndarray]:
    return {
        candidate: np.asarray(
            [float(row["metrics"][metric][field]) for row in outputs[candidate]["rows"]],
            dtype=float,
        )
        for candidate in CANDIDATES_ORDER
    }


def user_means(frame: dict, values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(values[np.asarray(frame["primary_by_user"][user], dtype=int)].mean())
            for user in frame["users"]
        ],
        dtype=float,
    )


def two_afc(a3: np.ndarray, a4: np.ndarray) -> np.ndarray:
    return np.where(a4 < a3, 1.0, np.where(a4 > a3, 0.0, 0.5))


def r1_readout(frame: dict, outputs: dict[str, dict], metric: str) -> dict:
    d3 = row_arrays(outputs, metric, "D_A3")
    d4 = row_arrays(outputs, metric, "D_A4")
    dmin3 = row_arrays(outputs, metric, "Dmin_A3")
    dmin4 = row_arrays(outputs, metric, "Dmin_A4")
    rows: dict[str, dict] = {}
    user_delta: dict[str, np.ndarray] = {}
    user_delta_min: dict[str, np.ndarray] = {}
    user_abs_a4: dict[str, np.ndarray] = {}
    user_abs_a4_min: dict[str, np.ndarray] = {}
    row_delta: dict[str, np.ndarray] = {}
    row_delta_min: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES_ORDER:
        delta = d4[candidate] - d3[candidate]
        delta_min = dmin4[candidate] - dmin3[candidate]
        row_delta[candidate] = delta
        row_delta_min[candidate] = delta_min
        user_delta[candidate] = user_means(frame, delta)
        user_delta_min[candidate] = user_means(frame, delta_min)
        user_abs_a4[candidate] = user_means(frame, d4[candidate])
        user_abs_a4_min[candidate] = user_means(frame, dmin4[candidate])
        afc = two_afc(d3[candidate], d4[candidate])
        afc_min = two_afc(dmin3[candidate], dmin4[candidate])
        delta_stats = bootstrap_mean(user_delta[candidate])
        afc_stats = bootstrap_mean(user_means(frame, afc))
        afc_stats.update(
            {
                "chance": 0.5,
                "ci_low_minus_chance": afc_stats["ci_low"] - 0.5,
                "ci_high_minus_chance": afc_stats["ci_high"] - 0.5,
            }
        )
        delta_min_stats = bootstrap_mean(user_means(frame, delta_min))
        afc_min_stats = bootstrap_mean(user_means(frame, afc_min))
        afc_min_stats.update(
            {
                "chance": 0.5,
                "ci_low_minus_chance": afc_min_stats["ci_low"] - 0.5,
                "ci_high_minus_chance": afc_min_stats["ci_high"] - 0.5,
            }
        )
        rows[candidate] = {
            "behavioral_delta_indiv": delta_stats,
            "two_afc": afc_stats,
            "behavioral_delta_indiv_Dmin": delta_min_stats,
            "two_afc_Dmin": afc_min_stats,
            "mean_D_A4": float(user_abs_a4[candidate].mean()),
            "mean_Dmin_A4": float(user_abs_a4_min[candidate].mean()),
        }
    return {
        "metric": metric,
        "endpoint_labels": ENDPOINT_LABELS,
        "rows": rows,
        "row_delta": row_delta,
        "row_delta_Dmin": row_delta_min,
        "user_delta": user_delta,
        "user_delta_Dmin": user_delta_min,
        "user_abs_a4": user_abs_a4,
        "user_abs_a4_Dmin": user_abs_a4_min,
    }


def classify(n_resolvable: int, fraction: float | None) -> str:
    if n_resolvable < 10:
        return "underpowered"
    if fraction >= SUPPORT_THRESHOLD:
        return "support"
    if fraction <= REFUTE_THRESHOLD:
        return "refute"
    return "inconclusive"


def r2_readout(
    user_delta: dict[str, np.ndarray],
    at_user: dict[str, np.ndarray],
    candidates: tuple[str, ...] = CANDIDATES_ORDER,
    endpoint: str = "avg@8",
) -> dict:
    candidates = tuple(candidates)
    at_means = {candidate: float(at_user[candidate].mean()) for candidate in candidates}
    behavioral_means = {candidate: float(user_delta[candidate].mean()) for candidate in candidates}
    pairs: list[dict] = []
    for left, right in itertools.combinations(candidates, 2):
        at_diff = at_means[left] - at_means[right]
        behavioral_diff = user_delta[left] - user_delta[right]
        stats = bootstrap_mean(behavioral_diff)
        at_sign = sign(at_diff)
        behavioral_sign = sign(float(behavioral_diff.mean()))
        resolvable = bool(stats["ci_high"] < 0 or stats["ci_low"] > 0)
        agreement = bool(resolvable and at_sign != 0 and at_sign == behavioral_sign)
        pairs.append(
            {
                "endpoint": endpoint,
                "left": left,
                "right": right,
                "at_delta_indiv_left_minus_right": at_diff,
                "at_preferred": preference(at_sign, left, right),
                "behavioral_delta_left_minus_right": stats,
                "behavioral_preferred": preference(behavioral_sign, left, right),
                "resolvable": resolvable,
                "agreement": agreement,
            }
        )
    resolvable_pairs = [pair for pair in pairs if pair["resolvable"]]
    support = sum(bool(pair["agreement"]) for pair in resolvable_pairs)
    count = len(resolvable_pairs)
    support_fraction = support / count if count else None

    at_vector = np.asarray([at_means[candidate] for candidate in candidates])
    behavioral_vector = np.asarray([behavioral_means[candidate] for candidate in candidates])
    return {
        "endpoint": endpoint,
        "support_threshold": SUPPORT_THRESHOLD,
        "refute_threshold": REFUTE_THRESHOLD,
        "pairs": pairs,
        "n_resolvable": count,
        "n_agree": support,
        "support_fraction": support_fraction,
        "classification": classify(count, support_fraction),
        "at_order_best_to_worst": sorted(candidates, key=lambda c: at_means[c]),
        "behavioral_order_best_to_worst": sorted(candidates, key=lambda c: behavioral_means[c]),
        "kendall_tau_point": kendall_tau(at_vector, behavioral_vector),
        "at_means": at_means,
        "behavioral_means": behavioral_means,
    }


def crossfit_readout(
    frame: dict,
    at_user: dict[str, np.ndarray],
    behavioral_user: dict[str, np.ndarray],
    candidates: tuple[str, ...] = CANDIDATES_ORDER,
) -> dict:
    candidates = tuple(candidates)
    users = frame["users"]
    at_matrix = np.asarray([at_user[candidate] for candidate in candidates]).T
    behavioral_matrix = np.asarray([behavioral_user[candidate] for candidate in candidates]).T
    fold_a = [i for i, user in enumerate(users) if frame["fold_of_user"][user] == "A"]
    fold_b = [i for i, user in enumerate(users) if frame["fold_of_user"][user] == "B"]
    result = {}
    for label, source, target in (("fold_A_to_B", fold_a, fold_b), ("fold_B_to_A", fold_b, fold_a)):
        result[label] = bootstrap_tau(at_matrix, behavioral_matrix, source, target)
        result[label]["at_order_best_to_worst"] = [
            candidates[i] for i in np.argsort(at_matrix[source].mean(axis=0))
        ]
        result[label]["behavioral_order_best_to_worst"] = [
            candidates[i] for i in np.argsort(behavioral_matrix[target].mean(axis=0))
        ]
    return result


def r3_readout(
    user_abs_a4: dict[str, np.ndarray],
    at_user: dict[str, np.ndarray],
    candidates: tuple[str, ...] = CANDIDATES_ORDER,
    endpoint: str = "avg@8",
) -> dict:
    candidates = tuple(candidates)
    means = {candidate: float(user_abs_a4[candidate].mean()) for candidate in candidates}
    at_means = {candidate: float(at_user[candidate].mean()) for candidate in candidates}
    pairs: list[dict] = []
    for left, right in itertools.combinations(candidates, 2):
        at_diff = at_means[left] - at_means[right]
        quality_diff = user_abs_a4[left] - user_abs_a4[right]
        stats = bootstrap_mean(quality_diff)
        at_sign = sign(at_diff)
        quality_sign = sign(float(quality_diff.mean()))
        resolvable = bool(stats["ci_high"] < 0 or stats["ci_low"] > 0)
        agreement = bool(resolvable and at_sign != 0 and at_sign == quality_sign)
        pairs.append(
            {
                "endpoint": endpoint,
                "left": left,
                "right": right,
                "at_delta_indiv_left_minus_right": at_diff,
                "at_preferred": preference(at_sign, left, right),
                "absolute_D_A4_left_minus_right": stats,
                "quality_preferred": preference(quality_sign, left, right),
                "resolvable": resolvable,
                "agreement": agreement,
            }
        )
    resolvable_pairs = [pair for pair in pairs if pair["resolvable"]]
    n_agree = sum(bool(pair["agreement"]) for pair in resolvable_pairs)
    fraction = n_agree / len(resolvable_pairs) if resolvable_pairs else None
    return {
        "absolute_mean_D_A4": means,
        "absolute_order_best_to_worst": sorted(candidates, key=lambda c: means[c]),
        "pairs": pairs,
        "n_resolvable": len(resolvable_pairs),
        "n_agree": n_agree,
        "support_fraction": fraction,
        "classification": classify(len(resolvable_pairs), fraction),
        "endpoint": endpoint,
        "support_threshold": SUPPORT_THRESHOLD,
        "refute_threshold": REFUTE_THRESHOLD,
    }


def restrict_pair_readout(readout: dict, candidates: tuple[str, ...]) -> dict:
    """Restrict an already bootstrapped pair readout to a subset of simulators."""
    candidates = tuple(candidates)
    allowed = set(candidates)
    pairs = [
        pair for pair in readout["pairs"] if pair["left"] in allowed and pair["right"] in allowed
    ]
    resolvable = [pair for pair in pairs if pair["resolvable"]]
    n_agree = sum(bool(pair["agreement"]) for pair in resolvable)
    n_resolvable = len(resolvable)
    fraction = n_agree / n_resolvable if n_resolvable else None
    at_means = {candidate: readout["at_means"][candidate] for candidate in candidates}
    behavioral_means = {
        candidate: readout["behavioral_means"][candidate] for candidate in candidates
    }
    return {
        "endpoint": readout["endpoint"],
        "support_threshold": SUPPORT_THRESHOLD,
        "refute_threshold": REFUTE_THRESHOLD,
        "pairs": pairs,
        "n_resolvable": n_resolvable,
        "n_agree": n_agree,
        "support_fraction": fraction,
        "classification": classify(n_resolvable, fraction),
        "at_order_best_to_worst": sorted(candidates, key=lambda c: at_means[c]),
        "behavioral_order_best_to_worst": sorted(candidates, key=lambda c: behavioral_means[c]),
        "kendall_tau_point": kendall_tau(
            np.asarray([at_means[c] for c in candidates]),
            np.asarray([behavioral_means[c] for c in candidates]),
        ),
        "at_means": at_means,
        "behavioral_means": behavioral_means,
    }


def empty_rate_readout(frame: dict, forms: dict[str, dict]) -> dict:
    """Empty-continuation and cap rates from the sampled labels, per arm."""
    n_samples = int(frame["n_primary_rows"] * 8)
    candidates: dict[str, dict] = {}
    for candidate in CANDIDATES_ORDER:
        candidate_arms: dict[str, dict] = {}
        for arm in ("A3", "A4"):
            samples = [sample for row in forms[candidate]["sampled_labels"][arm] for sample in row]
            if len(samples) != n_samples:
                raise AssertionError(f"{candidate}/{arm}: expected {n_samples} sampled labels")
            generated = np.asarray(
                [int(sample["generated_tokens"]) for sample in samples], dtype=int
            )
            terminal = np.asarray([bool(sample["terminal"]) for sample in samples], dtype=bool)
            terminal_count = int(terminal.sum())
            token_empty_count = int((generated <= 1).sum())
            cap_count = int((generated == 96).sum())
            candidate_arms[arm] = {
                "n_samples": n_samples,
                # Empty means the parser's terminal/empty flag; Qwen3 can decode
                # empty text at the 96-token cap after its reasoning wrapper is
                # stripped, so the literal token threshold is reported separately.
                "empty_count": terminal_count,
                "empty_rate": terminal_count / n_samples,
                "terminal_count": terminal_count,
                "terminal_rate": terminal_count / n_samples,
                "generated_tokens_le_1_count": token_empty_count,
                "generated_tokens_le_1_rate": token_empty_count / n_samples,
                "cap_count": cap_count,
                "cap_rate": cap_count / n_samples,
            }
        candidates[candidate] = {"arms": candidate_arms}
    return {
        "schema": "wildchat-empty-rate-v1",
        "frame_primary_digest": frame["primary_row_digest"],
        "n_users": frame["n_users"],
        "n_primary_rows": frame["n_primary_rows"],
        "n_samples_per_candidate_arm": n_samples,
        "max_new_tokens": 96,
        "empty_definition": "sampled_labels[*].terminal == true (empty decoded continuation)",
        "generated_tokens_le_1_definition": "sampled_labels[*].generated_tokens <= 1",
        "cap_definition": "sampled_labels[*].generated_tokens == 96",
        "candidates": candidates,
    }


def at_frame_rows(frame: dict, row_deltas: dict[str, dict]) -> dict[str, np.ndarray]:
    frame_keys = [row_key(row["user_id"], row) for row in frame["primary_rows"]]
    out = {}
    for candidate in CANDIDATES_ORDER:
        lookup = {
            key: float(value)
            for key, value in zip(row_deltas[candidate]["keys"], row_deltas[candidate]["indiv"])
        }
        if any(key not in lookup for key in frame_keys):
            raise AssertionError(f"{candidate}: AT row mapping does not cover the frame")
        out[candidate] = np.asarray([lookup[key] for key in frame_keys], dtype=float)
    return out


def r4_readout(
    frame: dict, at_rows: dict[str, np.ndarray], behavioral_rows: dict[str, np.ndarray]
) -> dict:
    users = frame["users"]
    result: dict[str, dict] = {}
    for candidate in CANDIDATES_ORDER:
        result[candidate] = {}
        for at_parity, downstream_parity in (("odd", "even"), ("even", "odd")):
            at_values = []
            downstream_values = []
            for user in users:
                at_indices = np.asarray(frame["row_splits"][user][at_parity], dtype=int)
                downstream_indices = np.asarray(
                    frame["row_splits"][user][downstream_parity], dtype=int
                )
                at_values.append(float(at_rows[candidate][at_indices].mean()))
                downstream_values.append(
                    float(behavioral_rows[candidate][downstream_indices].mean())
                )
            result[candidate][f"at_{at_parity}_downstream_{downstream_parity}"] = (
                bootstrap_spearman(np.asarray(at_values), np.asarray(downstream_values))
            )
    return result


def form_r5_readout(frame: dict, forms: dict[str, dict]) -> dict:
    result: dict[str, dict] = {}
    for candidate in CANDIDATES_ORDER:
        result[candidate] = {}
        node = forms[candidate]
        for dimension in DIMENSIONS:
            labels = np.asarray(node["realized_labels"][dimension], dtype=int)
            hist3 = np.asarray(node["histograms"][dimension]["A3"], dtype=float)
            hist4 = np.asarray(node["histograms"][dimension]["A4"], dtype=float)
            selected = np.arange(len(labels), dtype=int)
            p3 = hist3[selected, labels]
            p4 = hist4[selected, labels]
            row_delta = -np.log(np.clip(p4, 1e-12, None)) + np.log(np.clip(p3, 1e-12, None))
            per_user_delta = user_means(frame, row_delta)
            per_user_afc = user_means(frame, two_afc(p3, p4))
            delta = bootstrap_mean(per_user_delta)
            afc = bootstrap_mean(per_user_afc)
            afc.update(
                {
                    "chance": 0.5,
                    "ci_low_minus_chance": afc["ci_low"] - 0.5,
                    "ci_high_minus_chance": afc["ci_high"] - 0.5,
                }
            )
            result[candidate][dimension] = {
                "sampled_delta_indiv": delta,
                "two_afc_realized_label": afc,
            }
    return result


def outcome(classification: str) -> str:
    if classification == "support":
        return "A"
    if classification == "refute":
        return "C"
    if classification == "underpowered":
        return "B"
    return "INCONCLUSIVE"


def fmt(stats: dict) -> str:
    if stats.get("point") is None:
        return "n/a"
    return f"{stats['point']:+.5f} [{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"


def write_outputs(
    frame: dict,
    metric_readouts: dict[str, dict],
    form_readout: dict,
    r4: dict,
    primary_metric: str,
    empty_rates: dict,
) -> None:
    paths.ensure(RESULTS)
    primary_r2 = metric_readouts[primary_metric]["r2"]
    primary_outcome = outcome(primary_r2["classification"])
    table = {
        "schema": "wildchat-readouts-v1",
        "frame_primary_digest": frame["primary_row_digest"],
        "n_users": frame["n_users"],
        "n_primary_rows": frame["n_primary_rows"],
        "primary_metric": primary_metric,
        "secondary_metric": METRIC_M2 if primary_metric == METRIC_M1 else METRIC_M1,
        "endpoint_labels": ENDPOINT_LABELS,
        "classification_thresholds": {
            "support": SUPPORT_THRESHOLD,
            "refute": REFUTE_THRESHOLD,
            "minimum_resolvable_pairs": 10,
        },
        "outcome": primary_outcome,
        "outcome_endpoint": "avg@8",
        "metric_readouts": {
            metric: {
                "r1": readout["r1"]["rows"],
                "r2": readout["r2"],
                "r2_best": readout["r2_best"],
                "r3": readout["r3"],
                "r3_best": readout["r3_best"],
                "r4": r4[metric],
            }
            for metric, readout in metric_readouts.items()
        },
        "r5_form": form_readout,
        "empty_rate_file": "empty_rate.json",
    }
    (RESULTS / "readouts.json").write_text(json.dumps(table, indent=2) + "\n")

    def add_pair_section(lines: list[str], title: str, r2: dict, r3: dict) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"Endpoint: **{r2['endpoint']}**. Pairwise bootstrap unit is user, seed 0, 10,000 resamples.",
                "",
                "### R2 behavioral Δindiv",
                "",
                "| left | right | AT preferred | behavioral Δ left−right | behavioral preferred | resolvable | agree |",
                "|---|---|---|---|---|:---:|:---:|",
            ]
        )
        for pair in r2["pairs"]:
            lines.append(
                f"| {pair['left']} | {pair['right']} | {pair['at_preferred'] or 'tie'} | "
                f"{fmt(pair['behavioral_delta_left_minus_right'])} | {pair['behavioral_preferred'] or 'tie'} | "
                f"{'yes' if pair['resolvable'] else 'no'} | {'yes' if pair['agreement'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                f"R2: {r2['n_agree']}/{r2['n_resolvable']} agreement among resolvable pairs; "
                f"classification `{r2['classification']}`.",
                "",
                "### R3 absolute distance",
                "",
                "| left | right | AT preferred | absolute distance left−right | quality preferred | resolvable | agree |",
                "|---|---|---|---|---|:---:|:---:|",
            ]
        )
        for pair in r3["pairs"]:
            lines.append(
                f"| {pair['left']} | {pair['right']} | {pair['at_preferred'] or 'tie'} | "
                f"{fmt(pair['absolute_D_A4_left_minus_right'])} | {pair['quality_preferred'] or 'tie'} | "
                f"{'yes' if pair['resolvable'] else 'no'} | {'yes' if pair['agreement'] else 'no'} |"
            )
        lines.append(
            f"\nR3: {r3['n_agree']}/{r3['n_resolvable']} agreement among resolvable pairs; "
            f"classification `{r3['classification']}`."
        )

    pair_lines = [
        "# R2/R3 — simulator pairs",
        "",
        f"Primary metric: `{primary_metric}`. Endpoint labels: `avg@8` and `best@8`; lower distance is better.",
        f"Support threshold is {SUPPORT_THRESHOLD:.2f}; refute threshold is {REFUTE_THRESHOLD:.2f}; minimum resolvable pairs is 10.",
    ]
    for metric in METRICS:
        add_pair_section(
            pair_lines,
            f"{metric} — avg@8",
            metric_readouts[metric]["r2"],
            metric_readouts[metric]["r3"],
        )
        add_pair_section(
            pair_lines,
            f"{metric} — best@8",
            metric_readouts[metric]["r2_best"],
            metric_readouts[metric]["r3_best"],
        )

    no_coser = tuple(candidate for candidate in CANDIDATES_ORDER if candidate != "coser")
    no_coser_avg = restrict_pair_readout(metric_readouts[METRIC_M1]["r2"], no_coser)
    no_coser_best = restrict_pair_readout(metric_readouts[METRIC_M1]["r2_best"], no_coser)
    pair_lines.extend(
        [
            "",
            "## Subset without CoSER-8B",
            "",
            "| endpoint | left | right | AT preferred | behavioral preferred | resolvable | agree |",
            "|---|---|---|---|---|:---:|:---:|",
        ]
    )
    for subset in (no_coser_avg, no_coser_best):
        for pair in subset["pairs"]:
            pair_lines.append(
                f"| {subset['endpoint']} | {pair['left']} | {pair['right']} | "
                f"{pair['at_preferred'] or 'tie'} | {pair['behavioral_preferred'] or 'tie'} | "
                f"{'yes' if pair['resolvable'] else 'no'} | {'yes' if pair['agreement'] else 'no'} |"
            )
        pair_lines.append(
            f"\nWithout CoSER ({subset['endpoint']}): {subset['n_agree']}/{subset['n_resolvable']} "
            "agreement among resolvable pairs."
        )
    (RESULTS / "readout_pairs.md").write_text("\n".join(pair_lines) + "\n")
    (RESULTS / "empty_rate.json").write_text(json.dumps(empty_rates, indent=2) + "\n")


def main() -> None:
    frame = json.loads(FRAME_PATH.read_text())
    validate_frame(frame)
    outputs = load_content_outputs(frame)
    forms = load_form_outputs(frame)
    at, row_deltas = load_at(frame)
    at_user = {
        candidate: np.asarray(
            [at["indiv"][candidate][user] for user in frame["users"]], dtype=float
        )
        for candidate in CANDIDATES_ORDER
    }
    at_rows = at_frame_rows(frame, row_deltas)

    metric_readouts: dict[str, dict] = {}
    r4_all: dict[str, dict] = {}
    for metric in METRICS:
        r1 = r1_readout(frame, outputs, metric)
        r2 = r2_readout(r1["user_delta"], at_user, endpoint="avg@8")
        r2["crossfit"] = crossfit_readout(frame, at_user, r1["user_delta"])
        r2_best = r2_readout(r1["user_delta_Dmin"], at_user, endpoint="best@8")
        r2_best["crossfit"] = crossfit_readout(frame, at_user, r1["user_delta_Dmin"])
        r3 = r3_readout(r1["user_abs_a4"], at_user, endpoint="avg@8")
        r3_best = r3_readout(r1["user_abs_a4_Dmin"], at_user, endpoint="best@8")
        metric_readouts[metric] = {
            "r1": r1,
            "r2": r2,
            "r2_best": r2_best,
            "r3": r3,
            "r3_best": r3_best,
        }
        r4_all[metric] = r4_readout(frame, at_rows, r1["row_delta"])
    form_readout = form_r5_readout(frame, forms)
    empty_rates = empty_rate_readout(frame, forms)
    primary_metric = json.loads((RESULTS / "metric_screen.json").read_text())["primary_metric"]
    if primary_metric not in METRICS:
        raise AssertionError(f"unexpected primary metric: {primary_metric}")
    write_outputs(
        frame,
        metric_readouts,
        form_readout,
        r4_all,
        primary_metric,
        empty_rates,
    )
    primary_r2 = metric_readouts[primary_metric]["r2"]
    primary_r2_best = metric_readouts[primary_metric]["r2_best"]
    print(
        json.dumps(
            {
                "primary_metric": primary_metric,
                "r2": {
                    "avg@8": {
                        "classification": primary_r2["classification"],
                        "n_resolvable": primary_r2["n_resolvable"],
                        "n_agree": primary_r2["n_agree"],
                        "kendall_tau": primary_r2["kendall_tau_point"],
                    },
                    "best@8": {
                        "classification": primary_r2_best["classification"],
                        "n_resolvable": primary_r2_best["n_resolvable"],
                        "n_agree": primary_r2_best["n_agree"],
                        "kendall_tau": primary_r2_best["kendall_tau_point"],
                    },
                },
                "outputs": [
                    str(RESULTS / "readouts.json"),
                    str(RESULTS / "readout_pairs.md"),
                    str(RESULTS / "empty_rate.json"),
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
