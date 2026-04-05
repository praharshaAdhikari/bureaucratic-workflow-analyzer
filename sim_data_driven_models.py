from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as exc:
    raise ImportError("scipy is required. Install with: pip install scipy") from exc


_DIST_MAP = {
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
    "fisk": stats.fisk,
    "lognorm": stats.lognorm,
}


@dataclass
class DurationModel:
    municipality: int | None
    activity: str
    dist_name: str
    params: list[float]
    ks_stat: float
    n_obs: int


def _fit_best_distribution(values: np.ndarray) -> tuple[str, list[float], float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if len(arr) < 8:
        raise ValueError("Not enough positive observations to fit distribution")

    best_name = "lognorm"
    best_params: list[float] = [0.6, 0.0, float(np.median(arr))]
    best_ks = float("inf")

    for name, dist in _DIST_MAP.items():
        try:
            params = dist.fit(arr, floc=0)
            ks_stat = float(stats.kstest(arr, name, args=params).statistic)
            if ks_stat < best_ks:
                best_name = name
                best_params = [float(x) for x in params]
                best_ks = ks_stat
        except Exception:
            continue

    return best_name, best_params, best_ks


def fit_duration_models(
    real_df: pd.DataFrame,
    min_obs: int = 25,
    value_col: str = "time_since_prev_hours",
) -> dict[str, Any]:
    needed = {"municipality", "activity", value_col}
    missing = needed.difference(real_df.columns)
    if missing:
        raise ValueError(f"Missing columns for duration fitting: {sorted(missing)}")

    df = real_df[["municipality", "activity", value_col]].copy()
    df["municipality"] = pd.to_numeric(df["municipality"], errors="coerce").astype("Int64")
    df["activity"] = df["activity"].astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["municipality", "activity", value_col])
    df = df[df[value_col] > 0]

    by_m: dict[int, dict[str, dict[str, Any]]] = {}
    global_model: dict[str, dict[str, Any]] = {}

    for (m, act), grp in df.groupby(["municipality", "activity"]):
        vals = grp[value_col].to_numpy(dtype=float)
        if len(vals) < min_obs:
            continue
        try:
            dist_name, params, ks = _fit_best_distribution(vals)
        except Exception:
            continue
        by_m.setdefault(int(m), {})[act] = {
            "dist": dist_name,
            "params": params,
            "ks": float(ks),
            "n_obs": int(len(vals)),
        }

    for act, grp in df.groupby("activity"):
        vals = grp[value_col].to_numpy(dtype=float)
        if len(vals) < min_obs:
            continue
        try:
            dist_name, params, ks = _fit_best_distribution(vals)
        except Exception:
            continue
        global_model[act] = {
            "dist": dist_name,
            "params": params,
            "ks": float(ks),
            "n_obs": int(len(vals)),
        }

    return {"by_municipality": by_m, "global": global_model, "value_col": value_col}


def sample_duration_hours(
    duration_model: dict[str, Any],
    municipality: int,
    activity: str,
    rng: np.random.Generator,
    fallback_hours: float = 1.0,
) -> float:
    by_m = duration_model.get("by_municipality", {})
    global_model = duration_model.get("global", {})
    spec = by_m.get(int(municipality), {}).get(activity)
    if spec is None:
        spec = global_model.get(activity)
    if spec is None:
        return float(fallback_hours)

    dist_name = str(spec.get("dist", "lognorm"))
    params = spec.get("params", None)
    dist = _DIST_MAP.get(dist_name)
    if dist is None or not isinstance(params, list):
        return float(fallback_hours)

    try:
        draw = float(dist.rvs(*params, random_state=rng))
    except Exception:
        return float(fallback_hours)

    if not np.isfinite(draw) or draw <= 0:
        return float(fallback_hours)
    return max(draw, 1e-4)


def fit_loop_model(
    real_df: pd.DataFrame,
    max_visit_bucket: int = 3,
    smoothing: float = 2.0,
) -> dict[str, Any]:
    needed = {"municipality", "case_id", "step_index", "activity", "next_activity"}
    missing = needed.difference(real_df.columns)
    if missing:
        raise ValueError(f"Missing columns for loop model: {sorted(missing)}")

    df = real_df[["municipality", "case_id", "step_index", "activity", "next_activity"]].copy()
    df["municipality"] = pd.to_numeric(df["municipality"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["municipality", "case_id", "step_index", "activity"])
    df = df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)

    rows: list[tuple[int, int, int]] = []
    for (m, case_id), grp in df.groupby(["municipality", "case_id"], sort=False):
        seen: dict[str, int] = {}
        recs = grp[["activity", "next_activity"]].to_dict("records")
        for rec in recs:
            act = str(rec["activity"])
            nxt = rec.get("next_activity")
            if nxt is not None and str(nxt) not in {"nan", "None", ""}:
                nxt_s = str(nxt)
                visits = int(seen.get(nxt_s, 0))
                bucket = min(visits, max_visit_bucket)
                is_loop = 1 if visits > 0 else 0
                rows.append((int(m), int(bucket), int(is_loop)))
            seen[act] = seen.get(act, 0) + 1

    if not rows:
        return {"by_municipality": {}, "global": {"0": 0.1}, "max_visit_bucket": int(max_visit_bucket)}

    loop_df = pd.DataFrame(rows, columns=["municipality", "visit_bucket", "is_loop"])

    def _aggregate(frame: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for b in range(0, max_visit_bucket + 1):
            sub = frame[frame["visit_bucket"] == b]
            n = float(len(sub))
            k = float(sub["is_loop"].sum())
            p = (k + smoothing) / (n + 2.0 * smoothing) if n > 0 else 0.5
            out[str(b)] = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return out

    by_m: dict[int, dict[str, float]] = {}
    for m, grp in loop_df.groupby("municipality"):
        by_m[int(m)] = _aggregate(grp)

    global_probs = _aggregate(loop_df)
    return {
        "by_municipality": by_m,
        "global": global_probs,
        "max_visit_bucket": int(max_visit_bucket),
        "smoothing": float(smoothing),
    }


def apply_loop_model_to_options(
    options: list[tuple[str, float]],
    seen_counts: dict[str, int],
    municipality: int,
    loop_model: dict[str, Any],
) -> list[tuple[str, float]]:
    if not options:
        return options

    max_b = int(loop_model.get("max_visit_bucket", 3))
    probs = loop_model.get("by_municipality", {}).get(int(municipality), loop_model.get("global", {}))
    p_unseen = float(probs.get("0", 0.5))
    non_loop_scale = max(1e-6, 1.0 - p_unseen)

    weighted: list[tuple[str, float]] = []
    for tgt, prob in options:
        seen = int(seen_counts.get(tgt, 0))
        if seen <= 0:
            factor = non_loop_scale
        else:
            bucket = min(seen, max_b)
            factor = float(probs.get(str(bucket), probs.get(str(max_b), p_unseen)))
        weighted.append((tgt, float(prob) * max(factor, 1e-6)))

    total = sum(p for _, p in weighted)
    if total <= 0:
        return options
    return [(tgt, p / total) for tgt, p in weighted]


def fit_stop_model(
    real_df: pd.DataFrame,
    progress_bins: tuple[float, ...] = (0.0, 0.4, 0.7, 0.9, 1.1, 10.0),
    max_rework_bucket: int = 3,
    smoothing: float = 3.0,
) -> dict[str, Any]:
    needed = {"municipality", "case_id", "step_index", "activity"}
    missing = needed.difference(real_df.columns)
    if missing:
        raise ValueError(f"Missing columns for stop model: {sorted(missing)}")

    df = real_df[["municipality", "case_id", "step_index", "activity"]].copy()
    if "rework_count_activity" in real_df.columns:
        df["rework_count_activity"] = pd.to_numeric(real_df["rework_count_activity"], errors="coerce").fillna(0)
    else:
        df["rework_count_activity"] = 0.0

    df["municipality"] = pd.to_numeric(df["municipality"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["municipality", "case_id", "step_index", "activity"]) 
    df = df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)

    rows: list[tuple[int, int, int, int]] = []
    bins = np.asarray(progress_bins, dtype=float)
    for (m, case_id), grp in df.groupby(["municipality", "case_id"], sort=False):
        g = grp.reset_index(drop=True)
        trace_len = int(g["step_index"].max()) + 1
        for i, rec in g.iterrows():
            step = int(rec["step_index"])
            progress = (step + 1) / max(trace_len, 1)
            p_bin = int(np.clip(np.digitize(progress, bins, right=False) - 1, 0, len(bins) - 2))
            r_bin = int(np.clip(int(rec["rework_count_activity"]), 0, max_rework_bucket))
            is_terminal = 1 if i == (len(g) - 1) else 0
            rows.append((int(m), p_bin, r_bin, is_terminal))

    if not rows:
        return {
            "by_municipality": {},
            "global": {"0|0": 0.2},
            "progress_bins": list(progress_bins),
            "max_rework_bucket": int(max_rework_bucket),
        }

    stop_df = pd.DataFrame(rows, columns=["municipality", "progress_bin", "rework_bin", "is_terminal"])

    def _aggregate(frame: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        p_bins = sorted(frame["progress_bin"].unique().tolist())
        for pb in p_bins:
            for rb in range(0, max_rework_bucket + 1):
                sub = frame[(frame["progress_bin"] == pb) & (frame["rework_bin"] == rb)]
                n = float(len(sub))
                k = float(sub["is_terminal"].sum())
                p = (k + smoothing) / (n + 2.0 * smoothing) if n > 0 else 0.5
                out[f"{int(pb)}|{int(rb)}"] = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return out

    by_m: dict[int, dict[str, float]] = {}
    for m, grp in stop_df.groupby("municipality"):
        by_m[int(m)] = _aggregate(grp)

    global_probs = _aggregate(stop_df)
    return {
        "by_municipality": by_m,
        "global": global_probs,
        "progress_bins": [float(x) for x in progress_bins],
        "max_rework_bucket": int(max_rework_bucket),
        "smoothing": float(smoothing),
    }


def lookup_stop_probability(
    municipality: int,
    progress_ratio: float,
    rework_count: int,
    stop_model: dict[str, Any],
) -> float:
    bins = np.asarray(stop_model.get("progress_bins", [0.0, 0.5, 0.8, 1.0, 10.0]), dtype=float)
    p_bin = int(np.clip(np.digitize(float(progress_ratio), bins, right=False) - 1, 0, len(bins) - 2))
    r_max = int(stop_model.get("max_rework_bucket", 3))
    r_bin = int(np.clip(int(rework_count), 0, r_max))

    key = f"{p_bin}|{r_bin}"
    local = stop_model.get("by_municipality", {}).get(int(municipality), {})
    if key in local:
        return float(local[key])

    global_probs = stop_model.get("global", {})
    return float(global_probs.get(key, 0.2))


def fit_all_data_driven_models(
    real_df: pd.DataFrame,
    duration_min_obs: int = 25,
    loop_smoothing: float = 2.0,
    stop_smoothing: float = 3.0,
) -> dict[str, Any]:
    duration_model = fit_duration_models(real_df, min_obs=duration_min_obs)
    loop_model = fit_loop_model(real_df, smoothing=loop_smoothing)
    stop_model = fit_stop_model(real_df, smoothing=stop_smoothing)
    return {
        "duration_model": duration_model,
        "loop_model": loop_model,
        "stop_model": stop_model,
        "meta": {
            "duration_min_obs": int(duration_min_obs),
            "loop_smoothing": float(loop_smoothing),
            "stop_smoothing": float(stop_smoothing),
        },
    }