from __future__ import annotations

import concurrent.futures
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import simpy

from sim_data_driven_models import (
    fit_all_data_driven_models,
    lookup_loop_probability,
    lookup_stop_probability,
    sample_duration_hours,
)


OUTPUT_DIR = Path("./output")
REAL_FEATURES_PARQUET = OUTPUT_DIR / "case_step_features.parquet"
REAL_FEATURES_CSV = OUTPUT_DIR / "case_step_features.csv"

SEARCH_REPORT_PATH = OUTPUT_DIR / "sim_data_driven_calibration_search.csv"
BEST_CALIB_PATH = OUTPUT_DIR / "sim_data_driven_calibration_best.json"
RESOURCE_CALIB_PATH = OUTPUT_DIR / "resource_calibration.json"

METRICS_BY_RUN_PATH = OUTPUT_DIR / "sim_validation_metrics_by_run.csv"
BASELINE_METRICS_BY_RUN_PATH = OUTPUT_DIR / "sim_validation_metrics_baseline_by_run.csv"
REPORT_PATH = OUTPUT_DIR / "sim_validation_report.csv"
COMPARISON_REPORT_PATH = OUTPUT_DIR / "sim_validation_comparison.csv"

METRIC_COLUMNS = [
    "trace_length_dist_l1",
    "top20_activity_rel_error_mean",
    "transition_matrix_l1",
    "duration_median_rel_error",
    "loop_rel_error",
]

DEFAULT_OBJECTIVE_WEIGHTS = {
    "trace_length_dist_l1": 0.20,
    "top20_activity_rel_error_mean": 0.20,
    "transition_matrix_l1": 0.20,
    "duration_median_rel_error": 0.20,
    "loop_rel_error": 0.20,
}

DEFAULT_THRESHOLDS = {
    "trace_length_dist_l1": 0.60,
    "top20_activity_rel_error_mean": 0.20,
    "transition_matrix_l1": 0.80,
    "duration_median_rel_error": 0.25,
    "loop_rel_error": 0.20,
}


_TRAIN_DF: pd.DataFrame | None = None
_HOLDOUT_DF: pd.DataFrame | None = None
_EVAL_CFG: dict[str, Any] | None = None


def relative_error(real_value: float, est_value: float) -> float:
    denom = max(abs(float(real_value)), 1e-9)
    return abs(float(est_value) - float(real_value)) / denom


def distribution_l1_distance(real_series: pd.Series, est_series: pd.Series) -> float:
    r = real_series.value_counts(normalize=True)
    e = est_series.value_counts(normalize=True)
    idx = sorted(set(r.index).union(set(e.index)))
    rv = np.array([float(r.get(i, 0.0)) for i in idx])
    ev = np.array([float(e.get(i, 0.0)) for i in idx])
    return float(np.abs(rv - ev).sum())


def _relative_error_series(real_dist: pd.Series, sim_dist: pd.Series, keys: list[str]) -> float:
    errs: list[float] = []
    for key in keys:
        rv = float(real_dist.get(key, 0.0))
        sv = float(sim_dist.get(key, 0.0))
        errs.append(relative_error(rv, sv))
    return float(np.mean(errs)) if errs else 0.0


def _edge_distribution_l1(real_edges: pd.Series, sim_edges: pd.Series) -> float:
    return distribution_l1_distance(real_edges.astype(str), sim_edges.astype(str))


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in METRIC_COLUMNS:
                merged[k] = float(v)

    total = float(sum(max(0.0, merged[k]) for k in METRIC_COLUMNS))
    if total <= 0:
        return dict(DEFAULT_OBJECTIVE_WEIGHTS)
    return {k: float(max(0.0, merged[k]) / total) for k in METRIC_COLUMNS}


def compute_objective(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    w = _normalize_weights(weights)
    return float(sum(w[m] * float(metrics.get(m, 1.0)) for m in METRIC_COLUMNS))


def load_real_features() -> pd.DataFrame:
    if REAL_FEATURES_PARQUET.exists():
        df = pd.read_parquet(REAL_FEATURES_PARQUET)
    elif REAL_FEATURES_CSV.exists():
        df = pd.read_csv(REAL_FEATURES_CSV)
    else:
        raise FileNotFoundError("Missing real case features in ./output")

    req = {"municipality", "case_id", "step_index", "activity", "time_since_prev_hours"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["municipality"] = pd.to_numeric(df["municipality"], errors="coerce").astype("Int64")
    df["step_index"] = pd.to_numeric(df["step_index"], errors="coerce").fillna(0).astype(int)
    df["activity"] = df["activity"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df["time_since_prev_hours"] = pd.to_numeric(df["time_since_prev_hours"], errors="coerce").fillna(0.0)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    if "next_activity" not in df.columns:
        df = df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)
        df["next_activity"] = df.groupby(["municipality", "case_id"])["activity"].shift(-1)

    if "rework_count_activity" not in df.columns:
        df["rework_count_activity"] = 0

    return df.dropna(subset=["municipality"]).reset_index(drop=True)


def split_train_holdout(df: pd.DataFrame, holdout_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_tbl = df[["municipality", "case_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    mask = rng.random(len(case_tbl)) < float(holdout_frac)
    holdout_cases = case_tbl[mask]
    if holdout_cases.empty:
        holdout_cases = case_tbl.sample(n=max(1, int(len(case_tbl) * holdout_frac)), random_state=seed)

    key_holdout = set(zip(holdout_cases["municipality"].astype(str), holdout_cases["case_id"].astype(str)))
    is_holdout = df.apply(lambda r: (str(r["municipality"]), str(r["case_id"])) in key_holdout, axis=1)
    holdout_df = df[is_holdout].copy().reset_index(drop=True)
    train_df = df[~is_holdout].copy().reset_index(drop=True)
    return train_df, holdout_df


def load_resource_capacity_by_m(df: pd.DataFrame) -> dict[int, int]:
    if RESOURCE_CALIB_PATH.exists():
        try:
            with open(RESOURCE_CALIB_PATH, "r") as f:
                payload = json.load(f)
            by_m = payload.get("by_municipality", {})
            out: dict[int, int] = {}
            for m_str, row in by_m.items():
                try:
                    m = int(m_str)
                except Exception:
                    continue
                base = row.get("initial_workers", row.get("comfortable_workers", row.get("min_workers", 1)))
                out[m] = max(1, int(round(float(base))))
            if out:
                return out
        except Exception:
            pass

    case_counts = (
        df[["municipality", "case_id"]]
        .drop_duplicates()
        .groupby("municipality")
        .size()
        .to_dict()
    )
    fallback: dict[int, int] = {}
    for m, n_cases in case_counts.items():
        fallback[int(m)] = max(1, int(round(np.sqrt(max(float(n_cases), 1.0)))))
    return fallback


def _estimate_arrival_means_hours(df: pd.DataFrame) -> dict[int, float]:
    work = df.sort_values(["municipality", "case_id", "step_index"]).copy()

    by_m: dict[int, float] = {}
    has_ts = "timestamp" in work.columns and pd.api.types.is_datetime64_any_dtype(work["timestamp"])

    for m, grp in work.groupby("municipality"):
        municipality = int(m)

        if has_ts:
            starts = (
                grp.dropna(subset=["timestamp"])
                .groupby("case_id", as_index=False)["timestamp"]
                .min()
                .sort_values("timestamp")
            )
            if len(starts) >= 2:
                deltas = starts["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
                deltas = deltas[deltas > 0]
                if len(deltas) > 0:
                    by_m[municipality] = float(np.clip(deltas.median(), 0.10, 72.0))
                    continue

        case_duration = (
            grp.groupby("case_id")["time_since_prev_hours"]
            .sum()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        proxy = float(case_duration.median()) if len(case_duration) else 8.0
        by_m[municipality] = float(np.clip(max(proxy * 0.35, 0.25), 0.10, 72.0))

    return by_m


def _build_transition_and_start_models(df: pd.DataFrame) -> tuple[
    dict[int, dict[str, list[tuple[str, float]]]],
    dict[int, list[tuple[str, float]]],
    dict[int, float],
    int,
]:
    h = df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)

    trans = (
        h[h["next_activity"].notna()]
        .groupby(["municipality", "activity", "next_activity"])  # type: ignore[arg-type]
        .size()
        .reset_index(name="n")
    )
    trans["p"] = trans["n"] / trans.groupby(["municipality", "activity"])["n"].transform("sum")

    transition_by_m: dict[int, dict[str, list[tuple[str, float]]]] = {}
    for (m, src), grp in trans.groupby(["municipality", "activity"]):
        transition_by_m.setdefault(int(m), {})[str(src)] = [
            (str(r["next_activity"]), float(r["p"])) for _, r in grp.iterrows()
        ]

    first_events = (
        h.groupby(["municipality", "case_id"], as_index=False)
        .first()[["municipality", "activity"]]
    )

    start_dist: dict[int, list[tuple[str, float]]] = {}
    for m, grp in first_events.groupby("municipality"):
        vc = grp["activity"].value_counts(normalize=True)
        start_dist[int(m)] = [(str(k), float(v)) for k, v in vc.items()]

    expected_len_by_m = (
        h.groupby(["municipality", "case_id"])["step_index"].max().add(1)
        .groupby("municipality")
        .mean()
        .to_dict()
    )

    max_steps = int(max(8, h.groupby(["municipality", "case_id"])["step_index"].max().quantile(0.95) + 1))

    return transition_by_m, start_dist, {int(k): float(v) for k, v in expected_len_by_m.items()}, max_steps


def _sample_pairs(rng: np.random.Generator, pairs: list[tuple[str, float]]) -> str | None:
    if not pairs:
        return None

    vals = [v for v, _ in pairs]
    probs = np.asarray([max(float(p), 0.0) for _, p in pairs], dtype=float)
    if probs.sum() <= 0:
        return None
    probs = probs / probs.sum()
    return str(vals[int(rng.choice(len(vals), p=probs))])


def _choose_next_with_loop_gate(
    rng: np.random.Generator,
    options: list[tuple[str, float]],
    seen_counts: dict[str, int],
    municipality: int,
    loop_model: dict[str, Any],
) -> str | None:
    if not options:
        return None

    chosen = _sample_pairs(rng, options)
    if chosen is None:
        return None

    seen = int(seen_counts.get(chosen, 0))
    if seen <= 0:
        return chosen

    p_loop = lookup_loop_probability(
        municipality=municipality,
        seen_count=seen,
        loop_model=loop_model,
    )
    if float(rng.random()) < float(np.clip(p_loop, 0.0, 1.0)):
        return chosen

    unseen_options = [(tgt, p) for tgt, p in options if int(seen_counts.get(tgt, 0)) <= 0]
    if unseen_options:
        alt = _sample_pairs(rng, unseen_options)
        if alt is not None:
            return alt
    return chosen


def _simulate_holdout_once(
    models: dict[str, Any],
    holdout_df: pd.DataFrame,
    seed: int,
    n_cases_per_municipality: int,
    capacity_by_m: dict[int, int],
    capacity_multiplier: float = 1.0,
    arrival_mean_by_m: dict[int, float] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    h = holdout_df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)

    transition_by_m, start_dist, expected_len_by_m, max_steps = _build_transition_and_start_models(h)
    municipalities = sorted(set(int(x) for x in h["municipality"].dropna().unique()))
    arrival_by_m = arrival_mean_by_m or _estimate_arrival_means_hours(h)

    sim_trace_rows: list[dict[str, Any]] = []

    def _simulate_case(env: simpy.Environment, municipality: int, case_id: str, resource_pool: simpy.Resource) -> Any:
        start = _sample_pairs(rng, start_dist.get(municipality, []))
        if start is None:
            return

        current = start
        seen: dict[str, int] = {}
        steps = 0

        while steps < max_steps:
            seen[current] = int(seen.get(current, 0)) + 1
            options = transition_by_m.get(municipality, {}).get(current, [])
            nxt = _choose_next_with_loop_gate(
                rng=rng,
                options=options,
                seen_counts=seen,
                municipality=municipality,
                loop_model=models["loop_model"],
            )

            progress_ratio = (steps + 1) / max(float(expected_len_by_m.get(municipality, max_steps)), 1.0)
            rework_count = max(seen[current] - 1, 0)
            stop_prob = lookup_stop_probability(
                municipality=municipality,
                progress_ratio=progress_ratio,
                rework_count=rework_count,
                stop_model=models["stop_model"],
            )
            if nxt is not None and float(rng.random()) < float(np.clip(stop_prob, 0.0, 0.98)):
                nxt = None

            dur = sample_duration_hours(
                duration_model=models["duration_model"],
                municipality=municipality,
                activity=current,
                rng=rng,
                fallback_hours=1.0,
            )
            dur = max(float(dur), 1e-4)

            req_start = float(env.now)
            with resource_pool.request() as req:
                yield req
                queue_wait = max(float(env.now) - req_start, 0.0)
                yield env.timeout(dur)

            total_step = queue_wait + dur
            sim_trace_rows.append(
                {
                    "municipality": int(municipality),
                    "case_id": str(case_id),
                    "activity": str(current),
                    "next_activity": None if nxt is None else str(nxt),
                    "step_index": int(steps),
                    "time_since_prev_hours": float(total_step if steps > 0 else 0.0),
                    "queue_wait_hours": float(queue_wait),
                }
            )

            steps += 1
            if nxt is None:
                break
            current = str(nxt)

    def _arrival(env: simpy.Environment, municipality: int, n_cases: int, resource_pool: simpy.Resource) -> Any:
        mean_ia = float(np.clip(arrival_by_m.get(int(municipality), 1.0), 0.05, 720.0))
        for i in range(n_cases):
            cid = f"H{municipality}_SIM_{seed}_{i:05d}"
            env.process(_simulate_case(env, municipality, cid, resource_pool))
            yield env.timeout(float(rng.exponential(scale=mean_ia)))

    cap_mult = max(float(capacity_multiplier), 0.1)
    for municipality in municipalities:
        env = simpy.Environment()
        base_capacity = int(capacity_by_m.get(int(municipality), 1))
        scaled_capacity = max(1, int(round(float(base_capacity) * cap_mult)))
        resource_pool = simpy.Resource(env, capacity=scaled_capacity)
        env.process(
            _arrival(
                env,
                municipality,
                n_cases=int(max(1, n_cases_per_municipality)),
                resource_pool=resource_pool,
            )
        )
        env.run()

    return pd.DataFrame(sim_trace_rows)


def _compute_metrics(real_df: pd.DataFrame, sim_trace_df: pd.DataFrame) -> dict[str, float]:
    if sim_trace_df.empty:
        return {k: 1.0 for k in METRIC_COLUMNS}

    h = real_df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)
    s = sim_trace_df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)

    real_case_len = h.groupby(["municipality", "case_id"])["step_index"].max().add(1)
    sim_case_len = s.groupby(["municipality", "case_id"])["step_index"].max().add(1)
    trace_length_dist_l1 = distribution_l1_distance(real_case_len.astype(int), sim_case_len.astype(int))

    real_act_dist = h["activity"].astype(str).value_counts(normalize=True)
    sim_act_dist = s["activity"].astype(str).value_counts(normalize=True)
    top20 = real_act_dist.head(20).index.tolist()
    top20_activity_rel_error_mean = _relative_error_series(real_act_dist, sim_act_dist, [str(x) for x in top20])

    real_edges = h[h["next_activity"].notna()].apply(
        lambda r: f"{str(r['activity'])}=>{str(r['next_activity'])}", axis=1
    )
    sim_edges = s[s["next_activity"].notna()].apply(
        lambda r: f"{str(r['activity'])}=>{str(r['next_activity'])}", axis=1
    )
    transition_matrix_l1 = _edge_distribution_l1(real_edges, sim_edges)

    real_case_duration = h.groupby(["municipality", "case_id"])["time_since_prev_hours"].sum()
    sim_case_duration = s.groupby(["municipality", "case_id"])["time_since_prev_hours"].sum()
    duration_median_rel_error = relative_error(
        float(real_case_duration.median()) if len(real_case_duration) else 0.0,
        float(sim_case_duration.median()) if len(sim_case_duration) else 0.0,
    )

    def _case_loops(df_case: pd.DataFrame) -> int:
        vc = df_case["activity"].value_counts()
        return int((vc > 1).sum())

    real_loop_mean = float(
        h.groupby(["municipality", "case_id"], group_keys=False).apply(_case_loops, include_groups=False).mean()
    )
    sim_loop_mean = float(
        s.groupby(["municipality", "case_id"], group_keys=False).apply(_case_loops, include_groups=False).mean()
    )
    loop_rel_error = relative_error(real_loop_mean, sim_loop_mean)

    return {
        "trace_length_dist_l1": float(trace_length_dist_l1),
        "top20_activity_rel_error_mean": float(top20_activity_rel_error_mean),
        "transition_matrix_l1": float(transition_matrix_l1),
        "duration_median_rel_error": float(duration_median_rel_error),
        "loop_rel_error": float(loop_rel_error),
    }


def evaluate_candidate_simpy_holdout(
    models: dict,
    holdout_df: pd.DataFrame,
    seed: int = 123,
    n_cases_per_municipality: int = 30,
    capacity_by_m: dict[int, int] | None = None,
    capacity_multiplier: float = 1.0,
    arrival_mean_by_m: dict[int, float] | None = None,
) -> dict[str, float]:
    h = holdout_df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)
    if len(h) == 0:
        return {k: 1.0 for k in METRIC_COLUMNS}

    cap_map = capacity_by_m or load_resource_capacity_by_m(holdout_df)
    sim_trace_df = _simulate_holdout_once(
        models=models,
        holdout_df=h,
        seed=int(seed),
        n_cases_per_municipality=int(max(1, n_cases_per_municipality)),
        capacity_by_m=cap_map,
        capacity_multiplier=float(capacity_multiplier),
        arrival_mean_by_m=arrival_mean_by_m,
    )
    return _compute_metrics(h, sim_trace_df)


def evaluate_candidate(
    models: dict,
    holdout_df: pd.DataFrame,
    seed: int = 123,
    max_cases: int | None = None,
    n_cases_per_municipality: int = 30,
    seeds: list[int] | None = None,
    capacity_by_m: dict[int, int] | None = None,
    capacity_multiplier: float = 1.0,
    objective_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    h = holdout_df.sort_values(["municipality", "case_id", "step_index"]).reset_index(drop=True)

    if max_cases is not None and int(max_cases) > 0:
        case_tbl = h[["municipality", "case_id"]].drop_duplicates().reset_index(drop=True)
        max_cases_int = int(max_cases)
        if len(case_tbl) > max_cases_int:
            idx = rng.choice(len(case_tbl), size=max_cases_int, replace=False)
            sampled = case_tbl.iloc[np.sort(idx)]
            sampled_keys = set(zip(sampled["municipality"].astype(str), sampled["case_id"].astype(str)))
            keep_mask = h.apply(lambda r: (str(r["municipality"]), str(r["case_id"])) in sampled_keys, axis=1)
            h = h[keep_mask].copy().reset_index(drop=True)

    if len(h) == 0:
        out = {k: 1.0 for k in METRIC_COLUMNS}
        out["objective"] = compute_objective(out, objective_weights)
        return out

    seed_list = list(seeds) if seeds else [int(seed), int(seed) + 17, int(seed) + 37]
    cap_map = capacity_by_m or load_resource_capacity_by_m(h)
    arrival_map = _estimate_arrival_means_hours(h)

    rows: list[dict[str, float]] = []
    for s in seed_list:
        m = evaluate_candidate_simpy_holdout(
            models=models,
            holdout_df=h,
            seed=int(s),
            n_cases_per_municipality=int(max(1, n_cases_per_municipality)),
            capacity_by_m=cap_map,
            capacity_multiplier=float(capacity_multiplier),
            arrival_mean_by_m=arrival_map,
        )
        rows.append(m)

    by_seed_df = pd.DataFrame(rows)
    out = {metric: float(by_seed_df[metric].mean()) for metric in METRIC_COLUMNS}
    out["objective"] = compute_objective(out, objective_weights)
    return out


def _mark_pareto_front(df: pd.DataFrame, metric_cols: list[str]) -> pd.Series:
    vals = df[metric_cols].to_numpy(dtype=float)
    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        dominates_i = np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1)
        if np.any(dominates_i):
            is_pareto[i] = False
            continue
        dominated_by_i = np.all(vals[i] <= vals, axis=1) & np.any(vals[i] < vals, axis=1)
        is_pareto[dominated_by_i] = False
        is_pareto[i] = True

    return pd.Series(is_pareto, index=df.index)


def _run_single_candidate(d_min: int, loop_sm: float, stop_sm: float) -> tuple[dict[str, float], dict[str, Any]]:
    if _TRAIN_DF is None or _HOLDOUT_DF is None or _EVAL_CFG is None:
        raise RuntimeError("Worker dataset state is not initialized.")

    models = fit_all_data_driven_models(
        _TRAIN_DF,
        duration_min_obs=int(d_min),
        loop_smoothing=float(loop_sm),
        stop_smoothing=float(stop_sm),
    )

    metrics = evaluate_candidate(
        models=models,
        holdout_df=_HOLDOUT_DF,
        seed=int(_EVAL_CFG.get("seed", 123)),
        max_cases=_EVAL_CFG.get("max_cases"),
        n_cases_per_municipality=int(_EVAL_CFG.get("n_cases_per_municipality", 30)),
        seeds=list(_EVAL_CFG.get("seeds", [123, 140, 160])),
        capacity_by_m=dict(_EVAL_CFG.get("capacity_by_m", {})),
        capacity_multiplier=float(_EVAL_CFG.get("capacity_multiplier", 1.0)),
        objective_weights=dict(_EVAL_CFG.get("objective_weights", DEFAULT_OBJECTIVE_WEIGHTS)),
    )

    row = {
        "duration_min_obs": int(d_min),
        "loop_smoothing": float(loop_sm),
        "stop_smoothing": float(stop_sm),
        **{k: float(metrics[k]) for k in METRIC_COLUMNS},
        "objective": float(metrics["objective"]),
    }
    return row, models


def _init_worker(train_df: pd.DataFrame, holdout_df: pd.DataFrame, eval_cfg: dict[str, Any]) -> None:
    global _TRAIN_DF, _HOLDOUT_DF, _EVAL_CFG
    _TRAIN_DF = train_df
    _HOLDOUT_DF = holdout_df
    _EVAL_CFG = dict(eval_cfg)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is not None:
        return max(1, int(n_jobs))

    env_n_jobs = os.getenv("CALIB_N_JOBS", "").strip()
    if env_n_jobs:
        try:
            return max(1, int(env_n_jobs))
        except ValueError:
            pass

    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def _resolve_fast_mode(fast_mode: bool | None) -> bool:
    if fast_mode is not None:
        return bool(fast_mode)
    return os.getenv("CALIB_FAST", "0").strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_search_mode() -> str:
    mode = os.getenv("CALIB_SEARCH_MODE", "single").strip().lower()
    return mode if mode in {"single", "pareto"} else "single"


def _resolve_capacity_mode() -> str:
    mode = os.getenv("CALIB_CAPACITY_MODE", "grid").strip().lower()
    return mode if mode in {"grid", "mmc"} else "grid"


def _resolve_weights_from_env() -> dict[str, float]:
    raw = os.getenv("CALIB_METRIC_WEIGHTS", "").strip()
    if not raw:
        return dict(DEFAULT_OBJECTIVE_WEIGHTS)

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            parsed: dict[str, float] = {}
            for k, v in payload.items():
                if k in METRIC_COLUMNS:
                    parsed[k] = float(v)
            if parsed:
                return _normalize_weights(parsed)
    except Exception:
        pass

    return dict(DEFAULT_OBJECTIVE_WEIGHTS)


def _factorial_ratio_stable(a: int, b: int) -> float:
    if a < b:
        return 0.0
    return math.exp(math.lgamma(a + 1) - math.lgamma(b + 1))


def _erlang_c_probability_wait(lmbd: float, mu: float, c: int) -> float:
    if c <= 0 or lmbd <= 0 or mu <= 0:
        return 0.0

    rho = lmbd / (c * mu)
    if rho >= 1.0:
        return 1.0

    a = lmbd / mu
    sum_terms = 0.0
    for n in range(c):
        sum_terms += (a ** n) / _factorial_ratio_stable(n, 0)

    top = (a ** c) / (_factorial_ratio_stable(c, 0) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + top)
    p_wait = top * p0
    return float(np.clip(p_wait, 0.0, 1.0))


def _mmc_wait_p95_hours(lmbd: float, mu: float, c: int) -> float:
    if lmbd <= 0 or mu <= 0 or c <= 0:
        return 0.0
    if lmbd >= c * mu:
        return float("inf")

    p_wait = _erlang_c_probability_wait(lmbd, mu, c)
    if p_wait <= 0.05:
        return 0.0

    rate = c * mu - lmbd
    if rate <= 0:
        return float("inf")

    return float(-math.log(0.05 / p_wait) / rate)


def estimate_capacity_by_m_mmc(
    train_df: pd.DataFrame,
    target_wait_p95_hours: float = 24.0,
    max_servers: int = 80,
) -> dict[int, int]:
    if target_wait_p95_hours <= 0:
        raise ValueError("target_wait_p95_hours must be positive")

    arrival_mean_by_m = _estimate_arrival_means_hours(train_df)
    out: dict[int, int] = {}

    grouped = train_df.sort_values(["municipality", "case_id", "step_index"]).groupby("municipality")
    for m, grp in grouped:
        municipality = int(m)

        case_steps = grp.groupby("case_id")["step_index"].max().add(1)
        mean_steps = float(case_steps.mean()) if len(case_steps) else 1.0

        step_service = pd.to_numeric(grp["time_since_prev_hours"], errors="coerce")
        step_service = step_service[step_service > 0]
        mean_service_h = float(step_service.mean()) if len(step_service) else 1.0

        lambda_case = 1.0 / max(float(arrival_mean_by_m.get(municipality, 1.0)), 1e-6)
        lambda_step = lambda_case * max(mean_steps, 1e-6)
        mu_step = 1.0 / max(mean_service_h, 1e-6)

        chosen = max(1, int(round(np.sqrt(max(len(case_steps), 1)))))
        for c in range(1, int(max_servers) + 1):
            wait95 = _mmc_wait_p95_hours(lambda_step, mu_step, c)
            if np.isfinite(wait95) and wait95 <= float(target_wait_p95_hours):
                chosen = c
                break

        out[municipality] = int(max(1, chosen))

    return out


def _aggregate_metric_rows(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _build_metric_summary(metric_df: pd.DataFrame, label: str) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame(columns=["metric", "n_runs", "error_mean", "error_std", "ci95", "label"])

    rows = []
    for metric in METRIC_COLUMNS:
        vals = pd.to_numeric(metric_df[metric], errors="coerce").dropna()
        mean_error = float(vals.mean()) if len(vals) else np.nan
        std_error = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        ci95 = float(1.96 * std_error / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(
            {
                "metric": metric,
                "n_runs": int(len(vals)),
                "error_mean": mean_error,
                "error_std": std_error,
                "ci95": ci95,
                "error_ci95_low": mean_error - ci95 if np.isfinite(mean_error) else np.nan,
                "error_ci95_high": mean_error + ci95 if np.isfinite(mean_error) else np.nan,
                "label": str(label),
            }
        )
    return pd.DataFrame(rows)


def _run_multi_seed_validation(
    models: dict[str, Any],
    holdout_df: pd.DataFrame,
    n_runs: int,
    n_cases_per_municipality: int,
    capacity_by_m: dict[int, int],
    capacity_multiplier: float,
    base_seed: int,
) -> pd.DataFrame:
    arrival_mean_by_m = _estimate_arrival_means_hours(holdout_df)
    rows: list[dict[str, float]] = []
    for run_id in range(1, int(n_runs) + 1):
        seed = int(base_seed + run_id * 997)
        metrics = evaluate_candidate_simpy_holdout(
            models=models,
            holdout_df=holdout_df,
            seed=seed,
            n_cases_per_municipality=n_cases_per_municipality,
            capacity_by_m=capacity_by_m,
            capacity_multiplier=capacity_multiplier,
            arrival_mean_by_m=arrival_mean_by_m,
        )
        rows.append({"run_id": int(run_id), **metrics})

    return _aggregate_metric_rows(rows)


def compare_default_vs_best(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    best_config: dict[str, Any],
    n_runs: int = 20,
    n_cases_per_municipality: int = 120,
    base_seed: int = 1337,
) -> dict[str, pd.DataFrame]:
    base_capacity_by_m = load_resource_capacity_by_m(train_df)

    default_config = {
        "duration_min_obs": 20,
        "loop_smoothing": 2.0,
        "stop_smoothing": 3.0,
        "capacity_multiplier": 1.0,
    }

    default_models = fit_all_data_driven_models(
        train_df,
        duration_min_obs=int(default_config["duration_min_obs"]),
        loop_smoothing=float(default_config["loop_smoothing"]),
        stop_smoothing=float(default_config["stop_smoothing"]),
    )

    best_models = fit_all_data_driven_models(
        train_df,
        duration_min_obs=int(best_config.get("duration_min_obs", 20)),
        loop_smoothing=float(best_config.get("loop_smoothing", 2.0)),
        stop_smoothing=float(best_config.get("stop_smoothing", 3.0)),
    )

    if "capacity_by_m_direct" in best_config and isinstance(best_config["capacity_by_m_direct"], dict):
        best_capacity_by_m = {int(k): int(v) for k, v in best_config["capacity_by_m_direct"].items()}
        best_capacity_multiplier = 1.0
    else:
        best_capacity_by_m = dict(base_capacity_by_m)
        best_capacity_multiplier = float(best_config.get("capacity_multiplier", 1.0))

    baseline_by_run = _run_multi_seed_validation(
        models=default_models,
        holdout_df=holdout_df,
        n_runs=n_runs,
        n_cases_per_municipality=n_cases_per_municipality,
        capacity_by_m=base_capacity_by_m,
        capacity_multiplier=float(default_config["capacity_multiplier"]),
        base_seed=base_seed,
    )

    best_by_run = _run_multi_seed_validation(
        models=best_models,
        holdout_df=holdout_df,
        n_runs=n_runs,
        n_cases_per_municipality=n_cases_per_municipality,
        capacity_by_m=best_capacity_by_m,
        capacity_multiplier=float(best_capacity_multiplier),
        base_seed=base_seed,
    )

    baseline_summary = _build_metric_summary(baseline_by_run, label="baseline")
    best_summary = _build_metric_summary(best_by_run, label="best")

    comp = baseline_summary.merge(
        best_summary,
        on="metric",
        suffixes=("_baseline", "_best"),
        how="inner",
    )
    comp["delta_mean_best_minus_baseline"] = (
        pd.to_numeric(comp["error_mean_best"], errors="coerce")
        - pd.to_numeric(comp["error_mean_baseline"], errors="coerce")
    )

    return {
        "baseline_by_run": baseline_by_run,
        "best_by_run": best_by_run,
        "baseline_summary": baseline_summary,
        "best_summary": best_summary,
        "comparison": comp,
    }


def _build_report_from_summary(best_summary: pd.DataFrame) -> pd.DataFrame:
    report = best_summary.copy()
    report["threshold"] = report["metric"].map(DEFAULT_THRESHOLDS).astype(float)
    report["pass_rate"] = np.nan
    report["pass"] = report["error_mean"] < report["threshold"]
    report["status"] = report["pass"].map({True: "DONE (PASS)", False: "TODO (FAIL)"})
    return report[
        [
            "metric",
            "n_runs",
            "error_mean",
            "error_std",
            "error_ci95_low",
            "error_ci95_high",
            "threshold",
            "pass_rate",
            "pass",
            "status",
        ]
    ]


def main(n_jobs: int | None = None, fast_mode: bool | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_df = load_real_features()
    train_df, holdout_df = split_train_holdout(real_df, holdout_frac=0.20, seed=88)

    is_fast_mode = _resolve_fast_mode(fast_mode)
    search_mode = _resolve_search_mode()
    capacity_mode = _resolve_capacity_mode()
    objective_weights = _resolve_weights_from_env()

    if is_fast_mode:
        duration_min_obs_grid = [15, 20]
        loop_smoothing_grid = [1.0, 2.0, 3.0]
        stop_smoothing_grid = [1.0, 2.0, 3.0]
        eval_case_cap = 1200
        eval_cases_per_m = 30
        eval_seeds = [123, 321, 777]
        capacity_multiplier_grid = [0.75, 1.0, 1.25, 1.5, 2.0]
        capacity_eval_cases_per_m = 50
        capacity_eval_seeds = [2024, 2048, 2066]
    else:
        duration_min_obs_grid = [15, 20, 30]
        loop_smoothing_grid = [1.0, 2.0, 3.0, 5.0]
        stop_smoothing_grid = [1.0, 2.0, 3.0, 5.0]
        eval_case_cap = None
        eval_cases_per_m = 40
        eval_seeds = [123, 321, 777]
        capacity_multiplier_grid = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
        capacity_eval_cases_per_m = 60
        capacity_eval_seeds = [2024, 2048, 2066, 2084, 2102]

    print(
        f"Calibration mode: {'FAST' if is_fast_mode else 'FULL'} "
        f"(search_mode={search_mode}, capacity_mode={capacity_mode}, "
        f"objective_weights={objective_weights})"
    )

    base_capacity_by_m = load_resource_capacity_by_m(train_df)

    eval_cfg = {
        "seed": 123,
        "max_cases": eval_case_cap,
        "n_cases_per_municipality": eval_cases_per_m,
        "seeds": eval_seeds,
        "capacity_by_m": base_capacity_by_m,
        "capacity_multiplier": 1.0,
        "objective_weights": objective_weights,
    }

    candidate_grid = list(
        itertools.product(
            duration_min_obs_grid,
            loop_smoothing_grid,
            stop_smoothing_grid,
        )
    )

    workers = min(_resolve_n_jobs(n_jobs), len(candidate_grid))
    print(
        f"Calibration search: {len(candidate_grid)} candidates on {workers} worker(s), "
        f"cases/m={eval_cases_per_m}, seeds={eval_seeds}"
    )

    results: list[dict[str, float]] = []
    best_models: dict[str, Any] | None = None
    best_row: dict[str, float] | None = None

    if workers <= 1:
        _init_worker(train_df, holdout_df, eval_cfg)
        for idx, (d_min, loop_sm, stop_sm) in enumerate(candidate_grid, start=1):
            row, models = _run_single_candidate(int(d_min), float(loop_sm), float(stop_sm))
            results.append(row)
            if best_row is None or row["objective"] < best_row["objective"]:
                best_row = row
                best_models = models
            print(
                f"[{idx}/{len(candidate_grid)}] d_min={int(d_min)} loop_sm={float(loop_sm):.1f} "
                f"stop_sm={float(stop_sm):.1f} objective={float(row['objective']):.4f}"
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(train_df, holdout_df, eval_cfg),
        ) as executor:
            futures = {
                executor.submit(_run_single_candidate, int(d_min), float(loop_sm), float(stop_sm)): (
                    int(d_min),
                    float(loop_sm),
                    float(stop_sm),
                )
                for d_min, loop_sm, stop_sm in candidate_grid
            }

            completed = 0
            for future in concurrent.futures.as_completed(futures):
                row, models = future.result()
                results.append(row)
                if best_row is None or row["objective"] < best_row["objective"]:
                    best_row = row
                    best_models = models

                completed += 1
                d_min, loop_sm, stop_sm = futures[future]
                print(
                    f"[{completed}/{len(candidate_grid)}] d_min={int(d_min)} loop_sm={float(loop_sm):.1f} "
                    f"stop_sm={float(stop_sm):.1f} objective={float(row['objective']):.4f}"
                )

    search_df = pd.DataFrame(results).sort_values("objective").reset_index(drop=True)
    search_df["is_pareto"] = _mark_pareto_front(search_df, METRIC_COLUMNS).values

    if search_mode == "pareto":
        pareto_df = search_df[search_df["is_pareto"]].copy().sort_values("objective").reset_index(drop=True)
        chosen_row_df = pareto_df.head(1) if len(pareto_df) else search_df.head(1)
    else:
        chosen_row_df = search_df.head(1)

    chosen_row = chosen_row_df.iloc[0].to_dict()
    selected_tuple = (
        int(chosen_row["duration_min_obs"]),
        float(chosen_row["loop_smoothing"]),
        float(chosen_row["stop_smoothing"]),
    )

    if best_models is None or best_row is None or (
        int(best_row["duration_min_obs"]),
        float(best_row["loop_smoothing"]),
        float(best_row["stop_smoothing"]),
    ) != selected_tuple:
        best_models = fit_all_data_driven_models(
            train_df,
            duration_min_obs=selected_tuple[0],
            loop_smoothing=selected_tuple[1],
            stop_smoothing=selected_tuple[2],
        )
        best_row = chosen_row

    search_df.to_csv(SEARCH_REPORT_PATH, index=False)

    if best_models is None:
        raise RuntimeError("Calibration search failed to produce a model.")

    capacity_search_rows: list[dict[str, Any]] = []
    best_capacity_multiplier = 1.0
    best_capacity_metrics: dict[str, float] | None = None
    best_capacity_objective = float("inf")
    best_capacity_by_m_direct: dict[int, int] | None = None

    arrival_mean_by_m_holdout = _estimate_arrival_means_hours(holdout_df)

    if capacity_mode == "mmc":
        best_capacity_by_m_direct = estimate_capacity_by_m_mmc(
            train_df=train_df,
            target_wait_p95_hours=float(os.getenv("CALIB_TARGET_WAIT_P95_HOURS", "24")),
            max_servers=int(os.getenv("CALIB_MAX_SERVERS", "80")),
        )

        metrics_rows = []
        for seed in capacity_eval_seeds:
            m = evaluate_candidate_simpy_holdout(
                models=best_models,
                holdout_df=holdout_df,
                seed=int(seed),
                n_cases_per_municipality=capacity_eval_cases_per_m,
                capacity_by_m=best_capacity_by_m_direct,
                capacity_multiplier=1.0,
                arrival_mean_by_m=arrival_mean_by_m_holdout,
            )
            metrics_rows.append(m)

        cap_df = pd.DataFrame(metrics_rows)
        mean_metrics = {k: float(cap_df[k].mean()) for k in METRIC_COLUMNS}
        best_capacity_objective = compute_objective(mean_metrics, objective_weights)
        best_capacity_metrics = dict(mean_metrics)
        capacity_search_rows.append(
            {
                "capacity_mode": "mmc",
                "capacity_multiplier": 1.0,
                **mean_metrics,
                "objective": float(best_capacity_objective),
            }
        )
    else:
        for cap_mult in capacity_multiplier_grid:
            metrics_rows = []
            for seed in capacity_eval_seeds:
                m = evaluate_candidate_simpy_holdout(
                    models=best_models,
                    holdout_df=holdout_df,
                    seed=int(seed),
                    n_cases_per_municipality=capacity_eval_cases_per_m,
                    capacity_by_m=base_capacity_by_m,
                    capacity_multiplier=float(cap_mult),
                    arrival_mean_by_m=arrival_mean_by_m_holdout,
                )
                metrics_rows.append(m)

            cap_df = pd.DataFrame(metrics_rows)
            mean_metrics = {k: float(cap_df[k].mean()) for k in METRIC_COLUMNS}
            cap_objective = compute_objective(mean_metrics, objective_weights)

            row = {
                "capacity_mode": "grid",
                "capacity_multiplier": float(cap_mult),
                **mean_metrics,
                "objective": float(cap_objective),
            }
            capacity_search_rows.append(row)

            if cap_objective < best_capacity_objective:
                best_capacity_objective = float(cap_objective)
                best_capacity_multiplier = float(cap_mult)
                best_capacity_metrics = dict(mean_metrics)

    selected_config = {
        **{k: best_row[k] for k in ["duration_min_obs", "loop_smoothing", "stop_smoothing"]},
        "capacity_multiplier": float(best_capacity_multiplier),
    }
    if best_capacity_by_m_direct is not None:
        selected_config["capacity_by_m_direct"] = {str(k): int(v) for k, v in best_capacity_by_m_direct.items()}

    comparison = compare_default_vs_best(
        train_df=train_df,
        holdout_df=holdout_df,
        best_config=selected_config,
        n_runs=20,
        n_cases_per_municipality=120,
        base_seed=1337,
    )

    baseline_by_run_df = comparison["baseline_by_run"].copy()
    best_by_run_df = comparison["best_by_run"].copy()
    comp_df = comparison["comparison"].copy()
    best_summary_df = comparison["best_summary"].copy()

    baseline_by_run_df.to_csv(BASELINE_METRICS_BY_RUN_PATH, index=False)
    best_by_run_df.to_csv(METRICS_BY_RUN_PATH, index=False)
    comp_df.to_csv(COMPARISON_REPORT_PATH, index=False)

    report_df = _build_report_from_summary(best_summary_df)
    report_df.to_csv(REPORT_PATH, index=False)

    payload = {
        "best_config": selected_config,
        "fitted_models": best_models,
        "search_mode": search_mode,
        "objective_weights": objective_weights,
        "search_results_path": str(SEARCH_REPORT_PATH),
        "best_capacity_multiplier": float(best_capacity_multiplier),
        "best_capacity_mode": capacity_mode,
        "best_capacity_by_m_direct": best_capacity_by_m_direct,
        "capacity_multiplier_search": sorted(capacity_search_rows, key=lambda r: float(r["objective"])),
        "capacity_multiplier_objective": float(best_capacity_objective),
        "capacity_multiplier_best_metrics": best_capacity_metrics,
        "calibration_mode": "fast" if is_fast_mode else "full",
        "search_eval_cases_per_municipality": int(eval_cases_per_m),
        "search_eval_seeds": [int(x) for x in eval_seeds],
        "capacity_eval_cases_per_municipality": int(capacity_eval_cases_per_m),
        "capacity_eval_seeds": [int(x) for x in capacity_eval_seeds],
        "pareto_front": search_df[search_df["is_pareto"]].to_dict(orient="records"),
        "holdout_comparison_paths": {
            "baseline_by_run": str(BASELINE_METRICS_BY_RUN_PATH),
            "best_by_run": str(METRICS_BY_RUN_PATH),
            "summary_report": str(REPORT_PATH),
            "comparison": str(COMPARISON_REPORT_PATH),
        },
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
    }

    with open(BEST_CALIB_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved search report:", SEARCH_REPORT_PATH.resolve())
    print("Saved best calibration:", BEST_CALIB_PATH.resolve())
    print("Saved best-by-run metrics:", METRICS_BY_RUN_PATH.resolve())
    print("Saved baseline-by-run metrics:", BASELINE_METRICS_BY_RUN_PATH.resolve())
    print("Saved comparison report:", COMPARISON_REPORT_PATH.resolve())
    print("Saved validation report:", REPORT_PATH.resolve())

    print("Best selected config:")
    print(pd.Series(selected_config).to_string())

    print("Top 5 search rows by objective:")
    print(search_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
