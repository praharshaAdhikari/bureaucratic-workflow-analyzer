"""
_prom_sim_workers.py
~~~~~~~~~~~~~~~~~~~~
Worker functions for step7-4_prom_simulations.ipynb.

All simulation logic lives here so that ProcessPoolExecutor subprocesses
can import it cleanly (Jupyter's __main__ is not picklable by workers).
"""
from __future__ import annotations

import io
import sys
import time

import pandas as pd
import pm4py


# ---------------------------------------------------------------------------
# Recursion limit – set in every process that imports this module
# ---------------------------------------------------------------------------
sys.setrecursionlimit(50_000)


# ---------------------------------------------------------------------------
# PM4Py helpers
# ---------------------------------------------------------------------------

def to_pm4py_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    d = df_in[["case_id", "activity", "timestamp"]].copy()
    d["case:concept:name"] = d["case_id"].astype(str)
    d["concept:name"] = d["activity"].astype(str)
    d["time:timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"])
    d = d[["case:concept:name", "concept:name", "time:timestamp"]]
    return pm4py.format_dataframe(
        d,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )


def to_utc_timestamp(value, fallback=None) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return fallback
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def eventlog_to_trace_table(elog, municipality: int, run_id: int) -> pd.DataFrame:
    rows = []
    for trace_idx, trace in enumerate(elog):
        case_id = f"M{municipality}_SIM_{run_id}_{trace_idx + 1:06d}"
        if len(trace) == 0:
            continue

        first_ts = to_utc_timestamp(trace[0].get("time:timestamp"))
        t0 = first_ts if first_ts is not None else pd.Timestamp.now(tz="UTC")

        prev_act = None
        prev_ts = None
        seen_counts: dict[str, int] = {}

        for i, ev in enumerate(trace):
            raw_ts = ev.get("time:timestamp")
            ts = to_utc_timestamp(raw_ts, fallback=(t0 + pd.Timedelta(hours=i)))

            act = str(ev.get("concept:name", "UNKNOWN"))
            seen_counts[act] = seen_counts.get(act, 0) + 1

            rows.append(
                {
                    "run_id": int(run_id),
                    "municipality": int(municipality),
                    "case_id": case_id,
                    "event_id": f"{case_id}_E{i + 1:04d}",
                    "timestamp": ts,
                    "activity": act,
                    "prev_activity": prev_act,
                    "step_index": int(i),
                    "trace_length": int(len(trace)),
                    "time_since_case_start_hours": float(
                        (ts - t0).total_seconds() / 3600.0
                    ),
                    "time_since_prev_hours": (
                        0.0
                        if prev_ts is None
                        else float((ts - prev_ts).total_seconds() / 3600.0)
                    ),
                    "rework_count_activity": int(max(0, seen_counts[act] - 1)),
                    "seen_activity_before": bool(seen_counts[act] > 1),
                    "case_completed": True,
                }
            )
            prev_act = act
            prev_ts = ts

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out = out.sort_values(
        ["run_id", "municipality", "case_id", "step_index"]
    ).reset_index(drop=True)
    out["next_activity"] = out.groupby(["run_id", "municipality", "case_id"])[
        "activity"
    ].shift(-1)
    return out


def trace_to_episode_summary(trace_df: pd.DataFrame) -> pd.DataFrame:
    if len(trace_df) == 0:
        return pd.DataFrame(
            columns=[
                "run_id",
                "municipality",
                "case_id",
                "steps",
                "duration_hours",
                "loops",
                "case_completed",
            ]
        )

    def loops_for_case(df_case: pd.DataFrame) -> int:
        vc = df_case["activity"].value_counts()
        return int((vc > 1).sum())

    grp = ["run_id", "municipality", "case_id"]
    steps = trace_df.groupby(grp)["step_index"].max().add(1).rename("steps")
    dur = (
        trace_df.groupby(grp)["time_since_case_start_hours"]
        .max()
        .rename("duration_hours")
    )
    loops = (
        trace_df.groupby(grp, group_keys=False)
        .apply(loops_for_case, include_groups=False)
        .rename("loops")
    )
    comp = trace_df.groupby(grp)["case_completed"].max().rename("case_completed")

    return pd.concat([steps, dur, loops, comp], axis=1).reset_index()


# ---------------------------------------------------------------------------
# Discovery / playout helpers (version-compat wrappers)
# ---------------------------------------------------------------------------

def discover_inductive_tree_compat(event_log, noise_threshold: float):
    """PM4Py-version compatible inductive tree discovery."""
    attempts = [
        {"noise_threshold": noise_threshold},
        {"noiseThreshold": noise_threshold},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return pm4py.discover_process_tree_inductive(event_log, **kwargs)
        except TypeError as e:
            last_error = e
    raise last_error  # type: ignore[misc]


def discover_heuristics_net_compat(event_log, noise_threshold: float):
    """PM4Py-version compatible heuristics-net discovery."""
    attempts = [
        {"noise_threshold": noise_threshold},
        {"noiseThreshold": noise_threshold},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return pm4py.discover_heuristics_net(event_log, **kwargs)
        except TypeError as e:
            last_error = e
    raise last_error  # type: ignore[misc]


def play_out_compat(*args, n_traces: int):
    """PM4Py-version compatible playout."""
    attempts = [
        {"parameters": {"no_traces": n_traces}},
        {"no_traces": n_traces},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return pm4py.play_out(*args, **kwargs)
        except TypeError as e:
            last_error = e
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Per-municipality simulators
# ---------------------------------------------------------------------------

def simulate_inductive_for_municipality(
    df_m: pd.DataFrame,
    municipality: int,
    run_id: int = 1,
    noise_threshold: float = 0.0,
) -> pd.DataFrame:
    t0 = time.time()
    log_df = to_pm4py_dataframe(df_m)
    event_log = pm4py.convert_to_event_log(log_df)
    n_traces = int(df_m["case_id"].nunique())

    discovery_start = time.time()
    tree = discover_inductive_tree_compat(event_log, noise_threshold=noise_threshold)
    discovery_time = time.time() - discovery_start

    playout_start = time.time()
    sim_log = play_out_compat(tree, n_traces=n_traces)
    playout_time = time.time() - playout_start

    result = eventlog_to_trace_table(sim_log, municipality=municipality, run_id=run_id)
    print(
        f"  M{municipality} inductive: discovery={discovery_time:.1f}s,"
        f" playout={playout_time:.1f}s, total={time.time() - t0:.1f}s"
    )
    return result


def simulate_heuristic_for_municipality(
    df_m: pd.DataFrame,
    municipality: int,
    run_id: int = 1,
    noise_threshold: float = 0.0,
) -> pd.DataFrame:
    t0 = time.time()
    log_df = to_pm4py_dataframe(df_m)
    event_log = pm4py.convert_to_event_log(log_df)
    n_traces = int(df_m["case_id"].nunique())

    discovery_start = time.time()
    heu_net = discover_heuristics_net_compat(event_log, noise_threshold=noise_threshold)
    discovery_time = time.time() - discovery_start

    conversion_start = time.time()
    net, im, fm = pm4py.convert_to_petri_net(heu_net)
    conversion_time = time.time() - conversion_start

    playout_start = time.time()
    sim_log = play_out_compat(net, im, fm, n_traces=n_traces)
    playout_time = time.time() - playout_start

    result = eventlog_to_trace_table(sim_log, municipality=municipality, run_id=run_id)
    print(
        f"  M{municipality} heuristic: discovery={discovery_time:.1f}s,"
        f" conversion={conversion_time:.1f}s, playout={playout_time:.1f}s,"
        f" total={time.time() - t0:.1f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Top-level worker – must be module-level for ProcessPoolExecutor pickling
# ---------------------------------------------------------------------------

def simulate_municipality_worker(
    args: tuple,
) -> tuple[int, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """
    Worker entry-point for ProcessPoolExecutor.

    args = (municipality_id, df_parquet_bytes, run_id, noise_threshold, run_inductive)

    Returns (municipality_id, inductive_df_or_None, heuristic_df_or_None, error_list).
    """
    m, df_bytes, run_id, noise_threshold, run_inductive = args
    df_m = pd.read_parquet(io.BytesIO(df_bytes))

    tr_ind: pd.DataFrame | None = None
    tr_heu: pd.DataFrame | None = None
    errors: list[str] = []

    muni_start = time.time()
    n_cases = df_m["case_id"].nunique()
    print(f"[worker] Simulating M{m} ({len(df_m):,} events, {n_cases:,} cases)...")

    if run_inductive:
        try:
            tr_ind = simulate_inductive_for_municipality(
                df_m,
                municipality=m,
                run_id=run_id,
                noise_threshold=noise_threshold,
            )
        except Exception as e:
            errors.append(f"[inductive] M{m}: {type(e).__name__}: {e}")

    try:
        tr_heu = simulate_heuristic_for_municipality(
            df_m,
            municipality=m,
            run_id=run_id,
            noise_threshold=noise_threshold,
        )
    except Exception as e:
        errors.append(f"[heuristic] M{m}: {type(e).__name__}: {e}")

    print(f"[worker] ✓ M{m} done in {time.time() - muni_start:.1f}s")
    return m, tr_ind, tr_heu, errors
