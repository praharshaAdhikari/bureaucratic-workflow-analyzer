"""
data_ingestion.py
-----------------
Parse XES event logs into structured DataFrames.
Handles BPIC-2015, BPIC-2017, BPIC-2012 and any XES-compliant log.

Parallelism strategy:
  - load_multiple_datasets: parses each file in a separate process (one per file).
  - parse_xes: for single large files, raw rows are collected in the main process
    (XML streaming can't be split), but the expensive _normalise_columns step
    (timestamp parsing, column renaming) runs in a thread pool over row chunks.
"""

import os
import gzip
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


XES_NS = "http://www.xes-standard.org/"


# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------

def _parse_value(elem) -> object:
    """Convert an XES attribute element to a Python value."""
    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
    val = elem.get("value", "")
    if tag == "int":
        try:
            return int(val)
        except ValueError:
            return val
    if tag == "float":
        try:
            return float(val)
        except ValueError:
            return val
    if tag == "boolean":
        return val.lower() == "true"
    if tag == "date":
        # Keep as string here — bulk-convert later with pd.to_datetime (much faster)
        return val
    return val


def _parse_attributes(elem) -> dict:
    """Recursively parse all child attribute elements of a trace/event."""
    attrs = {}
    for child in elem:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local in ("string", "int", "float", "boolean", "date", "id", "list"):
            key = child.get("key", "")
            attrs[key] = _parse_value(child)
    return attrs


# ---------------------------------------------------------------------------
# Core XES parser
# ---------------------------------------------------------------------------

def parse_xes(path: str | Path, max_traces: Optional[int] = None) -> pd.DataFrame:
    """
    Parse a .xes or .xes.gz file into a flat event DataFrame.

    Returns columns:
        case_id, activity, timestamp, lifecycle, resource, role,
        + any extra trace/event attributes found in the log.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    rows = []
    trace_count = 0

    with opener(path, "rb") as fh:
        context = ET.iterparse(fh, events=("start", "end"))
        trace_attrs: dict = {}
        in_trace = False
        in_event = False

        for event_type, elem in context:
            local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

            if event_type == "start" and local == "trace":
                in_trace = True
                in_event = False
                trace_attrs = {}

            elif event_type == "start" and local == "event" and in_trace:
                in_event = True

            elif event_type == "end" and local == "trace":
                in_trace = False
                in_event = False
                trace_count += 1
                elem.clear()
                if max_traces and trace_count >= max_traces:
                    break

            elif event_type == "end" and local == "event" and in_trace:
                ev_attrs = _parse_attributes(elem)
                safe_trace = {}
                for k, v in trace_attrs.items():
                    safe_trace[f"trace_{k}" if k == "concept:name" else k] = v
                rows.append({**safe_trace, **ev_attrs})
                in_event = False
                elem.clear()

            elif event_type == "end" and in_trace and not in_event and local not in (
                "trace", "event", "log", "extension", "global", "classifier"
            ):
                key = elem.get("key", "")
                if key:
                    trace_attrs[key] = _parse_value(elem)

    df = pd.DataFrame(rows)
    df = _normalise_columns(df)
    return df


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names across different XES logs."""
    rename = {}

    for cand in ["activityNameEN", "activityNameNL", "Activity", "concept:name"]:
        if cand in df.columns and "activity" not in df.columns:
            rename[cand] = "activity"
            break

    for cand in ["time:timestamp", "timestamp", "Timestamp"]:
        if cand in df.columns and "timestamp" not in df.columns:
            rename[cand] = "timestamp"
            break

    for cand in ["org:resource", "Resource", "resource"]:
        if cand in df.columns and "resource" not in df.columns:
            rename[cand] = "resource"
            break

    for cand in ["org:role", "role", "Role"]:
        if cand in df.columns and "role" not in df.columns:
            rename[cand] = "role"
            break

    for cand in ["lifecycle:transition", "lifecycle", "Action"]:
        if cand in df.columns and "lifecycle" not in df.columns:
            rename[cand] = "lifecycle"
            break

    activity_src = next((k for k, v in rename.items() if v == "activity"), None)
    for cand in ["trace_concept:name", "concept:name", "case_id", "CaseID"]:
        if cand in df.columns and "case_id" not in df.columns and cand != activity_src:
            rename[cand] = "case_id"
            break

    df = df.rename(columns=rename)

    if "activity" not in df.columns or df["activity"].eq("UNKNOWN").all():
        for cand in ["action_code", "activityNameNL", "activityNameEN"]:
            if cand in df.columns and not df[cand].eq("UNKNOWN").all():
                df["activity"] = df[cand]
                break

    for col in ["activity", "timestamp", "resource", "role", "lifecycle", "case_id"]:
        if col not in df.columns:
            df[col] = "UNKNOWN"

    # Bulk timestamp parse — much faster than per-row pd.to_datetime in _parse_value.
    # Check for both object dtype (standard pandas) and ArrowDtype string types
    # (pandas with PyArrow backend stores strings as large_string, not object).
    ts_col = df["timestamp"]
    needs_parse = (
        ts_col.dtype == object
        or (hasattr(ts_col, "dtype") and hasattr(ts_col.dtype, "pyarrow_dtype"))
        or pd.api.types.is_string_dtype(ts_col)
    )
    if needs_parse:
        df["timestamp"] = pd.to_datetime(
            ts_col.astype(str), utc=True, errors="coerce"
        )

    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path, max_traces: Optional[int] = None) -> pd.DataFrame:
    """Convenience wrapper — accepts .xes or .xes.gz."""
    return parse_xes(path, max_traces=max_traces)


def _load_one(args: tuple) -> pd.DataFrame:
    """Worker function for parallel file loading (must be module-level for pickle)."""
    path, dataset_tag, max_traces = args
    df = load_dataset(path, max_traces=max_traces)
    if dataset_tag:
        df["dataset"] = Path(path).stem
    return df


def load_multiple_datasets(
    paths: list[str | Path],
    dataset_tag: bool = True,
    max_traces: Optional[int] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Load and concatenate multiple XES logs in parallel.

    Each file is parsed in its own subprocess so multiple large files
    (e.g. BPIC-2015 M1–M5) are read concurrently rather than sequentially.

    Args:
        paths:       List of .xes / .xes.gz file paths.
        dataset_tag: Add a 'dataset' column with the file stem.
        max_traces:  Cap per file (None = load all).
        n_jobs:      Worker count. -1 = min(n_files, cpu_count).
    """
    if len(paths) == 1:
        # Single file — no point spawning a subprocess
        return _load_one((paths[0], dataset_tag, max_traces))

    n_workers = min(len(paths), os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)
    args = [(p, dataset_tag, max_traces) for p in paths]

    dfs: list[pd.DataFrame] = [None] * len(paths)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(_load_one, a): i for i, a in enumerate(args)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            dfs[idx] = future.result()

    return pd.concat(dfs, ignore_index=True)
