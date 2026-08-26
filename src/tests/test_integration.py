"""
Integration tests for the Digital Twin Live Sync end-to-end pipeline.

Tests:
  - Round-trip fidelity: synthetic events → replay_into → buffer → Parquet → re-read
  - StateSyncLayer auto-wiring: update_registry called after push_batch via callback

Run with:
    cd src && python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from digital_twin import DigitalTwin
from stream_generator import SyntheticStreamGenerator
from stream_ingestion import StreamIngestionPipeline
from state_sync import StateSyncLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_twin(n_cases: int = 80, seed: int = 42) -> tuple[DigitalTwin, pd.DataFrame]:
    rows = []
    for i in range(n_cases):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i % 30)
        for act in ["A", "B", "C"]:
            rows.append({
                "case_id": f"case_{i}",
                "activity": act,
                "timestamp": ts,
                "resource": "R1",
                "role": "analyst",
                "lifecycle": "complete" if act == "C" else "start",
            })
            ts += pd.Timedelta(hours=1)
    df = pd.DataFrame(rows)
    twin = DigitalTwin(seed=seed)
    twin.fit(df)
    return twin, df


# ---------------------------------------------------------------------------
# Test 15.3: Round-trip fidelity
# ---------------------------------------------------------------------------

def test_round_trip_fidelity():
    """
    Synthetic events → replay_into → buffer → flush to Parquet → re-read.
    Asserts case_id, activity, and timestamp values are preserved exactly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        twin, _ = make_twin()
        gen = SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0)

        # Pipeline with small max_buffer_rows to force a Parquet flush
        pipeline = StreamIngestionPipeline(
            watch_dir=tmpdir,
            poll_interval_s=1.0,
            max_buffer_rows=10,  # force flush
            output_dir=tmpdir,
            dataset_name="roundtrip_test",
        )

        # Generate once, deterministically, and compare against exactly those
        # events.
        #
        # This test used to capture original_events from an initial start()
        # and then call reset(seed=42) before replaying. Those are two
        # different simulations: __init__ leaves _seed as None, so the first
        # start() takes the unseeded parallel path while reset(seed=42) takes
        # the seeded single-threaded one. The timestamps therefore differed by
        # hours and the test could never have passed on them; case_id and
        # activity matched only because make_twin() is a deterministic
        # A -> B -> C chain. A dtype mismatch was raised first and hid it.
        gen.start(n_cases=30)
        gen.reset(seed=42)
        original_df = pd.DataFrame(list(gen._events))[["case_id", "activity", "timestamp"]]

        gen.replay_into(pipeline, batch_size=20)

        # Collect all events: in-memory buffer + flushed Parquet files
        out_dir = Path(tmpdir) / "roundtrip_test"
        parquet_files = sorted(out_dir.glob("stream_buffer_*.parquet"))
        assert len(parquet_files) >= 1, "Expected at least one Parquet flush file"

        flushed_dfs = [pd.read_parquet(f) for f in parquet_files]
        all_df = pd.concat(flushed_dfs + [pipeline.buffer], ignore_index=True)

        # Deduplicate (same as pipeline does internally)
        all_df = all_df.drop_duplicates(
            subset=["case_id", "activity", "timestamp"], keep="first"
        ).reset_index(drop=True)

        # Verify all original events are present
        # Normalize timestamps for comparison
        all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=True)
        original_df["timestamp"] = pd.to_datetime(original_df["timestamp"], utc=True)

        # Sort both for comparison
        all_sorted = all_df[["case_id", "activity", "timestamp"]].sort_values(
            ["case_id", "activity", "timestamp"]
        ).reset_index(drop=True)
        orig_sorted = original_df.sort_values(
            ["case_id", "activity", "timestamp"]
        ).reset_index(drop=True)

        assert len(all_sorted) == len(orig_sorted), (
            f"Row count mismatch: got {len(all_sorted)}, expected {len(orig_sorted)}"
        )
        # check_dtype=False: the test is about values surviving the round trip,
        # and a Parquet round trip legitimately changes the storage dtype for
        # text columns under pandas 3 (str on the way in, object on the way
        # back). Values are still compared exactly.
        pd.testing.assert_frame_equal(
            all_sorted.reset_index(drop=True),
            orig_sorted.reset_index(drop=True),
            check_like=False,
            check_dtype=False,
        )


# ---------------------------------------------------------------------------
# Test: StateSyncLayer auto-wiring via callback
# ---------------------------------------------------------------------------

def test_state_sync_auto_wiring():
    """
    StateSyncLayer registers itself as a callback on the pipeline.
    After push_batch, update_registry should have been called automatically.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        twin, df = make_twin()

        pipeline = StreamIngestionPipeline(
            watch_dir=tmpdir,
            poll_interval_s=1.0,
            max_buffer_rows=100000,
            output_dir=tmpdir,
            dataset_name="wiring_test",
        )

        # Pre-populate buffer so check_drift doesn't fail on empty buffer
        pipeline.buffer = df.copy()

        from unittest.mock import patch
        sync = StateSyncLayer(
            twin=twin,
            pipeline=pipeline,
            output_dir=tmpdir,
            dataset_name="wiring_test",
        )

        # Patch check_drift to avoid side effects
        with patch.object(sync, "check_drift", return_value={
            "drifted": False, "jsd": 0.0, "l1": 0.0, "triggered_metrics": []
        }):
            # Push events via pipeline — should auto-call update_registry
            pipeline.push_batch([
                {"case_id": "auto_case_1", "activity": "A",
                 "timestamp": "2024-06-01T10:00:00+00:00", "lifecycle": "start"},
                {"case_id": "auto_case_2", "activity": "B",
                 "timestamp": "2024-06-01T11:00:00+00:00", "lifecycle": "start"},
            ])

        # Registry should have been updated automatically
        assert sync.get_case_state("auto_case_1") is not None, \
            "auto_case_1 should be in registry after push_batch via callback"
        assert sync.get_case_state("auto_case_2") is not None, \
            "auto_case_2 should be in registry after push_batch via callback"


# ---------------------------------------------------------------------------
# Test: Full pipeline schema conformance after replay
# ---------------------------------------------------------------------------

def test_synthetic_events_schema_after_replay():
    """
    After replay_into, all events in the pipeline buffer should have the
    correct schema columns and dtypes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        twin, _ = make_twin()
        gen = SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0)
        gen.start(n_cases=20)

        pipeline = StreamIngestionPipeline(
            watch_dir=tmpdir,
            poll_interval_s=1.0,
            max_buffer_rows=100000,
            output_dir=tmpdir,
            dataset_name="schema_test",
        )
        gen.replay_into(pipeline, batch_size=25)

        buf = pipeline.buffer
        required_cols = {"case_id", "activity", "timestamp", "resource", "role", "lifecycle"}
        assert required_cols.issubset(set(buf.columns)), \
            f"Missing columns: {required_cols - set(buf.columns)}"

        # Timestamps should be UTC-aware
        assert buf["timestamp"].dt.tz is not None, "Timestamps should be UTC-aware"

        # String columns should be object dtype
        for col in ("case_id", "activity", "resource", "role", "lifecycle"):
            assert buf[col].dtype == object, f"Column {col} should be object dtype"
