"""
Tests for StateSyncLayer (src/state_sync.py).

Run with:
    cd src && python -m pytest tests/test_state_sync.py -v
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from digital_twin import DigitalTwin
from stream_ingestion import StreamIngestionPipeline
from state_sync import StateSyncLayer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_minimal_twin():
    """Fit a DigitalTwin on a small synthetic event log."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(50):
        case_id = f"case_{i}"
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        for act in ["A", "B", "C"]:
            rows.append({
                "case_id": case_id,
                "activity": act,
                "timestamp": ts,
                "resource": "R1",
                "role": "analyst",
                "lifecycle": "complete" if act == "C" else "start",
            })
            ts += pd.Timedelta(hours=1)
    df = pd.DataFrame(rows)
    twin = DigitalTwin(seed=42)
    twin.fit(df)
    return twin, df


def make_sync(tmpdir):
    """Create a StateSyncLayer with a minimal fitted twin and pipeline."""
    pipeline = StreamIngestionPipeline(
        watch_dir=tmpdir,
        poll_interval_s=1.0,
        max_buffer_rows=10000,
        output_dir=tmpdir,
        dataset_name="test",
    )
    twin, df = make_minimal_twin()
    sync = StateSyncLayer(
        twin=twin,
        pipeline=pipeline,
        output_dir=tmpdir,
        dataset_name="test",
    )
    return sync, pipeline, twin, df


# ---------------------------------------------------------------------------
# Test 1: update_registry sets correct last_activity and last_timestamp
# ---------------------------------------------------------------------------

def test_update_registry_sets_last_activity_and_timestamp():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)

        # Pre-populate the buffer so auto-drift-check doesn't fail on empty buffer
        pipeline.buffer = df.copy()

        events = pd.DataFrame([
            {
                "case_id": "case_X",
                "activity": "A",
                "timestamp": pd.Timestamp("2024-06-01T10:00:00", tz="UTC"),
                "lifecycle": "start",
            },
            {
                "case_id": "case_X",
                "activity": "B",
                "timestamp": pd.Timestamp("2024-06-01T11:00:00", tz="UTC"),
                "lifecycle": "start",
            },
        ])

        # Patch check_drift to avoid side effects in this unit test
        with patch.object(sync, "check_drift", return_value={"drifted": False, "jsd": 0.0, "l1": 0.0, "triggered_metrics": []}):
            sync.update_registry(events)

        state = sync.get_case_state("case_X")
        assert state is not None, "case_X should be in the registry"
        assert state["last_activity"] == "B", f"Expected 'B', got {state['last_activity']}"
        assert state["last_timestamp"] == pd.Timestamp("2024-06-01T11:00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Test 2: "complete" lifecycle event removes the case from the registry
# ---------------------------------------------------------------------------

def test_update_registry_complete_lifecycle_removes_case():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)
        pipeline.buffer = df.copy()

        # First add a case
        start_events = pd.DataFrame([{
            "case_id": "case_Y",
            "activity": "A",
            "timestamp": pd.Timestamp("2024-06-01T10:00:00", tz="UTC"),
            "lifecycle": "start",
        }])
        with patch.object(sync, "check_drift", return_value={"drifted": False, "jsd": 0.0, "l1": 0.0, "triggered_metrics": []}):
            sync.update_registry(start_events)

        assert sync.get_case_state("case_Y") is not None, "case_Y should be active after start event"

        # Now send a complete event
        complete_events = pd.DataFrame([{
            "case_id": "case_Y",
            "activity": "C",
            "timestamp": pd.Timestamp("2024-06-01T12:00:00", tz="UTC"),
            "lifecycle": "complete",
        }])
        with patch.object(sync, "check_drift", return_value={"drifted": False, "jsd": 0.0, "l1": 0.0, "triggered_metrics": []}):
            sync.update_registry(complete_events)

        assert sync.get_case_state("case_Y") is None, "case_Y should be removed after 'complete' lifecycle"


# ---------------------------------------------------------------------------
# Test 3: get_active_cases returns correct columns and row count
# ---------------------------------------------------------------------------

def test_get_active_cases_columns_and_row_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)
        pipeline.buffer = df.copy()

        events = pd.DataFrame([
            {
                "case_id": f"case_{i}",
                "activity": "A",
                "timestamp": pd.Timestamp("2024-06-01T10:00:00", tz="UTC") + pd.Timedelta(hours=i),
                "lifecycle": "start",
            }
            for i in range(5)
        ])

        with patch.object(sync, "check_drift", return_value={"drifted": False, "jsd": 0.0, "l1": 0.0, "triggered_metrics": []}):
            sync.update_registry(events)

        active = sync.get_active_cases()

        assert set(active.columns) == {"case_id", "last_activity", "last_timestamp", "lifecycle"}, \
            f"Unexpected columns: {list(active.columns)}"
        assert len(active) == 5, f"Expected 5 active cases, got {len(active)}"


def test_get_active_cases_empty_returns_correct_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)

        active = sync.get_active_cases()

        assert list(active.columns) == ["case_id", "last_activity", "last_timestamp", "lifecycle"]
        assert len(active) == 0


# ---------------------------------------------------------------------------
# Test 4: get_case_state returns None for unknown case IDs
# ---------------------------------------------------------------------------

def test_get_case_state_returns_none_for_unknown():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)

        result = sync.get_case_state("nonexistent_case_id_xyz")
        assert result is None, f"Expected None for unknown case, got {result}"


# ---------------------------------------------------------------------------
# Test 5: check_drift returns warning: "insufficient_data" when buffer < window_size
# ---------------------------------------------------------------------------

def test_check_drift_insufficient_data_warning():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)

        # Buffer has fewer rows than window_size=500
        pipeline.buffer = df.head(10).copy()

        result = sync.check_drift(window_size=500)

        assert "warning" in result, "Expected 'warning' key in result"
        assert result["warning"] == "insufficient_data", \
            f"Expected 'insufficient_data', got {result['warning']}"


def test_check_drift_empty_buffer_returns_early():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)

        # Empty buffer
        pipeline.buffer = pd.DataFrame(columns=["case_id", "activity", "timestamp", "resource", "role", "lifecycle"])

        result = sync.check_drift(window_size=500)

        assert "warning" in result, "Expected 'warning' key for empty buffer"
        assert result["warning"] == "insufficient_data"
        # Should still return the required keys
        assert "drifted" in result
        assert "jsd" in result
        assert "l1" in result
        assert "triggered_metrics" in result


# ---------------------------------------------------------------------------
# Test 6: check_drift returns required keys: drifted, jsd, l1, triggered_metrics
# ---------------------------------------------------------------------------

def test_check_drift_returns_required_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        sync, pipeline, twin, df = make_sync(tmpdir)

        # Set buffer with enough data (window_size default is 500, use small window)
        pipeline.buffer = df.copy()

        result = sync.check_drift(window_size=50)

        assert "drifted" in result, "Missing 'drifted' key"
        assert "jsd" in result, "Missing 'jsd' key"
        assert "l1" in result, "Missing 'l1' key"
        assert "triggered_metrics" in result, "Missing 'triggered_metrics' key"

        assert isinstance(result["drifted"], bool), f"'drifted' should be bool, got {type(result['drifted'])}"
        assert isinstance(result["jsd"], float), f"'jsd' should be float, got {type(result['jsd'])}"
        assert isinstance(result["l1"], float), f"'l1' should be float, got {type(result['l1'])}"
        assert isinstance(result["triggered_metrics"], list), \
            f"'triggered_metrics' should be list, got {type(result['triggered_metrics'])}"
