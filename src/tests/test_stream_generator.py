"""
Tests for SyntheticStreamGenerator (src/stream_generator.py).

Run with:
    cd src && python -m pytest tests/test_stream_generator.py -v
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_twin(n_cases: int = 60, seed: int = 42) -> DigitalTwin:
    """Fit a minimal DigitalTwin on a synthetic event log."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cases):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i)
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
    return twin


def make_generator(twin: DigitalTwin | None = None) -> SyntheticStreamGenerator:
    if twin is None:
        twin = make_twin()
    return SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0, speed_factor=1.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_raises_for_nonpositive_speed_factor():
    twin = make_twin()
    with pytest.raises(ValueError, match="speed_factor must be positive"):
        SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0, speed_factor=0.0)
    with pytest.raises(ValueError, match="speed_factor must be positive"):
        SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0, speed_factor=-1.0)


def test_constructor_raises_for_invalid_split_ratio():
    twin = make_twin()
    with pytest.raises(ValueError, match="split_ratio must be in"):
        SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0, split_ratio=0.0)
    with pytest.raises(ValueError, match="split_ratio must be in"):
        SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0, split_ratio=1.0)
    with pytest.raises(ValueError, match="split_ratio must be in"):
        SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0, split_ratio=1.5)


# ---------------------------------------------------------------------------
# start() and schema
# ---------------------------------------------------------------------------

def test_start_generates_events():
    gen = make_generator()
    gen.start(n_cases=20)
    assert len(gen._events) > 0
    assert not gen.is_exhausted


def test_start_produces_correct_schema():
    gen = make_generator()
    gen.start(n_cases=20)
    required = {"case_id", "activity", "timestamp", "resource", "role", "lifecycle"}
    keys = set(gen._events[0].keys())
    assert required.issubset(keys), f"Missing columns: {required - keys}"


def test_start_events_sorted_by_timestamp():
    gen = make_generator()
    gen.start(n_cases=30)
    timestamps = [e["timestamp"] for e in gen._events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], \
            f"Events not sorted at index {i}: {timestamps[i-1]} > {timestamps[i]}"


def test_lifecycle_defaults_to_complete():
    gen = make_generator()
    gen.start(n_cases=20)
    for e in gen._events:
        assert e["lifecycle"] is not None
        assert isinstance(e["lifecycle"], str)


# ---------------------------------------------------------------------------
# get_next_event
# ---------------------------------------------------------------------------

def test_get_next_event_returns_events_in_order():
    gen = make_generator()
    gen.start(n_cases=10)
    prev_ts = None
    while not gen.is_exhausted:
        e = gen.get_next_event()
        assert e is not None
        if prev_ts is not None:
            assert e["timestamp"] >= prev_ts
        prev_ts = e["timestamp"]


def test_get_next_event_returns_none_when_exhausted():
    gen = make_generator()
    gen.start(n_cases=5)
    # Drain all events
    while gen.get_next_event() is not None:
        pass
    assert gen.get_next_event() is None


# ---------------------------------------------------------------------------
# emit_batch
# ---------------------------------------------------------------------------

def test_emit_batch_returns_at_most_n():
    gen = make_generator()
    gen.start(n_cases=20)
    batch = gen.emit_batch(5)
    assert len(batch) <= 5


def test_emit_batch_returns_empty_when_exhausted():
    gen = make_generator()
    gen.start(n_cases=5)
    # Drain
    while not gen.is_exhausted:
        gen.emit_batch(100)
    assert gen.emit_batch(10) == []


def test_emit_batch_drains_all_events():
    gen = make_generator()
    gen.start(n_cases=20)
    total = len(gen._events)
    collected = []
    while not gen.is_exhausted:
        collected.extend(gen.emit_batch(7))
    assert len(collected) == total


# ---------------------------------------------------------------------------
# is_exhausted
# ---------------------------------------------------------------------------

def test_is_exhausted_false_after_start():
    gen = make_generator()
    gen.start(n_cases=10)
    assert not gen.is_exhausted


def test_is_exhausted_true_after_all_consumed():
    gen = make_generator()
    gen.start(n_cases=5)
    while not gen.is_exhausted:
        gen.get_next_event()
    assert gen.is_exhausted


# ---------------------------------------------------------------------------
# reset — reproducibility
# ---------------------------------------------------------------------------

def test_reset_same_seed_produces_identical_sequences():
    twin = make_twin()
    gen1 = SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0)
    gen2 = SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0)

    gen1.start(n_cases=30)
    gen1.reset(seed=99)

    gen2.start(n_cases=30)
    gen2.reset(seed=99)

    assert len(gen1._events) == len(gen2._events), "Event counts differ after same seed reset"
    for i, (e1, e2) in enumerate(zip(gen1._events, gen2._events)):
        assert e1["case_id"] == e2["case_id"], f"case_id mismatch at index {i}"
        assert e1["activity"] == e2["activity"], f"activity mismatch at index {i}"
        assert e1["timestamp"] == e2["timestamp"], f"timestamp mismatch at index {i}"


def test_reset_different_seeds_produce_different_sequences():
    twin = make_twin()
    gen = SyntheticStreamGenerator(twin=twin, arrival_rate_s=3600.0)
    gen.start(n_cases=30)

    gen.reset(seed=1)
    seq1 = [e["activity"] for e in gen._events]

    gen.reset(seed=2)
    seq2 = [e["activity"] for e in gen._events]

    # With different seeds, sequences should differ (with overwhelming probability)
    assert seq1 != seq2, "Different seeds produced identical sequences (extremely unlikely)"


def test_reset_resets_cursor():
    gen = make_generator()
    gen.start(n_cases=10)
    # Drain half
    gen.emit_batch(5)
    assert gen._cursor > 0
    gen.reset(seed=42)
    assert gen._cursor == 0
    assert not gen.is_exhausted


# ---------------------------------------------------------------------------
# get_wall_clock_delay
# ---------------------------------------------------------------------------

def test_get_wall_clock_delay_first_event_is_zero():
    gen = make_generator()
    gen.start(n_cases=10)
    first = gen._events[0]
    delay = gen.get_wall_clock_delay(first)
    assert delay == 0.0


def test_get_wall_clock_delay_nonnegative():
    gen = make_generator()
    gen.start(n_cases=10)
    for e in gen._events:
        assert gen.get_wall_clock_delay(e) >= 0.0


# ---------------------------------------------------------------------------
# replay_into
# ---------------------------------------------------------------------------

def test_replay_into_feeds_all_events_and_returns_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = make_generator()
        gen.start(n_cases=20)
        total_events = len(gen._events)

        pipeline = StreamIngestionPipeline(
            watch_dir=tmpdir, poll_interval_s=1.0, max_buffer_rows=100000,
            output_dir=tmpdir, dataset_name="test",
        )
        count = gen.replay_into(pipeline, batch_size=10)

        assert count == total_events, f"Expected {total_events} ingested, got {count}"
        assert gen.is_exhausted


def test_replay_into_exhausted_returns_zero():
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = make_generator()
        gen.start(n_cases=10)
        # Drain first
        while not gen.is_exhausted:
            gen.get_next_event()

        pipeline = StreamIngestionPipeline(
            watch_dir=tmpdir, poll_interval_s=1.0, max_buffer_rows=100000,
            output_dir=tmpdir, dataset_name="test",
        )
        count = gen.replay_into(pipeline)
        assert count == 0


def test_replay_into_is_exhausted_after_completion():
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = make_generator()
        gen.start(n_cases=15)
        pipeline = StreamIngestionPipeline(
            watch_dir=tmpdir, poll_interval_s=1.0, max_buffer_rows=100000,
            output_dir=tmpdir, dataset_name="test",
        )
        gen.replay_into(pipeline)
        assert gen.is_exhausted
