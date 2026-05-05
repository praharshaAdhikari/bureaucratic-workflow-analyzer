"""
Tests for StreamIngestionPipeline (src/stream_ingestion.py).

Run with:
    cd src && python -m pytest tests/test_stream_ingestion.py -v
"""
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from stream_ingestion import StreamIngestionPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_XES = """\
<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xmlns="http://www.xes-standard.org/">
  <trace>
    <string key="concept:name" value="case1"/>
    <event>
      <string key="concept:name" value="ActivityA"/>
      <date key="time:timestamp" value="2024-01-01T10:00:00.000+00:00"/>
      <string key="org:resource" value="User1"/>
    </event>
  </trace>
</log>
"""

MALFORMED_XES = "this is not valid xml <<<"


def _make_pipeline(watch_dir, output_dir, max_buffer_rows=1000):
    return StreamIngestionPipeline(
        watch_dir=watch_dir,
        poll_interval_s=1.0,
        max_buffer_rows=max_buffer_rows,
        output_dir=output_dir,
        dataset_name="test_dataset",
    )


def _write_xes(directory, filename, content):
    path = Path(directory) / filename
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Test 1: poll_once detects a new XES file and parses it into the buffer
# ---------------------------------------------------------------------------

def test_poll_once_detects_new_xes_file():
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        pipeline = _make_pipeline(watch_dir, output_dir)
        _write_xes(watch_dir, "test.xes", MINIMAL_XES)

        count = pipeline.poll_once()

        assert count > 0, "Expected at least one row to be ingested"
        assert len(pipeline.buffer) == count
        assert "case_id" in pipeline.buffer.columns
        assert "activity" in pipeline.buffer.columns
        assert "timestamp" in pipeline.buffer.columns


# ---------------------------------------------------------------------------
# Test 2: same mtime → NOT re-ingested on second poll_once
# ---------------------------------------------------------------------------

def test_poll_once_does_not_reingest_same_mtime():
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        pipeline = _make_pipeline(watch_dir, output_dir)
        xes_path = _write_xes(watch_dir, "test.xes", MINIMAL_XES)

        first_count = pipeline.poll_once()
        assert first_count > 0

        # Second poll — same file, same mtime
        second_count = pipeline.poll_once()
        assert second_count == 0, "File with same mtime should not be re-ingested"
        assert len(pipeline.buffer) == first_count


# ---------------------------------------------------------------------------
# Test 3: malformed XES is skipped and logged without raising
# ---------------------------------------------------------------------------

def test_poll_once_skips_malformed_xes(caplog):
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        pipeline = _make_pipeline(watch_dir, output_dir)
        _write_xes(watch_dir, "bad.xes", MALFORMED_XES)

        with caplog.at_level(logging.ERROR):
            count = pipeline.poll_once()

        assert count == 0, "Malformed file should yield 0 ingested rows"
        assert len(pipeline.buffer) == 0
        # A warning/error should have been logged
        assert any("bad.xes" in r.message or "bad.xes" in str(r.args)
                   for r in caplog.records), \
            "Expected an error log mentioning the bad file"


# ---------------------------------------------------------------------------
# Test 4: buffer flush writes Parquet and trims in-memory buffer
# ---------------------------------------------------------------------------

def test_buffer_flush_writes_parquet_and_trims():
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        # max_buffer_rows=1 so any second row triggers a flush
        pipeline = _make_pipeline(watch_dir, output_dir, max_buffer_rows=1)

        # Push 3 events — should trigger flush
        records = [
            {"case_id": "c1", "activity": "A", "timestamp": "2024-01-01T10:00:00+00:00"},
            {"case_id": "c2", "activity": "B", "timestamp": "2024-01-01T11:00:00+00:00"},
            {"case_id": "c3", "activity": "C", "timestamp": "2024-01-01T12:00:00+00:00"},
        ]
        pipeline.push_batch(records)

        out_dir = Path(output_dir) / "test_dataset"
        parquet_files = list(out_dir.glob("stream_buffer_*.parquet"))
        assert len(parquet_files) >= 1, "Expected at least one Parquet flush file"

        # Buffer should be trimmed to at most max_buffer_rows
        assert len(pipeline.buffer) <= pipeline.max_buffer_rows


# ---------------------------------------------------------------------------
# Test 5: push_event raises ValueError for each missing required key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", ["case_id", "activity", "timestamp"])
def test_push_event_raises_for_missing_required_key(missing_key):
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        pipeline = _make_pipeline(watch_dir, output_dir)
        record = {
            "case_id": "c1",
            "activity": "A",
            "timestamp": "2024-01-01T10:00:00+00:00",
        }
        del record[missing_key]

        with pytest.raises(ValueError, match=missing_key):
            pipeline.push_event(record)


# ---------------------------------------------------------------------------
# Test 6: push_event drops records with un-parseable timestamps and logs warning
# ---------------------------------------------------------------------------

def test_push_event_drops_unparseable_timestamp(caplog):
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        pipeline = _make_pipeline(watch_dir, output_dir)
        record = {
            "case_id": "c1",
            "activity": "A",
            "timestamp": "not-a-timestamp",
        }

        with caplog.at_level(logging.WARNING):
            pipeline.push_event(record)

        assert len(pipeline.buffer) == 0, "Record with bad timestamp should be dropped"
        assert any("timestamp" in r.message.lower() or "unparseable" in r.message.lower()
                   for r in caplog.records), \
            "Expected a warning about the unparseable timestamp"


# ---------------------------------------------------------------------------
# Test 7: push_batch returns correct count and skips invalid records
# ---------------------------------------------------------------------------

def test_push_batch_returns_correct_count_and_skips_invalid():
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        pipeline = _make_pipeline(watch_dir, output_dir)
        records = [
            {"case_id": "c1", "activity": "A", "timestamp": "2024-01-01T10:00:00+00:00"},  # valid
            {"case_id": "c2", "activity": "B"},                                              # missing timestamp
            {"case_id": "c3", "activity": "C", "timestamp": "bad-ts"},                      # bad timestamp
            {"case_id": "c4", "activity": "D", "timestamp": "2024-01-01T12:00:00+00:00"},  # valid
        ]

        count = pipeline.push_batch(records)

        assert count == 2, f"Expected 2 valid records, got {count}"
        assert len(pipeline.buffer) == 2


# ---------------------------------------------------------------------------
# Test 8: pushing the same event twice results in exactly one row (deduplication)
# ---------------------------------------------------------------------------

def test_deduplication_same_event_twice():
    with tempfile.TemporaryDirectory() as watch_dir, \
         tempfile.TemporaryDirectory() as output_dir:

        pipeline = _make_pipeline(watch_dir, output_dir)
        record = {
            "case_id": "c1",
            "activity": "A",
            "timestamp": "2024-01-01T10:00:00+00:00",
        }

        pipeline.push_event(record)
        pipeline.push_event(record)

        assert len(pipeline.buffer) == 1, \
            f"Expected exactly 1 row after deduplication, got {len(pipeline.buffer)}"
