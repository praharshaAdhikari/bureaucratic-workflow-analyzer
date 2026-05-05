"""
stream_ingestion.py
-------------------
Event Stream Ingestion Pipeline for the Digital Twin Live Sync feature.

Consumes new XES or structured event-stream data in near-real-time,
normalizes it through the existing data_ingestion pipeline, and buffers
events for downstream state synchronization and drift detection.

Importable as:
    from stream_ingestion import StreamIngestionPipeline
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data_ingestion import parse_xes, _normalise_columns

logger = logging.getLogger(__name__)

# Standard event log schema columns
_SCHEMA_COLUMNS = ["case_id", "activity", "timestamp", "resource", "role", "lifecycle"]


class StreamIngestionPipeline:
    """
    Ingestion pipeline that watches a directory for new XES files and/or
    accepts programmatic event pushes, normalizes all events to the standard
    schema, and maintains an in-memory buffer with Parquet sink flushing.

    Parameters
    ----------
    watch_dir : str | Path
        Directory to poll for new or updated XES / XES.GZ files.
    poll_interval_s : float
        Polling interval in seconds between directory scans.
    max_buffer_rows : int
        Maximum number of rows to retain in the in-memory buffer before
        flushing the oldest events to a dated Parquet file.
    output_dir : str | Path
        Root output directory; Parquet sinks are written to
        ``output_dir / dataset_name /``.
    dataset_name : str
        Dataset identifier used to construct the output subdirectory path
        and artifact file names.
    seed : int | None, optional
        Seed for the internal NumPy RNG (``np.random.default_rng``).
        Defaults to ``None`` (non-deterministic).
    """

    def __init__(
        self,
        watch_dir: str | Path,
        poll_interval_s: float,
        max_buffer_rows: int,
        output_dir: str | Path,
        dataset_name: str,
        seed: int | None = None,
    ) -> None:
        self.watch_dir = Path(watch_dir)
        self.poll_interval_s = poll_interval_s
        self.max_buffer_rows = max_buffer_rows
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name

        # In-memory event buffer — always has the standard schema columns
        # Use explicit dtypes to avoid FutureWarning on concat with empty DataFrame
        self.buffer: pd.DataFrame = pd.DataFrame({
            "case_id": pd.Series(dtype="object"),
            "activity": pd.Series(dtype="object"),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "resource": pd.Series(dtype="object"),
            "role": pd.Series(dtype="object"),
            "lifecycle": pd.Series(dtype="object"),
        })

        # Maps file path (str) → last-processed mtime (float) to prevent re-ingestion
        self.last_ingestion_ts: dict[str, float] = {}

        # Registered callbacks fired after each ingestion event
        self._on_ingest_callbacks: list[Callable[[pd.DataFrame], None]] = []

        # RNG — use default_rng, never np.random.seed()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_callback(self, fn: Callable[[pd.DataFrame], None]) -> None:
        """
        Register a callable to be invoked after each ingestion event.

        The callable receives a single argument: a ``pd.DataFrame`` containing
        the newly ingested events (conforming to the standard schema).

        Parameters
        ----------
        fn : callable
            Function with signature ``fn(new_events: pd.DataFrame) -> None``.
        """
        self._on_ingest_callbacks.append(fn)

    def _fire_callbacks(self, df: pd.DataFrame) -> None:
        """
        Invoke all registered ``on_ingest`` callbacks with the new events.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of newly ingested events to pass to each callback.
        """
        for fn in self._on_ingest_callbacks:
            try:
                fn(df)
            except Exception:
                logger.exception(
                    "on_ingest callback %r raised an exception; continuing.", fn
                )

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _fill_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all 6 standard schema columns exist, filling missing ones with
        ``"UNKNOWN"``.  ``_normalise_columns`` (called inside ``parse_xes``)
        already handles renaming; this method only fills gaps.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to fill.

        Returns
        -------
        pd.DataFrame
            DataFrame guaranteed to have all columns in ``_SCHEMA_COLUMNS``.
        """
        for col in ("resource", "role", "lifecycle"):
            if col not in df.columns:
                df[col] = "UNKNOWN"
            else:
                df[col] = df[col].fillna("UNKNOWN")
        # Ensure remaining schema columns exist (case_id, activity, timestamp)
        for col in _SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = "UNKNOWN"
        return df

    # ------------------------------------------------------------------
    # Buffer flush
    # ------------------------------------------------------------------

    def _maybe_flush(self) -> None:
        """
        Flush the oldest rows from the in-memory buffer to a dated Parquet file
        when the buffer exceeds ``self.max_buffer_rows``.

        The flush writes to::

            self.output_dir / self.dataset_name / stream_buffer_{timestamp}.parquet

        and retains only the most recent ``self.max_buffer_rows`` rows in memory.
        """
        if len(self.buffer) <= self.max_buffer_rows:
            return

        n_flush = len(self.buffer) - self.max_buffer_rows
        flush_df = self.buffer.iloc[:n_flush].copy()
        self.buffer = self.buffer.iloc[n_flush:].reset_index(drop=True)

        timestamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S%f")
        out_dir = self.output_dir / self.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"stream_buffer_{timestamp}.parquet"

        flush_df.to_parquet(out_path, engine="pyarrow", index=False)
        logger.info("Flushed %d rows to %s", n_flush, out_path)

    # ------------------------------------------------------------------
    # XES directory polling
    # ------------------------------------------------------------------

    def poll_once(self) -> int:
        """
        Scan ``self.watch_dir`` for new or updated ``.xes`` / ``.xes.gz`` files
        and ingest any that have been modified since the last poll.

        For each qualifying file:

        * Calls ``data_ingestion.parse_xes`` (which already applies
          ``_normalise_columns`` internally).
        * Fills missing ``resource``, ``role``, ``lifecycle`` columns with
          ``"UNKNOWN"`` via :meth:`_fill_missing_columns`.
        * Appends the normalized events to :attr:`buffer`.
        * Updates :attr:`last_ingestion_ts` with the file's current ``mtime``.

        Any exception raised during parsing is caught, logged, and the file is
        skipped — the polling loop is never halted.

        After processing all new files:

        * Calls :meth:`_maybe_flush` to trim the buffer if needed.
        * Fires registered callbacks with the newly ingested events.

        Returns
        -------
        int
            Total number of newly ingested rows across all processed files.
        """
        new_frames: list[pd.DataFrame] = []

        for file_path in sorted(self.watch_dir.glob("**/*")):
            if file_path.suffix not in (".xes", ".gz"):
                continue
            # Accept .xes and .xes.gz only
            name = file_path.name
            if not (name.endswith(".xes") or name.endswith(".xes.gz")):
                continue

            key = str(file_path)
            try:
                mtime = os.path.getmtime(file_path)
            except OSError:
                continue

            if mtime <= self.last_ingestion_ts.get(key, 0.0):
                continue

            # New or updated file — attempt to parse
            try:
                df = parse_xes(file_path)
            except Exception:
                logger.exception("Failed to parse XES file: %s — skipping.", file_path)
                continue

            df = self._fill_missing_columns(df)
            new_frames.append(df)
            self.last_ingestion_ts[key] = mtime

        if not new_frames:
            self._maybe_flush()
            return 0

        new_events_df = pd.concat(new_frames, ignore_index=True)
        self.buffer = pd.concat([self.buffer, new_events_df], ignore_index=True)

        self._maybe_flush()
        self._fire_callbacks(new_events_df)

        return len(new_events_df)

    # ------------------------------------------------------------------
    # Programmatic push interface
    # ------------------------------------------------------------------

    def push_event(self, record: dict) -> None:
        """
        Push a single event record into the buffer.

        Parameters
        ----------
        record : dict
            Event record with at minimum ``case_id``, ``activity``, ``timestamp``.

        Raises
        ------
        ValueError
            If any required key is missing from the record.
        """
        for key in ("case_id", "activity", "timestamp"):
            if key not in record:
                raise ValueError(f"Missing required key: '{key}'")

        ts = pd.to_datetime(record["timestamp"], utc=True, errors="coerce")
        if ts is pd.NaT:
            logger.warning(
                "Unparseable timestamp for case_id=%r activity=%r — dropping record.",
                record["case_id"],
                record["activity"],
            )
            return

        row = {
            "case_id": record["case_id"],
            "activity": record["activity"],
            "timestamp": ts,
            "resource": record.get("resource", "UNKNOWN") or "UNKNOWN",
            "role": record.get("role", "UNKNOWN") or "UNKNOWN",
            "lifecycle": record.get("lifecycle", "UNKNOWN") or "UNKNOWN",
        }

        new_df = pd.DataFrame([row])
        self.buffer = pd.concat([self.buffer, new_df], ignore_index=True)
        self._deduplicate()
        self._maybe_flush()
        self._fire_callbacks(new_df)

    def push_batch(self, records: list[dict]) -> int:
        """
        Push a list of event records into the buffer.

        Invalid records (missing required keys or un-parseable timestamps) are
        skipped with a logged warning; processing continues for the remaining
        records.

        Parameters
        ----------
        records : list[dict]
            List of event record dicts.

        Returns
        -------
        int
            Count of successfully ingested records.
        """
        valid_rows: list[dict] = []

        for idx, record in enumerate(records):
            # Validate required keys
            missing_key = None
            for key in ("case_id", "activity", "timestamp"):
                if key not in record:
                    missing_key = key
                    break
            if missing_key is not None:
                logger.warning(
                    "push_batch index=%d: missing required key '%s' — skipping.",
                    idx,
                    missing_key,
                )
                continue

            ts = pd.to_datetime(record["timestamp"], utc=True, errors="coerce")
            if ts is pd.NaT:
                logger.warning(
                    "push_batch index=%d: unparseable timestamp for "
                    "case_id=%r activity=%r — skipping.",
                    idx,
                    record["case_id"],
                    record["activity"],
                )
                continue

            valid_rows.append(
                {
                    "case_id": record["case_id"],
                    "activity": record["activity"],
                    "timestamp": ts,
                    "resource": record.get("resource", "UNKNOWN") or "UNKNOWN",
                    "role": record.get("role", "UNKNOWN") or "UNKNOWN",
                    "lifecycle": record.get("lifecycle", "UNKNOWN") or "UNKNOWN",
                }
            )

        if not valid_rows:
            return 0

        new_df = pd.DataFrame(valid_rows)
        self.buffer = pd.concat([self.buffer, new_df], ignore_index=True)
        self._deduplicate()
        self._maybe_flush()
        self._fire_callbacks(new_df)

        return len(valid_rows)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(self) -> None:
        """
        Remove duplicate rows from the buffer based on the composite key
        ``(case_id, activity, timestamp)``, keeping the first occurrence.
        """
        self.buffer = (
            self.buffer
            .drop_duplicates(subset=["case_id", "activity", "timestamp"], keep="first")
            .reset_index(drop=True)
        )
