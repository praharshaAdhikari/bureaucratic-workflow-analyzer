"""
stream_generator.py
-------------------
Synthetic Event Stream Generator for the Digital Twin Live Sync feature.

Produces realistic, time-stamped event streams using the same SimPy engine as
DigitalTwin, enabling offline prototyping and integration testing of the
ingestion and synchronization modules without requiring a live process feed.

Importable as:
    from stream_generator import SyntheticStreamGenerator
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from digital_twin import DigitalTwin

if TYPE_CHECKING:
    from stream_ingestion import StreamIngestionPipeline

logger = logging.getLogger(__name__)

# Required schema columns for synthetic output
_SCHEMA_COLUMNS = ["case_id", "activity", "timestamp", "resource", "role", "lifecycle"]


class SyntheticStreamGenerator:
    """
    Generates synthetic event streams from a fitted DigitalTwin using SimPy.

    Events are sorted by timestamp and emitted sequentially, enabling
    realistic replay into a StreamIngestionPipeline for end-to-end testing.

    Parameters
    ----------
    twin : DigitalTwin
        A fitted DigitalTwin instance used to generate synthetic traces.
    arrival_rate_s : float
        Mean inter-arrival time in seconds between cases (Poisson process).
    speed_factor : float, optional
        Scales simulated time gaps between events. Values > 1.0 compress time,
        values < 1.0 expand it. Must be positive. Defaults to 1.0.
    split_ratio : float, optional
        Fraction of cases used as the training split when validating against
        real data. Must be in (0.0, 1.0) exclusive. Defaults to 0.8.

    Raises
    ------
    ValueError
        If ``speed_factor <= 0.0`` or ``split_ratio`` is outside (0.0, 1.0).
    """

    def __init__(
        self,
        twin: DigitalTwin,
        arrival_rate_s: float,
        speed_factor: float = 1.0,
        split_ratio: float = 0.8,
    ) -> None:
        if speed_factor <= 0.0:
            raise ValueError("speed_factor must be positive")
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio must be in (0.0, 1.0)")

        self.twin = twin
        self.arrival_rate_s = arrival_rate_s
        self.speed_factor = speed_factor
        self.split_ratio = split_ratio

        # Internal ordered event list (populated by start() / reset())
        self._events: list[dict] = []
        self._cursor: int = 0  # index of next event to emit
        self._n_cases: int = 0  # last requested n_cases (for reset)
        self._seed: int | None = None  # last seed used (for reset)

    # ------------------------------------------------------------------
    # Stream generation
    # ------------------------------------------------------------------

    def start(self, n_cases: int, n_jobs: int = -1) -> None:
        """
        Generate a synthetic Event_Log for ``n_cases`` cases and prepare
        the stream for sequential emission.

        Uses ``DigitalTwin.simulate`` (parallel, all CPU cores by default)
        when no seed is set, falling back to ``_simulate_single`` when a
        seed is active so that the RNG state is applied deterministically.

        Parameters
        ----------
        n_cases : int
            Number of synthetic cases to generate.
        n_jobs : int, optional
            Number of parallel workers passed to ``DigitalTwin.simulate``.
            ``-1`` = all CPU cores. Ignored when a seed is active (single-
            threaded path is used for reproducibility). Defaults to ``-1``.
        """
        self._n_cases = n_cases
        self._n_jobs = n_jobs
        self._generate(n_cases, seed=self._seed)

    def _generate(self, n_cases: int, seed: int | None = None) -> None:
        """Internal: (re-)generate the event list, optionally with a seed.

        When *seed* is ``None`` the parallel ``DigitalTwin.simulate`` path is
        used (``n_jobs`` workers).  When a seed is provided the single-threaded
        ``_simulate_single`` path is used so the RNG override is applied
        deterministically before simulation starts.
        """
        # Warn if training split is small
        train_cases = int(n_cases * self.split_ratio)
        if train_cases < 50:
            logger.warning(
                "Training split contains fewer than 50 cases; "
                "fidelity metrics may be unreliable"
            )

        if seed is not None:
            # Seeded path: override RNG then run single-threaded for reproducibility
            self.twin.rng = np.random.default_rng(seed)
            sim_df = self.twin._simulate_single(
                n_cases=n_cases,
                arrival_rate_s=self.arrival_rate_s,
            )
        else:
            # Unseeded path: use parallel simulate() for maximum throughput
            n_jobs = getattr(self, "_n_jobs", -1)
            sim_df = self.twin.simulate(
                n_cases=n_cases,
                arrival_rate_s=self.arrival_rate_s,
                n_jobs=n_jobs,
            )

        sim_df = self._ensure_schema(sim_df)
        sim_df = sim_df.sort_values("timestamp").reset_index(drop=True)

        self._events = sim_df.to_dict(orient="records")
        self._cursor = 0

    def _ensure_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required schema columns exist with correct types."""
        # Add lifecycle default for synthetic events
        if "lifecycle" not in df.columns:
            df = df.copy()
            df["lifecycle"] = "complete"
        else:
            df = df.copy()
            df["lifecycle"] = df["lifecycle"].fillna("complete")

        # Fill any other missing schema columns
        for col in ("resource", "role"):
            if col not in df.columns:
                df[col] = "UNKNOWN"
            else:
                df[col] = df[col].fillna("UNKNOWN")

        # Ensure case_id and activity are strings
        df["case_id"] = df["case_id"].astype(str)
        df["activity"] = df["activity"].astype(str)

        # Ensure timestamp is UTC-aware
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        return df[_SCHEMA_COLUMNS + [c for c in df.columns if c not in _SCHEMA_COLUMNS]]

    # ------------------------------------------------------------------
    # Emission interface
    # ------------------------------------------------------------------

    def get_next_event(self) -> dict | None:
        """
        Return the next event record in timestamp order, or ``None`` when
        the stream is exhausted.
        """
        if self._cursor >= len(self._events):
            return None
        event = self._events[self._cursor]
        self._cursor += 1
        return event

    def emit_batch(self, n: int) -> list[dict]:
        """
        Return up to ``n`` consecutive events from the stream.

        Returns an empty list if the stream is already exhausted.

        Parameters
        ----------
        n : int
            Maximum number of events to return.
        """
        if self.is_exhausted:
            return []
        batch = self._events[self._cursor: self._cursor + n]
        self._cursor += len(batch)
        return batch

    @property
    def is_exhausted(self) -> bool:
        """``True`` when all generated events have been emitted."""
        return self._cursor >= len(self._events)

    def reset(self, seed: int | None = None) -> None:
        """
        Re-generate the synthetic Event_Log from scratch.

        Using the same seed on two independent instances (each with their own
        deep-copied twin) produces bit-identical event sequences.

        Parameters
        ----------
        seed : int | None, optional
            RNG seed for reproducible generation. ``None`` = non-deterministic.

        Raises
        ------
        RuntimeError
            If ``start()`` has not been called yet (``_n_cases`` is 0).
        """
        if self._n_cases == 0:
            raise RuntimeError(
                "reset() called before start(). Call start(n_cases=N) first."
            )
        self._seed = seed
        self._generate(self._n_cases, seed=seed)

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    def get_wall_clock_delay(self, event: dict) -> float:
        """
        Compute the real seconds to wait before emitting ``event``.

        Calculated as the ``sim_time`` gap to the previous event divided by
        ``speed_factor``. Returns ``0.0`` for the first event.

        Parameters
        ----------
        event : dict
            An event record (must contain ``sim_time`` key).
        """
        idx = None
        for i, e in enumerate(self._events):
            if e is event:
                idx = i
                break

        if idx is None or idx == 0:
            return 0.0

        prev_sim_time = self._events[idx - 1].get("sim_time", 0.0)
        curr_sim_time = event.get("sim_time", 0.0)
        gap = max(0.0, curr_sim_time - prev_sim_time)
        return gap / self.speed_factor

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    def replay_into(self, pipeline: "StreamIngestionPipeline", batch_size: int = 50) -> int:
        """
        Feed all remaining events into ``pipeline`` via ``push_batch``.

        Parameters
        ----------
        pipeline : StreamIngestionPipeline
            Target ingestion pipeline.
        batch_size : int, optional
            Number of events per ``push_batch`` call. Defaults to 50.

        Returns
        -------
        int
            Total count of successfully ingested events. Returns ``0``
            immediately if the stream is already exhausted.
        """
        if self.is_exhausted:
            return 0

        total = 0
        while not self.is_exhausted:
            batch = self.emit_batch(batch_size)
            if not batch:
                break
            total += pipeline.push_batch(batch)

        return total
