"""
state_sync.py
-------------
State Synchronization Layer for the Digital Twin Live Sync feature.

Keeps the in-memory DigitalTwin model aligned with the real process by
tracking active cases, detecting drift between the live model and incoming
data, and triggering re-fits when divergence exceeds configurable thresholds.

Importable as:
    from state_sync import StateSyncLayer
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

import validation
from digital_twin import DigitalTwin
from stream_ingestion import StreamIngestionPipeline

logger = logging.getLogger(__name__)


class StateSyncLayer:
    """
    Synchronizes an in-memory DigitalTwin with a live StreamIngestionPipeline.

    Maintains a registry of active cases, detects model drift against a sliding
    window of recent events, and triggers refits when divergence exceeds
    configurable thresholds.

    Parameters
    ----------
    twin : DigitalTwin
        A fitted DigitalTwin instance. Replaced in-memory on successful refit.
    pipeline : StreamIngestionPipeline
        The ingestion pipeline whose buffer is used for drift detection and refit.
    output_dir : str | Path
        Root output directory; checkpoints are written to
        ``output_dir / dataset_name /``.
    dataset_name : str
        Dataset identifier used for checkpoint file naming.
    jsd_threshold : float, optional
        Jensen-Shannon divergence threshold above which drift is flagged.
        Defaults to 0.10.
    l1_threshold : float, optional
        Transition matrix L1 distance threshold above which drift is flagged.
        Defaults to 0.25.
    """

    def __init__(
        self,
        twin: DigitalTwin,
        pipeline: StreamIngestionPipeline,
        output_dir: str | Path,
        dataset_name: str,
        jsd_threshold: float = 0.10,
        l1_threshold: float = 0.25,
    ) -> None:
        self.twin = twin
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.jsd_threshold = jsd_threshold
        self.l1_threshold = l1_threshold

        # Registry: case_id → {last_activity: str, last_timestamp: pd.Timestamp, lifecycle: str}
        self._registry: dict[str, dict] = {}

        # Refit history: list of dicts recording each completed refit event
        self._refit_history: list[dict] = []

        # Wire: automatically call update_registry after every ingestion event
        self.pipeline.register_callback(self.update_registry)

    # ------------------------------------------------------------------
    # Methods to be implemented in subsequent tasks (6, 7, 8)
    # ------------------------------------------------------------------

    def update_registry(self, events: pd.DataFrame) -> None:
        """Update the active case registry from a batch of new events."""
        if events.empty:
            return
        for case_id, group in events.groupby("case_id"):
            latest = group.loc[group["timestamp"].idxmax()]
            lifecycle = str(latest.get("lifecycle", "UNKNOWN")).lower()
            if lifecycle in ("complete", "completed"):
                self._registry.pop(str(case_id), None)
            else:
                self._registry[str(case_id)] = {
                    "last_activity": str(latest["activity"]),
                    "last_timestamp": latest["timestamp"],
                    "lifecycle": str(latest.get("lifecycle", "UNKNOWN")),
                }

        # Auto-trigger drift-based refit
        try:
            drift = self.check_drift()
            if drift.get("drifted", False):
                logger.info(
                    "Drift detected (jsd=%.4f, l1=%.4f) — triggering auto-refit.",
                    drift["jsd"], drift["l1"],
                )
                self.force_refit(pd.DataFrame(), trigger="drift")
        except Exception as e:
            logger.warning("Drift check after registry update failed: %s", e)

    def get_active_cases(self) -> pd.DataFrame:
        """Return a DataFrame of all currently active cases."""
        if not self._registry:
            return pd.DataFrame(columns=["case_id", "last_activity", "last_timestamp", "lifecycle"])
        rows = [
            {"case_id": cid, **state}
            for cid, state in self._registry.items()
        ]
        return pd.DataFrame(rows)[["case_id", "last_activity", "last_timestamp", "lifecycle"]]

    def get_case_state(self, case_id: str) -> dict | None:
        """Return the state dict for a case, or None if not active."""
        return self._registry.get(case_id)

    def check_drift(self, window_size: int = 500) -> dict:
        """Compute drift metrics against a sliding window of recent events."""
        buffer = self.pipeline.buffer
        n = len(buffer)

        # Return early with safe defaults when buffer is empty
        if n == 0:
            return {
                "drifted": False,
                "jsd": 0.0,
                "l1": 0.0,
                "triggered_metrics": [],
                "warning": "insufficient_data",
            }

        result = {}
        if n < window_size:
            result["warning"] = "insufficient_data"

        window = buffer.iloc[-min(n, window_size):]

        # Generate a reference simulation from the twin to compare against
        try:
            ref_df = self.twin._simulate_single(n_cases=max(50, len(window) // 5))
        except Exception:
            ref_df = self.twin.simulate(n_cases=max(50, len(window) // 5), n_jobs=1)

        jsd = validation.activity_freq_jsd(window, ref_df)
        l1 = validation.transition_matrix_l1(window, ref_df)

        triggered = []
        if jsd > self.jsd_threshold:
            triggered.append("activity_freq_jsd")
        if l1 > self.l1_threshold:
            triggered.append("transition_matrix_l1")

        result.update({
            "drifted": len(triggered) > 0,
            "jsd": jsd,
            "l1": l1,
            "triggered_metrics": triggered,
        })
        return result

    def force_refit(self, event_log: pd.DataFrame, trigger: str = "manual") -> dict:
        """Trigger an immediate refit with the provided event log."""
        # Compute drift metrics before refit for history
        drift_result = {}
        try:
            drift_result = self.check_drift()
        except Exception:
            pass
        jsd_before = drift_result.get("jsd", float("nan"))
        l1_before = drift_result.get("l1", float("nan"))

        # Merge provided event_log with pipeline buffer
        buffer = self.pipeline.buffer
        if not buffer.empty and not event_log.empty:
            merged = pd.concat([event_log, buffer], ignore_index=True).drop_duplicates(
                subset=["case_id", "activity", "timestamp"], keep="first"
            )
        elif not buffer.empty:
            merged = buffer.copy()
        else:
            merged = event_log.copy()

        previous_twin = self.twin
        validation_metrics = {}
        validation_passed = False

        try:
            new_twin = DigitalTwin(seed=42)
            new_twin.fit(merged)
            self.twin = new_twin

            # Save checkpoint
            ts = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
            out_dir = self.output_dir / self.dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / f"digital_twin_sync_{ts}.pkl"
            joblib.dump(self.twin, ckpt_path)
            logger.info("Saved checkpoint to %s", ckpt_path)

            # Run validation
            try:
                sim_df = self.twin._simulate_single(n_cases=min(200, max(50, len(merged) // 10)))
                validation_metrics = validation.validate(merged, sim_df, verbose=False)
                validation_passed = bool(validation_metrics.get("overall_pass", False))
            except Exception as e:
                logger.warning("Validation after refit failed: %s", e)

            logger.info("Refit completed. validation_passed=%s", validation_passed)

        except Exception as e:
            logger.error("Refit failed: %s — retaining previous twin.", e)
            self.twin = previous_twin
            self._refit_history.append({
                "timestamp": pd.Timestamp.now("UTC"),
                "trigger": trigger,
                "jsd_before": jsd_before,
                "l1_before": l1_before,
                "validation_passed": False,
                "error": str(e),
            })
            return validation_metrics

        # Record successful refit in history
        self._refit_history.append({
            "timestamp": pd.Timestamp.now("UTC"),
            "trigger": trigger,
            "jsd_before": jsd_before,
            "l1_before": l1_before,
            "validation_passed": validation_passed,
        })

        return validation_metrics
