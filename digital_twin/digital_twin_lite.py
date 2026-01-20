import os

import numpy as np
import pandas as pd
from scipy import stats


class DigitalTwinLite:
    """
    Estimates parameters for a simple stochastic digital twin from logs.csv.
    """

    def __init__(self, logs_path="logs.csv"):
        self.logs_path = logs_path
        self.logs = None
        self.task_duration_distributions = {}  # task_name -> (dist_type, dist_params)
        self.arrival_rate = 0.0

    def load_logs(self, filename="logs.csv"):
        try:
            self.logs = pd.read_csv(filename, sep=';')
            # Validating present columns
            required_columns = ["CaseID", "Task", "Worker", "StartTime", "EndTime"]
            if not all(col in self.logs.columns for col in required_columns):
                missing_cols = [
                    col for col in required_columns if col not in self.logs.columns
                ]
                print(f"Error: Missing required columns in log file: {missing_cols}")
                self.logs = None
                return False

            # ------------------------------------------------------------------
            # Original (datetime-only) conversion kept here commented for reference
            # try:
            #     self.logs["StartTime"] = pd.to_datetime(self.logs["StartTime"])
            #     self.logs["EndTime"] = pd.to_datetime(self.logs["EndTime"])
            # except Exception as e:
            #     print(f"Error converting StartTime/EndTime to datetime: {e}")
            #     self.logs = None
            #     return False
            # ------------------------------------------------------------------
            # New: accept either numeric (minutes since start) or datetime strings.
            # Try to parse each column; if dtype is numeric we'll treat it as numeric
            # and compute durations accordingly. This lets the repo work with
            # dummy numeric logs (minutes) and real timestamped logs.

            def _parse_time_column(series):
                # If already numeric, return float Series and type 'numeric'
                if pd.api.types.is_numeric_dtype(series.dtype):
                    return series.astype(float), "numeric"

                # Try full datetime parsing. If it fails, attempt numeric coercion.
                try:
                    parsed = pd.to_datetime(series, errors="raise")
                    return parsed, "datetime"
                except Exception:
                    coerced = pd.to_numeric(series, errors="coerce")
                    if coerced.isna().any():
                        raise ValueError("StartTime/EndTime column could not be parsed as datetime or numeric")
                    return coerced.astype(float), "numeric"

            try:
                start_parsed, start_type = _parse_time_column(self.logs["StartTime"])
                end_parsed, end_type = _parse_time_column(self.logs["EndTime"])

                # Keep datetimes as datetimes; otherwise force numeric float columns.
                if start_type == "datetime" and end_type == "datetime":
                    self.logs["StartTime"] = start_parsed
                    self.logs["EndTime"] = end_parsed
                else:
                    # At least one side is numeric: make both numeric floats
                    self.logs["StartTime"] = start_parsed.astype(float)
                    self.logs["EndTime"] = end_parsed.astype(float)

            except Exception as e:
                print(f"Error parsing StartTime/EndTime: {e}")
                self.logs = None
                return False

            # ------------------------------------------------------------------
            # Original: datetime-only duration calculation (commented out)
            # try:
            #     self.logs["Duration"] = (
            #         self.logs["EndTime"] - self.logs["StartTime"]
            #     ).dt.total_seconds() / 60.0
            # except Exception as e:
            #     print(f"Error calculating duration: {e}")
            #     self.logs = None
            #     return False
            # ------------------------------------------------------------------
            # Compute numeric Duration (minutes). If columns are datetimes, use
            # total_seconds/60. Otherwise assume numeric timestamps are already
            # in minutes and simply subtract.
            try:
                if pd.api.types.is_datetime64_any_dtype(self.logs["StartTime"].dtype):
                    self.logs["Duration"] = (
                        self.logs["EndTime"] - self.logs["StartTime"]
                    ).dt.total_seconds() / 60.0
                else:
                    self.logs["Duration"] = (
                        self.logs["EndTime"].astype(float) - self.logs["StartTime"].astype(float)
                    )

                # Normalize/validate numeric dtype
                self.logs["Duration"] = pd.to_numeric(self.logs["Duration"], errors="coerce").astype(float)
                # If we can't compute any durations, fail early
                if self.logs["Duration"].isna().all():
                    raise ValueError("Could not compute Duration from StartTime/EndTime columns")

            except Exception as e:
                print(f"Error calculating duration: {e}")
                self.logs = None
                return False

        except FileNotFoundError:
            print(f"Error: Log file not found at {filename}")
            self.logs = None
            return False  # FAILURE
        return True  # SUCCESS

    def analyze_durations(self):
        if self.logs is None:
            print("Error: Logs not loaded. Call load_logs() first.")
            return

        grouped = self.logs.groupby("Task")["Duration"]
        for task, durations in grouped:
            durations = durations.dropna().values
            
            # Filter out non-positive durations (lognorm requires x > 0)
            valid_durations = durations[durations > 0]
            
            if len(valid_durations) < len(durations):
                filtered_count = len(durations) - len(valid_durations)
                print(
                    f"Note: Filtered out {filtered_count} non-positive duration(s) for task '{task}'"
                )
            
            if len(valid_durations) < 5:
                print(
                    f"Warning: Not enough valid data for task '{task}' to fit distribution. Using empirical samples."
                )
                self.task_duration_distributions[task] = (
                    "empirical",
                    valid_durations.tolist() if len(valid_durations) > 0 else [10.0],
                )
                continue
            
            # Fit a lognormal distribution (common for durations)
            try:
                shape, loc, scale = stats.lognorm.fit(valid_durations, floc=0)
                self.task_duration_distributions[task] = (
                    "lognorm",
                    (shape, loc, scale),
                )
                print(
                    f"Task '{task}': Fitted lognorm(shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f})"
                )
            except Exception as e:
                print(
                    f"Warning: Could not fit lognorm for task '{task}'. Using empirical samples. {e}"
                )
                self.task_duration_distributions[task] = (
                    "empirical",
                    valid_durations.tolist(),
                )  # fallback

    def estimate_arrival_rate(self):
        if self.logs is None:
            print("Error: Logs not loaded. Call load_logs() first.")
            return

        # ------------------------------------------------------------------
        # Original datetime-only behavior kept here commented for reference
        # start_times = pd.to_datetime(self.logs["StartTime"])
        # end_times = pd.to_datetime(self.logs["EndTime"])
        # duration_minutes = (end_times.max() - start_times.min()).total_seconds() / 60.0
        # ------------------------------------------------------------------
        # New: handle numeric or datetime Start/End types. If datetimes are used
        # then we compute total_seconds/60; otherwise treat numeric values as minutes.
        if pd.api.types.is_datetime64_any_dtype(self.logs["StartTime"].dtype) and pd.api.types.is_datetime64_any_dtype(
            self.logs["EndTime"].dtype
        ):
            start_times = pd.to_datetime(self.logs["StartTime"])
            end_times = pd.to_datetime(self.logs["EndTime"])
            duration_minutes = (end_times.max() - start_times.min()).total_seconds() / 60.0
        else:
            start_vals = pd.to_numeric(self.logs["StartTime"], errors="coerce").astype(float)
            end_vals = pd.to_numeric(self.logs["EndTime"], errors="coerce").astype(float)
            if end_vals.isna().all() or start_vals.isna().all():
                print("Error: Could not compute arrival_rate because StartTime/EndTime could not be parsed as numeric or datetime")
                return
            duration_minutes = float(end_vals.max() - start_vals.min())
        n_cases = self.logs["CaseID"].nunique()
        self.arrival_rate = n_cases / max(1.0, duration_minutes)
        print(
            f"Estimated arrival rate: {self.arrival_rate:.3f} cases/minute"
        )  # cases / minute

    def sample_duration(self, task_name):
        """Samples a duration from the distribution for a given task."""
        dist_type, dist_params = self.task_duration_distributions.get(
            task_name, ("empirical", [10.0])
        )  # default to 10
        if dist_type == "lognorm":
            shape, loc, scale = dist_params
            return max(1.0, stats.lognorm.rvs(shape, loc=loc, scale=scale))
        elif dist_type == "empirical":
            return max(1.0, np.random.choice(dist_params))
        else:
            return 10.0
