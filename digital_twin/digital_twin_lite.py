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
            self.logs = pd.read_csv(filename)
            # Validating present columns
            required_columns = ["CaseID", "Task", "Worker", "StartTime", "EndTime"]
            if not all(col in self.logs.columns for col in required_columns):
                missing_cols = [
                    col for col in required_columns if col not in self.logs.columns
                ]
                print(f"Error: Missing required columns in log file: {missing_cols}")
                self.logs = None
                return False

            try:
                self.logs["StartTime"] = pd.to_datetime(self.logs["StartTime"])
                self.logs["EndTime"] = pd.to_datetime(self.logs["EndTime"])
            except Exception as e:
                print(f"Error converting StartTime/EndTime to datetime: {e}")
                self.logs = None
                return False

            try:
                self.logs["Duration"] = (
                    self.logs["EndTime"] - self.logs["StartTime"]
                ).dt.total_seconds() / 60.0
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
            if len(durations) < 5:
                print(
                    f"Warning: Not enough data for task '{task}' to fit distribution.  Using empirical samples."
                )
                self.task_duration_distributions[task] = (
                    "empirical",
                    durations.tolist(),
                )
                continue
            # Fit a lognormal distribution (common for durations)
            try:
                shape, loc, scale = stats.lognorm.fit(durations, floc=0)
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
                    durations.tolist(),
                )  # fallback

    def estimate_arrival_rate(self):
        if self.logs is None:
            print("Error: Logs not loaded. Call load_logs() first.")
            return

        start_times = pd.to_datetime(self.logs["StartTime"])
        end_times = pd.to_datetime(self.logs["EndTime"])
        duration_minutes = (end_times.max() - start_times.min()).total_seconds() / 60.0
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
