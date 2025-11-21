"""
Workflow module for workflow simulation.

This module contains the Workflow class that orchestrates the entire
simulation process including reading logs, managing workers, and generating reports.
"""

import csv
import os
from typing import List, Optional

import pandas as pd
import simpy
from colorama import Fore, Style, init

from digital_twin.case import Case
from digital_twin.digital_twin_lite import DigitalTwinLite
from digital_twin.task import Task
from digital_twin.worker import Worker

# Initialize colorama for cross-platform colored output
init(autoreset=True)


class Workflow:
    """
    Main workflow orchestrator that manages the simulation process.

    Attributes:
        name (str): Name of the workflow
        env (simpy.Environment): The simulation environment
        logs (pd.DataFrame): DataFrame containing workflow logs
        tasks (Optional): Placeholder for tasks (currently unused)
        cases (List[Case]): List of cases in the workflow
        ideal_task_durations (pd.DataFrame): DataFrame with ideal task durations
        workers (List[Worker]): List of workers in the workflow
    """

    def __init__(self, name: str, env: simpy.Environment = simpy.Environment()) -> None:
        """
        Initialize a new Workflow.

        Args:
            name: Name of the workflow
            env: The simulation environment (defaults to new environment)
        """
        self.name = name
        self.env = env
        self.logs: Optional[pd.DataFrame] = None
        self.tasks = None
        self.cases: List[Case] = []
        self.ideal_task_durations: Optional[pd.DataFrame] = None
        self.workers: Optional[List[Worker]] = None
        self.digital_twin = DigitalTwinLite()

    def read_logs(self, filename: str = "logs.csv") -> pd.DataFrame | None:
        """
        Read workflow logs from a CSV file.

        Args:
            filename: Path to the CSV file containing logs

        Returns:
            DataFrame containing the processed logs
        """
        print(f"Reading logs for workflow: {self.name}")

        if self.digital_twin.load_logs(filename):
            self.logs = self.digital_twin.logs
            self.digital_twin.analyze_durations()
            self.digital_twin.estimate_arrival_rate()
            return self.logs
        else:
            print("Failed to load logs.")
            self.logs = None
            return None

    def prompt_ideal_durations(self, filename: str = "ideal_durations.csv") -> None:
        """
        Prompt user for ideal task durations and save to CSV.

        Args:
            filename: Path to save the ideal durations CSV file
        """
        if self.logs is None:
            print("Logs not loaded. Please run read_logs() first.")
            return

        tasks = self.logs["Task"].unique()
        durations = []
        print("Enter the ideal duration (in minutes) for each task:")

        for task in tasks:
            while True:
                try:
                    value = float(input(f"Task '{task}': "))
                    durations.append({"Task": task, "IdealDuration": value})
                    break
                except ValueError:
                    print("Please enter a valid number.")

        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Task", "IdealDuration"])
            writer.writeheader()
            writer.writerows(durations)
        print(f"Ideal durations saved to {filename}.")

        self.ideal_task_durations = self.read_ideal_durations(filename)

    def read_ideal_durations(
        self, filename: str = "ideal_durations.csv"
    ) -> pd.DataFrame:
        """
        Read ideal task durations from a CSV file.

        Args:
            filename: Path to the CSV file containing ideal durations

        Returns:
            DataFrame containing ideal durations, or empty DataFrame if file not found
        """
        if not os.path.exists(filename):
            print(
                f"{filename} not found. Running prompt_and_save_ideal_durations() first."
            )
            self.prompt_ideal_durations()
            return pd.DataFrame()

        df = pd.read_csv(filename)
        self.ideal_task_durations = df
        return df

    def define_workers(self) -> None:
        """Create worker objects from unique worker names in logs."""
        if self.logs is None:
            print("Logs not loaded. Please run read_logs() first.")
            return

        self.workers = [Worker(name, self.env) for name in self.logs["Worker"].unique()]

    def start(self, speed: float) -> None:
        """
        Start the workflow simulation.

        Args:
            speed: Speed multiplier for the simulation (higher = faster)
        """
        if self.logs is None:
            print("Logs not loaded. Please run read_logs() first.")
            return
        if self.workers is None:
            print("Workers not defined. Please run define_workers() first.")
            return

        print(f"Starting workflow: {self.name} at speed: {speed}x")

        # Process cases sequentially
        for case_id in self.logs["CaseID"].unique():
            case_ = Case(case_id, self.env)
            self.cases.append(case_)
            tasks = self.logs[self.logs["CaseID"] == case_id]
            case_.start_work()

            for _, task_row in tasks.iterrows():
                task_name = task_row["Task"]
                # Sample Duration
                duration = self.digital_twin.sample_duration(
                    task_name
                )  # Use sampled duration
                task = Task(task_name, duration)  # Pass name, duration

                worker = next(
                    (w for w in self.workers if w.name == task_row["Worker"]), None
                )
                if worker:
                    case_.tasks.append(task)
                    case_.workers.append(worker)

            # Then start all task processes
            for i, (_, task_row) in enumerate(tasks.iterrows()):
                task = case_.tasks[i]
                worker = case_.workers[i]
                if worker:
                    self.env.process(
                        case_.assign_task(worker, task, speed, task_row["Status"])
                    )

    def generate_report(self, filename: str = "reports/workflow_report.txt") -> None:
        """Generate a colored report showing case and task results, output to both console and file."""
        if not self.cases:
            print("No cases processed. Please run start() first.")
            return
        if self.ideal_task_durations is None:
            print(
                "Ideal task durations not loaded. Please run read_ideal_durations() first."
            )
            return

        # Open file for writing
        with open(filename, "w", encoding="utf-8") as f:

            def write_both(text: str, colored_text: Optional[str] = None):
                """Write to both console (with colors) and file (without colors)."""
                if colored_text:
                    print(colored_text)  # Console with colors
                    f.write(text + "\n")  # File without colors
                else:
                    print(text)  # Console
                    f.write(text + "\n")  # File

            header = f"Generating report for workflow: {self.name}"
            write_both(header)
            write_both("=" * len(header))
            write_both("")

            for case in self.cases:
                # Color case info based on status
                case_text = f"Case ID: {case.id}, Status: {case.status}"
                if case.status == "Accepted":
                    write_both(case_text, f"{Fore.GREEN}{case_text}{Style.RESET_ALL}")
                elif case.status == "Rejected":
                    write_both(case_text, f"{Fore.RED}{case_text}{Style.RESET_ALL}")
                else:
                    write_both(case_text)

                for task in case.tasks:
                    # Color task info based on status
                    task_text = f"  Task: {task.name}, Duration: {task.duration} mins, Status: {task.status}"
                    if task.status == "Completed":
                        write_both(
                            task_text,
                            f"  Task: {task.name}, Duration: {task.duration} mins, Status: {task.status}{Style.RESET_ALL}",
                        )
                    elif task.status == "Rejected":
                        write_both(
                            task_text,
                            f"  {Fore.RED}Task: {task.name}, Duration: {task.duration} mins, Status: {task.status}{Style.RESET_ALL}",
                        )
                    else:
                        write_both(task_text)

                    # Performance warnings/praise with colors
                    if task.status != "Rejected":
                        ideal_duration = self.ideal_task_durations[
                            self.ideal_task_durations["Task"] == task.name
                        ]["IdealDuration"].values

                        if len(ideal_duration) > 0:
                            ratio = task.duration / ideal_duration[0]
                            if ratio > 1.5:
                                warning_text = f"    WARNING: Task {task.name} took {ratio * 100:.1f}% of the ideal duration."
                                write_both(
                                    warning_text,
                                    f"    {Fore.YELLOW}{warning_text}{Style.RESET_ALL}",
                                )
                            elif ratio < 0.5:
                                praise_text = f"    CONGRATS: Task {task.name} took {ratio * 100:.1f}% of the ideal duration."
                                write_both(
                                    praise_text,
                                    f"    {Fore.GREEN}{praise_text}{Style.RESET_ALL}",
                                )

                write_both("")  # Empty line between cases

            # Summary statistics
            total_cases = len(self.cases)
            accepted_cases = len([c for c in self.cases if c.status == "Accepted"])
            rejected_cases = len([c for c in self.cases if c.status == "Rejected"])

            write_both("SUMMARY:")
            write_both("-" * 20)
            write_both(f"Total Cases: {total_cases}")
            write_both(
                f"Accepted Cases: {accepted_cases} ({accepted_cases / total_cases * 100:.1f}%)"
            )
            write_both(
                f"Rejected Cases: {rejected_cases} ({rejected_cases / total_cases * 100:.1f}%)"
            )

        print(f"\nReport saved to: {filename}")
