"""
Case module for workflow simulation.

This module contains the Case class that represents individual cases
containing multiple tasks to be processed by workers.
"""

from typing import Generator, List

import simpy

from digital_twin.task import Task
from digital_twin.worker import Worker


class Case:
    """
    Represents a case containing multiple tasks to be processed.

    Attributes:
        id (int): Unique identifier for the case
        env (simpy.Environment): The simulation environment
        tasks (List[Task]): List of tasks in this case
        workers (List[Worker]): List of workers assigned to tasks
        status (str): Current status of the case ('Pending', 'Accepted', 'Rejected')
        priority (int): The priority of the case. Lower number means higher priority.
        creation_time (float): The time when the case was created.
    """

    def __init__(self, id: int, env: simpy.Environment) -> None:
        """
        Initialize a new Case.

        Args:
            id: Unique identifier for the case
            env: The simulation environment
        """
        self.id = id
        self.env = env
        self.tasks: List[Task] = []
        self.workers: List[Worker] = []
        self.status = "Pending"
        self.priority = 0
        self.creation_time = env.now
        self.completion_time = None

    def start_work(self) -> None:
        """Start processing work for this case."""
        print(f"Starting work for case {self.id} at time {self.env.now}")

    def assign_task(
        self,
        worker: Worker,
        task: Task,
        speed: float = 1.0,
        expected_status: str = "Completed",
    ) -> Generator:
        """
        Assign a task to a worker and wait for completion.

        Args:
            worker: The worker to assign the task to
            task: The task to be performed
            speed: Speed multiplier for the simulation
            expected_status: The expected final status of the task

        Yields:
            SimPy process for the worker performing the task
        """
        print(f"Assigning task {task.name} to worker {worker.name}")
        # record which worker is assigned to the task (used by reassign logic)
        try:
            task.assigned_worker_name = worker.name
        except Exception:
            # If Task doesn't expose attribute, ignore — older versions
            pass
        yield self.env.process(
            worker.perform_task(task, self.env, speed, expected_status)
        )
        # After task is complete, check if the case is complete
        self.check_completion()

    def check_completion(self) -> None:
        """
        Check if the case is complete and update its status accordingly.

        A case is complete when all tasks are no longer pending.
        A case is accepted only if all tasks are completed successfully.
        A case is rejected if any task is rejected or failed.
        """
        if all(task.status != "Pending" for task in self.tasks):
            # Case is complete when all tasks are non-pending
            # Case is Accepted only if ALL tasks are Completed
            # Case is Rejected if ANY task is Rejected or not Completed
            self.status = (
                "Accepted"
                if all(task.status == "Completed" for task in self.tasks)
                else "Rejected"
            )
            if self.status == "Accepted":
                self.completion_time = self.env.now
            print(
                f"Case {self.id} completed with status: {self.status} at time {self.env.now}"
            )
        # Note: Early rejection logic could be added here if needed
        # else:
        #     if any(task.status == 'Rejected' for task in self.tasks):
        #         self.status = 'Rejected'
        #         print(f"Case {self.id} marked as Rejected due to failed task at time {self.env.now}")

    def __repr__(self) -> str:
        """String representation of the case."""
        return f"Case(id={self.id}, status='{self.status}', tasks={len(self.tasks)})"
