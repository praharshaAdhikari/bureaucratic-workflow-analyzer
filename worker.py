"""
Worker module for workflow simulation.

This module contains the Worker class that represents workers
who perform tasks in the simulation environment.
"""

from typing import Generator

import simpy

from task import Task


class Worker:
    """
    Represents a worker who can perform tasks in the simulation.
    Each worker has an associated simpy.Resource to ensure exclusivity.

    Attributes:
        name (str): The name/identifier of the worker
        env (simpy.Environment): The simulation environment
    """

    def __init__(self, name: str, env: simpy.Environment) -> None:
        """
        Initialize a new Worker.

        Args:
            name: The name/identifier of the worker
            env: The simulation environment
        """
        self.name = name
        self.env = env
        self.resource = simpy.Resource(
            env, capacity=1
        )  # Modelling exclusivity. capacity 1 means 1 task at a time

    def perform_task(
        self,
        task: Task,
        env: simpy.Environment,
        speed: float = 1.0,
        expected_status: str = "Completed",
    ) -> Generator:
        """
        Acquire exclusive access to the worker, simulate work, then release.

        Args:
            task: The task to perform
            env: The simulation environment
            speed: Speed multiplier for the simulation
            expected_status: The expected final status of the task

        Yields:
            SimPy timeout event for the task duration
        """
        print(
            f"Worker {self.name} requesting to perform task: {task.name} at time {self.env.now}"
        )
        with self.resource.request() as req:
            yield req  # Modelling semaphore like behaviour.
            print(
                f"Worker {self.name} started task: {task.name} at time {self.env.now}"
            )
            yield env.timeout(task.duration / speed)
            task.status = expected_status
            print(f"Worker {self.name} completed task: {task.name} at time {env.now}")
            print(f"Task {task.name} marked as {expected_status}.")

    def complete_task(self, task: Task, status: str) -> None:
        """
        Mark a task as completed with the given status.

        Args:
            task: The task to mark as completed
            status: The final status of the task
        """
        task.status = status
        print(f"Task {task.name} marked as {status}.")

    def __repr__(self) -> str:
        """String representation of the worker."""
        return f"Worker(name='{self.name}')"
