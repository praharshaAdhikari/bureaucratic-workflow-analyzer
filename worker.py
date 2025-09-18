"""
Worker module for workflow simulation.

This module contains the Worker class that represents workers
who perform tasks in the simulation environment.
"""

import simpy
from typing import Generator
from task import Task


class Worker:
    """
    Represents a worker who can perform tasks in the simulation.
    
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

    def perform_task(self, task: Task, env: simpy.Environment, speed: float = 1.0, expected_status: str = 'Completed') -> Generator:
        """
        Perform a task in the simulation environment.
        
        Args:
            task: The task to perform
            env: The simulation environment
            speed: Speed multiplier for the simulation
            expected_status: The expected final status of the task
            
        Yields:
            SimPy timeout event for the task duration
        """
        print(f"Worker {self.name} is performing task: {task.name} at time {env.now}")
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