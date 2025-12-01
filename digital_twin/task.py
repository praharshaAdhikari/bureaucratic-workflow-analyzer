"""
Task module for workflow simulation.

This module contains the Task class that represents individual tasks
in a workflow with their properties and status.
"""


from typing import Optional


class Task:
    """
    Represents a single task in a workflow.

    Attributes:
        name (str): The name/type of the task
        duration (float): Duration of the task in minutes
        status (str): Current status of the task ('Pending', 'Completed', 'Rejected')
    """

    def __init__(self, name: str, duration: float, case=None) -> None:
        """
        Initialize a new Task.

        Args:
            name: The name/type of the task
            duration: Duration of the task in minutes
        """
        self.name = name
        self.duration = duration
        self.status = "Pending"
        self.case = case
        # Name of the worker currently assigned to this task (if any)
        self.assigned_worker_name: Optional[str] = None

    def __repr__(self) -> str:
        """String representation of the task."""
        return f"Task(name='{self.name}', duration={self.duration}, status='{self.status}', assigned_worker={self.assigned_worker_name})"
