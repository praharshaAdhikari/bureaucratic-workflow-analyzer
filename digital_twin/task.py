"""
Task module for workflow simulation.

This module contains the Task class that represents individual tasks
in a workflow with their properties and status.
"""

from digital_twin.case import Case


class Task:
    """
    Represents a single task in a workflow.

    Attributes:
        name (str): The name/type of the task
        duration (float): Duration of the task in minutes
        status (str): Current status of the task ('Pending', 'Completed', 'Rejected')
    """

    def __init__(self, name: str, duration: float, case: Case = None) -> None:
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

    def __repr__(self) -> str:
        """String representation of the task."""
        return f"Task(name='{self.name}', duration={self.duration}, status='{self.status}')"
