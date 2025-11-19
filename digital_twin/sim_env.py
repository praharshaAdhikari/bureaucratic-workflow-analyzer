import os
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import simpy
from case import Case
from digital_twin_lite import DigitalTwinLite
from scipy import stats
from task import Task
from worker import Worker
from workflow import Workflow


class SimEnv:
    def __init__(
        self,
        twin_params,
        logs_path="data_collection/data/logs.csv",
        decision_interval=5.0,
    ):
        self.params = twin_params
        self.logs_path = logs_path
        self.decision_interval = decision_interval
        self.env = None
        self.workers = {}
        self.digital_twin = None
        self.arrival_process = None
        self.case_id_counter = 1000
        self.observation_space = None
        self.action_space = None
        self.state = None
        self.workflow = None

        self._define_spaces()

    def reset(self):
        self.env = simpy.Environment()
        self.workflow = Workflow("Workflow", self.env)
        self._setup_workers()
        self._load_digital_twin()
        self._start_arrival_process()

        self.state = self._get_observation()
        return self.state

    def step(self, action):
        self._apply_action(action)
        self.env.run(until=self.env.now + self.decision_interval)
        self.state = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        return self.state, reward, done, info

    def _setup_workers(self):
        """Set up workers by creating a Workflow object, and using its workers."""
        .
        if self.workflow:
            self.workflow.define_workers()
            self.workers = {worker.name: worker for worker in self.workflow.workers}
            print(f"Created workers: {list(self.workers.keys())}")
        else:
            print("Warning: Workflow not initialized. Cannot setup workers.")

    def _load_digital_twin(self):
        """Loads the logs and sets up the digital twin components using the Workflow."""
        if self.workflow:
            self.workflow.read_logs(self.logs_path)
            self.digital_twin = (
                self.workflow.digital_twin
            )
            if self.digital_twin:
                print("Digital Twin Loaded.")
            else:
                print("DigitalTwinLite failed to load")
        else:
            print(
                "Warning: Workflow not initialized. Cannot load logs or set up digital twin."
            )

    def _start_arrival_process(self):
        if self.workflow and self.workflow.digital_twin.arrival_rate > 0:
            self.arrival_process = self.env.process(self._generate_cases())

    def _generate_cases(self):
        """Generates new cases based on the estimated arrival rate."""
        while True:
            interarrival_time = np.random.exponential(
                1.0 / self.workflow.digital_twin.arrival_rate
            )
            yield self.env.timeout(interarrival_time)
            self.case_id_counter += 1
            self.env.process(
                self._create_case(self.case_id_counter)
            )

    def _create_case(self, case_id):
        """Creates a new case with tasks using the Workflow."""
        print(f"Creating new case {case_id} at time {self.env.now:.2f}")

        task_sequence = {
            "ReceiveApplication": ["Review"],
            "Review": ["Approval"],
            "Approval": [],
            "Finalize": [],
            "PaymentProcessing": [],
        }
        current_task = "ReceiveApplication"
        while current_task:
            task_name = current_task
            # 1. Sample Duration
            # 2. Assign a Worker (using round robin cuz it good)
            # # 3. Run task
            duration = self.workflow.digital_twin.sample_duration(task_name)
            task = Task(task_name, duration)

            worker_name = self.params.get("workers", [])[
                (case_id - 1) % len(self.params.get("workers", []))
            ]
            worker = self.workers.get(worker_name)
            if not worker:
                print(
                    f"Warning: No worker found named {worker_name} for task {task_name}"
                )
                break

            print(
                f"Case {case_id}: starting task '{task_name}' for {duration:.2f} minutes"
            )
            yield self.env.process(self._perform_task(worker, task))
            if task.status != "Completed":
                print(
                    f"Case {case_id}: Task {task.name} was not completed, breaking case workflow"
                )
                break

            next_tasks = task_sequence.get(task_name, [])
            if not next_tasks:
                print(f"Case {case_id} completed at time {self.env.now:.2f}")
                break
            current_task = next_tasks[0]

    def _perform_task(self, worker: Worker, task: Task):
        """Worker performs a task (uses the worker's resource)."""
        print(
            f"Worker {worker.name} is starting task {task.name} at time {self.env.now:.2f}"
        )
        yield self.env.timeout(task.duration)
        task.status = "Completed"
        print(
            f"Worker {worker.name} completed task {task.name} at time {self.env.now:.2f}"
        )

    def _sample_duration(self, task_name):
        """Samples a duration from the distribution for a given task."""
        dist_type, dist_params = (
            self.workflow.digital_twin.task_duration_distributions.get(
                task_name, ("empirical", [10.0])
            )
        )
        if dist_type == "lognorm":
            shape, loc, scale = dist_params
            return max(
                1.0, stats.lognorm.rvs(shape, loc=loc, scale=scale)
            )
        elif dist_type == "empirical":
            return max(1.0, np.random.choice(dist_params))
        else:
            return 10.0

    def _define_spaces(self):
        """
        Defines the observation and action spaces.
        This is a placeholder; you'll replace it with your specific
        observation and action space definitions. For example, using
        gym.spaces from the gym library.
        """

        self.observation_space = {
            "queue_lengths": [
                "Review",
                "Approval",
                "PaymentProcessing",
                "Finalize",
                "PaymentProcessing",
            ],
            "worker_utilization": {
                worker: 0 for worker in self.params.get("workers", [])
            },
            "case_age": 0,
            "task_completion_time": {},
            "rejected_tasks_count": 0,
        }

        # Action space:
        # 0: Do nothing
        # 1: Reassign a task (requires more complex action representation)
        # 2: Increase the priority of case Y
        # 3: Add a temporary worker to task type Z
        self.action_space = {
            "n_actions": 4  # actions: do nothing, reassign, increase priority, add worker
        }

    def _get_observation(self) -> Dict[str, Any]:
        """
        Gathers the current state of the simulation.
        """

        queue_lengths = {task: 0 for task in self.observation_space["queue_lengths"]}
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Pending":
                    queue_lengths[task.name] += 1

        worker_utilization = {worker_name: 0 for worker_name in self.workers}

        for worker_name, worker in self.workers.items():
            # Check if the worker is currently busy (in a task) - NEEDS IMPLEMENTATION
            # worker_utilization[worker_name] = 0 # Placeholder
            # Iterate through workflow's cases and tasks to check if a worker is busy.
            for case in self.workflow.cases:
                for task in case.tasks:
                    if task.status == "Pending" and task.name in queue_lengths:
                        if (
                            worker.name
                            == self.params.get("workers", [])[
                                (self.workflow.logs["Task"] == task.name)
                                & (self.workflow.logs["CaseID"] == case.id)
                            ]
                        ):
                            worker_utilization[worker_name] = 1  # worker is busy

        # Calculate case age
        # for case in self.workflow.cases:
        #     case_age = self.env.now - case.creation_time # Assuming you have a creation time
        #     self.observation_space['case_age'] = case_age

        # Calculate task completion time.
        task_completion_time = {}  # key will be task name.
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Completed":
                    task_completion_time[task.name] = task.duration

        # Rejected task count.
        rejected_tasks_count = 0
        for case in self.workflow.cases:
            if case.status == "Rejected":
                rejected_tasks_count += 1

        self.observation_space["queue_lengths"] = list(queue_lengths.values())
        self.observation_space["worker_utilization"] = worker_utilization
        self.observation_space["task_completion_time"] = task_completion_time
        self.observation_space["rejected_tasks_count"] = rejected_tasks_count

        return self.observation_space

    def _apply_action(self, action: int):
        """
        Applies an action to the simulation environment.
        """
        # For example:
        if action == 1:
            self._reassign_task()
        elif action == 2:
            self._increase_case_priority()
        elif action == 3:
            self._add_temporary_worker()
        pass

    def _reassign_task(self):
        """Reassigns a task from a queue to a different worker."""
        print("Reassigning a task (placeholder)")
        # Find the longest waiting task in any queue
        longest_waiting_task = None
        longest_wait_time = -1
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Pending":
                    wait_time = self.env.now
                    if wait_time > longest_wait_time:
                        longest_wait_time = wait_time
                        longest_waiting_task = (case, task)

        if longest_waiting_task:
            case, task = longest_waiting_task
            print(f"Reassigning task '{task.name}' from Case {case.id}")

            available_workers = [
                worker
                for worker_name, worker in self.workers.items()
                if worker_name
                not in [
                    t.worker
                    for case in self.workflow.cases
                    for t in case.tasks
                    if t.status == "Pending"
                ]
            ]
            if available_workers:
                new_worker = available_workers[
                    0
                ]
                print(f"Reassigning task '{task.name}' to worker {new_worker.name}")
                # TODO: Implement the reassignment
                # To reassign a task, you will need to
                # 1. Remove the task from the current worker.
                # 2. Assign the task to the new worker.

            else:
                print("No workers available to reassign the task.")
        else:
            print("No tasks are waiting to be reassigned.")

    def _increase_case_priority(self):
        """Increases the priority of a case (placeholder)."""
        print("Increasing case priority (placeholder)")
        # Implement case prioritization
        # Find the case
        # Increase its priority. Remaining to implement simpy.PriorityResource
        pass

    def _add_temporary_worker(self):
        """Adds a temporary worker (placeholder)."""
        print("Adding a temporary worker")

        temp_worker_name = f"TempWorker_{len(self.workers) + 1}"
        temp_worker = Worker(temp_worker_name, self.env)
        self.workers[temp_worker_name] = temp_worker
        print(f"Added temporary worker: {temp_worker_name}")

    def _compute_reward(self) -> float:
        """
        Computes the reward for the current state.
        # Reward components:
        # - Negative waiting time
        # - Penalty for rejected cases
        # - Cost for adding resources (if applicable)

        """
        # Negative wait time
        total_wait_time = 0
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Pending":
                    total_wait_time += task.duration
        reward = -total_wait_time * 0.1

        # Penalty for rejected cases
        rejected_count = self.observation_space[
            "rejected_tasks_count"
        ]
        reward -= rejected_count * 100.0

        # Reward for completed cases
        completed_cases = sum(
            1 for case in self.workflow.cases if case.status == "Accepted"
        )
        reward += completed_cases * 50.0

        return reward

    def _is_done(self) -> bool:
        """
        Checks if the episode is done (simulation end).
        """
        return self.env.now >= (24 * 60)
