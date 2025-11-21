from typing import Any, Dict, Generator, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import simpy
from scipy import stats

from digital_twin.case import Case
from digital_twin.digital_twin_lite import DigitalTwinLite
from digital_twin.task import Task
from digital_twin.worker import Worker
from digital_twin.workflow import Workflow


class SimEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
        self,
        twin_params: Dict[str, Any],
        logs_path: str = "logs.csv",
        decision_interval: float = 5.0,
    ) -> None:
        super().__init__()
        self.params: Dict[str, Any] = twin_params
        self.logs_path: str = logs_path
        self.decision_interval: float = decision_interval
        self.env: simpy.Environment = simpy.Environment()
        self.workers: Dict[str, Worker] = {}
        self.digital_twin: DigitalTwinLite = DigitalTwinLite()
        self.arrival_process: Optional[simpy.Process] = None
        self.case_id_counter: int = 1000
        self.observation_space: Dict[str, Any] = {}
        self.action_space: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.workflow: Workflow = Workflow("Workflow", self.env)

        self._load_digital_twin()
        self._setup_workers()
        self._define_spaces()

    def reset(self) -> Dict[str, Any]:
        self.workflow = Workflow("Workflow", self.env)
        self._load_digital_twin()
        self._setup_workers()
        self._start_arrival_process()
        self.state = self._get_observation()
        return self.state

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._apply_action(action)
        self.env.run(until=self.env.now + self.decision_interval)
        self.state = self._get_observation()
        reward: float = self._compute_reward()
        terminated: bool = self._is_done()
        truncated: bool = False
        info: Dict[str, Any] = {}
        return self.state, reward, terminated, truncated, info

    def _setup_workers(self) -> None:
        if self.workflow:
            self.workflow.define_workers()
            self.workers = {worker.name: worker for worker in self.workflow.workers}
            print(f"Created workers: {list(self.workers.keys())}")
        else:
            print("Warning: Workflow not initialized. Cannot setup workers.")

    def _load_digital_twin(self) -> None:
        if self.workflow:
            self.workflow.read_logs(self.logs_path)
            self.digital_twin = self.workflow.digital_twin
            if self.digital_twin:
                print("Digital Twin Loaded.")
            else:
                print("DigitalTwinLite failed to load")
        else:
            print(
                "Warning: Workflow not initialized. Cannot load logs or set up digital twin."
            )

    def _start_arrival_process(self) -> None:
        if self.workflow and self.workflow.digital_twin.arrival_rate > 0:
            self.arrival_process = self.env.process(self._generate_cases())

    def _generate_cases(self) -> Generator[Any, None, None]:
        while True:
            interarrival_time: float = np.random.exponential(
                1.0 / self.workflow.digital_twin.arrival_rate
            )
            yield self.env.timeout(interarrival_time)
            self.case_id_counter += 1
            yield self.env.process(self._create_case(self.case_id_counter))

    def _create_case(self, case_id):
        print(f"Creating new case {case_id} at time {self.env.now:.2f}")
        case_tasks = self.workflow.logs[self.workflow.logs["CaseID"] == case_id]
        if case_tasks.empty:
            print(f"Warning: No tasks found for case {case_id} in logs.")
            return

        case_tasks = case_tasks.sort_values(by="StartTime")

        for _, task_row in case_tasks.iterrows():
            task_name = task_row["Task"]
            duration = self.workflow.digital_twin.sample_duration(task_name)
            task = Task(task_name, duration, case=self)
            worker_name = task_row["Worker"]
            worker = self.workers.get(worker_name)
            if not worker:
                print(
                    f"Warning: No worker found named {worker_name} for task {task_name}"
                )
                continue

            print(
                f"Case {case_id}: starting task '{task_name}' for {duration:.2f} minutes"
            )
            yield self.env.process(self._perform_task(worker, task))
            if task.status != "Completed":
                print(
                    f"Case {case_id}: Task {task.name} was not completed, breaking case workflow"
                )
                break
        print(f"Case {case_id} completed at time {self.env.now:.2f}")

    def _perform_task(self, worker: Worker, task: Task) -> Generator[Any, None, None]:
        print(
            f"Worker {worker.name} is starting task {task.name} at time {self.env.now:.2f}"
        )
        yield self.env.timeout(task.duration)
        task.status = "Completed"
        print(
            f"Worker {worker.name} completed task {task.name} at time {self.env.now:.2f}"
        )

    def _sample_duration(self, task_name: str) -> float:
        dist_type: str
        dist_params: Any
        dist_type, dist_params = (
            self.workflow.digital_twin.task_duration_distributions.get(
                task_name, ("empirical", [10.0])
            )
        )
        if dist_type == "lognorm":
            shape: float
            loc: float
            scale: float
            shape, loc, scale = dist_params
            return max(1.0, stats.lognorm.rvs(shape, loc=loc, scale=scale))
        elif dist_type == "empirical":
            return max(1.0, np.random.choice(dist_params))
        else:
            return 10.0

    def _define_spaces(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "queue_lengths": gym.spaces.Box(
                    low=0,
                    high=100,
                    shape=(
                        len(self.workflow.digital_twin.task_duration_distributions),
                    ),
                    dtype=np.float32,
                ),
                "worker_utilization": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(self.workers),),
                    dtype=np.float32,
                ),
                "case_age": gym.spaces.Box(
                    low=0, high=1000, shape=(1,), dtype=np.float32
                ),
                "task_completion_time": gym.spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.float32
                ),
                "rejected_tasks_count": gym.spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(4)

    def _get_observation(self) -> Dict[str, Any]:
        queue_lengths: Dict[str, int] = {
            task: 0 for task in self.workflow.digital_twin.task_duration_distributions
        }
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Pending":
                    queue_lengths[task.name] += 1

        worker_utilization: Dict[str, int] = {
            worker_name: 0 for worker_name in self.workers
        }

        for worker_name, worker in self.workers.items():
            is_busy = False
            for case in self.workflow.cases:
                for task in case.tasks:
                    if task.status == "Pending" and task.worker == worker.name:
                        is_busy = True
                        break
                if is_busy:
                    break

        case_age = 0
        if self.workflow.cases:
            case_age = self.env.now - self.workflow.cases[0].creation_time

        task_completion_time: Dict[str, float] = {}
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Completed":
                    task_completion_time[task.name] = task.duration

        rejected_tasks_count: int = 0
        for case in self.workflow.cases:
            if case.status == "Rejected":
                rejected_tasks_count += 1

        return {
            "queue_lengths": np.array(list(queue_lengths.values()), dtype=np.float32),
            "worker_utilization": np.array(
                list(worker_utilization.values()), dtype=np.float32
            ),
            "case_age": np.array(case_age, dtype=np.float32),
            "task_completion_time": np.array(
                list(task_completion_time.values()), dtype=np.float32
            ),
            "rejected_tasks_count": np.array(rejected_tasks_count, dtype=np.float32),
        }

    def _apply_action(self, action: int) -> None:
        if action == 1:
            self._reassign_task()
        elif action == 2:
            self._increase_case_priority()
        elif action == 3:
            self._add_temporary_worker()
        pass

    def _reassign_task(self) -> None:
        print("Reassigning a task")
        longest_waiting_task: Optional[Tuple[Case, Task]] = None
        longest_wait_time: float = -1
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Pending":
                    wait_time: float = (
                        self.env.now - task.start_time
                        if hasattr(task, "start_time")
                        else 0
                    )
                    if wait_time > longest_wait_time:
                        longest_wait_time = wait_time
                        longest_waiting_task = (case, task)

        if longest_waiting_task:
            case: Case
            task: Task
            case, task = longest_waiting_task
            print(f"Reassigning task '{task.name}' from Case {case.id}")

            # Barebones Implementation
            available_workers: List[Worker] = [
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
                new_worker: Worker = available_workers[0]
                print(f"Reassigning task '{task.name}' to worker {new_worker.name}")

                if task in case.tasks:
                    case.tasks.remove(task)
                    task.worker = new_worker.name  # type: ignore[attr-defined]
                    case.tasks.append(task)
            else:
                print("No workers available to reassign the task.")
        else:
            print("No tasks are waiting to be reassigned.")

    def _increase_case_priority(self) -> None:
        print("Increasing case priority")
        case_id_to_increase: int = self.workflow.cases[0].id

        for case in self.workflow.cases:
            if case.id == case_id_to_increase:
                case.priority -= 1
                print(f"Increased priority of case {case.id} to {case.priority}")
                break

    def _add_temporary_worker(self) -> None:
        print("Adding a temporary worker")

        temp_worker_name = f"TempWorker_{len(self.workers) + 1}"
        temp_worker: Worker = Worker(temp_worker_name, self.env)
        self.workers[temp_worker_name] = temp_worker
        print(f"Added temporary worker: {temp_worker_name}")

    def _compute_reward(self) -> float:
        # Negative wait time
        total_wait_time: float = 0
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Pending":
                    total_wait_time += task.duration
        reward: float = -total_wait_time * 0.1

        rejected_count = self.observation_space["rejected_tasks_count"]  # type: ignore
        reward -= rejected_count * 100.0

        # Reward for completed cases
        completed_cases: int = sum(
            1 for case in self.workflow.cases if case.status == "Accepted"
        )
        reward += completed_cases * 50.0

        return reward

    def _is_done(self) -> bool:
        return self.env.now >= (24 * 60)

    def render(self) -> None:
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self) -> None:
        import pygame

        if self.window is None:
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Bureaucratic Workflow")
        font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.get_surface()
        assert self.screen is not None, "Something went wrong with pygame!"
        self.screen.fill((0, 0, 0))

        queue_lengths: List[Any] = self.state["queue_lengths"]  # type: ignore
        y: int = 50
        for i, task in enumerate(self.observation_space["queue_lengths"]):  # type: ignore
            text = font.render(f"{task}: {queue_lengths[i]}", True, (255, 255, 255))
            self.screen.blit(text, (50, y))
            y += 30
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
