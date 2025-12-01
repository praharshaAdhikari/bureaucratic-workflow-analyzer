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
        # Allow twin_params to override estimates from logs (e.g. arrival_rate)
        if isinstance(twin_params, dict) and "arrival_rate" in twin_params:
            try:
                # Apply override to the workflow's digital twin if present
                self.workflow.digital_twin.arrival_rate = float(twin_params["arrival_rate"])
                print(
                    f"Overriding digital twin arrival_rate -> {self.workflow.digital_twin.arrival_rate} cases/min"
                )
            except Exception:
                print("Warning: invalid arrival_rate in twin_params; ignoring override.")
        tasks = self.workflow.tasks or []
        print(f"Loaded tasks: {len(tasks)}")


        self._setup_workers()
        self._define_spaces()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.env = simpy.Environment()
        
      
        if hasattr(self, 'workflow') and self.workflow is not None:
            self.workflow.cases.clear()
            self.workflow.env = self.env 
        else:
            self.workflow = Workflow("Workflow", self.env)
            
        self._load_digital_twin()
        self._setup_workers()
        self._start_arrival_process()
        self.state = self._get_observation()
        return self.state, {}

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

        # Try to find a corresponding case in the historical logs
        case_tasks = (
            self.workflow.logs[self.workflow.logs["CaseID"] == case_id]
            if self.workflow.logs is not None
            else None
        )

        # If no exact match, sample an existing case template from the logs
        if case_tasks is None or case_tasks.empty:
            if self.workflow.logs is None or self.workflow.logs.empty:
                print(f"Warning: No logs available to create case {case_id}.")
                return
            # Pick a random template case from the logs
            template_case_id = np.random.choice(self.workflow.logs["CaseID"].unique())
            case_tasks = self.workflow.logs[self.workflow.logs["CaseID"] == template_case_id]
            print(
                f"No exact match for case {case_id}; sampling template case {template_case_id} from logs."
            )

        case_tasks = case_tasks.sort_values(by="StartTime")

        # Create a Case instance and add it to the workflow
        case = Case(case_id, self.env)
        self.workflow.cases.append(case)
        case.start_work()

        for _, task_row in case_tasks.iterrows():
            task_name = task_row["Task"]
            duration = self.workflow.digital_twin.sample_duration(task_name)
            # Attach the task to the case (so priority and completion checks work)
            task = Task(task_name, duration, case=case)

            worker_name = task_row.get("Worker") if isinstance(task_row, dict) else task_row["Worker"]
            worker = self.workers.get(worker_name)

            if worker is None:
                # If the exact worker doesn't exist in the current environment, pick any worker
                available = list(self.workers.values())
                if not available:
                    print(
                        f"No workers available to perform task '{task_name}' for case {case_id}. Skipping task."
                    )
                    continue
                worker = np.random.choice(available)
                print(
                    f"No worker named {worker_name} found — assigned {worker.name} for task '{task_name}'."
                )

            # Register the task and worker on the case
            case.tasks.append(task)
            case.workers.append(worker)

            print(f"Case {case_id}: scheduling task '{task_name}' for {duration:.2f} minutes with worker {worker.name}")

            # Use the Case.assign_task which will manage the worker.perform_task and case completion checks
            yield self.env.process(case.assign_task(worker, task, speed=1.0, expected_status="Completed"))

            if task.status != "Completed":
                print(
                    f"Case {case_id}: Task {task.name} was not completed, breaking case workflow"
                )
                break

        # case.check_completion() should have been called after tasks, but call again to be sure
        case.check_completion()
        print(f"Case {case_id} completed with status {case.status} at time {self.env.now:.2f}")

    def _perform_task(self, worker: Worker, task: Task) -> Generator[Any, None, None]:
        # Request worker resource to ensure exclusivity
        with worker.resource.request() as req:
            yield req # Wait for worker to be free
            
            print(f"Worker {worker.name} is starting task {task.name} at time {self.env.now:.2f}")
            yield self.env.timeout(task.duration)
            task.status = "Completed"
            print(f"Worker {worker.name} completed task {task.name} at time {self.env.now:.2f}")

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
        num_tasks = len(self.workflow.digital_twin.task_duration_distributions)
        
        # Base workers + max temp workers (e.g., 5)
        self.max_workers = 15
        
        if num_tasks == 0:
            num_tasks = 1 # Fallback

        print(f"Defining spaces with num_tasks={num_tasks}, max_workers={self.max_workers}")

        self.observation_space = gym.spaces.Dict(
            {
                "queue_lengths": gym.spaces.Box(
                    low=0,
                    high=100,
                    shape=(num_tasks,),
                    dtype=np.float32,
                ),
                "worker_utilization": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_workers,),
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
        task_names = sorted(list(self.workflow.digital_twin.task_duration_distributions.keys()))
        worker_names = sorted(list(self.workers.keys()))
        
        queue_lengths = np.zeros(len(task_names), dtype=np.float32)
        for i, task_name in enumerate(task_names):
            count = 0
            for case in self.workflow.cases:
                for task in case.tasks:
                    if task.name == task_name and task.status == "Pending":
                        count += 1
            queue_lengths[i] = count

        worker_utilization = np.zeros(self.max_workers, dtype=np.float32)
        
        for i, worker_name in enumerate(worker_names):
            if i >= self.max_workers:
                break 
            worker = self.workers[worker_name]
            if worker.resource.count > 0:
                worker_utilization[i] = 1.0
            else:
                worker_utilization[i] = 0.0

        case_age = 0.0
        if self.workflow.cases:
            pending_cases = [c for c in self.workflow.cases if c.status == "Pending"]
            if pending_cases:
                case_age = self.env.now - pending_cases[0].creation_time

        task_completion_times = []
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Completed":
                    task_completion_times.append(task.duration)
        
        avg_completion_time = 0.0
        if task_completion_times:
            avg_completion_time = np.mean(task_completion_times)

        rejected_tasks_count: int = 0
        for case in self.workflow.cases:
            if case.status == "Rejected":
                rejected_tasks_count += 1

        obs = {
            "queue_lengths": queue_lengths,
            "worker_utilization": worker_utilization,
            "case_age": np.array([case_age], dtype=np.float32),
            "task_completion_time": np.array(
                [avg_completion_time], dtype=np.float32
            ),
            "rejected_tasks_count": np.array([rejected_tasks_count], dtype=np.float32),
        }
        
        # Debug checks
        if obs['queue_lengths'].shape != self.observation_space['queue_lengths'].shape:
            print(f"CRITICAL MISMATCH: queue_lengths obs {obs['queue_lengths'].shape} != space {self.observation_space['queue_lengths'].shape}")
            
        if obs['worker_utilization'].shape != self.observation_space['worker_utilization'].shape:
            print(f"CRITICAL MISMATCH: worker_utilization obs {obs['worker_utilization'].shape} != space {self.observation_space['worker_utilization'].shape}")

        return obs

    def _apply_action(self, action: int) -> None:
        if action == 1:
            self._reassign_task()
        elif action == 2:
            self._increase_case_priority()
        elif action == 3:
            self._add_temporary_worker()
        pass

    def _reassign_task(self) -> None:
        longest_waiting_task: Optional[Tuple[Case, Task]] = None
        longest_wait_time: float = -1
        
        for case in self.workflow.cases:
            for task in case.tasks:
                if task.status == "Pending":
                    # Approximate wait time
                    wait_time = self.env.now - case.creation_time
                    if wait_time > longest_wait_time:
                        longest_wait_time = wait_time
                        longest_waiting_task = (case, task)

        if longest_waiting_task:
            case, task = longest_waiting_task
            # print(f"Reassigning task '{task.name}' from Case {case.id}")

            # Find a worker who is free or has less load
            # Be defensive in case task wasn't annotated with assigned_worker_name
            current_worker_name = getattr(task, "assigned_worker_name", None)
            
            best_worker_name = None
            min_load = float('inf')
            
            for name, worker in self.workers.items():
                if name == current_worker_name:
                    continue
                    
                # Calculating load
                load = len(worker.resource.queue) + worker.resource.count
                
                if load < min_load:
                    min_load = load
                    best_worker_name = name
                elif load == min_load:
                    if np.random.random() < 0.5:
                        best_worker_name = name

            if best_worker_name:
                task.assigned_worker_name = best_worker_name
                print(f"Reassigned task {task.name} (Case {case.id}) from {current_worker_name} to {best_worker_name} (Load: {min_load})")

    def _increase_case_priority(self) -> None:
        print("Increasing case priority")
        pending_cases = [c for c in self.workflow.cases if c.status == "Pending"]
        if pending_cases:
            case = pending_cases[0]
            case.priority -= 1
            print(f"Increased priority of case {case.id} to {case.priority}")

    def _add_temporary_worker(self) -> None:
        # Count current temp workers
        temp_workers = [w for w in self.workers.keys() if w.startswith("TempWorker")]
        if len(temp_workers) >= 5:
            print("Max temporary workers reached (5). Cannot add more.")
            return

        print("Adding a temporary worker")
        temp_worker_name = f"TempWorker_{len(self.workers) + 1}"
        temp_worker: Worker = Worker(temp_worker_name, self.env)
        self.workers[temp_worker_name] = temp_worker
        print(f"Added temporary worker: {temp_worker_name}")

    def _compute_reward(self) -> float:
        # Negative wait time
        total_wait_time: float = 0.0
        for case in self.workflow.cases:
             if case.status == "Pending":
                 total_wait_time += float(self.env.now - case.creation_time)
        
        reward: float = -total_wait_time * 0.1

        # Use state value, not observation space definition
        rejected_count = 0.0
        if "rejected_tasks_count" in self.state:
             rejected_count = float(self.state["rejected_tasks_count"][0])
        
        reward -= rejected_count * 100.0

        # Reward for completed cases
        completed_cases: int = sum(
            1 for case in self.workflow.cases if case.status == "Accepted"
        )
        reward += float(completed_cases * 50.0)

        return float(reward)

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
