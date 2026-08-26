"""
experiment.py
-------------
One definition of "train a policy" and "measure a policy", so that a sweep and
the headline run cannot disagree.

Why this module exists
----------------------
Every number in this project came from a single training run executed by hand
in ``notebooks/04_rl_training.ipynb``. Two of the open questions cannot be
answered that way:

* **A4, run-to-run spread.** No difference between policies is defensible until
  we know how much the same policy varies across seeds.
* **The ``effect_scale`` sensitivity sweep.** Every effect size in
  ``intervention_effects.py`` is an assumption the logs cannot support, so
  nothing about the catalogue can be claimed without showing which conclusions
  survive changing them.

Both need dozens of runs, which means a script. A script that re-implements the
training setup is exactly the failure this codebase already had once — notebook
04 applied the tuned reward and notebooks 05/06 did not, so the agent was
graded under a reward it never trained on (see ``reward_config.py``). So the
setup lives here, and notebook 04 calls it. There is one definition.

The measurement side is deliberately thin: it reuses
``insights.collect_trajectories`` — the same rollout notebook 06 analyses —
rather than rolling its own loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

#: Training settings shared by every run, headline or swept. Changing one of
#: these changes it everywhere, which is the point.
LOG_INTERVAL = 10_000
MAX_STEPS = 500_000
EARLY_STOPPING = {"window": 5, "min_delta": 0.3, "patience": 8}

#: Management actions that waive a rule or skip a required step. Their combined
#: share is the headline number for the compliance charge, so it is defined
#: once here rather than re-listed at each call site.
RULE_WAIVING_ACTIONS = ("skip_optional_subprocess", "relax_rules_for_low_risk")

DAY = 86_400.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_policy(
    env,
    seed: int = 0,
    log_path: "str | Path | None" = None,
    max_steps: int = MAX_STEPS,
    log_interval: int = LOG_INTERVAL,
    early_stopping: "dict | None" = None,
    verbose: int = 1,
):
    """
    Train MaskablePPO on `env` with the project's standard setup.

    Everything except the seed is an SB3 default. Early stopping is fixed for
    every dataset — it used to scale with ``w_terminal`` and the activity
    count, which contradicted the claim that identical hyperparameters are
    used throughout.

    Returns
    -------
    (model, run_info) where `run_info` records what was actually run, for
    ``env_factory.save_run_config``.
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList

    from early_stopping import EarlyStoppingCallback
    from training_logger import TrainingLogger

    es = dict(EARLY_STOPPING, **(early_stopping or {}))

    stopper = EarlyStoppingCallback(
        window=es["window"],
        min_delta=es["min_delta"],
        patience=es["patience"],
        check_freq=log_interval,
        verbose=verbose,
    )
    logger = TrainingLogger(
        log_interval=log_interval,
        log_path=str(log_path) if log_path else None,
        verbose=verbose,
        early_stopping=stopper,
    )

    model = MaskablePPO("MultiInputPolicy", env, verbose=0, seed=seed)

    started = time.time()
    model.learn(total_timesteps=max_steps, callback=CallbackList([logger, stopper]))
    wall_s = time.time() - started

    run_info = {
        "training_seed": int(seed),
        "algorithm": "MaskablePPO",
        "policy": "MultiInputPolicy",
        "total_timesteps_cap": int(max_steps),
        "timesteps_run": int(model.num_timesteps),
        "log_interval": int(log_interval),
        "early_stopping": es,
        "wall_seconds": round(wall_s, 1),
        "ppo_hyperparameters": {
            "learning_rate": model.learning_rate,
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "clip_range": "sb3 default (0.2)",
            "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef,
            "max_grad_norm": model.max_grad_norm,
        },
    }
    return model, run_info


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class PolicyMetrics:
    """
    Everything the handoff table reports, for one policy on one dataset.

    Ratios are simulated / real, so 1.0 is fidelity and 0.06 means the agent
    finishes sixteen times faster than any real case.
    """
    n_episodes: int

    mean_reward: float
    std_reward: float

    mean_length: float
    real_median_length: "float | None"
    length_ratio: "float | None"

    median_cycle_days: float
    mean_cycle_days: float
    real_median_cycle_days: "float | None"
    cycle_ratio: "float | None"
    #: median |log2(simulated / real)| — the quantity the reward now charges.
    cycle_log2_deviation: "float | None"

    concluded_rate: float
    truncated_rate: float
    good_rate_of_concluded: float
    real_good_rate: "float | None"

    self_loop_rate: float
    real_self_loop_rate: "float | None"

    no_op_rate: float
    rule_waiving_rate: float
    mgmt_action_rates: dict

    def to_dict(self) -> dict:
        return asdict(self)


def real_self_loop_rate(dataset_dir: "str | Path") -> "float | None":
    """
    Share of real transitions that repeat the current activity.

    Step-weighted over the log, which is the same denominator the agent's rate
    uses — one number per transition, not per activity. Averaging the twin's
    per-activity loop probabilities instead would weight a rare activity the
    same as a dominant one and does not reproduce the log.

    Returns 42.5% / 0.8% / 38.3% on BPIC2012 / BPIC2015 / BPIC2017, the rates
    the agent is compared against, and None if the event log is not on disk.
    """
    import pandas as pd

    from timeutils import ensure_utc_timestamps, sort_events

    dataset_dir = Path(dataset_dir)
    path = dataset_dir / f"events_{dataset_dir.name}_train.parquet"
    if not path.exists():
        return None

    df = sort_events(ensure_utc_timestamps(pd.read_parquet(path)))
    within_case = df["case_id"] == df["case_id"].shift(1)
    repeated = within_case & (df["activity"] == df["activity"].shift(1))
    n_transitions = int(within_case.sum())
    return float(repeated.sum() / n_transitions) if n_transitions else None


def measure_policy(
    env,
    model,
    n_episodes: int = 300,
    deterministic: bool = True,
    real_reference: "dict | None" = None,
    real_loop_rate: "float | None" = None,
) -> PolicyMetrics:
    """
    Roll out `model` on `env` and reduce it to the reported metrics.

    Uses ``insights.collect_trajectories`` — the same rollout notebook 06
    analyses — so a swept run and the headline run measure the same thing.

    `real_reference` is the dataset's ``terminal_classification.json`` and
    `real_loop_rate` comes from :func:`real_self_loop_rate`; without them the
    real-log columns come back None rather than invented.
    """
    from insights import collect_trajectories

    trajectories = collect_trajectories(
        env, model, n_episodes=n_episodes, deterministic=deterministic
    )

    rewards = np.array([t["total_reward"] for t in trajectories], dtype=float)
    lengths = np.array([t["n_steps"] for t in trajectories], dtype=float)
    concluded = np.array([t["terminated"] for t in trajectories], dtype=bool)
    cycle_days = np.array([t["cycle_time_s"] for t in trajectories], dtype=float) / DAY

    all_steps = [s for t in trajectories for s in t["steps"]]
    n_steps = max(len(all_steps), 1)

    self_loops = sum(s["from_activity"] == s["to_activity"] for s in all_steps)

    mgmt_counts: dict[str, int] = {}
    for step in all_steps:
        name = step["mgmt_action_name"]
        mgmt_counts[name] = mgmt_counts.get(name, 0) + 1
    mgmt_rates = {k: v / n_steps for k, v in sorted(mgmt_counts.items())}

    ref = real_reference or {}
    real_len = (ref.get("steps_to_outcome") or {}).get("p50")
    real_cycle = ref.get("real_cycle_days_median")
    real_good = (ref.get("outcome_base_rates") or {}).get("p_good")

    # Cycle time on concluded episodes only. A truncated episode ran into the
    # step cap, so its elapsed time measures the cap, not the policy.
    concluded_cycle = cycle_days[concluded]
    if concluded_cycle.size == 0:
        concluded_cycle = cycle_days
    median_cycle = float(np.median(concluded_cycle)) if concluded_cycle.size else float("nan")

    cycle_ratio = log2_dev = None
    if real_cycle and median_cycle == median_cycle and median_cycle > 0:
        cycle_ratio = median_cycle / real_cycle
        positive = concluded_cycle[concluded_cycle > 0]
        if positive.size:
            log2_dev = float(np.median(np.abs(np.log2(positive / real_cycle))))

    n_concluded = int(concluded.sum())
    good = sum(
        t["terminated"] and not t["hit_bad_terminal"] for t in trajectories
    )

    return PolicyMetrics(
        n_episodes=len(trajectories),
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std(ddof=1)) if rewards.size > 1 else 0.0,
        mean_length=float(lengths.mean()),
        real_median_length=float(real_len) if real_len else None,
        length_ratio=float(lengths.mean() / real_len) if real_len else None,
        median_cycle_days=median_cycle,
        mean_cycle_days=float(concluded_cycle.mean()) if concluded_cycle.size else float("nan"),
        real_median_cycle_days=float(real_cycle) if real_cycle else None,
        cycle_ratio=cycle_ratio,
        cycle_log2_deviation=log2_dev,
        concluded_rate=n_concluded / max(len(trajectories), 1),
        truncated_rate=1.0 - n_concluded / max(len(trajectories), 1),
        good_rate_of_concluded=good / n_concluded if n_concluded else float("nan"),
        real_good_rate=float(real_good) if real_good else None,
        self_loop_rate=self_loops / n_steps,
        real_self_loop_rate=real_loop_rate,
        no_op_rate=mgmt_rates.get("assign_to_primary_team", 0.0),
        rule_waiving_rate=sum(mgmt_rates.get(a, 0.0) for a in RULE_WAIVING_ACTIONS),
        mgmt_action_rates=mgmt_rates,
    )


# ---------------------------------------------------------------------------
# One run, end to end
# ---------------------------------------------------------------------------

def run_experiment(
    dataset_dir: "str | Path",
    reward_config,
    seed: int,
    run_dir: "str | Path",
    n_eval_episodes: int = 300,
    env_seed: int = 42,
    max_steps: int = MAX_STEPS,
    verbose: int = 0,
) -> dict:
    """
    Train and measure one (dataset, reward, seed) combination.

    Reads the fitted artefacts from `dataset_dir` and writes everything it
    produces to `run_dir`, so a sweep never touches ``output/``.
    """
    from env_factory import build_process_env

    dataset_dir = Path(dataset_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pass the reward explicitly: build_process_env otherwise prefers whatever
    # reward_config.json happens to be sitting in the dataset directory, and a
    # sweep would silently measure the same reward every time.
    env, env_meta = build_process_env(
        dataset_dir, reward_config=reward_config, seed=env_seed
    )

    model, run_info = train_policy(
        env, seed=seed, log_path=run_dir, max_steps=max_steps, verbose=verbose
    )
    model.save(run_dir / "model.zip")

    with open(dataset_dir / "terminal_classification.json", encoding="utf-8") as fh:
        reference = json.load(fh)

    metrics = measure_policy(
        env, model,
        n_episodes=n_eval_episodes,
        real_reference=reference,
        real_loop_rate=real_self_loop_rate(dataset_dir),
    )

    record = {
        "dataset": dataset_dir.name,
        "seed": int(seed),
        "reward_config": reward_config.to_dict(),
        "env": {k: v for k, v in env_meta.items()
                if k not in ("good_terminals", "bad_terminals", "steps_to_outcome_real")},
        "training": run_info,
        "metrics": metrics.to_dict(),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    return record
