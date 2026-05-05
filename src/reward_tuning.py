"""
reward_tuning.py
----------------
Data-driven reward weight tuning.

Approach:
  1. Extract KPI targets from the real event log (completion rate, rework, delay).
  2. Run short rollouts under a random policy with each candidate weight vector.
  3. Score each vector by how closely the achieved KPI outcomes match the real
     log targets — NOT by raw reward magnitude (which would just find weights
     that are easy to game).
  4. Return the best weights for use in RewardFunction.

Scoring rationale
-----------------
We want weights that make the reward signal *informative* — i.e. an agent
optimising this reward should end up with KPI outcomes close to the real log.
The proxy for this is: under a random policy, do the KPI signals stay in a
healthy range? If w_delay is too low the agent never learns to avoid delay;
if it's too high the reward is dominated by noise and the agent can't learn
anything useful.

The score is a weighted sum of:
  - completion_rate_error : |achieved - target| (lower is better)
  - rework_error          : |achieved - target| (lower is better)
  - delay_error           : |achieved - target| (lower is better)
  - reward_variance       : std of episode rewards (higher = more signal)

We minimise the KPI errors and maximise variance (so we negate it in the
combined score). This gives a Pareto-style objective that finds weights
where the reward is both accurate and informative.
"""

import numpy as np
import pandas as pd
from typing import Optional
from itertools import product


# ---------------------------------------------------------------------------
# KPI target extraction from real data
# ---------------------------------------------------------------------------

def extract_kpi_targets(kpi_df: pd.DataFrame) -> dict:
    """
    Derive target KPI values from real case KPI DataFrame.
    These become the optimization targets for reward tuning.

    Note on completion_rate: the real log completion rate is often 1.0 (all
    cases eventually closed), but a random routing policy will never achieve
    that in a single episode. We clamp to [0.3, 0.85] so the scorer measures
    meaningful signal rather than an impossible target.

    Args:
        kpi_df: Output of feature_engineering.compute_case_kpis(df)

    Returns:
        Dict of target values the reward-tuned agent should reproduce.
    """
    raw_cr = float(kpi_df["is_completed"].mean())
    return {
        # Clamp to a range achievable by a random policy in a single episode
        "target_completion_rate":  float(np.clip(raw_cr, 0.3, 0.85)),
        "target_median_age_days":  float(kpi_df["case_age_days"].median()),
        "target_mean_rework":      float(kpi_df["rework_count"].mean()),
        "target_p90_age_days":     float(kpi_df["case_age_days"].quantile(0.90)),
        "target_rework_zero_frac": float((kpi_df["rework_count"] == 0).mean()),
    }


# ---------------------------------------------------------------------------
# Weight search space
# ---------------------------------------------------------------------------

WEIGHT_GRID = {
    # w_completion → w_terminal in C2 env (terminal bonus)
    "w_completion": [5.0, 10.0, 15.0, 20.0, 30.0],
    # w_delay → w_progress in C2 env (progress toward terminal bonus)
    "w_delay":      [0.1, 0.3, 0.5, 1.0],
    # w_rework → w_loop in C2 env (excess loop penalty)
    "w_rework":     [0.2, 0.5, 1.0, 2.0],
    # w_risk → unused in C2 env (kept for legacy compat with old env)
    "w_risk":       [0.0, 0.5, 1.0],
    # w_throughput → w_step in C2 env (per-step cost)
    "w_throughput": [0.02, 0.05, 0.1, 0.2],
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _run_rollouts(weights: dict, env, n_episodes: int, seed: int) -> dict:
    """
    Run n_episodes under a random policy with the given weights applied to
    env. Returns aggregate KPI stats and reward variance.

    Works with both the old RewardFunction-based env and the new C2 routing env
    with MultiDiscrete([max_successors, N_MANAGEMENT_ACTIONS]) action space.
    """
    # Apply weights — C2 env stores weights directly on the env object
    if hasattr(env, 'reward_fn'):
        # Legacy: old abstract-action env with RewardFunction
        env.reward_fn.w_completion = weights["w_completion"]
        env.reward_fn.w_delay      = weights["w_delay"]
        env.reward_fn.w_rework     = weights["w_rework"]
        env.reward_fn.w_risk       = weights["w_risk"]
        env.reward_fn.w_throughput = weights.get("w_throughput", 0.1)
    else:
        # C2 routing env: weights are attributes on the env itself
        env.w_terminal = weights.get("w_completion", env.w_terminal)
        env.w_loop     = weights.get("w_rework",     env.w_loop)
        env.w_progress = weights.get("w_delay",      env.w_progress)
        env.w_step     = weights.get("w_throughput",  env.w_step)

    # Detect whether the env uses MultiDiscrete (routing + management) or Discrete
    import gymnasium as gym
    _is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)

    rng = np.random.default_rng(seed)
    ep_rewards:       list[float] = []
    ep_terminal_rate: list[float] = []
    ep_rework:        list[float] = []
    ep_delay:         list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0
        # Accumulate KPI signals across all steps (not just the last)
        step_rework: list[float] = []
        step_delay:  list[float] = []

        while not (done or truncated):
            combined_mask = env.action_masks()

            if _is_multidiscrete:
                # combined_mask = [routing_mask... | mgmt_mask...]
                max_succ = env.action_space.nvec[0]

                routing_mask = combined_mask[:max_succ]
                mgmt_mask    = combined_mask[max_succ:]

                valid_routing = np.where(routing_mask)[0]
                valid_mgmt    = np.where(mgmt_mask)[0]

                routing_idx = int(rng.choice(valid_routing))
                mgmt_idx    = int(rng.choice(valid_mgmt))
                action = np.array([routing_idx, mgmt_idx])
            else:
                # Legacy Discrete env
                valid = np.where(combined_mask)[0]
                action = int(rng.choice(valid))

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            kpi = info.get("kpi", {})
            if kpi:
                step_rework.append(float(kpi.get("rework_norm",  0.0)))
                step_delay.append( float(kpi.get("delay_proxy",  0.0)))

        ep_rewards.append(ep_reward)
        # done=True AND not truncated → reached a real terminal activity
        ep_terminal_rate.append(float(done and not truncated))
        if step_rework:
            ep_rework.append(float(np.mean(step_rework)))
        if step_delay:
            ep_delay.append(float(np.mean(step_delay)))

    return {
        "mean_reward":    float(np.mean(ep_rewards)),
        "std_reward":     float(np.std(ep_rewards)),
        "terminal_rate":  float(np.mean(ep_terminal_rate)),
        "mean_rework":    float(np.mean(ep_rework)) if ep_rework else 0.0,
        "mean_delay":     float(np.mean(ep_delay))  if ep_delay  else 0.0,
    }


def _score_weights(
    weights: dict,
    env,
    targets: dict,
    n_episodes: int,
    seed: int,
) -> float:
    """
    Score a weight vector. Higher = better.

    The score rewards:
      - completion rate (terminal_rate) close to the real log target
      - high reward variance (informative signal for the learner)

    Rework and delay errors are intentionally excluded: for short-episode
    processes (mean trace length < 10), rework_norm is near zero in every
    episode regardless of weights, making it a useless discriminator.
    Delay similarly saturates at near-zero for short episodes.

    The primary signal that varies meaningfully across weight vectors is:
      1. terminal_rate — do the weights incentivise reaching a terminal?
      2. reward variance — does the reward signal discriminate good from bad?

    Returns a scalar score (higher is better).
    """
    stats = _run_rollouts(weights, env, n_episodes, seed)

    target_cr = targets.get("target_completion_rate", 0.85)

    # Completion rate error: how far is the random-policy terminal rate from target?
    cr_error = abs(stats["terminal_rate"] - target_cr)

    # Variance bonus: more variance = more learning signal.
    # Normaliser is 20 because management action deltas inflate episode reward scale.
    variance_bonus = stats["std_reward"] / 20.0

    # Combined score: minimise completion error, maximise variance
    score = -(cr_error * 2.0) + variance_bonus

    return float(score)


# ---------------------------------------------------------------------------
# Main tuning entry point
# ---------------------------------------------------------------------------

def tune_reward_weights(
    env,
    kpi_df: pd.DataFrame,
    n_episodes_per_trial: int = 40,
    random_search: bool = True,
    n_random_trials: int = 60,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Search for reward weights that produce KPI outcomes matching the real log.

    Args:
        env:                  ProcessEnv instance (will have reward_fn mutated
                              during search; weights are restored to best at end).
        kpi_df:               Output of compute_case_kpis(df) on the real log.
        n_episodes_per_trial: Episodes per candidate weight vector.
        random_search:        True = random sampling from WEIGHT_GRID ranges.
                              False = full Cartesian grid (slow: 3^5 = 243 trials).
        n_random_trials:      Number of random trials (ignored if random_search=False).
        seed:                 RNG seed for reproducibility.
        verbose:              Print progress.

    Returns:
        Dict of best weights: {w_completion, w_delay, w_rework, w_risk, w_throughput}
    """
    targets = extract_kpi_targets(kpi_df)
    if verbose:
        print("KPI targets from real log:")
        for k, v in targets.items():
            print(f"  {k:<30} = {v:.4f}")
        print()

    rng = np.random.default_rng(seed)

    if random_search:
        trials = []
        for _ in range(n_random_trials):
            w = {k: float(rng.choice(v)) for k, v in WEIGHT_GRID.items()}
            trials.append(w)
    else:
        keys = list(WEIGHT_GRID.keys())
        vals = list(WEIGHT_GRID.values())
        trials = [dict(zip(keys, combo)) for combo in product(*vals)]

    if verbose:
        print(f"Searching {len(trials)} weight combinations "
              f"({n_episodes_per_trial} episodes each)...")

    best_score   = -np.inf
    best_weights = {k: v[1] for k, v in WEIGHT_GRID.items()}  # safe default = middle

    for i, w in enumerate(trials):
        score = _score_weights(w, env, targets, n_episodes_per_trial, seed + i)
        if score > best_score:
            best_score   = score
            best_weights = w.copy()
            if verbose:
                print(f"  [{i:3d}/{len(trials)}] new best  score={score:+.4f}  "
                      f"w_comp={w['w_completion']:.1f}  w_delay={w['w_delay']:.1f}  "
                      f"w_rework={w['w_rework']:.1f}  w_risk={w['w_risk']:.1f}  "
                      f"w_tp={w.get('w_throughput', 0.2):.2f}")

    # Apply best weights to env permanently
    if hasattr(env, 'reward_fn'):
        env.reward_fn.w_completion = best_weights["w_completion"]
        env.reward_fn.w_delay      = best_weights["w_delay"]
        env.reward_fn.w_rework     = best_weights["w_rework"]
        env.reward_fn.w_risk       = best_weights["w_risk"]
        env.reward_fn.w_throughput = best_weights.get("w_throughput", 0.1)
    else:
        env.w_terminal = best_weights.get("w_completion", env.w_terminal)
        env.w_loop     = best_weights.get("w_rework",     env.w_loop)
        env.w_progress = best_weights.get("w_delay",      env.w_progress)
        env.w_step     = best_weights.get("w_throughput",  env.w_step)

    if verbose:
        print(f"\nBest weights (score={best_score:+.4f}):")
        for k, v in best_weights.items():
            print(f"  {k:<20} = {v}")

    return best_weights
