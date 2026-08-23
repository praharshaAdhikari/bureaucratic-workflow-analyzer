"""
training_logger.py
------------------
SB3 callback that prints a compact one-line terminal log at fixed timestep
intervals and writes a richer CSV for post-hoc analysis.

Supports both:
  - Discrete(n)                  — legacy routing-only action space
  - MultiDiscrete([max_succ, 15]) — routing + KPI management action space

Terminal output (one line per interval, header every 20 lines):
  step | eps | rew±std | term% | good% | bad% | trunc% | H | fps

CSV columns (superset — everything useful for post-hoc analysis):
  timestep, episodes,
  ep_reward_mean, ep_reward_std, ep_reward_p10, ep_reward_p50, ep_reward_p90, ep_len_mean,
  delay_mean, rework_mean, risk_mean,
  terminal_rate, good_terminal_rate, bad_terminal_rate, truncated_rate,
  top_routing_action, routing_entropy,
  top_mgmt_action, mgmt_entropy,
  mgmt_<action_name>_rate  (one column per management action, if MultiDiscrete),
  action_entropy,           (alias for routing_entropy, kept for backward compat)
  top_action,               (alias for top_routing_action)
  value_loss, policy_loss, entropy_loss, explained_variance, clip_fraction,
  fps
"""

import time
import csv
import os
import numpy as np
from collections import Counter
from stable_baselines3.common.callbacks import BaseCallback

# Lazy import — only needed when env uses MultiDiscrete
_MANAGEMENT_ACTIONS = None
_N_MGMT = 0

def _get_mgmt_actions():
    global _MANAGEMENT_ACTIONS, _N_MGMT
    if _MANAGEMENT_ACTIONS is None:
        try:
            from kpi_actions import MANAGEMENT_ACTIONS, N_MANAGEMENT_ACTIONS
            _MANAGEMENT_ACTIONS = MANAGEMENT_ACTIONS
            _N_MGMT = N_MANAGEMENT_ACTIONS
        except ImportError:
            _MANAGEMENT_ACTIONS = []
            _N_MGMT = 0
    return _MANAGEMENT_ACTIONS, _N_MGMT

# ── Terminal column widths (keep total ≤ 110 chars) ───────────────────
_HDR = (
    f"{'step':>8}  {'eps':>5}  {'rew':>8}  {'±':>6}  "
    f"{'term':>5}  {'good':>5}  {'bad':>5}  {'trnc':>5}  "
    f"{'H':>5}  {'ev':>6}  {'fps':>5}"
)
_SEP = "─" * len(_HDR)
_HEADER_EVERY = 20   # reprint header every N log lines


class TrainingLogger(BaseCallback):
    """
    Logs training metrics every ``log_interval`` timesteps.

    Parameters
    ----------
    log_interval : int
        How often (in timesteps) to emit a log line / CSV row.
    log_path : str | None
        Directory for ``training_metrics.csv``.  Created if absent.
    verbose : int
        0 = CSV only (silent terminal), 1 = compact terminal lines (default).
    """

    def __init__(
        self,
        log_interval: int = 10_000,
        log_path: str | None = None,
        verbose: int = 1,
        early_stopping=None,   # EarlyStoppingCallback instance, optional
    ):
        super().__init__(verbose)
        self.log_interval    = log_interval
        self.log_path        = log_path
        self.early_stopping  = early_stopping
        self._csv_path    = os.path.join(log_path, "training_metrics.csv") if log_path else None
        self._csv_file    = None
        self._csv_writer  = None
        self._log_count   = 0   # how many lines printed (for header repeat)

        # Rolling episode buffers — cleared each interval
        self._ep_rewards:   list[float] = []
        self._ep_lengths:   list[int]   = []
        self._delay_vals:   list[float] = []
        self._rework_vals:  list[float] = []
        self._risk_vals:    list[float] = []
        self._terminal_eps: list[int]   = []
        self._good_terminal_eps: list[int] = []
        self._bad_terminal_eps: list[int] = []
        self._truncated_eps: list[int] = []
        # Routing actions (dim 0 for MultiDiscrete, or the single action for Discrete)
        self._routing_actions: list[int] = []
        # Management actions (dim 1 for MultiDiscrete; empty for Discrete)
        self._mgmt_actions:    list[int] = []
        # Whether the env uses MultiDiscrete — detected on first step
        self._is_multidiscrete: bool = False

        self._last_log_step  = 0
        self._last_log_time  = 0.0
        self._total_episodes = 0

        # Per-env episode accumulators (VecEnv-safe)
        self._ep_reward_buf: np.ndarray | None = None
        self._ep_len_buf:    np.ndarray | None = None

    # ------------------------------------------------------------------
    # SB3 lifecycle
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._ep_reward_buf = np.zeros(n, dtype=np.float64)
        self._ep_len_buf    = np.zeros(n, dtype=np.int32)
        self._last_log_time = time.time()

        # Detect action space type
        import gymnasium as gym
        try:
            aspace = self.training_env.action_space
            self._is_multidiscrete = isinstance(aspace, gym.spaces.MultiDiscrete)
        except Exception:
            self._is_multidiscrete = False

        # Build CSV fieldnames — include per-management-action columns if MultiDiscrete
        mgmt_actions, n_mgmt = _get_mgmt_actions()
        mgmt_rate_cols = (
            [f"mgmt_{a.name}_rate" for a in mgmt_actions]
            if self._is_multidiscrete and mgmt_actions else []
        )
        self._mgmt_rate_cols = mgmt_rate_cols

        fieldnames = [
            "timestep", "episodes",
            "ep_reward_mean", "ep_reward_std",
            "ep_reward_p10", "ep_reward_p50", "ep_reward_p90",
            "ep_len_mean",
            "delay_mean", "rework_mean", "risk_mean",
            "terminal_rate", "good_terminal_rate", "bad_terminal_rate", "truncated_rate",
            "top_action", "action_entropy",          # backward-compat aliases
            "top_routing_action", "routing_entropy",
            "top_mgmt_action", "mgmt_entropy",
        ] + mgmt_rate_cols + [
            "value_loss", "policy_loss", "entropy_loss",
            "explained_variance", "clip_fraction",
            "fps",
        ]

        if self._csv_path:
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
            self._csv_file   = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames,
                                              extrasaction='ignore')
            self._csv_writer.writeheader()
            self._csv_file.flush()

        if self.verbose >= 1:
            print(_SEP, flush=True)
            print(_HDR, flush=True)
            print(_SEP, flush=True)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(self.training_env.num_envs))
        dones   = self.locals.get("dones",   np.zeros(self.training_env.num_envs, dtype=bool))
        actions = self.locals.get("actions", [])
        infos   = self.locals.get("infos",   [])

        self._ep_reward_buf += rewards
        self._ep_len_buf    += 1

        # Handle both Discrete (scalar per env) and MultiDiscrete (array per env)
        for a in actions:
            a = np.asarray(a)
            if a.ndim == 0:
                # Discrete: single scalar action
                self._routing_actions.append(int(a))
            else:
                # MultiDiscrete: [routing_idx, mgmt_idx, ...]
                self._routing_actions.append(int(a[0]))
                if len(a) > 1:
                    self._mgmt_actions.append(int(a[1]))

        for i, (done, info) in enumerate(zip(dones, infos)):
            kpi = info.get("kpi", {})
            if kpi:
                self._delay_vals.append(float(kpi.get("delay_proxy", 0.0)))
                self._rework_vals.append(float(kpi.get("rework_norm", 0.0)))
                self._risk_vals.append(float(kpi.get("risk_score", 0.0)))

            if done:
                self._ep_rewards.append(float(self._ep_reward_buf[i]))
                self._ep_lengths.append(int(self._ep_len_buf[i]))
                is_terminal = int(kpi.get("is_terminal", 0))
                is_good = int(kpi.get("is_good", 0))
                self._terminal_eps.append(is_terminal)
                self._good_terminal_eps.append(1 if (is_terminal and is_good) else 0)
                self._bad_terminal_eps.append(1 if (is_terminal and not is_good) else 0)
                self._truncated_eps.append(1 if not is_terminal else 0)
                self._ep_reward_buf[i] = 0.0
                self._ep_len_buf[i]    = 0
                self._total_episodes  += 1

        if self.num_timesteps - self._last_log_step >= self.log_interval:
            self._flush_log()
            self._last_log_step = self.num_timesteps

        return True

    # ------------------------------------------------------------------
    # Core flush
    # ------------------------------------------------------------------

    def _flush_log(self) -> None:
        now     = time.time()
        elapsed = max(now - self._last_log_time, 1e-6)
        fps     = self.log_interval / elapsed
        self._last_log_time = now

        # ── Episode stats ──────────────────────────────────────────────
        def _mean(lst): return float(np.mean(lst)) if lst else float("nan")
        def _std(lst):  return float(np.std(lst))  if lst else float("nan")

        rew_mean  = _mean(self._ep_rewards)
        rew_std   = _std(self._ep_rewards)
        rew_p10   = float(np.percentile(self._ep_rewards, 10)) if self._ep_rewards else float("nan")
        rew_p50   = float(np.percentile(self._ep_rewards, 50)) if self._ep_rewards else float("nan")
        rew_p90   = float(np.percentile(self._ep_rewards, 90)) if self._ep_rewards else float("nan")
        len_mean  = _mean(self._ep_lengths)
        delay     = _mean(self._delay_vals)
        rework    = _mean(self._rework_vals)
        risk      = _mean(self._risk_vals)
        term_rate = _mean(self._terminal_eps)
        good_rate = _mean(self._good_terminal_eps)
        bad_rate  = _mean(self._bad_terminal_eps)
        trunc_rate = _mean(self._truncated_eps)

        # ── Action distribution ────────────────────────────────────────
        def _action_stats(action_list: list[int], n_bins: int | None = None) -> tuple[str, float]:
            """Returns (top_action_str, entropy) for a list of action indices."""
            if not action_list:
                return "n/a", float("nan")
            counts    = Counter(action_list)
            top       = str(counts.most_common(1)[0][0])
            total     = sum(counts.values())
            n_bins    = n_bins or (max(counts.keys()) + 1)
            probs     = np.array([counts.get(i, 0) / total for i in range(n_bins)])
            probs_nz  = probs[probs > 0]
            entropy   = float(-np.sum(probs_nz * np.log(probs_nz + 1e-10)))
            return top, entropy

        top_routing, routing_entropy = _action_stats(self._routing_actions)
        top_mgmt,    mgmt_entropy    = _action_stats(self._mgmt_actions)

        # Per-management-action usage rates
        mgmt_actions, n_mgmt = _get_mgmt_actions()
        mgmt_rate_row: dict = {}
        if self._is_multidiscrete and self._mgmt_actions and mgmt_actions:
            total_mgmt = len(self._mgmt_actions)
            mgmt_counts = Counter(self._mgmt_actions)
            for a in mgmt_actions:
                col = f"mgmt_{a.name}_rate"
                mgmt_rate_row[col] = round(mgmt_counts.get(a.index, 0) / max(total_mgmt, 1), 4)

        # Backward-compat aliases
        top_action = top_routing
        entropy    = routing_entropy

        # ── PPO train stats (from SB3 internal logger) ────────────────
        # Available after the first update; nan until then.
        def _sb3_stat(key: str) -> float:
            try:
                val = self.model.logger.name_to_value.get(key, float("nan"))
                return float(val) if val is not None else float("nan")
            except Exception:
                return float("nan")

        value_loss   = _sb3_stat("train/value_loss")
        policy_loss  = _sb3_stat("train/policy_gradient_loss")
        entropy_loss = _sb3_stat("train/entropy_loss")
        expl_var     = _sb3_stat("train/explained_variance")
        clip_frac    = _sb3_stat("train/clip_fraction")

        # ── CSV row (verbose) ──────────────────────────────────────────
        row = {
            "timestep":           self.num_timesteps,
            "episodes":           self._total_episodes,
            "ep_reward_mean":     round(rew_mean,   3),
            "ep_reward_std":      round(rew_std,    3),
            "ep_reward_p10":      round(rew_p10,    3) if not np.isnan(rew_p10) else "",
            "ep_reward_p50":      round(rew_p50,    3) if not np.isnan(rew_p50) else "",
            "ep_reward_p90":      round(rew_p90,    3) if not np.isnan(rew_p90) else "",
            "ep_len_mean":        round(len_mean,   1),
            "delay_mean":         round(delay,      3),
            "rework_mean":        round(rework,     3),
            "risk_mean":          round(risk,       3),
            "terminal_rate":      round(term_rate,  3),
            "good_terminal_rate": round(good_rate,  3),
            "bad_terminal_rate":  round(bad_rate,   3),
            "truncated_rate":     round(trunc_rate, 3),
            # backward-compat aliases
            "top_action":         top_action,
            "action_entropy":     round(entropy,    3) if not np.isnan(entropy) else "",
            # routing dimension
            "top_routing_action": top_routing,
            "routing_entropy":    round(routing_entropy, 3) if not np.isnan(routing_entropy) else "",
            # management dimension
            "top_mgmt_action":    top_mgmt,
            "mgmt_entropy":       round(mgmt_entropy, 3) if not np.isnan(mgmt_entropy) else "",
            # PPO stats
            "value_loss":         round(value_loss,  1) if not np.isnan(value_loss)  else "",
            "policy_loss":        round(policy_loss, 5) if not np.isnan(policy_loss) else "",
            "entropy_loss":       round(entropy_loss,3) if not np.isnan(entropy_loss)else "",
            "explained_variance": round(expl_var,   3)  if not np.isnan(expl_var)    else "",
            "clip_fraction":      round(clip_frac,  4)  if not np.isnan(clip_frac)   else "",
            "fps":                round(fps,        1),
        }
        row.update(mgmt_rate_row)  # per-management-action usage rates

        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        # Feed interval mean to early stopping callback if attached
        if self.early_stopping is not None and not np.isnan(rew_mean):
            self.early_stopping.record_interval_mean(rew_mean)

        # ── Terminal line (compact) ────────────────────────────────────
        if self.verbose >= 1:
            # Reprint header periodically so it's always visible
            if self._log_count % _HEADER_EVERY == 0 and self._log_count > 0:
                print(_SEP, flush=True)
                print(_HDR, flush=True)
                print(_SEP, flush=True)

            ev_str = f"{expl_var:6.3f}" if not np.isnan(expl_var) else "   n/a"

            print(
                f"{self.num_timesteps:>8d}  "
                f"{self._total_episodes:>5d}  "
                f"{rew_mean:>+8.2f}  "
                f"{rew_std:>6.2f}  "
                f"{term_rate:>4.0%}  "
                f"{good_rate:>4.0%}  "
                f"{bad_rate:>4.0%}  "
                f"{trunc_rate:>4.0%}  "
                f"{entropy:>5.2f}  "
                f"{ev_str}  "
                f"{fps:>5.0f}",
                flush=True,
            )
            self._log_count += 1

        # ── Clear rolling buffers ──────────────────────────────────────
        self._ep_rewards.clear()
        self._ep_lengths.clear()
        self._delay_vals.clear()
        self._rework_vals.clear()
        self._risk_vals.clear()
        self._terminal_eps.clear()
        self._good_terminal_eps.clear()
        self._bad_terminal_eps.clear()
        self._truncated_eps.clear()
        self._routing_actions.clear()
        self._mgmt_actions.clear()

    def _on_training_end(self) -> None:
        if self._ep_rewards or self._routing_actions:
            self._flush_log()
        if self.verbose >= 1:
            print(_SEP, flush=True)
        if self._csv_file:
            self._csv_file.close()
