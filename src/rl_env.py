"""
rl_env.py
---------
Gymnasium environment for process routing optimisation via Maskable PPO.

C2 redesign: the agent controls two simultaneous decisions at each step:

  1. ROUTING action  — which activity to transition to next (Discrete(max_successors))
  2. MANAGEMENT action — which KPI-based management intervention to apply
                         (Discrete(N_MANAGEMENT_ACTIONS = 15))

Combined action space: MultiDiscrete([max_successors, 15])

The 15 management actions are defined in kpi_actions.py and correspond to
the rule_descriptions dict:
  0  assign_to_primary_team          (default no-op)
  1  outsource_to_volunteer_pool
  2  rebalance_overloaded_queue
  3  merge_tasks_under_role
  4  prioritize_urgent_case
  5  defer_until_objections_resolved
  6  escalate_to_higher_authority
  7  skip_optional_subprocess
  8  add_temporary_staff
  9  adjust_staffing_by_case_volume
  10 enable_cross_trained_pool
  11 relax_rules_for_low_risk
  12 trigger_high_cost_escalation
  13 reroute_from_overloaded_employee
  14 close_case

Observation space (MultiInputDict):
  - case_embedding    : float32[embed_dim]              mean-pooled activity embedding
  - kpi_signals       : float32[7]                      [delay_norm, rework_norm, loop_rate,
                                                          case_age_norm, terminal_proximity,
                                                          volume_pressure, step_frac]
  - resource_state    : float32[n_resources]            normalised load per resource slot
  - mgmt_action_mask  : float32[N_MANAGEMENT_ACTIONS]   1.0 where action is valid, 0.0 otherwise

Action masks (for MaskablePPO):
  Combined boolean mask of shape (max_successors + N_MANAGEMENT_ACTIONS,).
  First max_successors entries mask routing actions.
  Next N_MANAGEMENT_ACTIONS entries mask management actions.
  Management action 0 (default) is always unmasked.

Reward: shaped to minimise trace length while reaching terminal.
  +w_terminal      on reaching a good terminal activity
  +w_length_bonus  shaped bonus peaking when episode length == empirical median
  -w_loop          each time the same activity repeats (excess loops)
  +w_progress      each step that moves toward terminal (terminal_proximity increases)
  -w_step          per-step cost to incentivise efficiency (0.1, raised from 0.05)
  ±mgmt_delta      shaped reward from the management action (from kpi_actions.py)
  w_bad_terminal   hard fixed penalty for bad terminal (not additive with w_terminal)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from feature_engineering import ActivityEmbedder, embed_trace
from feature_engineering import classify_bad_terminals
from kpi_actions import (
    MANAGEMENT_ACTIONS,
    N_MANAGEMENT_ACTIONS,
    get_management_mask,
    apply_management_action,
)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ProcessEnv(gym.Env):
    """
    Single-case process routing + management environment.

    Each episode = one simulated case.
    At each step the agent makes two simultaneous decisions:
      1. Routing: which activity to transition to next (Discrete(max_successors))
      2. Management: which KPI-based intervention to apply (Discrete(15))

    Combined action space: MultiDiscrete([max_successors, N_MANAGEMENT_ACTIONS])

    Invalid routing actions (successors that don't exist from the current
    activity) are masked out via action_masks().  Invalid management actions
    (those whose KPI preconditions are not met) are also masked out, but the
    default action (index 0) is always valid.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        twin,
        embed_model: ActivityEmbedder,
        kpi_baselines: dict,
        n_resources: int = 20,
        embed_dim: int = 32,
        max_steps: int = 150,
        seed: int = 42,
        bad_terminals: Optional[set] = None,
        good_terminals: Optional[set] = None,
    ):
        super().__init__()
        self.twin       = twin
        self.embed_model = embed_model
        self.max_steps  = max_steps
        self._rng       = np.random.default_rng(seed)

        # ── Build global sorted activity list and successor index map ──────
        self._all_activities: list[str] = sorted(twin.activities)
        self._act_to_idx: dict[str, int] = {a: i for i, a in enumerate(self._all_activities)}
        n_acts = len(self._all_activities)

        # max_successors = largest out-degree in the transition graph
        self._max_succ: int = max(
            len(v) for v in twin.transition_probs.values()
        ) if twin.transition_probs else 1

        # For each activity, sorted list of successor activities (stable order)
        self._successors: dict[str, list[str]] = {
            act: sorted(trans.keys())
            for act, trans in twin.transition_probs.items()
        }

        # ── Reward weights ─────────────────────────────────────────────────
        self.w_terminal      = 0.0    # no bonus just for closing — outcome matters
        self.w_bad_terminal  = -30.0  # hard penalty for bad terminal (declined/cancelled)
                                      # NOTE: bad terminals receive ONLY this value (no
                                      # w_terminal bonus added), making the signal
                                      # unambiguously negative regardless of w_terminal tuning.
        self.w_loop          = 1.0    # penalty per excess loop above baseline rate
        self.w_progress      = 0.3    # bonus for advancing to a new activity
        self.w_step          = 0.05   # per-step cost (tunable via reward_tuning)
        self.w_length_bonus  = 5.0    # max bonus awarded when episode length == median
                                      # (Fix 2): shaped bonus that peaks at _median_len
                                      # and decays linearly with |length - median|

        median_len  = max(kpi_baselines.get("median_trace_length", 20), 1)
        mean_rework = max(kpi_baselines.get("mean_rework", 1.0), 0.1)
        self._baseline_loop_rate = mean_rework / median_len
        self._median_len = median_len

        # ── Terminal classification (loaded from feature engineering artefact) ──
        # classify_bad_terminals() is run once in notebook 02 and saved to
        # terminal_classification.json alongside the other artefacts.
        # ProcessEnv accepts an optional `bad_terminals` set; if not provided
        # it falls back to running classify_bad_terminals() directly.
        if bad_terminals is not None:
            self._bad_terminals = set(bad_terminals)
        else:
            from feature_engineering import classify_bad_terminals
            self._bad_terminals = classify_bad_terminals(twin.activities)

        # ── Good terminal inference ────────────────────────────────────────
        # Priority order:
        #   1. Caller passes good_terminals explicitly (highest priority)
        #   2. Keyword + graph-structure inference (dynamic fallback)
        #
        # The recommended path is for the caller (notebook 04) to load
        # good_terminals from terminal_classification.json (written by notebook 02
        # using classify_good_terminals, which combines empirical trace-ending
        # frequency with keyword filtering and W_-prefix exclusion).
        # That is more reliable than any graph-based heuristic, especially for
        # cyclic processes like BPIC2017 where every activity has outgoing edges.
        if good_terminals is not None:
            # Caller passed good_terminals explicitly — trust it, just exclude bads
            self._good_terminals: set = set(good_terminals) - self._bad_terminals
        else:
            # ── Dynamic fallback ──────────────────────────────────────────
            # Used when terminal_classification.json has no good_terminals key
            # (old artefacts) or when ProcessEnv is constructed without one.
            #
            # Strategy:
            #   A. Keyword scan over all non-W_ activities
            #   B. Include twin.terminal_activities that pass two filters:
            #      - Not a W_/w_ subprocess prefix (work-queue loops)
            #      - Not a dominant self-loop (>50% self-transition)
            #   Both sets are unioned; no mid-process filter is applied to
            #   keyword matches because cyclic processes have no dead-end nodes.
            _GOOD_KEYWORDS = (
                "accept", "approv", "final", "complet", "grant", "success",
                "paid", "closed", "done", "finish", "confirm",
            )
            _BAD_KEYWORDS = (
                "cancel", "declin", "refus", "reject", "denied", "withdraw",
                "suspend", "abort", "fail", "incomplet",
            )
            _SUBPROCESS_PREFIXES = ("W_", "w_")

            inferred_good: set = set()
            for act in twin.activities:
                if any(act.startswith(pfx) for pfx in _SUBPROCESS_PREFIXES):
                    continue
                act_lower = act.lower()
                if (any(kw in act_lower for kw in _GOOD_KEYWORDS)
                        and not any(kw in act_lower for kw in _BAD_KEYWORDS)):
                    inferred_good.add(act)

            def _is_subprocess_or_loop(act: str) -> bool:
                """True for W_ prefix activities or dominant self-loops (>50%)."""
                if any(act.startswith(pfx) for pfx in _SUBPROCESS_PREFIXES):
                    return True
                return twin.transition_probs.get(act, {}).get(act, 0.0) > 0.5

            twin_good = {
                act for act in (set(twin.terminal_activities) - self._bad_terminals)
                if not _is_subprocess_or_loop(act)
            }

            self._good_terminals = inferred_good | twin_good

        # All terminals = bad + good (union of everything that ends an episode)
        self._all_terminals: set = self._bad_terminals | self._good_terminals

        # Warn if no good terminals found — agent will never get a positive outcome
        if not self._good_terminals:
            import warnings
            warnings.warn(
                f"ProcessEnv: no good terminals found in {len(twin.activities)} activities. "
                "All episodes will end at bad terminals. Check terminal_classification.json.",
                stacklevel=2,
            )

        # ── Spaces ─────────────────────────────────────────────────────────
        self._resource_list = list(twin.resource_pool.capacities.keys())[:n_resources]
        while len(self._resource_list) < n_resources:
            self._resource_list.append("UNKNOWN")
        self.n_resources = n_resources
        self.embed_dim   = embed_dim

        self.observation_space = spaces.Dict({
            "case_embedding":   spaces.Box(-np.inf, np.inf, shape=(embed_dim,),              dtype=np.float32),
            "kpi_signals":      spaces.Box(-np.inf, np.inf, shape=(7,),                      dtype=np.float32),
            "resource_state":   spaces.Box(0.0, 1.0,        shape=(n_resources,),            dtype=np.float32),
            "mgmt_action_mask": spaces.Box(0.0, 1.0,        shape=(N_MANAGEMENT_ACTIONS,),   dtype=np.float32),
        })

        # Combined action space: [routing_action, management_action]
        self.action_space = spaces.MultiDiscrete([self._max_succ, N_MANAGEMENT_ACTIONS])

        # ── Episode state ──────────────────────────────────────────────────
        self._trace:            list[str] = []
        self._step:             int       = 0
        self._current_activity: str       = ""
        self._done:             bool      = False
        self._prev_terminal_proximity: float = 0.0
        self._volume_pressure: float = 0.0

        # Episode-scoped KPI state flags (modified by management actions)
        self._episode_state: dict = {}

        # Track last management action for info dict
        self._last_mgmt_action: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step  = 0
        self._done  = False
        self._trace = []
        self._volume_pressure = float(self._rng.normal(0.5, 0.5))

        self.twin.reset_cross_training()
        self.twin.resource_pool.current_load = {
            r: 0 for r in self.twin.resource_pool.capacities
        }

        self._current_activity = self.twin._sample_start_activity()
        self._trace.append(self._current_activity)
        self._prev_terminal_proximity = self._terminal_proximity()

        # Reset episode-scoped KPI state flags
        self._episode_state = {
            "_delay_norm":         self._delay_norm(),
            "_rework_norm":        0.0,
            "_volume_pressure":    self._volume_pressure,
            "_risk_high":          False,
            "_objection_active":   False,
            "_suspension_active":  False,
            "_cross_train_active": False,
            "_staffing_boost":     1.0,
            "_skip_optional":      False,
            "_deferred":           False,
            "_merged":             False,
            "_rerouted":           False,
            "_at_terminal":        False,
        }
        self._last_mgmt_action = 0

        return self._get_obs(), {}

    def step(self, action):
        assert not self._done, "Call reset() before step()"

        # ── Unpack combined action ─────────────────────────────────────────
        # action may be a length-2 array [routing_idx, mgmt_idx] or a plain int
        # (plain int = legacy routing-only mode, mgmt defaults to 0)
        if np.ndim(action) == 0 or (hasattr(action, "__len__") and len(action) == 1):
            routing_action = int(action) if np.ndim(action) == 0 else int(action[0])
            mgmt_action    = 0
        else:
            routing_action = int(action[0])
            mgmt_action    = int(action[1])

        # ── Apply management action first (modifies episode state) ─────────
        kpi_vec = self._build_kpi_vec()
        self._episode_state["_at_terminal"] = False  # not terminal yet this step
        mgmt_reward = apply_management_action(
            mgmt_action, self._episode_state, self.twin, kpi_vec
        )
        self._last_mgmt_action = mgmt_action

        # Sync volume pressure from episode state (management actions may change it)
        self._volume_pressure = float(self._episode_state.get("_volume_pressure", self._volume_pressure))

        # ── Routing step ───────────────────────────────────────────────────
        successors = self._successors.get(self._current_activity, [])
        if routing_action < len(successors):
            next_act = successors[routing_action]
        else:
            # Fallback: sample from Markov chain (shouldn't happen with masking)
            next_act = self.twin._sample_next_activity(self._current_activity)

        self._current_activity = next_act
        self._trace.append(next_act)
        self._step += 1

        terminal  = next_act in self._all_terminals
        truncated = self._step >= self.max_steps
        self._done = terminal or truncated

        # Update episode state with current KPI signals
        self._episode_state["_delay_norm"]      = self._delay_norm()
        self._episode_state["_rework_norm"]      = self._rework_norm()
        self._episode_state["_at_terminal"]      = terminal

        # Detect objection/suspension from activity name (heuristic)
        act_lower = next_act.lower()
        if any(kw in act_lower for kw in ("bezwaar", "objection", "appeal", "beroep")):
            self._episode_state["_objection_active"] = True
        if any(kw in act_lower for kw in ("suspend", "opschort", "hold")):
            self._episode_state["_suspension_active"] = True

        routing_reward = self._compute_reward(terminal, next_act)
        reward = routing_reward + mgmt_reward
        obs    = self._get_obs()

        delay_proxy = self._delay_norm()
        rework_norm = self._rework_norm()
        risk_high = bool(self._episode_state.get("_risk_high", False))
        objection = bool(self._episode_state.get("_objection_active", False))
        suspension = bool(self._episode_state.get("_suspension_active", False))
        risk_score = float(np.clip(
            0.45 * delay_proxy
            + 0.45 * rework_norm
            + (0.6 if risk_high else 0.0)
            + (0.3 if objection else 0.0)
            + (0.3 if suspension else 0.0),
            0.0,
            3.0,
        ))

        info = {
            "current_activity":  self._current_activity,
            "trace_len":         len(self._trace),
            "mgmt_action":       mgmt_action,
            "mgmt_action_name":  MANAGEMENT_ACTIONS[mgmt_action].name,
            "mgmt_reward":       mgmt_reward,
            "routing_reward":    routing_reward,
            "kpi": {
                "delay_proxy":   delay_proxy,
                "rework_norm":   rework_norm,
                "risk_score":    risk_score,
                "is_terminal":   int(terminal),
                "is_good":       int(next_act in self._good_terminals),
                "case_age_norm": self._step / self.max_steps,
                "volume_pressure": self._volume_pressure,
                "risk_high":     risk_high,
                "objection":     objection,
                "suspension":    suspension,
            },
        }
        return obs, reward, bool(terminal), bool(truncated), info

    def action_masks(self) -> np.ndarray:
        """
        Combined boolean mask for MaskablePPO.

        Returns a flat array of shape (max_successors + N_MANAGEMENT_ACTIONS,):
          - First max_successors entries: True where routing action is valid
          - Next N_MANAGEMENT_ACTIONS entries: True where management action is valid

        Management action 0 (assign_to_primary_team) is always True.
        """
        # Routing mask
        successors = self._successors.get(self._current_activity, [])
        routing_mask = np.zeros(self._max_succ, dtype=bool)
        routing_mask[:len(successors)] = True
        if not routing_mask.any():
            routing_mask[0] = True

        # Management mask based on current KPI signals and episode state
        kpi_vec  = self._build_kpi_vec()
        mgmt_mask = get_management_mask(kpi_vec, self._episode_state)

        return np.concatenate([routing_mask, mgmt_mask])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, terminal: bool, activity: str = "") -> float:
        reward = -self.w_step  # per-step cost

        if terminal:
            if activity in self._bad_terminals:
                # Bad terminal: hard fixed penalty — NO w_terminal bonus.
                # This is intentionally separated from w_terminal so that
                # reward_tuning cannot accidentally make bad terminals positive
                # by raising w_terminal (the old bug: w_terminal=15 + w_declined=-8 = +7).
                return float(self.w_bad_terminal)
            else:
                # Good terminal: closure bonus only for positive outcomes
                reward += self.w_terminal

                # Length bonus (Fix 2): reward finishing close to the empirical
                # median trace length.  Peaks at w_length_bonus when
                # len(trace) == _median_len, decays linearly to 0 at 2× median.
                # Clamped to [0, w_length_bonus] so it never penalises short episodes.
                length_ratio = len(self._trace) / max(self._median_len, 1)
                length_bonus = self.w_length_bonus * (1.0 - abs(length_ratio - 1.0))
                reward += max(0.0, length_bonus)

                return float(reward)

        # Progress bonus: reward advancing to a new (non-repeated) activity
        if self._current_activity not in self._trace[:-1]:
            reward += self.w_progress

        # Loop penalty: penalise excess repetition above baseline rate
        rework = len(self._trace) - len(set(self._trace))
        excess = max(0.0, rework / max(self._step, 1) - self._baseline_loop_rate)
        reward -= self.w_loop * excess

        return float(reward)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _build_kpi_vec(self) -> np.ndarray:
        """Build the 7-element KPI signal vector from current episode state."""
        tp = self._terminal_proximity()
        self._volume_pressure = float(np.clip(
            self._volume_pressure + self._rng.normal(0, 0.05), -2.0, 3.0
        ))
        # Sync episode state volume pressure
        self._episode_state["_volume_pressure"] = self._volume_pressure

        return np.array([
            self._delay_norm(),
            self._rework_norm(),
            # Loop rate this episode vs baseline
            min(1.0, (len(self._trace) - len(set(self._trace))) / max(self._step, 1)
                / max(self._baseline_loop_rate, 0.01)),
            self._step / self.max_steps,          # case_age_norm
            tp,                                    # terminal_proximity
            np.clip(self._volume_pressure / 3.0, -1, 1),  # normalised volume
            len(self._successors.get(self._current_activity, [])) / max(self._max_succ, 1),
        ], dtype=np.float32)

    def _terminal_proximity(self) -> float:
        """P(next activity is any terminal) from transition probs."""
        trans = self.twin.transition_probs.get(self._current_activity, {})
        return float(sum(
            p for act, p in trans.items()
            if act in self._all_terminals
        ))

    def _delay_norm(self) -> float:
        """Normalised trace length relative to median."""
        return min(3.0, max(0.0,
            (len(self._trace) - self._median_len) / max(self._median_len, 1)
        ))

    def _rework_norm(self) -> float:
        """Normalised rework count."""
        rework = len(self._trace) - len(set(self._trace))
        mean_rework = max(self.twin.kpi_baselines.get("mean_rework", 1.0), 0.1)
        return min(3.0, rework / mean_rework)

    def _get_obs(self) -> dict:
        emb = embed_trace(self._trace, self.embed_model)

        kpi_vec = self._build_kpi_vec()

        # Management action validity mask as float observation
        mgmt_mask = get_management_mask(kpi_vec, self._episode_state).astype(np.float32)

        # Resource state: normalised load for top-N resources
        res_state = np.zeros(self.n_resources, dtype=np.float32)
        for i, res in enumerate(self._resource_list):
            cap  = self.twin.resource_pool.capacities.get(res, 1)
            load = self.twin.resource_pool.current_load.get(res, 0)
            res_state[i] = load / max(cap, 1)

        return {
            "case_embedding":   emb.astype(np.float32),
            "kpi_signals":      kpi_vec,
            "resource_state":   res_state,
            "mgmt_action_mask": mgmt_mask,
        }
