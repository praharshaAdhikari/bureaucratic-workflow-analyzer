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
from reward_config import RewardConfig
from intervention_effects import (
    scaled_duration_multiplier,
    scaled_cost,
    scaled_compliance_risk,
    MIN_CUMULATIVE_MULTIPLIER,
    MAX_CUMULATIVE_MULTIPLIER,
)
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
        reward_config: Optional["RewardConfig"] = None,
        min_steps_to_outcome: int = 1,
        verdict_mode: str = "environment",
        outcome_base_rates: Optional[dict] = None,
        real_median_cycle_s: Optional[float] = None,
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
        # All weights come from one RewardConfig so that training, evaluation
        # and analysis cannot drift apart. They used to be set here as
        # defaults and then overwritten in the training notebook only, which
        # meant the agent was graded on a different reward than it learned.
        # See src/reward_config.py.
        self.reward_config = reward_config if reward_config is not None else RewardConfig()

        median_len  = max(kpi_baselines.get("median_trace_length", 20), 1)
        mean_rework = max(kpi_baselines.get("mean_rework", 1.0), 0.1)
        self._baseline_loop_rate = mean_rework / median_len
        self._median_len = median_len

        self.w_terminal      = self.reward_config.w_terminal
        self.w_bad_terminal  = self.reward_config.w_bad_terminal
        self.w_loop          = self.reward_config.w_loop
        self.w_length_bonus  = self.reward_config.w_length_bonus

        # Per-step shaping is a share of the outcome spread over one
        # median-length episode, so the same config induces the same reward
        # structure on a 11-step process and a 45-step one.
        self.w_progress, self.w_step = self.reward_config.per_step_weights(median_len)

        # Cycle-time and intervention weights, as shares of the outcome bonus.
        # The interventions change how long a case takes, so duration has to be
        # in the reward or the catalogue stays decorative however it is wired.
        self.w_time = self.reward_config.time_share * self.reward_config.w_terminal
        self.time_penalty_mode = self.reward_config.time_penalty_mode
        self.time_penalty_cap = self.reward_config.time_penalty_cap
        # Normalised by median trace length, exactly as w_step is. A charge
        # levied per step must be, or the same config costs eight times more on
        # a 45-step process than an 11-step one — the scale bug from Fix 3.
        # With share = 1.0, intervening on every step of a median-length case
        # with a cost-0.2 action spends 0.2 x w_terminal in total.
        self.w_intervention = (
            self.reward_config.intervention_cost_share
            * self.reward_config.w_terminal / max(median_len, 1)
        )
        self.effect_scale = self.reward_config.effect_scale

        # Median real case duration, the yardstick the cycle-time charge is
        # measured against. None disables the charge rather than inventing a
        # scale — a reward term with a made-up denominator is worse than none.
        self._real_median_cycle_s = (
            float(real_median_cycle_s) if real_median_cycle_s else None
        )

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

        # ── Outcome floor ──────────────────────────────────────────────────
        # An episode may not reach an outcome sooner than the fastest real
        # case does. Every edge in the transition graph is real, but a
        # first-order Markov chain composes them into paths no case ever took:
        # on BPIC2015 the shortest simulated route to "permit irrevocable" is
        # 2 steps against a real minimum of 11, and the agent learned to take
        # it in every episode. Per-edge masking cannot catch that, because the
        # path is only implausible as a whole.
        #
        # Below the floor, terminal activities are removed from the routing
        # mask. If an activity has no non-terminal successor at all, the floor
        # is waived for that step rather than trapping the episode.
        self.min_steps_to_outcome: int = max(1, int(min_steps_to_outcome))

        # ── Who decides the verdict ────────────────────────────────────────
        # "environment" (default): the agent routes only among non-terminal
        #   activities. The environment draws the verdict once per episode
        #   from the log's base rate, and decides *when* the case concludes
        #   using the empirical probability of concluding from the current
        #   activity. The agent therefore controls how a case is handled and
        #   how long it takes, but not whether it succeeds.
        # "agent" (ablation): the pre-fix behaviour, where routing to a
        #   terminal was an ordinary action. Kept so the difference can be
        #   measured rather than asserted.
        #
        # Why: "route to A_APPROVED" was always available and always worth the
        # outcome bonus, so the agent reached a good outcome 94.8% of the time
        # on BPIC2012 against a real rate of 17.7%. None of the 15 management
        # actions plausibly changes whether an applicant is creditworthy, so
        # letting the policy pick the verdict does not model anything real.
        if verdict_mode not in {"environment", "agent"}:
            raise ValueError(
                f"verdict_mode must be 'environment' or 'agent', got {verdict_mode!r}"
            )
        self.verdict_mode = verdict_mode

        rates = outcome_base_rates or {}
        self._p_good = float(rates.get("p_good", 0.5))
        self._good_weights = dict(rates.get("good_terminal_weights") or {})
        self._bad_weights = dict(rates.get("bad_terminal_weights") or {})
        if verdict_mode == "environment" and not rates:
            raise ValueError(
                "verdict_mode='environment' needs outcome_base_rates from "
                "feature_engineering.outcome_base_rates(). Without it the "
                "environment cannot draw a verdict at the real base rate."
            )

        # Empirical chance that a case concludes at each activity, used to
        # decide when — not whether — the case ends.
        self._p_conclude: dict[str, float] = {
            act: float(sum(p for t, p in trans.items() if t in self._all_terminals))
            for act, trans in twin.transition_probs.items()
        }

        # Drawn fresh each episode in reset().
        self._verdict_good: bool = True

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
        self._no_move: bool = False

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

        # The verdict is a property of the case, fixed before the agent acts
        # and invisible to it. Drawing it here rather than letting routing
        # reach it is what stops the policy from approving every application.
        self._verdict_good = bool(self._rng.random() < self._p_good)
        self._no_move = False
        # Cumulative effect of the interventions taken this episode.
        self._speed_multiplier = 1.0
        self._intervention_charge = 0.0

        # Elapsed case time, accumulated from the activities actually visited.
        # The twin fits each activity's duration as the gap to the next event,
        # so summing them over a trace reconstructs first-to-last elapsed time
        # on the same scale as the real log.
        self._elapsed_s = 0.0

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

        # Advance the workload signal once for this step, before anything
        # reads it, so the mask and the validity check see the same state.
        self._advance_volume_pressure()

        # ── Apply management action first (modifies episode state) ─────────
        kpi_vec = self._build_kpi_vec()
        self._episode_state["_at_terminal"] = False  # not terminal yet this step
        mgmt_reward = apply_management_action(
            mgmt_action, self._episode_state, self.twin, kpi_vec
        )
        self._last_mgmt_action = mgmt_action

        # The intervention's causal effect: it changes how long the rest of the
        # case takes, and it costs something. Bounded and multiplicative, so
        # repeated use has diminishing returns instead of driving duration to
        # zero.
        _mgmt_name = MANAGEMENT_ACTIONS[mgmt_action].name
        self._speed_multiplier = float(np.clip(
            self._speed_multiplier * scaled_duration_multiplier(_mgmt_name, self.effect_scale),
            MIN_CUMULATIVE_MULTIPLIER, MAX_CUMULATIVE_MULTIPLIER,
        ))
        self._intervention_charge += (
            scaled_cost(_mgmt_name, self.effect_scale)
            + scaled_compliance_risk(_mgmt_name, self.effect_scale)
        )

        # Sync volume pressure from episode state (management actions may change it)
        self._volume_pressure = float(self._episode_state.get("_volume_pressure", self._volume_pressure))

        # ── Routing step ───────────────────────────────────────────────────
        successors = self._successors.get(self._current_activity, [])
        had_alternative = any(a not in self._all_terminals for a in successors)
        above_floor = self._step + 1 >= self.min_steps_to_outcome

        # A state whose only successors are terminals, reached before the
        # floor, is a dead end the model cannot continue from. Recording an
        # outcome there would reintroduce exactly the implausibly short
        # trajectory the floor exists to prevent, so the episode is abandoned
        # without an outcome instead.
        dead_end_below_floor = (not had_alternative) and not above_floor

        if dead_end_below_floor:
            # Nowhere legal to go, and too early for an outcome. Previously
            # this fell through to `twin._sample_next_activity`, whose global
            # fallback distribution teleports the case to a frequently-seen
            # activity that is not a successor of the current one — the last
            # remaining source of impossible transitions (A1.2 in the brief).
            # Observed once in ~7,000 steps on BPIC2015, from
            # 'environmental permit decision suspended', which has no outgoing
            # edges at all.
            #
            # The episode is already truncating, so end it where it stands
            # rather than inventing a move.
            next_act = self._current_activity
            terminal = False
            self._no_move = True
        elif self.verdict_mode == "environment":
            # The environment decides *when* the case concludes, using the
            # log's own chance of concluding from here. The agent influences
            # that only through where it routes, which is legitimate control
            # over cycle time.
            p_conclude = self._p_conclude.get(self._current_activity, 0.0)
            concludes = (not dead_end_below_floor) and (
                (not had_alternative)                       # nowhere else to go
                or (above_floor and self._rng.random() < p_conclude)
            )
            if concludes:
                next_act = self._draw_terminal()
                terminal = True
            else:
                next_act = self._route_non_terminal(routing_action, successors)
                terminal = False
        else:
            # Ablation: routing to a terminal is an ordinary action.
            if routing_action < len(successors):
                next_act = successors[routing_action]
            else:
                # Fallback: sample from Markov chain (shouldn't happen with masking)
                next_act = self.twin._sample_next_activity(self._current_activity)
            # Honour an outcome only at or after the floor. The exception is a
            # state with no non-terminal successor, where the mask had to
            # offer a terminal anyway. Mirrors the waiver in action_masks().
            terminal = next_act in self._all_terminals and above_floor

        # Charge the time spent on the activity we are leaving. Durations were
        # fitted as the gap to the following event, so this accumulates to the
        # case's first-to-last elapsed time — the same quantity the real log
        # reports, and one the agent's routing genuinely drives.
        self._elapsed_s += float(
            self.twin._sample_duration(self._current_activity) * self._speed_multiplier
        )

        self._current_activity = next_act
        if not self._no_move:
            self._trace.append(next_act)
        self._step += 1

        truncated = self._step >= self.max_steps or dead_end_below_floor
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
            "cycle_time_s":      self._elapsed_s,
            "env_no_move":       bool(self._no_move),
            "verdict_good":      bool(self._verdict_good),
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

    def _draw_terminal(self) -> str:
        """
        Pick the activity the case ends at, matching the verdict drawn in
        reset(), weighted by how often each one ends a real case.
        """
        weights = self._good_weights if self._verdict_good else self._bad_weights
        pool = self._good_terminals if self._verdict_good else self._bad_terminals
        candidates = [a for a in weights if a in pool] or sorted(pool)
        if not candidates:
            # No terminal of the drawn class exists — fall back to the other
            # class rather than failing to end the episode.
            candidates = sorted(self._all_terminals)
            self._verdict_good = candidates and candidates[0] in self._good_terminals
        probs = np.array([weights.get(a, 0.0) for a in candidates], dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(len(candidates), dtype=float)
        probs /= probs.sum()
        return str(self._rng.choice(candidates, p=probs))

    def _route_non_terminal(self, routing_action: int, successors: list[str]) -> str:
        """Apply the agent's routing choice, skipping terminals."""
        options = [a for a in successors if a not in self._all_terminals]
        if not options:
            return self.twin._sample_next_activity(self._current_activity)
        if routing_action < len(successors):
            chosen = successors[routing_action]
            if chosen in options:
                return chosen
        # Mask should have prevented this; pick the first legal option.
        return options[0]

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

        # Terminals are never offered as a routing choice when the environment
        # owns the verdict. Below the outcome floor they are withheld in
        # either mode. Both waive when every successor is a terminal, so the
        # episode is never left without a legal move.
        withhold_terminals = (
            self.verdict_mode == "environment"
            or self._step + 1 < self.min_steps_to_outcome
        )
        if withhold_terminals:
            blocked = routing_mask.copy()
            for i, act in enumerate(successors):
                if act in self._all_terminals:
                    blocked[i] = False
            if blocked.any():
                routing_mask = blocked

        if not routing_mask.any():
            routing_mask[0] = True

        # Management mask based on current KPI signals and episode state
        kpi_vec  = self._build_kpi_vec()
        mgmt_mask = get_management_mask(kpi_vec, self._episode_state)

        return np.concatenate([routing_mask, mgmt_mask])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _cycle_time_deviation(self) -> float:
        """
        How far this episode's elapsed time sits from the real median, in log2
        units, as a non-negative quantity to be charged.

        Charged on log2 of the ratio, not the ratio itself. A linear charge
        lets one slow case dominate the episode return; a hard cap on the ratio
        kills the gradient exactly where the agent operates — a random policy
        already sits at 3.6x the real median on BPIC2012, so with a cap at 3.0
        every improvement inside that range earned nothing and no policy could
        beat doing nothing. log2 doubles the charge for each doubling of
        duration, so it always has a gradient and never explodes.

        Two-sided by default. The original term floored the charge at zero
        below the real median, on the reasoning that paying the agent to go
        faster than reality would reward implausibility. It does not pay the
        agent to go faster — but it does make going faster *free*, and the
        per-step cost then breaks the tie. The agent duly raced: median
        simulated cycle time came out 16x, 5x and 4x faster than the real logs
        on BPIC2012 / BPIC2015 / BPIC2017, further from reality than before the
        term existed. Charging both directions makes the real median the
        target rather than the floor.

        Returns 0.0 when there is no real median to measure against — a reward
        term with an invented denominator is worse than no term.
        """
        if not self._real_median_cycle_s or self._real_median_cycle_s <= 0:
            return 0.0

        ratio = self._elapsed_s / self._real_median_cycle_s
        deviation = float(np.log2(max(ratio, 1e-12)))
        if self.time_penalty_mode == "slow_only":
            deviation = max(0.0, deviation)
        else:
            deviation = abs(deviation)
        return min(deviation, self.time_penalty_cap)

    def _compute_reward(self, terminal: bool, activity: str = "") -> float:
        reward = -self.w_step  # per-step cost

        if terminal:
            # When the environment owns the verdict the agent cannot affect
            # whether the case succeeds, so scoring it on the outcome would be
            # pure noise. What it *can* do is conclude the case, and do so in
            # a plausible number of steps — that is what is rewarded. The
            # good/bad split is still recorded in `info` and becomes a
            # fidelity check (does the simulation reproduce the real base
            # rate?) rather than a performance claim.
            if self.verdict_mode == "environment":
                reward += self.w_terminal
                length_ratio = len(self._trace) / max(self._median_len, 1)
                reward += max(0.0, self.w_length_bonus * (1.0 - abs(length_ratio - 1.0)))

                # Charge for how long the case actually took, relative to the
                # median real case. This is what gives the managerial
                # interventions something to bite on: they change duration, so
                # without it their effect is invisible and the catalogue is
                # decorative however it is wired.
                reward -= self.w_time * self._cycle_time_deviation()

                # And charge for the interventions used to get there, so
                # intervening is a trade-off rather than free.
                reward -= self.w_intervention * self._intervention_charge

                return float(reward)

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

    def _advance_volume_pressure(self) -> None:
        """
        Take one random-walk step of the workload signal.

        Called exactly once per environment step, from ``step()``.

        This used to live inside ``_build_kpi_vec``, which is called three
        times per step — from ``action_masks()``, from ``_get_obs()`` and from
        ``step()`` — with two consequences. The walk advanced three times
        faster than intended; and, worse, the action mask was built from a
        different KPI vector than the validity check applied a moment later
        inside ``apply_management_action``. When volume sat near a threshold
        the two disagreed, and the agent was charged the -0.1 invalid-action
        penalty for choosing an action the mask had just told it was legal.

        Measured before this fix: 9 steps in 3,964 on BPIC2012, 8 in 10,128 on
        BPIC2015, 15 in 9,131 on BPIC2017 — every single disagreement produced
        the penalty. Note this contradicts A7.1 of the remediation brief, which
        states the -0.1 term can never fire because masking suppresses invalid
        actions. It fires; masking is only as good as the state it was
        computed from.
        """
        self._volume_pressure = float(np.clip(
            self._volume_pressure + self._rng.normal(0, 0.05), -2.0, 3.0
        ))
        self._episode_state["_volume_pressure"] = self._volume_pressure

    def _build_kpi_vec(self) -> np.ndarray:
        """
        Build the 7-element KPI signal vector from current episode state.

        Pure: reading the KPI vector must not change it, or repeated reads
        within one step disagree with each other.
        """
        tp = self._terminal_proximity()

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
