"""
kpi_actions.py
--------------
KPI-based management action space for the RL process environment.

Each of the 15 management actions maps to a named rule with:
  - A human-readable description
  - KPI preconditions that determine when the action is *valid* (used for
    hard masking — invalid actions are suppressed by MaskablePPO)
  - A state-effect function that modifies the episode state when applied

REWARD DESIGN — NO INTRINSIC MANAGEMENT REWARDS
------------------------------------------------
Management actions carry zero intrinsic reward. Their value comes entirely
from their effects on episode state (KPI signals, twin overrides), which
then feed back into the routing reward through better outcomes:

  - Reducing _delay_norm makes the loop penalty smaller next step
  - Enabling cross-training unlocks better routing options
  - Boosting staffing reduces resource contention in the twin

This prevents the agent from farming management rewards by looping to
manufacture valid preconditions (reward hacking). The only rewards are:
  - Routing: +w_progress for new activities, -w_loop for excess loops,
             +w_terminal / +w_declined at episode end, -w_step per step
  - Invalid management action: -0.1 penalty (discourages precondition gaming)

The action index 0 is always "assign_to_primary_team" (the no-op default).
All other actions modify episode state only — no reward delta.

KPI signal indices (matching ProcessEnv kpi_signals vector, length 7):
  0  delay_norm          — normalised trace length vs median (0=fast, 3=very slow)
  1  rework_norm         — normalised rework count (0=clean, 3=heavy rework)
  2  loop_rate           — episode loop rate vs baseline (0=normal, 1=high)
  3  case_age_norm       — step / max_steps (0=early, 1=near timeout)
  4  terminal_proximity  — P(next is terminal) from transition probs
  5  volume_pressure     — normalised rolling case volume (-1=low, 3=high)
  6  step_frac           — successor count fraction (branching factor)

Episode state dict keys written/read by action effects:
  _delay_norm           float  — current delay signal
  _rework_norm          float  — current rework signal
  _volume_pressure      float  — current volume signal
  _risk_high            bool   — risk flag (set by escalation actions)
  _objection_active     bool   — objection/appeal flag
  _suspension_active    bool   — suspension/hold flag
  _cross_train_active   bool   — cross-training pool enabled
  _staffing_boost       float  — multiplier on resource capacity (default 1.0)
  _skip_optional        bool   — optional subprocess skip flag
  _deferred             bool   — case deferred (no routing until resolved)
  _merged               bool   — tasks merged under single role
  _rerouted             bool   — case rerouted away from overloaded employee
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# KPI thresholds (tunable)
# ---------------------------------------------------------------------------

DELAY_HIGH        = 1.0   # delay_norm > this → "high delay"
DELAY_EXTREME     = 2.0   # delay_norm > this → "extreme delay"
REWORK_HIGH       = 1.0   # rework_norm > this → "high rework"
VOLUME_HIGH       = 0.5   # volume_pressure > this → "high volume"
CASE_AGE_HIGH     = 0.6   # case_age_norm > this → "old case"
RISK_HIGH_THRESH  = 1.5   # delay_norm + rework_norm > this → "high risk proxy"
LOOP_HIGH         = 0.7   # loop_rate > this → "high loop rate"


# ---------------------------------------------------------------------------
# Action descriptor
# ---------------------------------------------------------------------------

@dataclass
class ManagementAction:
    """
    Descriptor for a single KPI-based management action.

    Attributes
    ----------
    index : int
        Unique integer index used in the action space.
    name : str
        Short snake_case identifier matching the rule_descriptions keys.
    description : str
        Human-readable description of when/why this action is used.
    is_valid : Callable[[np.ndarray, dict], bool]
        Returns True when the action is contextually valid given the current
        kpi_signals vector and episode_state dict.
    apply : Callable[[dict, object], float]
        Applies the action's effect to episode_state and the twin.
        Returns a shaped reward delta (positive = good, negative = bad).
    """
    index:       int
    name:        str
    description: str
    is_valid:    Callable[[np.ndarray, dict], bool]
    apply:       Callable[[dict, object], float]


# ---------------------------------------------------------------------------
# Helper: read KPI signal by name
# ---------------------------------------------------------------------------

def _kpi(kpi_vec: np.ndarray, idx: int) -> float:
    return float(kpi_vec[idx]) if idx < len(kpi_vec) else 0.0


# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------

def _build_actions() -> list[ManagementAction]:
    """
    Build and return the ordered list of 15 management actions.
    Index 0 = default no-op (assign_to_primary_team).
    """

    actions: list[ManagementAction] = []

    # ── 0. assign_to_primary_team ─────────────────────────────────────────
    # Default action — always valid, no state change, no reward delta.
    actions.append(ManagementAction(
        index=0,
        name="assign_to_primary_team",
        description="Default manager action for active non-terminal cases.",
        is_valid=lambda kpi, state: True,
        apply=lambda state, twin: 0.0,
    ))

    # ── 1. outsource_to_volunteer_pool ────────────────────────────────────
    def _outsource_valid(kpi: np.ndarray, state: dict) -> bool:
        return _kpi(kpi, 5) > VOLUME_HIGH and _kpi(kpi, 0) > DELAY_HIGH

    def _outsource_apply(state: dict, twin) -> float:
        state["_delay_norm"]      = max(0.0, state.get("_delay_norm", 0.0) - 0.3)
        state["_volume_pressure"] = max(-1.0, state.get("_volume_pressure", 0.0) - 0.2)
        return 0.0

    actions.append(ManagementAction(
        index=1,
        name="outsource_to_volunteer_pool",
        description="Valid when workload/volume pressure and delay are both high.",
        is_valid=_outsource_valid,
        apply=_outsource_apply,
    ))

    # ── 2. rebalance_overloaded_queue ─────────────────────────────────────
    def _rebalance_valid(kpi: np.ndarray, state: dict) -> bool:
        return _kpi(kpi, 1) > REWORK_HIGH or _kpi(kpi, 0) > DELAY_HIGH

    def _rebalance_apply(state: dict, twin) -> float:
        state["_rework_norm"] = max(0.0, state.get("_rework_norm", 0.0) - 0.25)
        state["_delay_norm"]  = max(0.0, state.get("_delay_norm", 0.0) - 0.15)
        return 0.0

    actions.append(ManagementAction(
        index=2,
        name="rebalance_overloaded_queue",
        description="Valid when rework or queue-delay suggests local overload.",
        is_valid=_rebalance_valid,
        apply=_rebalance_apply,
    ))

    # ── 3. merge_tasks_under_role ─────────────────────────────────────────
    def _merge_valid(kpi: np.ndarray, state: dict) -> bool:
        risk_high = (
            _kpi(kpi, 0) + _kpi(kpi, 1) > RISK_HIGH_THRESH
            or state.get("_risk_high", False)
        )
        return _kpi(kpi, 5) > VOLUME_HIGH and not risk_high

    def _merge_apply(state: dict, twin) -> float:
        state["_volume_pressure"] = max(-1.0, state.get("_volume_pressure", 0.0) - 0.3)
        state["_delay_norm"]      = max(0.0, state.get("_delay_norm", 0.0) - 0.1)
        state["_merged"]          = True
        return 0.0

    actions.append(ManagementAction(
        index=3,
        name="merge_tasks_under_role",
        description="Valid when volume pressure is high and risk is not high.",
        is_valid=_merge_valid,
        apply=_merge_apply,
    ))

    # ── 4. prioritize_urgent_case ─────────────────────────────────────────
    def _prioritize_valid(kpi: np.ndarray, state: dict) -> bool:
        return (
            _kpi(kpi, 3) > CASE_AGE_HIGH
            or state.get("_risk_high", False)
            or _kpi(kpi, 0) > DELAY_EXTREME
        )

    def _prioritize_apply(state: dict, twin) -> float:
        state["_delay_norm"] = max(0.0, state.get("_delay_norm", 0.0) - 0.4)
        return 0.0

    actions.append(ManagementAction(
        index=4,
        name="prioritize_urgent_case",
        description="Valid when case-age is high or risk branch indicates urgency.",
        is_valid=_prioritize_valid,
        apply=_prioritize_apply,
    ))

    # ── 5. defer_until_objections_resolved ───────────────────────────────
    def _defer_valid(kpi: np.ndarray, state: dict) -> bool:
        return (
            state.get("_objection_active", False)
            or state.get("_suspension_active", False)
        )

    def _defer_apply(state: dict, twin) -> float:
        state["_deferred"] = True
        return 0.0

    actions.append(ManagementAction(
        index=5,
        name="defer_until_objections_resolved",
        description="Valid when objection/appeal signals are present.",
        is_valid=_defer_valid,
        apply=_defer_apply,
    ))

    # ── 6. escalate_to_higher_authority ──────────────────────────────────
    def _escalate_valid(kpi: np.ndarray, state: dict) -> bool:
        return (
            _kpi(kpi, 0) > DELAY_EXTREME
            or state.get("_suspension_active", False)
            or _kpi(kpi, 1) > REWORK_HIGH * 2
        )

    def _escalate_apply(state: dict, twin) -> float:
        state["_rework_norm"] = max(0.0, state.get("_rework_norm", 0.0) * 0.5)
        state["_risk_high"]   = True
        state["_delay_norm"]  = max(0.0, state.get("_delay_norm", 0.0) - 0.5)
        return 0.0

    actions.append(ManagementAction(
        index=6,
        name="escalate_to_higher_authority",
        description="Valid under extreme delay, suspension/refusal, or persistent rework.",
        is_valid=_escalate_valid,
        apply=_escalate_apply,
    ))

    # ── 7. skip_optional_subprocess ──────────────────────────────────────
    def _skip_valid(kpi: np.ndarray, state: dict) -> bool:
        risk_high = (
            _kpi(kpi, 0) + _kpi(kpi, 1) > RISK_HIGH_THRESH
            or state.get("_risk_high", False)
        )
        return state.get("_skip_optional", False) and not risk_high

    def _skip_apply(state: dict, twin) -> float:
        state["_delay_norm"] = max(0.0, state.get("_delay_norm", 0.0) - 0.2)
        return 0.0

    actions.append(ManagementAction(
        index=7,
        name="skip_optional_subprocess",
        description="Valid only on optional-step signals for low-risk cases.",
        is_valid=_skip_valid,
        apply=_skip_apply,
    ))

    # ── 8. add_temporary_staff ────────────────────────────────────────────
    def _add_staff_valid(kpi: np.ndarray, state: dict) -> bool:
        return _kpi(kpi, 5) > VOLUME_HIGH and _kpi(kpi, 0) > DELAY_EXTREME

    def _add_staff_apply(state: dict, twin) -> float:
        state["_staffing_boost"] = min(2.0, state.get("_staffing_boost", 1.0) + 0.3)
        state["_delay_norm"]     = max(0.0, state.get("_delay_norm", 0.0) - 0.35)
        if twin is not None and hasattr(twin, "resource_pool"):
            for res in twin.resource_pool.capacities:
                twin.resource_pool.capacities[res] = max(
                    1, int(twin.resource_pool.capacities[res] * 1.3)
                )
        return 0.0

    actions.append(ManagementAction(
        index=8,
        name="add_temporary_staff",
        description="Valid under simultaneous high volume and extreme case delay.",
        is_valid=_add_staff_valid,
        apply=_add_staff_apply,
    ))

    # ── 9. adjust_staffing_by_case_volume ────────────────────────────────
    def _adjust_staff_valid(kpi: np.ndarray, state: dict) -> bool:
        return _kpi(kpi, 5) > VOLUME_HIGH

    def _adjust_staff_apply(state: dict, twin) -> float:
        state["_staffing_boost"]  = min(2.0, state.get("_staffing_boost", 1.0) + 0.15)
        state["_volume_pressure"] = max(-1.0, state.get("_volume_pressure", 0.0) - 0.15)
        return 0.0

    actions.append(ManagementAction(
        index=9,
        name="adjust_staffing_by_case_volume",
        description="Valid when municipality-level case volume is elevated.",
        is_valid=_adjust_staff_valid,
        apply=_adjust_staff_apply,
    ))

    # ── 10. enable_cross_trained_pool ─────────────────────────────────────
    def _cross_train_valid(kpi: np.ndarray, state: dict) -> bool:
        return _kpi(kpi, 5) > VOLUME_HIGH or _kpi(kpi, 1) > REWORK_HIGH

    def _cross_train_apply(state: dict, twin) -> float:
        state["_cross_train_active"] = True
        state["_rework_norm"]        = max(0.0, state.get("_rework_norm", 0.0) - 0.2)
        if twin is not None and hasattr(twin, "_cross_train_overrides"):
            for role in twin.role_activity_map:
                twin._cross_train_overrides[role] = set(twin.activities)
        return 0.0

    actions.append(ManagementAction(
        index=10,
        name="enable_cross_trained_pool",
        description="Valid when volume or rework pressure is elevated.",
        is_valid=_cross_train_valid,
        apply=_cross_train_apply,
    ))

    # ── 11. relax_rules_for_low_risk ──────────────────────────────────────
    def _relax_valid(kpi: np.ndarray, state: dict) -> bool:
        risk_high = (
            _kpi(kpi, 0) + _kpi(kpi, 1) > RISK_HIGH_THRESH
            or state.get("_risk_high", False)
        )
        objection = (
            state.get("_objection_active", False)
            or state.get("_suspension_active", False)
        )
        return not risk_high and not objection

    def _relax_apply(state: dict, twin) -> float:
        state["_skip_optional"] = True
        state["_delay_norm"]    = max(0.0, state.get("_delay_norm", 0.0) - 0.15)
        return 0.0

    actions.append(ManagementAction(
        index=11,
        name="relax_rules_for_low_risk",
        description="Valid only for low-risk, high-confidence, non-objection contexts.",
        is_valid=_relax_valid,
        apply=_relax_apply,
    ))

    # ── 12. trigger_high_cost_escalation ──────────────────────────────────
    def _high_cost_valid(kpi: np.ndarray, state: dict) -> bool:
        return (
            (_kpi(kpi, 0) + _kpi(kpi, 1) > RISK_HIGH_THRESH)
            and _kpi(kpi, 0) > DELAY_HIGH
        )

    def _high_cost_apply(state: dict, twin) -> float:
        state["_risk_high"]  = True
        state["_delay_norm"] = max(0.0, state.get("_delay_norm", 0.0) - 0.6)
        return 0.0

    actions.append(ManagementAction(
        index=12,
        name="trigger_high_cost_escalation",
        description="Valid on high-risk and high-delay proxy combinations.",
        is_valid=_high_cost_valid,
        apply=_high_cost_apply,
    ))

    # ── 13. reroute_from_overloaded_employee ──────────────────────────────
    def _reroute_valid(kpi: np.ndarray, state: dict) -> bool:
        return _kpi(kpi, 0) > DELAY_HIGH or _kpi(kpi, 1) > REWORK_HIGH

    def _reroute_apply(state: dict, twin) -> float:
        state["_rerouted"]    = True
        state["_rework_norm"] = max(0.0, state.get("_rework_norm", 0.0) - 0.3)
        state["_delay_norm"]  = max(0.0, state.get("_delay_norm", 0.0) - 0.2)
        return 0.0

    actions.append(ManagementAction(
        index=13,
        name="reroute_from_overloaded_employee",
        description="Valid when workload proxy is high (delay or rework).",
        is_valid=_reroute_valid,
        apply=_reroute_apply,
    ))

    # ── 14. close_case ────────────────────────────────────────────────────
    def _close_valid(kpi: np.ndarray, state: dict) -> bool:
        return state.get("_at_terminal", False)

    def _close_apply(state: dict, twin) -> float:
        return 0.0

    actions.append(ManagementAction(
        index=14,
        name="close_case",
        description="Valid only at terminal/completed states.",
        is_valid=_close_valid,
        apply=_close_apply,
    ))

    return actions


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

#: Ordered list of all 15 management actions (index 0 = default no-op).
MANAGEMENT_ACTIONS: list[ManagementAction] = _build_actions()

#: Number of management actions.
N_MANAGEMENT_ACTIONS: int = len(MANAGEMENT_ACTIONS)

#: Human-readable descriptions matching the original rule_descriptions dict.
RULE_DESCRIPTIONS: dict[str, str] = {a.name: a.description for a in MANAGEMENT_ACTIONS}

assert N_MANAGEMENT_ACTIONS == 15, (
    f"Expected 15 management actions, got {N_MANAGEMENT_ACTIONS}"
)


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------

def get_management_mask(
    kpi_signals: np.ndarray,
    episode_state: dict,
) -> np.ndarray:
    """
    Return a boolean mask of shape (N_MANAGEMENT_ACTIONS,) indicating which
    management actions are currently valid given the KPI signals and episode state.

    The default action (index 0) is always valid.

    Parameters
    ----------
    kpi_signals : np.ndarray, shape (7,)
        Current KPI signal vector from the environment observation.
    episode_state : dict
        Current episode state flags (see module docstring for keys).

    Returns
    -------
    np.ndarray[bool], shape (N_MANAGEMENT_ACTIONS,)
    """
    mask = np.zeros(N_MANAGEMENT_ACTIONS, dtype=bool)
    for action in MANAGEMENT_ACTIONS:
        mask[action.index] = action.is_valid(kpi_signals, episode_state)
    return mask


def apply_management_action(
    action_index: int,
    episode_state: dict,
    twin,
    kpi_signals: np.ndarray,
) -> float:
    """
    Apply a management action to the episode state and twin.

    Returns the shaped reward delta for the action.
    If the action is invalid given current KPI signals, applies a small
    penalty instead of the action effect (to discourage invalid use).

    Parameters
    ----------
    action_index : int
        Index of the management action to apply (0–14).
    episode_state : dict
        Mutable episode state dict (modified in-place).
    twin : DigitalTwin
        The digital twin instance for this episode.
    kpi_signals : np.ndarray, shape (7,)
        Current KPI signal vector.

    Returns
    -------
    float
        Shaped reward delta.
    """
    if action_index < 0 or action_index >= N_MANAGEMENT_ACTIONS:
        return -0.1  # out-of-range penalty

    action = MANAGEMENT_ACTIONS[action_index]

    if not action.is_valid(kpi_signals, episode_state):
        # Small penalty for using an action outside its valid context.
        # Kept small so it doesn't dominate the routing reward signal.
        return -0.1

    return action.apply(episode_state, twin)
