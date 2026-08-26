"""
intervention_effects.py
-----------------------
What each managerial intervention actually does to a case.

Why this module exists
----------------------
The 15-action catalogue was decorative. Every action wrote values into an
episode-state dictionary that nothing downstream read: the reward recomputes
delay and rework from the trace, and routing, conclusion timing and the verdict
never consulted the dictionary at all. Measured by holding routing fixed and
swapping the entire management policy, total reward moved by 0.02% and episode
length and outcome rate did not move at all. The two trained policies disagreed
completely about which action to use — 0.05% no-op on BPIC2015 against 91.9% on
BPIC2017 — which is what arbitrary drift looks like.

This module gives each intervention a causal effect on the one process quantity
the environment models honestly: **how long the case takes**. It also gives each
one a cost, so that intervening is a trade-off rather than free.

Where the numbers come from — read this before quoting any result
-----------------------------------------------------------------
**The effect sizes below are assumptions, not measurements.** Two things were
checked against the logs first, and both came back negative:

1. *Does congestion slow work down?* If busier resources were slower, staffing
   and rebalancing interventions would have a measurable basis. Spearman
   correlation between a resource's daily workload and the duration of its
   steps, computed within each activity:

       BPIC2012  median rho = -0.014   (5 of 23 activities significant)
       BPIC2017  median rho = -0.053   (22 of 24, but 13 of them negative)
       BPIC2015  median rho = -0.082   (53 of 90, 47 of them negative)

   The correlation is mostly *negative* — busier resources are faster. Step
   duration in these logs is dominated by waiting on applicants, objections and
   external parties, not by staff availability. There is no congestion effect
   to calibrate against.

2. *Are the interventions themselves recorded?* Only on BPIC2015, and only 4 of
   the 15 concepts (defer, escalate, skip, close). BPIC2012 and BPIC2017 record
   none at all. So the effect of an intervention cannot be estimated from what
   happened after it, because it was never logged as happening.

Consequently every multiplier here is a stated assumption with a rationale, and
``EFFECT_SCALE`` exists so results can be reported across a sensitivity sweep
rather than at one arbitrary point. A conclusion that survives only at
``scale = 1.0`` is not a conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionEffect:
    """
    What one intervention does, and what it costs.

    duration_multiplier
        Applied to the duration of subsequent steps. Below 1.0 speeds the case
        up, above 1.0 slows it down. Multiplicative and bounded, so repeated
        use has diminishing returns rather than driving duration to zero.
    cost
        Charged once, in reward units, each time the action is taken. Scaled by
        ``w_terminal`` at the point of use so it stays proportionate to the
        outcome bonus.
    compliance_risk
        Charged additionally when the intervention waives or skips work. Kept
        separate from ``cost`` so the compliance question can be reported on
        its own, and switched off, without disturbing the cost model.
    rationale
        Why this number and not another. Every entry is an assumption; this
        field is what makes it reviewable.
    """

    duration_multiplier: float = 1.0
    cost: float = 0.0
    compliance_risk: float = 0.0
    rationale: str = ""


#: Global multiplier on every effect, for sensitivity analysis.
#: 0.0 disables the mechanism entirely and reproduces the pre-fix behaviour,
#: which makes the management ablation a one-line change.
EFFECT_SCALE = 1.0

#: Floor on the cumulative duration multiplier within an episode. Without it,
#: an agent that intervenes every step drives case duration to zero.
MIN_CUMULATIVE_MULTIPLIER = 0.40
MAX_CUMULATIVE_MULTIPLIER = 2.50


EFFECTS: dict[str, InterventionEffect] = {
    "assign_to_primary_team": InterventionEffect(
        1.00, 0.00, 0.0,
        "The no-op default. Handling the case normally costs nothing extra and "
        "changes nothing.",
    ),
    "prioritize_urgent_case": InterventionEffect(
        0.85, 0.03, 0.0,
        "Moving a case up the queue is the cheapest lever a manager has and the "
        "most direct: the work is the same, it simply waits less. Modest cost "
        "because it delays other cases rather than consuming new resource.",
    ),
    "rebalance_overloaded_queue": InterventionEffect(
        0.92, 0.04, 0.0,
        "Redistributing work across an existing team. Smaller effect than "
        "prioritisation because the total capacity is unchanged.",
    ),
    "reroute_from_overloaded_employee": InterventionEffect(
        0.93, 0.03, 0.0,
        "A narrower version of rebalancing, applied to one handler.",
    ),
    "merge_tasks_under_role": InterventionEffect(
        0.90, 0.05, 0.0,
        "Combining steps under one role removes hand-offs, which is where "
        "administrative processes lose most time. Costs coordination effort.",
    ),
    "enable_cross_trained_pool": InterventionEffect(
        0.93, 0.08, 0.0,
        "Widening who may perform a step reduces waiting for a specific "
        "qualified person, but training and quality overhead make it dearer "
        "than simple rebalancing.",
    ),
    "adjust_staffing_by_case_volume": InterventionEffect(
        0.94, 0.06, 0.0,
        "Routine capacity adjustment. Small, cheap, sustainable.",
    ),
    "add_temporary_staff": InterventionEffect(
        0.88, 0.18, 0.0,
        "Real additional capacity, and correspondingly expensive. Temporary "
        "staff also need supervision, so the effect is smaller than headcount "
        "alone would suggest.",
    ),
    "outsource_to_volunteer_pool": InterventionEffect(
        0.90, 0.12, 0.0,
        "External capacity at lower unit cost than temporary staff, with more "
        "variable throughput.",
    ),
    "escalate_to_higher_authority": InterventionEffect(
        0.82, 0.20, 0.0,
        "Escalation forces a decision and is strongly effective, but consumes "
        "senior attention, which is the scarcest resource in the process.",
    ),
    "trigger_high_cost_escalation": InterventionEffect(
        0.72, 0.35, 0.0,
        "The strongest lever and the most expensive by design. If it is ever "
        "worth using, it should be rare.",
    ),
    "defer_until_objections_resolved": InterventionEffect(
        1.45, 0.02, 0.0,
        "The one intervention that deliberately makes a case slower. Waiting "
        "for an objection to resolve is correct handling, not a failure, and "
        "the environment should not reward pretending otherwise.",
    ),
    "skip_optional_subprocess": InterventionEffect(
        0.80, 0.02, 0.25,
        "Skipping work is fast and cheap, which is exactly why it needs a "
        "compliance charge. Without one the agent learns that the best way to "
        "run a permit process is not to do it.",
    ),
    "relax_rules_for_low_risk": InterventionEffect(
        0.83, 0.02, 0.30,
        "As above, and worse: waiving a rule in a public-administration process "
        "is a governance decision, not an efficiency one. The heaviest "
        "compliance charge of the catalogue.",
    ),
    "close_case": InterventionEffect(
        1.00, 0.00, 0.0,
        "Currently unreachable — the mask requires a terminal state, which ends "
        "the episode before the action can be selected. Kept at neutral so it "
        "cannot distort anything if that changes.",
    ),
}


def effect_for(action_name: str) -> InterventionEffect:
    """Effect for an action, defaulting to neutral for anything unlisted."""
    return EFFECTS.get(action_name, InterventionEffect())


def scaled_duration_multiplier(action_name: str, scale: float = EFFECT_SCALE) -> float:
    """
    Duration multiplier with the sensitivity scale applied.

    ``scale = 0`` returns 1.0 for everything, which turns the mechanism off.
    ``scale = 1`` returns the declared value. Intermediate and larger values
    interpolate and extrapolate around 1.0, so a sweep moves every effect
    together and by a comparable proportion.
    """
    m = effect_for(action_name).duration_multiplier
    return 1.0 + (m - 1.0) * scale


def scaled_cost(action_name: str, scale: float = EFFECT_SCALE) -> float:
    """Intervention cost with the sensitivity scale applied."""
    return effect_for(action_name).cost * scale


def scaled_compliance_risk(action_name: str, scale: float = EFFECT_SCALE) -> float:
    """Compliance charge with the sensitivity scale applied."""
    return effect_for(action_name).compliance_risk * scale


def summary_table():
    """The catalogue as a DataFrame, for the write-up and for review."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "action": name,
                "duration_multiplier": e.duration_multiplier,
                "cost": e.cost,
                "compliance_risk": e.compliance_risk,
                "rationale": e.rationale,
            }
            for name, e in EFFECTS.items()
        ]
    ).sort_values("duration_multiplier")
