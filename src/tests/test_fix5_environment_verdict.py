"""
test_fix5_environment_verdict.py
--------------------------------
Verifies that the agent cannot choose whether a case succeeds.

"Route to A_APPROVED" used to be an ordinary action, always available and
always worth the outcome bonus, so the trained agent reached a good outcome
94.8% of the time on BPIC2012 against a real rate of 17.7%. None of the 15
management actions plausibly changes whether an applicant is creditworthy,
so a policy that picks the verdict is not modelling anything real.

Conditioning the verdict on the current activity is not enough either — the
agent can park at the activity with the friendliest terminal distribution
(75% good on BPIC2012, 100% on BPIC2015) and wait. So the verdict is drawn
once per episode, before the agent acts.

Tests:
  1. Terminals are never offered as a routing choice
  2. Over many episodes the good-outcome rate matches the configured base rate
  3. The rate is unchanged by a policy that always picks the same action
  4. The reward at a conclusion does not depend on the verdict
  5. Episodes still conclude, and never before the floor
  6. Omitting the base rates is an error, not a silent default
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock

BASE_RATES = {
    "p_good": 0.2,
    "good_terminal_weights": {"GOOD": 1.0},
    "bad_terminal_weights": {"BAD": 1.0},
}


def _make_env(min_steps_to_outcome=1, p_good=0.2, **kwargs):
    """A -> B -> A loop, with GOOD/BAD reachable from B."""
    from rl_env import ProcessEnv

    twin = MagicMock()
    twin.activities = ["A", "B", "GOOD", "BAD"]
    twin.transition_probs = {
        "A":    {"B": 1.0},
        "B":    {"A": 0.6, "GOOD": 0.2, "BAD": 0.2},
        "GOOD": {},
        "BAD":  {},
    }
    twin.terminal_activities = {"GOOD", "BAD"}
    twin.resource_pool = MagicMock()
    twin.resource_pool.capacities = {"r1": 5}
    twin.resource_pool.current_load = {"r1": 0}
    twin.kpi_baselines = {"median_trace_length": 8, "mean_rework": 1.0}
    twin._sample_start_activity = MagicMock(return_value="A")
    twin._sample_next_activity = MagicMock(return_value="B")
    twin.reset_cross_training = MagicMock()
    twin._cross_train_overrides = {}
    twin.role_activity_map = {}

    embed_model = MagicMock()
    embed_model.encode = MagicMock(return_value=np.zeros(8, dtype=np.float32))

    rates = dict(BASE_RATES, p_good=p_good)
    return ProcessEnv(
        twin=twin,
        embed_model=embed_model,
        kpi_baselines=twin.kpi_baselines,
        n_resources=2,
        embed_dim=8,
        max_steps=200,
        seed=7,
        bad_terminals={"BAD"},
        good_terminals={"GOOD"},
        min_steps_to_outcome=min_steps_to_outcome,
        verdict_mode="environment",
        outcome_base_rates=rates,
        **kwargs,
    )


def _run(env, n=600, pick=0):
    """Roll out `n` episodes always choosing routing action `pick`."""
    good = concluded = 0
    lengths = []
    for _ in range(n):
        env.reset()
        done = trunc = False
        steps = 0
        while not (done or trunc):
            _, _, done, trunc, _ = env.step(np.array([pick, 0]))
            steps += 1
        if done:
            concluded += 1
            good += env._current_activity in env._good_terminals
        lengths.append(steps)
    return good, concluded, lengths


class TestEnvironmentVerdict:

    def test_terminals_never_offered_for_routing(self):
        env = _make_env()
        env.reset()
        for _ in range(40):
            successors = env._successors.get(env._current_activity, [])
            mask = env.action_masks()[: env._max_succ]
            offered = {a for i, a in enumerate(successors) if mask[i]}
            assert not (offered & env._all_terminals), (
                f"Routing offered a terminal: {offered & env._all_terminals}"
            )
            _, _, done, trunc, _ = env.step(np.array([0, 0]))
            if done or trunc:
                env.reset()

    def test_good_rate_matches_base_rate(self):
        env = _make_env(p_good=0.2)
        good, concluded, _ = _run(env, n=800)
        assert concluded > 700, f"Only {concluded} episodes concluded."
        rate = good / concluded
        assert abs(rate - 0.2) < 0.05, (
            f"Good-outcome rate {rate:.3f} should track the 0.2 base rate."
        )

    def test_rate_is_the_same_for_a_different_policy(self):
        """The verdict must not respond to how the agent routes."""
        rate = []
        for pick in (0, 1):
            env = _make_env(p_good=0.35)
            good, concluded, _ = _run(env, n=600, pick=pick)
            rate.append(good / concluded)
        assert abs(rate[0] - rate[1]) < 0.06, (
            f"Good rate moved from {rate[0]:.3f} to {rate[1]:.3f} when the "
            f"routing policy changed — the verdict is still steerable."
        )

    def test_reward_at_conclusion_ignores_the_verdict(self):
        """
        The agent cannot influence the verdict, so scoring it on the outcome
        would only add noise. Concluding is what earns the bonus.
        """
        env = _make_env()
        env.reset()
        env._trace = ["A"] * 8
        env._step = 7
        good_reward = env._compute_reward(terminal=True, activity="GOOD")
        bad_reward = env._compute_reward(terminal=True, activity="BAD")
        assert abs(good_reward - bad_reward) < 1e-9, (
            f"Conclusion reward differs by verdict ({good_reward} vs "
            f"{bad_reward}); the agent would be graded on something it "
            f"cannot control."
        )
        assert good_reward > 0

    def test_never_concludes_before_the_floor(self):
        env = _make_env(min_steps_to_outcome=6)
        _, concluded, lengths = _run(env, n=400)
        assert concluded > 350
        assert min(lengths) >= 6, (
            f"An episode concluded after {min(lengths)} steps, below the floor of 6."
        )

    def test_missing_base_rates_is_an_error(self):
        from rl_env import ProcessEnv

        with pytest.raises(ValueError, match="outcome_base_rates"):
            _make_env_without_rates(ProcessEnv)


def _make_env_without_rates(ProcessEnv):
    twin = MagicMock()
    twin.activities = ["A", "GOOD"]
    twin.transition_probs = {"A": {"GOOD": 1.0}, "GOOD": {}}
    twin.terminal_activities = {"GOOD"}
    twin.resource_pool = MagicMock()
    twin.resource_pool.capacities = {"r1": 1}
    twin.resource_pool.current_load = {"r1": 0}
    twin.kpi_baselines = {"median_trace_length": 5, "mean_rework": 1.0}
    twin._sample_start_activity = MagicMock(return_value="A")
    twin._sample_next_activity = MagicMock(return_value="GOOD")
    twin.reset_cross_training = MagicMock()
    twin._cross_train_overrides = {}
    twin.role_activity_map = {}
    embed_model = MagicMock()
    embed_model.encode = MagicMock(return_value=np.zeros(8, dtype=np.float32))
    return ProcessEnv(
        twin=twin,
        embed_model=embed_model,
        kpi_baselines=twin.kpi_baselines,
        n_resources=1,
        embed_dim=8,
        bad_terminals=set(),
        good_terminals={"GOOD"},
        verdict_mode="environment",
        outcome_base_rates=None,
    )
