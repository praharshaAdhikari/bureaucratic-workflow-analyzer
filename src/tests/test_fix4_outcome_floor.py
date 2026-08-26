"""
test_fix4_outcome_floor.py
--------------------------
Verifies the outcome floor: an episode may not reach an outcome sooner than
the fastest real case does.

Every edge in the fitted transition graph is real, but a first-order Markov
chain composes them into paths no case ever took. On BPIC2015 the shortest
simulated route to "permit irrevocable" was 2 steps against a real minimum of
11, and the trained agent took it in 100% of episodes — 0.94 reward variance,
2.3-step mean length. Per-edge masking cannot catch this because the path is
only implausible as a whole.

Tests:
  1. Below the floor, terminal successors are removed from the routing mask
  2. Below the floor, landing on a terminal does not end the episode
  3. At or above the floor, a terminal ends the episode as normal
  4. The floor is waived when every successor is a terminal (no dead ends)
  5. A floor of 1 reproduces the old unconstrained behaviour
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from unittest.mock import MagicMock


def _make_env(min_steps_to_outcome=1, dead_end=False):
    """Linear chain A -> B -> C -> GOOD, with A -> GOOD as a shortcut."""
    from rl_env import ProcessEnv

    twin = MagicMock()
    twin.activities = ["A", "B", "C", "GOOD", "BAD"]
    if dead_end:
        # Every successor of A is terminal — the floor must yield.
        twin.transition_probs = {
            "A":    {"GOOD": 0.5, "BAD": 0.5},
            "GOOD": {},
            "BAD":  {},
        }
    else:
        twin.transition_probs = {
            "A":    {"B": 0.5, "GOOD": 0.5},   # shortcut straight to the outcome
            "B":    {"C": 1.0},
            "C":    {"GOOD": 0.5, "BAD": 0.5},
            "GOOD": {},
            "BAD":  {},
        }
    twin.terminal_activities = {"GOOD", "BAD"}
    twin.resource_pool = MagicMock()
    twin.resource_pool.capacities = {"r1": 5}
    twin.resource_pool.current_load = {"r1": 0}
    twin.kpi_baselines = {"median_trace_length": 10, "mean_rework": 1.0}
    twin._sample_start_activity = MagicMock(return_value="A")
    twin._sample_next_activity = MagicMock(return_value="B")
    twin.reset_cross_training = MagicMock()
    twin._cross_train_overrides = {}
    twin.role_activity_map = {}

    embed_model = MagicMock()
    embed_model.encode = MagicMock(return_value=np.zeros(8, dtype=np.float32))

    return ProcessEnv(
        twin=twin,
        embed_model=embed_model,
        kpi_baselines=twin.kpi_baselines,
        n_resources=2,
        embed_dim=8,
        max_steps=50,
        bad_terminals={"BAD"},
        good_terminals={"GOOD"},
        min_steps_to_outcome=min_steps_to_outcome,
        # The floor is tested against agent-chosen terminals; in
        # environment mode terminals are withheld from routing entirely.
        verdict_mode="agent",
    )


def _routing_choices(env):
    """Successor activities the mask currently allows from the current state."""
    mask = env.action_masks()[: env._max_succ]
    successors = env._successors.get(env._current_activity, [])
    return {a for i, a in enumerate(successors) if mask[i]}


class TestOutcomeFloor:

    def test_terminals_masked_below_floor(self):
        env = _make_env(min_steps_to_outcome=3)
        env.reset()
        assert _routing_choices(env) == {"B"}, (
            "GOOD is reachable in one step from A and must be masked below the floor."
        )

    def test_terminals_offered_at_floor(self):
        env = _make_env(min_steps_to_outcome=1)
        env.reset()
        assert "GOOD" in _routing_choices(env), (
            "With a floor of 1 the shortcut must be available immediately."
        )

    def test_episode_does_not_end_below_floor(self):
        """Even if a terminal is somehow reached early, it must not terminate."""
        env = _make_env(min_steps_to_outcome=3)
        env.reset()
        successors = env._successors["A"]
        good_idx = successors.index("GOOD")
        _, _, terminated, truncated, _ = env.step(np.array([good_idx, 0]))
        assert not terminated, "Episode ended at step 1 despite a floor of 3."
        assert not truncated

    def test_episode_ends_at_or_above_floor(self):
        env = _make_env(min_steps_to_outcome=2)
        env.reset()
        env.step(np.array([env._successors["A"].index("B"), 0]))   # step 1 -> B
        successors = env._successors["B"]
        _, _, terminated, _, _ = env.step(np.array([successors.index("C"), 0]))  # step 2 -> C
        assert not terminated, "C is not a terminal."
        successors = env._successors["C"]
        _, _, terminated, _, _ = env.step(np.array([successors.index("GOOD"), 0]))
        assert terminated, "Step 3 is at/above the floor of 2 and must terminate."

    def test_dead_end_below_floor_truncates_without_an_outcome(self):
        """
        A state whose only successors are terminals, reached before the floor,
        is a dead end the model cannot continue from. Recording an outcome
        there would be exactly the implausibly short trajectory the floor
        exists to prevent, so the episode is abandoned instead.
        """
        env = _make_env(min_steps_to_outcome=5, dead_end=True)
        env.reset()
        allowed = _routing_choices(env)
        assert allowed, "Masking left no legal routing action."
        assert allowed <= {"GOOD", "BAD"}
        successors = env._successors["A"]
        idx = successors.index(sorted(allowed)[0])
        _, _, terminated, truncated, _ = env.step(np.array([idx, 0]))
        assert not terminated, (
            "A dead end below the floor must not record an outcome."
        )
        assert truncated, "The episode must end rather than continue forever."

    def test_dead_end_above_floor_still_concludes(self):
        """Above the floor the same dead end is a legitimate ending."""
        env = _make_env(min_steps_to_outcome=1, dead_end=True)
        env.reset()
        successors = env._successors["A"]
        idx = successors.index(sorted(_routing_choices(env))[0])
        _, _, terminated, _, _ = env.step(np.array([idx, 0]))
        assert terminated

    def test_floor_of_one_is_unconstrained(self):
        env = _make_env(min_steps_to_outcome=1)
        env.reset()
        successors = env._successors["A"]
        _, _, terminated, _, _ = env.step(np.array([successors.index("GOOD"), 0]))
        assert terminated, "A floor of 1 must reproduce the old behaviour."
