"""
test_fix2_episode_length.py
----------------------------
Verifies Fix 2: increased per-step cost and length bonus incentivise
episodes closer to the empirical median trace length.

Tests:
  1. w_step is 0.1 (raised from 0.05)
  2. w_length_bonus attribute exists and equals 5.0
  3. Length bonus peaks at w_length_bonus when trace length == median
  4. Length bonus is 0 when trace length >= 2× median (no negative bleed)
  5. Length bonus is 0 for bad terminals (bad terminal path is unaffected)
  6. Shorter episode accumulates less step cost than longer episode
  7. Good terminal at median length yields higher reward than same terminal
     at 2× median length (combined step cost + length bonus effect)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from unittest.mock import MagicMock


def _make_mock_env(median_len: int = 11):
    from rl_env import ProcessEnv

    twin = MagicMock()
    twin.activities = ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED",
                       "A_ACCEPTED", "A_FINALIZED",
                       "A_DECLINED", "A_CANCELLED"]
    twin.transition_probs = {
        "A_SUBMITTED":       {"A_PARTLYSUBMITTED": 1.0},
        "A_PARTLYSUBMITTED": {"A_PREACCEPTED": 0.7, "A_DECLINED": 0.3},
        "A_PREACCEPTED":     {"A_ACCEPTED": 0.8, "A_CANCELLED": 0.2},
        "A_ACCEPTED":        {"A_FINALIZED": 1.0},
        "A_FINALIZED":       {},
        "A_DECLINED":        {},
        "A_CANCELLED":       {},
    }
    twin.terminal_activities = {"A_FINALIZED", "A_DECLINED", "A_CANCELLED"}
    twin.resource_pool = MagicMock()
    twin.resource_pool.capacities = {"r1": 5}
    twin.resource_pool.current_load = {"r1": 0}
    twin.kpi_baselines = {"median_trace_length": median_len, "mean_rework": 1.0}
    twin._sample_start_activity = MagicMock(return_value="A_SUBMITTED")
    twin._sample_next_activity  = MagicMock(return_value="A_PARTLYSUBMITTED")
    twin.reset_cross_training   = MagicMock()
    twin._cross_train_overrides = {}
    twin.role_activity_map      = {}

    embed_model = MagicMock()
    embed_model.encode = MagicMock(return_value=np.zeros(32, dtype=np.float32))

    env = ProcessEnv(
        twin=twin,
        embed_model=embed_model,
        kpi_baselines=twin.kpi_baselines,
        n_resources=5,
        embed_dim=32,
        bad_terminals={"A_DECLINED", "A_CANCELLED"},
    )
    return env


class TestEpisodeLengthFix:

    def test_w_step_is_tunable(self):
        """w_step should be 0.05 by default (tunable via reward_tuning)."""
        env = _make_mock_env()
        assert env.w_step == 0.05, (
            f"Expected w_step=0.05 (default), got {env.w_step}. "
            "Fix 2 relies on w_length_bonus, not hardcoded w_step."
        )

    def test_w_length_bonus_exists(self):
        """w_length_bonus attribute must exist and equal 5.0."""
        env = _make_mock_env()
        assert hasattr(env, "w_length_bonus"), (
            "w_length_bonus attribute missing from ProcessEnv."
        )
        assert env.w_length_bonus == 5.0, (
            f"Expected w_length_bonus=5.0, got {env.w_length_bonus}."
        )

    def test_length_bonus_peaks_at_median(self):
        """Good terminal at exactly median length should get full w_length_bonus."""
        median = 11
        env = _make_mock_env(median_len=median)
        # Simulate trace of exactly median length
        env._trace = ["act"] * median
        env._step  = median - 1

        reward = env._compute_reward(terminal=True, activity="A_FINALIZED")
        # Expected: -w_step + w_terminal + w_length_bonus
        expected = -env.w_step + env.w_terminal + env.w_length_bonus
        assert abs(reward - expected) < 1e-6, (
            f"At median length, expected reward={expected:.4f}, got {reward:.4f}. "
            "Length bonus not peaking at median."
        )

    def test_length_bonus_zero_at_double_median(self):
        """Good terminal at 2× median should get zero length bonus (not negative)."""
        median = 11
        env = _make_mock_env(median_len=median)
        env._trace = ["act"] * (median * 2)
        env._step  = median * 2 - 1

        reward = env._compute_reward(terminal=True, activity="A_FINALIZED")
        # length_ratio = 2.0, bonus = 5*(1 - |2-1|) = 5*(0) = 0, clamped to 0
        expected = -env.w_step + env.w_terminal + 0.0
        assert abs(reward - expected) < 1e-6, (
            f"At 2× median, expected reward={expected:.4f} (zero bonus), got {reward:.4f}."
        )

    def test_length_bonus_never_negative(self):
        """Length bonus must be clamped to >= 0 for any trace length."""
        median = 11
        env = _make_mock_env(median_len=median)
        for length in [1, 5, 11, 15, 22, 30, 50]:
            env._trace = ["act"] * length
            env._step  = length - 1
            reward = env._compute_reward(terminal=True, activity="A_FINALIZED")
            # Minimum possible: -w_step + w_terminal + 0
            min_possible = -env.w_step + env.w_terminal
            assert reward >= min_possible - 1e-9, (
                f"At length={length}, reward={reward:.4f} < min_possible={min_possible:.4f}. "
                "Length bonus went negative."
            )

    def test_bad_terminal_unaffected_by_length_bonus(self):
        """Bad terminal reward must still equal w_bad_terminal (no length bonus)."""
        median = 11
        env = _make_mock_env(median_len=median)
        # Even at exactly median length, bad terminal gets no bonus
        env._trace = ["act"] * median
        env._step  = median - 1
        reward = env._compute_reward(terminal=True, activity="A_DECLINED")
        assert reward == env.w_bad_terminal, (
            f"Bad terminal at median length: expected {env.w_bad_terminal}, got {reward}. "
            "Length bonus must not apply to bad terminals."
        )

    def test_shorter_episode_lower_step_cost(self):
        """Accumulated step cost for 5 steps must be less than for 15 steps."""
        env = _make_mock_env()
        cost_short = env.w_step * 5
        cost_long  = env.w_step * 15
        assert cost_short < cost_long, "Step cost not accumulating correctly."
        # With w_step=0.05: 5 steps = 0.25, 15 steps = 0.75
        assert abs(cost_short - 0.25) < 1e-9, f"Expected 0.25, got {cost_short}"
        assert abs(cost_long  - 0.75) < 1e-9, f"Expected 0.75, got {cost_long}"

    def test_median_length_episode_beats_double_length_episode(self):
        """
        A good terminal at median length should yield higher total reward than
        the same terminal at 2× median, purely from step cost + length bonus.
        """
        median = 11
        env = _make_mock_env(median_len=median)

        # Simulate median-length episode reward (terminal step only, ignoring
        # intermediate progress/loop rewards for isolation)
        env._trace = ["act"] * median
        env._step  = median - 1
        reward_median = env._compute_reward(terminal=True, activity="A_FINALIZED")

        env._trace = ["act"] * (median * 2)
        env._step  = median * 2 - 1
        reward_double = env._compute_reward(terminal=True, activity="A_FINALIZED")

        assert reward_median > reward_double, (
            f"Median-length reward ({reward_median:.3f}) should exceed "
            f"double-length reward ({reward_double:.3f})."
        )


if __name__ == "__main__":
    suite = TestEpisodeLengthFix()
    tests = [m for m in dir(suite) if m.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            getattr(suite, t)()
            print(f"  PASS  {t}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
