"""
test_fix1_bad_terminal_reward.py
---------------------------------
Verifies Fix 1: bad terminal reward is unambiguously negative regardless of
how w_terminal is tuned.

Tests:
  1. Default weights: bad terminal reward < 0
  2. After reward_tuning sets w_terminal=15 (worst-case): bad terminal still < 0
  3. Good terminal reward > bad terminal reward (ordering preserved)
  4. w_bad_terminal is not overridden by reward_tuning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_mock_env():
    """Build a minimal ProcessEnv-like object to test _compute_reward in isolation."""
    # We import ProcessEnv but mock the heavy dependencies (twin, embedder)
    from rl_env import ProcessEnv

    # --- Mock twin ---
    twin = MagicMock()
    twin.activities = ["A_SUBMITTED", "A_PARTLYSUBMITTED", "A_PREACCEPTED",
                       "A_ACCEPTED", "A_FINALIZED",
                       "A_DECLINED", "A_CANCELLED", "O_CANCELLED"]
    twin.transition_probs = {
        "A_SUBMITTED":       {"A_PARTLYSUBMITTED": 1.0},
        "A_PARTLYSUBMITTED": {"A_PREACCEPTED": 0.7, "A_DECLINED": 0.3},
        "A_PREACCEPTED":     {"A_ACCEPTED": 0.8, "A_CANCELLED": 0.2},
        "A_ACCEPTED":        {"A_FINALIZED": 1.0},
        "A_FINALIZED":       {},
        "A_DECLINED":        {},
        "A_CANCELLED":       {},
        "O_CANCELLED":       {},
    }
    twin.terminal_activities = {"A_FINALIZED", "A_DECLINED", "A_CANCELLED", "O_CANCELLED"}
    twin.resource_pool = MagicMock()
    twin.resource_pool.capacities = {"r1": 5}
    twin.resource_pool.current_load = {"r1": 0}
    twin.kpi_baselines = {"median_trace_length": 11, "mean_rework": 1.0}
    twin._sample_start_activity = MagicMock(return_value="A_SUBMITTED")
    twin._sample_next_activity  = MagicMock(return_value="A_PARTLYSUBMITTED")
    twin.reset_cross_training   = MagicMock()
    twin._cross_train_overrides = {}
    twin.role_activity_map      = {}

    # --- Mock embedder ---
    embed_model = MagicMock()
    embed_model.encode = MagicMock(return_value=np.zeros(32, dtype=np.float32))

    bad_terminals = {"A_DECLINED", "A_CANCELLED", "O_CANCELLED"}

    env = ProcessEnv(
        twin=twin,
        embed_model=embed_model,
        kpi_baselines=twin.kpi_baselines,
        n_resources=5,
        embed_dim=32,
        bad_terminals=bad_terminals,
        # These tests cover the reward given when the *agent* routes to a
        # terminal. That is now the 'agent' ablation; the pipeline default
        # has the environment draw the verdict instead.
        verdict_mode="agent",
    )
    # Minimal episode state so _compute_reward can run
    env._trace = ["A_SUBMITTED", "A_PARTLYSUBMITTED"]
    env._step  = 1
    return env


class TestBadTerminalRewardIsNegative:

    def test_bad_terminal_default_weights_is_negative(self):
        """With default weights, bad terminal reward must be < 0."""
        env = _make_mock_env()
        reward = env._compute_reward(terminal=True, activity="A_DECLINED")
        assert reward < 0, (
            f"Bad terminal reward should be negative, got {reward}. "
            "Fix 1 not applied correctly."
        )

    def test_bad_terminal_with_high_w_terminal_still_negative(self):
        """Even if reward_tuning sets w_terminal=30, bad terminal must stay negative."""
        env = _make_mock_env()
        env.w_terminal = 30.0   # simulate worst-case reward_tuning output
        reward = env._compute_reward(terminal=True, activity="A_DECLINED")
        assert reward < 0, (
            f"Bad terminal reward should be negative even with w_terminal=30, got {reward}. "
            "w_bad_terminal is being overridden or the branch logic is wrong."
        )

    def test_bad_terminal_reward_equals_w_bad_terminal(self):
        """Bad terminal reward should equal exactly w_bad_terminal (no step cost added)."""
        env = _make_mock_env()
        reward = env._compute_reward(terminal=True, activity="A_CANCELLED")
        assert reward == env.w_bad_terminal, (
            f"Expected {env.w_bad_terminal}, got {reward}. "
            "Bad terminal should return w_bad_terminal directly."
        )

    def test_good_terminal_reward_greater_than_bad_terminal(self):
        """Good terminal reward must be strictly greater than bad terminal reward."""
        env = _make_mock_env()
        good_reward = env._compute_reward(terminal=True, activity="A_FINALIZED")
        bad_reward  = env._compute_reward(terminal=True, activity="A_DECLINED")
        assert good_reward > bad_reward, (
            f"Good terminal ({good_reward}) should exceed bad terminal ({bad_reward})."
        )

    def test_w_bad_terminal_not_affected_by_reward_tuning(self):
        """reward_tuning.tune_reward_weights must not change w_bad_terminal."""
        env = _make_mock_env()
        original_w_bad = env.w_bad_terminal

        # Simulate what reward_tuning does when applying best weights
        fake_best_weights = {
            "w_completion": 15.0,
            "w_delay":       0.5,
            "w_rework":      1.0,
            "w_risk":        0.5,
            "w_throughput":  0.05,
        }
        # Apply the same logic as reward_tuning.py
        env.w_terminal = fake_best_weights.get("w_completion", env.w_terminal)
        env.w_loop     = fake_best_weights.get("w_rework",     env.w_loop)
        env.w_progress = fake_best_weights.get("w_delay",      env.w_progress)
        env.w_step     = fake_best_weights.get("w_throughput",  env.w_step)
        # w_bad_terminal intentionally NOT in fake_best_weights

        assert env.w_bad_terminal == original_w_bad, (
            f"w_bad_terminal changed from {original_w_bad} to {env.w_bad_terminal}. "
            "reward_tuning must not override w_bad_terminal."
        )
        # And the reward is still negative after tuning
        reward = env._compute_reward(terminal=True, activity="A_DECLINED")
        assert reward < 0, (
            f"After reward_tuning, bad terminal reward is {reward} (should be < 0)."
        )

    def test_all_bad_terminals_yield_negative_reward(self):
        """Every activity in _bad_terminals must produce a negative terminal reward."""
        env = _make_mock_env()
        env.w_terminal = 30.0  # worst-case tuning
        for act in env._bad_terminals:
            reward = env._compute_reward(terminal=True, activity=act)
            assert reward < 0, (
                f"Activity '{act}' in _bad_terminals gave reward={reward} (should be < 0)."
            )


if __name__ == "__main__":
    # Quick smoke-test without pytest
    suite = TestBadTerminalRewardIsNegative()
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
