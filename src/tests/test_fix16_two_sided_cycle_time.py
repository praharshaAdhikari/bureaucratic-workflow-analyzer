"""
test_fix16_two_sided_cycle_time.py
----------------------------------
Verifies that the cycle-time charge punishes implausible *speed* as well as
implausible slowness.

The term was introduced (Fix 15) so the managerial interventions would have
something to bite on: they change how long a case takes, and duration was not
in the reward at all. It was floored at zero below the real median, on the
reasoning that paying the agent to finish faster than any real case would
reward implausibility.

Flooring does not pay the agent to go faster. It makes going faster *free* —
and the per-step cost then breaks the tie in favour of speed. Measured after
Fix 15, the simulated median cycle time was 16x, 5x and 4x faster than the
real logs on BPIC2012 / BPIC2015 / BPIC2017, i.e. further from reality than
before the term existed.

Tests:
  1. A case finishing at the real median is charged nothing
  2. Symmetry: half the median and twice the median cost the same
  3. Being 16x too fast is charged, and charged a lot (the BPIC2012 case)
  4. The charge is monotone in |log2(ratio)| on both sides
  5. ``slow_only`` reproduces the old one-sided shape exactly
  6. A degenerate near-zero elapsed time cannot blow up the return
  7. Without a real median there is no charge at all
  8. The charge reaches the episode reward with the right sign and size
  9. The mode is validated, and round-trips through the saved config
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock

from reward_config import RewardConfig

DAY = 86_400.0

#: BPIC2012's real median case duration.
REAL_MEDIAN_S = 0.8085351157407407 * DAY

BASE_RATES = {
    "p_good": 0.2,
    "good_terminal_weights": {"GOOD": 1.0},
    "bad_terminal_weights": {"BAD": 1.0},
}


def _make_env(real_median_cycle_s=REAL_MEDIAN_S, reward_config=None):
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
        reward_config=reward_config,
        min_steps_to_outcome=1,
        verdict_mode="environment",
        outcome_base_rates=BASE_RATES,
        real_median_cycle_s=real_median_cycle_s,
    )


def _deviation_at(env, ratio):
    """The charged deviation when the episode's elapsed time is `ratio` x real."""
    env._elapsed_s = ratio * REAL_MEDIAN_S
    return env._cycle_time_deviation()


class TestTwoSidedCycleTime:

    def test_real_median_is_free(self):
        env = _make_env()
        assert _deviation_at(env, 1.0) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_about_the_median(self):
        env = _make_env()
        for factor in (2.0, 4.0, 16.0):
            slow = _deviation_at(env, factor)
            fast = _deviation_at(env, 1.0 / factor)
            assert slow == pytest.approx(fast), (
                f"{factor}x too slow costs {slow:.3f} but {factor}x too fast "
                f"costs {fast:.3f} — the charge is not two-sided"
            )
            assert slow == pytest.approx(np.log2(factor))

    def test_racing_is_charged(self):
        """The BPIC2012 case: 0.05 days simulated against 0.81 days real."""
        env = _make_env()
        env._elapsed_s = 0.05 * DAY
        deviation = env._cycle_time_deviation()

        assert deviation > 3.5, (
            f"16x too fast is charged only {deviation:.2f} log2 units"
        )
        # w_time is 0.25 x 30 = 7.5, so this is worth about a conclusion bonus.
        charge = env.w_time * deviation
        assert charge > 0.5 * env.w_terminal, (
            f"charge {charge:.1f} is small next to the conclusion bonus "
            f"{env.w_terminal} — it will not change the policy"
        )

    def test_monotone_on_both_sides(self):
        env = _make_env()
        slow = [_deviation_at(env, r) for r in (1.0, 1.5, 2.0, 4.0, 8.0)]
        fast = [_deviation_at(env, 1.0 / r) for r in (1.0, 1.5, 2.0, 4.0, 8.0)]
        assert slow == sorted(slow) and len(set(slow)) == len(slow)
        assert fast == sorted(fast) and len(set(fast)) == len(fast)

    def test_slow_only_reproduces_the_old_shape(self):
        env = _make_env(reward_config=RewardConfig(time_penalty_mode="slow_only"))
        assert _deviation_at(env, 0.0625) == 0.0
        assert _deviation_at(env, 0.5) == 0.0
        assert _deviation_at(env, 1.0) == 0.0
        assert _deviation_at(env, 4.0) == pytest.approx(2.0)

    def test_degenerate_elapsed_time_is_bounded(self):
        env = _make_env()
        for elapsed in (0.0, 1e-9, 1.0):
            env._elapsed_s = elapsed
            assert env._cycle_time_deviation() == env.time_penalty_cap
        # And the cap sits well outside the range any real policy reaches.
        assert env.time_penalty_cap > _deviation_at(env, 0.0625) + 1.0

    def test_no_real_median_means_no_charge(self):
        env = _make_env(real_median_cycle_s=None)
        for ratio in (0.01, 1.0, 100.0):
            assert _deviation_at(env, ratio) == 0.0

    def test_charge_reaches_the_episode_reward(self):
        """Two envs differing only in elapsed time must differ by w_time x dev."""
        fast, slow = _make_env(), _make_env()
        for env in (fast, slow):
            env.reset()
            env._intervention_charge = 0.0
            env._trace = ["A"] * 8          # neutralise the length bonus difference
        fast._elapsed_s = 0.25 * REAL_MEDIAN_S
        slow._elapsed_s = 1.00 * REAL_MEDIAN_S

        r_fast = fast._compute_reward(terminal=True, activity="GOOD")
        r_slow = slow._compute_reward(terminal=True, activity="GOOD")

        assert r_fast < r_slow, (
            "finishing at a quarter of the real median scored no worse than "
            "finishing on it — the two-sided charge is not wired into the reward"
        )
        assert r_slow - r_fast == pytest.approx(fast.w_time * 2.0, rel=1e-6)


class TestConfig:

    def test_default_is_two_sided(self):
        assert RewardConfig().time_penalty_mode == "two_sided"

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="time_penalty_mode"):
            RewardConfig(time_penalty_mode="fast_only")

    def test_non_positive_cap_is_rejected(self):
        with pytest.raises(ValueError, match="time_penalty_cap"):
            RewardConfig(time_penalty_cap=0.0)

    def test_round_trips_through_disk(self, tmp_path):
        cfg = RewardConfig(time_penalty_mode="slow_only", time_penalty_cap=4.0)
        cfg.save(tmp_path)
        assert RewardConfig.load(tmp_path) == cfg

    def test_a_stale_config_without_the_key_still_loads(self):
        """Configs written before this fix must not become unreadable."""
        old = RewardConfig().to_dict()
        del old["time_penalty_mode"]
        del old["time_penalty_cap"]
        assert RewardConfig.from_dict(old).time_penalty_mode == "two_sided"
