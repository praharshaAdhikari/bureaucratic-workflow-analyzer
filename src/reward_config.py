"""
reward_config.py
----------------
The single definition of the reward function's weights.

Why this module exists
----------------------
The reward used to be configured in two places that disagreed.

``notebooks/04_rl_training.ipynb`` loaded ``reward_weights.json`` and applied
it to the environment::

    env.w_terminal = best_weights['w_completion']   # 30.0
    env.w_loop     = best_weights['w_rework']       # 2.0
    env.w_progress = best_weights['w_delay']        # 0.3
    env.w_step     = best_weights['w_throughput']   # 0.05

``notebooks/05_evaluation.ipynb`` and ``06_insights.ipynb`` built the same
environment and never loaded that file, so evaluation ran with the defaults —
in particular ``w_terminal = 0.0``.  The agent was trained where reaching a
good outcome was worth +30 and graded where it was worth nothing.  That is the
entire gap between the training curves (plateau ~+37 on BPIC2012) and the
reported evaluation means (+6.59).  Nothing else was needed to explain it.

Two further problems came with it:

* the key names did not correspond to what they set (``w_delay`` set the
  *progress* bonus, ``w_throughput`` set the *per-step cost*), so the mapping
  could not be checked by reading it;
* the tuned weights differed per dataset — the per-step cost was 0.05, 0.2 and
  0.02 on BPIC2012 / BPIC2015 / BPIC2017 — while the paper claims in three
  places that identical reward weights were used everywhere.

Decisions taken
---------------
1. One object, ``RewardConfig``, carries every weight, with the same names the
   environment uses. Training, evaluation and analysis all read it.
2. ``DEFAULT`` is shared across all datasets, so the paper's "no per-dataset
   tuning" claim is true as written.
3. The good-outcome bonus is +30 and the bad-outcome penalty is -30 — the
   magnitude training already used, now symmetric and applied on both sides.
4. Per-dataset tuned weights remain loadable (``RewardConfig.from_legacy_file``)
   so the earlier setup can be reproduced as an ablation, but nothing uses
   them by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


#: Filename written next to every result so a run's reward is self-documenting.
CONFIG_FILENAME = "reward_config.json"

#: Shapes the cycle-time charge can take. See ``RewardConfig.time_penalty_mode``.
TIME_PENALTY_MODES = frozenset({"two_sided", "slow_only"})


@dataclass(frozen=True)
class RewardConfig:
    """
    Weights for :meth:`rl_env.ProcessEnv._compute_reward`.

    Per step
        ``-w_step``                       always charged
        ``+w_progress``                   if the activity is new to this trace
        ``-w_loop * excess_loop_rate``    repetition above the log's own rate

    On reaching a good outcome
        ``-w_step + w_terminal + length_bonus - time_charge - intervention_charge``
        where ``length_bonus`` peaks at ``w_length_bonus`` when the episode
        length equals the median trace length and decays to 0 at twice that,
        and ``time_charge`` is ``w_time`` times the log2 deviation of the
        elapsed time from the real median (see ``time_penalty_mode``).

    On reaching a bad outcome
        ``w_bad_terminal`` alone — deliberately not combined with any bonus, so
        no tuning of the other weights can make a bad outcome profitable.

    Why the per-step weights are shares, not constants
    --------------------------------------------------
    ``w_step`` and ``w_progress`` are charged once per step, so their total
    over an episode grows with the length of the process while the outcome
    term does not. Fixed constants therefore mean different things on
    different logs. Measured with the old constants (0.05 and 0.3) under a
    random policy, the share of total reward coming from shaping rather than
    from the outcome was:

        BPIC2012 (median trace 11)    3.7%
        BPIC2017 (median trace 35)    4.7%
        BPIC2015 (median trace 45)   84.3%

    On BPIC2015 — 356 activities, ~48-step episodes, so nearly every step
    earns the "new activity" bonus — the agent was mostly being paid to
    wander. Those constants were implicitly fitted to BPIC2012's scale.

    So the two per-step weights are expressed as a fraction of the outcome
    magnitude spent over one median-length episode, and divided by that
    dataset's median trace length (see :meth:`per_step_weights`). The shares
    are identical for every dataset, which makes the *reward structure*
    uniform rather than merely the numbers. The resulting values on BPIC2012
    land almost exactly on the old 0.3 / 0.05, so this rescales the other two
    logs rather than changing the one the constants suited.
    """

    # Outcome terms — absolute, and the scale everything else is relative to.
    w_terminal:      float = 30.0
    w_bad_terminal:  float = -30.0
    w_length_bonus:  float = 5.0

    # Per-step shaping, as a share of w_terminal spent over a median episode.
    #
    # progress_share is 0. The progress bonus paid only for reaching an
    # activity not yet in the trace, which made repeating an activity strictly
    # worse than any alternative, whatever the process actually does. Measured
    # against the logs, the agent's repetition rate came out uncalibrated in
    # both directions:
    #
    #     dataset     agent self-loops   real log   real steps to a *new* activity
    #     BPIC2012            0.0%         42.5%              37.7%
    #     BPIC2015            0.0%          0.8%              92.4%
    #     BPIC2017           68.6%         38.3%              39.1%
    #
    # BPIC2012's real process repeats 62% of the time and the agent never did;
    # BPIC2015 matched only because that process genuinely does not repeat.
    # (BPIC2017 overshoots for a different reason — it camps on the activity
    # with the highest chance of concluding the case — which the progress
    # bonus was not preventing anyway.)
    #
    # The per-step cost and the length bonus already push toward finishing
    # efficiently, and neither of them cares whether a step is novel. Set this
    # back to 0.10 to reproduce the old behaviour as an ablation.
    progress_share:  float = 0.0
    step_share:      float = 0.02   # 2% of the outcome, over a median episode

    # Loop penalty is already a rate (excess loops per step), so it does not
    # accumulate with trace length and stays absolute.
    w_loop:          float = 1.0

    # Cycle-time term, as a share of w_terminal charged when a case takes the
    # median real time. Charged at the end, on elapsed/real_median.
    #
    # This exists so the managerial interventions have something to bite on.
    # They change how long a case takes; if duration is not in the reward, that
    # change is invisible and the catalogue stays decorative no matter how it
    # is wired. 0.25 makes a case that takes twice the real median cost half a
    # conclusion bonus, which is enough to matter without swamping the outcome.
    time_share:      float = 0.25

    # Which side of the real median the cycle-time charge acts on.
    #
    #   "two_sided"  charge |log2(elapsed / real_median)|
    #   "slow_only"  charge max(0, log2(elapsed / real_median))
    #
    # "slow_only" was the original shape and it is why the agent races. It
    # floors the charge at zero below the real median, so finishing in an hour
    # a case that really takes a day costs exactly the same as finishing it in
    # a day — and the per-step cost then breaks the tie in favour of the hour.
    # Measured under it, simulated median cycle time came out 16x, 5x and 4x
    # faster than the real logs on BPIC2012 / BPIC2015 / BPIC2017. The term
    # was doing the opposite of its job: it was introduced to make duration
    # visible to the reward, and it made implausible speed free.
    #
    # "two_sided" charges deviation in both directions, so the target is the
    # real median rather than zero. It is the default. Set "slow_only" to
    # reproduce the earlier behaviour as an ablation.
    time_penalty_mode: str = "two_sided"

    # Safety rail on the cycle-time charge, in log2 units — 6.0 is a factor of
    # 64 either way.
    #
    # Not a shaping cap. An earlier version capped the *ratio* at 3.0, which
    # killed the gradient exactly where the agent operates (a random policy
    # already sits at 3.6x the real median on BPIC2012), so no policy could
    # beat doing nothing. This sits far outside the operating range — the
    # worst deviation ever measured is 4.0 — and exists only so that a
    # degenerate episode with a near-zero elapsed time cannot contribute an
    # arbitrarily large negative return and swamp the batch.
    time_penalty_cap: float = 6.0

    # Multiplier on the declared intervention costs and compliance charges in
    # intervention_effects.py, again as a share of w_terminal.
    intervention_cost_share: float = 1.0

    # Global scale on the intervention *effects*, for sensitivity analysis.
    # 0.0 disables the mechanism and reproduces the pre-fix behaviour, which is
    # the management-action ablation.
    effect_scale:    float = 1.0

    # Escape hatches: set either to pin the absolute per-step weight and
    # ignore the corresponding share. Used to reproduce the pre-fix setup.
    w_progress_abs:  "float | None" = None
    w_step_abs:      "float | None" = None

    def __post_init__(self) -> None:
        if self.w_bad_terminal >= 0:
            raise ValueError(
                f"w_bad_terminal must be negative, got {self.w_bad_terminal}"
            )
        if self.w_terminal < 0:
            raise ValueError(
                f"w_terminal must be non-negative, got {self.w_terminal}"
            )
        for name in ("progress_share", "step_share"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.time_penalty_mode not in TIME_PENALTY_MODES:
            raise ValueError(
                f"time_penalty_mode must be one of {sorted(TIME_PENALTY_MODES)}, "
                f"got {self.time_penalty_mode!r}"
            )
        if self.time_penalty_cap <= 0:
            raise ValueError(
                f"time_penalty_cap must be positive, got {self.time_penalty_cap}"
            )

    def per_step_weights(self, median_trace_length: float) -> tuple[float, float]:
        """
        Resolve ``(w_progress, w_step)`` for a process of this typical length.

        Returns the absolute overrides when they are set, otherwise spreads
        each share across one median-length episode.
        """
        n = max(float(median_trace_length), 1.0)
        w_progress = (
            self.w_progress_abs if self.w_progress_abs is not None
            else self.progress_share * self.w_terminal / n
        )
        w_step = (
            self.w_step_abs if self.w_step_abs is not None
            else self.step_share * self.w_terminal / n
        )
        return float(w_progress), float(w_step)

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RewardConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown reward keys {sorted(unknown)}. Expected {sorted(known)}."
            )
        return cls(**data)

    def save(self, directory: "str | Path") -> Path:
        path = Path(directory) / CONFIG_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, directory: "str | Path") -> "RewardConfig | None":
        """Return the config saved in `directory`, or None if there is none."""
        path = Path(directory) / CONFIG_FILENAME
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- legacy ---------------------------------------------------------

    @classmethod
    def from_legacy_file(cls, path: "str | Path") -> "RewardConfig":
        """
        Read an old per-dataset ``reward_weights.json``.

        Provided only so the pre-fix, per-dataset setup can be reproduced for
        comparison. The key names in that file do not describe what they set;
        the mapping below is the one notebook 04 actually used.
        """
        with open(path, encoding="utf-8") as fh:
            legacy = json.load(fh)
        default = cls()
        return cls(
            w_terminal     = float(legacy.get("w_completion", default.w_terminal)),
            w_bad_terminal = default.w_bad_terminal,
            w_length_bonus = default.w_length_bonus,
            w_loop         = float(legacy.get("w_rework", default.w_loop)),
            # The old file held absolute per-step weights, so pin them and let
            # the shares be ignored.
            w_step_abs     = float(legacy.get("w_throughput", 0.05)),
            w_progress_abs = float(legacy.get("w_delay",      0.3)),
        )

    # -- comparison ------------------------------------------------------

    def differences(self, other: "RewardConfig") -> dict:
        """Weights that differ between two configs, as {name: (self, other)}."""
        return {
            f.name: (getattr(self, f.name), getattr(other, f.name))
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        }


#: Shared across every dataset. Training and evaluation both use this.
DEFAULT = RewardConfig()
