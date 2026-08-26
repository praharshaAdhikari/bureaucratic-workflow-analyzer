"""
fix5_verdict_control.py
-----------------------
Verifies that the agent cannot choose whether a case succeeds.

The test is policy-invariance: run two deliberately different policies and
compare the resulting good-outcome rate. If the verdict is the agent's to
pick, the rate moves with the policy. If the environment owns it, the rate
sits at the log's base rate whatever the policy does.

Both modes are measured, so the difference is reported rather than asserted.

The comparison is made on a confidence interval, not on the point estimates.
The always-take-the-first-successor policy concludes only about 2% of episodes
on BPIC2015 — 7 to 9 of 400 — so its good rate is a fraction with a
single-digit denominator. Compared point-to-point against a rate from ~390
episodes it swings by 10-20% from run to run purely by resampling, and the
check reported that as "the verdict is still steerable" or not depending on
the seed. Reward weights cannot touch the verdict (the rollout here uses no
model, and the environment's dynamics do not read the reward), so a check that
flips between runs was measuring its own sample size.

Run:
    python checks/fix5_verdict_control.py

Writes results/fix5_verdict_control/comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from env_factory import build_process_env    # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]
N_EPISODES = 400

#: Deliberately different routing policies. `random` is the reference; each
#: fixed policy is a separate contrast against it.
FIXED_POLICIES = ("first", "last")
POLICIES = ("random", *FIXED_POLICIES)

#: How far the good-outcome rate may move between two policies before we call
#: the verdict steerable. Applied to the *lower* end of the 95% interval on the
#: difference, so a swing has to be larger than the sampling noise to count.
POLICY_INVARIANCE_TOLERANCE = 0.10

#: Below this many concluded episodes an arm carries no information about the
#: base rate, and the comparison is reported as inconclusive rather than
#: silently passed or failed. A policy that concludes 7 of 400 episodes tells
#: us about its own conclusion rate, not about the verdict.
MIN_CONCLUDED = 30

#: How far the good-outcome rate may sit from the real base rate.
BASE_RATE_TOLERANCE = 0.10


def diff_ci_low(p1: float, n1: int, p2: float, n2: int, z: float = 1.96) -> float:
    """
    Lower bound of the 95% interval on |p1 - p2| for two independent rates.

    Returns 0.0 when either arm is empty. The upper bound is not used: the
    question is whether the swing is *at least* as large as the tolerance, and
    a wide interval on a tiny sample should not be able to answer yes.
    """
    if not n1 or not n2:
        return 0.0
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return max(0.0, abs(p1 - p2) - z * se)


def rollout(env, n: int, policy: str, seed: int = 0) -> dict:
    """
    `policy` is one of:

        random   uniform over the valid routing choices
        first    always the lowest valid successor index
        last     always the highest valid successor index

    Two fixed policies rather than one because ``first`` walks into a dead end
    on BPIC2015 and concludes 9 episodes in 400, which is not enough to say
    anything about its outcome rate. ``last`` concludes 47 and gives the
    contrast the check is trying to make.
    """
    rng = np.random.default_rng(seed)
    good = 0
    # Lengths are tracked only for episodes that actually reached an outcome.
    # A truncated episode can legitimately be shorter than the floor — a dead
    # end below it is abandoned rather than recorded as an outcome — so mixing
    # the two would make the floor look violated when it is not.
    concluded_lengths, truncated = [], 0
    for _ in range(n):
        env.reset()
        done = trunc = False
        steps = 0
        while not (done or trunc):
            mask = env.action_masks()
            split = env.action_space.nvec[0]
            valid = np.where(mask[:split])[0]
            if policy == "first":
                route = int(valid[0])
            elif policy == "last":
                route = int(valid[-1])
            else:
                route = int(rng.choice(valid))
            mgmt = int(rng.choice(np.where(mask[split:])[0]))
            _, _, done, trunc, _ = env.step(np.array([route, mgmt]))
            steps += 1
        if done:
            concluded_lengths.append(steps)
            good += env._current_activity in env._good_terminals
        else:
            truncated += 1
    n_conc = len(concluded_lengths)
    return {
        "concluded": n_conc / n,
        "truncated": truncated / n,
        "good_rate": good / n_conc if n_conc else float("nan"),
        "n_concluded": n_conc,
        "mean_len": float(np.mean(concluded_lengths)) if n_conc else float("nan"),
        "min_len": int(np.min(concluded_lengths)) if n_conc else -1,
    }


def main() -> int:
    rows, failures = [], []

    for name in DATASETS:
        out = REPO / "output" / name
        if not (out / "terminal_classification.json").exists():
            print(f"[skip] {name}: not built yet")
            continue

        try:
            env_env, meta = build_process_env(out, verdict_mode="environment")
            env_agent, _ = build_process_env(out, verdict_mode="agent")
        except ValueError as exc:
            print(f"[stale] {name}: {exc}")
            failures.append(f"{name}: {exc}")
            continue

        real_p_good = meta["outcome_base_rate_p_good"]
        floor = env_env.min_steps_to_outcome
        real_median = meta["steps_to_outcome_real"].get("p50")

        result = {}
        for mode, env in (("agent", env_agent), ("environment", env_env)):
            for policy in POLICIES:
                result[(mode, policy)] = rollout(env, N_EPISODES, policy)

        env_r = result[("environment", "random")]
        ag_r  = result[("agent", "random")]

        # Every fixed policy that concluded often enough to carry information
        # is a separate contrast. The verdict is steerable if any of them moves
        # the rate; it is untested if none of them concluded enough.
        contrasts = []
        for policy in FIXED_POLICIES:
            arm = result[("environment", policy)]
            if arm["n_concluded"] < MIN_CONCLUDED:
                continue
            contrasts.append((
                policy,
                abs(env_r["good_rate"] - arm["good_rate"]),
                diff_ci_low(env_r["good_rate"], env_r["n_concluded"],
                            arm["good_rate"], arm["n_concluded"]),
                arm,
            ))

        env_comparable = bool(contrasts)
        if contrasts:
            worst = max(contrasts, key=lambda c: c[2])
            worst_policy, env_swing, env_swing_low = worst[0], worst[1], worst[2]
            env_f = worst[3]
        else:
            worst_policy, env_swing, env_swing_low = "-", float("nan"), 0.0
            env_f = result[("environment", FIXED_POLICIES[0])]

        agent_arms = [result[("agent", p)] for p in FIXED_POLICIES
                      if result[("agent", p)]["n_concluded"] >= MIN_CONCLUDED]
        agent_swing = (
            max(abs(ag_r["good_rate"] - a["good_rate"]) for a in agent_arms)
            if agent_arms else float("nan")
        )
        env_vs_real = abs(result[("environment", "random")]["good_rate"] - real_p_good)

        rows.append({
            "dataset": name,
            "real_p_good": round(real_p_good, 4),
            "agent_good_random": round(result[("agent", "random")]["good_rate"], 4),
            "agent_good_fixed": round(min(
                (a["good_rate"] for a in agent_arms), default=float("nan")), 4),
            "agent_swing": round(agent_swing, 4),
            "env_good_random": round(result[("environment", "random")]["good_rate"], 4),
            "env_fixed_policy": worst_policy,
            "env_good_fixed": round(env_f["good_rate"], 4),
            "env_swing": round(env_swing, 4),
            "env_swing_ci_low": round(env_swing_low, 4),
            "env_n_concluded_random": env_r["n_concluded"],
            "env_n_concluded_fixed": env_f["n_concluded"],
            **{f"env_n_concluded_{p}": result[("environment", p)]["n_concluded"]
               for p in FIXED_POLICIES},
            "env_comparable": env_comparable,
            "env_vs_real": round(env_vs_real, 4),
            "env_mean_len": round(result[("environment", "random")]["mean_len"], 1),
            "real_median_steps": real_median,
            "floor": floor,
            "env_min_len": result[("environment", "random")]["min_len"],
        })

        print(f"=== {name}   real p(good) = {real_p_good:.1%}")
        for mode in ("agent", "environment"):
            r = result[(mode, "random")]
            arms = "  ".join(
                f"{p} {result[(mode, p)]['good_rate']:6.1%} "
                f"(n={result[(mode, p)]['n_concluded']:3d})"
                for p in FIXED_POLICIES
            )
            print(f"  {mode:12s} good: random {r['good_rate']:6.1%} "
                  f"(n={r['n_concluded']:3d})   {arms}")
        print(f"  environment episode length {result[('environment','random')]['mean_len']:.1f} "
              f"(real median to outcome {real_median:.0f}), shortest "
              f"{result[('environment','random')]['min_len']}, floor {floor}")
        print()

        if not env_comparable:
            failures.append(
                f"{name}: no fixed policy concluded {MIN_CONCLUDED} of "
                f"{N_EPISODES} episodes, so policy-invariance was not tested. "
                f"That is an untested claim, not a passing one."
            )
        elif env_swing_low > POLICY_INVARIANCE_TOLERANCE:
            failures.append(
                f"{name}: good rate moved {env_swing:.1%} between random and "
                f"'{worst_policy}' (at least {env_swing_low:.1%} after sampling "
                f"error) — the verdict is still steerable"
            )
        if env_vs_real > BASE_RATE_TOLERANCE:
            failures.append(
                f"{name}: good rate {result[('environment','random')]['good_rate']:.1%} "
                f"is {env_vs_real:.1%} off the real base rate {real_p_good:.1%}"
            )
        if result[("environment", "random")]["min_len"] < floor:
            failures.append(
                f"{name}: an episode concluded after "
                f"{result[('environment','random')]['min_len']} steps, below the "
                f"floor of {floor}"
            )

    if not rows:
        print("Nothing to check — run notebook 02 first.")
        return 1

    table = pd.DataFrame(rows)
    out_dir = REPO / "results" / "fix5_verdict_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(table.to_string(index=False))
    print()

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1

    print("PASS — the good-outcome rate tracks the real base rate and does not "
          "move with the policy.")
    print(f"Written: {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
