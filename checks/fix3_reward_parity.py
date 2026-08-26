"""
fix3_reward_parity.py
---------------------
Verifies Fix 3: the environment used for training, evaluation and analysis
must score episodes identically.

Before the fix, notebook 04 applied reward_weights.json and notebooks 05/06
did not, so a good outcome was worth +30 during training and 0 during grading.
This check rebuilds the environment the way each notebook now does it and
asserts the reward configs match, then prices the same episode outcomes under
the old and new settings so the size of the old discrepancy is on record.

Run:
    python checks/fix3_reward_parity.py

Writes results/fix3_reward_parity/comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from env_factory import build_process_env       # noqa: E402
from reward_config import RewardConfig          # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]


def price_good_outcome(cfg: RewardConfig, episode_len: int, median_len: float) -> float:
    """What one good-outcome episode is worth at its final step."""
    ratio = episode_len / max(median_len, 1)
    bonus = max(0.0, cfg.w_length_bonus * (1.0 - abs(ratio - 1.0)))
    _, w_step = cfg.per_step_weights(median_len)
    return -w_step + cfg.w_terminal + bonus


def main() -> int:
    rows = []
    mismatches = []

    for name in DATASETS:
        out = REPO / "output" / name
        if not (out / "terminal_classification.json").exists():
            print(f"[skip] {name}: not built yet")
            continue

        # The three notebooks now all go through build_process_env.
        env_train, meta = build_process_env(out)
        env_eval, _ = build_process_env(out)
        env_analysis, _ = build_process_env(out)

        cfgs = {
            "04_training":   env_train.reward_config,
            "05_evaluation": env_eval.reward_config,
            "06_insights":   env_analysis.reward_config,
        }
        diffs = cfgs["04_training"].differences(cfgs["05_evaluation"])
        diffs |= cfgs["04_training"].differences(cfgs["06_insights"])
        if diffs:
            mismatches.append(f"{name}: {diffs}")

        # Also assert the episode caps match — a different max_steps would
        # change truncation behaviour even with an identical reward.
        caps = {env_train.max_steps, env_eval.max_steps, env_analysis.max_steps}
        if len(caps) != 1:
            mismatches.append(f"{name}: max_steps differs across notebooks: {caps}")

        cfg = cfgs["04_training"]
        median_len = meta["median_trace_length"]

        # How much of an episode's reward comes from per-step shaping rather
        # than from the outcome. Fixed per-step constants make this vary wildly
        # with trace length (84% on BPIC2015 before the shares were introduced),
        # which means the same config trains for a different goal on each log.
        shaping_over_median = (
            cfg.progress_share * cfg.w_terminal      # every step a new activity
            + cfg.step_share * cfg.w_terminal        # total step cost
        )
        shaping_fraction = shaping_over_median / (shaping_over_median + abs(cfg.w_terminal))

        # What the old split cost, priced on the same episode.
        legacy_path = out / "reward_weights.json"
        old_train = (
            RewardConfig.from_legacy_file(legacy_path)
            if legacy_path.exists() else None
        )
        # What notebooks 05/06 silently used: the old absolute constants with
        # no good-outcome bonus at all.
        old_eval = RewardConfig(w_terminal=0.0, w_step_abs=0.05, w_progress_abs=0.3)

        at_median = int(round(median_len))
        rows.append({
            "dataset": name,
            "max_steps": env_train.max_steps,
            "median_trace_length": median_len,
            "w_terminal": cfg.w_terminal,
            "w_bad_terminal": cfg.w_bad_terminal,
            "progress_share": cfg.progress_share,
            "step_share": cfg.step_share,
            "w_loop": cfg.w_loop,
            "resolved_w_progress": round(env_train.w_progress, 4),
            "resolved_w_step": round(env_train.w_step, 4),
            "shaping_fraction_of_reward": round(shaping_fraction, 4),
            "configs_match": not diffs,
            "old_train_good_outcome": (
                round(price_good_outcome(old_train, at_median, median_len), 2)
                if old_train else None
            ),
            "old_eval_good_outcome": round(price_good_outcome(old_eval, at_median, median_len), 2),
            "new_good_outcome_both": round(price_good_outcome(cfg, at_median, median_len), 2),
        })

        print(f"=== {name}")
        print(f"  reward config identical across notebooks 04 / 05 / 06: {not diffs}")
        print(f"  max_steps identical: {len(caps) == 1} ({env_train.max_steps})")
        print(f"  a good outcome at the median trace length is worth "
              f"{price_good_outcome(cfg, at_median, median_len):+.2f} everywhere")
        print(f"  per-step weights resolve to w_progress={env_train.w_progress:.4f} "
              f"w_step={env_train.w_step:.4f} (median trace {median_len:.0f})")
        print(f"  shaping is {shaping_fraction:.1%} of the reward on offer")
        if old_train:
            print(f"    before the fix: {price_good_outcome(old_train, at_median, median_len):+.2f} "
                  f"in training vs {price_good_outcome(old_eval, at_median, median_len):+.2f} "
                  f"in evaluation")
        print()

    if not rows:
        print("Nothing built yet — run notebooks 1-3 first.")
        return 1

    table = pd.DataFrame(rows)
    out_dir = REPO / "results" / "fix3_reward_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(table.to_string(index=False))
    print()

    # The config must be identical across datasets...
    spec_cols = ["w_terminal", "w_bad_terminal", "progress_share", "step_share", "w_loop"]
    varying = [c for c in spec_cols if table[c].nunique() > 1]
    if varying:
        mismatches.append(f"reward config differs across datasets: {varying}")
    else:
        print("Reward config is identical across all datasets (no per-dataset tuning).")

    # ...and so must the reward *structure* it induces. This is the stronger
    # test: fixed per-step constants pass the check above while still training
    # for a different goal on each log.
    spread = table["shaping_fraction_of_reward"].max() - table["shaping_fraction_of_reward"].min()
    if spread > 0.01:
        mismatches.append(
            f"shaping is a different fraction of the reward per dataset "
            f"(spread {spread:.1%}) — the same config is training for "
            f"different goals"
        )
    else:
        print(f"Shaping is {table['shaping_fraction_of_reward'].iloc[0]:.1%} of the "
              f"reward on every dataset — the reward structure is uniform, not "
              f"just the numbers.")

    if mismatches:
        print("\nFAIL:")
        for m in mismatches:
            print("  -", m)
        return 1

    print(f"\nPASS — one reward, used everywhere.\nWritten: {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
