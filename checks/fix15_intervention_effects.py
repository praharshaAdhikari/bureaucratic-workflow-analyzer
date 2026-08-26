"""
fix15_intervention_effects.py
-----------------------------
Verifies that the managerial action catalogue actually does something.

Before this, holding routing fixed and swapping the entire management policy
moved total reward by 0.02% and left episode length and outcome rate unchanged
to the decimal. The two trained policies disagreed completely about which
action to use — 0.05% no-op on BPIC2015 against 91.9% on BPIC2017 — which is
what a policy choosing arbitrarily looks like.

Three things are checked:

1. **The ablation is clean.** With ``effect_scale = 0`` every management policy
   must give byte-identical results, reproducing the pre-fix behaviour. This is
   what makes "management actions on/off" a real experiment rather than an
   assertion.
2. **Interventions have causal effect.** With effects on, different management
   policies must produce different cycle times.
3. **Good management can beat no management.** If no policy beats always-no-op
   the catalogue is connected but mis-priced, and the agent still has nothing
   to learn.

Run:
    python checks/fix15_intervention_effects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from env_factory import build_process_env      # noqa: E402
from reward_config import RewardConfig         # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]
N_EPISODES = 300


def rollout(dataset: str, mgmt, cfg: RewardConfig, n: int = N_EPISODES) -> dict:
    """`mgmt` maps (kpi_vector, mgmt_mask) -> action index."""
    env, _ = build_process_env(REPO / "output" / dataset, reward_config=cfg)
    route = np.random.default_rng(7)           # identical routing every time
    rewards, cycles, used, steps = [], [], 0, 0
    for _ in range(n):
        env.reset()
        done = trunc = False
        total = 0.0
        info: dict = {}
        while not (done or trunc):
            mask = env.action_masks()
            split = env.action_space.nvec[0]
            r_idx = int(route.choice(np.where(mask[:split])[0]))
            m_idx = mgmt(env._build_kpi_vec(), mask[split:])
            steps += 1
            used += m_idx != 0
            _, r, done, trunc, info = env.step(np.array([r_idx, m_idx]))
            total += r
        rewards.append(total)
        cycles.append(info.get("cycle_time_s", np.nan) / 86400.0)
    return {
        "reward": float(np.mean(rewards)),
        "cycle": float(np.nanmedian(cycles)),
        "intervened": used / max(steps, 1),
    }


def noop(kpi, mask):
    return 0


def always(idx):
    return lambda kpi, mask: idx if mask[idx] else 0


def selective(kpi, mask):
    """Intervene in proportion to how far behind the case is running."""
    delay = kpi[0]
    if delay > 2.0 and mask[6]:
        return 6                                # escalate
    if delay > 1.0 and mask[2]:
        return 2                                # rebalance
    if delay > 0.5 and mask[4]:
        return 4                                # prioritise
    return 0


def main() -> int:
    rows, failures = [], []

    for name in DATASETS:
        if not (REPO / "output" / name / "terminal_classification.json").exists():
            print(f"[skip] {name}: not built yet")
            continue

        off = {label: rollout(name, pol, RewardConfig(effect_scale=0.0))
               for label, pol in [("no-op", noop), ("prioritise", always(4)),
                                  ("escalate", always(6))]}
        on = {label: rollout(name, pol, RewardConfig())
              for label, pol in [("no-op", noop), ("prioritise", always(4)),
                                 ("escalate", always(6)), ("selective", selective)]}

        ablation_clean = (
            abs(off["no-op"]["reward"] - off["prioritise"]["reward"]) < 1e-9
            and abs(off["no-op"]["reward"] - off["escalate"]["reward"]) < 1e-9
        )
        cycle_moves = abs(on["no-op"]["cycle"] - on["escalate"]["cycle"]) > 1e-6
        best_label = max(on, key=lambda k: on[k]["reward"])
        beats_noop = on[best_label]["reward"] - on["no-op"]["reward"]

        print(f"=== {name}")
        print(f"  effects OFF — every policy identical: {ablation_clean}")
        for label, r in on.items():
            print(f"  effects ON  {label:11s} reward {r['reward']:+8.3f} "
                  f"({r['reward'] - on['no-op']['reward']:+6.3f} vs no-op)  "
                  f"cycle {r['cycle']:8.2f}d  intervened {r['intervened']:5.1%}")
        print(f"  best policy: {best_label} ({beats_noop:+.3f} over no-op)")
        print()

        rows.append({
            "dataset": name,
            "ablation_clean": ablation_clean,
            "cycle_responds": cycle_moves,
            "best_policy": best_label,
            "best_margin_over_noop": round(beats_noop, 4),
            **{f"reward_{k}": round(v["reward"], 3) for k, v in on.items()},
            **{f"cycle_{k}": round(v["cycle"], 3) for k, v in on.items()},
        })

        if not ablation_clean:
            failures.append(f"{name}: effect_scale=0 does not reproduce identical results")
        if not cycle_moves:
            failures.append(f"{name}: cycle time does not respond to interventions")
        if beats_noop <= 0:
            failures.append(
                f"{name}: no management policy beats doing nothing "
                f"({beats_noop:+.3f}) — the catalogue is connected but priced so "
                f"that nothing pays"
            )

    if not rows:
        print("Nothing to check.")
        return 1

    table = pd.DataFrame(rows)
    out_dir = REPO / "results" / "fix15_intervention_effects"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(table[["dataset", "ablation_clean", "cycle_responds",
                 "best_policy", "best_margin_over_noop"]].to_string(index=False))
    print()

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1

    print("PASS — the catalogue is off cleanly, causal when on, and good "
          "management beats none.")
    print(f"Written: {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
