"""
fix21_insight_stability.py
--------------------------
Asks the only question that matters for the deliverable: **do the managerial
insights survive a change of training seed?**

`output/<DATASET>/routing_recommendations.csv` is what this project actually
hands a manager — statements of the form "the agent avoids W_Afhandelen leads
-> W_Beoordelen fraude; consider adding a routing constraint to discourage it".
Every one of them was produced by a single training run.

Fix 17 measured the across-seed spread of the *aggregate* metrics and found it
large (mean reward 14.24 +/- 12.26 on BPIC2015). That is a warning, not an
answer: aggregates can move while the underlying routing preferences stay put,
and they can also stay put while the preferences churn completely.

So this measures the recommendations themselves. For each dataset it loads the
five seed models trained at ``effect_scale = 1.0``, regenerates the routing
preference table from each, and reports:

* **Top-k Jaccard** — of the k most strongly preferred transitions, how many
  are the same set across seeds. This is the recommendation list.
* **Sign agreement** — over transitions that every seed exercised, how often
  all five seeds agree on whether the agent prefers or avoids it. A
  recommendation whose *direction* flips between seeds is not a finding.
* **Spearman rho** — rank correlation of ``log2_ratio`` over the common set,
  averaged over seed pairs.

Interpretation is stated up front so it cannot be chosen after seeing the
numbers:

    Jaccard >= 0.7 and sign agreement >= 0.9   the list is reproducible
    Jaccard >= 0.4 or  sign agreement >= 0.7   partially reproducible
    below that                                 the recommendations are an
                                               artefact of the training seed

Run:
    python checks/fix21_insight_stability.py

Writes results/fix21_insight_stability/{pairs,summary}.csv
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from env_factory import build_process_env          # noqa: E402
from insights import collect_trajectories, routing_preference_table  # noqa: E402
from reward_config import RewardConfig             # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]
SEEDS = [0, 1, 2, 3, 4]
SWEEP = REPO / "results" / "sweep"

N_EPISODES = 200
TOP_K = 10

#: A transition needs this many observations under a seed before that seed is
#: treated as having an opinion about it.
MIN_COUNT = 5

STRONG_JACCARD, STRONG_SIGN = 0.70, 0.90
PARTIAL_JACCARD, PARTIAL_SIGN = 0.40, 0.70

#: Minimum support before a sign-agreement or rank-correlation figure is
#: allowed to influence the verdict.
#:
#: The first version of this check did not have these guards and reported
#: BPIC2017 as "partially reproducible" on the strength of 100% sign agreement
#: over exactly one shared transition, with a mean Spearman rho of 0.943
#: computed over seed pairs sharing 1, 1, 2, 3, 3, 3, 4, 4, 4 and 7 points.
#: That is the same defect this repository just fixed in
#: checks/fix5_verdict_control.py: a statistic from a single-digit sample
#: compared against a threshold as though it carried information.
MIN_SHARED_FOR_SIGN = 10
MIN_COMMON_FOR_RHO = 10


def preference_table(dataset: str, seed: int) -> "pd.DataFrame | None":
    """Regenerate one seed's routing preference table."""
    model_path = SWEEP / dataset / f"scale1_seed{seed}" / "model.zip"
    if not model_path.exists():
        return None

    from sb3_contrib import MaskablePPO

    out = REPO / "output" / dataset
    env, _ = build_process_env(out, reward_config=RewardConfig(), seed=42)
    model = MaskablePPO.load(model_path, env=env)

    rl = collect_trajectories(env, model, n_episodes=N_EPISODES)

    # The random baseline is only a denominator here. It is regenerated per
    # seed from the same env seed so it is identical across seeds and cannot
    # itself contribute churn.
    rng = np.random.default_rng(0)

    class _Random:
        def predict(self, obs, action_masks=None, deterministic=True):
            nvec = env.action_space.nvec
            split = nvec[0]
            route = int(rng.choice(np.where(action_masks[:split])[0]))
            mgmt = int(rng.choice(np.where(action_masks[split:])[0]))
            return np.array([route, mgmt]), None

    rnd = collect_trajectories(env, _Random(), n_episodes=N_EPISODES)
    return routing_preference_table(rl, rnd)


def key_set(table: pd.DataFrame) -> pd.DataFrame:
    """Index the table by the transition, keeping only exercised rows."""
    t = table.copy()
    t["key"] = t["from_activity"] + " -> " + t["to_activity"]
    return t.set_index("key")


def main() -> int:
    pair_rows, summary_rows = [], []

    for dataset in DATASETS:
        tables = {}
        for seed in SEEDS:
            t = preference_table(dataset, seed)
            if t is not None:
                tables[seed] = key_set(t)
        if len(tables) < 2:
            print(f"[skip] {dataset}: need at least two seed models under "
                  f"{SWEEP / dataset}")
            continue

        print(f"=== {dataset}   ({len(tables)} seeds, {N_EPISODES} episodes each)")

        # Top-k preferred transitions per seed, by log2_ratio among exercised rows.
        tops = {}
        for seed, t in tables.items():
            exercised = t[t["rl_count"] >= MIN_COUNT]
            tops[seed] = set(
                exercised.sort_values("log2_ratio", ascending=False)
                .head(TOP_K).index
            )

        jaccards, rhos = [], []
        for a, b in itertools.combinations(sorted(tables), 2):
            inter = len(tops[a] & tops[b])
            union = len(tops[a] | tops[b])
            jac = inter / union if union else float("nan")

            common = tables[a].index.intersection(tables[b].index)
            common = [
                k for k in common
                if tables[a].loc[k, "rl_count"] >= MIN_COUNT
                and tables[b].loc[k, "rl_count"] >= MIN_COUNT
            ]
            if len(common) >= MIN_COMMON_FOR_RHO:
                x = tables[a].loc[common, "log2_ratio"].astype(float)
                y = tables[b].loc[common, "log2_ratio"].astype(float)
                rho = float(pd.Series(x.values).corr(pd.Series(y.values),
                                                     method="spearman"))
            else:
                rho = float("nan")

            jaccards.append(jac)
            rhos.append(rho)
            pair_rows.append({
                "dataset": dataset, "seed_a": a, "seed_b": b,
                "topk_jaccard": round(jac, 3),
                "topk_shared": inter,
                "n_common_transitions": len(common),
                "spearman_rho": None if rho != rho else round(rho, 3),
            })

        # Sign agreement across every seed, on transitions all seeds exercised.
        shared = None
        for t in tables.values():
            idx = set(t[t["rl_count"] >= MIN_COUNT].index)
            shared = idx if shared is None else (shared & idx)
        shared = sorted(shared or [])

        if shared:
            signs = np.array([
                [np.sign(tables[s].loc[k, "log2_ratio"]) for k in shared]
                for s in sorted(tables)
            ])
            unanimous = (np.abs(signs.sum(axis=0)) == len(tables)).mean()
        else:
            unanimous = float("nan")

        mean_jac = float(np.nanmean(jaccards)) if jaccards else float("nan")
        mean_rho = float(np.nanmean(rhos)) if not np.all(np.isnan(rhos)) else float("nan")
        mean_common = float(np.mean([r["n_common_transitions"]
                                     for r in pair_rows
                                     if r["dataset"] == dataset]))

        # Sign agreement only counts when enough transitions are shared.
        sign_supported = len(shared) >= MIN_SHARED_FOR_SIGN
        sign_term = unanimous if sign_supported else float("nan")

        if sign_supported and mean_jac >= STRONG_JACCARD and unanimous >= STRONG_SIGN:
            verdict = "reproducible"
        elif mean_jac >= PARTIAL_JACCARD or (sign_supported and unanimous >= PARTIAL_SIGN):
            verdict = "partially reproducible"
        else:
            verdict = "SEED ARTEFACT"

        print(f"  top-{TOP_K} preferred set, mean Jaccard over seed pairs   "
              f"{mean_jac:.3f}")
        print(f"  transitions every seed exercised                     "
              f"{len(shared)}")
        agree = ("n/a" if not sign_supported else f"{unanimous:.1%}")
        note = "" if sign_supported else f"  (< {MIN_SHARED_FOR_SIGN}, no support)"
        print(f"  of those, all seeds agree on the direction           "
              f"{agree}{note}")
        rho_txt = "n/a" if mean_rho != mean_rho else f"{mean_rho:.3f}"
        print(f"  mean Spearman rho on log2_ratio                      "
              f"{rho_txt}   (pairs with >= {MIN_COMMON_FOR_RHO} common)")
        print(f"  mean transitions shared by a seed PAIR               "
              f"{mean_common:.1f}")
        print(f"  --> {verdict}")
        print()

        summary_rows.append({
            "dataset": dataset,
            "n_seeds": len(tables),
            "topk": TOP_K,
            "mean_topk_jaccard": round(mean_jac, 3),
            "n_shared_transitions": len(shared),
            "sign_agreement": None if sign_term != sign_term else round(sign_term, 3),
            "sign_supported": sign_supported,
            "mean_pair_common": round(mean_common, 1),
            "mean_spearman_rho": None if mean_rho != mean_rho else round(mean_rho, 3),
            "verdict": verdict,
        })

    if not summary_rows:
        print("Nothing to check — run run_experiments.py first.")
        return 1

    out_dir = REPO / "results" / "fix21_insight_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(out_dir / "pairs.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(f"Written: {out_dir / 'summary.csv'}")

    artefacts = summary[summary["verdict"] == "SEED ARTEFACT"]["dataset"].tolist()
    if artefacts:
        print()
        print("The routing recommendations shipped in "
              "output/<DATASET>/routing_recommendations.csv come from a single "
              "training run. On " + ", ".join(artefacts) + " a different seed "
              "produces a different list, so those files state as findings "
              "something that is a property of the seed.")
        return 1

    print("Recommendations are stable enough across seeds to be reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
