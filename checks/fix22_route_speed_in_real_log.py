"""
fix22_route_speed_in_real_log.py
--------------------------------
The objective is to make the approve/reject decision **faster**, not to change
what it is. This check asks whether the agent's routing preferences point at
routes that are genuinely faster **in the real log** — with no digital twin, no
reward function and no simulated episode anywhere in the measurement.

Why this is the right test
--------------------------
Every speed claim so far has been measured inside the simulator, where the
agent controls which activities occur and the twin samples their durations.
Fix 16 showed how little that is worth: under the one-sided time charge the
agent raced to 16x faster than any real case, and under the two-sided charge it
hit the target by choosing slow activities and repeating them. In both cases
the "speed" was a property of the reward, not of the process.

Fix 21 then showed the routing recommendations differ between training seeds.
That means we cannot report *the* list. It does **not** establish that any
given list is wrong — different seeds may have found different but equally real
shortcuts. Instability is not invalidity, and the two need separating.

Both questions are settled by leaving the simulator entirely. The real log
records, for every case, the route it took and how long it took. So:

1. Take a trained policy's **preferred transitions** — those it uses more than
   a random policy does.
2. Score every **real** case by the fraction of its transitions that are in
   that preferred set.
3. Ask whether real cases with higher conformance reached their outcome faster.

If they did, the recommendation has support in the data independent of the
model that produced it, and seed-to-seed disagreement is a presentation problem
rather than a validity one. If they did not, the recommendation is an artefact
of the simulator whatever its stability.

Confounds handled
-----------------
* **Outcome.** Rejected cases are much faster than accepted ones on these logs,
  so conformance could just be tracking the outcome. Everything is computed
  **within** outcome group (good / bad) and never pooled across them.
* **Length.** Long cases have more transitions and more chances to leave the
  preferred set, and are trivially slower. Reported alongside the raw
  association is a partial correlation controlling for the case's step count,
  so "conformant" cannot simply mean "short".
* **Direction.** Reported as Spearman rho between conformance and cycle time.
  **Negative rho is the favourable result** — more conformance, less time.

Run:
    python checks/fix22_route_speed_in_real_log.py

Writes results/fix22_route_speed/{per_seed,summary}.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from env_factory import build_process_env                              # noqa: E402
from insights import collect_trajectories, routing_preference_table    # noqa: E402
from reward_config import RewardConfig                                 # noqa: E402
from timeutils import ensure_utc_timestamps, sort_events               # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]
SEEDS = [0, 1, 2, 3, 4]
SWEEP = REPO / "results" / "sweep"

N_EPISODES = 200
MIN_COUNT = 5
DAY = 86_400.0

#: A case needs at least this many transitions for its conformance score to
#: mean anything.
MIN_CASE_TRANSITIONS = 4

#: |rho| below this is no association regardless of significance — with ~13,000
#: cases, rho = 0.02 is significant and useless.
MEANINGFUL_RHO = 0.10


def preferred_transitions(dataset: str, seed: int) -> "set | None":
    """Transitions the trained policy uses more than a random policy does."""
    model_path = SWEEP / dataset / f"scale1_seed{seed}" / "model.zip"
    if not model_path.exists():
        return None

    from sb3_contrib import MaskablePPO

    out = REPO / "output" / dataset
    env, _ = build_process_env(out, reward_config=RewardConfig(), seed=42)
    model = MaskablePPO.load(model_path, env=env)

    rl = collect_trajectories(env, model, n_episodes=N_EPISODES)

    rng = np.random.default_rng(0)

    class _Random:
        def predict(self, obs, action_masks=None, deterministic=True):
            split = env.action_space.nvec[0]
            return np.array([
                int(rng.choice(np.where(action_masks[:split])[0])),
                int(rng.choice(np.where(action_masks[split:])[0])),
            ]), None

    rnd = collect_trajectories(env, _Random(), n_episodes=N_EPISODES)
    table = routing_preference_table(rl, rnd)

    preferred = table[(table["rl_count"] >= MIN_COUNT) & (table["log2_ratio"] > 0)]
    return set(zip(preferred["from_activity"], preferred["to_activity"]))


def real_cases(dataset: str) -> pd.DataFrame:
    """One row per real case: route, cycle time, steps, outcome."""
    out = REPO / "output" / dataset
    df = sort_events(ensure_utc_timestamps(
        pd.read_parquet(out / f"events_{dataset}_train.parquet")))
    with open(out / "terminal_classification.json", encoding="utf-8") as fh:
        tc = json.load(fh)
    good, bad = set(tc["good_terminals"]), set(tc["bad_terminals"])

    rows = []
    for case_id, g in df.groupby("case_id", sort=False):
        acts = g["activity"].tolist()
        ts = g["timestamp"].tolist()

        # Truncate at the first outcome marker: everything after the decision
        # is post-decision admin and is not part of time-to-decision.
        cut = next((i for i, a in enumerate(acts) if a in good or a in bad), None)
        if cut is None or cut < MIN_CASE_TRANSITIONS:
            continue
        outcome = "good" if acts[cut] in good else "bad"

        route = acts[: cut + 1]
        transitions = list(zip(route[:-1], route[1:]))
        rows.append({
            "case_id": case_id,
            "outcome": outcome,
            "n_transitions": len(transitions),
            "days_to_decision": (ts[cut] - ts[0]).total_seconds() / DAY,
            "transitions": transitions,
        })
    return pd.DataFrame(rows)


def partial_rho(x, y, z) -> float:
    """Spearman rho between x and y, controlling for z, via rank residuals."""
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    def resid(a):
        b = np.polyfit(rz, a, 1)
        return a - (b[0] * rz + b[1])
    ex, ey = resid(rx), resid(ry)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def main() -> int:
    per_seed, summary = [], []

    for dataset in DATASETS:
        cases = real_cases(dataset)
        if cases.empty:
            print(f"[skip] {dataset}: no cases with a recorded outcome")
            continue

        print(f"=== {dataset}   ({len(cases)} real cases with a decision)")
        for outcome, g in cases.groupby("outcome"):
            print(f"    {outcome}: {len(g)} cases, median "
                  f"{g['days_to_decision'].median():.2f} d to decision")

        for seed in SEEDS:
            pref = preferred_transitions(dataset, seed)
            if pref is None:
                continue

            conf = cases["transitions"].apply(
                lambda ts: sum(t in pref for t in ts) / len(ts))
            work = cases.assign(conformance=conf)

            for outcome, g in work.groupby("outcome"):
                if len(g) < 50 or g["conformance"].std() == 0:
                    continue
                rho = float(pd.Series(g["conformance"].values).corr(
                    pd.Series(g["days_to_decision"].values), method="spearman"))
                prho = partial_rho(g["conformance"].values,
                                   g["days_to_decision"].values,
                                   g["n_transitions"].values)

                lo, hi = g["conformance"].quantile([1 / 3, 2 / 3])
                bottom = g[g["conformance"] <= lo]["days_to_decision"].median()
                top = g[g["conformance"] >= hi]["days_to_decision"].median()

                per_seed.append({
                    "dataset": dataset, "seed": seed, "outcome": outcome,
                    "n_cases": len(g),
                    "n_preferred_transitions": len(pref),
                    "rho_conformance_vs_days": round(rho, 3),
                    "partial_rho_ctrl_length": round(prho, 3),
                    "median_days_low_conformance": round(bottom, 2),
                    "median_days_high_conformance": round(top, 2),
                    "speedup_ratio": round(top / bottom, 3) if bottom else None,
                })

    if not per_seed:
        print("Nothing to check — run run_experiments.py first.")
        return 1

    table = pd.DataFrame(per_seed)
    out_dir = REPO / "results" / "fix22_route_speed"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "per_seed.csv", index=False)

    print()
    print("Spearman rho between a REAL case's conformance to the agent's")
    print("preferred routes and its days-to-decision. NEGATIVE = conformant")
    print("real cases were faster = the recommendation has real support.")
    print()
    print(table.to_string(index=False))

    agg = (table.groupby(["dataset", "outcome"])
           .agg(n_seeds=("seed", "nunique"),
                mean_rho=("rho_conformance_vs_days", "mean"),
                min_rho=("rho_conformance_vs_days", "min"),
                max_rho=("rho_conformance_vs_days", "max"),
                mean_partial_rho=("partial_rho_ctrl_length", "mean"),
                mean_speedup=("speedup_ratio", "mean"))
           .round(3).reset_index())

    def verdict(r):
        if r["max_rho"] < -MEANINGFUL_RHO:
            return "supported (every seed)"
        if r["mean_rho"] < -MEANINGFUL_RHO:
            return "supported on average"
        if r["min_rho"] > MEANINGFUL_RHO:
            return "REVERSED — conformant cases are slower"
        return "no association"

    agg["verdict"] = agg.apply(verdict, axis=1)
    agg.to_csv(out_dir / "summary.csv", index=False)

    print()
    print(agg.to_string(index=False))
    print()
    print(f"Written: {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
