"""
fix4_outcome_floor.py
---------------------
Verifies Fix 4: an episode may not reach an outcome faster than the fastest
real case does.

Every edge in the fitted transition graph is real, but a first-order Markov
chain composes them into paths no case ever took. Per-edge masking cannot see
this — the path is only implausible as a whole. Before the floor:

    dataset    shortest simulated route    real minimum    trained agent
    BPIC2012            4 steps                 3            9.5 steps  (fine)
    BPIC2015            2 steps                11            2.3 steps  (exploit)
    BPIC2017            3 steps                13            3.4 steps  (exploit)

Run:
    python checks/fix4_outcome_floor.py

Writes results/fix4_outcome_floor/comparison.csv
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from env_factory import build_process_env    # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]
N_ROLLOUTS = 300


def shortest_route_to_good(twin, good: set, terminals: set, start: str) -> int | None:
    """Fewest transitions from `start` to any good outcome, terminals absorbing."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        act, dist = queue.popleft()
        for nxt in twin.transition_probs.get(act, {}):
            if nxt in good:
                return dist + 1
            if nxt in terminals or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def random_rollouts(env, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # Only episodes that reached an outcome count towards the length stats.
    # A dead end below the floor is abandoned without an outcome, so those
    # episodes are legitimately shorter and would look like floor violations.
    concluded = []
    truncated = 0
    for _ in range(n):
        env.reset()
        done = trunc = False
        steps = 0
        while not (done or trunc):
            mask = env.action_masks()
            split = env.action_space.nvec[0]
            action = np.array([
                int(rng.choice(np.where(mask[:split])[0])),
                int(rng.choice(np.where(mask[split:])[0])),
            ])
            _, _, done, trunc, _ = env.step(action)
            steps += 1
        if done:
            concluded.append(steps)
        else:
            truncated += 1
    return {
        "mean_len": float(np.mean(concluded)) if concluded else float("nan"),
        "min_len": int(np.min(concluded)) if concluded else -1,
        "terminated": len(concluded) / n,
        "truncated": truncated / n,
    }


def main() -> int:
    rows, failures = [], []

    for name in DATASETS:
        out = REPO / "output" / name
        tc_path = out / "terminal_classification.json"
        if not tc_path.exists():
            print(f"[skip] {name}: not built yet")
            continue

        terminals = json.loads(tc_path.read_text())
        if "min_steps_to_outcome" not in terminals:
            failures.append(f"{name}: terminal_classification.json predates the "
                            f"floor — re-run notebook 02")
            print(f"[stale] {name}: no min_steps_to_outcome; re-run notebook 02")
            continue

        try:
            env, meta = build_process_env(out)
        except ValueError as exc:
            failures.append(f"{name}: {exc}")
            print(f"[stale] {name}: {exc}")
            continue
        floor = env.min_steps_to_outcome
        real = terminals["steps_to_outcome"]

        good = set(terminals["good_terminals"])
        alls = good | set(terminals["bad_terminals"])
        start = max(env.twin.start_activities, key=env.twin.start_activities.get)
        graph_shortest = shortest_route_to_good(env.twin, good, alls, start)

        roll = random_rollouts(env, N_ROLLOUTS)

        rows.append({
            "dataset": name,
            "graph_shortest_route": graph_shortest,
            "real_min": real.get("min"),
            "real_p1": real.get("p1"),
            "real_p50": real.get("p50"),
            "floor_applied": floor,
            "random_min_episode_len": roll["min_len"],
            "random_mean_episode_len": round(roll["mean_len"], 1),
            "random_terminated": roll["terminated"],
        })

        print(f"=== {name}")
        print(f"  transition graph still permits a {graph_shortest}-step route to a good outcome")
        print(f"  real log: min {real.get('min'):.0f}, p1 {real.get('p1'):.0f}, "
              f"median {real.get('p50'):.0f}")
        print(f"  floor applied: {floor} steps")
        print(f"  random policy, episodes that reached an outcome: shortest "
              f"{roll['min_len']} steps, mean {roll['mean_len']:.1f}, "
              f"{roll['terminated']:.0%} of episodes ({roll['truncated']:.0%} abandoned)")

        # The floor must actually bind: no episode may finish before it.
        if roll["min_len"] < floor and roll["terminated"] > 0:
            failures.append(
                f"{name}: an episode ended after {roll['min_len']} steps, "
                f"below the floor of {floor}"
            )
        # And the floor must not strand episodes at the step cap.
        if roll["terminated"] < 0.5:
            failures.append(
                f"{name}: only {roll['terminated']:.0%} of episodes reach an "
                f"outcome — the floor may be trapping them"
            )
        print()

    if not rows:
        print("Nothing to check.")
        return 1

    table = pd.DataFrame(rows)
    out_dir = REPO / "results" / "fix4_outcome_floor"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(table.to_string(index=False))
    print()

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1

    print("PASS — no episode reaches an outcome faster than the fastest real case.")
    print(f"Written: {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
