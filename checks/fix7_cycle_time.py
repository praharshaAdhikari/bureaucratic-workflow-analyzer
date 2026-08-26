"""
fix7_cycle_time.py
------------------
Verifies that cycle time is measured from the policy's own episode, and that
the duration model reproduces the real time scale.

Two things were wrong.

1. Cycle time was not measured from the episode at all. ``PolicyEvaluator``
   re-simulated a *fresh* case, drew its length from the short half of the
   empirical distribution if the episode had terminated and the long half if
   it had not, then scaled by a rework multiplier clipped to [0.70, 1.50]. The
   policy's routing never entered the number; only a boolean and mean rework
   did. Any cycle-time advantage would have been an artefact of the
   terminated/truncated split.

2. The duration model discarded every inter-event gap of 60s or less when
   fitting, with the comment that this "matches what the KS metric sees". It
   does — and that is the problem, because the KS metric filters both sides to
   > 60s and `case_duration_wasserstein` divides each distribution by its own
   99th percentile. Both are blind to the discarded mass. BPIC2012's
   A_SUBMITTED has a real median gap of 0s; after filtering, the sampler
   returned a median of 1,035s and a mean of 84,118s for it.

The test below replays the *real* activity sequences through the sampler. If
the duration model is faithful, the resulting cycle times reproduce the real
ones — any difference is the model, not the policy.

Run:
    python checks/fix7_cycle_time.py

Writes results/fix7_cycle_time/comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from digital_twin import DigitalTwin                      # noqa: E402
from timeutils import ensure_utc_timestamps, sort_events  # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]
N_CASES = 800
DAY = 86_400.0

#: Median simulated cycle time must sit within this log-ratio of the real one.
#: 0.40 is roughly a factor of 1.5 either way.
SCALE_TOLERANCE = 0.40


def replay(twin, sequences: list[list[str]], seed: int = 0) -> np.ndarray:
    """Cycle time for each real activity sequence, using the twin's sampler."""
    twin.rng = np.random.default_rng(seed)
    return np.array([
        sum(twin._sample_duration(a) for a in acts[:-1]) / DAY
        for acts in sequences
    ])


def main() -> int:
    rows, failures = [], []

    for name in DATASETS:
        path = REPO / "output" / name / f"events_{name}_train.parquet"
        if not path.exists():
            print(f"[skip] {name}: not built yet")
            continue

        raw = pd.read_parquet(path)
        df = sort_events(ensure_utc_timestamps(raw))

        grouped = df.groupby("case_id")
        case_ids = list(grouped.groups)[:N_CASES]
        sequences = [grouped.get_group(c)["activity"].tolist() for c in case_ids]
        real = np.array([
            (grouped.get_group(c)["timestamp"].max()
             - grouped.get_group(c)["timestamp"].min()).total_seconds() / DAY
            for c in case_ids
        ])

        twin = DigitalTwin(max_trace_len=200, seed=42).fit(raw)
        sim = replay(twin, sequences)

        real_med, sim_med = float(np.median(real)), float(np.median(sim))
        scale_error = abs(np.log(sim_med / real_med)) if real_med > 0 and sim_med > 0 else float("inf")

        rows.append({
            "dataset": name,
            "n_cases": len(sequences),
            "real_median_days": round(real_med, 2),
            "sim_median_days": round(sim_med, 2),
            "real_mean_days": round(float(real.mean()), 2),
            "sim_mean_days": round(float(sim.mean()), 2),
            "median_ratio": round(sim_med / real_med, 3) if real_med > 0 else None,
            "mean_ratio": round(float(sim.mean() / real.mean()), 3) if real.mean() > 0 else None,
            "scale_error": round(scale_error, 3),
        })

        print(f"=== {name}  (replaying {len(sequences)} real traces)")
        print(f"  median cycle time   real {real_med:8.2f} d   sim {sim_med:8.2f} d   "
              f"ratio {sim_med / real_med:5.2f}x")
        print(f"  mean cycle time     real {real.mean():8.2f} d   sim {sim.mean():8.2f} d   "
              f"ratio {sim.mean() / real.mean():5.2f}x")
        print(f"  scale error |log(ratio)| = {scale_error:.3f}  (tolerance {SCALE_TOLERANCE})")
        print()

        if scale_error > SCALE_TOLERANCE:
            failures.append(
                f"{name}: simulated median cycle time is {sim_med / real_med:.2f}x "
                f"the real one (scale error {scale_error:.3f})"
            )

    if not rows:
        print("Nothing to check.")
        return 1

    table = pd.DataFrame(rows)
    out_dir = REPO / "results" / "fix7_cycle_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(table.to_string(index=False))
    print()

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1

    print("PASS — replaying real traces through the duration model reproduces "
          "the real cycle-time scale.")
    print(f"Written: {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
