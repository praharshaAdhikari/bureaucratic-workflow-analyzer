"""
fix1_transition_integrity.py
----------------------------
Verifies Fix 1: the digital twin's transition graph must contain only
step-to-step transitions that actually occur in the training log.

For each dataset it reports:
  nat_old        events the OLD parsing turned into NaT (the bug)
  nat_new        events the NEW parsing turns into NaT (must be 0)
  real_edges     directly-follows pairs in the real log
  twin_edges     directly-follows pairs the fitted twin believes in
  fabricated     twin edges with no counterpart in the log (must be 0)

Run:
    python checks/fix1_transition_integrity.py

Writes results/fix1_transition_integrity/comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from digital_twin import DigitalTwin          # noqa: E402
from timeutils import ensure_utc_timestamps, sort_events   # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]


def directly_follows(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Ordered activity pairs observed within each case."""
    pairs: set[tuple[str, str]] = set()
    for _, grp in sort_events(df).groupby("case_id"):
        acts = grp["activity"].tolist()
        pairs.update(zip(acts[:-1], acts[1:]))
    return pairs


def count_nat_old_parser(df: pd.DataFrame) -> int:
    """How many timestamps the pre-fix astype(str) round-trip destroyed."""
    reparsed = pd.to_datetime(df["timestamp"].astype(str), utc=True, errors="coerce")
    return int(reparsed.isna().sum())


def main() -> int:
    rows = []
    for name in DATASETS:
        path = REPO / "output" / name / f"events_{name}_train.parquet"
        if not path.exists():
            print(f"[skip] {name}: {path.name} not found")
            continue

        raw = pd.read_parquet(path)
        clean = ensure_utc_timestamps(raw)          # raises if anything fails

        real = directly_follows(clean)

        twin = DigitalTwin(max_trace_len=200, seed=42).fit(raw)
        twin_edges = {(a, b) for a, succ in twin.transition_probs.items() for b in succ}

        fabricated = twin_edges - real
        rows.append({
            "dataset":     name,
            "events":      len(raw),
            "cases":       raw["case_id"].nunique(),
            "activities":  raw["activity"].nunique(),
            "nat_old":     count_nat_old_parser(raw),
            "nat_new":     int(clean["timestamp"].isna().sum()),
            "real_edges":  len(real),
            "twin_edges":  len(twin_edges),
            "fabricated":  len(fabricated),
            "fabricated_pct": round(100 * len(fabricated) / max(len(twin_edges), 1), 2),
        })

    if not rows:
        print("No datasets found — run notebook 01 first.")
        return 1

    table = pd.DataFrame(rows)
    out_dir = REPO / "results" / "fix1_transition_integrity"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison.csv", index=False)

    print(table.to_string(index=False))
    print()

    failed = table[table["fabricated"] > 0]
    if not failed.empty:
        print("FAIL — fabricated transitions remain:")
        print(failed[["dataset", "fabricated"]].to_string(index=False))
        return 1

    print("PASS — every transition the twin knows about occurs in the real log.")
    print(f"Written: {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
