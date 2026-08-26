"""
fix2_terminal_labels.py
-----------------------
Verifies Fix 2: every activity the environment treats as the end of a case
must actually be an ending in the real log.

The old rule produced "good terminals" that end zero real cases and sit in the
first third of a typical trace. The new rule keeps only activities that occur
near the end of a case whenever they occur at all.

Run:
    python checks/fix2_terminal_labels.py

Writes results/fix2_terminal_labels/comparison.csv and, per dataset,
results/fix2_terminal_labels/<dataset>_markers.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from feature_engineering import (                                         # noqa: E402
    classify_terminals, terminal_diagnostics, load_terminal_overrides,
)
from timeutils import ensure_utc_timestamps, sort_events                  # noqa: E402

DATASETS = ["BPIC2012", "BPIC2015", "BPIC2017"]

# What the old rule produced, for the before/after column.
OLD_GOOD = {
    "BPIC2012": ["A_ACCEPTED", "A_APPROVED", "A_FINALIZED", "A_PREACCEPTED", "O_ACCEPTED"],
    "BPIC2015": ["close case", "enter date publication decision environmental permit",
                 "enter senddate decision environmental permit", "register deadline",
                 "set phase: phase permitting irrevocable"],
    "BPIC2017": ["A_Accepted", "A_Complete", "O_Accepted"],
}


def main() -> int:
    out_dir = REPO / "results" / "fix2_terminal_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    for name in DATASETS:
        path = REPO / "output" / name / f"events_{name}_train.parquet"
        if not path.exists():
            print(f"[skip] {name}: {path.name} not found")
            continue

        df = ensure_utc_timestamps(pd.read_parquet(path))
        diag = terminal_diagnostics(df)
        overrides = load_terminal_overrides(name)
        result = classify_terminals(df, overrides=overrides)

        good, bad = result["good_terminals"], result["bad_terminals"]
        markers = pd.DataFrame(result["diagnostics"]).set_index("activity")
        markers["label"] = ["bad" if a in bad else "good" for a in markers.index]
        markers.sort_values("case_coverage", ascending=False).to_csv(
            out_dir / f"{name}_markers.csv"
        )

        # The headline test: does a labelled ending ever actually end a case?
        old_good = OLD_GOOD.get(name, [])
        old_end = float(diag.reindex(old_good)["end_frac"].fillna(0).sum())
        new_end = float(diag.reindex(good + bad)["end_frac"].fillna(0).sum())
        worst_new_pos = float(diag.reindex(good + bad)["rel_pos_p25"].min())
        worst_old_pos = float(diag.reindex(old_good)["rel_pos_p25"].min()) if old_good else float("nan")

        # Real-log reference: of the cases that do reach a settled outcome,
        # how many end well, and after how many steps?
        ordered = sort_events(df)
        ordered = ordered.assign(step=ordered.groupby("case_id").cumcount() + 1)
        hit = ordered[ordered["activity"].isin(good + bad)].groupby("case_id").first()
        n_cases = ordered["case_id"].nunique()
        coverage = len(hit) / n_cases
        good_rate = float(hit["activity"].isin(good).mean()) if len(hit) else float("nan")
        median_steps = float(hit["step"].median()) if len(hit) else float("nan")
        median_len = float(ordered.groupby("case_id").size().median())

        rows.append({
            "dataset": name,
            "old_n_good": len(old_good),
            "old_traces_ending_at_a_good_terminal": round(old_end, 4),
            "old_earliest_good_terminal_pos_p25": round(worst_old_pos, 3),
            "new_n_good": len(good),
            "new_n_bad": len(bad),
            "new_n_excluded": len(result["excluded_markers"]),
            "manual_labels": len(overrides),
            "new_earliest_terminal_pos_p25": round(worst_new_pos, 3),
            "real_cases_reaching_an_outcome": round(coverage, 4),
            "real_good_rate_of_those": round(good_rate, 4),
            "real_median_steps_to_outcome": median_steps,
            "median_trace_length": median_len,
        })

        print(f"=== {name}" + ("  [manual labels applied]" if overrides else ""))
        print(f"  good ({len(good)}): {good if len(good) <= 8 else good[:8] + ['...']}")
        print(f"  bad  ({len(bad)}): {bad if len(bad) <= 8 else bad[:8] + ['...']}")
        if result["excluded_markers"]:
            print(f"  excluded as outcome-neutral: {len(result['excluded_markers'])}")
        if result["unknown_overrides"]:
            print(f"  WARNING unknown override names: {result['unknown_overrides']}")
        print(f"  earliest any terminal appears (p25 of position): {worst_new_pos:.3f}")
        print(f"  real log: {coverage:.1%} of cases reach an outcome; "
              f"of those {good_rate:.1%} good / {1 - good_rate:.1%} bad; "
              f"median {median_steps:.0f} steps (trace median {median_len:.0f})")

        if worst_new_pos < 0.80:
            failures.append(f"{name}: a terminal sits at position {worst_new_pos:.3f}")
        print()

    if not rows:
        print("No datasets found — run notebook 01 first.")
        return 1

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "comparison.csv", index=False)
    print(table.to_string(index=False))
    print()

    if failures:
        print("FAIL:")
        for f in failures:
            print("  -", f)
        return 1

    print("PASS — every labelled terminal occurs near the end of a real case.")
    print(f"Written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
