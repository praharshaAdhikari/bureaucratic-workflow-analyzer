"""
inspect_artefacts.py
--------------------
Prints kpi_baselines and terminal classification cosine similarities
for every dataset found in the output/ directory.

Usage:
    python inspect_artefacts.py
    python inspect_artefacts.py --dataset BPIC2012
    python inspect_artefacts.py --threshold 0.4   # change cosine threshold
"""

import sys
import json
import argparse
import re
from pathlib import Path

# ── Repo root detection ───────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here / "src"))

import joblib
import numpy as np

OUTPUT_ROOT = _here / "output"


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inspect KPI baselines and terminal cosine similarities.")
parser.add_argument("--dataset",   default=None,  help="Single dataset name (default: all)")
parser.add_argument("--threshold", type=float, default=0.35, help="Cosine similarity threshold (default: 0.35)")
args = parser.parse_args()


# ── Discover datasets ─────────────────────────────────────────────────────────
if args.dataset:
    dataset_dirs = [OUTPUT_ROOT / args.dataset]
else:
    dataset_dirs = sorted(
        d for d in OUTPUT_ROOT.iterdir()
        if d.is_dir() and (d / f"digital_twin_{d.name}_train.pkl").exists()
    )

if not dataset_dirs:
    print(f"No datasets found in {OUTPUT_ROOT}")
    sys.exit(1)


# ── Cosine similarity helper ──────────────────────────────────────────────────
_sentence_model = None

def _get_cosine_scores(activities: list[str], threshold: float) -> list[tuple[str, float, bool]]:
    """
    Returns list of (activity, cosine_similarity, is_bad_terminal) sorted by score desc.
    Falls back to keyword matching with score=1.0/0.0 if sentence-transformers unavailable.
    """
    global _sentence_model

    fallback_keywords = ["declin", "denied", "deny", "cancel", "cancelled",
                         "refus", "refused", "reject"]

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        if _sentence_model is None:
            print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        def _clean(name: str) -> str:
            name = re.sub(r"^[A-Z]_", "", name)
            return name.replace("_", " ").strip()

        clean = [_clean(a) for a in activities]
        vecs  = _sentence_model.encode(clean, normalize_embeddings=True, show_progress_bar=False)
        anchor = _sentence_model.encode(
            ["application declined denied refused cancelled rejected"],
            normalize_embeddings=True, show_progress_bar=False,
        )[0]
        sims = cosine_similarity(vecs, [anchor]).flatten()

        return sorted(
            [(act, float(sim), float(sim) >= threshold) for act, sim in zip(activities, sims)],
            key=lambda x: -x[1],
        )

    except ImportError:
        # Keyword fallback — assign 1.0 for keyword match, 0.0 otherwise
        results = []
        for act in activities:
            is_bad = any(k in act.lower() for k in fallback_keywords)
            results.append((act, 1.0 if is_bad else 0.0, is_bad))
        return sorted(results, key=lambda x: -x[1])


# ── Main loop ─────────────────────────────────────────────────────────────────
SEP = "─" * 72

for ds_dir in dataset_dirs:
    ds_name   = ds_dir.name
    twin_path = ds_dir / f"digital_twin_{ds_name}_train.pkl"
    tc_path   = ds_dir / "terminal_classification.json"

    print()
    print(SEP)
    print(f"  DATASET: {ds_name}")
    print(SEP)

    # ── KPI baselines ─────────────────────────────────────────────────────────
    print()
    print("  KPI BASELINES  (from twin.kpi_baselines, fitted in notebook 03)")
    print()

    twin = joblib.load(twin_path)
    baselines = twin.kpi_baselines

    labels = {
        "median_trace_length":  ("Median trace length",  "activities/case"),
        "p95_trace_length":     ("P95 trace length",     "activities/case"),
        "median_case_age_days": ("Median case age",      "days"),
        "mean_rework":          ("Mean rework count",    "duplicate activities/case"),
        "completion_rate":      ("Completion rate",      "fraction of cases"),
    }
    for key, (label, unit) in labels.items():
        val = baselines.get(key, "N/A")
        if isinstance(val, float):
            print(f"    {label:<28}  {val:>10.4f}  ({unit})")
        else:
            print(f"    {label:<28}  {val!r}")

    # Derived values used by ProcessEnv
    median_len  = max(baselines.get("median_trace_length", 20), 1)
    mean_rework = max(baselines.get("mean_rework", 1.0), 0.1)
    baseline_loop_rate = mean_rework / median_len
    print()
    print(f"  Derived (used by ProcessEnv):")
    print(f"    baseline_loop_rate  =  mean_rework / median_len")
    print(f"                        =  {mean_rework:.4f} / {median_len:.1f}  =  {baseline_loop_rate:.4f}")

    # ── Terminal classification ───────────────────────────────────────────────
    print()
    print(f"  TERMINAL CLASSIFICATION  (cosine threshold = {args.threshold})")
    print(f"  Source: classify_bad_terminals() in feature_engineering.py")
    print()

    # Load saved bad_terminals from JSON if it exists
    saved_bad: list = []
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        saved_bad = tc.get("bad_terminals", [])
        print(f"  Saved in {tc_path.name}:")
        print(f"    bad_terminals = {saved_bad}")
        print()

    # Recompute cosine scores live
    activities = sorted(twin.activities)
    scores = _get_cosine_scores(activities, args.threshold)

    print(f"  Live cosine scores (all {len(activities)} activities):")
    print(f"  {'Activity':<40}  {'Cosine sim':>10}  {'≥ threshold':>11}  {'In saved JSON':>13}")
    print(f"  {'─'*40}  {'─'*10}  {'─'*11}  {'─'*13}")
    for act, sim, is_bad in scores:
        in_saved = "yes" if act in saved_bad else "no"
        flag     = "BAD" if is_bad else ""
        print(f"  {act:<40}  {sim:>10.4f}  {flag:>11}  {in_saved:>13}")

    # Highlight mismatches between saved JSON and live threshold
    live_bad  = {act for act, _, is_bad in scores if is_bad}
    saved_set = set(saved_bad)
    only_live  = live_bad - saved_set
    only_saved = saved_set - live_bad
    if only_live or only_saved:
        print()
        print(f"  ⚠  Mismatch between saved JSON and live threshold={args.threshold}:")
        if only_live:
            print(f"     In live but not saved:  {sorted(only_live)}")
        if only_saved:
            print(f"     In saved but not live:  {sorted(only_saved)}")
    else:
        print()
        print(f"  ✓  Saved JSON matches live threshold={args.threshold}")

print()
print(SEP)
print("Done.")
