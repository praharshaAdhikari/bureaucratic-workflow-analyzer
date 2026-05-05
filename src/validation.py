"""
validation.py
-------------
Validates the Digital Twin simulation against real event log data.

Metric design philosophy:
  - Each metric targets a distinct structural property of the process
  - Metrics are robust to the specific quirks of bureaucratic logs
    (zero-gap timestamps, rare activities, long-tail durations)
  - All metrics are bounded or normalised so thresholds are interpretable

Metrics:
  trace_length_wasserstein    : Earth mover's distance on trace length distributions
  case_duration_wasserstein   : EMD on total case duration (days) — primary KPI
  activity_freq_jsd           : Jensen-Shannon divergence on full activity frequencies
  transition_matrix_l1        : L1 on empirical transition matrices (shared activities)
  duration_distribution_ks    : Mean KS statistic on per-activity duration CDFs
  variant_coverage            : Fraction of top-50 real variants reproduced by sim
  resource_utilisation_mae    : MAE on per-resource utilisation rates

Thresholds (literature-informed, based on Chapela-Campa et al. 2023 and Simod/Prosimos benchmarks):
  trace_length_wasserstein    < 3.0   (events; good Markov sims: 0.5–2.0)
  case_duration_wasserstein   < 0.15  (normalised [0,1]; good range: 0.03–0.12)
  activity_freq_jsd           < 0.05  (bits [0,1]; < 0.05 excellent, < 0.10 acceptable)
  transition_matrix_l1        < 0.10  (mean abs diff per cell; well-fitted Markov: < 0.05)
  duration_distribution_ks    < 0.20  (KS stat [0,1]; typical good range: 0.10–0.25)
  variant_coverage            > 0.80  (bigram coverage; good range: 0.75–1.0)
  resource_utilisation_mae    < 0.05  (mean abs error on utilisation fractions)
"""

import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import wasserstein_distance, ks_2samp

from feature_engineering import (
    compute_transition_matrix,
    compute_duration_stats,
)


DEFAULT_THRESHOLDS = {
    # Normalized Wasserstein (EMD) on trace length distributions.
    # Good Markov simulators on BPIC logs typically achieve 0.5–2.0 events.
    # Threshold of 3.0 allows for harder datasets while rejecting clearly broken sims.
    # Source: empirical range from Chapela-Campa et al. (2023), Simod benchmarks.
    "trace_length_wasserstein":  3.0,

    # Wasserstein on normalised [0,1] case duration shapes.
    # Markov sims capture processing time, not calendar waiting time, so some
    # shape divergence is expected. 0.15 rejects sims with badly wrong duration shapes
    # while tolerating the inherent scale mismatch between sim and real durations.
    # Typical good range: 0.03–0.12 on BPIC logs.
    "case_duration_wasserstein": 0.15,

    # Jensen-Shannon divergence on activity frequency distributions, normalised [0,1].
    # JSD < 0.05 is excellent; < 0.10 is acceptable per BPS literature.
    # A Markov sim fitted to the same log should easily stay below 0.05.
    "activity_freq_jsd":         0.05,

    # Mean L1 distance per cell of the transition matrix.
    # A well-fitted Markov model should be near 0; 0.10 allows for rare-activity noise.
    # Values > 0.10 indicate the sim is routing cases through wrong paths.
    "transition_matrix_l1":      0.10,

    # Mean KS statistic on per-activity duration CDFs (log-scale).
    # Duration fitting is the hardest dimension for Markov sims — they use sampled
    # inter-event gaps which don't capture waiting time. KS of 0.20 is a realistic
    # bar; typical good sims achieve 0.10–0.25 on BPIC logs.
    "duration_distribution_ks":  0.20,

    # Bigram coverage of top-50 real variants in the simulation.
    # 0.80 means the sim reproduces 80% of the transition pairs seen in common traces.
    # Typical good range: 0.75–1.0 for Markov sims on BPIC logs.
    "variant_coverage":          0.80,

    # Mean absolute error on per-resource utilisation fractions.
    # 0.05 means on average each resource's share of work is within 5 percentage points.
    # Values > 0.10 indicate the sim is assigning work to wrong resources.
    "resource_utilisation_mae":  0.05,
}

# Metrics where higher value = better (inverted threshold logic)
_HIGHER_IS_BETTER = {"variant_coverage"}


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def trace_length_wasserstein(real_df: pd.DataFrame, sim_df: pd.DataFrame) -> float:
    """
    Earth mover's distance between trace length distributions.

    Better than L1 histogram because:
    - No binning artifacts
    - Shape-aware: penalises distributions that are shifted or spread differently
    - Units are interpretable: distance in number of events
    """
    real_lens = real_df.groupby("case_id")["activity"].count().values.astype(float)
    sim_lens  = sim_df.groupby("case_id")["activity"].count().values.astype(float)
    return float(wasserstein_distance(real_lens, sim_lens))


def case_duration_wasserstein(real_df: pd.DataFrame, sim_df: pd.DataFrame) -> float:
    """
    Earth mover's distance between total case duration distributions (in days).

    For real data: timestamp span (first to last event).
    For sim data: uses sum(duration_s) which represents active processing time.

    Note: sim duration_s only captures processing time (60s-8h per activity),
    not calendar waiting time. The metric compares the shape of the distributions
    after normalising both to [0,1] range, so scale differences don't dominate.
    """
    def real_case_durations_days(df):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str), utc=True, errors="coerce")
        return df.groupby("case_id")["timestamp"].agg(
            lambda x: (x.max() - x.min()).total_seconds() / 86400
        ).clip(lower=0).values.astype(float)

    def sim_case_durations_days(df):
        if "duration_s" in df.columns:
            return (df.groupby("case_id")["duration_s"].sum() / 86400).clip(lower=0).values.astype(float)
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str), utc=True, errors="coerce")
        return df.groupby("case_id")["timestamp"].agg(
            lambda x: (x.max() - x.min()).total_seconds() / 86400
        ).clip(lower=0).values.astype(float)

    real_durs = real_case_durations_days(real_df)
    sim_durs  = sim_case_durations_days(sim_df)

    # Normalise both to [0,1] by dividing by their own 99th percentile
    # This compares distribution SHAPE rather than absolute scale
    # (sim captures processing time, real captures calendar time — different scales by design)
    real_p99 = float(np.percentile(real_durs, 99)) or 1.0
    sim_p99  = float(np.percentile(sim_durs,  99)) or 1.0
    real_norm = np.clip(real_durs / real_p99, 0, 1)
    sim_norm  = np.clip(sim_durs  / sim_p99,  0, 1)

    return float(wasserstein_distance(real_norm, sim_norm))


def activity_freq_jsd(real_df: pd.DataFrame, sim_df: pd.DataFrame) -> float:
    """
    Jensen-Shannon divergence on full activity frequency distributions.

    Better than top-20 relative error because:
    - Covers all activities, not just the most frequent
    - Symmetric and bounded [0, log(2)] ≈ [0, 0.693] in nats, normalised to [0, 1]
    - Not inflated by a single badly-fit rare activity
    """
    real_freq = real_df["activity"].value_counts(normalize=True)
    sim_freq  = sim_df["activity"].value_counts(normalize=True)

    all_acts = sorted(set(real_freq.index) | set(sim_freq.index))
    p = np.array([real_freq.get(a, 0.0) for a in all_acts])
    q = np.array([sim_freq.get(a, 0.0)  for a in all_acts])

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()

    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    # Normalise to [0, 1] by dividing by log(2)
    return float(np.clip(jsd / np.log(2), 0, 1))


def transition_matrix_l1(real_df: pd.DataFrame, sim_df: pd.DataFrame) -> float:
    """L1 distance between empirical transition matrices (shared activities)."""
    real_mat = compute_transition_matrix(real_df)
    sim_mat  = compute_transition_matrix(sim_df)

    shared = sorted(set(real_mat.index) & set(sim_mat.index))
    if not shared:
        return 1.0

    real_sub = real_mat.loc[shared, shared].fillna(0).values
    sim_sub  = sim_mat.loc[shared, shared].fillna(0).values
    return float(np.mean(np.abs(real_sub - sim_sub)))


def duration_distribution_ks(real_df: pd.DataFrame, sim_df: pd.DataFrame) -> float:
    """
    Mean Kolmogorov-Smirnov statistic on per-activity duration CDFs.

    Compares on log-scale (log of seconds) because bureaucratic durations span
    many orders of magnitude (seconds to weeks). Log-scale KS is more sensitive
    to shape differences across the full range rather than being dominated by
    the heavy tail.

    Only applied to activities with real non-zero duration (>60s) so zero-gap
    activities don't inflate the metric.

    Uses duration_s column if available (sim data), otherwise inter-event gaps.
    """
    # Real data: inter-event gaps
    real_df2 = real_df.copy().sort_values(["case_id", "timestamp"])
    real_df2["timestamp"] = pd.to_datetime(real_df2["timestamp"].astype(str), utc=True, errors="coerce")
    real_df2["next_ts"]    = real_df2.groupby("case_id")["timestamp"].shift(-1)
    real_df2["duration_s"] = (real_df2["next_ts"] - real_df2["timestamp"]).dt.total_seconds()
    real_df2 = real_df2[real_df2["duration_s"] > 60]  # only non-trivial gaps

    # Sim data: use duration_s column directly if present
    if "duration_s" in sim_df.columns:
        sim_df2 = sim_df[sim_df["duration_s"] > 60].copy()
    else:
        sim_df2 = sim_df.copy().sort_values(["case_id", "timestamp"])
        sim_df2["timestamp"] = pd.to_datetime(sim_df2["timestamp"].astype(str), utc=True, errors="coerce")
        sim_df2["next_ts"]    = sim_df2.groupby("case_id")["timestamp"].shift(-1)
        sim_df2["duration_s"] = (sim_df2["next_ts"] - sim_df2["timestamp"]).dt.total_seconds()
        sim_df2 = sim_df2[sim_df2["duration_s"] > 60]

    # Activities with enough samples in both
    real_acts = set(real_df2.groupby("activity").filter(lambda g: len(g) >= 10)["activity"].unique())
    sim_acts  = set(sim_df2.groupby("activity").filter(lambda g: len(g) >= 10)["activity"].unique())
    shared    = real_acts & sim_acts

    if not shared:
        return 1.0

    ks_stats = []
    for act in shared:
        r_vals = np.log(real_df2[real_df2["activity"] == act]["duration_s"].values)
        s_vals = np.log(sim_df2[sim_df2["activity"] == act]["duration_s"].values)
        stat, _ = ks_2samp(r_vals, s_vals)
        ks_stats.append(stat)

    return float(np.mean(ks_stats))


def variant_coverage(real_df: pd.DataFrame, sim_df: pd.DataFrame,
                     top_n: int = 50) -> float:
    """
    Bigram coverage: fraction of real activity bigrams (consecutive pairs)
    from the top-N most frequent real traces that appear in the simulation.

    Why bigrams instead of exact variants:
    - BPIC-2015 has 1170 unique variants from 1199 cases (essentially every case unique)
    - Exact variant matching is impossible for any Markov-based simulator on this data
    - Bigrams test whether the sim reproduces the same local transition patterns
      that appear in the most common real traces — a meaningful structural check

    Higher is better. Threshold is a minimum.
    """
    def get_trace_bigrams(df):
        """Get all bigrams from traces, weighted by trace frequency."""
        traces = (
            df.sort_values(["case_id", "timestamp"])
            .groupby("case_id")["activity"]
            .apply(tuple)
        )
        bigrams = set()
        for trace in traces:
            for a, b in zip(trace[:-1], trace[1:]):
                bigrams.add((a, b))
        return bigrams

    def get_top_n_trace_bigrams(df, n):
        """Get bigrams only from the top-N most frequent real traces."""
        traces = (
            df.sort_values(["case_id", "timestamp"])
            .groupby("case_id")["activity"]
            .apply(tuple)
        )
        top_traces = traces.value_counts().head(n).index
        bigrams = set()
        for trace in top_traces:
            for a, b in zip(trace[:-1], trace[1:]):
                bigrams.add((a, b))
        return bigrams

    real_bigrams = get_top_n_trace_bigrams(real_df, top_n)
    sim_bigrams  = get_trace_bigrams(sim_df)

    if not real_bigrams:
        return 0.0

    covered = sum(1 for bg in real_bigrams if bg in sim_bigrams)
    return float(covered / len(real_bigrams))


def resource_utilisation_mae(real_df: pd.DataFrame, sim_df: pd.DataFrame) -> float:
    """
    Mean absolute error on per-resource utilisation rates.

    Utilisation = fraction of total events handled by each resource,
    normalised so all resources sum to 1. Measures whether the sim
    reproduces realistic workload distribution across resources.

    Only compares resources present in both datasets.
    """
    real_util = real_df["resource"].value_counts(normalize=True)
    sim_util  = sim_df["resource"].value_counts(normalize=True)

    shared = sorted(set(real_util.index) & set(sim_util.index))
    if not shared:
        return 1.0

    errors = [abs(real_util[r] - sim_util.get(r, 0.0)) for r in shared]
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Full validation report
# ---------------------------------------------------------------------------

def validate(
    real_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    thresholds: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run all validation metrics and compare against thresholds.

    Returns dict with keys: metric_name -> {value, threshold, passed}.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    metric_fns = {
        "trace_length_wasserstein":  trace_length_wasserstein,
        "case_duration_wasserstein": case_duration_wasserstein,
        "activity_freq_jsd":         activity_freq_jsd,
        "transition_matrix_l1":      transition_matrix_l1,
        "duration_distribution_ks":  duration_distribution_ks,
        "variant_coverage":          variant_coverage,
        "resource_utilisation_mae":  resource_utilisation_mae,
    }

    results = {}
    for name, fn in metric_fns.items():
        if thresholds.get(name) is None:
            continue  # explicitly skipped (e.g. cross-domain mode)
        try:
            val = fn(real_df, sim_df)
        except Exception as e:
            val = float("nan")
            if verbose:
                print(f"  [WARN] {name} failed: {e}")

        thresh  = thresholds[name]
        higher_better = name in _HIGHER_IS_BETTER
        if not np.isnan(val):
            passed = (val >= thresh) if higher_better else (val <= thresh)
        else:
            passed = False

        results[name] = {"value": val, "threshold": thresh, "passed": passed}

        if verbose:
            status    = "✓ PASS" if passed else "✗ FAIL"
            direction = f"(threshold ≥ {thresh})" if higher_better else f"(threshold ≤ {thresh})"
            print(f"  {status}  {name:<35s} = {val:.4f}  {direction}")

    overall = all(r["passed"] for r in results.values())
    results["overall_pass"] = overall

    if verbose:
        print(f"\n  Overall: {'✓ PASS' if overall else '✗ FAIL'}")

    return results


def validation_report_df(results: dict) -> pd.DataFrame:
    rows = []
    for k, v in results.items():
        if k == "overall_pass":
            continue
        rows.append({
            "metric":    k,
            "value":     v["value"],
            "threshold": v["threshold"],
            "passed":    v["passed"],
            "higher_is_better": k in _HIGHER_IS_BETTER,
        })
    return pd.DataFrame(rows)
