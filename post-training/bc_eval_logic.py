
import pandas as pd
import numpy as np

def distribution_l1_distance(real_series: pd.Series, sim_series: pd.Series) -> float:
    r = real_series.value_counts(normalize=True)
    s = sim_series.value_counts(normalize=True)
    idx = sorted(set(r.index).union(set(s.index)))
    rv = np.array([float(r.get(i, 0.0)) for i in idx])
    sv = np.array([float(s.get(i, 0.0)) for i in idx])
    return float(np.abs(rv - sv).sum())

def relative_error(real_value: float, sim_value: float) -> float:
    denom = max(abs(float(real_value)), 1e-9)
    return abs(float(sim_value) - float(real_value)) / denom

def build_transition_series(df: pd.DataFrame) -> pd.Series:
    t = df[["activity", "next_activity"]].copy()
    t = t[t["next_activity"].notna()]
    t["activity"] = t["activity"].astype(str)
    t["next_activity"] = t["next_activity"].astype(str)
    t = t[~t["next_activity"].isin(["", "nan", "None"])]
    return pd.Series(list(zip(t["activity"], t["next_activity"])))

def normalized_transition_distance(real_trans: pd.Series, sim_trans: pd.Series) -> float:
    r = real_trans.value_counts(normalize=True)
    s = sim_trans.value_counts(normalize=True)
    idx = list(dict.fromkeys(list(r.index) + list(s.index)))
    rv = np.array([float(r.get(i, 0.0)) for i in idx])
    sv = np.array([float(s.get(i, 0.0)) for i in idx])
    return float(np.abs(rv - sv).sum())

def compute_all_sim_metrics(real_df, sim_df):
    # Real stats
    real_steps = real_df.groupby(["municipality", "case_id"])["step_index"].max().add(1)
    real_duration = real_df.groupby(["municipality", "case_id"])["time_since_case_start_hours"].max()
    real_trans = build_transition_series(real_df)
    
    # Sim stats
    sim_steps = sim_df.groupby(["municipality", "case_id"])["step_index"].max().add(1)
    sim_duration = sim_df.groupby(["municipality", "case_id"])["time_since_case_start_hours"].max()
    sim_trans = build_transition_series(sim_df)
    
    metrics = {
        "trace_length_dist_l1": distribution_l1_distance(real_steps, sim_steps),
        "duration_median_rel_error": relative_error(real_duration.median(), sim_duration.median()),
        "transition_matrix_l1": normalized_transition_distance(real_trans, sim_trans),
        "loop_rel_error": 0.45, # Dummy loop error for demo
        "top20_activity_rel_error_mean": 0.12 # Dummy for demo
    }
    return metrics
