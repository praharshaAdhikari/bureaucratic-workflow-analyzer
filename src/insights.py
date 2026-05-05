"""
insights.py
-----------
Backtrack RL routing decisions to produce human-readable workflow recommendations.

C2 routing env: actions = MultiDiscrete([routing_idx, mgmt_idx]).

Three analysis layers:
  1. Routing preference table — compares RL vs random transition frequencies
     with smoothed ratios and impossible-transition flagging.
  2. Decision tree extraction — converts the policy into if-then rules
     readable by business stakeholders.
  3. Recommendation generation — translates routing patterns into actionable
     process improvement rules.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Optional
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectories(
    env,
    model,
    n_episodes: int = 200,
    deterministic: bool = True,
) -> list[dict]:
    """
    Run policy for n_episodes and collect full routing trajectories.

    Handles MultiDiscrete([max_succ, n_mgmt]) and legacy Discrete(max_succ).

    Each episode dict contains:
        episode, steps, total_reward, n_steps, terminated,
        final_activity, hit_bad_terminal, mgmt_action_counts,
        kpi_snapshots  — list of kpi_signals dicts per step (for decision tree)
    """
    import gymnasium as gym
    _is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)

    try:
        from kpi_actions import MANAGEMENT_ACTIONS
        _mgmt_names = {a.index: a.name for a in MANAGEMENT_ACTIONS}
    except ImportError:
        _mgmt_names = {}

    trajectories = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        steps = []
        total_reward = 0.0
        mgmt_counts: Counter = Counter()

        while not (done or truncated):
            from_act = env._current_activity
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)

            if _is_multidiscrete:
                action = np.asarray(action)
                routing_idx = int(action[0])
                mgmt_idx    = int(action[1]) if len(action) > 1 else 0
            else:
                routing_idx = int(action)
                mgmt_idx    = 0

            successors = env._successors.get(from_act, [])
            to_act    = successors[routing_idx] if routing_idx < len(successors) else "UNKNOWN"
            mgmt_name = _mgmt_names.get(mgmt_idx, str(mgmt_idx))

            # Snapshot KPI signals before stepping (for decision tree)
            kpi_vec = obs.get("kpi_signals", np.zeros(7)) if isinstance(obs, dict) else np.zeros(7)

            obs, reward, done, truncated, info = env.step(
                np.array([routing_idx, mgmt_idx]) if _is_multidiscrete else routing_idx
            )
            total_reward += reward
            mgmt_counts[mgmt_name] += 1

            steps.append({
                "from_activity":    from_act,
                "to_activity":      to_act,
                "routing_action":   routing_idx,
                "mgmt_action":      mgmt_idx,
                "mgmt_action_name": mgmt_name,
                "reward":           reward,
                "trace_len":        info["trace_len"],
                "is_bad_terminal":  to_act in env._bad_terminals,
                "kpi_delay":        float(kpi_vec[0]),
                "kpi_rework":       float(kpi_vec[1]),
                "kpi_loop_rate":    float(kpi_vec[2]),
                "kpi_case_age":     float(kpi_vec[3]),
                "kpi_term_prox":    float(kpi_vec[4]),
                "kpi_volume":       float(kpi_vec[5]),
            })

        trajectories.append({
            "episode":            ep,
            "steps":              steps,
            "total_reward":       total_reward,
            "n_steps":            len(steps),
            "terminated":         bool(done),
            "final_activity":     env._current_activity,
            "hit_bad_terminal":   env._current_activity in env._bad_terminals,
            "mgmt_action_counts": dict(mgmt_counts),
        })

    return trajectories


# ---------------------------------------------------------------------------
# Routing preference table — with smoothing and impossible-transition flags
# ---------------------------------------------------------------------------

def routing_preference_table(
    rl_trajectories: list[dict],
    random_trajectories: list[dict],
    real_log_df: Optional[pd.DataFrame] = None,
    smoothing_alpha: float = 0.5,
) -> pd.DataFrame:
    """
    For each (from_activity, to_activity) pair, compare RL vs random frequency.

    Improvements over naive ratio:
    - Laplace smoothing (alpha=0.5) prevents division-by-zero and caps extreme
      ratios caused by small sample sizes.
    - Cross-references against the real event log to classify avoided routes as
      either "impossible in twin" (never in real log either) or "learned to avoid"
      (exists in real log but agent avoids it).

    Parameters
    ----------
    real_log_df : pd.DataFrame, optional
        Event log with columns [case_id, activity, timestamp]. When provided,
        each row is annotated with transition_type:
          'impossible'    — not in real log transitions (twin artefact)
          'learned_avoid' — in real log but RL avoids it
          'learned_prefer'— in real log and RL prefers it
          'neutral'       — no strong preference either way

    smoothing_alpha : float
        Laplace smoothing pseudocount added to both RL and random counts
        before computing the ratio. Default 0.5 (Jeffreys prior).

    Returns DataFrame with columns:
        from_activity, to_activity, rl_count, random_count,
        rl_pct, random_pct, preference_ratio, log2_ratio,
        avg_reward_rl, in_real_log, transition_type
    """
    def count_transitions(trajs):
        counts: Counter = Counter()
        rewards: dict = defaultdict(list)
        for traj in trajs:
            for step in traj["steps"]:
                key = (step["from_activity"], step["to_activity"])
                counts[key] += 1
                rewards[key].append(step["reward"])
        return counts, rewards

    rl_counts, rl_rewards = count_transitions(rl_trajectories)
    rnd_counts, _         = count_transitions(random_trajectories)

    # Build real-log transition set for cross-referencing
    real_transitions: set = set()
    if real_log_df is not None:
        _df = real_log_df.copy()
        _df["timestamp"] = pd.to_datetime(_df["timestamp"], utc=True, errors="coerce")
        for _, grp in _df.groupby("case_id"):
            acts = grp.sort_values("timestamp")["activity"].tolist()
            for a, b in zip(acts[:-1], acts[1:]):
                real_transitions.add((a, b))

    all_keys = set(rl_counts) | set(rnd_counts)

    rows = []
    for (frm, to) in sorted(all_keys):
        rl_c   = rl_counts.get((frm, to), 0)
        rnd_c  = rnd_counts.get((frm, to), 0)

        # Total steps from this activity (denominator for pct)
        rl_total  = sum(v for (f, _), v in rl_counts.items()  if f == frm) or 1
        rnd_total = sum(v for (f, _), v in rnd_counts.items() if f == frm) or 1

        rl_pct  = rl_c  / rl_total  * 100
        rnd_pct = rnd_c / rnd_total * 100

        # Smoothed ratio: (rl_c + alpha) / (rnd_c + alpha) normalised by totals
        # This prevents 0/0 and caps extreme values from tiny samples
        rl_smooth  = (rl_c  + smoothing_alpha) / (rl_total  + smoothing_alpha)
        rnd_smooth = (rnd_c + smoothing_alpha) / (rnd_total + smoothing_alpha)
        ratio      = rl_smooth / rnd_smooth
        log2_ratio = float(np.log2(ratio))

        avg_rew = float(np.mean(rl_rewards[(frm, to)])) if rl_rewards[(frm, to)] else 0.0

        # Cross-reference with real log
        in_real = (frm, to) in real_transitions if real_transitions else None

        # Classify transition type
        if in_real is None:
            t_type = "unknown"
        elif not in_real:
            t_type = "impossible"          # not in real log — twin artefact
        elif log2_ratio < -1.0:
            t_type = "learned_avoid"       # in real log, RL avoids it
        elif log2_ratio > 1.0:
            t_type = "learned_prefer"      # in real log, RL prefers it
        else:
            t_type = "neutral"

        rows.append({
            "from_activity":    frm,
            "to_activity":      to,
            "rl_count":         rl_c,
            "random_count":     rnd_c,
            "rl_pct":           round(rl_pct,  1),
            "random_pct":       round(rnd_pct, 1),
            "preference_ratio": round(ratio,   3),
            "log2_ratio":       round(log2_ratio, 3),
            "avg_reward_rl":    round(avg_rew, 3),
            "in_real_log":      in_real,
            "transition_type":  t_type,
        })

    return pd.DataFrame(rows).sort_values("log2_ratio", ascending=False)


# ---------------------------------------------------------------------------
# Management action summary
# ---------------------------------------------------------------------------

def management_action_summary(
    rl_trajectories: list[dict],
    random_trajectories: list[dict],
) -> pd.DataFrame:
    """
    Compare management action usage rates between RL and random policies.

    Returns DataFrame sorted by rl_rate descending.
    """
    def action_rates(trajs):
        total_steps = sum(t["n_steps"] for t in trajs)
        counts: Counter = Counter()
        for t in trajs:
            for name, cnt in t["mgmt_action_counts"].items():
                counts[name] += cnt
        return {name: cnt / max(total_steps, 1) for name, cnt in counts.items()}

    rl_rates  = action_rates(rl_trajectories)
    rnd_rates = action_rates(random_trajectories)
    all_names = sorted(set(rl_rates) | set(rnd_rates))

    rows = []
    for name in all_names:
        rl_r  = rl_rates.get(name, 0.0)
        rnd_r = rnd_rates.get(name, 0.0)
        rows.append({
            "mgmt_action": name,
            "rl_rate":     round(rl_r,  4),
            "random_rate": round(rnd_r, 4),
            "ratio":       round(rl_r / max(rnd_r, 1e-6), 2),
        })

    return pd.DataFrame(rows).sort_values("rl_rate", ascending=False)


# ---------------------------------------------------------------------------
# Episode outcome summary
# ---------------------------------------------------------------------------

def episode_outcome_summary(trajectories: list[dict]) -> dict:
    n = len(trajectories)
    return {
        "n_episodes":        n,
        "mean_reward":       round(float(np.mean([t["total_reward"] for t in trajectories])), 3),
        "std_reward":        round(float(np.std([t["total_reward"]  for t in trajectories])), 3),
        "mean_length":       round(float(np.mean([t["n_steps"]      for t in trajectories])), 1),
        "terminal_rate":     round(sum(t["terminated"]        for t in trajectories) / n, 3),
        "bad_terminal_rate": round(sum(t["hit_bad_terminal"]  for t in trajectories) / n, 3),
    }


# ---------------------------------------------------------------------------
# Decision tree extraction
# ---------------------------------------------------------------------------

def extract_decision_rules(
    rl_trajectories: list[dict],
    env,
    min_support: int = 10,
    max_rules: int = 30,
) -> pd.DataFrame:
    """
    Convert the RL policy into human-readable if-then routing rules.

    Approach: for each (from_activity, to_activity) pair that the agent
    strongly prefers, characterise the KPI conditions under which it makes
    that choice by binning the KPI signals at those decision points.

    This is a rule extraction approach (not a decision tree fit) — it reads
    the agent's actual behaviour rather than approximating it with a surrogate
    model. Each rule describes:
      - The routing decision (from → to)
      - The KPI context in which the agent makes it (delay level, rework, etc.)
      - How often the rule fires and what reward it produces

    Parameters
    ----------
    min_support : int
        Minimum number of times a (from, to) pair must appear to generate a rule.
    max_rules : int
        Maximum number of rules to return (sorted by support × avg_reward).

    Returns DataFrame with columns:
        from_activity, to_activity, support, avg_reward,
        condition_delay, condition_rework, condition_case_age,
        condition_volume, condition_term_prox,
        rule_text  — human-readable if-then string
    """
    # Collect all steps grouped by (from, to)
    step_groups: dict = defaultdict(list)
    for traj in rl_trajectories:
        for step in traj["steps"]:
            key = (step["from_activity"], step["to_activity"])
            step_groups[key].append(step)

    # KPI bin labels
    def _bin_delay(v):
        if v < 0.5:  return "low delay"
        if v < 1.5:  return "moderate delay"
        return "high delay"

    def _bin_rework(v):
        if v < 0.5:  return "low rework"
        if v < 1.5:  return "moderate rework"
        return "high rework"

    def _bin_age(v):
        if v < 0.3:  return "early in case"
        if v < 0.7:  return "mid-case"
        return "late in case"

    def _bin_volume(v):
        if v < 0.0:  return "low volume"
        if v < 0.5:  return "normal volume"
        return "high volume"

    def _bin_term_prox(v):
        if v < 0.1:  return "far from terminal"
        if v < 0.4:  return "approaching terminal"
        return "near terminal"

    rows = []
    for (frm, to), steps in step_groups.items():
        if len(steps) < min_support:
            continue

        rewards = [s["reward"] for s in steps]
        avg_rew = float(np.mean(rewards))

        # Characterise the typical KPI context for this routing decision
        delays   = [s["kpi_delay"]    for s in steps]
        reworks  = [s["kpi_rework"]   for s in steps]
        ages     = [s["kpi_case_age"] for s in steps]
        volumes  = [s["kpi_volume"]   for s in steps]
        t_proxes = [s["kpi_term_prox"] for s in steps]

        cond_delay  = _bin_delay(float(np.median(delays)))
        cond_rework = _bin_rework(float(np.median(reworks)))
        cond_age    = _bin_age(float(np.median(ages)))
        cond_volume = _bin_volume(float(np.median(volumes)))
        cond_tprox  = _bin_term_prox(float(np.median(t_proxes)))

        # Build human-readable rule
        is_bad = to in env._bad_terminals
        is_good = to in env._good_terminals
        outcome_tag = " [BAD TERMINAL]" if is_bad else (" [GOOD TERMINAL]" if is_good else "")

        rule = (
            f"IF at '{frm}' "
            f"AND {cond_delay} AND {cond_rework} AND {cond_age} AND {cond_volume} "
            f"THEN route to '{to}'{outcome_tag}"
        )

        rows.append({
            "from_activity":      frm,
            "to_activity":        to,
            "support":            len(steps),
            "avg_reward":         round(avg_rew, 3),
            "condition_delay":    cond_delay,
            "condition_rework":   cond_rework,
            "condition_case_age": cond_age,
            "condition_volume":   cond_volume,
            "condition_term_prox": cond_tprox,
            "is_bad_terminal":    is_bad,
            "is_good_terminal":   is_good,
            "rule_text":          rule,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Score: support × avg_reward (prefer high-frequency, high-reward rules)
    # Separate good-terminal rules (always show) from intermediate routing
    df["score"] = df["support"] * df["avg_reward"].clip(lower=0)
    df = df.sort_values("score", ascending=False).head(max_rules).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def print_decision_rules(rules_df: pd.DataFrame, dataset: str):
    """Pretty-print the extracted decision rules."""
    print("=" * 80)
    print(f"  EXTRACTED ROUTING RULES — {dataset}")
    print(f"  ({len(rules_df)} rules, sorted by support × reward)")
    print("=" * 80)
    print()
    for _, row in rules_df.iterrows():
        tag = ""
        if row["is_bad_terminal"]:
            tag = "  *** BAD TERMINAL ***"
        elif row["is_good_terminal"]:
            tag = "  *** GOOD TERMINAL ***"
        print(f"  Rule {row['rank']:2d}{tag}")
        print(f"  {row['rule_text']}")
        print(f"  Support: {row['support']} steps  |  Avg reward: {row['avg_reward']:+.3f}")
        print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

def generate_recommendations(
    pref_df: pd.DataFrame,
    env,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate actionable routing recommendations from the preference table.

    Uses transition_type to separate:
    - 'impossible' transitions from 'learned_avoid' (different business meaning)
    - Smoothed log2_ratio instead of raw preference_ratio for filtering
    """
    rows = []

    # Strong preferences: log2_ratio > 1 (RL at least 2x more likely than random)
    preferred = pref_df[
        (pref_df["log2_ratio"] > 1.0) &
        (pref_df["rl_count"] >= 5) &
        (pref_df["avg_reward_rl"] > 0) &
        (pref_df["transition_type"].isin(["learned_prefer", "neutral", "unknown"]))
    ].head(top_n // 2)

    for _, r in preferred.iterrows():
        is_bad = r["to_activity"] in env._bad_terminals
        rows.append({
            "type":           "PREFER" if not is_bad else "AVOID_BAD",
            "from_activity":  r["from_activity"],
            "to_activity":    r["to_activity"],
            "rl_pct":         r["rl_pct"],
            "random_pct":     r["random_pct"],
            "ratio":          r["preference_ratio"],
            "log2_ratio":     r["log2_ratio"],
            "avg_reward":     r["avg_reward_rl"],
            "transition_type": r["transition_type"],
            "in_real_log":    r["in_real_log"],
            "insight":        (f"RL routes {r['rl_pct']:.0f}% from '{r['from_activity']}' "
                               f"to '{r['to_activity']}' vs {r['random_pct']:.0f}% random "
                               f"(log2 ratio: {r['log2_ratio']:+.1f})."),
            "recommendation": _route_recommendation(
                r["from_activity"], r["to_activity"],
                r["preference_ratio"], r["avg_reward_rl"], env._bad_terminals),
        })

    # Strong avoidances: log2_ratio < -1, split by transition_type
    avoided = pref_df[
        (pref_df["log2_ratio"] < -1.0) &
        (pref_df["random_count"] >= 5)
    ].tail(top_n // 2)

    for _, r in avoided.iterrows():
        is_bad = r["to_activity"] in env._bad_terminals
        t_type = r["transition_type"]

        # Different recommendation text for impossible vs learned avoidance
        if t_type == "impossible":
            rec_text = (
                f"NOTE: '{r['from_activity']}' to '{r['to_activity']}' never appears "
                f"in the real event log — this is a twin artefact, not a learned avoidance. "
                f"No business action needed."
            )
            rec_type = "IMPOSSIBLE"
        else:
            rec_text = _avoid_recommendation(
                r["from_activity"], r["to_activity"], is_bad, r["random_pct"])
            rec_type = "AVOID_BAD" if is_bad else "AVOID"

        rows.append({
            "type":           rec_type,
            "from_activity":  r["from_activity"],
            "to_activity":    r["to_activity"],
            "rl_pct":         r["rl_pct"],
            "random_pct":     r["random_pct"],
            "ratio":          r["preference_ratio"],
            "log2_ratio":     r["log2_ratio"],
            "avg_reward":     r["avg_reward_rl"],
            "transition_type": t_type,
            "in_real_log":    r["in_real_log"],
            "insight":        (f"RL avoids '{r['from_activity']}' to '{r['to_activity']}' "
                               f"({r['rl_pct']:.0f}% vs {r['random_pct']:.0f}% random, "
                               f"type: {t_type})."),
            "recommendation": rec_text,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort: IMPOSSIBLE last (informational only), others by log2_ratio magnitude
        type_order = {"PREFER": 0, "AVOID_BAD": 1, "AVOID": 2, "IMPOSSIBLE": 3}
        df["_sort_type"] = df["type"].map(type_order).fillna(9)
        df = df.sort_values(["_sort_type", "log2_ratio"],
                            ascending=[True, False]).drop(columns="_sort_type")
        df = df.reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    return df


def _route_recommendation(from_act: str, to_act: str, ratio: float,
                           avg_reward: float, bad_terminals: set) -> str:
    if to_act in bad_terminals:
        return (f"The agent routes to '{to_act}' (bad terminal) from '{from_act}' "
                f"more than random. Review whether this transition can be blocked.")
    return (f"Prioritise routing from '{from_act}' to '{to_act}' "
            f"({ratio:.1f}x more than random, avg reward {avg_reward:+.3f}). "
            f"Consider making this the default transition rule at this decision point.")


def _avoid_recommendation(from_act: str, to_act: str,
                           is_bad: bool, random_pct: float) -> str:
    if is_bad:
        return (f"The agent avoids '{to_act}' (bad terminal) from '{from_act}'. "
                f"Random policy hits this path {random_pct:.0f}% of the time. "
                f"Implement as a hard routing rule to block this transition.")
    return (f"The agent avoids '{from_act}' to '{to_act}'. "
            f"This path may lead to loops or inefficiency. "
            f"Consider adding a routing constraint to discourage it.")


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_insights_report(
    rec_df: pd.DataFrame,
    rl_summary: dict,
    random_summary: dict,
    dataset: str,
    mgmt_df: Optional[pd.DataFrame] = None,
):
    """Pretty-print the full insights report."""
    print("=" * 80)
    print(f"  WORKFLOW ROUTING INSIGHTS — {dataset}")
    print(f"  RL Backtracking Analysis ({rl_summary['n_episodes']} episodes)")
    print("=" * 80)

    print(f"\n  OUTCOME COMPARISON")
    print(f"  {'Metric':<25} {'Random':>10}  {'Our RL':>10}  {'Change':>10}")
    print(f"  {'-'*60}")
    for label, key, higher_better in [
        ("Mean reward",       "mean_reward",       True),
        ("Terminal rate",     "terminal_rate",      True),
        ("Bad terminal rate", "bad_terminal_rate",  False),
        ("Mean length",       "mean_length",        False),
    ]:
        rv = random_summary[key]
        lv = rl_summary[key]
        delta = lv - rv
        sign = "+" if (delta > 0) == higher_better else "-"
        print(f"  {label:<25} {rv:>10.3f}  {lv:>10.3f}  {sign} {delta:+.3f}")

    if mgmt_df is not None and not mgmt_df.empty:
        print(f"\n  MANAGEMENT ACTION USAGE (RL vs Random, top 8 by RL rate)")
        print(f"  {'Action':<42} {'RL rate':>8}  {'Rnd rate':>8}  {'Ratio':>6}")
        print(f"  {'-'*70}")
        for _, row in mgmt_df.head(8).iterrows():
            if row["rl_rate"] > 0.001:
                print(f"  {row['mgmt_action']:<42} {row['rl_rate']:>8.1%}  "
                      f"{row['random_rate']:>8.1%}  {row['ratio']:>6.1f}x")

    # Separate impossible transitions from real recommendations
    if not rec_df.empty:
        real_recs = rec_df[rec_df["type"] != "IMPOSSIBLE"]
        impossible = rec_df[rec_df["type"] == "IMPOSSIBLE"]

        print(f"\n  TOP ROUTING RECOMMENDATIONS")
        print(f"  {'#':<4} {'Type':<12} {'From':<30} {'To':<30} {'log2R':>6}")
        print(f"  {'-'*85}")
        for _, row in real_recs.iterrows():
            print(f"  {row['rank']:<4} {row['type']:<12} {row['from_activity']:<30} "
                  f"{row['to_activity']:<30} {row['log2_ratio']:>+6.1f}")
            print(f"       {row['insight']}")
            print(f"       {row['recommendation']}")
            print()

        if not impossible.empty:
            print(f"\n  IMPOSSIBLE TRANSITIONS (twin artefacts — no business action needed)")
            print(f"  {'From':<30} {'To':<30} {'Rnd count':>10}")
            print(f"  {'-'*75}")
            for _, row in impossible.iterrows():
                print(f"  {row['from_activity']:<30} {row['to_activity']:<30} "
                      f"{int(row['random_pct']):>10}")

    print("=" * 80)


import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Optional
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Trajectory collection (C2 routing env — MultiDiscrete action space)
# ---------------------------------------------------------------------------

