"""
figures.py
----------
Every saved figure in the pipeline, in one place.

Why this module exists
----------------------
The six notebooks each hand-rolled their own matplotlib. That produced 21
panels, and an audit of all of them found the same failures repeated:

* **Duplicates.** ``data_overview.png`` had four panels holding two series —
  all-activity counts beside top-15 counts, and an events-per-case histogram
  beside a bar chart of that histogram's own quantiles.
* **Dead-flat series.** ``terminal_rate`` ran 1.000..1.000 across the whole of
  BPIC2012's training, ``truncated_rate`` 0.000..0.000, and after the verdict
  moved to the environment ``ep_reward_mean`` moves about one point on a scale
  of 34 — so the headline reward curve reads as "nothing was learned" when
  episode length halved.
* **Duplicated series inside one figure.** ``action_entropy`` is byte-identical
  to ``routing_entropy`` on all three datasets.
* **A wrong computation.** The routing heatmap plotted
  ``log2(max(log2_ratio, 0.01))`` — a log of a log, with every avoided route
  clamped to one value. 74.6% of BPIC2012's cells rendered the same shade.
* **Unreadable axes.** 57.1% of BPIC2012 cases fall in the first of 30
  histogram bins; validation metrics span 5,224x on a linear axis.

Each function here takes plain data and a path, saves, and returns the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent across every figure so a reader can carry meaning between them.
C = {
    "rl":       "#1f4e9c",   # the learned policy
    "baseline": "#8c9bab",   # any non-learned policy
    "real":     "#a44235",   # the real log — always the reference
    "good":     "#3f7a66",
    "bad":      "#a44235",
    "neutral":  "#b6bfca",
    "accent":   "#8a6512",
    "grid":     "#dde3ea",
}

POLICY_ORDER = ["Random", "FIFO", "Greedy", "EmpMk", "RwdG", "RL"]


def _style(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, out_path: "str | Path") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def _annotate(ax, xs, values, fmt="{:.0f}", dy=0.02) -> None:
    top = max(values) if len(values) else 1
    for x, v in zip(xs, values):
        ax.text(x, v + top * dy, fmt.format(v), ha="center", va="bottom",
                fontsize=8, fontweight="bold")


# ---------------------------------------------------------------------------
# Notebook 01 — the log itself
# ---------------------------------------------------------------------------

def log_overview(df: pd.DataFrame, out_path, dataset: str, top_n: int = 15):
    """
    Two panels replacing the previous four.

    Dropped: the all-activities bar chart (the same counts as the top-N panel,
    and a 356-row chart on BPIC2015), and the events-per-case percentile bars
    (quantiles of the histogram sitting next to it). The percentiles are now
    marked on the histogram itself, and the events axis is log-scaled because
    45-57% of BPIC2012 cases otherwise land in a single bin.
    """
    counts = df["activity"].value_counts()
    per_case = df.groupby("case_id").size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    head = counts.head(top_n).iloc[::-1]
    ax.barh(range(len(head)), head.values, color=C["rl"], alpha=0.85)
    ax.set_yticks(range(len(head)))
    ax.set_yticklabels([a[:38] for a in head.index], fontsize=8)
    ax.set_xlabel("Events")
    ax.set_title(f"Top {top_n} activities of {counts.size}", fontweight="bold")
    _style(ax)

    ax = axes[1]
    ax.hist(per_case, bins=np.logspace(np.log10(max(per_case.min(), 1)),
                                       np.log10(per_case.max()), 40),
            color=C["baseline"], edgecolor="white")
    ax.set_xscale("log")
    for q, style in ((0.50, "-"), (0.90, "--"), (0.99, ":")):
        v = per_case.quantile(q)
        ax.axvline(v, color=C["real"], ls=style, lw=1.4,
                   label=f"p{int(q * 100)} = {v:.0f}")
    ax.set_xlabel("Events per case (log scale)")
    ax.set_ylabel("Cases")
    ax.set_title("Case length", fontweight="bold")
    ax.legend(fontsize=8)
    _style(ax)

    fig.suptitle(f"{dataset} — {len(df):,} events, {per_case.size:,} cases, "
                 f"{counts.size} activities", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Notebook 02 — why each activity got its label
# ---------------------------------------------------------------------------

def terminal_evidence(diag: pd.DataFrame, good: set, bad: set, excluded: set,
                      thresholds: dict, out_path, dataset: str):
    """
    Replaces the trace-ending-rate bar chart.

    That chart coloured the top-15 trace-enders by good/bad, but the outcome
    markers chosen by ``classify_terminals`` end roughly 0% of traces — they
    are activities that occur *near* the end, not ones that end a case. So
    almost every bar rendered grey and the figure contradicted the labelling
    it was meant to illustrate.

    This plots the two quantities that actually decide the label: how late in a
    trace an activity occurs, and how many cases contain it.
    """
    min_pos = thresholds.get("min_end_position", 0.80)
    min_cov = thresholds.get("min_case_coverage", 0.02)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.axvspan(min_pos, 1.02, color=C["grid"], alpha=0.55, zorder=0)
    ax.axvline(min_pos, color=C["accent"], lw=1.4, ls="--",
               label=f"qualifies at p25 ≥ {min_pos}")
    ax.axhline(min_cov, color=C["accent"], lw=1.0, ls=":",
               label=f"and coverage ≥ {min_cov:.0%}")

    groups = [
        (good,     C["good"],    "good outcome",  70, "o"),
        (bad,      C["bad"],     "bad outcome",   70, "s"),
        (excluded, C["accent"],  "near the end, but outcome-neutral", 45, "^"),
    ]
    labelled = set().union(good, bad, excluded)
    rest = diag.loc[[a for a in diag.index if a not in labelled]]
    ax.scatter(rest["rel_pos_p25"], rest["case_coverage"], s=18,
               color=C["neutral"], alpha=0.65, label="not an outcome marker",
               zorder=2)

    for members, colour, label, size, marker in groups:
        sub = diag.loc[[a for a in diag.index if a in members]]
        if sub.empty:
            continue
        ax.scatter(sub["rel_pos_p25"], sub["case_coverage"], s=size,
                   color=colour, marker=marker, edgecolor="white", lw=0.8,
                   label=f"{label} ({len(sub)})", zorder=3)

    for act in list(good) + list(bad):
        if act in diag.index:
            r = diag.loc[act]
            ax.annotate(act[:30], (r["rel_pos_p25"], r["case_coverage"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_yscale("log")
    ax.set_xlabel("Position in the trace — 25th percentile of where this activity occurs\n"
                  "(1.0 = always at the very end)")
    ax.set_ylabel("Share of cases containing it (log scale)")
    ax.set_title(f"{dataset} — what qualifies an activity as an outcome",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    _style(ax)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Notebook 03 — fidelity
# ---------------------------------------------------------------------------

def validation_figure(results: dict, out_path, dataset: str):
    """
    One panel instead of two.

    The old figure plotted metric value against threshold on a linear axis,
    but the values span 5,224x on BPIC2012 (0.0002 to 1.0000), so five of the
    seven bars were invisible slivers. Its second panel was a pass/fail
    scorecard — seven green ticks that repeated the colour coding of the first.

    Plotting value ÷ threshold on a log axis makes every metric comparable and
    puts the pass line at 1.0, so how *much* headroom each metric has is
    readable at a glance.
    """
    rows = []
    for name, entry in results.items():
        if not isinstance(entry, dict) or entry.get("threshold") in (None, 0):
            continue
        value, thr = float(entry["value"]), float(entry["threshold"])
        higher_better = name == "variant_coverage"
        # Ratio below 1 always means "passing", whichever direction is good.
        ratio = (thr / value if value > 0 else np.inf) if higher_better else (value / thr)
        rows.append({"metric": name, "ratio": ratio,
                     "passed": bool(entry["passed"]), "value": value,
                     "threshold": thr})

    table = pd.DataFrame(rows).sort_values("ratio")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(table) + 1.5)))

    colours = [C["good"] if p else C["bad"] for p in table["passed"]]
    ax.barh(range(len(table)), table["ratio"], color=colours, alpha=0.85)
    ax.axvline(1.0, color="#222", lw=1.6)
    ax.text(1.0, len(table) - 0.35, "  threshold", fontsize=9, va="center")

    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["metric"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Metric value ÷ threshold (log scale) — under 1.0 passes")

    for i, r in enumerate(table.itertuples()):
        ax.text(r.ratio * 1.08, i, f"{r.value:.4g} / {r.threshold:g}",
                va="center", fontsize=7.5)

    n_fail = int((~table["passed"]).sum())
    verdict = "all metrics pass" if n_fail == 0 else f"{n_fail} metric(s) FAIL"
    ax.set_title(f"{dataset} — simulation fidelity: {verdict}", fontweight="bold")
    _style(ax)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Notebook 04 — training
# ---------------------------------------------------------------------------

def training_progress(metrics: pd.DataFrame, out_path, dataset: str,
                      real_steps_median: "float | None" = None,
                      base_rate: "float | None" = None,
                      w_terminal: float = 30.0):
    """
    Rebuilt around what actually moves.

    Removed: the completion-rate panel (1.000..1.000 on BPIC2012), and one of
    ``action_entropy`` / ``routing_entropy`` (byte-identical on all three
    datasets).

    Promoted: episode length, which is the learning curve now — 9.3..17.2 on
    BPIC2012 while total reward moved 33.35..34.25.

    Added: a reward decomposition, because total reward is dominated by the
    fixed conclusion bonus. Showing the shaped part separately explains why
    the total looks flat instead of leaving a reader to misread it as "nothing
    was learned". And the outcome rate, explicitly labelled as a line that
    should *not* move — it is drawn by the environment, so drift there means
    the verdict is leaking back to the agent.
    """
    steps = metrics["timestep"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # 1. Episode length — the learning signal.
    ax = axes[0, 0]
    ax.plot(steps, metrics["ep_len_mean"], color=C["rl"], lw=2)
    if real_steps_median:
        ax.axhline(real_steps_median, color=C["real"], ls="--", lw=1.4,
                   label=f"real median = {real_steps_median:.0f}")
        ax.legend(fontsize=8)
    ax.set_ylabel("Steps per case")
    ax.set_title("Episode length", fontweight="bold")
    _style(ax)

    # 2. Reward, split so the flat part is explained rather than hidden.
    #    The shaped part gets its own axis: it runs 3.3..4.2 on BPIC2012 while
    #    the total runs 33.4..34.2, so on a shared axis the only part that
    #    actually moves would be an invisible wiggle.
    ax = axes[0, 1]
    total = metrics["ep_reward_mean"]
    concluded = metrics["terminal_rate"] * w_terminal
    shaped = total - concluded
    ax.plot(steps, total, color=C["rl"], lw=2, label="total (left)")
    ax.plot(steps, concluded, color=C["neutral"], lw=1.4, ls="--",
            label=f"conclusion bonus, {w_terminal:g} × completion (left)")
    ax.set_ylabel("Reward", color=C["rl"])
    _style(ax)

    ax2 = ax.twinx()
    ax2.plot(steps, shaped, color=C["accent"], lw=1.8,
             label="shaping + length (right)")
    ax2.set_ylabel("Shaped component", color=C["accent"])
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7)
    ax.set_title("Reward, decomposed —\nthe total is dominated by a constant",
                 fontweight="bold", fontsize=10)

    # 3. Outcome rate — a line that should not move. Fixed window with a
    #    tolerance band, so ordinary sampling noise is not auto-scaled into
    #    something that looks like a trend.
    ax = axes[0, 2]
    ax.plot(steps, metrics["good_terminal_rate"] * 100, color=C["good"], lw=2)
    if base_rate is not None:
        pct = base_rate * 100
        ax.axhspan(pct - 5, pct + 5, color=C["good"], alpha=0.12,
                   label="±5pp sampling band")
        ax.axhline(pct, color=C["real"], ls="--", lw=1.4,
                   label=f"real base rate = {base_rate:.1%}")
        ax.set_ylim(max(0, pct - 20), min(100, pct + 20))
        ax.legend(fontsize=8, loc="lower right")
    ax.set_ylabel("Good outcomes (%)")
    ax.set_title("Outcome rate — drawn by the environment,\nshould stay flat",
                 fontweight="bold", fontsize=10)
    _style(ax)

    # 4. KPI signals.
    ax = axes[1, 0]
    for col, colour, label in (("delay_mean", C["bad"], "delay"),
                               ("rework_mean", C["accent"], "rework"),
                               ("risk_mean", C["rl"], "risk")):
        if col in metrics:
            ax.plot(steps, metrics[col], color=colour, lw=1.5, label=label)
    ax.set_ylabel("Normalised")
    ax.set_title("KPI signals", fontweight="bold")
    ax.legend(fontsize=8)
    _style(ax)

    # 5. Exploration. Only one of the two identical entropies.
    ax = axes[1, 1]
    ax.plot(steps, metrics["routing_entropy"], color=C["rl"], lw=1.6,
            label="routing")
    if "mgmt_entropy" in metrics:
        ax.plot(steps, metrics["mgmt_entropy"], color=C["accent"], lw=1.6,
                label="management")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Exploration", fontweight="bold")
    ax.legend(fontsize=8)
    _style(ax)

    # 6. Critic.
    ax = axes[1, 2]
    ax.plot(steps, metrics["value_loss"], color=C["bad"], lw=1.5, label="value loss")
    ax.set_ylabel("Value loss", color=C["bad"])
    _style(ax)
    if "explained_variance" in metrics:
        ax2 = ax.twinx()
        ax2.plot(steps, metrics["explained_variance"], color=C["good"], lw=1.5,
                 ls="--", label="explained variance")
        ax2.set_ylabel("Explained variance", color=C["good"])
        ax2.spines["top"].set_visible(False)
    ax.set_title("Critic", fontweight="bold")

    for ax in axes.flat:
        ax.set_xlabel("Timesteps", fontsize=9)

    fig.suptitle(f"Training — {dataset} ({int(steps.max()):,} steps)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Notebook 05 — policy comparison
# ---------------------------------------------------------------------------

def policy_comparison(all_results: dict, out_path, real_reference: dict):
    """
    The headline figure, rebuilt around what separates the policies.

    Reward no longer does. RL beats the best baseline by +1.16, +0.86 and
    +2.60 on a scale of ~33, while finishing in 9.5 steps against 15.4, and
    22.6 against 48.2. Every reward-based panel understated the result.

    ``real_reference[dataset]`` supplies ``steps_median``, ``cycle_days_median``
    and ``p_good`` from the log.

    Row 1 — episode length against the real median.
    Row 2 — cycle time against the real median (meaningful only since the
            duration fix; it was previously re-simulated independently of the
            policy).
    Row 3 — conclusion rate. This is where the baselines genuinely fail:
            Empirical Markov truncates 92.8% of BPIC2015 episodes.
    Row 4 — outcome rate against the real base rate, as validation. No policy
            can move it; the panel exists to show that it doesn't.
    """
    datasets = list(all_results.keys())
    n = len(datasets)
    fig, axes = plt.subplots(4, n, figsize=(5.2 * n, 15))
    if n == 1:
        axes = axes.reshape(4, 1)

    keys = ["random", "fifo", "greedy", "empirical", "reward", "rl"]

    for col, ds in enumerate(datasets):
        res = all_results[ds]
        ref = real_reference.get(ds, {})
        colours = [C["rl"] if k == "rl" else C["baseline"] for k in keys]
        xs = np.arange(len(keys))

        # Row 0 — episode length
        ax = axes[0, col]
        vals = [res[k]["mean_length"] for k in keys]
        ax.bar(xs, vals, color=colours, alpha=0.9, width=0.62)
        _annotate(ax, xs, vals, "{:.0f}")
        if ref.get("steps_median"):
            ax.axhline(ref["steps_median"], color=C["real"], ls="--", lw=1.5,
                       label=f"real = {ref['steps_median']:.0f}")
            ax.legend(fontsize=8)
        ax.set_ylabel("Steps per case")
        ax.set_title(f"[{ds}] Episode length", fontweight="bold")

        # Row 1 — cycle time
        ax = axes[1, col]
        vals = [res[k].get("median_cycle_days", np.nan) for k in keys]
        if not np.all(np.isnan(vals)):
            ax.bar(xs, vals, color=colours, alpha=0.9, width=0.62)
            _annotate(ax, xs, vals, "{:.1f}")
            if ref.get("cycle_days_median"):
                ax.axhline(ref["cycle_days_median"], color=C["real"], ls="--",
                           lw=1.5, label=f"real = {ref['cycle_days_median']:.1f}d")
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "cycle time not collected", ha="center",
                    va="center", transform=ax.transAxes, color="#888")
        ax.set_ylabel("Median cycle time (days)")
        ax.set_title(f"[{ds}] Cycle time", fontweight="bold")

        # Row 2 — conclusion rate
        ax = axes[2, col]
        vals = [100.0 - res[k]["truncated_pct"] for k in keys]
        ax.bar(xs, vals, color=colours, alpha=0.9, width=0.62)
        _annotate(ax, xs, vals, "{:.0f}%")
        ax.set_ylim(0, 118)
        ax.set_ylabel("% of cases concluded")
        ax.set_title(f"[{ds}] Cases actually finished", fontweight="bold")

        # Row 3 — outcome rate, as validation
        ax = axes[3, col]
        vals = []
        for k in keys:
            concluded = res[k]["good_term_pct"] + res[k]["bad_term_pct"]
            vals.append(res[k]["good_term_pct"] / concluded * 100 if concluded else np.nan)
        ax.bar(xs, vals, color=[C["good"]] * len(keys), alpha=0.75, width=0.62)
        _annotate(ax, xs, vals, "{:.0f}%")
        if ref.get("p_good") is not None:
            ax.axhline(ref["p_good"] * 100, color=C["real"], ls="--", lw=1.5,
                       label=f"real = {ref['p_good']:.1%}")
            ax.legend(fontsize=8)
        ax.set_ylim(0, 118)
        ax.set_ylabel("Good outcomes (% of concluded)")
        ax.set_title(f"[{ds}] Outcome mix — set by the environment,\n"
                     f"identical by design", fontweight="bold", fontsize=10)

        for row in range(4):
            a = axes[row, col]
            a.set_xticks(xs)
            a.set_xticklabels(POLICY_ORDER, fontsize=9)
            _style(a)

    fig.suptitle("RL vs baselines — lower is better on rows 1-2, "
                 "higher on row 3, flat on row 4",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def steps_to_outcome(all_results: dict, real_reference: dict, out_path):
    """
    Replaces "Final Activity Distribution", which compared RL against Random on
    a quantity the environment now draws — so the two were identical by
    construction and the panel could only ever show noise.

    Distribution of how long a case takes, RL against Random against the log.
    """
    datasets = list(all_results.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8))
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        res = all_results[ds]
        ref = real_reference.get(ds, {})
        bins = np.linspace(0, max(
            np.percentile(res["random"]["lengths"], 99),
            np.percentile(res["rl"]["lengths"], 99),
            ref.get("steps_median", 0) * 2,
        ), 40)
        ax.hist(res["random"]["lengths"], bins=bins, color=C["baseline"],
                alpha=0.7, label="Random", density=True)
        ax.hist(res["rl"]["lengths"], bins=bins, color=C["rl"], alpha=0.7,
                label="RL", density=True)
        if ref.get("steps_median"):
            ax.axvline(ref["steps_median"], color=C["real"], ls="--", lw=1.8,
                       label=f"real median = {ref['steps_median']:.0f}")
        ax.set_xlabel("Steps to conclusion")
        ax.set_ylabel("Density")
        ax.set_title(f"[{ds}]", fontweight="bold")
        ax.legend(fontsize=8)
        _style(ax)

    fig.suptitle("How long a case takes — learned policy vs random vs the real log",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# Notebook 06 — routing behaviour
# ---------------------------------------------------------------------------

def routing_heatmap(pref_df: pd.DataFrame, activities: list, out_path,
                    dataset: str, bad_terminals: set = frozenset(),
                    max_activities: int = 80):
    """
    Fixes a genuine bug.

    The old code computed ``np.log2(max(row['log2_ratio'], 0.01))`` — a log2 of
    a value that is already a log2. The ``max`` also clamped every negative
    entry, meaning every route the agent avoids, to a single number. On
    BPIC2012 that collapsed 47 of 63 populated cells (74.6%) to one shade, so
    the whole "RL avoids" half of the legend carried no information, while the
    axis label described a quantity that was not being plotted.

    Two further changes. ``log2_ratio`` is now plotted directly, and cells
    neither policy ever visits are masked to grey rather than rendered as 0 —
    on BPIC2012 that is 89% of the matrix, and 0 previously read as "both
    policies did this equally often".
    """
    n = len(activities)
    if n > max_activities:
        return None

    idx = {a: i for i, a in enumerate(activities)}
    mat = np.full((n, n), np.nan)
    for row in pref_df.itertuples():
        i, j = idx.get(row.from_activity), idx.get(row.to_activity)
        if i is not None and j is not None:
            mat[i, j] = row.log2_ratio          # already log2 — do not log again

    fig, ax = plt.subplots(figsize=(max(10, n * 0.62), max(8, n * 0.55)))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#9aa5b1")                     # never visited by either policy

    # Scale on a robust quantile rather than the extremes. The distribution is
    # lopsided — BPIC2012 runs to -8.8 but only +3.0 — so scaling to the max
    # washes every mid-range cell out to near-white. Outliers saturate instead.
    finite = np.abs(mat[np.isfinite(mat)])
    vmax = float(np.quantile(finite, 0.90)) if finite.size else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto")

    short = [a.replace("W_", "W:").replace("A_", "A:").replace("O_", "O:")[:20]
             for a in activities]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("To activity"); ax.set_ylabel("From activity")

    for j, act in enumerate(activities):
        if act in bad_terminals:
            ax.axvline(j - 0.5, color=C["bad"], lw=0.8, alpha=0.5)
            ax.axvline(j + 0.5, color=C["bad"], lw=0.8, alpha=0.5)

    covered = np.isfinite(mat).sum()
    ax.set_title(
        f"[{dataset}] Routing preference, RL vs Random\n"
        f"blue = RL prefers, red = RL avoids, grey = neither policy took it "
        f"({covered}/{n * n} cells populated)",
        fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, extend="both", shrink=0.8,
                 label=f"log2(RL rate / Random rate), clipped at ±{vmax:.1f}")
    fig.tight_layout()
    _save(fig, out_path)
    return fig
