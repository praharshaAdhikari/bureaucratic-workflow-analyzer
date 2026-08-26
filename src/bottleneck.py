"""
bottleneck.py
-------------
Where does the time actually go, and is it a staffing problem or a process
problem?

Why this module exists
----------------------
The project's goal is to find weak points in a bureaucratic workflow and say
whether to put more people on them or remove the step. That is a *diagnostic*
question, and it does not need a policy. Everything here is measured directly
from the event log:

* no trained agent, so no seed dependence (cf. Fix 21);
* no reward function, so nothing to game (cf. Fix 16);
* no assumed intervention effects (cf. Fix 15).

The one thing it does need is **start timestamps**. An event log with only
completion events records the gap between two activities as a single number
that fuses queueing and work. "Nobody was free to pick this up for six days"
and "this task genuinely takes six days" are then indistinguishable, and they
have opposite remedies. So this module refuses to guess: an activity without
paired start/complete events is reported as unmeasurable rather than estimated.

The decomposition
-----------------
For one execution of an activity:

    schedule ──────────► start ──────────► complete
             queue time          processing time

* **queue time** — the task existed and nobody worked on it. Driven by how
  many qualified people are free. This is the quantity that responds to
  *hiring*.
* **processing time** — someone was working on it. Driven by how hard the step
  is. This responds to simplification, automation, or removing the step; it
  does **not** respond to hiring.
* **suspended time** (BPIC2017 only, via suspend/resume) — work was
  deliberately paused, typically waiting on the applicant or a third party.
  Responds to neither hiring nor simplification: it is outside the office.

Splitting these three is the whole point. A step that looks slow is a hiring
problem only if its time is mostly queue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOUR = 3600.0
DAY = 86_400.0

#: Lifecycle transitions, lowercased. Logs disagree on case.
SCHEDULE = "schedule"
START = "start"
COMPLETE = "complete"
SUSPEND = "suspend"
RESUME = "resume"

#: An activity needs this many completed executions before its statistics are
#: reported. Below it the medians are anecdote.
MIN_EXECUTIONS = 30

#: Share of an activity's total elapsed time that must sit in one component
#: before the diagnosis names that component as the cause.
DOMINANT_SHARE = 0.60


@dataclass(frozen=True)
class Diagnosis:
    """What to do about an activity, and why."""
    label: str
    action: str
    rationale: str


#: The diagnoses this module can return. Deliberately few — a diagnostic that
#: can say anything says nothing.
DIAGNOSES = {
    "staffing": Diagnosis(
        "staffing bottleneck",
        "add capacity to the roles serving this step",
        "most of the elapsed time is spent queued, so the work is waiting for "
        "a person rather than taking a long time to do",
    ),
    "process": Diagnosis(
        "slow step",
        "simplify, automate, or split this step",
        "most of the elapsed time is spent working, so more people would not "
        "make an individual case finish sooner",
    ),
    "external": Diagnosis(
        "external dependency",
        "chase, set a deadline, or redesign the hand-off",
        "most of the elapsed time is suspended, i.e. waiting on someone "
        "outside the office; neither hiring nor simplification touches it",
    ),
    "rework": Diagnosis(
        "rework loop",
        "fix the upstream cause of the repeats",
        "the step is repeated far more often than it is reached, so the same "
        "work is being redone",
    ),
    "mixed": Diagnosis(
        "no single cause",
        "investigate before acting",
        "no component dominates, so any single remedy addresses less than "
        "60% of the delay",
    ),
    "unmeasurable": Diagnosis(
        "not measurable from this log",
        "instrument the process before drawing conclusions",
        "the log records no start timestamps for this step, so queueing and "
        "work cannot be separated and the two have opposite remedies",
    ),
}


def _pair_executions(case_events: pd.DataFrame) -> list[dict]:
    """
    Reconstruct individual executions from lifecycle events within one case.

    Events for the same activity repeat within a case, so schedule/start/
    complete are matched per activity rather than by assuming one execution
    each. Two subtleties, both of which produced wrong numbers in the first
    version of this module:

    **Schedules are matched backwards from the start, not FIFO.** Popping the
    oldest pending schedule can pair a start with a schedule that happened
    *after* it, which yields a negative queue time. BPIC2017's
    ``W_Validate application`` came out at −662.9 queue-days that way. The
    schedule chosen is now the latest one at or before the start; if there is
    none, queue time is left unknown rather than invented.

    **Suspension sits inside the start→complete window.** ``complete − start``
    already contains any suspend/resume gap, so reporting both as separate
    components counts that time twice — on BPIC2017 it made processing and
    suspended shares sum to ~1.0 with queueing squeezed to zero. Suspended
    time is now subtracted out, leaving processing as time genuinely spent
    working.
    """
    pending_sched: dict[str, list] = {}
    open_start: dict[str, list] = {}
    suspend_open: dict[str, list] = {}
    suspend_acc: dict[str, float] = {}
    out: list[dict] = []

    for row in case_events.itertuples(index=False):
        act, lc, ts = row.activity, row.lifecycle, row.timestamp

        if lc == SCHEDULE:
            pending_sched.setdefault(act, []).append(ts)

        elif lc == START:
            open_start.setdefault(act, []).append(ts)
            suspend_acc.setdefault(act, 0.0)

        elif lc == SUSPEND:
            if open_start.get(act):
                suspend_open.setdefault(act, []).append(ts)

        elif lc == RESUME:
            stack = suspend_open.get(act) or []
            if stack:
                suspend_acc[act] = (suspend_acc.get(act, 0.0)
                                    + (ts - stack.pop()).total_seconds())

        elif lc == COMPLETE:
            starts = open_start.get(act) or []
            if not starts:
                # Completion with no start: an instantaneous system event, or
                # a step this log does not instrument. Recorded so coverage is
                # visible rather than silently dropped.
                out.append({
                    "activity": act, "resource": getattr(row, "resource", None),
                    "queue_s": np.nan, "processing_s": np.nan,
                    "suspended_s": np.nan, "measured": False, "anomaly": False,
                })
                continue

            t_start = starts.pop()

            # Latest schedule at or before the start. Anything later belongs to
            # a different execution and would give a negative queue.
            scheds = pending_sched.get(act) or []
            eligible = [i for i, t in enumerate(scheds) if t <= t_start]
            t_sched = scheds.pop(eligible[-1]) if eligible else None

            # Any suspension still open at completion ends now.
            for t_susp in suspend_open.pop(act, []):
                suspend_acc[act] = (suspend_acc.get(act, 0.0)
                                    + (ts - t_susp).total_seconds())

            queue_s = ((t_start - t_sched).total_seconds()
                       if t_sched is not None else np.nan)
            window_s = (ts - t_start).total_seconds()
            susp_s = min(suspend_acc.pop(act, 0.0), max(window_s, 0.0))
            proc_s = window_s - susp_s

            anomaly = window_s < 0 or (queue_s is not np.nan and queue_s < 0)
            out.append({
                "activity": act,
                "resource": getattr(row, "resource", None),
                "queue_s": np.nan if anomaly else queue_s,
                "processing_s": np.nan if anomaly else proc_s,
                "suspended_s": np.nan if anomaly else susp_s,
                "measured": not anomaly,
                "anomaly": anomaly,
            })

    return out


def executions(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per activity execution, with its time split into components.

    `df` needs columns case_id, activity, timestamp, lifecycle and optionally
    resource. Lifecycle values are lowercased here so BPIC2012's uppercase and
    BPIC2017's lowercase behave identically.
    """
    work = df.copy()
    work["lifecycle"] = work["lifecycle"].astype(str).str.lower()
    work = work.sort_values(["case_id", "timestamp"], kind="stable")

    rows = []
    for case_id, g in work.groupby("case_id", sort=False):
        for rec in _pair_executions(g):
            rec["case_id"] = case_id
            rows.append(rec)
    return pd.DataFrame(rows)


def _diagnose(row) -> str:
    if not row["measurable"]:
        return "unmeasurable"
    if row["repeats_per_case"] >= 2.0:
        return "rework"
    shares = {
        "staffing": row["share_queue"],
        "process": row["share_processing"],
        "external": row["share_suspended"],
    }
    top = max(shares, key=lambda k: shares[k])
    return top if shares[top] >= DOMINANT_SHARE else "mixed"


def activity_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-activity time decomposition and diagnosis, ranked by time lost.

    Ranked by *total* queue time rather than median, because the target is
    total time removed from the process: a step that wastes two days on every
    one of 20,000 cases matters more than one that wastes a month on ten.
    """
    ex = executions(df)
    if ex.empty:
        return pd.DataFrame()

    n_cases = df["case_id"].nunique()
    rows = []

    for activity, g in ex.groupby("activity"):
        measured = g[g["measured"]]
        n_exec = len(g)
        n_measured = len(measured)

        q = measured["queue_s"].dropna()
        p = measured["processing_s"].dropna()
        s = measured["suspended_s"].dropna()

        total_q, total_p, total_s = q.sum(), p.sum(), s.sum()
        total = total_q + total_p + total_s

        measurable = n_measured >= MIN_EXECUTIONS and len(p) >= MIN_EXECUTIONS

        rows.append({
            "activity": activity,
            "n_executions": n_exec,
            "n_measured": n_measured,
            "cases_touched": g["case_id"].nunique(),
            "repeats_per_case": round(n_exec / max(g["case_id"].nunique(), 1), 2),
            "coverage_of_cases": round(g["case_id"].nunique() / max(n_cases, 1), 3),
            "n_resources": measured["resource"].nunique() if n_measured else 0,

            "median_queue_h": round(q.median() / HOUR, 2) if len(q) else np.nan,
            "median_processing_h": round(p.median() / HOUR, 2) if len(p) else np.nan,
            "median_suspended_h": round(s.median() / HOUR, 2) if len(s) else np.nan,
            "n_anomalies": int(g["anomaly"].sum()),

            "total_queue_days": round(total_q / DAY, 1),
            "total_processing_days": round(total_p / DAY, 1),
            "total_suspended_days": round(total_s / DAY, 1),

            "share_queue": round(total_q / total, 3) if total else np.nan,
            "share_processing": round(total_p / total, 3) if total else np.nan,
            "share_suspended": round(total_s / total, 3) if total else np.nan,

            "measurable": measurable,
        })

    table = pd.DataFrame(rows)
    table["diagnosis"] = table.apply(_diagnose, axis=1)
    table["recommendation"] = table["diagnosis"].map(
        lambda k: DIAGNOSES[k].action)
    table["because"] = table["diagnosis"].map(lambda k: DIAGNOSES[k].rationale)

    return table.sort_values("total_queue_days", ascending=False).reset_index(drop=True)


def case_time_budget(df: pd.DataFrame) -> dict:
    """
    Where the elapsed time of an average case goes.

    ``unattributed`` is the part of case duration that no instrumented
    execution accounts for — the gaps between steps, when the case is simply
    sitting in the system. It is reported rather than hidden because on these
    logs it is usually the largest single component, and it is the honest
    upper bound on what any within-step remedy can recover.
    """
    ex = executions(df)
    work = df.sort_values(["case_id", "timestamp"], kind="stable")
    span = work.groupby("case_id")["timestamp"].agg(["min", "max"])
    total_elapsed = (span["max"] - span["min"]).dt.total_seconds().sum()

    measured = ex[ex["measured"]]
    q = measured["queue_s"].dropna().sum()
    p = measured["processing_s"].dropna().sum()
    s = measured["suspended_s"].dropna().sum()

    return {
        "n_cases": int(df["case_id"].nunique()),
        "total_elapsed_days": round(total_elapsed / DAY, 1),
        "queue_days": round(q / DAY, 1),
        "processing_days": round(p / DAY, 1),
        "suspended_days": round(s / DAY, 1),
        "unattributed_days": round((total_elapsed - q - p - s) / DAY, 1),
        "share_queue": round(q / total_elapsed, 3) if total_elapsed else np.nan,
        "share_processing": round(p / total_elapsed, 3) if total_elapsed else np.nan,
        "share_suspended": round(s / total_elapsed, 3) if total_elapsed else np.nan,
        "share_unattributed": (round((total_elapsed - q - p - s) / total_elapsed, 3)
                               if total_elapsed else np.nan),
        "instrumented_executions": int(measured.shape[0]),
        "uninstrumented_executions": int((~ex["measured"]).sum()),
        "anomalous_executions": int(ex["anomaly"].sum()),
    }
