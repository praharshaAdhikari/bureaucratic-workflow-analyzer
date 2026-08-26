"""
timeutils.py
------------
One safe way to get a UTC-aware timestamp column, used everywhere.

Why this module exists
----------------------
The pipeline used to normalise timestamps like this:

    pd.to_datetime(df["timestamp"].astype(str), utc=True, errors="coerce")

On an already-typed datetime column that round-trip is lossy.  Values that land
exactly on a whole second render without a fractional part, so the column
becomes a *mixed*-format string set; pandas then fails to parse the
whole-second ones and ``errors="coerce"`` turns them into NaT.

NaT is not neutral.  ``sort_values(["case_id", "timestamp"])`` places NaT rows
last within each case, so an event from the middle of a trace is moved to the
end — and the directly-follows relation learned from that ordering contains
edges that never occurred in the log.

Measured on the committed training logs before the fix:

    BPIC2012   286 NaT   ->  173 DF edges vs 125 real   (48 fabricated, 27.7%)
    BPIC2017 1,177 NaT   ->  243 DF edges vs 178 real   (65 fabricated, 26.7%)
    BPIC2015     0 NaT   ->  9,064 DF edges vs 9,064    (clean)

Use ``ensure_utc_timestamps(df)`` at every entry point that sorts or diffs
timestamps.  It never stringifies a column that is already a datetime, and it
refuses to hand back silent NaT.
"""

from __future__ import annotations

import pandas as pd


class TimestampParseError(ValueError):
    """Raised when a timestamp column cannot be fully parsed."""


def to_utc_series(series: pd.Series, *, strict: bool = True) -> pd.Series:
    """
    Convert `series` to UTC-aware datetimes without a lossy string round-trip.

    Parameters
    ----------
    series : pd.Series
        Datetime-like (numpy or Arrow backed) or string timestamps.
    strict : bool
        When True (default), raise ``TimestampParseError`` if any value fails
        to parse.  When False, unparseable values come back as NaT — only use
        this where the caller drops NaT rows explicitly.

    Returns
    -------
    pd.Series of dtype datetime64[..., UTC]
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        # Already a datetime (including Arrow timestamp) — convert directly.
        # Never route this through astype(str); that is the original bug.
        out = pd.to_datetime(series, utc=True)
    else:
        # Genuine string / object column, e.g. straight out of the XES parser.
        # format="mixed" lets whole-second and sub-second values coexist.
        try:
            out = pd.to_datetime(series, utc=True, format="mixed")
        except (TypeError, ValueError):
            out = pd.to_datetime(
                series.astype("string"), utc=True, format="mixed", errors="coerce"
            )

    n_bad = int(out.isna().sum()) - int(pd.isna(series).sum())
    if strict and n_bad > 0:
        sample = series[out.isna() & series.notna()].head(3).tolist()
        raise TimestampParseError(
            f"{n_bad} timestamp value(s) could not be parsed. "
            f"Examples: {sample}. Fix the source data rather than coercing to "
            f"NaT — NaT rows sort to the end of their case and fabricate "
            f"directly-follows edges."
        )
    return out


def ensure_utc_timestamps(
    df: pd.DataFrame,
    column: str = "timestamp",
    *,
    strict: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Return `df` with `column` guaranteed to be UTC-aware datetimes.

    Set ``copy=False`` only when the caller already owns a private copy.
    """
    if column not in df.columns:
        raise KeyError(f"DataFrame has no '{column}' column")
    out = df.copy() if copy else df
    out[column] = to_utc_series(out[column], strict=strict)
    return out


def sort_events(
    df: pd.DataFrame,
    case_col: str = "case_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Chronologically sort an event log within each case.

    Uses a stable sort so that events sharing a timestamp keep their original
    (log) order rather than being shuffled — BPIC logs record simultaneous
    lifecycle events, and their recorded order is meaningful.
    """
    return df.sort_values([case_col, time_col], kind="stable")
