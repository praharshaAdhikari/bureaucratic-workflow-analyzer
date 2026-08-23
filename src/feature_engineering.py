"""
feature_engineering.py
-----------------------
Derives KPI signals, activity/role embeddings, and transition statistics
from parsed event log DataFrames.

Embeddings use TF-IDF on activity n-gram sequences + TruncatedSVD (LSA),
giving a dense vector per activity with no external dependencies beyond sklearn.
The ActivityEmbedder object is saved/loaded with joblib (also sklearn ecosystem).
"""

import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import joblib


# ---------------------------------------------------------------------------
# Role derivation helpers
# ---------------------------------------------------------------------------

def _ensure_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the 'timestamp' column is a proper UTC-aware datetime dtype.
    Handles object, ArrowDtype(large_string), and already-parsed datetime columns.
    Called at the top of every function that does timestamp arithmetic.
    """
    ts = df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        # Already datetime — make sure it's tz-aware
        if ts.dt.tz is None:
            df = df.copy()
            df["timestamp"] = ts.dt.tz_localize("UTC")
        return df
    # String or ArrowDtype string — parse
    df = df.copy()
    df["timestamp"] = pd.to_datetime(ts.astype(str), utc=True, errors="coerce")
    return df

def _is_numeric_column(series: pd.Series) -> bool:
    """True if >80% of non-null values are numeric strings (e.g. resource IDs)."""
    sample = series.dropna().astype(str).head(100)
    if len(sample) == 0:
        return True
    return sample.str.match(r"^\d+$").mean() > 0.8


def _derive_role_from_activity(activity: str) -> str:
    """
    Derive a role/department label from activity name patterns.
    BPIC-2015: '01_HOOFD_110' → 'HOOFD'
    BPIC-2017: 'A_Create Application' → 'Create'
    Generic  : first meaningful token uppercased.
    """
    if not isinstance(activity, str) or activity == "UNKNOWN":
        return "UNKNOWN"
    m = re.match(r"^\d+_([A-Z]+)_", activity)
    if m:
        return m.group(1)
    m = re.match(r"^[A-Z]_(\w+)", activity)
    if m:
        return m.group(1)
    parts = activity.replace("_", " ").split()
    return parts[0].upper() if parts else "UNKNOWN"


# ---------------------------------------------------------------------------
# Activity embeddings — TF-IDF + SVD (no gensim)
# ---------------------------------------------------------------------------

class ActivityEmbedder:
    """
    Learns dense activity embeddings from trace sequences using TF-IDF + SVD (LSA).

    Each trace is treated as a "document" where activities are "words".
    TF-IDF captures activity importance across traces; SVD reduces to a dense
    low-dimensional space. The result is a lookup table: activity → float32 vector.

    Why this instead of Word2Vec?
    - Zero extra dependencies (sklearn is already required for validation)
    - Deterministic, fast, works well on small corpora
    - SimPy-friendly: no background threads or C extensions
    """

    def __init__(self, vector_size: int = 32, seed: int = 42):
        self.vector_size = vector_size
        self.seed = seed
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._activity_vectors: dict[str, np.ndarray] = {}
        self._zero: np.ndarray = np.zeros(vector_size, dtype=np.float32)

    def fit(self, df: pd.DataFrame) -> "ActivityEmbedder":
        """
        Fit on an event log DataFrame.
        Each trace becomes one document; each activity is a token.
        """
        # Build one "document" per trace (activity sequence as space-joined string)
        traces = (
            df.sort_values(["case_id", "timestamp"])
            .groupby("case_id")["activity"]
            .apply(lambda acts: " ".join(acts.astype(str).tolist()))
            .tolist()
        )

        n_components = min(self.vector_size, len(traces) - 1,
                           df["activity"].nunique() - 1)
        n_components = max(n_components, 2)

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\S+",   # treat any non-space token as a word
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = self._vectorizer.fit_transform(traces)  # (n_traces, n_activities)

        self._svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        # We want activity vectors, not trace vectors → transpose
        # SVD on activity×trace matrix: each activity gets a vector
        activity_tfidf = tfidf_matrix.T  # (n_activities, n_traces)
        activity_vecs = self._svd.fit_transform(activity_tfidf)  # (n_activities, n_components)
        activity_vecs = normalize(activity_vecs, norm="l2")

        # Pad or trim to vector_size
        if activity_vecs.shape[1] < self.vector_size:
            pad = np.zeros((activity_vecs.shape[0], self.vector_size - activity_vecs.shape[1]),
                           dtype=np.float32)
            activity_vecs = np.hstack([activity_vecs, pad])
        else:
            activity_vecs = activity_vecs[:, :self.vector_size]

        vocab = self._vectorizer.get_feature_names_out()
        self._activity_vectors = {
            act: activity_vecs[i].astype(np.float32)
            for i, act in enumerate(vocab)
        }
        self._zero = np.zeros(self.vector_size, dtype=np.float32)
        return self

    def get_vector(self, activity: str) -> np.ndarray:
        """Return embedding for an activity, or zeros if unseen."""
        return self._activity_vectors.get(str(activity), self._zero).copy()

    def embed_trace(self, activities: list[str]) -> np.ndarray:
        """Mean-pool activity embeddings for a trace."""
        if not activities:
            return self._zero.copy()
        vecs = [self.get_vector(a) for a in activities]
        return np.mean(vecs, axis=0).astype(np.float32)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ActivityEmbedder":
        return joblib.load(path)

    def __len__(self):
        return len(self._activity_vectors)


def train_activity_embeddings(
    df: pd.DataFrame,
    vector_size: int = 32,
    seed: int = 42,
    **_kwargs,          # absorb unused Word2Vec kwargs (window, epochs, etc.)
) -> ActivityEmbedder:
    """Fit and return an ActivityEmbedder. Drop-in replacement for Word2Vec training."""
    embedder = ActivityEmbedder(vector_size=vector_size, seed=seed)
    embedder.fit(df)
    return embedder


def embed_trace(activities: list[str], model: ActivityEmbedder) -> np.ndarray:
    """Convenience wrapper — matches old gensim call signature."""
    return model.embed_trace(activities)


def get_activity_vector(activity: str, model: ActivityEmbedder) -> np.ndarray:
    """Convenience wrapper — matches old gensim call signature."""
    return model.get_vector(activity)


# ---------------------------------------------------------------------------
# KPI signal extraction
# ---------------------------------------------------------------------------

def compute_case_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-case KPI signals used as RL state features and reward proxies.

    Fully vectorized — no Python-level case loop.

    Returns a DataFrame indexed by case_id with columns:
        case_age_days, trace_length, rework_count, unique_resources,
        unique_roles, has_objection, has_suspension, has_refusal,
        is_completed, risk_score_proxy, delay_proxy, volume_pressure
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    if "role" not in df.columns or _is_numeric_column(df["role"]):
        df["role"] = df["activity"].apply(_derive_role_from_activity)

    # ── Vectorized aggregations ───────────────────────────────────────
    g = df.groupby("case_id")

    # Case age: max_ts - min_ts in days
    ts_min = g["timestamp"].min()
    ts_max = g["timestamp"].max()
    case_age_days = ((ts_max - ts_min).dt.total_seconds() / 86400).clip(lower=0)

    trace_length     = g["activity"].count()
    unique_resources = g["resource"].nunique()
    unique_roles     = g["role"].nunique()

    # Rework: duplicated activities per case
    rework_count = (
        df.assign(_dup=df.duplicated(subset=["case_id", "activity"]))
        .groupby("case_id")["_dup"].sum()
        .astype(int)
    )

    # Keyword flags — join all activities per case into one string, then str.contains
    act_lower = df.copy()
    act_lower["_act_low"] = act_lower["activity"].str.lower().fillna("")
    case_acts = act_lower.groupby("case_id")["_act_low"].apply(" ".join)

    has_objection  = case_acts.str.contains("bezwaar|objection|appeal|beroep",  regex=True).astype(int)
    has_suspension = case_acts.str.contains("suspend|opschort|hold",            regex=True).astype(int)
    has_refusal    = case_acts.str.contains("refus|weiger|reject|denied",       regex=True).astype(int)

    # Completion: any lifecycle in {complete, completed}
    lc_lower = df["lifecycle"].str.lower().fillna("")
    is_completed = (
        df.assign(_done=lc_lower.isin(["complete", "completed"]))
        .groupby("case_id")["_done"].any()
        .astype(int)
    )

    risk_score_proxy = has_refusal * 2 + has_suspension + rework_count * 0.1

    kpi_df = pd.DataFrame({
        "case_age_days":    case_age_days,
        "trace_length":     trace_length,
        "rework_count":     rework_count,
        "unique_resources": unique_resources,
        "unique_roles":     unique_roles,
        "has_objection":    has_objection,
        "has_suspension":   has_suspension,
        "has_refusal":      has_refusal,
        "is_completed":     is_completed,
        "risk_score_proxy": risk_score_proxy,
    })

    mu    = kpi_df["case_age_days"].mean()
    sigma = kpi_df["case_age_days"].std() + 1e-9
    kpi_df["delay_proxy"]     = (kpi_df["case_age_days"] - mu) / sigma
    kpi_df["volume_pressure"] = _compute_volume_pressure(df)

    return kpi_df


def _compute_volume_pressure(df: pd.DataFrame) -> pd.Series:
    df = _ensure_timestamps(df)
    case_starts = df.groupby("case_id")["timestamp"].min().dropna().sort_values()
    cs_df = case_starts.reset_index()
    cs_df.columns = ["case_id", "start"]
    cs_df = cs_df.set_index("start").sort_index()
    rolling = cs_df.rolling("30D").count()["case_id"]
    cs_df["volume_pressure"] = rolling.values
    result = cs_df.reset_index().set_index("case_id")["volume_pressure"]
    return (result - result.mean()) / (result.std() + 1e-9)


def _kpi_worker(args: tuple) -> tuple[str, pd.DataFrame]:
    """Module-level worker for parallel KPI computation."""
    label, df = args
    return label, compute_case_kpis(df)


def compute_kpis_parallel(
    splits: dict[str, pd.DataFrame],
    n_jobs: int = -1,
) -> dict[str, pd.DataFrame]:
    """
    Compute KPIs for multiple splits concurrently.

    Args:
        splits:  dict of {label: DataFrame}, e.g. {'train': df_train, 'val': df_val}
        n_jobs:  worker count; -1 = min(n_splits, cpu_count)

    Returns:
        dict of {label: kpi_DataFrame}
    """
    non_empty = {k: v for k, v in splits.items() if not v.empty}
    if not non_empty:
        return {k: pd.DataFrame() for k in splits}

    n_workers = min(len(non_empty), os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)

    results: dict[str, pd.DataFrame] = {}
    if n_workers == 1 or len(non_empty) == 1:
        for label, df in non_empty.items():
            results[label] = compute_case_kpis(df)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_kpi_worker, (k, v)): k for k, v in non_empty.items()}
            for future in as_completed(futures):
                label, kpi_df = future.result()
                results[label] = kpi_df

    # Fill empty splits with empty DataFrames
    for k in splits:
        if k not in results:
            results[k] = pd.DataFrame()
    return results


# ---------------------------------------------------------------------------
# Transition statistics
# ---------------------------------------------------------------------------

def compute_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_timestamps(df)
    transitions: dict = defaultdict(lambda: defaultdict(int))
    for _, grp in df.groupby("case_id"):
        acts = grp.sort_values("timestamp")["activity"].tolist()
        for a, b in zip(acts[:-1], acts[1:]):
            transitions[a][b] += 1

    all_acts = sorted(set(df["activity"].dropna().unique()))
    mat = pd.DataFrame(0.0, index=all_acts, columns=all_acts)
    for src, targets in transitions.items():
        total = sum(targets.values())
        for tgt, cnt in targets.items():
            if src in mat.index and tgt in mat.columns:
                mat.loc[src, tgt] = cnt / total
    return mat


def compute_duration_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-activity duration statistics.

    Uses `duration_s` column directly if present (sim data from DigitalTwin,
    which stores the sampled activity duration). Falls back to computing
    inter-event gaps from timestamps for real log data.

    This ensures real vs sim comparison is apples-to-apples: both measure
    the time spent on the activity itself, not queuing/waiting time.
    """
    df = _ensure_timestamps(df)

    if "duration_s" in df.columns and df["duration_s"].notna().any():
        dur_col = df[["activity", "duration_s"]].copy()
        dur_col = dur_col[dur_col["duration_s"] > 0]
    else:
        df = df.sort_values(["case_id", "timestamp"])
        df["next_ts"]    = df.groupby("case_id")["timestamp"].shift(-1)
        df["duration_s"] = (df["next_ts"] - df["timestamp"]).dt.total_seconds()
        dur_col = df[["activity", "duration_s"]].dropna()
        dur_col = dur_col[dur_col["duration_s"] >= 0]

    return (
        dur_col.groupby("activity")["duration_s"]
        .agg(mean_dur="mean", median_dur="median", std_dur="std")
        .reset_index()
    )


def compute_loop_rates(df: pd.DataFrame) -> pd.Series:
    total_traces = df["case_id"].nunique()
    return (
        df.groupby(["case_id", "activity"]).size()
        .reset_index(name="cnt")
        .query("cnt > 1")
        .groupby("activity")["case_id"]
        .nunique()
        / total_traces
    )


def compute_resource_workload(df: pd.DataFrame) -> pd.DataFrame:
    wl = (
        df.groupby("resource")
        .agg(event_count=("activity", "count"), case_count=("case_id", "nunique"))
        .reset_index()
    )
    wl["load_score"] = (wl["event_count"] - wl["event_count"].mean()) / (wl["event_count"].std() + 1e-9)
    return wl


# ---------------------------------------------------------------------------
# Semantic activity classifier (uses sentence-transformers if available)
# ---------------------------------------------------------------------------

_SENTENCE_MODEL = None  # module-level cache — loaded once per process

def classify_bad_terminals(
    activities: list[str],
    threshold: float = 0.35,
    fallback_keywords: Optional[list[str]] = None,
) -> set[str]:
    """
    Classify which activities represent bad terminal states (rejection,
    cancellation, denial) using semantic sentence embeddings.

    Uses ``sentence-transformers`` (all-MiniLM-L6-v2) to embed each activity
    name and computes cosine similarity against a rejection anchor phrase.
    Activities above ``threshold`` are classified as bad terminals.

    Falls back to keyword matching if sentence-transformers is not installed.

    Args:
        activities:        List of activity names from the process.
        threshold:         Cosine similarity threshold above which an activity
                           is classified as a bad terminal. Default 0.35 was
                           validated on BPIC-2012 and BPIC-2017:
                             - Known bad (declined/cancelled): 0.396–0.514
                             - Known good (accepted/approved): 0.055–0.278
        fallback_keywords: Keywords to use if sentence-transformers is not
                           available. Defaults to common BPM rejection terms.

    Returns:
        Set of activity names classified as bad terminals.

    Example:
        >>> bad = classify_bad_terminals(twin.activities)
        >>> # BPIC-2012: {'A_DECLINED', 'A_CANCELLED', 'O_DECLINED', 'O_CANCELLED'}
        >>> # BPIC-2017: {'A_Denied', 'A_Cancelled', 'O_Cancelled', 'O_Refused'}
    """
    if fallback_keywords is None:
        fallback_keywords = [
            'declin', 'denied', 'deny',
            'cancel', 'cancelled',
            'refus', 'refused', 'reject',
        ]

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

        global _SENTENCE_MODEL
        if _SENTENCE_MODEL is None:
            _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        model = _SENTENCE_MODEL
        # Clean activity names: strip common prefixes and underscores
        def _clean(name: str) -> str:
            name = re.sub(r'^[A-Z]_', '', name)   # remove A_, O_, W_ etc.
            return name.replace('_', ' ').strip()

        clean_names = [_clean(a) for a in activities]
        vecs = model.encode(clean_names, normalize_embeddings=True, show_progress_bar=False)

        rejection_anchor = model.encode(
            ['application declined denied refused cancelled rejected'],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        sims = _cos_sim(vecs, [rejection_anchor]).flatten()

        bad = {act for act, sim in zip(activities, sims) if sim >= threshold}
        return bad

    except ImportError:
        # sentence-transformers not installed — fall back to keywords
        return {
            a for a in activities
            if any(k in a.lower() for k in fallback_keywords)
        }


def classify_good_terminals(
    df: pd.DataFrame,
    bad_terminals: set[str],
    min_end_rate: float = 0.05,
    require_keyword: bool = False,
    exclude_subprocess_prefixes: tuple = ("W_", "w_"),
) -> set[str]:
    """
    Identify good terminal activities from the real event log.

    Strategy — empirical end rate is the primary signal:
      1. Compute the fraction of traces that end at each activity.
      2. Keep activities that end >= ``min_end_rate`` of traces AND are not
         in ``bad_terminals`` AND don't have a subprocess prefix.
      3. If ``require_keyword=True``: additionally require a positive outcome
         keyword. Use this only for processes where outcome names are semantic
         (e.g. BPIC2012 "A_FINALIZED"). For administrative processes like
         BPIC2015 (where traces end at "close case", "set phase: ..."), leave
         this False (the default) so empirical rate alone decides.

    Fallback: if the empirical filter yields nothing, fall back to keyword-only
    matching so the env always has at least one good terminal.

    Args:
        df:                          Event log DataFrame with [case_id, activity, timestamp].
        bad_terminals:               Set of bad terminal activity names.
        min_end_rate:                Minimum fraction of traces ending at an activity.
                                     Default 0.05 (5%).
        require_keyword:             If True, also require a positive outcome keyword.
                                     Default False — empirical rate is sufficient.
        exclude_subprocess_prefixes: Activity name prefixes to always exclude.
                                     Default (\"W_\", \"w_\").

    Returns:
        Set of activity names that are good terminals.

    Example:
        >>> good = classify_good_terminals(df_train, bad_terminals)
        >>> # BPIC-2012: {'A_FINALIZED', 'A_APPROVED', ...}
        >>> # BPIC-2017: {'A_Complete', 'O_Accepted', ...}
        >>> # BPIC-2015: {'close case', 'set phase: phase permitting irrevocable', ...}
    """
    _GOOD_KEYWORDS = (
        "accept", "approv", "final", "complet", "grant", "success",
        "paid", "closed", "done", "finish", "confirm",
    )
    _BAD_KEYWORDS = (
        "cancel", "declin", "refus", "reject", "denied", "withdraw",
        "suspend", "abort", "fail", "incomplet",
    )

    df = _ensure_timestamps(df)

    # Empirical: last activity per trace
    last_activities = (
        df.sort_values(["case_id", "timestamp"])
        .groupby("case_id")["activity"]
        .last()
    )
    n_traces = len(last_activities)
    end_rates = last_activities.value_counts() / n_traces

    def _passes_filters(act: str, rate: float) -> bool:
        if rate < min_end_rate:
            return False
        if act in bad_terminals:
            return False
        if any(act.startswith(pfx) for pfx in exclude_subprocess_prefixes):
            return False
        if require_keyword:
            act_lower = act.lower()
            has_good = any(kw in act_lower for kw in _GOOD_KEYWORDS)
            has_bad  = any(kw in act_lower for kw in _BAD_KEYWORDS)
            if not has_good or has_bad:
                return False
        return True

    good = {act for act, rate in end_rates.items() if _passes_filters(act, rate)}

    # Fallback: if nothing passed the empirical filter, use keyword-only
    # so the env always has at least one good terminal to learn toward.
    if not good:
        all_activities = sorted(df["activity"].unique().tolist())
        for act in all_activities:
            if act in bad_terminals:
                continue
            if any(act.startswith(pfx) for pfx in exclude_subprocess_prefixes):
                continue
            act_lower = act.lower()
            has_good = any(kw in act_lower for kw in _GOOD_KEYWORDS)
            has_bad  = any(kw in act_lower for kw in _BAD_KEYWORDS)
            if has_good and not has_bad:
                good.add(act)

    return good
