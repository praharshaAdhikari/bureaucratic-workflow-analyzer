"""
digital_twin.py
---------------
Generalized Digital Twin simulator using SimPy discrete-event simulation.

Key design decisions based on empirical analysis of bureaucratic logs:

1. EMPIRICAL TRACE LENGTH SAMPLING (primary stopping mechanism)
   Pre-sample the target trace length at case start by drawing directly from
   the empirical distribution (bootstrap/inverse-CDF). This guarantees the
   output trace length distribution matches the real one exactly, regardless
   of process structure or loop rates. The old hazard function approach failed
   for high-rework processes (e.g. BPIC-2017 with mean rework=23) because
   Markov loops kept the chain running past the target length.

2. EMPIRICAL DURATION SAMPLING (per-activity)
   Store the actual observed inter-event gaps per activity and sample from
   them directly (bootstrap). This handles any distribution shape — multi-modal,
   heavy-tailed, zero-inflated — without parametric assumptions. The old
   two-component log-normal model failed for BPIC-2015 where many activities
   have bimodal or irregular gap distributions.

3. PARALLEL SIMULATION (multiprocessing)
   simulate() splits cases across N workers (default: all CPU cores) using
   concurrent.futures.ProcessPoolExecutor. Each worker runs an independent
   SimPy environment with a deterministically seeded RNG, so results are
   reproducible. simulate_case() (used by the RL env) is single-threaded.

4. FALLBACK TRANSITIONS (non-terminal only)
   Activities with no outgoing transitions get a fallback sampled from the
   global frequency distribution, excluding terminal activities. This prevents
   the sim from getting stuck and producing degenerate short traces.

5. RESOURCE CAPACITY FROM SQRT(N_CASES)
   Standard operations-research approximation for service capacity.
"""

import simpy
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from timeutils import ensure_utc_timestamps, sort_events


# ---------------------------------------------------------------------------
# Parallel simulation worker (module-level so it's picklable)
# ---------------------------------------------------------------------------

def _simulate_chunk(twin: "DigitalTwin", case_ids: list[str], seed: int,
                    arrival_rate_s: float) -> list[dict]:
    """
    Simulate a subset of cases in an independent SimPy environment.
    Called by ProcessPoolExecutor workers — must be a module-level function.
    Each worker gets its own RNG seeded from `seed` for reproducibility.
    """
    twin.rng = np.random.default_rng(seed)
    env        = simpy.Environment()
    pool       = ResourcePool(twin._resource_capacities, env)
    events_out: list[dict] = []

    def arrival_process():
        for cid in case_ids:
            env.process(twin._case_process(env, pool, cid, events_out, {}))
            yield env.timeout(float(twin.rng.exponential(arrival_rate_s)))

    env.process(arrival_process())
    env.run()
    return events_out


# ---------------------------------------------------------------------------
# Resource pool (SimPy-backed)
# ---------------------------------------------------------------------------

class ResourcePool:
    def __init__(self, capacities: dict[str, int], env: simpy.Environment):
        self.capacities = capacities
        self._resources: dict[str, simpy.Resource] = {
            res: simpy.Resource(env, capacity=max(1, cap))
            for res, cap in capacities.items()
        }
        self._default = simpy.Resource(env, capacity=9999)

    def get(self, resource: str) -> simpy.Resource:
        return self._resources.get(resource, self._default)

    @property
    def current_load(self) -> dict[str, int]:
        return {r: res.count for r, res in self._resources.items()}

    def is_available(self, resource: str) -> bool:
        res = self._resources.get(resource)
        return True if res is None else res.count < res.capacity


# ---------------------------------------------------------------------------
# Digital Twin
# ---------------------------------------------------------------------------

class DigitalTwin:
    """
    SimPy-based generalized process simulator fitted from an event log.

    Usage:
        twin = DigitalTwin()
        twin.fit(df)
        sim_df = twin.simulate(n_cases=500)
    """

    def __init__(self, max_trace_len: int = 200, seed: int = 42):
        self.max_trace_len = max_trace_len
        self.rng = np.random.default_rng(seed)

        self.transition_probs:     dict[str, dict[str, float]] = {}
        self.start_activities:     dict[str, float] = {}
        self.terminal_activities:  set = set()
        self._fallback_transitions: dict[str, float] = {}
        self._trace_len_hazard:    np.ndarray = np.array([])
        self._trace_len_empirical: np.ndarray = np.array([])  # for direct sampling

        # Duration model: empirical arrays per activity (primary)
        # plus parametric params kept for backward compat / diagnostics
        self._duration_empirical:  dict[str, np.ndarray] = {}
        self.processing_params:    dict[str, tuple] = {}
        self.waiting_params:       dict = {}   # kept for compat, not used in sampling
        self._zero_gap_activities: set = set()
        self._has_waiting:         set = set()  # kept for compat

        # Keep duration_params as unified interface for diagnostics
        self.duration_params:     dict[str, tuple] = {}

        self.role_per_activity:   dict[str, dict[str, float]] = {}
        self.resource_per_role:   dict[str, dict[str, float]] = {}
        self._resource_capacities: dict[str, int] = {}
        self.role_map:            dict[str, str] = {}
        self.activities:          list[str] = []
        self.kpi_baselines:       dict = {}
        self._empirical_loop_rates: dict[str, float] = {}

        # Role-based constraints:
        # role_activity_map[role] = frozenset of activities that role is qualified for
        # activity_role_map[activity] = frozenset of roles qualified to perform it
        # Both are derived from observed (role, activity) pairs in the training log.
        self.role_activity_map:   dict[str, frozenset] = {}
        self.activity_role_map:   dict[str, frozenset] = {}

        # Cross-training overrides applied by RL actions (episode-scoped).
        # Maps role -> set of extra activities temporarily unlocked.
        self._cross_train_overrides: dict[str, set] = {}

        self.resource_pool = _LightResourcePool()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DigitalTwin":
        # Coerce timestamp to UTC-aware datetime regardless of dtype
        # (handles ArrowDtype large_string / timestamp from PyArrow-backed parquet).
        # Must not go via astype(str) — see timeutils for why.
        df = ensure_utc_timestamps(df)
        df = sort_events(df)
        self.activities = sorted(df["activity"].dropna().unique().tolist())
        self._fit_transitions(df)
        self._fit_trace_length_cdf(df)
        self._fit_durations(df)
        self._fit_roles(df)
        self._fit_resource_capacities(df)
        self._fit_kpi_baselines(df)
        return self

    def _fit_transitions(self, df: pd.DataFrame):
        counts:          dict = defaultdict(lambda: defaultdict(int))
        start_counts:    dict = defaultdict(int)
        terminal_counts: dict = defaultdict(int)

        for _, grp in df.groupby("case_id"):
            acts = grp["activity"].tolist()
            if not acts:
                continue
            start_counts[acts[0]] += 1
            terminal_counts[acts[-1]] += 1
            for a, b in zip(acts[:-1], acts[1:]):
                counts[a][b] += 1

        total_start = sum(start_counts.values())
        self.start_activities = {a: c / total_start for a, c in start_counts.items()}

        for src, targets in counts.items():
            total = sum(targets.values())
            self.transition_probs[src] = {t: c / total for t, c in targets.items()}

        # Terminal detection: end_frac > 15% AND terminal_rate > 40%
        n_traces = df["case_id"].nunique()
        activity_counts = df["activity"].value_counts()

        self.terminal_activities = set()
        for act, end_count in terminal_counts.items():
            end_frac      = end_count / n_traces
            terminal_rate = end_count / max(activity_counts.get(act, 1), 1)
            if end_frac > 0.15 and terminal_rate > 0.40:
                self.terminal_activities.add(act)

        if not self.terminal_activities:
            best = max(terminal_counts, key=lambda a: terminal_counts[a])
            self.terminal_activities.add(best)

        # Fallback distribution: global activity frequency EXCLUDING terminals
        # Used when an activity has no outgoing transitions
        non_terminal_freq = (
            activity_counts[~activity_counts.index.isin(self.terminal_activities)]
        )
        total_non_term = non_terminal_freq.sum()
        self._fallback_transitions = (non_terminal_freq / total_non_term).to_dict()

        # Empirical self-loop rates
        for act in activity_counts.index:
            total_out = sum(counts[act].values())
            if total_out > 0:
                self._empirical_loop_rates[act] = counts[act].get(act, 0) / total_out

    def _fit_trace_length_cdf(self, df: pd.DataFrame):
        """
        Store the empirical trace length distribution for direct inverse-CDF sampling.

        Instead of a hazard function (which can fail to stop loops in high-rework
        processes), we pre-sample the target trace length at case start using the
        empirical CDF. This guarantees the output trace length distribution matches
        the real one exactly, regardless of process structure or loop rates.

        We also keep the hazard array for backward compatibility with any code that
        reads it, but _should_stop() is no longer the primary stopping mechanism.
        """
        trace_lens = df.groupby("case_id")["activity"].count().values
        max_len    = int(np.max(trace_lens))

        # Store empirical distribution for diagnostics
        self._trace_len_mean = float(np.mean(trace_lens))
        self._trace_len_p50  = float(np.median(trace_lens))
        self._trace_len_p99  = float(np.percentile(trace_lens, 99))

        # Hard cap at p99 + 10 so sim can't produce unrealistically long traces
        p99_len = int(self._trace_len_p99) + 10
        self.max_trace_len = p99_len

        # Store the raw empirical trace lengths for inverse-CDF sampling
        # Clip at p99 to avoid extreme outliers dominating
        self._trace_len_empirical = trace_lens[trace_lens <= p99_len].copy()

        # Build PMF over trace lengths 1..max_len (kept for diagnostics / hazard compat)
        counts_per_len = np.bincount(trace_lens, minlength=max_len + 1).astype(float)
        pmf      = counts_per_len / counts_per_len.sum()
        survival = np.clip(1.0 - np.cumsum(pmf), 0, 1)

        hazard = np.zeros(max_len + 2)
        for s in range(0, max_len + 1):
            surv_s = survival[s]
            if surv_s > 1e-9:
                p_stop = pmf[s + 1] if s + 1 < len(pmf) else 0.0
                hazard[s] = p_stop / surv_s
            else:
                hazard[s] = 1.0

        self._trace_len_hazard = np.clip(hazard, 0, 1)

    def _sample_trace_length(self) -> int:
        """
        Sample a target trace length directly from the empirical distribution
        (inverse-CDF / bootstrap sampling). This is the primary stopping mechanism.
        """
        return int(self.rng.choice(self._trace_len_empirical))

    def _fit_durations(self, df: pd.DataFrame):
        """
        Empirical duration model per activity using inverse-CDF (quantile) sampling.

        Instead of fitting a parametric log-normal, we store the actual observed
        inter-event gaps per activity and sample from them directly. This is fully
        data-driven and handles any distribution shape — multi-modal, heavy-tailed,
        or zero-inflated — without parametric assumptions.

        For activities with very few observations (< MIN_SAMPLES), we fall back to
        a log-normal fitted from the available data, or a global fallback.

        Zero-gap activities (>50% of gaps == 0s) get a fixed ~10s duration.
        """
        MIN_SAMPLES = 5  # minimum observations to use empirical sampling

        df = sort_events(ensure_utc_timestamps(df))
        df["next_ts"] = df.groupby("case_id")["timestamp"].shift(-1)
        df["dur_s"]   = (df["next_ts"] - df["timestamp"]).dt.total_seconds()
        df_gaps = df[df["dur_s"].notna()].copy()

        # Classify zero-gap activities (>50% of gaps are 0s)
        zero_frac = df_gaps.groupby("activity").apply(
            lambda g: (g["dur_s"] == 0).mean()
        )
        self._zero_gap_activities = set(zero_frac[zero_frac > 0.5].index)

        # Global fallback: all non-zero gaps > 60s across all activities
        global_gaps = df_gaps[df_gaps["dur_s"] > 60]["dur_s"].values
        if len(global_gaps) < MIN_SAMPLES:
            global_gaps = np.array([300.0])  # 5 minutes

        # Per-activity empirical duration arrays.
        #
        # These hold EVERY observed gap, including zeros and sub-minute ones.
        # They used to hold only gaps > 60s, with the comment "matches what the
        # KS metric sees" — and that is exactly the problem: the duration
        # metric filters both the real and simulated sides to > 60s, so it is
        # structurally blind to whatever the fit discards. The two agreed with
        # each other while disagreeing with the log.
        #
        # The effect was large. A_SUBMITTED in BPIC2012 is followed within a
        # second by A_PARTLYSUBMITTED in essentially every case, so its real
        # median gap is 0s. Dropping everything under a minute left only the
        # rare long tail, and the sampler returned a median of 1,035s and a
        # mean of 84,118s for it. Summed over a trace, simulated cycle times
        # came out 3-11x longer than the real ones.
        self._duration_empirical: dict[str, np.ndarray] = {}

        # Keep parametric params for backward compat (used by duration_params)
        fallback_mu = float(np.log(5 * 60))

        for act in self.activities:
            grp = df_gaps[df_gaps["activity"] == act]
            observed = grp["dur_s"].clip(lower=0.0).values

            if len(observed) >= MIN_SAMPLES:
                # Empirical: store sorted array for fast quantile sampling
                self._duration_empirical[act] = np.sort(observed)
            elif len(observed) > 0:
                # Too few samples: widen with a log-normal fitted from what we
                # have, keeping the observations themselves.
                log_vals = np.log(np.clip(observed, 1, None))
                mu    = float(np.median(log_vals))
                sigma = float(max(np.std(log_vals), 0.3))
                synth = np.exp(np.random.default_rng(42).normal(mu, sigma, 50))
                self._duration_empirical[act] = np.sort(
                    np.concatenate([observed, synth])
                )
            else:
                # Activity never observed with a following event: fall back to
                # the global gap distribution.
                self._duration_empirical[act] = np.sort(global_gaps)

            # Parametric params (kept for backward compat / diagnostics)
            arr = self._duration_empirical[act]
            log_arr = np.log(np.clip(arr, 1, None))
            mu    = float(np.median(log_arr))
            sigma = float(max(np.std(log_arr), 0.1))
            self.processing_params[act] = (mu, sigma)
            self.duration_params[act]   = (mu, sigma)
            self.waiting_params[act]    = None  # type: ignore[assignment]

        # Fit case-level duration distribution for reference
        case_durs = df.groupby("case_id")["timestamp"].agg(
            lambda x: (x.max() - x.min()).total_seconds() / 86400
        ).clip(lower=0)
        p99 = float(case_durs.quantile(0.99))
        case_durs_capped = case_durs.clip(upper=p99)
        log_case = np.log(np.clip(case_durs_capped.values, 0.01, None))
        self._case_duration_mu    = float(np.median(log_case))
        self._case_duration_sigma = float(np.std(log_case))

    def _fit_roles(self, df: pd.DataFrame):
        from feature_engineering import _is_numeric_column, _derive_role_from_activity
        if "role" not in df.columns or _is_numeric_column(df["role"]):
            df = df.copy()
            df["role"] = df["activity"].apply(_derive_role_from_activity)

        for act, grp in df.groupby("activity"):
            self.role_per_activity[act] = grp["role"].value_counts(normalize=True).to_dict()

        for res, grp in df.groupby("resource"):
            mode = grp["role"].mode()
            self.role_map[res] = mode.iloc[0] if len(mode) > 0 else "UNKNOWN"

        for role, grp in df.groupby("role"):
            self.resource_per_role[role] = grp["resource"].value_counts(normalize=True).to_dict()

        # Build hard-constraint maps from observed (role, activity) pairs.
        # A role is "qualified" for an activity if it performed it at least once.
        role_acts: dict[str, set] = {}
        for act, role_dist in self.role_per_activity.items():
            for role in role_dist:
                role_acts.setdefault(role, set()).add(act)

        self.role_activity_map = {r: frozenset(acts) for r, acts in role_acts.items()}
        self.activity_role_map = {
            act: frozenset(
                role for role, acts in self.role_activity_map.items() if act in acts
            )
            for act in self.activities
        }

    def _fit_resource_capacities(self, df: pd.DataFrame):
        resource_caps: dict[str, int] = {}
        for res, grp in df.groupby("resource"):
            n_cases = grp["case_id"].nunique()
            cap = max(2, int(np.ceil(np.sqrt(n_cases))))
            resource_caps[res] = cap
        self._resource_capacities = resource_caps
        self.resource_pool = _LightResourcePool(resource_caps)

    def _fit_kpi_baselines(self, df: pd.DataFrame):
        from feature_engineering import compute_case_kpis
        kpis = compute_case_kpis(df)
        trace_lens = df.groupby("case_id")["activity"].count()
        self.kpi_baselines = {
            "median_trace_length":  float(trace_lens.median()),
            "p95_trace_length":     float(trace_lens.quantile(0.95)),
            "median_case_age_days": float(kpis["case_age_days"].median()),
            "mean_rework":          float(kpis["rework_count"].mean()),
            "completion_rate":      float(kpis["is_completed"].mean()),
        }

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_start_activity(self) -> str:
        acts  = list(self.start_activities.keys())
        probs = list(self.start_activities.values())
        return str(self.rng.choice(acts, p=probs))

    def _sample_next_activity(self, current: str) -> str:
        """Sample next activity. Falls back to non-terminal frequency if no transitions."""
        if current in self.transition_probs:
            targets = self.transition_probs[current]
            acts    = list(targets.keys())
            probs   = list(targets.values())
            return str(self.rng.choice(acts, p=probs))
        acts  = list(self._fallback_transitions.keys())
        probs = list(self._fallback_transitions.values())
        return str(self.rng.choice(acts, p=probs))

    def _should_stop(self, step: int) -> bool:
        """Check whether to stop after completing `step` (0-indexed).
        hazard[step] = P(stop | reached step)."""
        if step < len(self._trace_len_hazard):
            return bool(self.rng.random() < self._trace_len_hazard[step])
        return True

    def _sample_duration(self, activity: str) -> float:
        """
        Sample inter-event gap for an activity using empirical inverse-CDF sampling.

        Draws uniformly from the stored empirical gap array (bootstrap sampling).
        This exactly reproduces the real duration distribution for each activity,
        handling any shape — multi-modal, heavy-tailed, zero-inflated — without
        parametric assumptions.

        Zero gaps are part of the empirical arrays now, so activities that
        complete instantly sample 0 naturally. ``_zero_gap_activities`` is kept
        as a diagnostic set but no longer short-circuits sampling — doing so
        replaced a true 0s gap with a synthetic ~10s one.
        """
        empirical = self._duration_empirical.get(activity)
        if empirical is not None and len(empirical) > 0:
            return float(self.rng.choice(empirical))

        # Fallback: log-normal from stored params
        mu, sigma = self.processing_params.get(activity, (np.log(5 * 60), 0.5))
        return float(np.exp(self.rng.normal(mu, sigma)))

    def _sample_role(self, activity: str) -> str:
        """
        Sample a role for the given activity, restricted to qualified roles.

        Qualified roles = those observed performing this activity in the training
        log (role_activity_map), plus any cross-training overrides active in the
        current episode (_cross_train_overrides).

        Falls back to the full role_per_activity distribution if no qualified
        role has a resource available (should be rare).
        """
        if activity not in self.role_per_activity:
            return "UNKNOWN"

        # Roles that are qualified for this activity (hard constraint)
        qualified = self.activity_role_map.get(activity, frozenset())

        # Add any cross-training overrides: roles that have this activity unlocked
        if self._cross_train_overrides:
            extra = frozenset(
                role for role, extra_acts in self._cross_train_overrides.items()
                if activity in extra_acts
            )
            qualified = qualified | extra

        # Filter role_per_activity to qualified roles only
        role_dist = self.role_per_activity[activity]
        filtered = {r: p for r, p in role_dist.items() if r in qualified}

        if not filtered:
            # Fallback: use full distribution (shouldn't happen for trained activities)
            filtered = role_dist

        roles = list(filtered.keys())
        probs = np.array(list(filtered.values()), dtype=float)
        probs /= probs.sum()
        return str(self.rng.choice(roles, p=probs))

    def _sample_resource(self, role: str, pool: ResourcePool, activity: str = "") -> str:
        """
        Sample a resource for the given role, preferring available workers.

        When an activity is provided, further restricts candidates to resources
        whose role is qualified for that activity — enforcing the hard constraint
        that workers can only be assigned to tasks their role covers.
        """
        if role not in self.resource_per_role:
            return "UNKNOWN"
        resources = list(self.resource_per_role[role].keys())
        probs     = list(self.resource_per_role[role].values())

        # If activity is known, filter to resources whose role qualifies for it
        if activity:
            qualified_roles = self.activity_role_map.get(activity, frozenset())
            # Add cross-training overrides
            if self._cross_train_overrides:
                extra = frozenset(
                    r for r, extra_acts in self._cross_train_overrides.items()
                    if activity in extra_acts
                )
                qualified_roles = qualified_roles | extra

            resources_filtered = [
                r for r in resources
                if self.role_map.get(r, "UNKNOWN") in qualified_roles
            ]
            if resources_filtered:
                probs_filtered = [probs[resources.index(r)] for r in resources_filtered]
                resources, probs = resources_filtered, probs_filtered

        probs_arr = np.array(probs, dtype=float)
        probs_arr /= probs_arr.sum()

        available = [r for r in resources if pool.is_available(r)]
        if available:
            ap = np.array([probs_arr[resources.index(r)] for r in available], dtype=float)
            ap /= ap.sum()
            return str(self.rng.choice(available, p=ap))
        return str(self.rng.choice(resources, p=probs_arr))

    # ------------------------------------------------------------------
    # SimPy case process
    # ------------------------------------------------------------------

    def _case_process(
        self,
        env: simpy.Environment,
        pool: ResourcePool,
        case_id: str,
        events_out: list,
        rl_overrides: dict,
    ):
        # Pre-sample target trace length from empirical distribution.
        # This is the primary stopping mechanism — it guarantees the output
        # trace length distribution matches the real one exactly.
        target_len = self._sample_trace_length()
        current_act = self._sample_start_activity()

        for step in range(target_len):
            role     = self._sample_role(current_act)
            resource = self._sample_resource(role, pool, activity=current_act)
            dur      = self._sample_duration(current_act)

            simpy_res = pool.get(resource)
            with simpy_res.request() as req:
                yield req
                yield env.timeout(dur)

            events_out.append({
                "case_id":    case_id,
                "activity":   current_act,
                "role":       role,
                "resource":   resource,
                "sim_time":   env.now,
                "duration_s": dur,
                "step":       step,
            })

            if step in rl_overrides:
                current_act = rl_overrides[step]
                continue

            # Stop if we've reached the last step of the target length
            if step == target_len - 1:
                break

            current_act = self._sample_next_activity(current_act)

    # ------------------------------------------------------------------
    # Public simulation API
    # ------------------------------------------------------------------

    def simulate(
        self,
        n_cases: int = 500,
        rl_overrides_per_case: Optional[dict] = None,
        arrival_rate_s: float = 3600.0,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Simulate n_cases cases in parallel across CPU cores.

        Each worker runs an independent SimPy environment over its chunk of
        cases, with a deterministically seeded RNG so results are reproducible.
        Resource contention is per-chunk (not global), which is a valid
        approximation since the metrics we optimise don't depend on cross-case
        queuing — and it's what makes parallelism possible.

        Args:
            n_cases:        Number of cases to simulate.
            arrival_rate_s: Mean inter-arrival time in seconds (Poisson).
            n_jobs:         Number of parallel workers. -1 = all CPU cores.
        """
        import os
        if rl_overrides_per_case:
            # RL overrides need the original single-env path (overrides are per case-id)
            return self._simulate_single(n_cases, rl_overrides_per_case, arrival_rate_s)

        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        n_workers = min(n_workers, n_cases)  # no point spawning more workers than cases

        # Split case IDs into chunks — one per worker
        all_ids = [f"SIM_{i:05d}" for i in range(n_cases)]
        chunks  = [all_ids[i::n_workers] for i in range(n_workers)]

        # Each worker gets a unique seed derived from self's seed for reproducibility
        base_seed = int(self.rng.integers(0, 2**31))
        worker_seeds = [base_seed + i for i in range(n_workers)]

        all_events: list[dict] = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_simulate_chunk, self, chunk, seed, arrival_rate_s): i
                for i, (chunk, seed) in enumerate(zip(chunks, worker_seeds))
            }
            for future in as_completed(futures):
                all_events.extend(future.result())

        df = pd.DataFrame(all_events)
        if df.empty:
            return df

        MAX_SIM_SECONDS = 10 * 365 * 86400
        origin = pd.Timestamp("2020-01-01", tz="UTC")
        df["timestamp"] = origin + pd.to_timedelta(
            df["sim_time"].clip(upper=MAX_SIM_SECONDS), unit="s"
        )
        return df.sort_values(["case_id", "sim_time"]).reset_index(drop=True)

    def _simulate_single(
        self,
        n_cases: int = 500,
        rl_overrides_per_case: Optional[dict] = None,
        arrival_rate_s: float = 3600.0,
    ) -> pd.DataFrame:
        """Single-threaded simulation — used when rl_overrides are present."""
        rl_overrides_per_case = rl_overrides_per_case or {}
        env        = simpy.Environment()
        pool       = ResourcePool(self._resource_capacities, env)
        events_out: list[dict] = []

        def arrival_process():
            for i in range(n_cases):
                cid       = f"SIM_{i:05d}"
                overrides = rl_overrides_per_case.get(cid, {})
                env.process(self._case_process(env, pool, cid, events_out, overrides))
                yield env.timeout(float(self.rng.exponential(arrival_rate_s)))

        env.process(arrival_process())
        env.run()

        df = pd.DataFrame(events_out)
        if df.empty:
            return df

        MAX_SIM_SECONDS = 10 * 365 * 86400
        origin = pd.Timestamp("2020-01-01", tz="UTC")
        df["timestamp"] = origin + pd.to_timedelta(
            df["sim_time"].clip(upper=MAX_SIM_SECONDS), unit="s"
        )
        return df.sort_values(["case_id", "sim_time"]).reset_index(drop=True)

    def __setstate__(self, state: dict):
        """Backward-compat unpickling: fill in attributes added after initial release."""
        self.__dict__.update(state)
        if not hasattr(self, "role_activity_map"):
            self.role_activity_map = {}
        if not hasattr(self, "activity_role_map"):
            self.activity_role_map = {}
        if not hasattr(self, "_cross_train_overrides"):
            self._cross_train_overrides = {}

    def simulate_case(
        self,
        case_id: str = "SIM_0",
        rl_overrides: Optional[dict] = None,
    ) -> list[dict]:
        """Simulate a single case (no resource contention). Used by RL env."""
        rl_overrides = rl_overrides or {}
        env  = simpy.Environment()
        pool = ResourcePool(self._resource_capacities, env)
        events_out: list[dict] = []
        env.process(self._case_process(env, pool, case_id, events_out, rl_overrides))
        env.run()
        self.resource_pool.current_load = pool.current_load
        return events_out

    # ------------------------------------------------------------------
    # Role-constraint helpers (used by RL env actions)
    # ------------------------------------------------------------------

    def enable_cross_training(self, role: str, activities: list[str]) -> None:
        """
        Temporarily unlock ``activities`` for ``role`` in the current episode.

        Called by the RL env when the ``enable_cross_trained_pool`` action is
        taken.  Overrides are episode-scoped — call ``reset_cross_training()``
        at the start of each episode.
        """
        self._cross_train_overrides.setdefault(role, set()).update(activities)

    def reset_cross_training(self) -> None:
        """Clear all cross-training overrides (call at episode reset)."""
        self._cross_train_overrides.clear()

    def get_qualified_resources(self, activity: str) -> list[str]:
        """
        Return all resources (workers) qualified to perform ``activity``,
        including any active cross-training overrides.
        """
        qualified_roles = set(self.activity_role_map.get(activity, frozenset()))
        if self._cross_train_overrides:
            for role, extra_acts in self._cross_train_overrides.items():
                if activity in extra_acts:
                    qualified_roles.add(role)

        result = []
        for role in qualified_roles:
            result.extend(self.resource_per_role.get(role, {}).keys())
        return list(dict.fromkeys(result))  # deduplicate, preserve order

    def get_overloaded_resources(self, threshold: float = 0.8) -> list[str]:
        """
        Return resources whose current load exceeds ``threshold`` fraction of capacity.
        Used by the ``reroute_from_overloaded_employee`` RL action.
        """
        overloaded = []
        for res, cap in self.resource_pool.capacities.items():
            load = self.resource_pool.current_load.get(res, 0)
            if cap > 0 and load / cap >= threshold:
                overloaded.append(res)
        return overloaded


# ---------------------------------------------------------------------------
# Lightweight resource pool for RL env
# ---------------------------------------------------------------------------

@dataclass
class _LightResourcePool:
    capacities:   dict = field(default_factory=dict)
    current_load: dict = field(default_factory=dict)

    def __init__(self, capacities: Optional[dict] = None):
        self.capacities   = capacities or {}
        self.current_load = {r: 0 for r in self.capacities}

    def is_available(self, resource: str) -> bool:
        return self.current_load.get(resource, 0) < self.capacities.get(resource, 1)

    def assign(self, resource: str):
        self.current_load[resource] = self.current_load.get(resource, 0) + 1

    def release(self, resource: str):
        self.current_load[resource] = max(0, self.current_load.get(resource, 0) - 1)
