import concurrent.futures
import os
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pm4py
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
    from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
    from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    from pm4py.objects.process_tree.obj import ProcessTree
except Exception as e:
    raise ImportError(
        'pm4py is required for process-model quality metrics. Install with: pip install pm4py'
    ) from e

OUTPUT_DIR = Path('./output')
REAL_FEATURES_PARQUET = OUTPUT_DIR / 'case_step_features.parquet'
REAL_FEATURES_CSV = OUTPUT_DIR / 'case_step_features.csv'

SIMPY_TRACE_PATH = OUTPUT_DIR / 'sim_trace_table.csv'
PROM_INDUCTIVE_TRACE_PATH = OUTPUT_DIR / 'prom_inductive_sim_trace_table.csv'
PROM_HEURISTIC_TRACE_PATH = OUTPUT_DIR / 'prom_heuristic_sim_trace_table.csv'

PROCESS_QUALITY_PATH = OUTPUT_DIR / 'process_discovery_quality_comparison.csv'

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_source_df(source_name: str, trace_path: Path):
    if source_name == 'real':
        if REAL_FEATURES_PARQUET.exists():
            df = pd.read_parquet(REAL_FEATURES_PARQUET)
        elif REAL_FEATURES_CSV.exists():
            df = pd.read_csv(REAL_FEATURES_CSV)
        else:
            raise FileNotFoundError("Missing real features file.")
    else:
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing trace file: {trace_path}")
        df = pd.read_csv(trace_path)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    return df


def ensure_event_schema(df: pd.DataFrame) -> pd.DataFrame:
    req = {'case_id', 'activity', 'step_index', 'municipality'}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    x = df.copy()
    x = x.sort_values(['municipality', 'case_id', 'step_index']).reset_index(drop=True)

    if 'timestamp' in x.columns and x['timestamp'].notna().any():
        x['timestamp'] = pd.to_datetime(x['timestamp'], utc=True, errors='coerce')
    elif 'time_since_case_start_hours' in x.columns:
        t0 = pd.Timestamp('2020-01-01T00:00:00Z')
        hrs = pd.to_numeric(x['time_since_case_start_hours'], errors='coerce').fillna(0.0)
        x['timestamp'] = t0 + pd.to_timedelta(hrs, unit='h')
    else:
        order_in_case = x.groupby(['municipality', 'case_id']).cumcount()
        t0 = pd.Timestamp('2020-01-01T00:00:00Z')
        x['timestamp'] = t0 + pd.to_timedelta(order_in_case, unit='m')

    x['case_key'] = x['municipality'].astype(str) + '::' + x['case_id'].astype(str)
    x = x[['case_key', 'activity', 'timestamp']].rename(
        columns={
            'case_key': 'case:concept:name',
            'activity': 'concept:name',
            'timestamp': 'time:timestamp',
        }
    )

    x['concept:name'] = x['concept:name'].astype(str)
    x['case:concept:name'] = x['case:concept:name'].astype(str)
    x['time:timestamp'] = pd.to_datetime(x['time:timestamp'], utc=True, errors='coerce')
    x = x.dropna(subset=['case:concept:name', 'concept:name', 'time:timestamp'])
    return x


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_model(log_df: pd.DataFrame, miner_name: str):
    """Discovers a Petri net model using the specified miner.
    
    Ensures the return value is always a (net, im, fm) triplet, 
    converting from ProcessTree if necessary.
    """
    if miner_name == 'inductive':
        # High-level API usually returns (net, im, fm)
        res = pm4py.discover_petri_net_inductive(log_df)
    elif miner_name == 'heuristic':
        res = pm4py.discover_petri_net_heuristics(log_df)
    else:
        raise ValueError(f'Unknown miner: {miner_name}')

    # Robust check: If we got a single ProcessTree instead of a triplet
    if isinstance(res, ProcessTree):
        return pt_converter.apply(res)
    
    return res


# ---------------------------------------------------------------------------
# Evaluation – metrics run in parallel threads
# ---------------------------------------------------------------------------

def _replay_fitness(log_df, net, im, fm):
    """Token-replay pass + pm4py aggregate fitness (both need the full replay)."""
    tr = token_replay.apply(log_df, net, im, fm)
    n_traces = len(tr)
    fit_trace_count = sum(1 for r in tr if bool(r.get('trace_is_fit', False)))

    trace_fitness_vals = [float(r.get('trace_fitness', np.nan)) for r in tr]
    trace_fitness_vals = [v for v in trace_fitness_vals if np.isfinite(v)]

    avg_trace_fitness = float(np.mean(trace_fitness_vals)) if trace_fitness_vals else np.nan
    percentage_fit_traces = (100.0 * fit_trace_count / n_traces) if n_traces else np.nan

    fitness_dict = pm4py.fitness_token_based_replay(log_df, net, im, fm)
    log_fitness = float(fitness_dict.get('log_fitness', np.nan))

    return percentage_fit_traces, avg_trace_fitness, log_fitness


def _precision(log_df, net, im, fm):
    """Skipping precision as requested (too slow on this dataset)."""
    return np.nan


def _generalization(log_df, net, im, fm):
    """Skipping generalization as requested (too slow on this dataset)."""
    return np.nan


def _simplicity(net):
    return float(simplicity_evaluator.apply(net))


def evaluate_model(log_df: pd.DataFrame, net, im, fm) -> dict:
    """Run all five quality metrics concurrently using threads.

    pm4py's heavy lifting is C-extension code that releases the GIL, so
    ThreadPoolExecutor gives real concurrency here without pickling overhead.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as tex:
        f_fit    = tex.submit(_replay_fitness,   log_df, net, im, fm)
        f_prec   = tex.submit(_precision,        log_df, net, im, fm)
        f_gen    = tex.submit(_generalization,   log_df, net, im, fm)
        f_sim    = tex.submit(_simplicity,       net)

        percentage_fit_traces, avg_trace_fitness, log_fitness = f_fit.result()
        precision_val    = f_prec.result()
        generalization_val = f_gen.result()
        simplicity_val   = f_sim.result()

    return {
        'percentage_fit_traces': percentage_fit_traces,
        'average_trace_fitness': avg_trace_fitness,
        'log_fitness': log_fitness,
        'precision': precision_val,
        'generalization': generalization_val,
        'simplicity': simplicity_val,
    }


# ---------------------------------------------------------------------------
# Per-task worker (runs in a separate process)
# ---------------------------------------------------------------------------

def worker_evaluate(source_name: str, trace_path, miner_key: str, miner_label: str,
                    preloaded_log: pd.DataFrame | None = None):
    """Evaluate one (source, miner) combination."""
    if preloaded_log is not None:
        log_df = preloaded_log
    else:
        try:
            df_src = load_source_df(source_name, Path(trace_path) if trace_path else None)
        except FileNotFoundError:
            return {'source': source_name, 'miner': miner_label, 'error': 'trace file not found'}

        log_df = ensure_event_schema(df_src)

    if log_df.empty:
        return {'source': source_name, 'miner': miner_label, 'error': 'empty log after schema normalization'}

    # ------------------------------------------------------------------
    # SPEED OPTIMIZATION: Sample the log to a representative size
    # ------------------------------------------------------------------
    MAX_EVAL_CASES = 2000
    unique_cases = log_df['case:concept:name'].unique()
    if len(unique_cases) > MAX_EVAL_CASES:
        subset_cases = np.random.choice(unique_cases, size=MAX_EVAL_CASES, replace=False)
        log_df = log_df[log_df['case:concept:name'].isin(subset_cases)].copy()
        print(f"  [info] Sampled {source_name} log to {MAX_EVAL_CASES} cases for speed.")

    try:
        net, im, fm = discover_model(log_df, miner_key)
        metrics = evaluate_model(log_df, net, im, fm)

        result_str = (
            f"  {miner_label} on {source_name}: "
            f"fit_traces={metrics['percentage_fit_traces']:.2f}%, "
            f"avg_trace_fitness={metrics['average_trace_fitness']:.4f}, "
            f"log_fitness={metrics['log_fitness']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"generalization={metrics['generalization']:.4f}, "
            f"simplicity={metrics['simplicity']:.4f}"
        )
        print(result_str)

        return {
            'source': source_name,
            'miner': miner_label,
            **metrics,
        }
    except Exception as ex:
        return {'source': source_name, 'miner': miner_label, 'error': str(ex)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sources = [
        ('real',           None),
        ('simpy',          SIMPY_TRACE_PATH),
        ('prom_inductive', PROM_INDUCTIVE_TRACE_PATH),
        ('prom_heuristic', PROM_HEURISTIC_TRACE_PATH),
    ]

    MINER_MAP = {
        'inductive': 'Inductive Miner',
        'heuristic': 'Heuristic Miner',
    }

    # ------------------------------------------------------------------
    # Pre-load and normalise the real log ONCE in the main process so
    # we do not repeat disk I/O for every miner variant.
    # ------------------------------------------------------------------
    real_log_df: pd.DataFrame | None = None
    try:
        df_real = load_source_df('real', None)
        real_log_df = ensure_event_schema(df_real)
        print(f"Pre-loaded real log: {len(real_log_df)} events, "
              f"{real_log_df['case:concept:name'].nunique()} cases.")
    except FileNotFoundError as e:
        print(f"[warn] Could not pre-load real log: {e}. Will skip 'real' source tasks.")

    # Build task list; attach the pre-loaded log for 'real' tasks so
    # worker processes don't each re-read the parquet file.
    tasks = []
    for source_name, trace_path in sources:
        preloaded = real_log_df if source_name == 'real' else None
        if source_name == 'real' and real_log_df is None:
            # Skip — we already warned above
            continue
        for miner_key, miner_label in MINER_MAP.items():
            tasks.append((source_name, trace_path, miner_key, miner_label, preloaded))

    # Number of parallel processes: one per task, capped at CPU count.
    # Very large logs are memory-heavy, so cap at min(tasks, cpu_count).
    n_workers = min(len(tasks), os.cpu_count() or 4)
    print(f"\nStarting parallel evaluation — {len(tasks)} tasks across {n_workers} workers …")

    quality_rows = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(worker_evaluate, src, str(tp) if tp else None, mk, ml, pre): (src, ml)
            for src, tp, mk, ml, pre in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            src_name, m_label = futures[future]
            try:
                result = future.result()
                if 'error' in result:
                    print(f"  [skip] {m_label} on {src_name}: {result['error']}")
                else:
                    quality_rows.append(result)
            except Exception as exc:
                print(f"  [skip] {m_label} on {src_name} generated an exception: {exc}")

    if not quality_rows:
        print('No process-discovery quality results were produced.')
        return

    quality_df = pd.DataFrame(quality_rows)
    quality_df = quality_df.sort_values(['source', 'miner']).reset_index(drop=True)

    for c in [
        'percentage_fit_traces',
        'average_trace_fitness',
        'log_fitness',
        'precision',
        'generalization',
        'simplicity',
    ]:
        quality_df[c] = pd.to_numeric(quality_df[c], errors='coerce').round(4)

    quality_df.to_csv(PROCESS_QUALITY_PATH, index=False)
    print(f'\nSaved process quality comparison: {PROCESS_QUALITY_PATH.resolve()}')

    print('\n=== PRoM-style process discovery quality (all sources) ===')
    print(quality_df.to_string(index=False))

    real_only = quality_df[quality_df['source'] == 'real'].copy()
    if not real_only.empty:
        print('\n=== Real-log discovery quality (closest to PRoM table style) ===')
        print(
            real_only[
                [
                    'miner',
                    'percentage_fit_traces',
                    'average_trace_fitness',
                    'log_fitness',
                    'precision',
                    'generalization',
                    'simplicity',
                ]
            ].to_string(index=False)
        )


if __name__ == '__main__':
    main()
