#!/usr/bin/env python3
"""
run_experiments.py
------------------
Run the multi-run studies the single headline run cannot answer.

Two studies, both from the same runner:

``seeds``
    Train the same configuration under several training seeds and report the
    across-seed spread. Every number in this project came from one run; until
    the spread is known, no difference between policies is defensible (A4).

``effect-scale``
    Sweep ``RewardConfig.effect_scale``, the global multiplier on the
    intervention effect sizes in ``src/intervention_effects.py``. Those effect
    sizes are assumptions the logs cannot support — congestion correlations are
    near zero and mostly negative, and BPIC2012/BPIC2017 record no
    interventions at all — so nothing about the catalogue can be claimed
    without showing which conclusions survive changing them. ``0.0`` disables
    the mechanism entirely and is the management-action ablation (A3) for free.

Usage
-----
    # Run-to-run spread, five seeds, all datasets
    python run_experiments.py seeds --seeds 0 1 2 3 4

    # Sensitivity to the assumed effect sizes
    python run_experiments.py effect-scale --values 0.0 0.5 1.0 1.5 2.0

    # Both, three seeds each, four at a time
    python run_experiments.py seeds effect-scale --seeds 0 1 2 --jobs 4

    # What would run, without running it
    python run_experiments.py seeds --dry-run

Results go to ``results/sweep/`` — never to ``output/``, so a sweep cannot
overwrite the headline run. Each run writes its own ``result.json`` and
training log; the studies share the runs they have in common and write
``summary.csv``, ``seed_spread.csv`` and ``effect_scale.csv`` across them.

Re-running skips any run whose ``result.json`` already exists unless
``--force`` is given, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
OUTPUT = REPO / "output"

STUDIES = ("seeds", "effect-scale")

#: Both studies write here. They share runs — see :func:`_plan`.
STUDY_DIR = "sweep"

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_EFFECT_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)

#: Columns lifted out of each run's metrics into the study summary, in the
#: order the handoff table reports them.
SUMMARY_COLUMNS = (
    "mean_length", "real_median_length", "length_ratio",
    "median_cycle_days", "real_median_cycle_days", "cycle_ratio",
    "cycle_log2_deviation",
    "good_rate_of_concluded", "real_good_rate",
    "truncated_rate", "no_op_rate", "rule_waiving_rate",
    "self_loop_rate", "real_self_loop_rate",
    "mean_reward", "std_reward",
)


def _datasets(requested: "list[str] | None") -> list[str]:
    from dataset_registry import REGISTRY

    available = [
        d for d in sorted(REGISTRY)
        if (OUTPUT / d / "terminal_classification.json").exists()
    ]
    if not requested:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise SystemExit(
            f"Not built yet: {', '.join(missing)}. "
            f"Run notebooks 1-3 for them first. Available: {', '.join(available)}"
        )
    return list(requested)


def _plan(studies, datasets, seeds, sweep_seeds, effect_scales) -> list[dict]:
    """
    Every (dataset, seed, effect_scale) run the requested studies need.

    The two studies overlap — the seed study is the effect-scale sweep at
    ``effect_scale = 1.0`` — so runs are keyed by their output directory and
    the union is taken. Asking for both costs nine runs less than running them
    separately, and guarantees the shared cell is literally the same run rather
    than two runs that ought to agree.
    """
    combos: set[tuple[int, float]] = set()
    if "seeds" in studies:
        combos |= {(int(s), 1.0) for s in seeds}
    if "effect-scale" in studies:
        # Sweeping at a single seed cannot separate a real trend from seed
        # noise, so each value carries several seeds and the summary reports
        # the spread at each one.
        combos |= {
            (int(s), float(v))
            for s, v in itertools.product(sweep_seeds, effect_scales)
        }

    jobs = []
    for dataset, (seed, scale) in itertools.product(datasets, sorted(combos)):
        jobs.append({
            "dataset": dataset,
            "seed": seed,
            "effect_scale": scale,
            "run_dir": RESULTS / STUDY_DIR / dataset / f"scale{scale:g}_seed{seed}",
        })
    return jobs


def _run_one(job: dict, n_eval_episodes: int, max_steps: int, threads: int) -> dict:
    """Worker body: train and measure one run. Runs in its own process."""
    # One BLAS thread per worker. Torch otherwise grabs every core in each
    # process, and N workers then thrash rather than run N times faster.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)

    sys.path.insert(0, str(REPO / "src"))
    import torch

    torch.set_num_threads(threads)

    from experiment import run_experiment
    from reward_config import RewardConfig

    started = time.time()
    record = run_experiment(
        dataset_dir=OUTPUT / job["dataset"],
        reward_config=RewardConfig(effect_scale=job["effect_scale"]),
        seed=job["seed"],
        run_dir=job["run_dir"],
        n_eval_episodes=n_eval_episodes,
        max_steps=max_steps,
        verbose=0,
    )
    record["effect_scale"] = job["effect_scale"]
    record["wall_seconds"] = round(time.time() - started, 1)
    with open(job["run_dir"] / "result.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    return record


def _collect() -> "pd.DataFrame | None":                       # noqa: F821
    """Every finished run under ``results/sweep/`` as one table."""
    import pandas as pd

    rows = []
    for path in sorted((RESULTS / STUDY_DIR).glob("*/*/result.json")):
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        metrics = record.get("metrics", {})
        rows.append({
            "dataset": record.get("dataset"),
            "effect_scale": record.get(
                "effect_scale", record["reward_config"]["effect_scale"]),
            "seed": record.get("seed"),
            "timesteps": record.get("training", {}).get("timesteps_run"),
            **{c: metrics.get(c) for c in SUMMARY_COLUMNS},
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(["dataset", "effect_scale", "seed"])


def _fmt(v) -> str:
    return "" if v is None or v != v else f"{v:.3f}"


def _report_seeds(table) -> None:
    """
    A4: how much does the same configuration move between training seeds?

    Reported as mean +/- standard deviation across seeds at the default
    ``effect_scale = 1.0``. Any claimed difference between two policies has to
    clear this to mean anything.
    """
    at_default = table[table["effect_scale"] == 1.0]
    if at_default.empty:
        print("\nNo runs at effect_scale = 1.0 — nothing to say about seed spread.")
        return

    metrics = ["mean_length", "cycle_ratio", "good_rate_of_concluded",
               "no_op_rate", "rule_waiving_rate", "self_loop_rate", "mean_reward"]
    stats = (at_default.groupby("dataset")[metrics]
             .agg(["mean", "std", "count"]))

    print("\n=== A4: across-seed spread at effect_scale = 1.0 ===")
    for dataset, row in stats.iterrows():
        n = int(row[(metrics[0], "count")])
        print(f"\n{dataset}  ({n} seed{'s' if n != 1 else ''})")
        for m in metrics:
            mean, sd = row[(m, "mean")], row[(m, "std")]
            spread = "" if sd != sd else f" +/- {sd:.3f}"
            print(f"  {m:<24s} {_fmt(mean):>8s}{spread}")

    path = RESULTS / STUDY_DIR / "seed_spread.csv"
    stats.to_csv(path)
    print(f"\nWritten: {path}")


def _report_effect_scale(table) -> None:
    """
    Which conclusions survive changing the assumed intervention effect sizes?

    Every effect size in ``intervention_effects.py`` is an assumption, so a
    conclusion that only holds at ``effect_scale = 1.0`` is a conclusion about
    the assumption. ``0.0`` is the management-action ablation.
    """
    metrics = ["mean_length", "cycle_ratio", "good_rate_of_concluded",
               "no_op_rate", "rule_waiving_rate", "self_loop_rate", "mean_reward"]
    stats = (table.groupby(["dataset", "effect_scale"])[metrics]
             .agg(["mean", "std", "count"]))

    print("\n=== effect_scale sensitivity (mean across seeds) ===")
    for dataset in table["dataset"].unique():
        block = stats.loc[dataset]
        print(f"\n{dataset}")
        header = "  scale  n  " + "".join(f"{m[:14]:>16s}" for m in metrics)
        print(header)
        for scale, row in block.iterrows():
            n = int(row[(metrics[0], "count")])
            cells = "".join(f"{_fmt(row[(m, 'mean')]):>16s}" for m in metrics)
            print(f"  {scale:<5g}  {n}  {cells}")

    path = RESULTS / STUDY_DIR / "effect_scale.csv"
    stats.to_csv(path)
    print(f"\nWritten: {path}")


def _write_summary(studies) -> "Path | None":
    table = _collect()
    if table is None:
        return None

    path = RESULTS / STUDY_DIR / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"\nAll runs: {path}")

    if "seeds" in studies:
        _report_seeds(table)
    if "effect-scale" in studies:
        _report_effect_scale(table)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("studies", nargs="+", choices=STUDIES,
                        help="Which study or studies to run.")
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="Defaults to every dataset already built.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                        help=f"Training seeds (default {list(DEFAULT_SEEDS)}).")
    parser.add_argument("--values", nargs="+", type=float,
                        default=list(DEFAULT_EFFECT_SCALES),
                        help=f"effect_scale values (default {list(DEFAULT_EFFECT_SCALES)}).")
    parser.add_argument("--sweep-seeds", nargs="+", type=int, metavar="N",
                        help="Seeds run at every effect_scale value. Defaults to "
                             "the first three of --seeds; the full seed list is "
                             "still run at effect_scale = 1.0.")
    parser.add_argument("--jobs", type=int, default=3,
                        help="Runs in parallel (default 3).")
    parser.add_argument("--threads", type=int, default=2,
                        help="BLAS threads per run (default 2).")
    parser.add_argument("--max-steps", type=int, default=500_000,
                        help="Training step cap; early stopping usually wins.")
    parser.add_argument("--eval-episodes", type=int, default=300,
                        help="Episodes rolled out to measure each policy.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run runs that already have a result.json.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Rebuild summary.csv from existing results and exit.")
    args = parser.parse_args()

    studies = list(dict.fromkeys(args.studies))

    sweep_seeds = args.sweep_seeds or args.seeds[:3]

    if args.summary_only:
        if _write_summary(studies) is None:
            print(f"No results under {RESULTS / STUDY_DIR}")
        return 0

    datasets = _datasets(args.datasets)
    jobs = _plan(studies, datasets, args.seeds, sweep_seeds, args.values)

    done = [j for j in jobs if (j["run_dir"] / "result.json").exists()]
    todo = jobs if args.force else [j for j in jobs if j not in done]

    print(f"Studies:  {', '.join(studies)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Seeds:    {args.seeds}")
    if "effect-scale" in studies:
        print(f"Scales:   {args.values}  at seeds {sweep_seeds}")
    print(f"Runs:     {len(jobs)} planned, {len(done)} already done, "
          f"{len(todo)} to run  ({args.jobs} at a time)")
    for job in todo:
        print(f"    {job['dataset']}  seed={job['seed']}  "
              f"effect_scale={job['effect_scale']:g}")

    if args.dry_run:
        print("\nDry run — nothing executed.")
        return 0
    if not todo:
        print("\nNothing to run.")
        _write_summary(studies)
        return 0

    started = time.time()
    failures = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_run_one, job, args.eval_episodes,
                        args.max_steps, args.threads): job
            for job in todo
        }
        for i, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            tag = (f"{job['dataset']} seed={job['seed']} "
                   f"scale={job['effect_scale']:g}")
            try:
                record = future.result()
            except Exception as exc:                      # noqa: BLE001
                failures.append((tag, repr(exc)))
                print(f"  {i}/{len(todo)}  FAILED  {tag}: {exc}")
                continue
            m = record["metrics"]
            print(f"  {i}/{len(todo)}  {tag}  "
                  f"len={m['mean_length']:.1f}  "
                  f"cycle_ratio={m['cycle_ratio'] if m['cycle_ratio'] is None else round(m['cycle_ratio'], 2)}  "
                  f"good={m['good_rate_of_concluded']:.3f}  "
                  f"reward={m['mean_reward']:+.2f}  "
                  f"({record['wall_seconds'] / 60:.1f} min)")

    print(f"\nFinished in {(time.time() - started) / 60:.1f} min.")
    _write_summary(studies)

    if failures:
        print("\nFAILED runs:")
        for tag, err in failures:
            print(f"  {tag}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
