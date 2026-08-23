#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Execute the full notebook pipeline (01–06) for one or more datasets.

Usage
-----
  # All datasets registered in dataset_registry.py:
  python run_pipeline.py

  # Specific dataset(s):
  python run_pipeline.py --datasets BPIC2012
  python run_pipeline.py --datasets BPIC2012 BPIC2017

  # Only run certain notebooks (1-indexed, matching filenames):
  python run_pipeline.py --datasets BPIC2012 --notebooks 1 2 3

  # Skip notebooks that already have output artefacts (re-run only what's missing):
  python run_pipeline.py --skip-existing

  # Dry run — show what would be executed without running anything:
  python run_pipeline.py --dry-run

  # Increase timeout per notebook (default: 3600 s = 1 h):
  python run_pipeline.py --timeout 7200

Options
-------
  --datasets      One or more dataset names from dataset_registry.REGISTRY.
                  Defaults to all registered datasets.
  --notebooks     Notebook numbers to run (1–6). Defaults to all.
  --skip-existing Skip a notebook/dataset pair if its primary output artefact
                  already exists (see SKIP_ARTEFACTS below).
  --dry-run       Print the execution plan without running anything.
  --timeout       Per-notebook execution timeout in seconds (default 3600).
  --output-dir    Root directory for output artefacts (default: ./output).
  --kernel        Jupyter kernel name to use (default: auto-detect from .venv).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parent
NOTEBOOKS   = REPO_ROOT / "notebooks"
OUTPUT_ROOT = REPO_ROOT / "output"

# Ordered list of notebooks. Each entry:
#   (number, filename, dataset_cell_pattern, primary_output_artefact_template)
# The artefact template uses {dataset} and {output_dir} — if the file exists
# and --skip-existing is set, the notebook is skipped.
PIPELINE: list[tuple[int, str, str]] = [
    (1, "01_data_ingestion.ipynb",      "events_{dataset}_train.parquet"),
    (2, "02_feature_engineering.ipynb", "terminal_classification.json"),
    (3, "03_digital_twin.ipynb",        "digital_twin_{dataset}_train.pkl"),
    (4, "04_rl_training.ipynb",         "rl_model/best_model.zip"),
    (5, "05_evaluation.ipynb",          "evaluation_all_models.csv"),
    (6, "06_insights.ipynb",            "routing_divergences.png"),
]

# Notebooks that are cross-dataset: they scan output/ for all models and only
# need to run once after all per-dataset notebooks complete.
# Notebook 05 (evaluation) iterates OUTPUT_ROOT for every trained model.
# Notebook 06 (insights) is per-dataset — it loads a specific DATASET's twin/model.
CROSS_DATASET_NOTEBOOKS = {5}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_kernel() -> str:
    """Return the kernel name for the .venv if it exists, else 'python3'."""
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        # Check if a kernel spec named 'python3' points to this venv
        try:
            result = subprocess.run(
                [str(venv_python), "-m", "ipykernel", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return "python3"
        except Exception:
            pass
    return "python3"


def _inject_dataset(nb: nbformat.NotebookNode, dataset: str) -> nbformat.NotebookNode:
    """
    Replace the DATASET = "..." line in the first cell that contains it.
    Also injects N_EPISODES = 2000 in notebook 06 (Fix 5 — larger sample).
    """
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        new_lines = []
        changed = False
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("DATASET") and "=" in stripped:
                # Preserve indentation
                indent = line[: len(line) - len(stripped)]
                new_lines.append(f'{indent}DATASET = "{dataset}"')
                changed = True
            elif stripped.startswith("N_EPISODES") and "=" in stripped:
                # Do NOT override N_EPISODES — notebook 06 sets it adaptively
                # based on dataset size to avoid OOM on large datasets.
                new_lines.append(line)
            else:
                new_lines.append(line)
        if changed:
            cell.source = "\n".join(new_lines)
    return nb


def _artefact_exists(nb_num: int, dataset: str, template: str) -> bool:
    """Return True if the primary output artefact for this notebook/dataset exists."""
    if nb_num in CROSS_DATASET_NOTEBOOKS:
        path = OUTPUT_ROOT / template
    else:
        path = OUTPUT_ROOT / dataset / template.format(dataset=dataset)
    return path.exists()


def _run_notebook(
    nb_path: Path,
    dataset: str,
    kernel: str,
    timeout: int,
    dry_run: bool,
) -> bool:
    """
    Execute a notebook with the DATASET variable injected.
    Returns True on success, False on failure.
    """
    label = f"[{dataset}] {nb_path.name}"

    if dry_run:
        print(f"  DRY-RUN  {label}")
        return True

    print(f"\n{'='*70}")
    print(f"  RUNNING  {label}")
    print(f"{'='*70}")
    t0 = time.time()

    # Load notebook
    nb = nbformat.read(nb_path, as_version=4)

    # Inject dataset parameter
    nb = _inject_dataset(nb, dataset)

    # Execute
    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name=kernel,
    )

    try:
        ep.preprocess(nb, {"metadata": {"path": str(REPO_ROOT)}})
    except CellExecutionError as exc:
        elapsed = time.time() - t0
        print(f"\n  FAILED   {label}  ({elapsed:.0f}s)")
        print(f"  Error: {exc.ename}: {exc.evalue}")
        # Save the partially-executed notebook for debugging
        err_path = nb_path.with_suffix(f".{dataset}.ERROR.ipynb")
        nbformat.write(nb, err_path)
        print(f"  Partial output saved to: {err_path}")
        return False
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  FAILED   {label}  ({elapsed:.0f}s)")
        print(f"  Unexpected error: {exc}")
        return False

    elapsed = time.time() - t0
    print(f"\n  DONE     {label}  ({elapsed:.0f}s)")

    # Save executed notebook alongside the original (overwrite)
    nbformat.write(nb, nb_path)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full notebook pipeline for one or more datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="DATASET",
        help="Dataset name(s) from dataset_registry. Default: all registered.",
    )
    parser.add_argument(
        "--notebooks", nargs="+", type=int, metavar="N",
        help="Notebook numbers to run (1–6). Default: all.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip notebook if its primary output artefact already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print execution plan without running anything.",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Per-notebook timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_ROOT,
        help="Root output directory (default: ./output).",
    )
    parser.add_argument(
        "--kernel", default=None,
        help="Jupyter kernel name (default: auto-detect from .venv).",
    )
    args = parser.parse_args()

    # ── Resolve datasets ───────────────────────────────────────────────────
    # Import here so the script works even if src/ isn't on PYTHONPATH yet
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from dataset_registry import REGISTRY

    if args.datasets:
        unknown = [d for d in args.datasets if d not in REGISTRY]
        if unknown:
            print(f"ERROR: Unknown dataset(s): {unknown}")
            print(f"Available: {list(REGISTRY.keys())}")
            sys.exit(1)
        datasets = args.datasets
    else:
        # Default: all registered datasets EXCEPT per-municipality sub-entries.
        # BPIC2015_M1..M5 are research entries for per-municipality comparison;
        # the pooled BPIC2015 entry covers the general case.
        # Users can still run them explicitly: --datasets BPIC2015_M1 BPIC2015_M2 ...
        _SKIP_BY_DEFAULT = {"BPIC2015_M1", "BPIC2015_M2", "BPIC2015_M3",
                            "BPIC2015_M4", "BPIC2015_M5", "BPIC2012W"}
        datasets = [d for d in REGISTRY if d not in _SKIP_BY_DEFAULT]

    # ── Resolve notebooks ──────────────────────────────────────────────────
    nb_filter = set(args.notebooks) if args.notebooks else {nb[0] for nb in PIPELINE}
    pipeline  = [(n, f, a) for n, f, a in PIPELINE if n in nb_filter]

    if not pipeline:
        print("ERROR: No notebooks selected.")
        sys.exit(1)

    # ── Kernel ─────────────────────────────────────────────────────────────
    kernel = args.kernel or _detect_kernel()

    # ── Build execution plan ───────────────────────────────────────────────
    # Cross-dataset notebooks (5, 6) run once after all per-dataset notebooks.
    per_dataset_steps = [
        (nb_num, nb_file, artefact, ds)
        for ds in datasets
        for nb_num, nb_file, artefact in pipeline
        if nb_num not in CROSS_DATASET_NOTEBOOKS
    ]
    cross_dataset_steps = [
        (nb_num, nb_file, artefact, datasets[0])   # dataset arg is ignored for these
        for nb_num, nb_file, artefact in pipeline
        if nb_num in CROSS_DATASET_NOTEBOOKS
    ]
    # Deduplicate cross-dataset steps (only run each once)
    seen_cross = set()
    deduped_cross = []
    for step in cross_dataset_steps:
        key = step[0]
        if key not in seen_cross:
            seen_cross.add(key)
            deduped_cross.append(step)

    all_steps = per_dataset_steps + deduped_cross

    # ── Print plan ─────────────────────────────────────────────────────────
    print(f"\nPipeline execution plan")
    print(f"  Datasets:  {datasets}")
    print(f"  Notebooks: {[nb[0] for nb in pipeline]}")
    print(f"  Kernel:    {kernel}")
    print(f"  Timeout:   {args.timeout}s per notebook")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry run:   {args.dry_run}")
    print()

    skipped = []
    planned = []
    for nb_num, nb_file, artefact, ds in all_steps:
        nb_path = NOTEBOOKS / nb_file
        if not nb_path.exists():
            print(f"  WARNING: notebook not found: {nb_path}")
            continue
        if args.skip_existing and _artefact_exists(nb_num, ds, artefact):
            label = f"[{ds}] {nb_file}" if nb_num not in CROSS_DATASET_NOTEBOOKS else f"[all] {nb_file}"
            skipped.append(label)
            print(f"  SKIP     {label}  (artefact exists)")
        else:
            planned.append((nb_num, nb_file, artefact, ds, nb_path))

    if skipped:
        print(f"\n  {len(skipped)} step(s) skipped.")

    if not planned:
        print("\nNothing to run.")
        return

    print(f"\n  {len(planned)} step(s) to run:")
    for nb_num, nb_file, _, ds, _ in planned:
        label = f"[{ds}] {nb_file}" if nb_num not in CROSS_DATASET_NOTEBOOKS else f"[all] {nb_file}"
        print(f"    {label}")

    if args.dry_run:
        print("\nDry run complete — nothing executed.")
        return

    # ── Execute ────────────────────────────────────────────────────────────
    print()
    results: list[tuple[str, bool]] = []
    t_total = time.time()

    for nb_num, nb_file, _, ds, nb_path in planned:
        label = f"[{ds}] {nb_file}" if nb_num not in CROSS_DATASET_NOTEBOOKS else f"[all] {nb_file}"
        ok = _run_notebook(nb_path, ds, kernel, args.timeout, dry_run=False)
        results.append((label, ok))
        if not ok:
            print(f"\n  Pipeline halted after failure in: {label}")
            print("  Fix the error and re-run. Use --skip-existing to resume from this point.")
            break

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    passed = sum(1 for _, ok in results if ok)
    failed = sum(1 for _, ok in results if not ok)

    print(f"\n{'='*70}")
    print(f"  SUMMARY  {passed} passed  {failed} failed  ({elapsed_total:.0f}s total)")
    print(f"{'='*70}")
    for label, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {label}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
