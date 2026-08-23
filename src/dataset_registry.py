"""
dataset_registry.py
-------------------
Registry of known datasets with their XES paths and metadata.
New datasets can be added here or passed directly via config — nothing is hardcoded
in the notebooks.

Each entry describes:
  - files             : list of XES file paths (relative to REPO_ROOT)
  - split_key         : column used to split into train/validation subsets
  - train_tags        : which split_key values to use for training the twin
  - description       : human-readable description
  - domain            : used to decide whether cross-dataset validation is
                        same-domain or cross-domain
  - subprocess_filter : optional dict that extracts a sub-process from the
                        full event log before fitting the twin.  Keys:
                          activity_prefix  (str)  keep only activities whose
                                                   name starts with this prefix
                          lifecycle        (str)  keep only events with this
                                                   lifecycle value (e.g. 'COMPLETE')
                        When set, the filter is applied by apply_subprocess_filter()
                        after loading the XES file.  The output directory is
                        automatically suffixed with the filter key so artefacts
                        from the full log and the sub-process don't collide.
  - municipality_files: optional dict {tag: xes_path} for multi-municipality
                        datasets (BPIC2015).  When set, notebook 01 adds a
                        'municipality' column to the DataFrame so the twin and
                        evaluation can distinguish between sub-populations.
                        Use BPIC2015 (pooled) for a single general model, or
                        BPIC2015_M1..M5 for per-municipality models.

        Example — BPIC2012 W subprocess:
            subprocess_filter={
                "activity_prefix": "W_",
                "lifecycle": "COMPLETE",
            }
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Repo root = the directory that contains the src/ folder this file lives in.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# XES paths for each BPIC2015 municipality (relative to REPO_ROOT)
_BPIC2015_FILES = {
    "M1": "datasets/BPIC-2015/M1/BPIC15_1.xes",
    "M2": "datasets/BPIC-2015/M2/BPIC15_2.xes",
    "M3": "datasets/BPIC-2015/M3/BPIC15_3.xes",
    "M4": "datasets/BPIC-2015/M4/BPIC15_4.xes",
    "M5": "datasets/BPIC-2015/M5/BPIC15_5.xes",
}


@dataclass
class DatasetConfig:
    name: str                              # short identifier, e.g. 'BPIC2015'
    files: list[str]                       # XES paths relative to REPO_ROOT
    train_tags: list[str]                  # dataset tag values used for twin training
    description: str = ""
    domain: str = "unknown"               # e.g. 'permit', 'loan', 'healthcare'
    max_traces: Optional[int] = None      # None = load all
    split_key: str = "dataset"            # column that identifies the sub-split
    subprocess_filter: Optional[dict] = None  # see module docstring
    # Optional: {tag: xes_path} for multi-municipality datasets.
    # When set, a 'municipality' column is added during ingestion.
    municipality_files: Optional[dict] = None


# ---------------------------------------------------------------------------
# Known dataset registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, DatasetConfig] = {

    # ── BPIC2015: pooled across all 5 municipalities ──────────────────────
    # All 5 XES files are concatenated with a 'municipality' tag column added.
    # The twin is fitted on the pooled data, giving a general permit-process
    # model that covers all municipalities.
    #
    # Activity overlap between municipalities is high (Jaccard ≈ 0.79–0.88),
    # so the pooled twin is a reasonable approximation of each individual one.
    # Use BPIC2015_M1..M5 below for per-municipality models.
    "BPIC2015": DatasetConfig(
        name="BPIC2015",
        files=list(_BPIC2015_FILES.values()),
        train_tags=list(_BPIC2015_FILES.keys()),  # all municipalities
        description=(
            "Dutch municipal permit application process — pooled across all 5 "
            "municipalities (M1–M5). Activity overlap Jaccard ≈ 0.79–0.88. "
            "Use BPIC2015_M1..M5 for per-municipality models."
        ),
        domain="permit",
        max_traces=None,
        municipality_files=_BPIC2015_FILES,
    ),

    # ── BPIC2015 per-municipality entries ─────────────────────────────────
    # Each entry trains a twin on a single municipality only.
    # Useful for comparing routing policies across municipalities.
    "BPIC2015_M1": DatasetConfig(
        name="BPIC2015_M1",
        files=[_BPIC2015_FILES["M1"]],
        train_tags=["M1"],
        description="BPIC2015 — Municipality 1 only (1,199 cases, 289 activities).",
        domain="permit",
        municipality_files={"M1": _BPIC2015_FILES["M1"]},
    ),
    "BPIC2015_M2": DatasetConfig(
        name="BPIC2015_M2",
        files=[_BPIC2015_FILES["M2"]],
        train_tags=["M2"],
        description="BPIC2015 — Municipality 2 only (832 cases, 304 activities).",
        domain="permit",
        municipality_files={"M2": _BPIC2015_FILES["M2"]},
    ),
    "BPIC2015_M3": DatasetConfig(
        name="BPIC2015_M3",
        files=[_BPIC2015_FILES["M3"]],
        train_tags=["M3"],
        description="BPIC2015 — Municipality 3 only (1,409 cases, 277 activities).",
        domain="permit",
        municipality_files={"M3": _BPIC2015_FILES["M3"]},
    ),
    "BPIC2015_M4": DatasetConfig(
        name="BPIC2015_M4",
        files=[_BPIC2015_FILES["M4"]],
        train_tags=["M4"],
        description="BPIC2015 — Municipality 4 only (1,053 cases, 272 activities).",
        domain="permit",
        municipality_files={"M4": _BPIC2015_FILES["M4"]},
    ),
    "BPIC2015_M5": DatasetConfig(
        name="BPIC2015_M5",
        files=[_BPIC2015_FILES["M5"]],
        train_tags=["M5"],
        description="BPIC2015 — Municipality 5 only (1,156 cases, 285 activities).",
        domain="permit",
        municipality_files={"M5": _BPIC2015_FILES["M5"]},
    ),

    "BPIC2017": DatasetConfig(
        name="BPIC2017",
        files=[
            "datasets/BPIC-2017/BPI Challenge 2017.xes",
        ],
        train_tags=["BPI Challenge 2017"],
        description="Dutch financial institute loan application process",
        domain="loan",
        max_traces=None,
    ),

    "BPIC2012": DatasetConfig(
        name="BPIC2012",
        files=[
            "datasets/BPIC-2012/BPI_Challenge_2012.xes",
        ],
        train_tags=["BPI_Challenge_2012"],
        description="Dutch financial institute loan application process (2012 edition)",
        domain="loan",
        max_traces=None,
        subprocess_filter=None,  # full log — all A_, O_, W_ activities
    ),

    # BPIC2012W: W_ subprocess only, COMPLETE lifecycle events.
    # This is the exact sub-process used by RIMS_DRL (BPI Challenge 2012 W).
    "BPIC2012W": DatasetConfig(
        name="BPIC2012W",
        files=[
            "datasets/BPIC-2012/BPI_Challenge_2012.xes",
        ],
        train_tags=["BPI_Challenge_2012"],
        description=(
            "BPI Challenge 2012 — W subprocess only (work-item activities, "
            "COMPLETE lifecycle)."
        ),
        domain="loan",
        max_traces=None,
        subprocess_filter={
            "activity_prefix": "W_",   # keep only W_* activities
            "lifecycle": "COMPLETE",   # keep only COMPLETE lifecycle events
        },
    ),

}


# ---------------------------------------------------------------------------
# Subprocess filter
# ---------------------------------------------------------------------------

def apply_subprocess_filter(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """
    Apply config.subprocess_filter to a loaded event log DataFrame.

    Filters rows by activity prefix and/or lifecycle value, then drops cases
    that become empty after filtering.  The case_id column is preserved so
    the filtered DataFrame can still be grouped by case.

    Args:
        df:     Full event log DataFrame (output of parse_xes).
        config: DatasetConfig — subprocess_filter is read from here.

    Returns:
        Filtered DataFrame, or the original df if subprocess_filter is None.
    """
    filt = config.subprocess_filter
    if not filt:
        return df

    mask = pd.Series(True, index=df.index)

    prefix = filt.get("activity_prefix")
    if prefix:
        mask &= df["activity"].str.startswith(prefix)

    lifecycle = filt.get("lifecycle")
    if lifecycle and "lifecycle" in df.columns:
        mask &= df["lifecycle"] == lifecycle

    filtered = df[mask].copy()

    # Drop cases that have fewer than 2 events after filtering (degenerate traces)
    case_sizes = filtered.groupby("case_id")["activity"].transform("count")
    filtered = filtered[case_sizes >= 2].reset_index(drop=True)

    n_before = df["case_id"].nunique()
    n_after  = filtered["case_id"].nunique()
    print(
        f"  subprocess_filter applied: "
        f"prefix={prefix!r}  lifecycle={lifecycle!r}  "
        f"cases {n_before} → {n_after}  "
        f"events {len(df)} → {len(filtered)}"
    )
    return filtered


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------

def get_config(name: str) -> DatasetConfig:
    """Retrieve a registered dataset config by name, with paths resolved to absolute."""
    if name not in REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not found in registry.\n"
            f"Available: {list(REGISTRY.keys())}\n"
            f"To add a new dataset, call register_dataset() or edit REGISTRY directly."
        )
    cfg = REGISTRY[name]
    # Resolve relative paths against the repo root so notebooks work regardless
    # of the kernel's working directory.
    resolved_files = [
        str(_REPO_ROOT / f) if not Path(f).is_absolute() else f
        for f in cfg.files
    ]
    return DatasetConfig(
        name=cfg.name,
        files=resolved_files,
        train_tags=cfg.train_tags,
        description=cfg.description,
        domain=cfg.domain,
        max_traces=cfg.max_traces,
        split_key=cfg.split_key,
        subprocess_filter=cfg.subprocess_filter,
        municipality_files={
            tag: str(_REPO_ROOT / path) if not Path(path).is_absolute() else path
            for tag, path in cfg.municipality_files.items()
        } if cfg.municipality_files else None,
    )


def register_dataset(config: DatasetConfig):
    """Register a new dataset at runtime."""
    REGISTRY[config.name] = config
    print(f"Registered dataset: '{config.name}'")


def list_datasets() -> None:
    """Print all registered datasets."""
    print(f"{'Name':<15} {'Domain':<12} {'Filter':<20} {'Files':<5} {'Train tags'}")
    print("-" * 75)
    for name, cfg in REGISTRY.items():
        filt = cfg.subprocess_filter
        filt_str = (
            f"{filt.get('activity_prefix','*')}/{filt.get('lifecycle','*')}"
            if filt else "—"
        )
        print(
            f"{name:<15} {cfg.domain:<12} {filt_str:<20} {len(cfg.files):<5} "
            f"{str(cfg.train_tags)}"
        )
