"""
env_factory.py
--------------
One way to build the training/evaluation environment.

Why this module exists
----------------------
``ProcessEnv`` used to be constructed by hand in three notebooks — 04
(training), 05 (evaluation) and 06 (analysis). Each copy repeated the same
dozen lines: load the twin, load the embedder, read the terminal labels,
derive ``embed_dim`` and ``max_steps``. Because they were copies, they drifted:
notebook 04 additionally applied the tuned reward weights and the other two did
not, so the agent was graded under a reward it was never trained on.

Any fix that only edits the notebooks would let the same drift happen again.
Instead there is now a single ``build_process_env()`` that all three call, and
``config_used.json`` records what it produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from reward_config import RewardConfig
from rl_env import ProcessEnv

#: Written next to the results so a run's exact setup is recoverable.
RUN_CONFIG_FILENAME = "config_used.json"


def _max_steps_for(median_trace_length: float) -> int:
    """Episode cap: enough room for a full-length trace plus rework."""
    return max(150, int(median_trace_length * 4))


def build_process_env(
    output_dir: "str | Path",
    reward_config: "RewardConfig | None" = None,
    n_resources: int = 20,
    seed: int = 42,
    verdict_mode: str = "environment",
) -> tuple[ProcessEnv, dict]:
    """
    Build the environment for one dataset directory.

    Parameters
    ----------
    output_dir
        ``output/<DATASET>/`` — must contain the fitted twin, the activity
        embedder and ``terminal_classification.json``.
    reward_config
        Defaults to ``reward_config.json`` in `output_dir` if present,
        otherwise the shared :data:`reward_config.DEFAULT`. Pass one
        explicitly only to run an ablation.
    seed
        Environment RNG seed. Note this is *not* the training seed — that
        belongs to the learning algorithm.

    Returns
    -------
    (env, meta) where `meta` is the resolved configuration, suitable for
    writing out with :func:`save_run_config`.
    """
    output_dir = Path(output_dir)
    dataset = output_dir.name

    twin_path = output_dir / f"digital_twin_{dataset}_train.pkl"
    emb_path = output_dir / "activity_embeddings.model"
    tc_path = output_dir / "terminal_classification.json"

    for path in (twin_path, emb_path, tc_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} missing from {output_dir}. "
                f"Run notebooks 1-3 for {dataset} first."
            )

    twin = joblib.load(twin_path)
    embedder = joblib.load(emb_path)
    with open(tc_path, encoding="utf-8") as fh:
        terminals = json.load(fh)

    good_terminals = set(terminals.get("good_terminals", []))
    bad_terminals = set(terminals.get("bad_terminals", []))
    if not good_terminals:
        raise ValueError(
            f"{tc_path} lists no good terminals. The agent would have no "
            f"reachable positive outcome. Re-run notebook 02 for {dataset}."
        )

    if "min_steps_to_outcome" not in terminals:
        raise ValueError(
            f"{tc_path} has no 'min_steps_to_outcome'. It was written before "
            f"the outcome floor existed. Re-run notebook 02 for {dataset} — "
            f"without the floor the agent can reach an outcome faster than any "
            f"real case does."
        )
    min_steps = int(terminals["min_steps_to_outcome"])

    if verdict_mode == "environment" and "outcome_base_rates" not in terminals:
        raise ValueError(
            f"{tc_path} has no 'outcome_base_rates'. Re-run notebook 02 for "
            f"{dataset}. Without them the environment cannot draw a verdict "
            f"at the real base rate, and the agent would be able to route to "
            f"a good outcome itself."
        )
    rates = terminals.get("outcome_base_rates", {})

    if reward_config is None:
        reward_config = RewardConfig.load(output_dir) or RewardConfig()

    median_len = twin.kpi_baselines.get("median_trace_length", 20)
    embed_dim = embedder.vector_size
    max_steps = _max_steps_for(median_len)

    env = ProcessEnv(
        twin=twin,
        embed_model=embedder,
        kpi_baselines=twin.kpi_baselines,
        n_resources=n_resources,
        embed_dim=embed_dim,
        max_steps=max_steps,
        seed=seed,
        bad_terminals=bad_terminals,
        good_terminals=good_terminals,
        reward_config=reward_config,
        min_steps_to_outcome=min_steps,
        verdict_mode=verdict_mode,
        outcome_base_rates=rates,
        real_median_cycle_s=(
            terminals["real_cycle_days_median"] * 86400.0
            if terminals.get("real_cycle_days_median") else None
        ),
    )

    meta = {
        "dataset": dataset,
        "n_activities": len(twin.activities),
        "median_trace_length": float(median_len),
        "embed_dim": int(embed_dim),
        "max_steps": int(max_steps),
        "n_resources": int(n_resources),
        "env_seed": int(seed),
        "min_steps_to_outcome": min_steps,
        "steps_to_outcome_real": terminals.get("steps_to_outcome", {}),
        "verdict_mode": verdict_mode,
        "outcome_base_rate_p_good": rates.get("p_good"),
        "real_cycle_days_median": terminals.get("real_cycle_days_median"),
        "good_terminals": sorted(good_terminals),
        "bad_terminals": sorted(bad_terminals),
        "reward_config": reward_config.to_dict(),
        # The shares above are identical for every dataset; these are what
        # they resolve to for this one's trace length.
        "resolved_per_step_weights": {
            "w_progress": env.w_progress,
            "w_step": env.w_step,
        },
    }
    return env, meta


def save_run_config(output_dir: "str | Path", meta: dict, **extra) -> Path:
    """
    Write ``config_used.json`` beside the results.

    `extra` is merged in, for anything the caller knows and the factory does
    not — training seed, algorithm hyperparameters, number of timesteps.
    """
    path = Path(output_dir) / RUN_CONFIG_FILENAME
    payload = {**meta, **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path
