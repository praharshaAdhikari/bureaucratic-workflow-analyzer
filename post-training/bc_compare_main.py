
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("output")
PPO_METRICS_PATH = OUTPUT_DIR / "live_progress" / "ppo_metrics_live.csv"
PPO_EPISODE_REWARDS_PATH = OUTPUT_DIR / "live_progress" / "ppo_episode_rewards_live.csv"
BC_CASE_REWARD_PATH = OUTPUT_DIR / "reward_case_summary.csv"
BC_COMPONENTS_PATH = OUTPUT_DIR / "reward_stats_by_municipality.csv"
PPO_COMPONENT_CANDIDATES = [
    OUTPUT_DIR / "live_progress" / "ppo_reward_components_live.csv",
    OUTPUT_DIR / "ppo_reward_components.csv",
]


def load_ppo_reward_summary() -> dict:
    if PPO_METRICS_PATH.exists():
        ppo_df = pd.read_csv(PPO_METRICS_PATH)
        if "final_rl_avg_reward" in ppo_df.columns:
            rewards = pd.to_numeric(ppo_df["final_rl_avg_reward"], errors="coerce").dropna()
            return {
                "model": "PPO",
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std(ddof=0)) if len(rewards) else np.nan,
                "n_samples": int(len(rewards)),
                "source_file": str(PPO_METRICS_PATH),
            }

    if PPO_EPISODE_REWARDS_PATH.exists():
        ep_df = pd.read_csv(PPO_EPISODE_REWARDS_PATH)
        rewards = pd.to_numeric(ep_df.get("episode_reward", pd.Series(dtype=float)), errors="coerce").dropna()
        return {
            "model": "PPO",
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(ddof=0)) if len(rewards) else np.nan,
            "n_samples": int(len(rewards)),
            "source_file": str(PPO_EPISODE_REWARDS_PATH),
        }

    raise FileNotFoundError(
        "Could not find PPO reward summaries. Expected one of: "
        f"{PPO_METRICS_PATH} or {PPO_EPISODE_REWARDS_PATH}"
    )


def load_bc_reward_summary() -> dict:
    if not BC_CASE_REWARD_PATH.exists():
        raise FileNotFoundError(f"Missing BC reward summary file: {BC_CASE_REWARD_PATH}")

    bc_df = pd.read_csv(BC_CASE_REWARD_PATH)
    rewards = pd.to_numeric(bc_df.get("cumulative_reward", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "model": "BC_policy",
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=0)) if len(rewards) else np.nan,
        "n_samples": int(len(rewards)),
        "source_file": str(BC_CASE_REWARD_PATH),
    }


def load_component_comparison() -> pd.DataFrame:
    components = [
        "delay_penalty_mean",
        "rework_penalty_mean",
        "invalid_penalty_mean",
        "completion_bonus_mean",
    ]

    rows = []

    if BC_COMPONENTS_PATH.exists():
        bc_comp_df = pd.read_csv(BC_COMPONENTS_PATH)
        for comp in components:
            value = pd.to_numeric(bc_comp_df.get(comp, pd.Series(dtype=float)), errors="coerce").mean()
            rows.append({"model": "BC_policy", "component": comp, "value": float(value) if pd.notna(value) else np.nan})
    else:
        for comp in components:
            rows.append({"model": "BC_policy", "component": comp, "value": np.nan})

    ppo_component_df = None
    for candidate in PPO_COMPONENT_CANDIDATES:
        if not candidate.exists():
            continue
        candidate_df = pd.read_csv(candidate)
        if set(components).issubset(candidate_df.columns):
            ppo_component_df = candidate_df
            break

    if ppo_component_df is not None:
        for comp in components:
            value = pd.to_numeric(ppo_component_df.get(comp, pd.Series(dtype=float)), errors="coerce").mean()
            rows.append({"model": "PPO", "component": comp, "value": float(value) if pd.notna(value) else np.nan})
    else:
        for comp in components:
            rows.append({"model": "PPO", "component": comp, "value": np.nan})

    return pd.DataFrame(rows)


def plot_model_reward_comparison(reward_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(reward_df))
    means = pd.to_numeric(reward_df["reward_mean"], errors="coerce").to_numpy()
    stds = pd.to_numeric(reward_df["reward_std"], errors="coerce").to_numpy()
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors[: len(reward_df)], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(reward_df["model"].tolist())
    ax.set_ylabel("Average reward")
    ax.set_title("Reward Comparison: BC vs PPO")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, mean in zip(bars, means):
        if np.isfinite(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_component_comparison(component_df: pd.DataFrame, output_path: Path) -> None:
    pivot_df = component_df.pivot(index="component", columns="model", values="value")
    components = pivot_df.index.tolist()
    models = pivot_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.38
    x = np.arange(len(components))

    for idx, model in enumerate(models):
        vals = pd.to_numeric(pivot_df[model], errors="coerce").to_numpy(dtype=float)
        plot_vals = np.nan_to_num(vals, nan=0.0)
        offset = (idx - (len(models) - 1) / 2.0) * width
        bars = ax.bar(x + offset, plot_vals, width=width, label=model, alpha=0.85)

        for bar, raw_val in zip(bars, vals):
            if np.isnan(raw_val):
                bar.set_hatch("//")
                bar.set_alpha(0.35)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.01,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Delay penalty",
            "Rework penalty",
            "Invalid penalty",
            "Completion bonus",
        ]
    )
    ax.set_ylabel("Mean component value")
    ax.set_title("Reward Component Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    print("Step 1/3: Building reward comparison table...")
    ppo_reward = load_ppo_reward_summary()
    bc_reward = load_bc_reward_summary()

    reward_df = pd.DataFrame([ppo_reward, bc_reward]).sort_values("model").reset_index(drop=True)
    reward_csv = OUTPUT_DIR / "bc_vs_ppo_reward_comparison_latest.csv"
    reward_df.to_csv(reward_csv, index=False)

    print("Step 2/3: Building reward component comparison...")
    component_df = load_component_comparison()
    component_csv = OUTPUT_DIR / "bc_vs_ppo_reward_components_latest.csv"
    component_df.to_csv(component_csv, index=False)

    print("Step 3/3: Rendering comparison charts...")
    reward_chart = OUTPUT_DIR / "bc_vs_ppo_reward_comparison.png"
    component_chart = OUTPUT_DIR / "bc_vs_ppo_reward_components.png"
    plot_model_reward_comparison(reward_df, reward_chart)
    plot_reward_component_comparison(component_df, component_chart)

    print("\nREWARD COMPARISON TABLE")
    print(reward_df.to_string(index=False))
    print("\nREWARD COMPONENT TABLE")
    print(component_df.to_string(index=False))
    print(f"\nSaved: {reward_csv}")
    print(f"Saved: {component_csv}")
    print(f"Saved: {reward_chart}")
    print(f"Saved: {component_chart}")

if __name__ == "__main__":
    main()
