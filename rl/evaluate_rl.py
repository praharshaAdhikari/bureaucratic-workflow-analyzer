from typing import Any, Dict

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from digital_twin.sim_env import SimEnv


def evaluate(sim_env: SimEnv, num_episodes: int = 1000) -> float:
    """Main execution function to evaluate the RL agent."""
    # Load the trained agent
    model = PPO.load("ppo_workflow")

    # Evaluate the agent
    obs: Dict[str, Any] = sim_env.reset()
    total_reward: float = 0
    for _ in range(num_episodes):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = sim_env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs = sim_env.reset()

    print(f"Total reward after {num_episodes} steps: {total_reward}")
    return total_reward
