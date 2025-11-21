from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from digital_twin.sim_env import SimEnv


def train(sim_env: SimEnv):
    """Main execution function to train the RL agent."""

    model = PPO("MultiInputPolicy", sim_env, verbose=1)

    model.learn(total_timesteps=100000)

    model.save("ppo_workflow")
