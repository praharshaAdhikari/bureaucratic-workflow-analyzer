import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from digital_twin.sim_env import SimEnv

def train():
    twin_params = {
        "arrival_rate": 0.5,
    }

    env = SimEnv(twin_params=twin_params, logs_path="logs.csv")
    
    print("Checking environment compatibility...")
    try:
        check_env(env)
        print("Environment is compatible!")
    except Exception as e:
        print(f"Environment check failed: {e}")
    
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    print("Starting training...")
    model.learn(total_timesteps=10000)
    print("Training finished.")
    
    model.save("ppo_bureaucracy_v1")
