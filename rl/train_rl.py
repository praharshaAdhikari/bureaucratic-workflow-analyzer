import time
from stable_baselines3 import PPO
from digital_twin.sim_env import SimEnv
import torch

def train():
    twin_params = {"arrival_rate": 0.5}
    env = SimEnv(twin_params=twin_params, logs_path="logs.csv", decision_interval=10.0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    total_timesteps = 50_000  # Increase for better learning
    model = PPO("MultiInputPolicy", env, verbose=0, device=device)

    start_time = time.time()
    for i in range(10):  # chunks of 5k steps
        model.learn(total_timesteps=5_000, reset_num_timesteps=False)

        elapsed = time.time() - start_time
        progress = ((i+1) * 5000) / total_timesteps
        eta = elapsed / progress - elapsed

        print(f"[{(progress*100):.1f}%] ⏳ Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min")

    model.save("ppo_bureaucracy_v1")
