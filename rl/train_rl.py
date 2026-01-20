import time
from stable_baselines3 import PPO
from digital_twin.sim_env import SimEnv
import torch

def train(logs_path="logs.csv", sep=';'):
    twin_params = {"arrival_rate": 0.5}
    env = SimEnv(twin_params=twin_params, logs_path=logs_path, decision_interval=10.0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    total_timesteps = 50_000  # Increase for better learning
    # Add learning_rate and entropy coefficient to improve exploration
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=0, 
        device=device,
        learning_rate=3e-4,  # Slightly higher default
        ent_coef=0.01,       # Encourage exploration
        gae_lambda=0.95,     # Better advantage estimation
    )

    start_time = time.time()
    for i in range(10):  # chunks of 5k steps
        model.learn(total_timesteps=5_000, reset_num_timesteps=False)

        elapsed = time.time() - start_time
        progress = ((i+1) * 5000) / total_timesteps
        eta = elapsed / progress - elapsed

        print(f"[{(progress*100):.1f}%] ⏳ Elapsed: {elapsed/60:.2f} min | ETA: {eta/60:.2f} min")

    model.save("ppo_bureaucracy_v1")
    print("Model training complete. Saved as 'ppo_bureaucracy_v1'.")
