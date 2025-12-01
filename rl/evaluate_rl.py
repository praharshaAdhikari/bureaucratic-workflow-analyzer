import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
from digital_twin.sim_env import SimEnv
import matplotlib.pyplot as plt
import torch

# def calculate_baseline_metrics(logs_path="logs.csv"):
#     """Calculate metrics from the historical logs for comparison."""
#     try:
#         df = pd.read_csv(logs_path)
#         df['StartTime'] = pd.to_datetime(df['StartTime'])
#         df['EndTime'] = pd.to_datetime(df['EndTime'])
        
#         # Case Duration
#         case_durations = []
#         for case_id in df['CaseID'].unique():
#             case_df = df[df['CaseID'] == case_id]
#             start = case_df['StartTime'].min()
#             end = case_df['EndTime'].max()
#             duration = (end - start).total_seconds() / 60.0
#             case_durations.append(duration)
            
#         avg_duration = np.mean(case_durations) if case_durations else 0
#         median_duration = np.median(case_durations) if case_durations else 0
        
#         # Throughput (Cases per day)
#         total_duration_days = (df['EndTime'].max() - df['StartTime'].min()).total_seconds() / (60*60*24)
#         throughput = len(case_durations) / max(1.0, total_duration_days)
        
#         return {
#             "avg_duration": avg_duration,
#             "median_duration": median_duration,
#             "throughput_per_day": throughput,
#             "total_cases": len(case_durations)
#         }
#     except Exception as e:
#         print(f"Error calculating baseline: {e}")
#         return None


def calculate_baseline_metrics(logs_path="logs.csv"):
    import pandas as pd
    import numpy as np
    
    try:
        df = pd.read_csv(logs_path)

        # Convert to numeric minutes instead of datetime
        df['StartTime'] = df['StartTime'].astype(float)
        df['EndTime']   = df['EndTime'].astype(float)

        # Case Duration
        case_durations = []
        for case_id in df['CaseID'].unique():
            case_df = df[df['CaseID'] == case_id]
            start = case_df['StartTime'].min()
            end = case_df['EndTime'].max()
            duration = end - start
            case_durations.append(duration)

        avg_duration = np.mean(case_durations) if case_durations else 0
        median_duration = np.median(case_durations) if case_durations else 0

        # Throughput (Cases / day)
        total_minutes = df['EndTime'].max() - df['StartTime'].min()
        total_days = total_minutes / 1440  # convert minutes → days
        throughput = len(case_durations) / max(1.0, total_days)

        return {
            "avg_duration": avg_duration,
            "median_duration": median_duration,
            "throughput_per_day": throughput,
            "total_cases": len(case_durations)
        }
    except Exception as e:
        print(f"Error calculating baseline: {e}")
        return None


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU found. Using CPU.")


    # Load the environment with a higher arrival rate to stress test
    # 0.5 cases/min = 30 cases/hour = 720 cases/day (much higher than logs likely, but good for stress test)
    # Or we can try to match a busy day.
    twin_params = {
        "arrival_rate": 0.5, 
    }
    # Run for a longer time: 1440 mins = 24 hours
    sim_duration = 1440 * 5 # 5 days
    
    env = SimEnv(twin_params=twin_params, logs_path="logs.csv", decision_interval=10.0)

    # Load the trained model
    try:
        model = PPO.load("ppo_bureaucracy_v1", device=device)  # <-- THIS IS THE IMPORTANT CHANGE; USING GPU
        print("Model loaded successfully on", device)
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return

    # Baseline
    print("Calculating baseline metrics from logs...")
    baseline = calculate_baseline_metrics("logs.csv")
    if baseline:
        print(f"Baseline Avg Duration: {baseline['avg_duration']:.2f} mins")
        print(f"Baseline Throughput: {baseline['throughput_per_day']:.2f} cases/day")

    # Evaluation loop
    num_episodes = 1 # Run one long episode for stability
    
    print(f"Starting evaluation for {num_episodes} long episode(s) of {sim_duration} minutes...")

    all_case_durations = []
    all_queue_lengths = []
    all_worker_utilization = []
    total_completed_cases = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_completed_cases = []  # Track completions during episode
        
        # We need to manually stop if the env doesn't have a time limit, 
        # but SimEnv usually runs until no events or we can enforce time.
        # SimEnv.step() doesn't enforce time limit unless we wrap it or SimEnv has it.
        # SimEnv uses self.env.now.
        
        while env.env.now < sim_duration:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            all_queue_lengths.append(np.mean(obs['queue_lengths']))
            all_worker_utilization.append(np.mean(obs['worker_utilization']))
            
            # Track newly completed cases DURING the episode
            for case in env.workflow.cases:
                if case.status == "Accepted" and case not in episode_completed_cases:
                    episode_completed_cases.append(case)
                    if case.completion_time:
                        duration = case.completion_time - case.creation_time
                        all_case_durations.append(duration)
            
            if done or truncated:
                break

        # Post-episode metrics
        print(f"DEBUG: Total cases in workflow: {len(env.workflow.cases)}")
        print(f"DEBUG: Episode completed cases: {len(episode_completed_cases)}")
        for c in episode_completed_cases:
            print(f"DEBUG: Case {c.id} Status: {c.status}, Duration: {c.completion_time - c.creation_time if c.completion_time else 'N/A'}")
            
        total_completed_cases += len(episode_completed_cases)
        
        print(f"  Episode Reward: {episode_reward:.2f}")
        print(f"  Completed Cases: {len(episode_completed_cases)}")
        if all_case_durations:
            print(f"  Avg Case Duration: {np.mean(all_case_durations):.2f} mins")

    # Final Summary
    print("\n" + "="*40)
    print("EVALUATION SUMMARY (RL vs Baseline)")
    print("="*40)
    
    rl_avg_duration = np.mean(all_case_durations) if all_case_durations else 0
    rl_throughput = total_completed_cases / (sim_duration / (60*24)) # cases per day
    
    print(f"{'Metric':<25} | {'Baseline':<15} | {'RL Agent':<15}")
    print("-" * 60)
    print(f"{'Avg Case Duration (min)':<25} | {baseline['avg_duration'] if baseline else 'N/A':<15.2f} | {rl_avg_duration:<15.2f}")
    print(f"{'Throughput (cases/day)':<25} | {baseline['throughput_per_day'] if baseline else 'N/A':<15.2f} | {rl_throughput:<15.2f}")
    print(f"{'Avg Queue Length':<25} | {'N/A':<15} | {np.mean(all_queue_lengths):<15.2f}")
    print(f"{'Avg Utilization':<25} | {'N/A':<15} | {np.mean(all_worker_utilization):<15.2f}")
    print("="*40)
    
    if baseline and rl_avg_duration > 0:
        improvement = (baseline['avg_duration'] - rl_avg_duration) / baseline['avg_duration'] * 100
        print(f"Improvement in Duration: {improvement:.2f}%")
    
    print("\nNote: RL Agent was tested with arrival_rate=0.5 cases/min (High Load).")
    print("Baseline metrics are from historical logs which may have different load.")

if __name__ == "__main__":
    evaluate()
