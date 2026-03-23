
import torch
import numpy as np
import pickle
from bc_model import BCModel

class BCPolicy:
    """
    Policy wrapper that uses a trained BC model for predictions.
    Compatible with stable-baselines3 evaluation harness.
    """
    def __init__(self, model_path="output/bc_model.pt", encoder_path="output/bc_data/joint_encoder.pkl"):
        # Load encoders
        try:
            with open(encoder_path, "rb") as f:
                self.joint_encoder = pickle.load(f)
            num_classes = len(self.joint_encoder.classes_)
        except:
            print("Warning: Joint encoder not found. Using default dummy class count.")
            num_classes = 100
            
        # Hardcoded input size from dataset construction (5 features)
        input_size = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BCModel(input_size, num_classes)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.model.eval()

    def _extract_features(self, observation):
        """
        Convert PPO dictionary observation to BC model input features.
        X features: time_since_case_start_hours, time_since_prev_hours, rework_count_activity, branch_confidence, step_index
        Since these are not directly in the PPO obs (which is environment level), we use proxies.
        """
        # PPO observation has: queue_lengths, active_case_ages, available_workers, current_hour
        # We'll take the age of the oldest active case as a proxy for time_since_case_start_hours.
        # queue_lengths as a proxy for rework_count.
        
        ages = observation.get('active_case_ages', [0])
        max_age = np.max(ages) if len(ages) > 0 else 0.0
        
        avg_queue = np.mean(observation.get('queue_lengths', [0]))
        
        # We'll use a simplified mapping for the demonstration
        features = [
            float(max_age),      # proxy for time_since_case_start
            0.0,                 # proxy for time_since_prev
            float(avg_queue),    # proxy for rework
            0.8,                 # proxy for branch_confidence
            0.0                  # proxy for step_index
        ]
        return np.array(features, dtype=np.float32)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        Main prediction interface for SB3.
        Action Space: 15 management actions (0-14).
        """
        # 1. Prepare features
        feat_vec = self._extract_features(observation)
        feat_tensor = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 2. Get Resource/Actor prediction from BC model
        with torch.no_grad():
            output = self.model(feat_tensor)
            if deterministic:
                _, predicted_idx = torch.max(output, 1)
            else:
                prob = torch.softmax(output, dim=1)
                predicted_idx = torch.multinomial(prob, 1)
            
            raw_label = self.joint_encoder.inverse_transform([predicted_idx.item()])[0]
            
        # 3. Map Resource/Actor to management action (0-14)
        # For simplicity, we implement a name-based heuristic or default to 0.
        action = 0 # Default: assign_to_primary_team
        
        label_lower = str(raw_label).lower()
        if "manager" in label_lower or "chief" in label_lower:
            action = 6 # Escalate
        elif "priorit" in label_lower:
            action = 4 # Prioritize
        elif "pool" in label_lower or "volunteer" in label_lower:
            action = 1 # Outsource
            
        return np.array(action, dtype=np.int64), None

if __name__ == "__main__":
    policy = BCPolicy()
    dummy_obs = {
        'queue_lengths': np.zeros(15),
        'active_case_ages': np.zeros(100),
        'available_workers': np.array([10]),
        'current_hour': np.array([12.0])
    }
    action, _ = policy.predict(dummy_obs)
    print(f"Predicted action: {action}")
