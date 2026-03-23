
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class BCDatasetBuilder:
    def __init__(self, data_dir="./dataset", output_dir="./output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.features_path = self.output_dir / "case_step_features.parquet"
        
        self.resource_encoder = LabelEncoder()
        self.actor_encoder = LabelEncoder()
        self.joint_encoder = LabelEncoder()
        
    def load_raw_ocel_proxies(self):
        """Extract resource and actor from raw OCEL logs."""
        proxies = []
        for m in range(1, 6):
            log_path = self.data_dir / f"BPIC15_Municipality{m}.jsonocel"
            if not log_path.exists():
                print(f"Warning: {log_path} not found.")
                continue
            
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            events = data.get('ocel:events', data.get('events', {}))
            for eid, ev in events.items():
                vmap = ev.get('ocel:vmap', ev.get('vmap', {}))
                resource = vmap.get('resource', 'unknown')
                actor = vmap.get('Responsible_actor', 'unknown')
                proxies.append({
                    'event_id': eid,
                    'resource': str(resource),
                    'Responsible_actor': str(actor)
                })
        return pd.DataFrame(proxies)

    def build_ppo_state(self, df):
        """
        Approximate the PPO state representation from the features table.
        State: [queue_lengths (15), active_case_ages (100), available_workers (1), current_hour (1)]
        """
        # We need to simulate the environment's state-building logic for historical points.
        # For simplicity in this BC baseline, we use the event-level features as proxies.
        # But to be 'the same representation', we should ideally aggregate.
        
        # 1. Activity encoding (first 15 activities)
        activities = sorted(df['activity'].unique())[:15]
        activity_map = {a: i for i, a in enumerate(activities)}
        
        df = df.copy()
        df['activity_idx'] = df['activity'].map(lambda x: activity_map.get(x, 14))
        
        # Vectorize state
        # In a real BC, we'd reconstruct the full queue at timestamp T.
        # Here we use the features already computed in Step 2:
        # - elapsed_hours, rework_count, branch_confidence, etc.
        
        state_cols = [
            'time_since_case_start_hours', 
            'time_since_prev_hours',
            'rework_count_activity',
            'branch_confidence',
            'step_index'
        ]
        # Pad with zeros if columns are missing
        for col in state_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[state_cols].values

    def create_dataset(self, test_size=0.2, val_size=0.1):
        print("Loading features...")
        df = pd.read_parquet(self.features_path)
        
        print("Loading OCEL proxies...")
        proxies_df = self.load_raw_ocel_proxies()
        
        print("Merging...")
        df = df.merge(proxies_df, on='event_id', how='left')
        df['resource'] = df['resource'].fillna('unknown')
        df['Responsible_actor'] = df['Responsible_actor'].fillna('unknown')
        
        # LIMIT DATASET FOR SPEED
        if len(df) > 20000:
            print(f"Sampling 20,000 events from {len(df)}...")
            df = df.sample(20000, random_state=42)
            
        # Create joint label
        df['joint_label'] = df['resource'] + "__" + df['Responsible_actor']
        
        print("Encoding labels...")
        y = self.joint_encoder.fit_transform(df['joint_label'])
        
        print("Building states...")
        X = self.build_ppo_state(df)
        
        # Split
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=42)
        
        print(f"Dataset summary: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        print(f"Number of classes: {len(self.joint_encoder.classes_)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    builder = BCDatasetBuilder()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = builder.create_dataset()
    
    # Save for training
    os.makedirs("output/bc_data", exist_ok=True)
    np.save("output/bc_data/X_train.npy", X_train)
    np.save("output/bc_data/y_train.npy", y_train)
    np.save("output/bc_data/X_val.npy", X_val)
    np.save("output/bc_data/y_val.npy", y_val)
    np.save("output/bc_data/X_test.npy", X_test)
    np.save("output/bc_data/y_test.npy", y_test)
    
    # Save encoder
    import pickle
    with open("output/bc_data/joint_encoder.pkl", "wb") as f:
        pickle.dump(builder.joint_encoder, f)
    print("Dataset saved to output/bc_data/")
