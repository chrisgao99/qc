import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = 'data/antmaze_raw_sequences.pkl'
OUTPUT_CLF_PATH = 'data/antmaze_classifiers_nofilter.pkl'

LABEL_NAMES = {1: "Forward", 2: "Left", 3: "Right", 4: "Backward"}

# =============================================================================
# Helper: Labeling Logic
# =============================================================================
def get_moving_label(obs_start, obs_future, stop_threshold=0.5):
    pos_start = obs_start[:2]
    pos_future = obs_future[:2]
    
    # Quaternion indices 3,4,5,6 -> w, x, y, z
    w, x, y, z = obs_start[3:7]
    current_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    dx_global = pos_future[0] - pos_start[0]
    dy_global = pos_future[1] - pos_start[1]
    
    dist = np.sqrt(dx_global**2 + dy_global**2)
    if dist < stop_threshold:
        return 0  # STOP (Will be filtered out)
        
    x_local = dx_global * np.cos(current_yaw) + dy_global * np.sin(current_yaw)
    y_local = -dx_global * np.sin(current_yaw) + dy_global * np.cos(current_yaw)
    
    angle_deg = np.degrees(np.arctan2(y_local, x_local))
    
    if -45 <= angle_deg <= 45:
        return 1  # FORWARD
    elif 45 < angle_deg <= 135:
        return 2  # LEFT
    elif -135 <= angle_deg < -45:
        return 3  # RIGHT
    else:
        return 4  # BACKWARD

# =============================================================================
# Main Linear Probing Loop (4-Way Classification)
# =============================================================================
def main():
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Raw Arrays
    all_features = data['features']
    all_states = data['states']
    episode_ids = data['episode_ids']
    total_chunks = len(all_states)
    
    # Dictionary to store trained models
    trained_classifiers = {}

    print(f"Loaded {total_chunks} total samples.")
    print("Dropping 'Stuck' (0) labels for 4-way classification.")
    print("\n" + "="*85)
    print(f"{'Horizon':<10} | {'Samples':<10} | {'Test Acc':<10} | {'Baseline':<10} | {'Improvement':<12}")
    print("="*85)

    # Loop for N = 1 to 10
    for n_future in range(1, 11):
        
        # 1. Identify valid sequence pairs
        target_indices = np.arange(n_future, total_chunks)
        source_indices = np.arange(0, total_chunks - n_future)
        
        valid_mask = (episode_ids[source_indices] == episode_ids[target_indices])
        
        valid_source_idx = source_indices[valid_mask]
        valid_target_idx = target_indices[valid_mask]
        
        if len(valid_source_idx) == 0:
            continue

        # 2. Compute Labels for ALL valid sequences first
        start_states_subset = all_states[valid_source_idx]
        future_states_subset = all_states[valid_target_idx]
        
        temp_y = np.array([
            get_moving_label(s, f, stop_threshold=0.5) 
            for s, f in zip(start_states_subset, future_states_subset)
        ])
        
        # 3. FILTER: Remove 'Stuck' (0) labels
        active_mask = (temp_y != 0)
        
        X = all_features[valid_source_idx][active_mask]
        y = temp_y[active_mask]

        if len(X) < 50:
            print(f"N={n_future:<8} | Not enough active samples ({len(X)}). Skipping.")
            continue

        # 4. Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 6. Train Linear Probe
        clf = LogisticRegression(
            max_iter=2000, 
            multi_class='multinomial', 
            solver='lbfgs'
        )
        clf.fit(X_train_scaled, y_train)

        # 7. Evaluate
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        # 8. Baseline (Majority Class among the 4 active classes)
        vals, counts = np.unique(y_test, return_counts=True)
        majority_acc = np.max(counts) / np.sum(counts)

        # 9. Store for Saving
        trained_classifiers[n_future] = {
            'model': clf,
            'scaler': scaler,
            'test_acc': acc,
            'horizon_steps': n_future * 5
        }

        # Print Result
        print(f"N={n_future:<2} ({n_future*5:2}s) | "
              f"{len(X):<10} | "
              f"{acc*100:6.2f}%    | "
              f"{majority_acc*100:6.2f}%    | "
              f"+{(acc-majority_acc)*100:5.2f}%")

    print("="*85)

    # Save Classifiers
    if trained_classifiers:
        os.makedirs(os.path.dirname(OUTPUT_CLF_PATH), exist_ok=True)
        with open(OUTPUT_CLF_PATH, 'wb') as f:
            pickle.dump(trained_classifiers, f)
        print(f"\nSaved {len(trained_classifiers)} classifiers to: {OUTPUT_CLF_PATH}")

if __name__ == "__main__":
    main()