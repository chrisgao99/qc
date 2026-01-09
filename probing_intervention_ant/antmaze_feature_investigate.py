import pickle
import numpy as np
from collections import Counter

# Path to your collected data
DATA_PATH = 'data/antmaze_raw_sequences.pkl'

# =============================================================================
# Helper: Labeling Logic (Same as before)
# =============================================================================
def get_moving_label(obs_start, obs_future, stop_threshold=0.5):
    """
    Returns a label for the movement direction.
    0: Stop/Stuck, 1: Forward, 2: Left, 3: Right, 4: Backward
    """
    # 1. Extract Positions (X, Y)
    pos_start = obs_start[:2]
    pos_future = obs_future[:2]
    
    # 2. Extract Heading (Yaw) from Quaternion (Indices 3,4,5,6)
    w, x, y, z = obs_start[3:7]
    current_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    # 3. Global Displacement
    dx_global = pos_future[0] - pos_start[0]
    dy_global = pos_future[1] - pos_start[1]
    
    # Check for "Stop/Stuck"
    dist = np.sqrt(dx_global**2 + dy_global**2)
    if dist < stop_threshold:
        return 0  # STOP
        
    # 4. Egocentric Rotation
    x_local = dx_global * np.cos(current_yaw) + dy_global * np.sin(current_yaw)
    y_local = -dx_global * np.sin(current_yaw) + dy_global * np.cos(current_yaw)
    
    # 5. Classify Angle
    angle_deg = np.degrees(np.arctan2(y_local, x_local))
    
    if -45 <= angle_deg <= 45:
        return 1  # FORWARD
    elif 45 < angle_deg <= 135:
        return 2  # LEFT
    elif -135 <= angle_deg < -45:
        return 3  # RIGHT
    else:
        return 4  # BACKWARD

LABEL_NAMES = {0: "Stop", 1: "Forward", 2: "Left", 3: "Right", 4: "Backward"}

# =============================================================================
# Main Analysis
# =============================================================================
def main():
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    # Unpack arrays
    states = data['states']          # State at START of chunk i
    episode_ids = data['episode_ids'] # Episode ID for chunk i
    
    total_chunks = len(states)
    print(f"Loaded {total_chunks} chunks from {len(np.unique(episode_ids))} episodes.")
    print("-" * 65)
    print(f"{'Horizon':<10} | {'Stop':<8} {'Fwd':<8} {'Left':<8} {'Right':<8} {'Back':<8} | {'Valid Samples'}")
    print("-" * 65)

    # Iterate from 1 to 10 chunks into the future
    for n_future in range(1, 11):
        labels = []
        
        # We iterate through the dataset.
        # We compare chunk [i] with chunk [i + n_future]
        for i in range(total_chunks):
            target_idx = i + n_future
            
            # Boundary check: Ensure we don't go out of bounds
            if target_idx >= total_chunks:
                continue
                
            # Episode check: Ensure both chunks belong to the same episode
            if episode_ids[i] == episode_ids[target_idx]:
                start_state = states[i]
                future_state = states[target_idx] # State at start of chunk (i + n)
                
                # Note: You might want to tune stop_threshold based on horizon length.
                # A threshold of 0.5 is good for 5 steps, but for 50 steps (N=10),
                # "Stuck" might need a slightly larger threshold or stay the same depending on definition.
                # Here we keep it fixed at 0.5.
                lbl = get_moving_label(start_state, future_state, stop_threshold=0.5)
                labels.append(lbl)

        # Calculate Statistics
        if not labels:
            print(f"N={n_future:<8} | No valid samples found.")
            continue
            
        counts = Counter(labels)
        total_valid = len(labels)
        
        # Format counts as percentages
        def get_pct(k):
            return f"{(counts[k]/total_valid)*100:4.1f}%"

        print(f"N={n_future:<2} ({n_future*5:2}s) | "
              f"{get_pct(0):<8} {get_pct(1):<8} {get_pct(2):<8} {get_pct(3):<8} {get_pct(4):<8} | "
              f"{total_valid}")

    print("-" * 65)

if __name__ == "__main__":
    main()