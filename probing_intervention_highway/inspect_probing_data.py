import pickle
import numpy as np
import os

def inspect_pickle(file_path):
    print(f"=== Inspecting {file_path} ===\n")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type of loaded data: {type(data)}")
        
        if isinstance(data, dict):
            print("\n--- Top Level Keys ---")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"Key: '{key}' | Type: np.ndarray | Shape: {value.shape} | Dtype: {value.dtype}")
                elif isinstance(value, list):
                    print(f"Key: '{key}' | Type: list | Length: {len(value)}")
                else:
                    print(f"Key: '{key}' | Type: {type(value)}")
            
            # Detailed checks
            if 'labels' in data:
                print("\n--- Label Distribution ---")
                unique, counts = np.unique(data['labels'], return_counts=True)
                total = sum(counts)
                for u, c in zip(unique, counts):
                    label_name = {0: "Stay/Unknown", 1: "Left", 2: "Right"}.get(u, "Unknown")
                    print(f"Label {u} ({label_name}): {c} samples ({c/total*100:.2f}%)")

            if 'features' in data:
                print("\n--- Feature Stats ---")
                print(f"Mean: {np.mean(data['features']):.4f}")
                print(f"Std:  {np.std(data['features']):.4f}")
                print(f"Min:  {np.min(data['features']):.4f}")
                print(f"Max:  {np.max(data['features']):.4f}")

            if 'lane_before' in data:
                print("\n--- Sample Trajectory Check (First 20 steps) ---")
                print(f"{'Step':<6} | {'Lane':<6}")
                print("-" * 20)
                for i in range(min(20, len(data['lane_before']))):
                    print(f"{i:<6} | {data['lane_before'][i]:<6}")

            if 'lane_after' in data:
                print("\n--- Sample Trajectory Check (First 20 steps) ---")
                print(f"{'Step':<6} | {'Lane':<6}")
                print("-" * 20)
                for i in range(min(20, len(data['lane_after']))):
                    print(f"{i:<6} | {data['lane_after'][i]:<6}")

        else:
            print("Data is not a dictionary.")

    except Exception as e:
        print(f"An error occurred while loading the file: {e}")

if __name__ == "__main__":
    # You can change this path if your file is located elsewhere
    FILE_PATH = "data/ckpt115_2m_probing_lane.pkl"
    inspect_pickle(FILE_PATH)