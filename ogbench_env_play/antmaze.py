import numpy as np

def explore_dataset(file_path):
    try:
        # Load the dataset
        data = np.load(file_path)
        
        print(f"--- Exploring {file_path} ---")
        print(f"{'Key':<20} | {'Shape':<20} | {'Data Type'}")
        print("-" * 60)
        
        # Iterate through keys and print details
        for key in data.files:
            array = data[key]
            print(f"{key:<20} | {str(array.shape):<20} | {array.dtype}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# file path
file_path = '/p/yufeng/qc/.ogbench/data/antmaze-large-navigate-v0.npz'

# Run the function
if __name__ == "__main__":
    explore_dataset(file_path)