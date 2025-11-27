import numpy as np
import argparse
import json

def inspect_array(name, arr, show_values=False):
    print(f"\n====== {name} ======")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")

    # numeric stats
    if np.issubdtype(arr.dtype, np.number):
        print(f"min: {arr.min()}, max: {arr.max()}, mean: {arr.mean():.3f}")
    else:
        print(f"Non-numeric array")

    # show a small sample
    if show_values:
        flat = arr.reshape(-1)
        print("sample values:", flat[:10])


def print_transition(data, idx):
    print("\n====== Example Transition ======")
    for key in data.files:
        arr = data[key]
        if arr.ndim == 1:
            print(f"{key}: {arr[idx]}")
        else:
            print(f"{key}: {arr[idx].shape}")


def main(path):
    print(f"\nLoading dataset: {path}")
    data = np.load(path)

    print("\n=============== KEYS ===============")
    print(data.files)

    # Print structure for each key
    for key in data.files:
        inspect_array(key, data[key], show_values=False)

    # Print example transition (index 0)
    print_transition(data, 0)

    # Print episode boundary info if available
    if "terminals" in data.files or "timeouts" in data.files:
        print("\n=============== EPISODE INFO ===============")
        terminals = data["terminals"] if "terminals" in data.files else None
        timeouts = data["timeouts"] if "timeouts" in data.files else None

        if terminals is not None:
            print(f"Number of terminals (episode ends): {terminals.sum()}")

        if timeouts is not None:
            print(f"Number of timeouts: {timeouts.sum()}")

    # Print observation + action dimensions
    if "observations" in data.files:
        print("\n=============== OBS/ACTION SHAPES ===============")
        obs = data["observations"]
        print(f"Observation dimension: {obs.shape[1:]}")

    if "actions" in data.files:
        act = data["actions"]
        print(f"Action dimension: {act.shape[1:]}")

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to .npz dataset")
    args = parser.parse_args()

    main(args.path)
