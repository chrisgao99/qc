import pickle
import numpy as np
import os

# Path provided
classifier_path = '/p/yufeng/qc/data/h5_lane_classifier.pkl'

def analyze_classifier(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    print(f"Loading classifier from: {path}")
    with open(path, 'rb') as f:
        clf = pickle.load(f)

    # W shape is typically (n_classes, n_features) -> (3, 512)
    # b shape is (n_classes,) -> (3,)
    W = clf.coef_
    b = clf.intercept_
    
    classes = clf.classes_ # Typically [0, 1, 2]
    class_names = ["Straight (0)", "Left (1)", "Right (2)"]
    
    print(f"\nWeight Matrix Shape: {W.shape}")
    print("-" * 60)

    # 1. Individual Vector Properties
    print(f"{'Class':<15} | {'L2 Norm':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Near Zero (<1e-3)':<15}")
    print("-" * 60)
    
    threshold = 1e-3
    
    for i, class_idx in enumerate(classes):
        w_vec = W[i]
        
        l2_norm = np.linalg.norm(w_vec)
        mean_val = np.mean(w_vec)
        std_val = np.std(w_vec)
        
        # Count weights that are effectively zero
        near_zero_count = np.sum(np.abs(w_vec) < threshold)
        pct_zero = (near_zero_count / len(w_vec)) * 100
        
        name = class_names[class_idx] if class_idx < 3 else str(class_idx)
        print(f"{name:<15} | {l2_norm:<10.4f} | {mean_val:<10.4f} | {std_val:<10.4f} | {near_zero_count} ({pct_zero:.1f}%)")

    print("-" * 60)
    print("\n")

    # 2. Interaction Analysis (Testing Independence)
    # If vectors were independent, Cosine Sim would be ~0.
    # If Left is opposite of Right, Cosine Sim would be ~ -1.
    
    print("=== Geometric Relationships (Cosine Similarity) ===")
    print("Values close to 0  => Independent (Orthogonal)")
    print("Values close to 1  => Identical")
    print("Values close to -1 => Opposites")
    print("-" * 40)
    
    # Normalize vectors to unit length for easy comparison
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_normalized = W / norms
    
    # Compute Cosine Similarity Matrix
    cosine_sim_matrix = np.dot(W_normalized, W_normalized.T)
    
    labels = ["S", "L", "R"]
    print("      S       L       R")
    for i in range(3):
        row_str = f"{labels[i]}   "
        for j in range(3):
            val = cosine_sim_matrix[i, j]
            row_str += f"{val:>7.3f} "
        print(row_str)

    print("\n")
    
    # 3. Raw Dot Products (Magnitude of Interaction)
    # This answers: is w_i * w_j << w_i * w_i?
    print("=== Raw Dot Products (Interactions) ===")
    print("Compare Diagonal (Self) vs Off-Diagonal (Cross)")
    print("-" * 40)
    
    dot_prod_matrix = np.dot(W, W.T)
    
    print("      S       L       R")
    for i in range(3):
        row_str = f"{labels[i]}   "
        for j in range(3):
            val = dot_prod_matrix[i, j]
            row_str += f"{val:>7.2f} "
        print(row_str)

    # 4. summation of 3 weights vetors
    print("\n=== Summation of Weight Vectors ===")
    w_sum = np.sum(W, axis=0)
    # print(w_sum)
    print(f"Summed Weights Vector Shape: {w_sum.shape}")
    # number of elements near zero
    near_zero_count_sum = np.sum(np.abs(w_sum) < threshold)
    pct_zero_sum = (near_zero_count_sum / len(w_sum)) * 100
    print(f"Number of elements near zero (<{threshold}): {near_zero_count_sum} ({pct_zero_sum:.1f}%)")
    l2_norm_sum = np.linalg.norm(w_sum)
    print(f"L2 Norm of Summed Weights: {l2_norm_sum:.4f}")

if __name__ == "__main__":
    analyze_classifier(classifier_path)