import pickle
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    lane_before = data['lane_before']
    lane_after = data['lane_after']
    
    print(f"Data loaded. Shape: {features.shape}")
    return features, lane_before, lane_after

def create_labels(lane_before, lane_after):
    """
    Generate classification labels from lane changes.
    Logic:
      0 = Stay   (lane_after == lane_before)
      1 = Left   (lane_after < lane_before)
      2 = Right  (lane_after > lane_before)
    """
    labels = np.zeros_like(lane_before)
    
    # Calculate difference
    diff = lane_after - lane_before
    
    # Stay is already 0
    # Left: diff < 0
    labels[diff < 0] = 1
    # Right: diff > 0
    labels[diff > 0] = 2
    
    return labels

def train_probe(features, labels):
    # Split data (80% train, 20% test)
    # stratify=labels ensures we keep the same class distribution in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTraining Linear Classifier (Logistic Regression)...")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Initialize Logistic Regression (Linear Probe)
    # multi_class='multinomial' makes it softmax regression
    clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000, 
        class_weight='balanced' # Optional: handles imbalance if sampling wasn't perfect
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    print("\n=== Evaluation Results ===")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    target_names = ['Stay', 'Left', 'Right']
    
    # Check if we actually have all classes in the test set to avoid errors
    unique_labels = np.unique(y_test)
    present_names = [target_names[i] for i in unique_labels]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=present_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Normalized Confusion Matrix for easier reading
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("\nNormalized Confusion Matrix (Recall):")
    # Print formatted matrix
    print(f"{'':<10} {'Pred Stay':<10} {'Pred Left':<10} {'Pred Right':<10}")
    for i, label in enumerate(present_names):
        row = cm_norm[i]
        # Handle cases where some classes might be missing in mapping
        # We assume 0,1,2 map to Stay, Left, Right
        row_str = "  ".join([f"{val:.2f}".ljust(10) for val in row])
        print(f"{label:<10} {row_str}")

    return clf

def main():
    parser = argparse.ArgumentParser(description="Train linear probe for lane change direction.")
    parser.add_argument('--data_path', type=str, default='data/ckpt115_2m_probing_lane_100k.pkl',
                        help='Path to the pickle dataset.')
    parser.add_argument('--save_path', type=str, default='data/lane_classifier.pkl',
                        help='Path to save the trained classifier weights.')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: File not found at {args.data_path}")
        return

    # 1. Load
    features, lane_before, lane_after = load_data(args.data_path)
    
    # 2. Process Labels
    labels = create_labels(lane_before, lane_after)
    
    # Print stats
    unique, counts = np.unique(labels, return_counts=True)
    label_map = {0: 'Stay', 1: 'Left', 2: 'Right'}
    print("\nClass Distribution:")
    for u, c in zip(unique, counts):
        print(f"  {label_map[u]}: {c} ({c/len(labels)*100:.1f}%)")
    
    # 3. Train & Eval
    clf = train_probe(features, labels)

    # 4. Save
    print(f"\nSaving classifier to {args.save_path}...")
    try:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'wb') as f:
            pickle.dump(clf, f)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()