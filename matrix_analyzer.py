import os
import numpy as np
import pandas as pd

from tqdm import tqdm

def calculate_accuracy(conf_matrix):
    correct = np.trace(conf_matrix)
    total = conf_matrix.sum()
    return correct / total if total > 0 else 0.0

def collect_results(confusion_matrices_root_dir):
    results = []

    for model_dir in tqdm(os.listdir(confusion_matrices_root_dir)):
        model_path = os.path.join(confusion_matrices_root_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        for test_subdir in os.listdir(model_path):
            test_path = os.path.join(model_path, test_subdir)
            cm_file = os.path.join(test_path, "confusion_matrix.npy")
            if not os.path.exists(cm_file):
                continue

            try:
                cm = np.load(cm_file)
                acc = calculate_accuracy(cm)
                results.append({
                    "model": model_dir,
                    "test_case": test_subdir,
                    "accuracy": acc
                })
            except Exception as e:
                print(f"⚠️ Failed to load {cm_file}: {e}")

    return pd.DataFrame(results)

def analyze_confusion_matrices(confusion_matrices_root_dir):
    df = collect_results(confusion_matrices_root_dir)

    # Pivot for overview (rows: models, cols: test cases)
    pivot_table = df.pivot(index="model", columns="test_case", values="accuracy")

    # Compute average accuracy per model
    model_means = pivot_table.mean(axis=1).sort_values(ascending=False)

    print("\n📊 Average Accuracy per Model:")
    print(model_means)

    best_model = model_means.idxmax()
    print(f"\n🏆 Best Performing Model: {best_model} with mean accuracy {model_means.max():.4f}")

    return pivot_table, model_means

# === Example usage ===
# Set this to your actual directory path
confusion_matrices_root_dir =  r'\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_4\confusion_matrices'
pivot_table, model_means = analyze_confusion_matrices(confusion_matrices_root_dir)

# Optionally, save to CSV
pivot_table.to_csv("confusion_accuracy_matrix.csv")
model_means.to_csv("model_average_accuracies.csv")
