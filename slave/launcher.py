import os
import json

# === Config ===
main_script = "main.py"
venv_python = "venv/bin/python"  # Path to your Python inside venv

base_data_dir = "./data/MNIST_WIN"
merged_dir = os.path.join(base_data_dir, "merged_datasets")
dataset_mnist_non_rotated = os.path.join(base_data_dir, "dataset_mnist_non_rotated")

base_save_dir = "./saves"
base_log_dir = "./logs/json/"
model_name = "cyvgg19"
polar_transform = "linearpolar"

overwrite_logs = True
overwrite_models = False

required_files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

# === Utilities ===
def dataset_valid(path):
    return all(os.path.exists(os.path.join(path, f)) for f in required_files)

def run_command(cmd, log_file=None):
    if log_file:
        if not overwrite_logs and os.path.exists(log_file):
            print(f"⚠️ Skipping existing log: {log_file}")
            return
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        cmd = f"{cmd} > {log_file} 2>&1"
    print(f"🚀 Running: {cmd}")
    os.system(cmd)

# === Setup ===
os.makedirs(base_save_dir, exist_ok=True)
os.makedirs(base_log_dir, exist_ok=True)

# === Load Scenarios ===
try:
    with open("train_test_scenarios.json") as f:
        train_test_dict = json.load(f)
except FileNotFoundError:
    print("❌ train_test_scenarios.json not found.")
    exit(1)

# === Main Launcher ===
def main():
    for train_set, test_sets in train_test_dict.items():
        train_data_dir = os.path.join(merged_dir, train_set)
        model_save_path = os.path.join(base_save_dir, f"{train_set}.pt")
        train_log_file = os.path.join(base_log_dir, f"{train_set}_train.txt")

        if not dataset_valid(train_data_dir):
            print(f"❌ Missing training data files for: {train_set}")
            continue

        print(f"\n=== TRAINING on {train_set} ===")
        if overwrite_models or not os.path.exists(model_save_path):
            train_cmd = (
                f"{venv_python} {main_script} "
                f"--train --model={model_name} --dataset=mnist-custom "
                f"--polar-transform={polar_transform} "
                f"--data-dir={train_data_dir}"
            )
            run_command(train_cmd, train_log_file)
        else:
            print(f"✅ Model already exists: {model_save_path}")

        for test_set in test_sets:
            test_data_dir = (
                dataset_mnist_non_rotated if test_set == "dataset_mnist_non_rotated"
                else os.path.join(merged_dir, test_set)
            )

            if not dataset_valid(test_data_dir):
                print(f"❌ Missing test data for: {test_set}, skipping.")
                continue

            print(f"--- TESTING {train_set} model on {test_set} ---")
            test_log_file = os.path.join(base_log_dir, f"{train_set}_test_on_{test_set}.txt")
            test_cmd = (
                f"{venv_python} {main_script} "
                f"--test --model={model_name} --dataset=mnist-custom "
                f"--polar-transform={polar_transform} "
                f"--data-dir={train_data_dir} "
                f"--test-data-dir={test_data_dir}"
            )
            run_command(test_cmd, test_log_file)

    print("\n✅ All training and testing complete!")

if __name__ == "__main__":
    main()
