import os

# Paths
main_script = "main.py"
venv_python = "venv/bin/python"  # WSL/Linux

# Datasets
train_datasets = [
    "merged_datasets/merged_nonrot_45",
    "merged_datasets/merged_20-50_45",
    # "merged_20-50_45",
    # Add more training datasets here
]

test_datasets = [
    "dataset_mnist_non_rotated",
    "rotated-0-20",
    "rotated-20-50",
    "rotated-45",
    "rotated-50-90",
    "rotated-90-120",
    # "rotated-120-150",
    # "rotated-120-180",
    # "rotated-150-180",
    # Add more testing datasets here
]

# Paths and Settings
# base_data_dir = "./dataset/MNIST"
base_data_dir = "./data/MNIST_WIN"
base_save_dir = "./saves"
base_log_dir = "./logs"
model_name = "cyvgg19"
polar_transform = "linearpolar"

overwrite_logs = True  # Skip if file already exists
overwrite_models = False

# Ensure save/log directories exist
os.makedirs(base_save_dir, exist_ok=True)
os.makedirs(base_log_dir, exist_ok=True)

def run_command(cmd, log_file=None):
    if log_file:
        if not overwrite_logs and os.path.exists(log_file):
            print(f"⚠️ Skipping existing log: {log_file}")
            return
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        cmd = f"{cmd} > {log_file}"
    print(f"🚀 Running: {cmd}")
    os.system(cmd)

def main():
    for train_set in train_datasets:
        print(f"\n=== TRAINING on {train_set} ===")

        train_data_dir = os.path.join(base_data_dir, train_set)
        train_log_file = os.path.join(base_log_dir, f"{train_set}_train.txt")
        model_save_name = f"{train_set}.pt"
        model_save_path = os.path.join(base_save_dir, model_save_name)

        if not os.path.exists(model_save_path):
            print(f"⛔ Model not found: {model_save_path}, skipping test.")
            continue  # Skip test if model is missing

        # Train model
        if overwrite_models or not os.path.exists(model_save_path):
            train_cmd = (
                f"{venv_python} {main_script} "
                f"--train --model={model_name} --dataset=mnist-custom "
                f"--polar-transform={polar_transform} --data-dir={train_data_dir}"
            )
            run_command(train_cmd, train_log_file)
        else:
            print(f"✅ Model already exists: {model_save_path}")

        # Test model on all test datasets
        for test_set in test_datasets:
            print(f"--- TESTING {train_set} model on {test_set} ---")

            test_data_dir = os.path.join(base_data_dir, test_set)
            test_log_file = os.path.join(base_log_dir, f"{train_set}_test_on_{test_set}.txt")

            test_cmd = (
                f"{venv_python} {main_script} "
                f"--test --model={model_name} --dataset=mnist-custom "
                f"--polar-transform={polar_transform} "
                f"--data-dir={train_data_dir} "   # still needed for generating name
                f"--test-data-dir={test_data_dir}"
            )
            run_command(test_cmd, test_log_file)

    print("\n✅ All training and testing complete!")

if __name__ == "__main__":
    main()
