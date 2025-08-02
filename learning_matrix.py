import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# === CONFIGURATION ===
DATASETS = ["MNIST"]
BASE_DIR = Path.cwd()
LOGS_BASE_PATH = BASE_DIR / "log_files_from_slave" / "logs"
OUTPUT_BASE_PATH = BASE_DIR / "heatmaps"

# Pattern to extract accuracy percentage from log files
ACCURACY_PATTERN = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)%\s*\)")

MODEL_KEYS = [
    "cyresnet56_logpolar",
    "cyresnet56_linearpolar",
    "cyvgg19_logpolar",
    "cyvgg19_linearpolar"
]

def determine_group_key(model_name: str):
    for key in MODEL_KEYS:
        model_type, activation = key.split("_")
        if model_type in model_name and activation in model_name:
            return key
    return None

def rotation_sort_key(name: str):
    if name.startswith("dataset"):
        base = 0
    elif name.startswith("merged"):
        base = 1
    elif name.startswith("rotated"):
        base = 2
    else:
        base = 3
    match = re.search(r"(\d+)", name)
    angle = int(match.group(1)) if match else 9999
    return (base, angle)

def extract_sort_key(col_name: str):
    match = re.search(r"(\d+)", col_name)
    return int(match.group(1)) if match else float('inf')

def process_dataset(dataset_name: str):
    log_path = LOGS_BASE_PATH / f"json_{dataset_name}" / "test"
    output_path = OUTPUT_BASE_PATH / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    results = {key: defaultdict(dict) for key in MODEL_KEYS}

    for model_dir in log_path.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for log_file in model_dir.glob("*.txt"):
            try:
                content = log_file.read_text()
                match = ACCURACY_PATTERN.search(content)
                if not match:
                    continue
                accuracy = float(match.group(1))
                test_case = log_file.stem.split("_test_on_")[-1]
                group_key = determine_group_key(model_name)

                if group_key:
                    results[group_key][model_name][test_case] = accuracy
            except Exception as e:
                print(f"⚠️ Error processing {log_file}: {e}")

    for group_key, data in results.items():
        if not data:
            print(f"⚠️ No data found for: {group_key}")
            continue

        df = pd.DataFrame(data).T
        sorted_columns = sorted(df.columns, key=extract_sort_key)
        df = df[sorted_columns]

        df.index = df.index.str.replace(r'^.+?-[^-]+-[^_]+_', '', regex=True)
        df.columns = df.columns.str.replace(r'^.+?-[^-]+-[^_]+_', '', regex=True)

        df = df.sort_index(key=lambda idx: [rotation_sort_key(name) for name in idx])
        df = df[sorted(df.columns, key=rotation_sort_key)]

        # Save CSV
        csv_path = output_path / f"accuracy_matrix_{group_key}.csv"
        df.to_csv(csv_path)
        print(f"✅ Saved CSV: {csv_path}")

        # Plot heatmap
        n_rows, n_cols = df.shape
        figsize = (max(20, n_cols * 1.5), max(8, n_rows * 1.5))

        plt.figure(figsize=figsize)
        sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="Purples", cbar_kws={'label': 'Accuracy [%]'})
        plt.title(f"Accuracy Heatmap – {group_key.replace('_', ' ').title()}")
        plt.ylabel("Train Model")
        plt.xlabel("Test Dataset")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        heatmap_path = output_path / f"heatmap_{group_key}.png"
        plt.savefig(heatmap_path)
        plt.close()
        print(f"🖼️ Saved heatmap: {heatmap_path}")

    print(f"✅ Finished dataset: {dataset_name}")

if __name__ == "__main__":
    for dataset in DATASETS:
        process_dataset(dataset)
    print("🏁 All datasets processed.")
