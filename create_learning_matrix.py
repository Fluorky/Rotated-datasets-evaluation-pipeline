import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Base directories
base_dir = Path("/Users/maciej/PycharmProjects/MasterThesis")
log_root = base_dir / "log_files_from_slave/logs/json_4/test"
output_dir = base_dir / "heatmaps"
output_dir.mkdir(parents=True, exist_ok=True)

# Accuracy extraction pattern (match percentage in parentheses)
accuracy_pattern = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)%\s*\)")

# Four groups: (model, mapping)
results = {
    "cyresnet56_logpolar": defaultdict(dict),
    "cyresnet56_linearpolar": defaultdict(dict),
    "cyvgg19_logpolar": defaultdict(dict),
    "cyvgg19_linearpolar": defaultdict(dict)
}

# Walk through all test logs
for model_dir in log_root.iterdir():
    if not model_dir.is_dir():
        continue
    model_name = model_dir.name

    for txt_file in model_dir.glob("*.txt"):
        try:
            content = txt_file.read_text()
            match = accuracy_pattern.search(content)
            if not match:
                continue
            accuracy = float(match.group(1))  # now a true percentage like 91.93
            test_case = txt_file.stem.split("_test_on_")[-1]

            # Route to correct group
            key = None
            if "cyresnet56" in model_name and "logpolar" in model_name:
                key = "cyresnet56_logpolar"
            elif "cyresnet56" in model_name and "linearpolar" in model_name:
                key = "cyresnet56_linearpolar"
            elif "cyvgg19" in model_name and "logpolar" in model_name:
                key = "cyvgg19_logpolar"
            elif "cyvgg19" in model_name and "linearpolar" in model_name:
                key = "cyvgg19_linearpolar"

            if key:
                results[key][model_name][test_case] = accuracy

        except Exception as e:
            print(f"Error in {txt_file}: {e}")
# Save results
for group_key, data in results.items():
    if not data:
        print(f"⚠️ No data found for: {group_key}")
        continue

    # df = pd.DataFrame(data).T
    df = pd.DataFrame(data).T


    def extract_sort_key(col_name):
        match = re.search(r'(\d+)', col_name)
        return int(match.group(1)) if match else float('inf')

    sorted_columns = sorted(df.columns, key=extract_sort_key)
    df = df[sorted_columns]

    # sorted_index = sorted(df.index, key=extract_sort_key)
    # df = df.loc[sorted_index]
    #
    # # Optional: clean model names
    # df.index = df.index.str.replace(r"^mnist-custom-", "", regex=True)

    # Clean and sort row labels (i.e., training datasets)

    # def extract_train_set_name(model_name):
    #     """
    #     Extracts the training dataset part from full model name.
    #     E.g., 'mnist-custom-cyresnet56-logpolar_rotated-90' → 'rotated-90'
    #     """
    #     match = re.search(r'_(rotated[^_]*|rotated-\d+-\d+|merged_datasets/[^_]*|dataset_mnist_non_rotated)',
    #                       model_name)
    #     return match.group(1) if match else model_name

    #
    # def sort_key(name):
    #     """
    #     Sort key that extracts numeric value from dataset name, e.g., 'rotated-30-60' → 30
    #     Used to sort by rotation angle.
    #     """
    #     match = re.search(r'(\d+)', name)
    #     return int(match.group(1)) if match else float('inf')
    #
    #
    # # Apply name extraction and sorting
    # df.index = df.index.map(extract_train_set_name)
    # df = df.sort_index(key=lambda idx: [sort_key(name) for name in idx])
    # Clean row names: remove model prefix like 'mnist-custom-cyresnet56-logpolar_'

    # Clean row and column names: remove model prefix like 'mnist-custom-cyresnet56-logpolar_'
    df.index = df.index.str.replace(r'^mnist-custom-[^-]+-[^_]+_', '', regex=True)
    df.columns = df.columns.str.replace(r'^mnist-custom-[^-]+-[^_]+_', '', regex=True)


    # Define custom sort key (dataset < merged < rotated, then by first numeric value)
    def rotation_sort_key(name):
        if name.startswith("dataset"):
            base = 0
        elif name.startswith("merged"):
            base = 1
        elif name.startswith("rotated"):
            base = 2
        else:
            base = 3
        match = re.search(r'(\d+)', name)
        angle = int(match.group(1)) if match else 9999
        return (base, angle)


    # Sort rows (train models) and columns (test sets)
    df = df.sort_index(key=lambda idx: [rotation_sort_key(name) for name in idx])
    df = df[sorted(df.columns, key=rotation_sort_key)]

    # df.index = df.index.str.replace(r'^mnist-custom-[^-]+-[^_]+_', '', regex=True)
    #
    #
    # # Sort rows by rotation angle or type
    # def rotation_sort_key(name):
    #     """
    #     Custom sort key that sorts by type and first angle in name (if applicable).
    #     """
    #     # Prioritize: dataset_mnist < merged_datasets < rotated
    #     if name.startswith("dataset"):
    #         base = 0
    #     elif name.startswith("merged"):
    #         base = 1
    #     elif name.startswith("rotated"):
    #         base = 2
    #     else:
    #         base = 3
    #
    #     # Extract numeric angle for finer sort
    #     match = re.search(r'(\d+)', name)
    #     angle = int(match.group(1)) if match else 9999
    #     return (base, angle)
    #
    #
    # df = df.sort_index(key=lambda idx: [rotation_sort_key(name) for name in idx])

    # Save CSV
    csv_path = output_dir / f"accuracy_matrix_{group_key}.csv"
    df.to_csv(csv_path)
    print(f"✅ Saved CSV to: {csv_path}")

    # Plot heatmap with dynamic sizing
    n_rows, n_cols = df.shape
    figsize = (max(20, n_cols * 1.5), max(8, n_rows * 1.5))

    plt.figure(figsize=figsize)
    # sns.heatmap(df.astype(float), annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Accuracy [%]'})
    # sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="jet", cbar_kws={'label': 'Accuracy [%]'})
    # sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="Purples", cbar_kws={'label': 'Accuracy [%]'}, vmin=0, vmax=100)
    sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="Purples", cbar_kws={'label': 'Accuracy [%]'})
    plt.title(f"Accuracy Heatmap – {group_key.replace('_', ' ').title()}")
    plt.ylabel("Train Model")
    plt.xlabel("Test Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = output_dir / f"heatmap_{group_key}.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"🖼️ Saved heatmap to: {heatmap_path}")

print("✅ All heatmaps and CSVs saved.")
