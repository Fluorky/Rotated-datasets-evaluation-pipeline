# src/analysis/learning_matrix.py
"""
Build accuracy matrices and heatmaps from test log files.

Changes in this version:
- Do NOT canonicalize arches: cyresnet56, resnet56, cyvgg19, vgg19 are distinct.
- Detect (arch, activation) ONLY from the TRAIN label (before "_test_on_").
- Case-insensitive, recursive scan; supports flat and nested test layouts.
- Color scale is GLOBAL per dataset (across all model groups): vmin = global min,
  vmax = global max, so the gradient always starts at the smallest value.
- Heatmap tiles are SQUARE (square=True + equal aspect).
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Distinct groups we want to output
MODEL_KEYS: List[str] = [
    "cyresnet56_logpolar",
    "cyresnet56_linearpolar",
    "cyvgg19_logpolar",
    "cyvgg19_linearpolar",
    "resnet56_logpolar",
    "resnet56_linearpolar",
    "vgg19_logpolar",
    "vgg19_linearpolar",
]

# Accuracy pattern like: "... (98.1234%) ..."
ACCURACY_PATTERN = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)%\s*\)")

# Order matters (avoid 'vgg19' catching 'cyvgg19', etc.)
ARCH_TOKENS: List[str] = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]
ACTIVATIONS: List[str] = ["linearpolar", "logpolar"]  # deterministic order


# --------------------------- helpers ---------------------------------------

def resolve_log_path(logs_base: Path, dataset_name: str) -> Optional[Path]:
    """
    Given a logs base (e.g., E:\\MasterThesisLogsAll or E:\\...\\json_GTSRB),
    try to locate the '.../test' directory for the dataset.
    Return the first existing path, or None.
    """
    candidates = [
        logs_base / "test",
        logs_base / f"json_{dataset_name}" / "test",
        logs_base / f"json_{dataset_name.lower()}" / "test",
        logs_base / f"json_{dataset_name.upper()}" / "test",
        logs_base / dataset_name / "test",
        logs_base / dataset_name.lower() / "test",
        logs_base / dataset_name.upper() / "test",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def extract_test_case(stem: str) -> str:
    """Return the part after '_test_on_' (case-insensitive); if none, return whole stem."""
    parts = re.split(r"(?i)_test_on_", stem)
    return parts[-1]


def extract_train_label(stem: str) -> str:
    """Return the part before '_test_on_' (case-insensitive); if none, return whole stem."""
    return re.split(r"(?i)_test_on_", stem)[0]


def detect_arch_from_train_label(train_label: str) -> Optional[str]:
    """Detect architecture token from the TRAIN label (checks longer tokens first)."""
    t = train_label.lower()
    for token in ARCH_TOKENS:  # cy* first, then bare
        if token in t:
            return token
    return None


def detect_activation_from_train_label(train_label: str) -> Optional[str]:
    """Detect activation token from the TRAIN label."""
    t = train_label.lower()
    for act in ACTIVATIONS:
        if act in t:
            return act
    return None


def rotation_sort_key(name: str) -> Tuple[int, int]:
    """
    Sort key for dataset/test-case names:
      dataset < merged < rotated < other
    and then by the first integer (angle) found in the name.
    """
    n = name.lower()
    if n.startswith("dataset"):
        base = 0
    elif n.startswith("merged"):
        base = 1
    elif n.startswith("rotated"):
        base = 2
    else:
        base = 3
    match = re.search(r"(\d+)", n)
    angle = int(match.group(1)) if match else 9999
    return (base, angle)


def extract_sort_key(col_name: str):
    """Extract the first integer from the column name to sort columns; inf if none."""
    match = re.search(r"(\d+)", col_name)
    return int(match.group(1)) if match else float("inf")


# ----------------------------- main ----------------------------------------

def process_dataset(dataset_name: str, logs_base: Path, output_base: Path):
    """
    Process test logs for the given dataset and produce:
      - CSV accuracy matrices for each group in MODEL_KEYS
      - Heatmap PNGs (all using the same vmin/vmax within the dataset)
    """
    # Locate /test
    log_path = resolve_log_path(Path(logs_base), dataset_name)
    output_path = Path(output_base) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    if not log_path:
        print(f"❌ Not found: no expected 'test' directory for '{dataset_name}' under base '{logs_base}'")
        print("   Checked for paths like:")
        print(f"   - {Path(logs_base) / f'json_{dataset_name}' / 'test'}")
        print(f"   - {Path(logs_base) / dataset_name / 'test'}")
        return

    # results[group_key][train_label][test_case] = accuracy
    results: Dict[str, Dict[str, Dict[str, float]]] = {key: defaultdict(dict) for key in MODEL_KEYS}

    # accepted log file extensions (recursive)
    patterns = ("**/*.txt", "**/*.log", "**/*.out")

    # Scan files recursively to handle flat or nested layouts
    log_files = []
    for pat in patterns:
        log_files.extend(log_path.glob(pat))

    if not log_files:
        print(f"⚠️ No log files found under: {log_path}")
        return

    # Collect values and track global min/max for a unified color scale
    all_values: List[float] = []

    for log_file in sorted(log_files):
        if not log_file.is_file():
            continue
        try:
            content = log_file.read_text(encoding="utf-8", errors="ignore")
            m = ACCURACY_PATTERN.search(content)
            if not m:
                continue

            accuracy = float(m.group(1))  # e.g., 98.1234 (percent)
            stem = log_file.stem

            train_label = extract_train_label(stem)
            test_case = extract_test_case(stem)

            arch = detect_arch_from_train_label(train_label)
            act  = detect_activation_from_train_label(train_label)
            if not arch or not act:
                # Unknown or unsupported model/activation — skip
                continue

            group_key = f"{arch}_{act}"
            if group_key not in results:
                # Ignore unexpected groups (keeps outputs consistent)
                continue

            results[group_key][train_label][test_case] = accuracy
            all_values.append(accuracy)

        except Exception as e:
            print(f"⚠️ Error processing {log_file}: {e}")

    # Determine a GLOBAL color scale per dataset
    vmin = min(all_values) if all_values else None
    vmax = max(all_values) if all_values else None

    # Build CSVs and heatmaps per model group
    for group_key, data in results.items():
        if not data:
            print(f"⚠️ No data found for: {group_key}")
            continue

        df = pd.DataFrame(data).T  # rows: train labels; cols: test cases
        if df.empty:
            print(f"⚠️ Empty DataFrame for: {group_key}")
            continue

        # sort columns by first integer (if any)
        df = df[sorted(df.columns, key=extract_sort_key)]

        # shorten labels (drop long prefixes)
        df.index = df.index.str.replace(r'^.+?-[^-]+-[^_]+_', '', regex=True)
        df.columns = df.columns.str.replace(r'^.+?-[^-]+-[^_]+_', '', regex=True)

        # semantic sorting for train/test labels (dataset/merged/rotated + angle)
        df = df.sort_index(key=lambda idx: [rotation_sort_key(name) for name in idx])
        df = df[sorted(df.columns, key=rotation_sort_key)]

        # --- CSV
        csv_path = output_path / f"accuracy_matrix_{group_key}.csv"
        df.to_csv(csv_path)
        print(f"✅ Saved CSV: {csv_path}")

        # --- Heatmap (square tiles)
        n_rows, n_cols = df.shape
        # Figure size can stay generous; square=True forces square cells in the Axes
        figsize = (max(20, n_cols * 1.5), max(8, n_rows * 1.5))

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            df.astype(float),
            annot=True,
            fmt=".4f",
            cmap="Purples",
            vmin=vmin,   # global minimum across the dataset
            vmax=vmax,   # global maximum across the dataset
            square=True,  # <-- make tiles square
            cbar_kws={"label": "Accuracy [%]"},
        )
        # Ensure equal aspect for safety (helps with very rectangular matrices)
        ax.set_aspect("equal", adjustable="box")

        plt.title(f"Accuracy Heatmap – {group_key.replace('_', ' ').title()}")
        plt.ylabel("Train Model")
        plt.xlabel("Test Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        heatmap_path = output_path / f"heatmap_{group_key}.png"
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"🖼️ Saved heatmap: {heatmap_path}")

    print(f"✅ Finished dataset: {dataset_name}")
