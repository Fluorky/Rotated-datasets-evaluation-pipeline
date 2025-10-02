"""
Optuna logs analyzer:
- Parses training curves (with last checkpoint marker) from Optuna .log files.
- Extracts [TEST] results to build a 1-row accuracy "heatmap" per run.
- Saves CSV + PNG per run, grouped by (arch, activation).

Usage via CLI command added in src/cli.py:
    python -m src.cli optuna-analyze -d GTSRB --optuna-logs "<path>/optuna_checked/logs" --output-dir "<OUT>"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse robust parsing & plotting from learning_curves
from .learning_curves import (
    parse_training_log,        # returns (df, last_ckpt_epoch)
    detect_arch,
    detect_activation,
    shorten_label,
    # we won't import _plot_combined to avoid tight coupling; we'll keep a local plotter below
)
from .learning_matrix import rotation_sort_key  # for consistent column ordering


# ----------------------- parsing [TEST] lines -------------------------------

# Example:
# [TEST] cyresnet56-linearpolar_dataset_GTSRB_non_rotated_test_on_rotated-330-360  loss=0.6516  acc=87.43%
TEST_LINE_RE = re.compile(
    r"^\[TEST\]\s+([^\s]+)\s+loss=\s*([0-9.]+)\s+acc=\s*([0-9.]+)%",
    re.MULTILINE
)

def parse_optuna_test_results(path: Path) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Returns (train_label, {test_case -> acc%}) from a single Optuna .log file.
    train_label is the token BEFORE '_test_on_' from the first [TEST] line.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, {}

    results: Dict[str, float] = {}
    train_label: Optional[str] = None

    for m in TEST_LINE_RE.finditer(text):
        token = m.group(1)                # "<train_label>_test_on_<test_case>"
        acc = float(m.group(3))           # percent
        if "_test_on_" in token:
            tr, _, te = token.partition("_test_on_")
        else:
            # fallback: take the whole token as test_case
            tr, te = token, token
        if train_label is None:
            train_label = tr
        results[te] = acc

    return train_label, results


# ----------------------- utilities ------------------------------------------

def _dataset_token_candidates(dataset_name: str) -> List[str]:
    d = dataset_name.lower()
    if d == "gtsrb_rgb":
        return ["gtsrb_rgb", "gtsrb-rgb"]
    if d == "gtsrb":
        return ["gtsrb_", "gtsrb-"]
    return [d]

def _matches_dataset(filename: str, dataset_name: str) -> bool:
    f = filename.lower()
    return any(tok in f for tok in _dataset_token_candidates(dataset_name))

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _plot_learning_curve_combined(df: pd.DataFrame, title: str, out_path: Path, ckpt_epoch: Optional[int]) -> None:
    """
    Minimal copy of the combined plotter (no dependency on private funcs).
    Left: train/val loss; Right: train/val acc (0..100%); vertical line at checkpoint.
    Colors kept consistent with learning_curves.py.
    """
    from matplotlib.ticker import PercentFormatter

    TRAIN_LOSS_COLOR = "#2ca02c"
    VAL_LOSS_COLOR   = "#1f77b4"
    TRAIN_ACC_COLOR  = "#ff7f0e"
    VAL_ACC_COLOR    = "#d62728"
    CKPT_LINE_COLOR  = "#555555"

    if df.empty:
        return

    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))
    lines, labels = [], []

    if "loss" in df and df["loss"].notna().any():
        l1, = ax.plot(df["epoch"], df["loss"], marker="o", color=TRAIN_LOSS_COLOR, label="train_loss")
        lines.append(l1); labels.append("train_loss")
    if "val_loss" in df and df["val_loss"].notna().any():
        l2, = ax.plot(df["epoch"], df["val_loss"], marker="o", linestyle="--", alpha=0.95,
                      color=VAL_LOSS_COLOR, label="val_loss")
        lines.append(l2); labels.append("val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    acc_series = pd.concat(
        [df.get("acc", pd.Series(dtype=float)), df.get("val_acc", pd.Series(dtype=float))],
        ignore_index=True
    ).dropna()

    if "acc" in df and df["acc"].notna().any():
        l3, = ax2.plot(df["epoch"], df["acc"], marker="s", color=TRAIN_ACC_COLOR, label="train_acc")
        lines.append(l3); labels.append("train_acc")
    if "val_acc" in df and df["val_acc"].notna().any():
        l4, = ax2.plot(df["epoch"], df["val_acc"], marker="s", linestyle="--", alpha=0.95,
                       color=VAL_ACC_COLOR, label="val_acc")
        lines.append(l4); labels.append("val_acc")

    ax2.set_ylabel("Accuracy [%]")
    if not acc_series.empty and acc_series.min() >= 0.0 and acc_series.max() <= 100.0:
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    if ckpt_epoch is not None:
        v = ax.axvline(x=ckpt_epoch, color=CKPT_LINE_COLOR, linestyle=":", linewidth=2, label=f"ckpt@{ckpt_epoch}")
        lines.append(v); labels.append(f"ckpt@{ckpt_epoch}")

    ncols = min(4, max(1, len(labels)))
    fig.legend(handles=lines, labels=labels,
               loc="upper center", bbox_to_anchor=(0.5, -0.08),
               ncol=ncols, frameon=True, title="Metrics")
    fig.subplots_adjust(bottom=0.18)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------- main API -------------------------------------------

def generate_optuna_learning_curves(dataset_name: str, optuna_logs_dir: Path, output_base: Path) -> None:
    """
    Build combined learning curves (one PNG per run) from Optuna logs.
    """
    logs_dir = Path(optuna_logs_dir)
    if not logs_dir.exists():
        print(f"❌ Optuna logs dir not found: {logs_dir}")
        return

    files = [p for p in logs_dir.glob("*.log") if p.is_file() and _matches_dataset(p.name, dataset_name)]
    if not files:
        print(f"⚠️ No Optuna .log files found for dataset '{dataset_name}' in: {logs_dir}")
        return

    out_root = Path(output_base) / dataset_name / "optuna" / "learning_curves"
    _ensure_dir(out_root)

    for f in sorted(files):
        df, ckpt_epoch = parse_training_log(f)
        # Label: prefer [TEST] train label if present; otherwise use filename stem
        train_label, _ = parse_optuna_test_results(f)
        label = shorten_label(train_label) if train_label else f.stem

        arch = detect_arch(label) or detect_arch(f.stem) or ""
        act  = detect_activation(label) or detect_activation(f.stem) or ""
        group = f"{arch}_{act}".strip("_") if arch and act else "unknown"

        out_dir = out_root / group
        _ensure_dir(out_dir)
        _plot_learning_curve_combined(df, f"Learning Curve – {label}", out_dir / f"{label}_combined.png", ckpt_epoch)

    print(f"✅ Optuna learning curves saved under: {out_root}")


def generate_optuna_heatmaps(dataset_name: str, optuna_logs_dir: Path, output_base: Path) -> None:
    """
    Build a 1-row accuracy heatmap (and CSV) from [TEST] lines per run.
    """
    logs_dir = Path(optuna_logs_dir)
    if not logs_dir.exists():
        print(f"❌ Optuna logs dir not found: {logs_dir}")
        return

    files = [p for p in logs_dir.glob("*.log") if p.is_file() and _matches_dataset(p.name, dataset_name)]
    if not files:
        print(f"⚠️ No Optuna .log files found for dataset '{dataset_name}' in: {logs_dir}")
        return

    out_root = Path(output_base) / dataset_name / "optuna" / "heatmaps"
    _ensure_dir(out_root)

    for f in sorted(files):
        train_label, results = parse_optuna_test_results(f)
        if not results:
            # No [TEST] section in this file — skip
            continue

        label = shorten_label(train_label) if train_label else f.stem

        # Columns sorted semantically (dataset/merged/rotated + angle)
        cols = sorted(results.keys(), key=rotation_sort_key)
        df = pd.DataFrame([ [results[c] for c in cols] ], columns=cols, index=[label])

        # save CSV
        arch = detect_arch(label) or detect_arch(f.stem) or ""
        act  = detect_activation(label) or detect_activation(f.stem) or ""
        group = f"{arch}_{act}".strip("_") if arch and act else "unknown"

        out_dir = out_root / group
        _ensure_dir(out_dir)

        csv_path = out_dir / f"{label}_accuracy.csv"
        df.to_csv(csv_path, float_format="%.4f")

        # heatmap (single row) – square tiles & nice width
        plt.figure(figsize=(max(16, len(cols) * 1.1), 3.8))
        ax = sns.heatmap(
            df.astype(float),
            annot=True, fmt=".2f",
            cmap="Purples",
            cbar_kws={"label": "Accuracy [%]"},
            square=True
        )
        ax.set_aspect("equal", adjustable="box")
        plt.title(f"Optuna Accuracy – {label}")
        plt.xlabel("Test Dataset")
        plt.ylabel("Run")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        png_path = out_dir / f"{label}_heatmap.png"
        plt.savefig(png_path, dpi=150)
        plt.close()

        print(f"🧾 {csv_path}")
        print(f"🖼️ {png_path}")

    print(f"✅ Optuna heatmaps saved under: {out_root}")
