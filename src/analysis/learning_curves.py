# src/analysis/learning_curves.py
"""
Parse training logs and generate learning curves (loss/accuracy vs. epoch).

- Distinguishes arches: cyresnet56, resnet56, cyvgg19, vgg19 (no canonicalization).
- Uses only the TRAIN label (filename stem of *train* logs) to detect (arch, activation).
- Scans logs recursively under .../train (works with flat or nested layouts).
- Recognizes common metric formats, e.g.:
    Epoch 1/90 - loss: 0.54 - acc: 87.2% - val_loss: 0.60 - val_accuracy: 0.84
    epoch: 2 loss: 0.432 accuracy: 0.905 val_loss: 0.51 val_acc: 88.1%
- Converts accuracy to percent if it looks like fraction (<= 1.5).
- Outputs PNGs per run (loss + accuracy) under:
    <output>/<DATASET>/learning_curves/<group_key>/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# Distinct groups we support
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

ARCH_TOKENS: List[str] = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]
ACTIVATIONS: List[str] = ["linearpolar", "logpolar"]

# Epoch/metric patterns (case-insensitive)
EPOCH_RE = re.compile(r"(?i)\bepoch(?:\s*[:#]|\s+)?\s*(\d+)(?:\s*/\s*\d+)?")
LOSS_RE = re.compile(r"(?i)\bval[_\s-]?loss\s*[:=]\s*([0-9]*\.?[0-9]+)|\bloss\s*[:=]\s*([0-9]*\.?[0-9]+)")
ACC_RE  = re.compile(r"(?i)\bval[_\s-]?(acc|accuracy)\s*[:=]\s*([0-9]*\.?[0-9]+)%?|\b(acc|accuracy)\s*[:=]\s*([0-9]*\.?[0-9]+)%?")

def resolve_train_path(logs_base: Path, dataset_name: str) -> Optional[Path]:
    candidates = [
        logs_base / "train",
        logs_base / f"json_{dataset_name}" / "train",
        logs_base / f"json_{dataset_name.lower()}" / "train",
        logs_base / f"json_{dataset_name.upper()}" / "train",
        logs_base / dataset_name / "train",
        logs_base / dataset_name.lower() / "train",
        logs_base / dataset_name.upper() / "train",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None

def detect_arch(train_label: str) -> Optional[str]:
    t = train_label.lower()
    for token in ARCH_TOKENS:  # cy* first, then bare
        if token in t:
            return token
    return None

def detect_activation(train_label: str) -> Optional[str]:
    t = train_label.lower()
    for act in ACTIVATIONS:
        if act in t:
            return act
    return None

def shorten_label(label: str) -> str:
    # Similar shortening as in heatmaps (drop long prefixes like "...-xxx-yyy_")
    return re.sub(r'^.+?-[^-]+-[^_]+_', '', label)

def _to_percent(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    # If number looks like a fraction (<= 1.5), treat as 0..1 and convert to percent
    return x * 100.0 if x <= 1.5 else x

def parse_training_log(path: Path) -> pd.DataFrame:
    """
    Parse a single training log into a tidy DataFrame with columns:
    epoch, loss, val_loss, acc, val_acc  (accuracy in percent if possible).
    """
    rows: List[Dict[str, float]] = []
    epoch_counter = 0

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "acc", "val_acc"])

    for line in text.splitlines():
        # Try to detect epoch
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            epoch_counter = epoch
        else:
            # Increment when we see metrics without explicit epoch (best-effort)
            # Do not increment on empty/noise lines
            epoch = None

        # Extract losses (prefer val_loss if present)
        loss_match = LOSS_RE.search(line)
        loss = None
        val_loss = None
        if loss_match:
            # LOSS_RE has two alternatives; pick groups accordingly
            val_loss_str, loss_str = loss_match.groups()
            if val_loss_str is not None:
                val_loss = float(val_loss_str)
            if loss_str is not None:
                loss = float(loss_str)

        # Extract accuracies (val first if present)
        acc = None
        val_acc = None
        acc_match = ACC_RE.search(line)
        if acc_match:
            # The regex has 4 capturing groups; pick non-None values
            groups = acc_match.groups()
            # groups: (val_tag, val_val, acc_tag, acc_val)
            if groups[1] is not None:
                val_acc = float(groups[1])
            if groups[3] is not None:
                acc = float(groups[3])

        # If we captured anything metric-like, record a row
        if any(v is not None for v in (loss, val_loss, acc, val_acc)):
            if epoch is None:
                epoch_counter += 1
                epoch = epoch_counter
            rows.append({
                "epoch": int(epoch),
                "loss": loss,
                "val_loss": val_loss,
                "acc": _to_percent(acc) if acc is not None else None,
                "val_acc": _to_percent(val_acc) if val_acc is not None else None,
            })

    if not rows:
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "acc", "val_acc"])

    df = pd.DataFrame(rows).sort_values("epoch")
    # Deduplicate per epoch keeping the last occurrence
    df = df.drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    return df

def _plot_loss(df: pd.DataFrame, title: str, out_path: Path):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    if "loss" in df and df["loss"].notna().any():
        plt.plot(df["epoch"], df["loss"], marker="o", label="loss")
    if "val_loss" in df and df["val_loss"].notna().any():
        plt.plot(df["epoch"], df["val_loss"], marker="o", label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_acc(df: pd.DataFrame, title: str, out_path: Path):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    if "acc" in df and df["acc"].notna().any():
        plt.plot(df["epoch"], df["acc"], marker="o", label="acc")
    if "val_acc" in df and df["val_acc"].notna().any():
        plt.plot(df["epoch"], df["val_acc"], marker="o", label="val_acc")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_learning_curves(dataset_name: str, logs_base: Path, output_base: Path):
    """
    Generate learning curves from training logs for the given dataset.
    Saves per-run PNGs under:
      <output>/<DATASET>/learning_curves/<group_key>/
    """
    train_path = resolve_train_path(Path(logs_base), dataset_name)
    output_root = Path(output_base) / dataset_name / "learning_curves"
    output_root.mkdir(parents=True, exist_ok=True)

    if not train_path:
        print(f"❌ Not found: no expected 'train' directory for '{dataset_name}' under base '{logs_base}'")
        return

    patterns = ("**/*train*.txt", "**/*.txt", "**/*.log", "**/*.out")
    files: List[Path] = []
    for pat in patterns:
        files.extend(train_path.glob(pat))

    if not files:
        print(f"⚠️ No training log files found under: {train_path}")
        return

    # results[group][list of (label, path)]
    groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

    for f in sorted({p for p in files if p.is_file()}):
        stem = f.stem
        tlabel = stem  # full stem as train label
        arch = detect_arch(tlabel)
        act  = detect_activation(tlabel)
        if not arch or not act:
            continue
        group_key = f"{arch}_{act}"
        if group_key not in MODEL_KEYS:
            continue
        groups[group_key].append((tlabel, f))

    if not groups:
        print("⚠️ No recognizable training runs found.")
        return

    for group_key, runs in groups.items():
        out_dir = output_root / group_key
        out_dir.mkdir(parents=True, exist_ok=True)

        for train_label, path in runs:
            df = parse_training_log(path)
            short = shorten_label(train_label)

            # Loss curves
            _plot_loss(df, title=f"Loss – {short}", out_path=out_dir / f"{short}_loss.png")
            # Accuracy curves
            _plot_acc(df, title=f"Accuracy – {short}", out_path=out_dir / f"{short}_acc.png")

    print(f"✅ Learning curves generated under: {output_root}")
