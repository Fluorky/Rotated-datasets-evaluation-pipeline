# src/analysis/learning_curves.py
"""
Parse training logs and generate combined learning curves:
- Left Y: train loss + val loss (one color)
- Right Y: train accuracy + val accuracy (second color)

Notes:
- Distinguishes arches: cyresnet56, resnet56, cyvgg19, vgg19 (no canonicalization).
- Uses only the TRAIN label (filename stem) to detect (arch, activation).
- Scans logs recursively under .../train (flat or nested layouts).
- Recognizes common metric formats, e.g.:
    Epoch 1/90 - loss: 0.54 - acc: 87.2% - val_loss: 0.60 - val_accuracy: 0.84
    epoch: 2 loss: 0.432 accuracy: 0.905 val_loss: 0.51 val_acc: 88.1%
- Converts accuracy to percent if it looks like fraction (<= 1.5).
- Outputs one combined PNG per run:
    <output>/<DATASET>/learning_curves/<group_key>/<short>_combined.png
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# --- Config: two-color scheme -----------------------------------------------

LOSS_COLOR = "#1f77b4"   # blue for loss (train/val share color; styles differ)
ACC_COLOR  = "#ff7f0e"   # orange for accuracy (train/val share color; styles differ)

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

# Epoch and metric regexes (case-insensitive)
EPOCH_RE = re.compile(r"(?i)\bepoch(?:\s*[:#]|\s+)?\s*(\d+)(?:\s*/\s*\d+)?")

VAL_LOSS_RE = re.compile(r"(?i)\bval(?:[_\s-]?loss)\s*[:=]\s*([0-9]*\.?[0-9]+)")
LOSS_RE     = re.compile(r"(?i)(?<!val_)(?<!val-)(?<!val )(?<!val)\bloss\s*[:=]\s*([0-9]*\.?[0-9]+)")

VAL_ACC_RE  = re.compile(r"(?i)\bval(?:[_\s-]?(?:acc|accuracy))\s*[:=]\s*([0-9]*\.?[0-9]+)%?")
ACC_RE      = re.compile(r"(?i)(?<!val_)(?<!val-)(?<!val )(?<!val)\b(?:acc|accuracy)\s*[:=]\s*([0-9]*\.?[0-9]+)%?")

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
    for token in ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]:  # longer tokens first
        if token in t:
            return token
    return None

def detect_activation(train_label: str) -> Optional[str]:
    t = train_label.lower()
    for act in ["linearpolar", "logpolar"]:
        if act in t:
            return act
    return None

def shorten_label(label: str) -> str:
    # Drop prefixes like "...-xxx-yyy_" to keep plot titles readable
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
        # Detect epoch if present
        epoch = None
        m_ep = EPOCH_RE.search(line)
        if m_ep:
            epoch = int(m_ep.group(1))
            epoch_counter = epoch

        # Find metrics (both train & val if present)
        val_loss = None
        loss     = None
        val_acc  = None
        acc      = None

        m_vl = VAL_LOSS_RE.search(line)
        if m_vl:
            val_loss = float(m_vl.group(1))
        m_l = LOSS_RE.search(line)
        if m_l:
            loss = float(m_l.group(1))

        m_va = VAL_ACC_RE.search(line)
        if m_va:
            val_acc = float(m_va.group(1))
        m_a = ACC_RE.search(line)
        if m_a:
            acc = float(m_a.group(1))

        # If we captured any metric, emit a row
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
    # For duplicate epochs, keep the last seen values
    df = df.drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    return df

def _plot_combined(df: pd.DataFrame, title: str, out_path: Path):
    """
    One figure per run with two colors:
      - Left Y-axis (LOSS_COLOR): loss + val_loss
      - Right Y-axis (ACC_COLOR): acc + val_acc  (in %)
    Train = solid, Val = dashed, same color family.
    """
    if df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # LOSS curves (left axis)
    lines = []
    labels = []
    if "loss" in df and df["loss"].notna().any():
        l1, = ax.plot(df["epoch"], df["loss"], marker="o", color=LOSS_COLOR, label="loss")
        lines.append(l1); labels.append("loss")
    if "val_loss" in df and df["val_loss"].notna().any():
        l2, = ax.plot(df["epoch"], df["val_loss"], marker="o", linestyle="--", alpha=0.8,
                      color=LOSS_COLOR, label="val_loss")
        lines.append(l2); labels.append("val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss", color=LOSS_COLOR)
    ax.tick_params(axis="y", colors=LOSS_COLOR)
    ax.spines["left"].set_color(LOSS_COLOR)
    ax.grid(True, which="both", alpha=0.3)

    # ACC curves (right axis)
    ax2 = ax.twinx()
    if "acc" in df and df["acc"].notna().any():
        l3, = ax2.plot(df["epoch"], df["acc"], marker="s", color=ACC_COLOR, label="acc")
        lines.append(l3); labels.append("acc")
    if "val_acc" in df and df["val_acc"].notna().any():
        l4, = ax2.plot(df["epoch"], df["val_acc"], marker="s", linestyle="--", alpha=0.8,
                       color=ACC_COLOR, label="val_acc")
        lines.append(l4); labels.append("val_acc")
    ax2.set_ylabel("Accuracy [%]", color=ACC_COLOR)
    ax2.tick_params(axis="y", colors=ACC_COLOR)
    ax2.spines["right"].set_color(ACC_COLOR)

    # Legend (single, for both axes)
    ax.legend(lines, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Metrics")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def generate_learning_curves(dataset_name: str, logs_base: Path, output_base: Path):
    """
    Generate combined learning curves from training logs for the given dataset.
    Saves per-run PNGs under:
      <output>/<DATASET>/learning_curves/<group_key>/<short>_combined.png
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

            # Combined plot: loss/val_loss (left, LOSS_COLOR) + acc/val_acc (right, ACC_COLOR)
            _plot_combined(
                df,
                title=f"Learning Curve – {short}",
                out_path=out_dir / f"{short}_combined.png",
            )

    print(f"✅ Learning curves generated under: {output_root}")
