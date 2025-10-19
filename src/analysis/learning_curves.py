# src/analysis/learning_curves.py
"""
Generate combined learning curves from training logs.

Per run it plots:
- Left Y:  train_loss (green, solid) + val_loss (blue, dashed)
- Right Y: train_acc  (orange, solid) + val_acc  (red, dashed) — in %

Extras:
- Detects the LAST "Model saved at:" checkpoint and marks it with a vertical line.
- Robust parsing:
    * "Train/Training/trn loss", plain "loss", and "Validation loss".
    * Accuracy as A/B -> %, "(xx.xx%)", or plain number (0..1 -> %, >100 treated as count).
    * Converts raw counts to % using dataset sizes from lines like:
      "✔️ Train data: 35289 samples" / "✔️ Validation data: 3920 samples".
- Works when --logs-dir points either to the parent of json_* or directly to json_DATASET.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ---------- Colors ----------
TRAIN_LOSS_COLOR = "#2ca02c"  # green
VAL_LOSS_COLOR   = "#1f77b4"  # blue
TRAIN_ACC_COLOR  = "#ff7f0e"  # orange
VAL_ACC_COLOR    = "#d62728"  # red
CKPT_LINE_COLOR  = "#555555"  # grey

# ---------- Grouping tokens (optional) ----------
MODEL_KEYS: List[str] = [
    "cyresnet56_logpolar", "cyresnet56_linearpolar",
    "cyvgg19_logpolar",    "cyvgg19_linearpolar",
    "resnet56_logpolar",   "resnet56_linearpolar",
    "vgg19_logpolar",      "vgg19_linearpolar",
]
ARCH_TOKENS: List[str] = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]
ACTIVATIONS: List[str] = ["linearpolar", "logpolar"]

# ---------- Regexes ----------
EPOCH_RE = re.compile(r"(?i)\bepoch(?:\s*[:#]|\s+)?\s*(\d+)(?:\s*/\s*\d+)?")

NUM  = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
FRAC = rf"({NUM})\s*/\s*({NUM})"

# Loss
VAL_LOSS_RE   = re.compile(rf"(?i)\b(?:val(?:idation)?[_\s-]?)loss\s*[:=]\s*({NUM})")
TRAIN_LOSS_RE = re.compile(rf"(?i)\b(?:trn|train(?:ing)?)[_\s-]?loss\s*[:=]\s*({NUM})")
LOSS_RE       = re.compile(
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\bloss\s*[:=]\s*({NUM})"
)

# Accuracy as fraction A/B
VAL_ACC_FRAC_RE = re.compile(rf"(?i)\b(?:val(?:idation)?[_\s-]?(?:acc|accuracy|top1|top-1))[^0-9]*{FRAC}")
ACC_FRAC_RE     = re.compile(
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\b(?:acc|accuracy|top1|top-1)[^0-9]*{FRAC}"
)

# Accuracy as plain number (block the A/B form)
VAL_ACC_RE = re.compile(
    rf"(?i)\b(?:val(?:idation)?[_\s-]?(?:acc|accuracy|top1|top-1))\s*[:=]\s*(?!{NUM}\s*/)\s*({NUM})%?"
)
ACC_RE = re.compile(
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\b(?:acc|accuracy|top1|top-1)\s*[:=]\s*(?!{NUM}\s*/)\s*({NUM})%?"
)

# Explicit percent in parentheses: "(80.94%)"
PAREN_PCT_RE = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\)")
VAL_KEYWORD_RE = re.compile(r"(?i)\bval(?:idation)?\b")

# Dataset sizes
TRAIN_SAMPLES_RE = re.compile(r"(?i)\btrain\s+data:\s*([0-9][0-9,]*)\s*samples")
VAL_SAMPLES_RE   = re.compile(r"(?i)\bval(?:idation)?\s+data:\s*([0-9][0-9,]*)\s*samples")

# Checkpoint (accepts optional emoji)
CKPT_RE = re.compile(r"(?i)(?:💾\s*)?model\s+saved\s+at\s*:")


# ---------- Small helpers ----------

def resolve_train_path(logs_base: Path, dataset_name: str) -> Optional[Path]:
    """Find '<...>/train' under various common layouts."""
    for p in [
        logs_base / "train",
        logs_base / f"json_{dataset_name}" / "train",
        logs_base / f"json_{dataset_name.lower()}" / "train",
        logs_base / f"json_{dataset_name.upper()}" / "train",
        logs_base / dataset_name / "train",
        logs_base / dataset_name.lower() / "train",
        logs_base / dataset_name.upper() / "train",
    ]:
        if p.exists() and p.is_dir():
            return p
    return None


def detect_arch(train_label: str) -> Optional[str]:
    t = train_label.lower()
    for token in ARCH_TOKENS:  # order matters (cy* first)
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
    """Drop long prefixes like '...-xxx-yyy_' for nicer titles."""
    return re.sub(r'^.+?-[^-]+-[^_]+_', '', label)


def _to_percent_if_fraction(x: float) -> float:
    """If x looks like 0..1 -> convert to percent; otherwise return as-is."""
    return x * 100.0 if x <= 1.5 else x


def _parse_acc_from_line(line: str, prefer_val: bool) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (acc, val_acc) in percent if possible.
    Priority per line:
      1) fraction A/B -> %
      2) percent in parentheses "(xx.x%)" (if prefer_val)
      3) plain number after acc/accuracy (block A/B with negative lookahead)
    If prefer_val=True and we see plain "Accuracy:" on a validation line,
    treat it as val_acc.
    """
    acc = val_acc = None
    acc_frac_found = val_acc_frac_found = False

    # 1) Fractions
    m = VAL_ACC_FRAC_RE.search(line)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            val_acc = (num / den) * 100.0
            val_acc_frac_found = True

    m = ACC_FRAC_RE.search(line)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            if prefer_val:
                val_acc = (num / den) * 100.0
                val_acc_frac_found = True
            else:
                acc = (num / den) * 100.0
                acc_frac_found = True

    # 2) Explicit percent "(xx.xx%)" – usually appears on validation lines
    if prefer_val and val_acc is None:
        m = PAREN_PCT_RE.search(line)
        if m:
            val_acc = float(m.group(1))

    # 3) Plain numbers
    if not val_acc_frac_found and val_acc is None:
        m = VAL_ACC_RE.search(line)
        if m:
            val_acc = _to_percent_if_fraction(float(m.group(1)))

    if not acc_frac_found:
        m = ACC_RE.search(line)
        if m:
            value = _to_percent_if_fraction(float(m.group(1)))
            if prefer_val and val_acc is None:
                val_acc = value
            elif acc is None:
                acc = value

    return acc, val_acc


# ---------- Core parsing & plotting ----------

def parse_training_log(path: Path) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Parse one training log into a DataFrame with columns:
      epoch, loss, val_loss, acc, val_acc
    Returns (df, last_checkpoint_epoch).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "acc", "val_acc"]), None

    rows: List[Dict[str, float]] = []
    current_epoch: Optional[int] = None
    current: Dict[str, Optional[float]] = {"epoch": None, "loss": None, "val_loss": None, "acc": None, "val_acc": None}

    # dataset sizes to convert counts to %
    train_n: Optional[int] = None
    val_n: Optional[int] = None

    # last checkpoint epoch
    last_ckpt_epoch: Optional[int] = None

    def flush():
        if current["epoch"] is not None and any(v is not None for k, v in current.items() if k != "epoch"):
            rows.append({
                "epoch": int(current["epoch"]),
                "loss": current["loss"],
                "val_loss": current["val_loss"],
                "acc": current["acc"],
                "val_acc": current["val_acc"],
            })

    for line in text.splitlines():
        # dataset sizes (often printed once near the top)
        m_trn = TRAIN_SAMPLES_RE.search(line)
        if m_trn:
            train_n = int(m_trn.group(1).replace(",", ""))
        m_val = VAL_SAMPLES_RE.search(line)
        if m_val:
            val_n = int(m_val.group(1).replace(",", ""))

        # epoch boundary
        m_ep = EPOCH_RE.search(line)
        if m_ep:
            if current_epoch is not None:
                flush()
            current_epoch = int(m_ep.group(1))
            current = {"epoch": current_epoch, "loss": None, "val_loss": None, "acc": None, "val_acc": None}

        if current_epoch is None:
            current_epoch = 1
            current["epoch"] = current_epoch

        # losses: val first, then explicit train, then generic
        m_vl = VAL_LOSS_RE.search(line)
        if m_vl:
            current["val_loss"] = float(m_vl.group(1))

        m_tl = TRAIN_LOSS_RE.search(line)
        if m_tl:
            current["loss"] = float(m_tl.group(1))
        if current["loss"] is None:
            m_l = LOSS_RE.search(line)
            if m_l:
                current["loss"] = float(m_l.group(1))

        # accuracy (with validation context heuristic)
        prefer_val = bool(m_vl) or bool(VAL_KEYWORD_RE.search(line))
        acc, val_acc = _parse_acc_from_line(line, prefer_val=prefer_val)

        # convert raw counts to percent if we know dataset sizes
        if acc is not None and acc > 100 and train_n:
            if acc <= train_n * 1.01:
                acc = (acc / train_n) * 100.0
        if val_acc is not None and val_acc > 100 and val_n:
            if val_acc <= val_n * 1.01:
                val_acc = (val_acc / val_n) * 100.0

        if acc is not None:
            current["acc"] = acc
        if val_acc is not None:
            current["val_acc"] = val_acc

        # checkpoint detection: keep the LAST occurrence
        if CKPT_RE.search(line):
            if current_epoch is not None:
                last_ckpt_epoch = current_epoch

    flush()

    df = pd.DataFrame(rows, columns=["epoch", "loss", "val_loss", "acc", "val_acc"])
    if df.empty:
        return df, last_ckpt_epoch

    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    return df, last_ckpt_epoch


def _plot_combined(df: pd.DataFrame, title: str, out_path: Path, ckpt_epoch: Optional[int] = None):
    """
    One figure per run:
      - Left Y: train_loss (green) + val_loss (blue)
      - Right Y: train_acc (orange) + val_acc (red) in %
      - Vertical dotted line at the last checkpoint epoch (if present).
    """
    if df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    lines, labels = [], []

    # Loss (left)
    if "loss" in df and df["loss"].notna().any():
        l1, = ax.plot(df["epoch"], df["loss"], marker="o",
                      color=TRAIN_LOSS_COLOR, label="train_loss")
        lines.append(l1); labels.append("train_loss")

    if "val_loss" in df and df["val_loss"].notna().any():
        l2, = ax.plot(df["epoch"], df["val_loss"], marker="o", linestyle="--", alpha=0.95,
                      color=VAL_LOSS_COLOR, label="val_loss")
        lines.append(l2); labels.append("val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)

    # Accuracy (right)
    ax2 = ax.twinx()
    acc_series = pd.concat(
        [df.get("acc", pd.Series(dtype=float)),
         df.get("val_acc", pd.Series(dtype=float))],
        ignore_index=True
    ).dropna()

    if "acc" in df and df["acc"].notna().any():
        l3, = ax2.plot(df["epoch"], df["acc"], marker="s",
                       color=TRAIN_ACC_COLOR, label="train_acc")
        lines.append(l3); labels.append("train_acc")

    if "val_acc" in df and df["val_acc"].notna().any():
        l4, = ax2.plot(df["epoch"], df["val_acc"], marker="s", linestyle="--", alpha=0.95,
                       color=VAL_ACC_COLOR, label="val_acc")
        lines.append(l4); labels.append("val_acc")

    ax2.set_ylabel("Accuracy [%]")
    is_percent = (not acc_series.empty and acc_series.min() >= 0.0 and acc_series.max() <= 100.0)
    if is_percent:
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    # Checkpoint marker (vertical line on the left axis)
    if ckpt_epoch is not None:
        v = ax.axvline(x=ckpt_epoch, color=CKPT_LINE_COLOR, linestyle=":", linewidth=2,
                       label=f"ckpt@{ckpt_epoch}")
        lines.append(v); labels.append(f"ckpt@{ckpt_epoch}")

    # Legend below
    ncols = min(4, max(1, len(labels)))
    fig.legend(handles=lines, labels=labels,
               loc="upper center", bbox_to_anchor=(0.5, -0.08),
               ncol=ncols, frameon=True, title="Metrics")
    fig.subplots_adjust(bottom=0.18)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_learning_curves(dataset_name: str, logs_base: Path, output_base: Path):
    """
    Scan training logs and save per-run PNGs under:
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

    groups: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

    for f in sorted({p for p in files if p.is_file()}):
        stem = f.stem
        tlabel = stem
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
            df, ckpt_epoch = parse_training_log(path)
            short = shorten_label(train_label)
            _plot_combined(
                df,
                title=f"Learning Curve – {short}",
                out_path=out_dir / f"{short}_combined.png",
                ckpt_epoch=ckpt_epoch,
            )

    print(f"✅ Learning curves generated under: {output_root}")
