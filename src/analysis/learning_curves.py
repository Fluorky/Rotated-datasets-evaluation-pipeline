# src/analysis/learning_curves.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

# --- Colors (distinct) ------------------------------------------------------
TRAIN_LOSS_COLOR = "#2ca02c"  # green
VAL_LOSS_COLOR   = "#1f77b4"  # blue
TRAIN_ACC_COLOR  = "#ff7f0e"  # orange
VAL_ACC_COLOR    = "#d62728"  # red

# Supported groups
MODEL_KEYS: List[str] = [
    "cyresnet56_logpolar", "cyresnet56_linearpolar",
    "cyvgg19_logpolar",    "cyvgg19_linearpolar",
    "resnet56_logpolar",   "resnet56_linearpolar",
    "vgg19_logpolar",      "vgg19_linearpolar",
]
ARCH_TOKENS: List[str] = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]
ACTIVATIONS: List[str] = ["linearpolar", "logpolar"]

# --- Regexes (case-insensitive) ---------------------------------------------
EPOCH_RE = re.compile(r"(?i)\bepoch(?:\s*[:#]|\s+)?\s*(\d+)(?:\s*/\s*\d+)?")

NUM  = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
FRAC = rf"({NUM})\s*/\s*({NUM})"

# val_loss / train_loss / generic loss
VAL_LOSS_RE   = re.compile(rf"(?i)\b(?:val(?:idation)?[_\s-]?)loss\s*[:=]\s*({NUM})")
TRAIN_LOSS_RE = re.compile(rf"(?i)\b(?:trn|train(?:ing)?)[_\s-]?loss\s*[:=]\s*({NUM})")
LOSS_RE       = re.compile(
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\bloss\s*[:=]\s*({NUM})"
)

# accuracy as fraction A/B
VAL_ACC_FRAC_RE = re.compile(rf"(?i)\b(?:val(?:idation)?[_\s-]?(?:acc|accuracy|top1|top-1))[^0-9]*{FRAC}")
ACC_FRAC_RE     = re.compile(rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
                             rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
                             rf"\b(?:acc|accuracy|top1|top-1)[^0-9]*{FRAC}")

# accuracy as single number (block fractions with negative lookahead)
VAL_ACC_RE = re.compile(
    rf"(?i)\b(?:val(?:idation)?[_\s-]?(?:acc|accuracy|top1|top-1))\s*[:=]\s*(?!{NUM}\s*/)\s*({NUM})%?"
)
ACC_RE = re.compile(
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\b(?:acc|accuracy|top1|top-1)\s*[:=]\s*(?!{NUM}\s*/)\s*({NUM})%?"
)

VAL_KEYWORD_RE = re.compile(r"(?i)\bval(?:idation)?\b")  # heuristic

# ---------------------------------------------------------------------------

def resolve_train_path(logs_base: Path, dataset_name: str) -> Optional[Path]:
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
    for token in ARCH_TOKENS:
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
    return re.sub(r'^.+?-[^-]+-[^_]+_', '', label)

def _to_percent_if_fraction(x: float) -> float:
    """If 0..1 -> %, else leave as-is."""
    return x * 100.0 if x <= 1.5 else x

def _parse_acc_from_line(line: str, prefer_val: bool) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (acc, val_acc) in percent if possible.
    If prefer_val is True and we see a plain 'Accuracy/acc' without 'val',
    we treat it as val_acc (for lines like 'Validation loss: ..., Accuracy: ...').
    """
    acc = val_acc = None
    acc_frac_found = val_acc_frac_found = False

    # Fraction first (A/B)
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

    # Plain numeric (maybe already %), but only if NOT a fraction already found
    if not val_acc_frac_found:
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

def parse_training_log(path: Path) -> pd.DataFrame:
    """
    Parse a training log into a DataFrame with columns:
    epoch, loss, val_loss, acc, val_acc (accuracy in percent if possible).
    Aggregates multiple lines per epoch; prefers explicit train-loss if present.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "acc", "val_acc"])

    rows: List[Dict[str, float]] = []
    current_epoch: Optional[int] = None
    current: Dict[str, Optional[float]] = {"epoch": None, "loss": None, "val_loss": None, "acc": None, "val_acc": None}

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
        # Epoch boundary
        m_ep = EPOCH_RE.search(line)
        if m_ep:
            if current_epoch is not None:
                flush()
            current_epoch = int(m_ep.group(1))
            current = {"epoch": current_epoch, "loss": None, "val_loss": None, "acc": None, "val_acc": None}

        if current_epoch is None:
            current_epoch = 1
            current["epoch"] = current_epoch

        # Losses (val first, then explicit train, then generic)
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

        # Accuracy (with context heuristic)
        prefer_val = bool(m_vl) or bool(VAL_KEYWORD_RE.search(line))
        acc, val_acc = _parse_acc_from_line(line, prefer_val=prefer_val)
        if acc is not None:
            current["acc"] = acc
        if val_acc is not None:
            current["val_acc"] = val_acc

    flush()

    if not rows:
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "acc", "val_acc"])

    df = pd.DataFrame(rows).sort_values("epoch")
    df = df.drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    return df

def _plot_combined(df: pd.DataFrame, title: str, out_path: Path):
    """
    One figure per run:
      - Left Y-axis: train_loss (green) + val_loss (blue)
      - Right Y-axis: train_acc (orange) + val_acc (red)
    Train = solid, Val = dashed.
    """
    if df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    lines, labels = [], []

    # --- LOSS (left axis) ---
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

    # --- ACCURACY (right axis) ---
    ax2 = ax.twinx()
    acc_series = pd.concat(
        [df.get("acc", pd.Series(dtype=float)), df.get("val_acc", pd.Series(dtype=float))],
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
    if not acc_series.empty and acc_series.max() <= 100.0 and acc_series.min() >= 0.0:
        ax2.set_ylim(0, 100)

    # --- Legend below ---
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
            df = parse_training_log(path)
            short = shorten_label(train_label)

            _plot_combined(
                df,
                title=f"Learning Curve – {short}",
                out_path=out_dir / f"{short}_combined.png",
            )

    print(f"✅ Learning curves generated under: {output_root}")
