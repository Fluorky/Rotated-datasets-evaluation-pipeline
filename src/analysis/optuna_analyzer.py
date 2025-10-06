# src/analysis/optuna_analyzer.py
"""
Optuna logs analyzer (self-contained).

Features
--------
1) Learning curves per run:
   - Left Y: train_loss (green) + val_loss (blue)
   - Right Y: train_acc (orange) + val_acc (red), in percent
   - Vertical dotted line at the LAST "Model saved at:" checkpoint.

2) One-row accuracy heatmap per run from [TEST] lines (plus CSV).
   - Global 0..100% color scale
   - Wrapped x tick labels
   - Title shows model + transform (linearpolar/logpolar)

CLI (hook in src/cli.py as `optuna-analyze`):
    python -m src.cli optuna-analyze -d GTSRB --optuna-logs "E:\\MasterThesisLogsAll\\optuna_checked\\logs" --output-dir "<OUT>"
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns

# ----------------------- tokens & helpers -----------------------------------

ARCH_TOKENS: List[str] = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]  # longer tokens first
ACT_TOKENS:  List[str] = ["linearpolar", "logpolar"]

def detect_arch_from_text(text: str) -> Optional[str]:
    t = text.lower()
    for tok in ARCH_TOKENS:
        if tok in t:
            return tok
    return None

def detect_act_from_text(text: str) -> Optional[str]:
    t = text.lower()
    for tok in ACT_TOKENS:
        if tok in t:
            return tok
    return None

def shorten_label(label: str) -> str:
    """Drop 'arch-activation_' prefix for nicer plot/heatmap titles."""
    return re.sub(r'^[^-]+-[^_]+_', '', label)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _wrap_tick(s: str, every: int = 22) -> str:
    s = s.replace("_", " ")
    return "\n".join(s[i:i+every] for i in range(0, len(s), every))

def rotation_sort_key(name: str) -> Tuple[int, int]:
    n = name.lower()
    if n.startswith("dataset"):
        base = 0
    elif n.startswith("merged"):
        base = 1
    elif n.startswith("rotated"):
        base = 2
    else:
        base = 3
    m = re.search(r"(\d+)", n)
    ang = int(m.group(1)) if m else 9999
    return (base, ang)

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


# ----------------------- training log parsing -------------------------------

# Epoch markers like "[Epoch 36]" (optionally "/N")
EPOCH_RE = re.compile(r"(?i)\bepoch(?:\s*[:#]|\s+)?\s*(\d+)(?:\s*/\s*\d+)?")

NUM  = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
FRAC = rf"({NUM})\s*/\s*({NUM})"

# Losses
VAL_LOSS_RE   = re.compile(rf"(?i)\b(?:val(?:idation)?[_\s-]?)loss\s*[:=]\s*({NUM})")
TRAIN_LOSS_RE = re.compile(rf"(?i)\b(?:trn|train(?:ing)?)[_\s-]?loss\s*[:=]\s*({NUM})")
LOSS_RE       = re.compile(  # generic 'loss:' but not 'validation ...'
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\bloss\s*[:=]\s*({NUM})"
)

# Accuracies
VAL_ACC_FRAC_RE = re.compile(rf"(?i)\b(?:val(?:idation)?[_\s-]?(?:acc|accuracy|top1|top-1))[^0-9]*{FRAC}")
ACC_FRAC_RE     = re.compile(
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\b(?:acc|accuracy|top1|top-1)[^0-9]*{FRAC}"
)
VAL_ACC_RE = re.compile(
    rf"(?i)\b(?:val(?:idation)?[_\s-]?(?:acc|accuracy|top1|top-1))\s*[:=]\s*(?!{NUM}\s*/)\s*({NUM})%?"
)
ACC_RE = re.compile(
    rf"(?i)(?<!val_)(?<!val-)(?<!val\s)(?<!val)"
    rf"(?<!validation_)(?<!validation-)(?<!validation\s)(?<!validation)"
    rf"\b(?:acc|accuracy|top1|top-1)\s*[:=]\s*(?!{NUM}\s*/)\s*({NUM})%?"
)
PAREN_PCT_RE = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*\)")

CKPT_RE = re.compile(r"(?i)model saved at")

def _to_percent(x: float) -> float:
    """If 0..1 -> %, else leave as-is."""
    return x * 100.0 if 0.0 <= x <= 1.5 else x

def _parse_acc_from_line(line: str, prefer_val: bool) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (train_acc, val_acc) in percent.
    Priority:
      1) fraction A/B  -> %,
      2) percent in parentheses "(xx.x%)" (when prefer_val),
      3) plain numeric after acc/accuracy  (blocked from A/B by negative lookahead)
    """
    acc = val_acc = None

    # 1) fractions
    m = VAL_ACC_FRAC_RE.search(line)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den:
            val_acc = (num / den) * 100.0
    m = ACC_FRAC_RE.search(line)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den:
            if prefer_val:
                val_acc = (num / den) * 100.0
            else:
                acc = (num / den) * 100.0

    # 2) percent in parens
    if prefer_val and val_acc is None:
        m = PAREN_PCT_RE.search(line)
        if m:
            val_acc = float(m.group(1))

    # 3) plain numeric
    if val_acc is None:
        m = VAL_ACC_RE.search(line)
        if m:
            val_acc = _to_percent(float(m.group(1)))
    m = ACC_RE.search(line)
    if m:
        v = _to_percent(float(m.group(1)))
        if prefer_val and val_acc is None:
            val_acc = v
        elif acc is None:
            acc = v

    return acc, val_acc

# If accuracy parsed as counts (>100), infer denominator from logs text
_DENOM_RE = re.compile(r"(?i)\bAccuracy\s*:\s*\d+\s*/\s*(\d+)")
_ANY_FRAC_DENOM_RE = re.compile(r"\b\d+\s*/\s*(\d+)\b")

def _coerce_accuracy_to_percent(df: pd.DataFrame, log_text: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    def needs(series: pd.Series) -> bool:
        return series.notna().any() and series.max() > 100.0

    if ("acc" in df and needs(df["acc"])) or ("val_acc" in df and needs(df["val_acc"])):
        cand = [int(x) for x in _DENOM_RE.findall(log_text)] or [int(x) for x in _ANY_FRAC_DENOM_RE.findall(log_text)]
        denom = Counter(cand).most_common(1)[0][0] if cand else None
        if denom and denom > 0:
            for col in ("acc", "val_acc"):
                if col in df:
                    df[col] = df[col].apply(lambda v: (v / denom) * 100.0 if pd.notna(v) and v > 100.0 else v)

    for col in ("acc", "val_acc"):
        if col in df and df[col].notna().any() and df[col].max() <= 1.5:
            df[col] = df[col] * 100.0

    return df

def parse_training_log(path: Path) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Parse a training log into a DataFrame with columns:
        epoch, loss, val_loss, acc, val_acc  (accuracy in % if possible)
    Returns (df, last_checkpoint_epoch).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "acc", "val_acc"]), None

    rows: List[Dict[str, float]] = []
    current_epoch: Optional[int] = None
    last_ckpt_epoch: Optional[int] = None
    cur = {"epoch": None, "loss": None, "val_loss": None, "acc": None, "val_acc": None}


    TL_WITH_EPOCH_RE = re.compile(
        rf"(?mi)\[\s*epoch\s*(\d+)\s*]\s*.*?\btrain(?:ing)?\s*loss\s*[:=]\s*({NUM})"
    )
    explicit_train_loss: Dict[int, float] = {
        int(m.group(1)): float(m.group(2)) for m in TL_WITH_EPOCH_RE.finditer(text)
    }

    def flush():
        if cur["epoch"] is not None and any(v is not None for k, v in cur.items() if k != "epoch"):
            rows.append({
                "epoch": int(cur["epoch"]),
                "loss": cur["loss"],
                "val_loss": cur["val_loss"],
                "acc": cur["acc"],
                "val_acc": cur["val_acc"],
            })

    for line in text.splitlines():
        # Epoch boundary
        m_ep = EPOCH_RE.search(line)
        if m_ep:
            if current_epoch is not None:
                flush()
            current_epoch = int(m_ep.group(1))
            cur = {"epoch": current_epoch, "loss": None, "val_loss": None, "acc": None, "val_acc": None}

        if current_epoch is None:
            current_epoch = 1
            cur["epoch"] = current_epoch

        # Losses
        m_vl = VAL_LOSS_RE.search(line)
        if m_vl:
            cur["val_loss"] = float(m_vl.group(1))

        m_tl = TRAIN_LOSS_RE.search(line) or re.search(rf"(?i)\btrain(?:ing)?\s+loss\s*[:=]\s*({NUM})", line)
        if m_tl:
            cur["loss"] = float(m_tl.group(1))

        if cur["loss"] is None:
            m_l = LOSS_RE.search(line)
            if m_l:
                cur["loss"] = float(m_l.group(1))

        # Accuracies
        prefer_val = bool(m_vl) or ("validation" in line.lower())
        acc, val_acc = _parse_acc_from_line(line, prefer_val=prefer_val)
        if acc is not None:
            cur["acc"] = acc
        if val_acc is not None:
            cur["val_acc"] = val_acc

        # Checkpoint
        if CKPT_RE.search(line):
            last_ckpt_epoch = current_epoch

    flush()


    df = pd.DataFrame(rows).sort_values("epoch")
    if not df.empty:
        df = df.drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
        df = _coerce_accuracy_to_percent(df, text)

    if explicit_train_loss:
        if df.empty:
            df = pd.DataFrame({
                "epoch": sorted(explicit_train_loss.keys()),
                "loss":  [explicit_train_loss[e] for e in sorted(explicit_train_loss.keys())],
                "val_loss": [None]*len(explicit_train_loss),
                "acc": [None]*len(explicit_train_loss),
                "val_acc": [None]*len(explicit_train_loss),
            })
        else:

            tl_map = explicit_train_loss
            df["loss"] = df.apply(
                lambda r: tl_map.get(int(r["epoch"]), r["loss"]),
                axis=1
            )

    return df, last_ckpt_epoch


# ----------------------- [TEST] section parsing -----------------------------

TEST_LINE_RE = re.compile(
    r"""(?mix)
    ^\s*\[TEST\]\s+
    (\S+)\s+                # token: <train_label>_test_on_<test_case>
    loss\s*=\s*([0-9.]+)\s+
    acc\s*=\s*([0-9.]+)\s*%?\s*$
    """
)

def parse_optuna_test_results(path: Path) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Parse one Optuna .log file and return:
      (train_label, { test_case -> accuracy(%) }).
    If the same test_case appears multiple times, the LAST value wins.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, {}

    results: Dict[str, float] = {}
    train_label: Optional[str] = None

    for m in TEST_LINE_RE.finditer(text):
        token = m.group(1).strip()
        acc = float(m.group(3))
        if "_test_on_" in token:
            tr, _, te = token.partition("_test_on_")
        else:
            tr, te = token, token
        if train_label is None:
            train_label = tr
        results[te] = acc  # last wins

    return train_label, results


# ----------------------- plotting ------------------------------------------

def _plot_learning_curve(df: pd.DataFrame, title: str, out_path: Path, ckpt_epoch: Optional[int]) -> None:
    if df.empty:
        return

    _ensure_dir(out_path.parent)

    TRAIN_LOSS_COLOR = "#2ca02c"  # green
    VAL_LOSS_COLOR   = "#1f77b4"  # blue
    TRAIN_ACC_COLOR  = "#ff7f0e"  # orange
    VAL_ACC_COLOR    = "#d62728"  # red
    CKPT_LINE_COLOR  = "#555555"

    fig, ax = plt.subplots(figsize=(10, 6))
    lines, labels = [], []

    # Loss (left)
    if df["loss"].notna().any():
        l1, = ax.plot(df["epoch"], df["loss"], marker="o", color=TRAIN_LOSS_COLOR, label="train_loss")
        lines.append(l1); labels.append("train_loss")
    if df["val_loss"].notna().any():
        l2, = ax.plot(df["epoch"], df["val_loss"], marker="o", linestyle="--", alpha=0.95,
                      color=VAL_LOSS_COLOR, label="val_loss")
        lines.append(l2); labels.append("val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.3)

    # Accuracy (right)
    ax2 = ax.twinx()
    # if "acc" in df and df["acc"].notna().any():
    #     l3, = ax2.plot(df["epoch"], df["acc"], marker="s", color=TRAIN_ACC_COLOR, label="train_acc")
    #     lines.append(l3); labels.append("train_acc")
    if "val_acc" in df and df["val_acc"].notna().any():
        l4, = ax2.plot(df["epoch"], df["val_acc"], marker="s", linestyle="--", alpha=0.95,
                       color=VAL_ACC_COLOR, label="val_acc")
        lines.append(l4); labels.append("val_acc")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    # Last checkpoint
    if ckpt_epoch is not None:
        v = ax.axvline(x=ckpt_epoch, color=CKPT_LINE_COLOR, linestyle=":", linewidth=2, label=f"ckpt@{ckpt_epoch}")
        lines.append(v); labels.append(f"ckpt@{ckpt_epoch}")

    # Legend
    ncols = min(4, max(1, len(labels)))
    fig.legend(handles=lines, labels=labels,
               loc="upper center", bbox_to_anchor=(0.5, -0.08),
               ncol=ncols, frameon=True, title="Metrics")
    fig.subplots_adjust(bottom=0.18)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------- public API -----------------------------------------

def generate_optuna_learning_curves(dataset_name: str, optuna_logs_dir: Path, output_base: Path) -> None:
    """
    Build combined learning curves (one PNG per Optuna run).
    Output:
      <output>/<DATASET>/optuna/learning_curves/<group>/<short>_combined.png
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

        # Prefer run label from [TEST]
        train_label, _ = parse_optuna_test_results(f)
        full_label = train_label or f.stem             # full string (with arch/act)
        short_label = shorten_label(full_label)        # nice tail

        # Detect model/transform from the FULL label (not shortened)
        arch = detect_arch_from_text(full_label) or detect_arch_from_text(f.stem) or "model?"
        act  = detect_act_from_text(full_label)  or detect_act_from_text(f.stem)  or "transform?"
        group = f"{arch}_{act}".strip("_") if arch and act else "unknown"

        out_dir = out_root / group
        _ensure_dir(out_dir)

        title = f"Learning Curve – {arch} · {act} – {short_label}"
        _plot_learning_curve(df, title, out_dir / f"{short_label}_combined.png", ckpt_epoch)

    print(f"✅ Optuna learning curves saved under: {out_root}")


def generate_optuna_heatmaps(dataset_name: str, optuna_logs_dir: Path, output_base: Path) -> None:
    """
    Build a 1-row accuracy heatmap (and CSV) from [TEST] lines per Optuna run.
    Output:
      <output>/<DATASET>/optuna/heatmaps/<group>/<run>_accuracy.csv
      <output>/<DATASET>/optuna/heatmaps/<group>/<run>_heatmap.png
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
            continue

        full_label = train_label or f.stem
        short_label = shorten_label(full_label)

        # semantic column order
        cols = sorted(results.keys(), key=rotation_sort_key)

        # CSV (original column names)
        df_csv = pd.DataFrame([[results[c] for c in cols]], columns=cols, index=[short_label])

        arch = detect_arch_from_text(full_label) or detect_arch_from_text(f.stem) or "model?"
        act  = detect_act_from_text(full_label)  or detect_act_from_text(f.stem)  or "transform?"
        group = f"{arch}_{act}".strip("_") if arch and act else "unknown"

        out_dir = out_root / group
        _ensure_dir(out_dir)

        csv_path = out_dir / f"{short_label}_accuracy.csv"
        df_csv.to_csv(csv_path, float_format="%.2f")

        # Plot with wrapped ticks for readability
        plot_cols = [_wrap_tick(c, every=22) for c in cols]
        df_plot = pd.DataFrame([[results[c] for c in cols]], columns=plot_cols, index=[short_label])

        fig_w = min(max(14, len(cols) * 0.7), 36)  # 14..36 inches
        fig_h = 6.0
        plt.figure(figsize=(fig_w, fig_h), dpi=200)

        ax = sns.heatmap(
            df_plot.astype(float),
            annot=True, fmt=".2f",
            cmap="jet",
            vmin=0, vmax=100,
            cbar_kws={"label": "Accuracy [%]"},
            square=False,
            annot_kws={"fontsize": 8},
        )
        ax.set_aspect("auto")

        plt.title(f"Optuna Accuracy – {arch} · {act} – {short_label}")
        plt.xlabel("Test dataset")
        plt.ylabel("Run")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        png_path = out_dir / f"{short_label}_heatmap.png"
        plt.savefig(png_path, dpi=200)
        plt.close()

        print(f"🧾 {csv_path}")
        print(f"🖼️ {png_path}")

    print(f"✅ Optuna heatmaps saved under: {out_root}")
