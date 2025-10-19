# -*- coding: utf-8 -*-
"""
optuna_delta_table.py

Build a comparison table (Optuna-tuned run vs baseline family summary) for one dataset.

Outputs:
    <out>/optuna_delta_<dataset_lower>_micro.csv

Columns:
    run, arch, act, avg, AUC_theta, worst,
    base_avg, base_auc, base_worst,
    Δavg, ΔAUC, Δworst

Usage:
    python scripts/optuna_delta_table.py \
        --dataset LEGO \
        --optuna-logs "E:\\MasterThesisLogsAll\\optuna_checked\\logs" \
        --out results\\fig \
        --family-summary results\\exports_family\\LEGO\\micro\\family_summary_LEGO_micro.csv \
        --print-mapping
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------ dataset/file matching ------------------------------

def _ds_norm(s: str) -> str:
    return s.strip().lower().replace("-", "_")

def _aliases(ds: str) -> set[str]:
    ds = _ds_norm(ds)
    return {
        ds,
        ds.replace("_", "-"),
    }

def matches_dataset(filename: str, dataset: str) -> bool:
    """
    Strict-enough matcher that avoids mixing GTSRB with GTSRB_RGB.

    Positive examples:
      LEGO_dataset_LEGO_non_rotated_...
      dataset_LEGO_non_rotated_...
      custom_dataset_LEGO_non_rotated_...
      GTSRB-RGB_... (when dataset=GTSRB_RGB)

    Negative examples:
      if dataset=GTSRB -> filenames that contain GTSRB_RGB must NOT match.
    """
    name = filename.lower()
    ds = _ds_norm(dataset)

    # Special disambiguation for GTSRB vs GTSRB_RGB
    if ds == "gtsrb":
        if "gtsrb_rgb" in name or "gtsrb-rgb" in name:
            return False
    if ds == "gtsrb_rgb":
        if not ("gtsrb_rgb" in name or "gtsrb-rgb" in name):
            return False

    # Prefer explicit "...dataset_<ds>_non_rotated..." or "<ds>_dataset_<ds>_non_rotated..."
    for a in _aliases(ds):
        patt1 = rf'(?i)(?:^|[^a-z0-9])dataset[_-]{re.escape(a)}[_-]non[_-]rotated'
        patt2 = rf'(?i)(?:^|[^a-z0-9]){re.escape(a)}[_-]dataset[_-]{re.escape(a)}[_-]non[_-]rotated'
        if re.search(patt1, name) or re.search(patt2, name):
            return True

    # Fallback: token-level match, guarded by non-alnum boundaries
    for a in _aliases(ds):
        patt = rf'(?i)(?:^|[^a-z0-9]){re.escape(a)}(?:[^a-z0-9]|$)'
        if re.search(patt, name):
            return True

    return False


# ------------------------------ arch/act tokenization ------------------------------

ARCH_TOKENS = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]
ACT_TOKENS = ["linearpolar", "logpolar", "linear", "log"]

def _norm_arch(s: str) -> str:
    t = s.lower()
    for tok in ARCH_TOKENS:
        if tok in t:
            return tok
    return t

def _norm_act(s: str) -> str:
    t = s.lower()
    if "logpolar" in t or (("log" in t) and ("polar" in t or t.strip() == "log")):
        return "logpolar"
    if "linearp" in t or (("linear" in t) and ("polar" in t or t.strip() == "linear")):
        return "linearpolar"
    return t

def detect_arch(text: str) -> Optional[str]:
    return next((tok for tok in ARCH_TOKENS if tok in text.lower()), None)

def detect_act(text: str) -> Optional[str]:
    t = text.lower()
    if "logpolar" in t or t.strip() == "log":
        return "logpolar"
    if "linearpolar" in t or t.strip() == "linear":
        return "linearpolar"
    return None


# ------------------------------ Δθ helpers ------------------------------

ANGLE_TOKEN = re.compile(
    r'(?i)(?:rotated-(\d+)(?:-(\d+))?)|(?:range[_-](\d+)[_-](\d+))|(?:full[_-]0[_-]360)'
)

def _interval_from_token(name: str) -> Optional[Tuple[float, float]]:
    s = name.lower()
    m = ANGLE_TOKEN.search(s)
    if not m:
        if "non_rotated" in s:
            return (0.0, 0.0)
        return None
    if m.group(1):  # rotated-a or rotated-a-b
        a = float(m.group(1)); b = float(m.group(2)) if m.group(2) else float(m.group(1))
        return (a, b)
    if m.group(3):  # range_a_b
        a = float(m.group(3)); b = float(m.group(4))
        return (a, b)
    return (0.0, 360.0)  # full_0_360

def _center_deg(iv: Tuple[float, float]) -> float:
    a, b = iv
    if b >= a:
        return (a + b) / 2.0
    return (a + ((b + 360.0 - a) / 2.0)) % 360.0

def _delta_deg(train_name: str, test_name: str) -> Optional[float]:
    it = _interval_from_token(train_name)
    ie = _interval_from_token(test_name)
    if not it or not ie:
        return None
    ct = _center_deg(it); ce = _center_deg(ie)
    d = abs(ct - ce) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d

def _bin_delta(d: float, step: int = 15) -> int:
    b = int(round(d / step) * step)
    return min(b, 180)


# ------------------------------ parse Optuna [TEST] lines ------------------------------

TEST_LINE_RE = re.compile(
    r"""(?mix)
    ^\s*\[TEST\]\s+
    (\S+)\s+                 # <train_label>_test_on_<test_case>
    loss\s*=\s*([0-9.]+)\s+
    acc\s*=\s*([0-9.]+)\s*%?\s*$   # accuracy (maybe in %)
    """
)

def parse_optuna_test_results(path: Path) -> Tuple[Optional[str], Dict[str, float]]:
    """Return (train_label, { test_case -> accuracy_fraction(0..1) })."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, {}

    results: Dict[str, float] = {}
    train_label: Optional[str] = None

    for m in TEST_LINE_RE.finditer(text):
        token = m.group(1).strip()
        acc = float(m.group(3))
        acc = acc / 100.0 if acc > 1.5 else acc  # normalize to 0..1

        if "_test_on_" in token:
            tr, _, te = token.partition("_test_on_")
        else:
            tr, te = token, token

        if train_label is None:
            train_label = tr
        results[te] = acc

    return train_label, results


# ------------------------------ metrics from Acc(Δθ) ------------------------------

def acc_delta_curve_and_auc(train_label: str,
                            test_map: Dict[str, float],
                            theta_step: int = 15) -> Tuple[list[int], list[float], float, float]:
    """
    Build Acc(Δθ) binned every theta_step degrees and compute:
      xs: [0, theta_step, ..., 180]
      ys: mean accuracy in each bin (forward/back filled)
      auc_norm: trapezoidal AUC normalized by 180
      worst: min(ys)
    """
    by_bin: Dict[int, List[float]] = defaultdict(list)
    for test_name, acc in test_map.items():
        d = _delta_deg(train_label, test_name)
        if d is None:
            continue
        by_bin[_bin_delta(d, theta_step)].append(acc)

    xs = list(range(0, 181, theta_step))
    ys: List[Optional[float]] = []
    for x in xs:
        vals = by_bin.get(x, [])
        ys.append(float(np.mean(vals)) if vals else None)

    # forward fill then back-fill head
    last = None
    for i, v in enumerate(ys):
        if v is None:
            ys[i] = last
        else:
            last = v
    first_known = next((v for v in ys if v is not None), 0.0)
    ys = [first_known if v is None else v for v in ys]  # type: ignore

    auc = 0.0
    for i in range(1, len(xs)):
        auc += (ys[i - 1] + ys[i]) * (xs[i] - xs[i - 1]) / 2.0
    auc_norm = auc / 180.0
    worst = min(ys) if ys else 0.0
    return xs, ys, float(auc_norm), float(worst)


# ------------------------------ family summary (baseline) ------------------------------

def _pick_col(cols_lower: Dict[str, str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None

def load_family_baseline(path: Path, *, print_mapping: bool = False) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Load baseline metrics per (arch, act) from family summary CSV.

    Recognized column names (case-insensitive):
      - arch: arch
      - act : act
      - or a generic "model/name/label/config" with tokens inside.

    Metrics (many aliases supported):
      avg    : avg, mean, avg_acc, accuracy
      auc    : auc_theta, auctheta, auc, auc_theta_norm, auc_norm, "auc_θ"
      worst  : worst, acc_worst, worst_case, min, worst_acc
    """
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    arch_col = _pick_col(cols_lower, ["arch"])
    act_col  = _pick_col(cols_lower, ["act"])
    model_col = _pick_col(cols_lower, ["model", "name", "config", "label"])

    avg_col   = _pick_col(cols_lower, ["avg", "mean", "avg_acc", "accuracy"])
    auc_col   = _pick_col(cols_lower, ["auc_theta", "auctheta", "auc", "auc_theta_norm", "auc_norm", "auc_θ"])
    worst_col = _pick_col(cols_lower, ["worst", "acc_worst", "worst_case", "min", "worst_acc"])

    baseline: Dict[Tuple[str, str], Dict[str, float]] = {}
    keys_seen: list[Tuple[str, str]] = []

    for _, row in df.iterrows():
        if arch_col and act_col:
            arch = _norm_arch(str(row[arch_col]))
            act  = _norm_act(str(row[act_col]))
        elif model_col:
            m = str(row[model_col])
            arch = _norm_arch(m)
            act  = _norm_act(m)
        else:
            continue

        entry: Dict[str, float] = {}
        if avg_col and pd.notna(row.get(avg_col)):
            entry["avg"] = float(row[avg_col])
        if auc_col and pd.notna(row.get(auc_col)):
            entry["auc"] = float(row[auc_col])
        if worst_col and pd.notna(row.get(worst_col)):
            entry["worst"] = float(row[worst_col])

        key = (arch, act)
        if entry:
            baseline[key] = entry
            keys_seen.append(key)

    if print_mapping:
        uniq = sorted(set(keys_seen))
        if uniq:
            print("Baseline (arch, act) keys found:", ", ".join([f"{a}/{b}" for a, b in uniq]))
        else:
            print("Baseline mapping is empty – check columns in family summary.")

    return baseline


# ------------------------------ main builder ------------------------------

def build_table_for_dataset(dataset: str,
                            logs_dir: Path,
                            out_dir: Path,
                            family_summary: Optional[Path] = None,
                            theta_step: int = 15,
                            print_mapping: bool = False) -> Tuple[Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"optuna_delta_{dataset.lower()}_micro.csv"

    files = [p for p in logs_dir.glob("*.log") if p.is_file() and matches_dataset(p.name, dataset)]
    if not files:
        header = ["run","arch","act","avg","AUC_theta","worst","base_avg","base_auc","base_worst","Δavg","ΔAUC","Δworst"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
        print(f"WARNING: No logs for dataset={dataset}. Saved empty: {out_csv}")
        return out_csv, 0

    baseline = {}
    if family_summary is not None and Path(family_summary).exists():
        baseline = load_family_baseline(Path(family_summary), print_mapping=print_mapping)

    rows: List[List[object]] = []
    unmatched: list[str] = []

    for f in sorted(files):
        train_label, test_map = parse_optuna_test_results(f)
        if not test_map or not train_label:
            continue

        avg = float(np.mean(list(test_map.values()))) if test_map else 0.0
        _, _, auc_theta, worst = acc_delta_curve_and_auc(train_label, test_map, theta_step=theta_step)

        arch = detect_arch(train_label) or detect_arch(f.stem) or "unknown"
        act  = detect_act(train_label)  or detect_act(f.stem)  or "unknown"

        base_avg = base_auc = base_worst = None
        d_avg = d_auc = d_worst = None

        if baseline:
            key = (_norm_arch(arch), _norm_act(act))
            b = baseline.get(key)
            if b:
                base_avg = float(b.get("avg")) if "avg" in b else None
                base_auc = float(b.get("auc")) if "auc" in b else None
                base_worst = float(b.get("worst")) if "worst" in b else None
            else:
                unmatched.append(f"{f.name} -> key {key} not in baseline")

            if base_avg is not None: d_avg = avg - base_avg
            if base_auc is not None: d_auc = auc_theta - base_auc
            if base_worst is not None: d_worst = worst - base_worst

        rows.append([
            f.name, arch, act,
            round(avg, 12), round(auc_theta, 12), round(worst, 12),
            (None if base_avg is None else round(base_avg, 12)),
            (None if base_auc is None else round(base_auc, 12)),
            (None if base_worst is None else round(base_worst, 12)),
            (None if d_avg is None else round(d_avg, 12)),
            (None if d_auc is None else round(d_auc, 12)),
            (None if d_worst is None else round(d_worst, 12)),
        ])

    if print_mapping and unmatched:
        print("Unmatched Optuna runs (no baseline key):")
        for u in unmatched:
            print("  -", u)

    header = ["run","arch","act","avg","AUC_theta","worst","base_avg","base_auc","base_worst","Δavg","ΔAUC","Δworst"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

    print(f"Saved: {out_csv}")
    print(f"Pairs: {len(rows)} (dataset={dataset}, metric=micro)")
    return out_csv, len(rows)


# ------------------------------ CLI ------------------------------

def main():
    p = argparse.ArgumentParser(description="Build Optuna vs baseline delta table for a dataset.")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. LEGO, MNIST, GTSRB, GTSRB_RGB")
    p.add_argument("--optuna-logs", required=True, help="Directory with Optuna .log files")
    p.add_argument("--out", default="results/fig", help="Output directory (default: results/fig)")
    p.add_argument("--family-summary", default=None,
                   help="Optional path to results/exports_family/<DATASET>/micro/family_summary_<DATASET>_micro.csv")
    p.add_argument("--theta-step", type=int, default=15, help="Bin size (deg) for Δθ curve/AUC (default: 15)")
    p.add_argument("--print-mapping", action="store_true", help="Print baseline keys and unmatched runs")
    args = p.parse_args()

    logs_dir = Path(args.optuna_logs)
    out_dir = Path(args.out)

    build_table_for_dataset(
        dataset=args.dataset,
        logs_dir=logs_dir,
        out_dir=out_dir,
        family_summary=Path(args.family_summary) if args.family_summary else None,
        theta_step=args.theta_step,
        print_mapping=args.print_mapping
    )

if __name__ == "__main__":
    main()
