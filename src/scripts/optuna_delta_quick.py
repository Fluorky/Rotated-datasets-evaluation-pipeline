# scripts/optuna_delta_quick.py
# -*- coding: utf-8 -*-
"""
Build Optuna-vs-Baseline delta tables for one or multiple datasets.

For each dataset D, the script:
  1) Parses Optuna logs: [TEST] <train_label>_test_on_<test_case> loss=... acc=X%
  2) Aggregates accuracy per Δθ-bin to compute Acc(Δθ), AUCθ (normalized), and worst-case
  3) Looks up a baseline in family summary (preferred) and/or in family Acc(Δθ) CSVs
  4) Writes CSV: <out>/optuna_delta_<dataset_lower>_micro.csv

Inputs you likely have:
  --optuna-logs  E:\MasterThesisLogsAll\optuna_checked\logs
  --family-dir   results\exports_family\<DATASET>\micro
  --family-summary results\exports_family\<DATASET>\micro\family_summary_<DATASET>_micro.csv

Usage examples:
  Single dataset:
    python scripts/optuna_delta_quick.py \
      --dataset LEGO \
      --optuna-logs "E:\\MasterThesisLogsAll\\optuna_checked\\logs" \
      --family-dir "results\\exports_family\\LEGO\\micro" \
      --family-summary "results\\exports_family\\LEGO\\micro\\family_summary_LEGO_micro.csv" \
      --out results\\fig

  Multiple datasets (space-separated):
    python scripts/optuna_delta_quick.py \
      --dataset GTSRB GTSRB_RGB LEGO MNIST \
      --optuna-logs "E:\\MasterThesisLogsAll\\optuna_checked\\logs" \
      --family-root "results\\exports_family" \
      --out results\\fig \
      --print-mapping

Notes:
- If --family-summary is NOT given, the script tries to read family curves
  (family_acc_vs_delta_*.csv) from --family-dir (or from --family-root/<DATASET>/micro).
- AUCθ is trapezoidal over 0..180 and normalized by 180°.
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


# --------------------------- dataset-safe filename matching ---------------------------

def _ds_norm(s: str) -> str:
    return s.strip().lower().replace("-", "_")

def matches_dataset(filename: str, dataset: str) -> bool:
    """
    Strict-enough matcher to avoid mixing GTSRB and GTSRB_RGB.
    Accepts common variants like 'gtsrb-rgb', 'custom_dataset_mnist', etc.
    """
    name = filename.lower()
    ds = _ds_norm(dataset)

    # Disambiguate GTSRB vs GTSRB_RGB
    if ds == "gtsrb":
        if "gtsrb_rgb" in name or "gtsrb-rgb" in name:
            return False
        return ("gtsrb" in name)
    if ds == "gtsrb_rgb":
        return ("gtsrb_rgb" in name) or ("gtsrb-rgb" in name)

    # LEGO / MNIST (looser, but safe)
    if ds in ("lego", "mnist"):
        return (ds in name) or (f"dataset_{ds}" in name) or (f"custom_dataset_{ds}" in name)

    # Generic fallback
    return ds in name


# --------------------------- Δθ helpers ---------------------------

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
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) else float(m.group(1))
        return (a, b)
    if m.group(3):  # range_a_b
        a = float(m.group(3)); b = float(m.group(4))
        return (a, b)
    return (0.0, 360.0)  # full_0_360

def _center_deg(iv: Tuple[float, float]) -> float:
    a, b = iv
    if b >= a:
        return (a + b) / 2.0
    # wrap (e.g., 330..30)
    return (a + ((b + 360.0 - a) / 2.0)) % 360.0

def _delta_deg(train_name: str, test_name: str) -> Optional[float]:
    it = _interval_from_token(train_name)
    ie = _interval_from_token(test_name)
    if not it or not ie:
        return None
    ct = _center_deg(it); ce = _center_deg(ie)
    d = abs(ct - ce) % 360.0
    return 360.0 - d if d > 180.0 else d

def _bin_delta(d: Optional[float], step: int = 15) -> Optional[int]:
    if d is None:
        return None
    b = int(round(d / step) * step)
    return min(b, 180)


# --------------------------- Optuna [TEST] parsing ---------------------------

TEST_LINE_RE = re.compile(
    r"""(?mix)
    ^\s*\[TEST\]\s+
    (\S+)\s+                 # <train_label>_test_on_<test_case>
    loss\s*=\s*([0-9.]+)\s+
    acc\s*=\s*([0-9.]+)\s*%?\s*$   # accuracy (maybe in %)
    """
)

def parse_optuna_test_results(path: Path) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Return (train_label, { test_case -> accuracy_fraction(0..1) }).
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
        acc = acc / 100.0 if acc > 1.5 else acc

        if "_test_on_" in token:
            tr, _, te = token.partition("_test_on_")
        else:
            tr, te = token, token

        if train_label is None:
            train_label = tr
        results[te] = acc

    return train_label, results


# --------------------------- model tokens ---------------------------

ARCH_TOK = ["cyresnet56", "cyvgg19", "resnet56", "vgg19"]
ACT_TOK  = ["linearpolar", "logpolar"]

def _detect_arch_act(label: str) -> Tuple[Optional[str], Optional[str]]:
    L = label.lower()
    arch = next((t for t in ARCH_TOK if t in L), None)
    act  = next((t for t in ACT_TOK if t in L), None)
    # also allow "log" / "linear" short tokens
    if act is None:
        if re.search(r'(?i)\blog(?:polar)?\b', L):
            act = "logpolar"
        elif re.search(r'(?i)\blinear(?:polar)?\b', L):
            act = "linearpolar"
    return arch, act


# --------------------------- baseline: family summary (preferred) ---------------------------

def _pick_col(cols_lower: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None

def _parse_family_label(family: str) -> Tuple[Optional[str], Optional[str]]:
    """
    'CyResNet56-linear' -> ('cyresnet56', 'linearpolar')
    'CyVGG19-log'       -> ('cyvgg19', 'logpolar')
    """
    t = family.strip().lower()
    arch = next((tok for tok in ARCH_TOK if tok in t), None)
    act = "logpolar" if "log" in t else ("linearpolar" if "linear" in t else None)
    return arch, act

def load_family_summary(path: Path, print_mapping: bool = False) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Load baseline metrics per (arch, act) from family summary CSV.

    Expected (case-insensitive) columns:
      family  (like 'CyResNet56-linear')  OR  arch/act  OR  model/name/config
    Metrics:
      avg:   avg_mean, avg, mean, accuracy
      auc:   auc_theta_mean, auc_theta, auc_norm, auctheta, 'auc_θ'
      worst: acc_worst_mean, worst, worst_case, min, worst_acc
    """
    if not path or not path.exists():
        return {}

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    family_col = _pick_col(cols_lower, ["family"])
    arch_col   = _pick_col(cols_lower, ["arch"])
    act_col    = _pick_col(cols_lower, ["act"])
    model_col  = _pick_col(cols_lower, ["model", "name", "config", "label"])

    avg_col   = _pick_col(cols_lower, ["avg_mean", "avg", "mean", "accuracy"])
    auc_col   = _pick_col(cols_lower, ["auc_theta_mean", "auc_theta", "auc_norm", "auctheta", "auc_θ"])
    worst_col = _pick_col(cols_lower, ["acc_worst_mean", "worst", "worst_case", "min", "worst_acc"])

    mapping: Dict[Tuple[str, str], Dict[str, float]] = {}
    keys_seen: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        arch = act = None
        if family_col:
            arch, act = _parse_family_label(str(row[family_col]))
        if (arch is None or act is None) and arch_col and act_col:
            arch = str(row[arch_col]).lower()
            act  = str(row[act_col]).lower()
            if "log" in act and "polar" not in act: act = "logpolar"
            if "linear" in act and "polar" not in act: act = "linearpolar"
        if (arch is None or act is None) and model_col:
            m = str(row[model_col]).lower()
            arch, act = _detect_arch_act(m)

        if arch is None or act is None:
            continue

        entry: Dict[str, float] = {}
        if avg_col   and pd.notna(row.get(avg_col)):   entry["avg"]   = float(row[avg_col])
        if auc_col   and pd.notna(row.get(auc_col)):   entry["auc"]   = float(row[auc_col])
        if worst_col and pd.notna(row.get(worst_col)): entry["worst"] = float(row[worst_col])

        key = (arch, act)
        if entry:
            mapping[key] = entry
            keys_seen.append(key)

    if print_mapping:
        if keys_seen:
            distinct = sorted(set(keys_seen))
            print("Baseline (family summary) keys:", ", ".join([f"{a}/{b}" for a, b in distinct]))
        else:
            print("Baseline mapping from family summary is empty.")

    return mapping


# --------------------------- baseline: family Acc(Δθ) CSVs ---------------------------

def load_family_curve_baseline(family_dir: Path, arch: str, act: str) -> Optional[Tuple[float, float, float]]:
    """
    Read family_acc_vs_delta_<Arch>-<linear|log>.csv, average rows -> (avg, auc, worst).
    Returns None if files are not present.
    """
    if not family_dir or not family_dir.exists():
        return None

    # build candidate filenames
    arch_name = {
        "cyresnet56": "CyResNet56",
        "cyvgg19": "CyVGG19",
        "resnet56": "ResNet56",
        "vgg19": "VGG19",
    }.get(arch.lower(), arch)

    act_tag = "log" if "log" in act.lower() else "linear"

    candidates = [
        family_dir / f"family_acc_vs_delta_{arch_name}-{act_tag}.csv",
        family_dir / f"family_acc_vs_delta_{arch_name.lower()}-{act_tag}.csv",
    ]
    fpath = next((p for p in candidates if p.exists()), None)
    if not fpath:
        return None

    df = pd.read_csv(fpath)
    dcols = [c for c in df.columns if c.lower().startswith("d")]
    if not dcols:
        return None

    vals = df[dcols].mean(axis=0).values.astype(float)  # mean over rows
    if vals.max() > 1.5:
        vals = vals / 100.0

    # AUC (normalized)
    xs = np.arange(0, 181, 15)
    # ensure we have exactly len(xs) points; if not, try to align by column names
    if len(vals) != len(xs):
        # try reindex by exact d0..d180 columns
        try:
            ordered = [f"d{x}" for x in xs]
            vals = df[ordered].mean(axis=0).values.astype(float)
            if vals.max() > 1.5:
                vals = vals / 100.0
        except Exception:
            pass

    auc = 0.0
    for i in range(1, len(xs)):
        auc += (vals[i - 1] + vals[i]) * (xs[i] - xs[i - 1]) / 2.0
    auc_norm = auc / 180.0
    avg = float(np.mean(vals))
    worst = float(np.min(vals))
    return avg, float(auc_norm), worst


# --------------------------- core: compute deltas for one dataset ---------------------------

def compute_for_dataset(
    dataset: str,
    logs_dir: Path,
    out_dir: Path,
    family_dir: Optional[Path] = None,
    family_summary: Optional[Path] = None,
    theta_step: int = 15,
    print_mapping: bool = False,
) -> Tuple[Path, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"optuna_delta_{dataset.lower()}_micro.csv"

    # load baseline mappings (preferred: family summary)
    baseline_map = load_family_summary(family_summary, print_mapping=print_mapping) if (family_summary and family_summary.exists()) else {}

    rows: List[List[object]] = []

    # iterate logs
    logs = [p for p in logs_dir.glob("*.log") if p.is_file() and matches_dataset(p.name, dataset)]
    if not logs:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["run","arch","act","avg","AUC_theta","worst","base_avg","base_auc","base_worst","Δavg","ΔAUC","Δworst"])
        print(f"WARNING: No logs for dataset={dataset}. Saved empty: {out_csv}")
        return out_csv, 0

    for log in sorted(logs):
        train_label, test_map = parse_optuna_test_results(log)
        if not test_map or not train_label:
            continue

        # aggregate Acc(Δθ)
        by_bin: Dict[int, List[float]] = defaultdict(list)
        all_acc = []
        for test_name, acc in test_map.items():
            d = _bin_delta(_delta_deg(train_label, test_name), theta_step)
            if d is not None:
                by_bin[int(d)].append(acc)
            all_acc.append(acc)

        bins = list(range(0, 181, theta_step))
        ys: List[Optional[float]] = []
        last = None
        for b in bins:
            v = float(np.mean(by_bin[b])) if by_bin[b] else None
            if v is None:
                v = last
            ys.append(v)
            if v is not None:
                last = v
        first_known = next((v for v in ys if v is not None), 0.0)
        ys = [first_known if v is None else v for v in ys]  # type: ignore

        auc = 0.0
        for i in range(1, len(bins)):
            auc += (ys[i - 1] + ys[i]) * (bins[i] - bins[i - 1]) / 2.0
        auc /= 180.0

        avg   = float(np.mean(all_acc))
        worst = float(np.min(ys)) if ys else float(np.min(all_acc))

        arch, act = _detect_arch_act(log.name)
        arch = arch or "unknown"
        act  = act  or "unknown"

        # baseline lookup: prefer family_summary, fallback to family_acc_vs_delta
        base_avg = base_auc = base_worst = None
        d_avg = d_auc = d_worst = None

        key = (arch, act)
        if baseline_map.get(key):
            b = baseline_map[key]
            base_avg   = float(b.get("avg"))   if "avg"   in b else None
            base_auc   = float(b.get("auc"))   if "auc"   in b else None
            base_worst = float(b.get("worst")) if "worst" in b else None
        elif family_dir and family_dir.exists():
            got = load_family_curve_baseline(family_dir, arch, act)
            if got:
                base_avg, base_auc, base_worst = got

        if base_avg is not None:   d_avg   = avg   - base_avg
        if base_auc is not None:   d_auc   = auc   - base_auc
        if base_worst is not None: d_worst = worst - base_worst

        rows.append([
            log.name, arch, act,
            round(avg, 12), round(auc, 12), round(worst, 12),
            ("" if base_avg   is None else round(base_avg, 12)),
            ("" if base_auc   is None else round(base_auc, 12)),
            ("" if base_worst is None else round(base_worst, 12)),
            ("" if d_avg   is None else round(d_avg, 12)),
            ("" if d_auc   is None else round(d_auc, 12)),
            ("" if d_worst is None else round(d_worst, 12)),
        ])

    # write CSV
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run","arch","act","avg","AUC_theta","worst","base_avg","base_auc","base_worst","Δavg","ΔAUC","Δworst"])
        for r in rows:
            w.writerow(r)

    print(f"Saved: {out_csv}  (rows={len(rows)})")
    return out_csv, len(rows)


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Build Optuna delta tables for one or many datasets.")
    ap.add_argument("--dataset", nargs="+", required=True,
                    help="One or more dataset names, e.g. GTSRB GTSRB_RGB LEGO MNIST")
    ap.add_argument("--optuna-logs", required=True,
                    help="Directory with Optuna .log files")
    ap.add_argument("--out", default="results/fig",
                    help="Output directory for CSVs (default: results/fig)")

    # You can provide either per-dataset --family-dir/--family-summary,
    # or a --family-root to auto-resolve <root>/<DATASET>/micro
    ap.add_argument("--family-dir", default=None,
                    help="Path to family exports for ONE dataset (results/exports_family/<DATASET>/micro)")
    ap.add_argument("--family-summary", default=None,
                    help="Path to family_summary_<DATASET>_micro.csv for ONE dataset")
    ap.add_argument("--family-root", default=None,
                    help="If provided, use <root>/<DATASET>/micro for family-dir and family-summary")

    ap.add_argument("--theta-step", type=int, default=15,
                    help="Δθ bin size in degrees (default: 15)")
    ap.add_argument("--print-mapping", action="store_true",
                    help="Print baseline keys detected from family summary")
    args = ap.parse_args()

    logs_dir = Path(args.optuna_logs)
    out_dir  = Path(args.out)

    # process each dataset
    for ds in args.dataset:
        ds_upper = ds.strip().upper()
        ds_lower = ds.strip().lower()

        # Resolve family paths:
        fam_dir = Path(args.family_dir) if args.family_dir else None
        fam_sum = Path(args.family_summary) if args.family_summary else None

        if args.family_root:
            # Prefer per-dataset subdir under family_root
            root = Path(args.family_root)
            candidate_dir = root / ds_upper / "micro"
            candidate_sum = candidate_dir / f"family_summary_{ds_upper}_micro.csv"
            if candidate_dir.exists():
                fam_dir = candidate_dir
            if candidate_sum.exists():
                fam_sum = candidate_sum

        print(f"--- DATASET: {ds_upper} ---")
        if fam_dir:  print(f"family-dir: {fam_dir}")
        if fam_sum:  print(f"family-summary: {fam_sum}")

        compute_for_dataset(
            dataset=ds_upper,
            logs_dir=logs_dir,
            out_dir=out_dir,
            family_dir=fam_dir,
            family_summary=fam_sum,
            theta_step=args.theta_step,
            print_mapping=args.print_mapping
        )


if __name__ == "__main__":
    main()
