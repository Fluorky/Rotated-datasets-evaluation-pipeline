# -*- coding: utf-8 -*-
"""
Compare Optuna (non_rotated) vs Baseline (non_rotated) for the same arch/act.
Strict dataset matching so that GTSRB and GTSRB_RGB never mix.
"""

from __future__ import annotations
import re, csv, argparse, sqlite3
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

# ---------- strict dataset matching ----------
def matches_dataset_strict(text: str, dataset: str) -> bool:
    n = (text or "").lower()
    ds = (dataset or "").lower()
    if ds == "gtsrb":
        if re.search(r"gtsrb[_-]rgb", n): return False
        return re.search(r"(^|[^a-z0-9])gtsrb([^a-z0-9]|$)", n) is not None
    if ds == "gtsrb_rgb":
        return re.search(r"(^|[^a-z0-9])gtsrb[_-]rgb([^a-z0-9]|$)", n) is not None
    pat = rf"(^|[^a-z0-9]){re.escape(ds)}([^a-z0-9]|$)"
    return re.search(pat, n) is not None

# ---------- Δθ utils ----------
ANGLE_TOKEN = re.compile(r'(?i)(?:rotated-(\d+)(?:-(\d+))?)|(?:range[_-](\d+)[_-](\d+))|(?:full[_-]0[_-]360)')
def interval_from_token(name: str):
    s = (name or "").lower()
    m = ANGLE_TOKEN.search(s)
    if not m:
        return (0.0,0.0) if "non_rotated" in s else None
    if m.group(1):
        a = float(m.group(1)); b = float(m.group(2)) if m.group(2) else float(m.group(1))
        return (a,b)
    if m.group(3): return (float(m.group(3)), float(m.group(4)))
    return (0.0,360.0)
def center_deg(iv): a,b=iv; return (a+b)/2.0 if b>=a else (a+((b+360.0-a)/2.0))%360.0
def delta_deg(train_label, test_case):
    it = interval_from_token(train_label); ie = interval_from_token(test_case)
    if not it or not ie: return None
    d = abs(center_deg(it)-center_deg(ie))%360.0
    return 360.0-d if d>180.0 else d
def bin_delta(d, step=15):
    if d is None: return None
    b = int(round(d/step)*step); return min(b,180)

# ---------- arch/act detection ----------
ARCH_TOK = ["cyresnet56","cyvgg19","resnet56","vgg19"]
ACT_TOK  = ["linearpolar","logpolar"]
def detect_arch_act(label: str):
    L = (label or "").lower()
    arch = next((t for t in ARCH_TOK if t in L), None)
    act  = next((t for t in ACT_TOK  if t in L), None)
    return arch, act

# ---------- parse Optuna ----------
TEST_LINE = re.compile(r'(?mi)^\s*\[TEST\]\s+(\S+)\s+loss=\S+\s+acc=([0-9.]+)%\s*$')
def parse_optuna_log(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, []
    matches = TEST_LINE.findall(text)
    if not matches: return None, []
    train_label = None; pairs = []
    for token, acc_pct in matches:
        acc = float(acc_pct)/100.0
        if "_test_on_" in token:
            tr, _, te = token.partition("_test_on_")
            if train_label is None: train_label = tr
            pairs.append((te, acc))
        else:
            train_label = train_label or token
            pairs.append((token, acc))
    return train_label, pairs

def metrics_from_pairs(train_label: str, pairs: List[Tuple[str,float]], theta_step=15):
    if not pairs: return 0.0,0.0,0.0,0
    accs = [a for _,a in pairs]
    by_bin = defaultdict(list)
    for te,a in pairs:
        d = delta_deg(train_label, te); b = bin_delta(d, theta_step)
        if b is not None: by_bin[b].append(a)
    bins = list(range(0,181,theta_step))
    ys, last = [], None
    for b in bins:
        v = float(np.mean(by_bin[b])) if by_bin.get(b) else None
        if v is None: v = last
        ys.append(v);
        if v is not None: last = v
    first = next((v for v in ys if v is not None), 0.0)
    ys = [first if v is None else float(v) for v in ys]
    auc = 0.0
    for i in range(1,len(bins)):
        auc += (ys[i-1]+ys[i])*(bins[i]-bins[i-1])/2.0
    auc /= 180.0
    return float(np.mean(accs)), float(auc), float(np.min(ys) if ys else np.min(accs)), len(accs)

# ---------- baseline from SQLite (non_rotated, same arch/act) ----------
def fetch_baseline_nonrot_from_db(db_path: Path, dataset: str, arch: str, act: str, theta_step=15):
    conn = sqlite3.connect(str(db_path)); cur = conn.cursor()
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(evaluations)")]
        if "model" not in cols or "test_case" not in cols or "accuracy" not in cols:
            conn.close(); return None, None, None, 0

        cur.execute("SELECT model, test_case, accuracy FROM evaluations")
        rows = cur.fetchall()
    finally:
        conn.close()

    arch = (arch or "").lower(); act = (act or "").lower()
    def ok_model(m: str) -> bool:
        L = (m or "").lower()
        if "non_rotated" not in L: return False
        if arch not in L or act not in L: return False
        if not matches_dataset_strict(L, dataset): return False
        return True

    pairs_by_model = defaultdict(list); all_accs = []
    for m, t, a in rows:
        if not ok_model(m): continue
        try: acc = float(a)
        except: continue
        pairs_by_model[m].append((t, acc)); all_accs.append(acc)

    if not pairs_by_model: return None, None, None, 0

    bins = list(range(0,181,theta_step))
    curves = []
    for m, pts in pairs_by_model.items():
        by_bin = defaultdict(list)
        for te, a in pts:
            d = delta_deg(m, te); b = bin_delta(d, theta_step)
            if b is not None: by_bin[b].append(a)
        ys, last = [], None
        for b in bins:
            v = float(np.mean(by_bin[b])) if by_bin.get(b) else None
            if v is None: v = last
            ys.append(v);
            if v is not None: last = v
        first = next((v for v in ys if v is not None), 0.0)
        curves.append([first if v is None else float(v) for v in ys])

    curve_avg = np.mean(np.array(curves), axis=0)
    auc = 0.0
    for i in range(1,len(bins)):
        auc += (curve_avg[i-1]+curve_avg[i])*(bins[i]-bins[i-1])/2.0
    auc /= 180.0

    return float(np.mean(all_accs)), float(auc), float(np.min(curve_avg)), len(all_accs)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)     # GTSRB | GTSRB_RGB | LEGO | MNIST
    ap.add_argument("--optuna-logs", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--theta-step", type=int, default=15)
    ap.add_argument("--out", default="results\\fig")
    args = ap.parse_args()

    logs_dir = Path(args.optuna_logs)
    out_dir  = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for p in sorted(logs_dir.glob("*.log")):
        if not matches_dataset_strict(p.name, args.dataset):
            continue
        train_label, pairs = parse_optuna_log(p)
        if not pairs or not train_label:
            continue

        avg_o, auc_o, worst_o, n_opt = metrics_from_pairs(train_label, pairs, args.theta_step)
        arch, act = detect_arch_act(p.name)
        if not arch or not act:
            a2, t2 = detect_arch_act(train_label or "")
            arch = arch or a2; act = act or t2

        avg_b, auc_b, worst_b, n_base = fetch_baseline_nonrot_from_db(Path(args.db), args.dataset, arch or "", act or "", args.theta_step)

        d_avg = (avg_o - avg_b) if avg_b is not None else ""
        d_auc = (auc_o - auc_b) if auc_b is not None else ""
        d_wst = (worst_o - worst_b) if worst_b is not None else ""

        out_rows.append([
            p.name, arch or "", act or "",
            avg_o, auc_o, worst_o,
            "" if avg_b is None else avg_b,
            "" if auc_b is None else auc_b,
            "" if worst_b is None else worst_b,
            d_avg, d_auc, d_wst,
            n_opt, n_base
        ])

    out_csv = out_dir / f"optuna_vs_baseline_nonrot_{args.dataset}_micro.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run","arch","act",
            "avg_opt","AUC_opt","worst_opt",
            "avg_base","AUC_base","worst_base",
            "d_avg","d_AUC","d_worst",
            "n_opt_tests","n_base_pairs"
        ])
        for r in out_rows:
            w.writerow(r)

    print(f"Saved: {out_csv}  rows={len(out_rows)}")

if __name__ == "__main__":
    main()
