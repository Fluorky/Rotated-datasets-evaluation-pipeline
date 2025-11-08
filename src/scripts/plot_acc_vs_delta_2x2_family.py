#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Acc(Δθ) — 2×2 panel per dataset.
Each panel = one family (VGG19, CyVGG19, ResNet56, CyResNet56),
and shows two lines: linear vs log (=> 8 lines total per figure).

Expected CSVs (wide format) from family_aggregator:
  results/exports_family/<DATASET>/micro/family_curves/
    family_acc_vs_delta_VGG19-linear.csv
    family_acc_vs_delta_VGG19-log.csv
    family_acc_vs_delta_CyVGG19-linear.csv
    family_acc_vs_delta_CyVGG19-log.csv
    family_acc_vs_delta_ResNet56-linear.csv
    family_acc_vs_delta_ResNet56-log.csv
    family_acc_vs_delta_CyResNet56-linear.csv
    family_acc_vs_delta_CyResNet56-log.csv
"""

from __future__ import annotations
import argparse, csv, os, re, math
from glob import glob
from typing import Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt

ANGLE_COL_RE = re.compile(r'^[dD](\d+)$')  # d0, d15, ..., d180

# ---------- IO helpers -------------------------------------------------------

def _ffill_then_fill0(arr: List[Optional[float]]) -> np.ndarray:
    out = []
    last = None
    for v in arr:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(last)
        else:
            out.append(v)
            last = v
    first_known = next((v for v in out if v is not None), 0.0)
    out = [first_known if v is None else v for v in out]
    return np.array(out, dtype=float)

def load_curve_wide(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a single-row 'wide' CSV with columns d0,d15,...,d180 and return (x_deg, y_acc).
    If multiple rows are present, the first row is used.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        if not rows:
            raise ValueError(f"No rows in {csv_path}")
        angle_cols = []
        for h in rdr.fieldnames or []:
            m = ANGLE_COL_RE.match(h.strip()) if h else None
            if m:
                angle_cols.append((int(m.group(1)), h))
        if not angle_cols:
            raise ValueError(f"CSV doesn't look 'wide' (no dXX columns): {csv_path}")
        angle_cols.sort(key=lambda t: t[0])
        row0 = rows[0]
        x = [deg for deg, _ in angle_cols]
        y = []
        for _, col in angle_cols:
            try:
                y.append(float(row0[col]))
            except Exception:
                y.append(np.nan)
        y = _ffill_then_fill0(y)
        return np.array(x, dtype=float), np.array(y, dtype=float)

def find_family_csv(curves_dir: str, family: str, act: str) -> Optional[str]:
    """
    family in {VGG19, CyVGG19, ResNet56, CyResNet56}
    act    in {linear, log}   (tolerates - or _; also *polar suffixes)
    """
    names = [
        f"family_acc_vs_delta_{family}-{act}.csv",
        f"family_acc_vs_delta_{family}_{act}.csv",
        f"family_acc_vs_delta_{family}-{act}polar.csv",
        f"family_acc_vs_delta_{family}_{act}polar.csv",
    ]
    for n in names:
        p = os.path.join(curves_dir, n)
        if os.path.isfile(p):
            return p
    pat = os.path.join(curves_dir, f"family_acc_vs_delta_{family}*{act}*.csv")
    hits = sorted(glob(pat))
    return hits[0] if hits else None

# ---------- Plotting ---------------------------------------------------------

FAMILIES = ["VGG19", "CyVGG19", "ResNet56", "CyResNet56"]
PANEL_ORDER = [["VGG19", "CyVGG19"], ["ResNet56", "CyResNet56"]]

LINE_STYLE = {
    "linear": {"color": "#1f77b4", "lw": 2.2, "label": "linear"},
    "log":    {"color": "#ff7f0e", "lw": 2.2, "label": "log"},
}
MISSING_ALPHA = 0.25

def plot_dataset_panel(
    dataset: str,
    curves_dir: str,
    out_png: str,
    *,
    title_font: float = 16.0,
    label_font: float = 12.0,
    tick_font: float = 11.0,
    dpi: int = 250,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    """
    Build a 2×2 figure:
        TL:  VGG19       — {linear, log}
        TR:  CyVGG19     — {linear, log}
        BL:  ResNet56    — {linear, log}
        BR:  CyResNet56  — {linear, log}
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    axes = axes.ravel()

    # collect paths
    paths: Dict[Tuple[str, str], Optional[str]] = {}
    for fam in FAMILIES:
        for act in ("linear", "log"):
            paths[(fam, act)] = find_family_csv(curves_dir, fam, act)

    # plot
    for ax, fam in zip(axes, sum(PANEL_ORDER, [])):  # flatten 2x2 -> [VGG, CyVGG, Res, CyRes]
        ax.set_title(f"{fam} — linear vs log", fontsize=13)
        for act in ("linear", "log"):
            p = paths[(fam, act)]
            style = LINE_STYLE[act]
            if p and os.path.isfile(p):
                x, y = load_curve_wide(p)
                ax.plot(x, y, **style)
            else:
                # faint "missing" baseline to keep legend consistent
                ax.plot([0, 180], [0, 0], color=style["color"], lw=2.2, alpha=MISSING_ALPHA)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(0, 180)
        # Y limits (optionally zoom to emphasize differences)
        if ymin is not None or ymax is not None:
            y0 = 0.0 if ymin is None else ymin
            y1 = 1.0 if ymax is None else ymax
            ax.set_ylim(y0, y1)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="both", labelsize=tick_font)

    # axis labels on lower row / left column
    axes[2].set_xlabel("Δθ [deg]", fontsize=label_font)
    axes[3].set_xlabel("Δθ [deg]", fontsize=label_font)
    axes[0].set_ylabel("Acc(Δθ)", fontsize=label_font)
    axes[2].set_ylabel("Acc(Δθ)", fontsize=label_font)

    # global title & legend
    fig.suptitle(f"Acc(Δθ) curves — {dataset}", fontsize=title_font)
    leg_lines = [
        plt.Line2D([0], [0], **LINE_STYLE["linear"]),
        plt.Line2D([0], [0], **LINE_STYLE["log"]),
    ]
    fig.legend(leg_lines, ["linear", "log"], loc="lower center", ncol=2, frameon=False, fontsize=11)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_png}")

# ---------- CLI --------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="2×2 Acc(Δθ) panel per dataset (linear vs log in each family panel)."
    )
    ap.add_argument("--dataset", required=True, help="MNIST | GTSRB | GTSRB_RGB | LEGO")
    ap.add_argument("--curves-dir", default=None,
                    help="Folder with family_acc_vs_delta_*.csv. "
                         "Default: results/exports_family/<DATASET>/micro/family_curves")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--ymin", type=float, default=None, help="Lower y-limit for Acc(Δθ)")
    ap.add_argument("--ymax", type=float, default=None, help="Upper y-limit for Acc(Δθ)")
    return ap.parse_args()

def main():
    args = parse_args()
    ds = args.dataset
    curves_dir = args.curves_dir or os.path.join("results", "exports_family", ds, "micro", "family_curves")
    plot_dataset_panel(ds, curves_dir, args.out, dpi=args.dpi, ymin=args.ymin, ymax=args.ymax)

if __name__ == "__main__":
    main()
