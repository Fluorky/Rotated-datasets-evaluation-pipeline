#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Acc(Δθ) — single plot with 8 curves for a given dataset.

Change according to supervisor’s suggestion:
Instead of a 2×2 panel, draw everything on **one** plot,
because there are only 8 lines (4 model families × 2 variants: linear / log).

Expected “wide” CSV files (from family_aggregator):
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

# columns d0, d15, ..., d180
ANGLE_COL_RE = re.compile(r"^[dD](\d+)$")

# ---------------------------------------------------------------------------

def _ffill_then_fill0(arr: List[Optional[float]]) -> np.ndarray:
    """Simple missing value handling: forward-fill, then fill remaining gaps with the first known value or 0."""
    out: List[Optional[float]] = []
    last: Optional[float] = None
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
    Loads a “wide” CSV with columns d0, d15, ... and returns (angles, accuracy).
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        if not rows:
            raise ValueError(f"No data in {csv_path}")
        # find angle columns
        angle_cols: List[Tuple[int, str]] = []
        for h in rdr.fieldnames or []:
            m = ANGLE_COL_RE.match(h.strip()) if h else None
            if m:
                angle_cols.append((int(m.group(1)), h))
        if not angle_cols:
            raise ValueError(f"File does not look like dXX format: {csv_path}")
        angle_cols.sort(key=lambda t: t[0])
        row0 = rows[0]
        x = [deg for deg, _ in angle_cols]
        y: List[Optional[float]] = []
        for _, col in angle_cols:
            try:
                y.append(float(row0[col]))
            except Exception:
                y.append(None)
        y_arr = _ffill_then_fill0(y)
        return np.array(x, dtype=float), y_arr

def find_family_csv(curves_dir: str, family: str, act: str) -> Optional[str]:
    """
    Search for a file for a given family and variant (linear/log) with name tolerance.
    """
    candidates = [
        f"family_acc_vs_delta_{family}-{act}.csv",
        f"family_acc_vs_delta_{family}_{act}.csv",
        f"family_acc_vs_delta_{family}-{act}polar.csv",
        f"family_acc_vs_delta_{family}_{act}polar.csv",
    ]
    for name in candidates:
        p = os.path.join(curves_dir, name)
        if os.path.isfile(p):
            return p
    # fallback: glob pattern search
    pat = os.path.join(curves_dir, f"family_acc_vs_delta_{family}*{act}*.csv")
    hits = sorted(glob(pat))
    return hits[0] if hits else None

# ---------------------------------------------------------------------------

FAMILIES = ["VGG19", "CyVGG19", "ResNet56", "CyResNet56"]

# color by family, line style (solid/dashed) by linear/log
FAMILY_COLOR = {
    "VGG19": "#1f77b4",
    "CyVGG19": "#2ca02c",
    "ResNet56": "#ff7f0e",
    "CyResNet56": "#d62728",
}
ACT_STYLE = {
    "linear": "-",
    "log": "--",
}

def plot_all_on_one(
    dataset: str,
    curves_dir: str,
    out_png: str,
    *,
    dpi: int = 250,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    """
    Single plot, 8 lines: 4 model families × (linear, log).
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    plotted_any = False
    legend_items = []

    for family in FAMILIES:
        for act in ("linear", "log"):
            csv_path = find_family_csv(curves_dir, family, act)
            label = f"{family} – {act}"
            if csv_path and os.path.isfile(csv_path):
                x, y = load_curve_wide(csv_path)
                line = ax.plot(
                    x,
                    y,
                    ACT_STYLE[act],
                    color=FAMILY_COLOR[family],
                    linewidth=2.0,
                    label=label,
                )[0]
                legend_items.append(line)
                plotted_any = True
            else:
                # file not found – skip, but don’t break the plot
                continue

    if not plotted_any:
        raise RuntimeError(f"No curves found in {curves_dir}")

    ax.set_title(f"Acc(Δθ) — {dataset}", fontsize=13)
    ax.set_xlabel("Δθ [deg]", fontsize=11)
    ax.set_ylabel("Acc(Δθ)", fontsize=11)
    ax.set_xlim(0, 180)
    if ymin is not None or ymax is not None:
        ax.set_ylim(0.0 if ymin is None else ymin, 1.0 if ymax is None else ymax)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)

    # # legend with 8 items in 2 columns to save space
    # ax.legend(
    #     handles=legend_items,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.25),
    #     ncol=2,
    #     frameon=False,
    #     fontsize=9,
    # )
    #
    # plt.tight_layout()
    # os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    # fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    # plt.close(fig)
    # print(f"Saved: {out_png}")
    # legend with 8 items in 2 columns, placed slightly lower with more spacing
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),   # was -0.25
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # adds bottom margin
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Draws a single Acc(Δθ) plot for one dataset (4 families × linear/log)."
    )
    ap.add_argument("--dataset", required=True, help="MNIST | GTSRB | GTSRB_RGB | LEGO")
    ap.add_argument(
        "--curves-dir",
        default=None,
        help="Directory containing family_acc_vs_delta_*.csv "
             "(default: results/exports_family/<DATASET>/micro/family_curves)",
    )
    ap.add_argument("--out", required=True, help="Output PNG file path")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    curves_dir = (
        args.curves_dir
        or os.path.join("results", "exports_family", args.dataset, "micro", "family_curves")
    )
    plot_all_on_one(
        dataset=args.dataset,
        curves_dir=curves_dir,
        out_png=args.out,
        dpi=args.dpi,
        ymin=args.ymin,
        ymax=args.ymax,
    )

if __name__ == "__main__":
    main()
