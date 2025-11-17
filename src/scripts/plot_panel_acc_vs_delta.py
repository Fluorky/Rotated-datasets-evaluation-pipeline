import os
import re
import sys
import csv
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ------------- CSV loaders (wide or long) -----------------

ANGLE_COL_RE = re.compile(r'^[dD](\d+)$')  # d0, d15, ..., d180

def load_curve(path):
    """
    Returns (x_deg, y_acc, label_guess).
    Supports:
      - WIDE: columns like d0,d15,...,d180 (single-row, or takes first row)
      - LONG: columns: delta/delta_deg + acc
    """
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    if not rows:
        raise ValueError(f"No rows in {path}")

    # Try WIDE header: d0,d15,...,d180
    angle_cols = []
    for h in rdr.fieldnames:
        m = ANGLE_COL_RE.match(h.strip()) if h else None
        if m:
            angle_cols.append((int(m.group(1)), h))
    if angle_cols:
        angle_cols.sort(key=lambda t: t[0])
        # take first row
        row0 = rows[0]
        x = [deg for deg, _ in angle_cols]
        y = []
        for _, col in angle_cols:
            try:
                y.append(float(row0[col]))
            except:
                y.append(np.nan)
        # forward fill NaNs
        y = _ffill_then_fill0(y)
        return (np.array(x, dtype=float), np.array(y, dtype=float), _guess_label_from_path(path))

    # Try LONG: delta / delta_deg + acc
    # Find column names
    delta_col = None
    for cand in ["delta","delta_deg","angle","theta"]:
        if cand in rows[0]:
            delta_col = cand; break
    acc_col = None
    for cand in ["acc","accuracy","acc_avg"]:
        if cand in rows[0]:
            acc_col = cand; break
    if not delta_col or not acc_col:
        # last resort: attempt to parse any numeric pairs
        # but better fail explicitly for clarity
        raise ValueError(f"Unsupported CSV structure in {path}")

    x, y = [], []
    for r in rows:
        try:
            xd = float(r[delta_col])
            ya = float(r[acc_col])
            x.append(xd); y.append(ya)
        except:
            continue
    # Sort by x and forward-fill if needed
    idx = np.argsort(x)
    x = np.array(x, dtype=float)[idx]
    y = np.array(y, dtype=float)[idx]
    y = _ffill_then_fill0(y)
    return (x, y, _guess_label_from_path(path))

def _ffill_then_fill0(arr):
    out = []
    last = None
    for v in arr:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(last)
        else:
            out.append(v); last = v
    # fill initial None with first known or 0.0
    first_known = next((v for v in out if v is not None), 0.0)
    out = [first_known if v is None else v for v in out]
    return np.array(out, dtype=float)

def _guess_label_from_path(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    # e.g., family_acc_vs_delta_CyResNet56-linear -> CyResNet56-linear
    m = re.search(r"family_acc_vs_delta_(.+)$", name)
    return m.group(1) if m else name

# ------------- plotting -----------------

def plot_panel(pairs, out_png):
    """
    pairs: list of dicts:
      [
        {"title": "MNIST", "base_csv": "...VGG19-linear.csv", "cy_csv": "...CyVGG19-linear.csv"},
        {"title": "GTSRB", ...},
        {"title": "GTSRB_RGB", ...},
        {"title": "LEGO", ...},
      ]
    """
    if len(pairs) != 4:
        raise ValueError("Provide exactly 4 pairs (for 2×2 panel).")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, cfg in zip(axes, pairs):
        xb, yb, lb = load_curve(cfg["base_csv"])
        xc, yc, lc = load_curve(cfg["cy_csv"])

        # align on common x-grid if needed (prefer union, then interpolate)
        xgrid = sorted(set(list(xb) + list(xc)))
        yb_i = np.interp(xgrid, xb, yb)
        yc_i = np.interp(xgrid, xc, yc)

        ax.plot(xgrid, yb_i, label=lb, linewidth=2)
        ax.plot(xgrid, yc_i, label=lc, linewidth=2)
        ax.set_title(cfg["title"])
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[2].set_xlabel("Δθ [deg]")
    axes[3].set_xlabel("Δθ [deg]")
    axes[0].set_ylabel("Acc(Δθ)")
    axes[2].set_ylabel("Acc(Δθ)")

    # single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0,0.05,1,1])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=220)
    print("Saved:", out_png)

# ------------- CLI ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Make 2×2 panel of Acc(Δθ) for four datasets.")
    ap.add_argument("--mnist", nargs=2, metavar=("BASE_CSV","CY_CSV"))
    ap.add_argument("--gtsrb", nargs=2, metavar=("BASE_CSV","CY_CSV"))
    ap.add_argument("--gtsrb_rgb", nargs=2, metavar=("BASE_CSV","CY_CSV"))
    ap.add_argument("--lego", nargs=2, metavar=("BASE_CSV","CY_CSV"))
    ap.add_argument("--out", required=True, help="Output PNG, e.g. results/fig/acc_vs_delta_panel_2x2.png")
    return ap.parse_args()

def main():
    args = parse_args()
    pairs = []
    if args.mnist:
        pairs.append({"title":"MNIST","base_csv":args.mnist[0],"cy_csv":args.mnist[1]})
    if args.gtsrb:
        pairs.append({"title":"GTSRB","base_csv":args.gtsrb[0],"cy_csv":args.gtsrb[1]})
    if args.gtsrb_rgb:
        pairs.append({"title":"GTSRB_RGB","base_csv":args.gtsrb_rgb[0],"cy_csv":args.gtsrb_rgb[1]})
    if args.lego:
        pairs.append({"title":"LEGO","base_csv":args.lego[0],"cy_csv":args.lego[1]})
    # ensure order MNIST, GTSRB, GTSRB_RGB, LEGO
    title_order = ["MNIST","GTSRB","GTSRB_RGB","LEGO"]
    pairs.sort(key=lambda d: title_order.index(d["title"]))
    plot_panel(pairs, args.out)

if __name__ == "__main__":
    main()
