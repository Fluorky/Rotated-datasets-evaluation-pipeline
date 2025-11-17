# save as: scripts/plot_acc_vs_delta_vgg_vs_cyvgg_gtsrb_rgb.py
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
VGG_CSV = "results\\exports_family\\GTSRB_RGB\\micro\\family_curves\\family_acc_vs_delta_VGG19-log.csv"
CYVGG_CSV = "results\\exports_family\\GTSRB_RGB\\micro\\family_curves\\family_acc_vs_delta_CyVGG19-log.csv"
SEP = ","  # change to ";" if your CSVs use the European format
OUT_PNG = "results/fig/acc_vs_delta_GTSRB_RGB_VGG19_vs_CyVGG19_log.png"

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

def load_curve(path, sep=","):
    """Return (x_deg_sorted, acc_sorted) for a CSV in one of two formats:
       (A) wide: model,d0,d15,...,d180 (single row), or
       (B) long: delta|delta_deg, acc|accuracy (many rows).
    """
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    # (B) long format?
    lc = [c.lower() for c in df.columns]
    has_delta = any(c in ("delta","delta_deg") or "delta" in c for c in lc)
    has_acc   = any(c in ("acc","accuracy","acc_mean","mean_acc") or "acc" in c for c in lc)
    if has_delta and has_acc and len(df) > 1:
        # find the exact column names
        delta_col = next(c for c in df.columns if "delta" in c.lower())
        acc_col   = next(c for c in df.columns if ("acc" in c.lower()) or ("accuracy" in c.lower()))
        # aggregate in case of replicates
        g = df.groupby(delta_col)[acc_col].mean().reset_index()
        x = g[delta_col].astype(float).values
        y = g[acc_col].astype(float).values
        idx = np.argsort(x)
        return x[idx], y[idx]
    # (A) wide format
    # pick columns dXXX or bare numeric degrees
    deg_cols = []
    for c in df.columns:
        m = re.fullmatch(r"d?(\d+)", c.lower())
        if m:
            deg_cols.append((int(m.group(1)), c))
    if not deg_cols:
        # maybe names like "d0","d15",... with extra prefix/suffix?
        for c in df.columns:
            m = re.search(r"(\d+)", c)
            if m and c.lower().startswith("d"):
                deg_cols.append((int(m.group(1)), c))
    if not deg_cols:
        raise ValueError(f"Could not recognize Δθ columns in {path}. Columns: {df.columns.tolist()}")
    deg_cols.sort(key=lambda t: t[0])
    degrees = [d for d,_ in deg_cols]
    cols    = [c for _,c in deg_cols]
    # typically a single row; if there are more, take the row-wise mean
    vals = df[cols].astype(float).mean(axis=0).values
    return np.array(degrees, dtype=float), vals

def auc_theta_norm(x_deg, y):
    """Trapezoidal integration over x (in degrees), normalized by 180°."""
    # fill possible NaNs with simple forward/back fill
    y = pd.Series(y).ffill().bfill().values
    auc = np.trapz(y, x_deg)
    return float(auc / 180.0)

def worst_case(y):
    y = pd.Series(y).ffill().bfill().values
    return float(np.min(y))

# --- load both curves ---
x_vgg, y_vgg = load_curve(VGG_CSV, SEP)
x_cy , y_cy  = load_curve(CYVGG_CSV, SEP)

# sanity: if grids differ, resample to a common grid (assume VGG grid)
if len(x_vgg) != len(x_cy) or np.any(x_vgg != x_cy):
    # simple linear resampling of Cy onto VGG grid
    y_cy = np.interp(x_vgg, x_cy, y_cy)
    x_cy = x_vgg

# --- metrics ---
auc_vgg = auc_theta_norm(x_vgg, y_vgg)
auc_cy  = auc_theta_norm(x_cy , y_cy )
w_vgg   = worst_case(y_vgg)
w_cy    = worst_case(y_cy)

# --- plot ---
plt.figure(figsize=(8.8, 5.2), dpi=150)
plt.plot(x_vgg, y_vgg, label=f"VGG19-log  (AUCθ={auc_vgg:.3f}, worst={w_vgg:.3f})")
plt.plot(x_cy , y_cy , label=f"CyVGG19-log (AUCθ={auc_cy:.3f}, worst={w_cy:.3f})")
plt.xlabel("Δθ [deg]")
plt.ylabel("Acc (micro)")
plt.title("GTSRB_RGB — Acc(Δθ): VGG19-log vs CyVGG19-log")
plt.xlim(0, 180)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PNG)
print(f"[OK] Saved: {OUT_PNG}")

# --- print differences (for use in text) ---
print("\nDifference summary (CyVGG19-log minus VGG19-log):")
print(f"AUCθ: {auc_cy:.3f} – {auc_vgg:.3f}  (Δ {auc_cy-auc_vgg:+.3f})")
print(f"worst: {w_cy:.3f} – {w_vgg:.3f}  (Δ {w_cy-w_vgg:+.3f})")
