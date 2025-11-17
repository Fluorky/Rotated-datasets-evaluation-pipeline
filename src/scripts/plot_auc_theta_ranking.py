# save as: scripts/plot_auc_theta_ranking.py
import os, re, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === SETTINGS ===
DATASET = "LEGO"  # e.g. "GTSRB"
CSV = f"results/exports/{DATASET}/micro/delta_curves/auc_theta_ranking.csv"
OUT_PNG = f"results/fig/auc_theta_ranking_{DATASET}_micro.png"
TOP_N = 20

FULL_LABELS = True          # show full labels (wrapped into multiple lines)
AGGREGATE_DUPLICATES = False  # for full labels, it's better to keep this False
WRAP_WIDTH = 40             # line wrap width for long labels
ROW_HEIGHT = 0.5            # height of one bar (increase for long text)
MAX_FIG_HEIGHT = 18         # maximum figure height (in inches)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ---------- HELPERS ----------
def find_col(df, candidates):
    cand_lower = [c.lower() for c in candidates]
    low2orig = {c.lower(): c for c in df.columns}
    for c in cand_lower:
        if c in low2orig:
            return low2orig[c]
    for c in df.columns:
        lc = c.lower()
        if any(x in lc for x in cand_lower):
            return c
    raise KeyError(f"Could not find any of {candidates} in {df.columns.tolist()}")

def shorten_label(model: str):
    s = model.lower()
    if "vgg19" in s:
        fam = "VGG19"
    elif any(k in s for k in ("resnet56", "resnet-56", "resnet_56", "resnet")):
        fam = "ResNet56"
    else:
        m = re.search(r"[a-z]+[a-z0-9]*", s)
        fam = m.group(0).capitalize() if m else model
    is_cy = ("cy" in s and "cyc" not in s) or "cy" in s.split("-")[0] or "cy" in s.split("_")[0]
    fam = ("Cy" + fam) if is_cy else fam
    if "logpolar" in s or re.search(r"(^|[^a-z])log([^a-z]|$)", s):
        tr = "log"
    elif "linearpolar" in s or "linear" in s:
        tr = "linear"
    else:
        tr = "?"
    return f"{fam}-{tr}", is_cy

def wrap_label(s: str, width=WRAP_WIDTH):
    s = s.replace("_", " ")
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=True)

# ---------- LOAD DATA ----------
df = pd.read_csv(CSV, encoding="utf-8")

model_col = find_col(df, ["model"])
auc_col   = find_col(df, ["auc_theta_norm", "auc_theta", "auc"])
worst_col = find_col(df, ["acc_worst", "worst"]) if any("worst" in c.lower() for c in df.columns) else None
sd_col    = find_col(df, ["sd_theta", "sd", "std"]) if any(("sd" in c.lower()) or ("std" in c.lower()) for c in df.columns) else None

# sorting: AUC descending, then worst descending, then SD ascending
sort_cols = [auc_col]
ascending = [False]
if worst_col: sort_cols += [worst_col]; ascending += [False]
if sd_col:    sort_cols += [sd_col];    ascending += [True]
df_sorted = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

# labels and Cy flag
short_and_flag = df_sorted[model_col].astype(str).map(shorten_label)
df_sorted["short_label"] = [t[0] for t in short_and_flag]
df_sorted["is_cy"]       = [t[1] for t in short_and_flag]

# select rows for plotting
plot_df = df_sorted.head(TOP_N).copy()
labels_raw = plot_df[model_col].astype(str).tolist() if FULL_LABELS else plot_df["short_label"].astype(str).tolist()
flags_cy   = plot_df["is_cy"].tolist()
auc_vals   = plot_df[auc_col].astype(float).values

# prepare labels (multi-line wrapping)
labels = [wrap_label(s) for s in labels_raw]

# ---------- PLOT (HORIZONTAL) ----------
colors = ["#2b8cbe" if cy else "#9e9e9e" for cy in flags_cy]  # Cy = blue, Classical = gray

# dynamic figure height
fig_height = min(MAX_FIG_HEIGHT, max(6, ROW_HEIGHT * len(labels)))
fig_width = 12  # horizontal plots don’t need to be too wide
plt.figure(figsize=(fig_width, fig_height), dpi=150)

y = np.arange(len(labels))
bars = plt.barh(y, auc_vals, color=colors, edgecolor="black", linewidth=0.3)

# labels on the left, aligned to the right side of y-ticks
plt.yticks(y, labels, fontsize=8)
plt.gca().invert_yaxis()  # top = best model (more natural ranking)

plt.xlabel("AUC$_\\theta$ (micro)")
plt.title(f"{DATASET} — AUC$_\\theta$ Ranking (TOP {TOP_N})")

# margins: more space on the left for long labels, a bit on the right for legend
plt.subplots_adjust(left=0.42, right=0.86, top=0.92, bottom=0.06)

# X-axis range (contrast)
xmin = max(0.0, float(auc_vals.min()) - 0.02)
xmax = min(1.0, float(auc_vals.max()) + 0.01)
plt.xlim(xmin, xmax)

# text values at the end of bars (on the right side)
for i, b in enumerate(bars):
    v = auc_vals[i]
    plt.text(v + (xmax - xmin) * 0.005, b.get_y() + b.get_height()/2,
             f"{v:.3f}", va="center", ha="left", fontsize=8)

# legend outside the plot area
cy_patch = plt.Rectangle((0,0),1,1,color="#2b8cbe")
cl_patch = plt.Rectangle((0,0),1,1,color="#9e9e9e")
plt.legend([cy_patch, cl_patch], ["Cyclic (Cy*)", "Classical"],
           loc="center left", bbox_to_anchor=(0.89, 0.5), frameon=True)

plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_PNG)
print(f"[OK] Plot saved: {OUT_PNG}")

# ---------- (optional) PRINT TABLE ----------
cols_to_show = [model_col, auc_col] + ([worst_col] if worst_col else []) + ([sd_col] if sd_col else [])
print("\nTOP configurations (no aggregation, full labels):")
print(plot_df[cols_to_show].to_string(index=False))
