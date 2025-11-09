# save as: scripts/plot_perf_vs_time_scatter_multi.py
# Python 3.x
import os, re, csv, math, argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ========================= Defaults & Styles =========================
DEFAULT_DATASETS = ["MNIST", "GTSRB", "GTSRB_RGB", "LEGO"]
DEFAULT_METRIC   = "micro"
EXPORT_ROOT_TPL  = "results/exports/{dataset}/{metric}"

FAMILY_COLORS = {
    "VGG19":      "#F28E2B",
    "ResNet56":   "#E15759",
    "CyVGG19":    "#4E79A7",
    "CyResNet56": "#59A14F",
    "Other":      "#9e9e9e",
}

# Label appearance
USE_SHORT_LABELS = False        # <- FULL labels by default
MAX_LABEL_LEN    = 2000         # effectively unlimited when USE_SHORT_LABELS=False
FONT_SIZE        = 7.5
BOX_ALPHA        = 0.55
ARROW_ALPHA      = 0.6

# Repulsion (applied after selection)
REPEL_ITERS      = 400
REPEL_STRENGTH   = 0.004
MIN_SEP_X_CHARS  = 1.1
MIN_SEP_Y_LINES  = 1.0

# ============================== Helpers ==============================
FAM_RX = [
    ("CyResNet56", re.compile(r"(?i)\bcy[-_ ]?resnet(?:[-_ ]?56)?\b")),
    ("ResNet56",   re.compile(r"(?i)\bresnet[-_ ]?56\b")),
    ("CyVGG19",    re.compile(r"(?i)\bcy[-_ ]?vgg[-_ ]?19\b")),
    ("VGG19",      re.compile(r"(?i)\bvgg[-_ ]?19\b")),
]

def detect_family(model: str) -> str:
    for fam, rx in FAM_RX:
        if rx.search(model):
            return fam
    return "Other"

def make_label_full(model_name: str) -> str:
    """Return the full model name (optionally shortened if the flag is on)."""
    if not USE_SHORT_LABELS:
        return model_name
    # optional compact form (kept for completeness; off by default)
    s = re.sub(r"[\s]+", " ", model_name).strip()
    return s if len(s) <= MAX_LABEL_LEN else (s[:MAX_LABEL_LEN-1] + "…")

def read_avg(root_dir: str) -> Dict[str, float]:
    """Load average accuracy per model from ranking_quality.csv."""
    p = os.path.join(root_dir, "ranking_quality.csv")
    if not os.path.exists(p): return {}
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f); hdr = next(r); H = {h:i for i,h in enumerate(hdr)}
        for row in r:
            try: out[row[H["model"]]] = float(row[H["avg"]])
            except: pass
    return out

def read_avgperf(root_dir: str) -> List[Tuple[str,float,float]]:
    """Load avg_perf (= mean(acc)/time) per model from ranking_timeaware_avgperf.csv."""
    p = os.path.join(root_dir, "ranking_timeaware_avgperf.csv")
    if not os.path.exists(p): return []
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f); hdr = next(r); H = {h:i for i,h in enumerate(hdr)}
        for row in r:
            try:
                model = row[H["model"]]
                avg_perf = float(row[H["avg_perf"]])
                time_s = float(row[H["time_s"]]) if "time_s" in H and row[H["time_s"]] != "" else None
                rows.append((model, avg_perf, time_s))
            except: pass
    return rows

def build_points(avg_map: Dict[str,float], avgperf_rows: List[Tuple[str,float,float]]):
    """Join avg and avg_perf on model; return list[(model, avg, avg_perf, family)] and per-family counts."""
    pts = []
    fam_count = defaultdict(int)
    for model, avg_perf, _t in avgperf_rows:
        if model not in avg_map:
            continue
        avg = avg_map[model]
        fam = detect_family(model)
        pts.append((model, avg, avg_perf, fam))
        fam_count[fam] += 1
    return pts, fam_count

# --- approximate label bbox in *data* units (for spacing/repel) --------------
def approx_text_bbox(x, y, text, ax, fontsize=FONT_SIZE):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    span_x = (x1 - x0); span_y = (y1 - y0)
    char_w = span_x * 0.006 * (fontsize / 10.0)
    line_h = span_y * 0.018 * (fontsize / 10.0)
    nlines = 1 + text.count("\n")
    return (max(1, len(text)) * char_w)/2.0, (nlines * line_h)/2.0

def repel_labels(points, labels, ax, iters=REPEL_ITERS, strength=REPEL_STRENGTH):
    """Simple pairwise repulsion to reduce label overlaps."""
    pos = {}
    for (model, x, y, fam), lab in zip(points, labels):
        seed = (hash(model) % 997) / 997.0
        dx = (ax.get_xlim()[1]-ax.get_xlim()[0]) * 0.002 * (seed-0.5)
        dy = (ax.get_ylim()[1]-ax.get_ylim()[0]) * 0.004 * (seed-0.5)
        pos[model] = [x+dx, y+dy]

    xmin,xmax = ax.get_xlim(); ymin,ymax = ax.get_ylim()
    models = [m for (m, *_rest) in points]

    for _ in range(iters):
        moved = 0
        for i, mi in enumerate(models):
            xi, yi = pos[mi]
            wi, hi = approx_text_bbox(xi, yi, labels[i], ax)
            for j in range(i+1, len(models)):
                mj = models[j]
                xj, yj = pos[mj]
                wj, hj = approx_text_bbox(xj, yj, labels[j], ax)
                sep_x = max(wi, wj) * MIN_SEP_X_CHARS
                sep_y = max(hi, hj) * MIN_SEP_Y_LINES
                if abs(xi-xj) < (wi+wj+sep_x) and abs(yi-yj) < (hi+hj+sep_y):
                    vx, vy = xi-xj, yi-yj
                    norm = math.hypot(vx, vy) or 1.0
                    ux, uy = vx/norm, vy/norm
                    step = strength * norm
                    xi += ux*step; yi += uy*step
                    xj -= ux*step; yj -= uy*step
                    pos[mi] = [xi, yi]; pos[mj] = [xj, yj]; moved += 1
            # keep inside plot area with small margins
            xm = (xmax-xmin)*0.01; ym = (ymax-ymin)*0.01
            xi = min(max(xi, xmin+xm), xmax-xm)
            yi = min(max(yi, ymin+ym), ymax-ym)
            pos[mi] = [xi, yi]
        if moved == 0:
            break
    return pos

# ===================== Label selection strategies =====================
def select_labels_per_family(points, k_per_family=1):
    """Pick up to K labels per family by highest avg_perf."""
    idx = []
    byfam = defaultdict(list)
    for i, p in enumerate(points):
        byfam[p[3]].append((i, p))
    for fam, arr in byfam.items():
        arr.sort(key=lambda t: -t[1][2])  # by avg_perf desc
        idx += [i for (i, _) in arr[:k_per_family]]
    return sorted(set(idx))

def select_labels_grid(points, ncols=5, nrows=4):
    """Split plot into a grid and keep 1 representative (highest z-score) per cell."""
    if not points: return []
    xs = np.array([p[1] for p in points]); ys = np.array([p[2] for p in points])
    xmin,xmax = xs.min(), xs.max(); ymin,ymax = ys.min(), ys.max()
    bx = np.linspace(xmin, xmax, ncols+1)
    by = np.linspace(ymin, ymax, nrows+1)
    a = (xs - xs.mean())/(xs.std()+1e-9)
    b = (ys - ys.mean())/(ys.std()+1e-9)
    score = np.hypot(a, b)
    chosen = {}
    for i,(m,x,y,f) in enumerate(points):
        cx = np.searchsorted(bx, x, side="right")-1
        cy = np.searchsorted(by, y, side="right")-1
        cx = max(0, min(ncols-1, cx)); cy = max(0, min(nrows-1, cy))
        key = (cx,cy)
        if key not in chosen or score[i] > chosen[key][0]:
            chosen[key] = (score[i], i)
    return sorted(set([i for _, i in chosen.values()]))

def filter_min_distance(points, base_idx, min_dx=0.02, min_dy=0.0002):
    """Drop labels closer than thresholds (data units) to already-kept labels."""
    keep, used = [], []
    for i in base_idx:
        xi, yi = points[i][1], points[i][2]
        ok = True
        for j in used:
            xj, yj = points[j][1], points[j][2]
            if abs(xi-xj) < min_dx and abs(yi-yj) < min_dy:
                ok = False; break
        if ok:
            keep.append(i); used.append(i)
    return keep

# =============================== Plotting ===============================
def plot_scatter(points, fam_count, dataset, metric, out_path,
                 label_strategy="per_family",
                 k_per_family=1, grid_cols=5, grid_rows=4,
                 min_dx=0.02, min_dy=0.0002):
    plt.figure(figsize=(12, 9), dpi=140); ax = plt.gca()

    # scatter points
    for (model, avg, avg_perf, fam) in points:
        ax.scatter(avg, avg_perf, s=22, alpha=0.78,
                   color=FAMILY_COLORS.get(fam, FAMILY_COLORS["Other"]),
                   linewidths=0)

    # legend
    handles, labels = [], []
    for fam, col in FAMILY_COLORS.items():
        if fam_count.get(fam, 0) > 0:
            h = plt.Line2D([0],[0], marker="o", linestyle="", markersize=6, color=col, label=fam)
            handles.append(h); labels.append(fam)
    if handles:
        ax.legend(handles, labels, title="Family", loc="upper left")

    # axes & title
    ax.set_xlabel("Avg accuracy")
    ax.set_ylabel("Avg perf (mean(acc) / train_time)")
    ax.set_title(f"{dataset} — per-time vs average (metric={metric})")
    ax.grid(True, alpha=0.25)

    # comfortable margins
    if points:
        xs = np.array([p[1] for p in points]); ys = np.array([p[2] for p in points])
        pad_x = (xs.max() - xs.min()) * 0.08 if xs.max() > xs.min() else 0.02
        pad_y = (ys.max() - ys.min()) * 0.15 if ys.max() > ys.min() else 0.005
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    # ----- label selection -----
    if label_strategy == "none":
        idx_to_label = []
    elif label_strategy == "per_family":
        idx = select_labels_per_family(points, k_per_family=k_per_family)
        idx_to_label = filter_min_distance(points, idx, min_dx=min_dx, min_dy=min_dy)
    elif label_strategy == "grid":
        idx = select_labels_grid(points, ncols=grid_cols, nrows=grid_rows)
        idx_to_label = filter_min_distance(points, idx, min_dx=min_dx, min_dy=min_dy)
    else:
        idx_to_label = []

    labels = []
    sel_points = []
    for i in idx_to_label:
        m, x, y, fam = points[i]
        labels.append(make_label_full(m))
        sel_points.append(points[i])

    # repelled annotations
    if sel_points:
        pos = repel_labels(sel_points, labels, ax)
        for (model, x, y, fam), lab in zip(sel_points, labels):
            lx, ly = pos[model]
            ax.annotate(
                lab, (x, y), xytext=(lx, ly), textcoords="data",
                fontsize=FONT_SIZE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=BOX_ALPHA),
                arrowprops=dict(arrowstyle="-", lw=0.6,
                                color=FAMILY_COLORS.get(fam, "#666666"), alpha=ARROW_ALPHA),
                ha="left", va="center"
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

# ================================ Runner =================================
def run_for_dataset(dataset: str, metric: str, exports_root: str, out_dir: str, args):
    root = EXPORT_ROOT_TPL.format(dataset=dataset, metric=metric)
    out  = os.path.join(out_dir, f"perf_vs_time_scatter_{dataset}_{metric}.png")

    avg_map      = read_avg(root)
    avgperf_rows = read_avgperf(root)
    points, fam_count = build_points(avg_map, avgperf_rows)
    if not points:
        print(f"[WARN] No points for {dataset}/{metric}"); return

    # stable ordering (family, then by accuracy/perf)
    points.sort(key=lambda t: (t[3], -t[1], -t[2]))
    print(f"[INFO] {dataset}: points per family -> {dict(fam_count)} (total={len(points)})")

    plot_scatter(
        points, fam_count, dataset, metric, out,
        label_strategy=args.label_strategy,
        k_per_family=args.labels_per_family,
        grid_cols=args.grid_cols, grid_rows=args.grid_rows,
        min_dx=args.min_dx, min_dy=args.min_dy
    )

def main():
    global USE_SHORT_LABELS, REPEL_ITERS, REPEL_STRENGTH

    ap = argparse.ArgumentParser(
        description="Scatter plots (avg accuracy vs per-time performance) with readable labels."
    )
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Datasets to plot. If omitted and --all not set, uses default list.")
    ap.add_argument("--all", action="store_true", help="Plot all default datasets.")
    ap.add_argument("--metric", default="micro", choices=["micro","macro"])
    ap.add_argument("--exports-root", default="results/exports")
    ap.add_argument("--out-dir", default="results/fig")

    # Labeling strategies
    ap.add_argument("--label-strategy", default="per_family",
                    choices=["none","per_family","grid"],
                    help="none = no labels; per_family = up to K per family; grid = up to 1 per grid cell.")
    ap.add_argument("--labels-per-family", type=int, default=1,
                    help="K labels per family (used by --label-strategy=per_family).")
    ap.add_argument("--grid-cols", type=int, default=5, help="Grid columns (for grid strategy).")
    ap.add_argument("--grid-rows", type=int, default=4, help="Grid rows (for grid strategy).")
    ap.add_argument("--min-dx", type=float, default=0.02, help="Min X distance between labels after selection.")
    ap.add_argument("--min-dy", type=float, default=0.0002, help="Min Y distance between labels after selection.")
    ap.add_argument("--short", action="store_true", help="Use shortened labels (OFF by default).")
    ap.add_argument("--repel-iters", type=int, default=REPEL_ITERS)
    ap.add_argument("--repel-strength", type=float, default=REPEL_STRENGTH)

    args = ap.parse_args()
    USE_SHORT_LABELS = args.short
    REPEL_ITERS      = args.repel_iters
    REPEL_STRENGTH   = args.repel_strength

    datasets = DEFAULT_DATASETS if args.all else (args.datasets or DEFAULT_DATASETS)
    for ds in datasets:
        run_for_dataset(ds, args.metric, args.exports_root, args.out_dir, args)

if __name__ == "__main__":
    main()
