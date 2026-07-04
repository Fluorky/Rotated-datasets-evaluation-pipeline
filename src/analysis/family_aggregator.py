import os
import re
import csv
import math
import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

# ---------------------------- config (defaults) -----------------------------

DEFAULT_DATASET = "LEGO"  # MNIST | GTSRB | GTSRB_RGB | LEGO
DEFAULT_METRIC = "micro"
EXPORT_ROOT_TPL = "results/exports/{dataset}/{metric}"
OUT_TPL = "results/fig/perf_vs_time_scatter_{dataset}_{metric}.png"

# Labeling
LABEL_ALL = True  # label every point
SHORTEN_LABELS = True  # shorten long model names
MAX_LABEL_LEN = 60  # hard cap after shortening
FONT_SIZE = 7.5
BOX_ALPHA = 0.55
ARROW_ALPHA = 0.6

# Collision-avoidance (label repelling)
REPEL_ITERS = 250  # iterations of the repelling loop
REPEL_STRENGTH = 0.003  # base step size in data units (scaled internally)
MIN_SEP_X_CHARS = 1.2  # desired horizontal separation in "character widths"
MIN_SEP_Y_LINES = 1.0  # desired vertical separation in "text line heights"

# Colors per family (consistent & colorblind-friendly-ish)
FAMILY_COLORS = {
    "VGG19": "#F28E2B",
    "ResNet56": "#E15759",
    "CyVGG19": "#4E79A7",
    "CyResNet56": "#59A14F",
}

# ------------------------------ helpers -------------------------------------

FAM_RX = [
    ("CyResNet56", re.compile(r"(?i)\bcy[-_ ]?resnet(?:[-_ ]?56)?\b")),
    ("ResNet56", re.compile(r"(?i)\bresnet[-_ ]?56\b")),
    ("CyVGG19", re.compile(r"(?i)\bcy[-_ ]?vgg[-_ ]?19\b")),
    ("VGG19", re.compile(r"(?i)\bvgg[-_ ]?19\b")),
]


def detect_family(model: str) -> str:
    """Return family name based on model string."""
    for fam, rx in FAM_RX:
        if rx.search(model):
            return fam
    return "Other"


def shorten_model_label(s: str) -> str:
    """Make labels shorter but still informative."""
    if not SHORTEN_LABELS:
        return s[:MAX_LABEL_LEN]
    # Drop dataset prefix if present (e.g., "LEGO-" at start)
    s2 = re.sub(r"^(MNIST|GTSRB|GTSRB[_-]RGB|LEGO)[-_]", "", s, flags=re.IGNORECASE)
    # Collapse multiple underscores/dashes/spaces
    s2 = re.sub(r"[\s_-]+", "_", s2).strip("_")

    # Keep the most useful tail (often transform + angle); ensure family stays
    fam = detect_family(s)
    if fam != "Other":
        # ensure family token present at front
        if not s2.lower().startswith(fam.lower()):
            s2 = f"{fam}_{s2}"
    return (s2[:MAX_LABEL_LEN]).strip("_")


def read_avg(root_dir: str) -> dict:
    """Read avg accuracy from ranking_quality.csv (model -> avg)."""
    p = os.path.join(root_dir, "ranking_quality.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Missing file: {p}")
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        H = {h: i for i, h in enumerate(hdr)}
        for row in r:
            try:
                model = row[H["model"]]
                avg = float(row[H["avg"]])
                out[model] = avg
            except Exception:
                continue
    return out


def read_avgperf(root_dir: str) -> list:
    """Read avg_perf from ranking_timeaware_avgperf.csv → list of (model, avg_perf, time_s/None)."""
    p = os.path.join(root_dir, "ranking_timeaware_avgperf.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Missing file: {p}")
    pts = []
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        H = {h: i for i, h in enumerate(hdr)}
        for row in r:
            try:
                model = row[H["model"]]
                avg_perf = float(row[H["avg_perf"]])
                time_s = float(row[H["time_s"]]) if "time_s" in H and row[H["time_s"]] != "" else None
                pts.append((model, avg_perf, time_s))
            except Exception:
                continue
    return pts


def build_points(avg_map: dict, avgperf_rows: list):
    """Join by model → (model, avg, avg_perf, family)."""
    points = []
    fam_count = defaultdict(int)
    for model, avg_perf, _time in avgperf_rows:
        if model not in avg_map:
            continue
        avg = avg_map[model]
        fam = detect_family(model)
        points.append((model, avg, avg_perf, fam))
        fam_count[fam] += 1
    return points, fam_count


# --- geometry helpers for label repelling (data-space) ---

def approx_text_bbox(x, y, text, ax, fontsize=FONT_SIZE):
    """
    Very rough bbox in data units:
    width ~ len(text) * char_w, height ~ 1 line height.
    Char width & line height are scaled from current data limits.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = (x1 - x0)
    span_y = (y1 - y0)

    # scale factors tuned for typical matplotlib default font metrics
    char_w = span_x * 0.006 * (fontsize / 10.0)
    line_h = span_y * 0.018 * (fontsize / 10.0)

    # allow for multi-line after shortening (if underscores etc.)
    nlines = 1 + text.count("\n")
    width = max(1, len(text)) * char_w
    height = nlines * line_h
    # return half sizes for convenience
    return width / 2.0, height / 2.0


def repel_labels(points, ax, iters=REPEL_ITERS, strength=REPEL_STRENGTH):
    """
    Naive label repelling: move label anchors (x,y) away from overlaps.
    Returns dict: model -> (lx, ly) label position in data coords.
    """
    # start from small offsets (so arrows are visible)
    pos = {}
    for model, x, y, fam in points:
        # small deterministic jitter based on model hash
        seed = (hash(model) % 997) / 997.0
        dx = (seed - 0.5) * (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.002
        dy = (seed - 0.5) * (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.004
        pos[model] = [x + dx, y + dy]

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for _ in range(iters):
        moved = 0
        models = [m for m, *_ in points]
        for i in range(len(models)):
            mi = models[i]
            xi, yi = pos[mi]
            wi, hi = approx_text_bbox(xi, yi, mi, ax)
            for j in range(i + 1, len(models)):
                mj = models[j]
                xj, yj = pos[mj]
                wj, hj = approx_text_bbox(xj, yj, mj, ax)

                # check overlap in data space (expanded by desired min sep)
                sep_x = max(wi, wj) * MIN_SEP_X_CHARS
                sep_y = max(hi, hj) * MIN_SEP_Y_LINES

                if abs(xi - xj) < (wi + wj + sep_x) and abs(yi - yj) < (hi + hj + sep_y):
                    # push apart along the connecting vector
                    vx = xi - xj
                    vy = yi - yj
                    norm = math.hypot(vx, vy) or 1.0
                    ux, uy = vx / norm, vy / norm
                    step = strength * norm  # a bit adaptive
                    xi += ux * step
                    yi += uy * step
                    xj -= ux * step
                    yj -= uy * step
                    pos[mi] = [xi, yi]
                    pos[mj] = [xj, yj]
                    moved += 1

            # keep inside axes with a small margin
            xm = (xmax - xmin) * 0.01
            ym = (ymax - ymin) * 0.01
            xi = min(max(xi, xmin + xm), xmax - xm)
            yi = min(max(yi, ymin + ym), ymax - ym)
            pos[mi] = [xi, yi]

        if moved == 0:
            break
    return pos


# ------------------------------- plotting -----------------------------------

def plot_scatter(points, fam_count, dataset, metric, out_path):
    plt.figure(figsize=(10.5, 8.0), dpi=140)
    ax = plt.gca()

    # draw points
    for (model, avg, avg_perf, fam) in points:
        ax.scatter(avg, avg_perf, s=22, alpha=0.75,
                   color=FAMILY_COLORS.get(fam, "#999999"), linewidths=0.0)

    # legend
    handles = []
    labels = []
    for fam, col in FAMILY_COLORS.items():
        if fam_count.get(fam, 0) > 0:
            h = plt.Line2D([0], [0], marker="o", linestyle="",
                           markersize=6, color=col, label=fam)
            handles.append(h);
            labels.append(fam)
    ax.legend(handles, labels, title="Family", loc="upper left")

    ax.set_xlabel("Avg accuracy")
    ax.set_ylabel("Avg perf (mean(acc) / train_time)")
    ax.set_title(f"{dataset} — per-time vs average (metric={metric})")
    ax.grid(True, alpha=0.25)

    # label everything (with repelling)
    if LABEL_ALL and points:
        # shorten labels
        labels_map = {m: shorten_model_label(m) for (m, *_rest) in points}

        # ensure fixed limits before repelling
        xs = [p[1] for p in points];
        ys = [p[2] for p in points]
        pad_x = (max(xs) - min(xs)) * 0.04 if xs else 0.05
        pad_y = (max(ys) - min(ys)) * 0.08 if ys else 0.005
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

        pos = repel_labels(points, ax, iters=REPEL_ITERS, strength=REPEL_STRENGTH)

        for (model, x, y, fam) in points:
            lx, ly = pos[model]
            txt = labels_map[model]
            ax.annotate(
                txt, (x, y), xytext=(lx, ly),
                textcoords="data",
                fontsize=FONT_SIZE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=BOX_ALPHA),
                arrowprops=dict(arrowstyle="-", lw=0.6,
                                color=FAMILY_COLORS.get(fam, "#666666"),
                                alpha=ARROW_ALPHA),
                ha="left", va="center"
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


# ------------------------------- main ---------------------------------------

def main():
    global LABEL_ALL, SHORTEN_LABELS, REPEL_ITERS, REPEL_STRENGTH

    ap = argparse.ArgumentParser(
        description="Scatter: avg accuracy vs per-time performance with non-overlapping labels.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--metric", default=DEFAULT_METRIC, choices=["micro", "macro"])
    ap.add_argument("--exports-root", default="results/exports")
    ap.add_argument("--out", default=None, help="Output PNG; default based on dataset/metric.")
    ap.add_argument("--label-all", action="store_true", help="Force labeling of all points.")
    ap.add_argument("--no-short", action="store_true", help="Do not shorten labels.")
    ap.add_argument("--repel-iters", type=int, default=REPEL_ITERS)
    ap.add_argument("--repel-strength", type=float, default=REPEL_STRENGTH)
    args = ap.parse_args()

    if args.label_all:
        LABEL_ALL = True
    SHORTEN_LABELS = not args.no_short
    REPEL_ITERS = args.repel_iters
    REPEL_STRENGTH = args.repel_strength

    root = os.path.join(args.exports_root, args.dataset, args.metric)
    out = args.out or OUT_TPL.format(dataset=args.dataset, metric=args.metric)

    avg_map = read_avg(root)
    avgperf_rows = read_avgperf(root)
    points, fam_count = build_points(avg_map, avgperf_rows)

    if not points:
        raise SystemExit("No common models between ranking_quality.csv and ranking_timeaware_avgperf.csv.")

    print("[INFO] Points per family:", dict(fam_count))
    # Sort for stable rendering (helps legend/overdraw)
    points.sort(key=lambda t: (t[3], -t[1], -t[2]))

    plot_scatter(points, fam_count, args.dataset, args.metric, out)


if __name__ == "__main__":
    main()
