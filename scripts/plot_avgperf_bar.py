#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-time ranking (avg_perf) — readable bar chart for many models.

Examples:
  python scripts/plot_avgperf_bar.py results/exports/LEGO/micro/ranking_timeaware_avgperf.csv results/fig/avg_perf_LEGO.png --top-k 100
  python scripts/plot_avgperf_bar.py <in.csv> <out.png> --horizontal   # (default)
  python scripts/plot_avgperf_bar.py <in.csv> <out.png> --vertical
"""

import csv
import os
import argparse
import textwrap
import matplotlib.pyplot as plt


def _clean_label(name: str) -> str:
    """Simplify very long model names (safe heuristics)."""
    s = name
    for tok in (
        "custom_dataset_", "dataset_", "merged_datasets_", "non_rotated_", "train_on_",
        "results_", "logs_", "GTSRB_RGB_", "GTSRB_", "MNIST_", "LEGO_"
    ):
        s = s.replace(tok, "")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_-")


def _wrap_label(s: str, width: int) -> str:
    """Word-wrap long labels at nice break points."""
    s = s.replace("/", " / ")
    s = s.replace("-", " - ")
    s = s.replace("_", " _ ")
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False))


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                rows.append({
                    "model": row["model"],
                    "avg_perf": float(row["avg_perf"]),
                    "robust_perf": float(row["robust_perf"]) if row.get("robust_perf") not in (None, "",) else None,
                    "time_s": float(row["time_s"]) if row.get("time_s") not in (None, "",) else None,
                })
            except Exception:
                # skip malformed/unparsable rows
                pass
    rows.sort(key=lambda r: r["avg_perf"], reverse=True)
    return rows


def plot_avgperf(csv_path, out_png, top_k=15, title=None,
                 wrap=28, do_clean=True, horizontal=True):
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No data rows in CSV.")
    rows = rows[:top_k]

    labels_raw = [r["model"] for r in rows]
    if do_clean:
        labels_raw = [_clean_label(x) for x in labels_raw]
    labels = [_wrap_label(x, wrap) for x in labels_raw]

    avgp = [r["avg_perf"] for r in rows]
    robp = [r["robust_perf"] for r in rows]

    # dynamic figure size based on item count and orientation
    if horizontal:
        h = max(6.0, 0.35 * len(rows))   # ~0.35" per row
        w = 12.0
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
        y = list(range(len(rows)))[::-1]  # best at the top
        ax.barh(y, avgp[::-1], label="avg_perf")
        rp = robp[::-1]
        ys = [y[i] for i, v in enumerate(rp) if v is not None]
        xs = [v for v in rp if v is not None]
        if xs:
            ax.plot(xs, ys, "o", label="robust_perf")
        ax.set_yticks(list(range(len(labels)))[::-1], labels[::-1])
        ax.invert_yaxis()
        ax.set_xlabel("performance per time (Acc/time)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    else:
        w = max(12.0, 0.28 * len(rows))
        h = 7.0
        fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
        x = list(range(len(rows)))
        ax.bar(x, avgp, label="avg_perf")
        xs = [i for i, v in enumerate(robp) if v is not None]
        ys = [v for v in robp if v is not None]
        if ys:
            ax.plot(xs, ys, "o", label="robust_perf")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("performance per time (Acc/time)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    if title:
        ax.set_title(title)

    ax.legend(loc="best", frameon=True)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="ranking_timeaware_avgperf.csv")
    ap.add_argument("out_png", help="output PNG path")
    ap.add_argument("--top-k", type=int, default=15, help="how many top items to show")
    ap.add_argument("--wrap", type=int, default=28, help="wrap labels every N characters")
    ap.add_argument("--no-clean", action="store_true", help="do not clean/simplify model names")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--horizontal", action="store_true", help="horizontal bar chart (default)")
    g.add_argument("--vertical", action="store_true", help="vertical bar chart")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    horizontal = True
    if args.vertical:
        horizontal = False
    elif args.horizontal:
        horizontal = True

    plot_avgperf(
        args.csv_path,
        args.out_png,
        top_k=args.top_k,
        title=args.title,
        wrap=args.wrap,
        do_clean=(not args.no_clean),
        horizontal=horizontal,
    )


if __name__ == "__main__":
    main()
