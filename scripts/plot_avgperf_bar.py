#!/usr/bin/env python3
# scripts/plot_avgperf_bar.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import csv, os, sys, argparse
import matplotlib.pyplot as plt

def load_rows(csv_path: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                rows.append({
                    "model": row["model"],
                    "avg_perf": float(row["avg_perf"]),
                    "robust_perf": float(row["robust_perf"]) if row.get("robust_perf") not in ("", None) else None,
                    "time_s": float(row["time_s"]) if row.get("time_s") not in ("", None) else None,
                })
            except Exception:
                pass
    rows.sort(key=lambda r: r["avg_perf"], reverse=True)
    return rows

def plot_avgperf(
    csv_path: str,
    out_png: str,
    *,
    top_k: int = 30,
    horizontal: bool = True,
    title: str | None = None,
    base_font: float = 8.0,
    tick_font: float = 8.0,
    label_font: float = 9.0,
    title_font: float = 10.0,
    legend_font: float = 8.0,
    figsize_w: float | None = None,
    figsize_h: float | None = None,
    dpi: int = 300,
    strip_prefix: str | None = None,
):
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No data rows in CSV.")
    rows = rows[:max(1, top_k)]

    labels = [(r["model"][len(strip_prefix):] if strip_prefix and r["model"].startswith(strip_prefix) else r["model"])
              for r in rows]
    avgp = [r["avg_perf"] for r in rows]
    robp = [r["robust_perf"] for r in rows]

    # ---- figure size (no wrapping): widen automatically for long labels ----
    max_label_len = max(len(s) for s in labels)
    if figsize_w is None or figsize_h is None:
        if horizontal:
            n = len(rows)
            auto_h = max(4.0, 0.28 * n)        # 0.28" per item
            # add width if labels are long (rough heuristic: 0.12" per char)
            auto_w = max(10.0, 8.0 + 0.12 * max_label_len)
            figsize = (figsize_w or auto_w, figsize_h or auto_h)
        else:
            n = len(rows)
            auto_w = max(10.0, 8.0 + 0.06 * max_label_len)
            auto_h = 6.0
            figsize = (figsize_w or auto_w, figsize_h or auto_h)
    else:
        figsize = (figsize_w, figsize_h)

    plt.rcParams["font.size"] = base_font
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    if horizontal:
        y = list(range(len(rows)))
        ax.barh(y, avgp, label="avg_perf")
        if any(v is not None for v in robp):
            ys = [i for i, v in enumerate(robp) if v is not None]
            xs = [v for v in robp if v is not None]
            ax.plot(xs, ys, marker="o", linestyle="none", label="robust_perf")

        ax.set_yticks(y, labels=labels)
        ax.tick_params(axis="y", labelsize=tick_font)
        ax.tick_params(axis="x", labelsize=tick_font)
        ax.invert_yaxis()
        ax.set_xlabel("performance per time (Acc/time)", fontsize=label_font)

        # leave extra room for long y-labels on the left
        plt.subplots_adjust(left=0.28, right=0.98, top=0.90, bottom=0.06)
    else:
        x = list(range(len(rows)))
        ax.bar(x, avgp, label="avg_perf")
        if any(v is not None for v in robp):
            xs = [i for i, v in enumerate(robp) if v is not None]
            ys = [v for v in robp if v is not None]
            ax.plot(xs, ys, marker="o", linestyle="none", label="robust_perf")

        ax.set_xticks(x, labels=labels, rotation=30, ha="right")
        ax.tick_params(axis="x", labelsize=tick_font)
        ax.tick_params(axis="y", labelsize=tick_font)
        ax.set_ylabel("performance per time (Acc/time)", fontsize=label_font)
        plt.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.35)

    if title:
        ax.set_title(title, fontsize=title_font)

    ax.grid(True, axis="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, prop={"size": legend_font})

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print("Saved:", out_png)

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Plot time-aware performance (avg_perf, robust_perf) without wrapping labels.")
    ap.add_argument("csv", help="ranking_timeaware_avgperf.csv")
    ap.add_argument("out", help="output PNG")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--horizontal", action="store_true", help="use horizontal bars (default)")
    ap.add_argument("--vertical", action="store_true", help="use vertical bars")
    ap.add_argument("--base-font", type=float, default=8.0)
    ap.add_argument("--tick-font", type=float, default=8.0)
    ap.add_argument("--label-font", type=float, default=9.0)
    ap.add_argument("--title-font", type=float, default=10.0)
    ap.add_argument("--legend-font", type=float, default=8.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fig-w", type=float, default=None)
    ap.add_argument("--fig-h", type=float, default=None)
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--strip-prefix", type=str, default=None)
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    horizontal = True
    if args.vertical:
        horizontal = False
    elif args.horizontal:
        horizontal = True

    plot_avgperf(
        args.csv, args.out,
        top_k=args.top_k, horizontal=horizontal, title=args.title,
        base_font=args.base_font, tick_font=args.tick_font, label_font=args.label_font,
        title_font=args.title_font, legend_font=args.legend_font,
        figsize_w=args.fig_w, figsize_h=args.fig_h, dpi=args.dpi,
        strip_prefix=args.strip_prefix,
    )

if __name__ == "__main__":
    main()
