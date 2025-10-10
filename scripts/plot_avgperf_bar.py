#!/usr/bin/env python3
# scripts/plot_avgperf_bar.py
# -*- coding: utf-8 -*-
"""
Bar chart of time-aware performance from ranking_timeaware_avgperf.csv

- Reads columns: model, avg_perf, [robust_perf], [time_s]
- Shows avg_perf as bars; robust_perf as overlay points (if present)
- Defaults to HORIZONTAL bars (best for long labels)
- Smart label wrapping + small fonts to avoid overlap
- CLI flags to control top-k, fonts, wrapping, figure size, orientation, DPI

Examples
--------
python scripts/plot_avgperf_bar.py \
  results/exports/LEGO/micro/ranking_timeaware_avgperf.csv \
  results/fig/avg_perf_LEGO.png \
  --top-k 80 --wrap 22 --horizontal --dpi 300

python scripts/plot_avgperf_bar.py input.csv out.png \
  --vertical --top-k 25 --tick-font 7 --label-font 8 --title-font 9
"""

from __future__ import annotations
import csv
import os
import sys
import argparse
from textwrap import wrap
import matplotlib.pyplot as plt


# ------------------------------- IO -----------------------------------------

def load_rows(csv_path: str):
    """Load rows from ranking_timeaware_avgperf.csv and sort by avg_perf desc."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                rows.append({
                    "model": row["model"],
                    "avg_perf": float(row["avg_perf"]),
                    "robust_perf": float(row["robust_perf"]) if row.get("robust_perf", "") not in ("", None) else None,
                    "time_s": float(row["time_s"]) if row.get("time_s", "") not in ("", None) else None,
                })
            except Exception:
                # skip malformed rows
                pass
    rows.sort(key=lambda r: r["avg_perf"], reverse=True)
    return rows


# ---------------------------- label helpers ---------------------------------

def wrap_label(s: str, width: int) -> str:
    """Wrap a long label at 'width' characters per line (safe fallback)."""
    if width <= 0:
        return s
    # Prefer to break on separators if possible
    s2 = s.replace("/", " / ").replace("_", " _ ").replace("-", " - ")
    parts = wrap(s2, width=width, break_long_words=False, break_on_hyphens=True)
    if not parts:
        return s
    # Re-join and clean spaces around separators
    txt = "\n".join(p.replace(" _ ", "_").replace(" - ", "-").replace(" / ", "/") for p in parts)
    return txt


def maybe_shorten(s: str, strip_prefix: str | None) -> str:
    """Optionally strip a common prefix from labels (e.g., 'LEGO-' or 'GTSRB-')."""
    if strip_prefix and s.startswith(strip_prefix):
        return s[len(strip_prefix):]
    return s


# ------------------------------- plot ---------------------------------------

def plot_avgperf(
    csv_path: str,
    out_png: str,
    *,
    top_k: int = 30,
    horizontal: bool = True,
    title: str | None = None,
    wrap_chars: int = 22,
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
    labels = [maybe_shorten(r["model"], strip_prefix) for r in rows]
    if wrap_chars and wrap_chars > 0:
        labels = [wrap_label(s, wrap_chars) for s in labels]

    avgp = [r["avg_perf"] for r in rows]
    robp = [r["robust_perf"] for r in rows]

    # Figure size: auto if not provided
    if figsize_w is None or figsize_h is None:
        if horizontal:
            # Height scales with number of bars; width enough for wrapped labels
            n = len(rows)
            auto_h = max(4.0, 0.28 * n)   # 0.28" per item is readable
            auto_w = 10.0
            figsize = (figsize_w or auto_w, figsize_h or auto_h)
        else:
            # Vertical: make width scale with number of bars, fixed height
            n = len(rows)
            auto_w = max(8.0, 0.28 * n)
            auto_h = 6.0
            figsize = (figsize_w or auto_w, figsize_h or auto_h)
    else:
        figsize = (figsize_w, figsize_h)

    plt.rcParams["font.size"] = base_font
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)

    if horizontal:
        y = list(range(len(rows)))
        ax.barh(y, avgp, label="avg_perf")
        # overlay robust_perf as points
        if any(v is not None for v in robp):
            ys = [i for i, v in enumerate(robp) if v is not None]
            xs = [v for v in robp if v is not None]
            ax.plot(xs, ys, marker="o", linestyle="none", label="robust_perf")

        ax.set_yticks(y, labels=labels)
        ax.tick_params(axis="y", labelsize=tick_font)
        ax.tick_params(axis="x", labelsize=tick_font)
        ax.invert_yaxis()  # highest at top (since sorted desc)
        ax.set_xlabel("performance per time (Acc/time)", fontsize=label_font)

    else:
        x = list(range(len(rows)))
        ax.bar(x, avgp, label="avg_perf")
        if any(v is not None for v in robp):
            xs = [i for i, v in enumerate(robp) if v is not None]
            ys = [v for v in robp if v is not None]
            ax.plot(xs, ys, marker="o", linestyle="none", label="robust_perf")

        ax.set_xticks(x, labels=labels, rotation=35, ha="right")
        ax.tick_params(axis="x", labelsize=tick_font)
        ax.tick_params(axis="y", labelsize=tick_font)
        ax.set_ylabel("performance per time (Acc/time)", fontsize=label_font)

    if title:
        ax.set_title(title, fontsize=title_font)

    ax.grid(True, axis="both", linestyle="--", alpha=0.35)
    leg = ax.legend(frameon=True, prop={"size": legend_font})
    leg.set_alpha(0.9)

    # Tight layout, but leave room for long wrapped labels
    if horizontal:
        plt.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.06)
    else:
        plt.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.30)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print("Saved:", out_png)


# ------------------------------- CLI ----------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Plot time-aware performance (avg_perf, robust_perf).")
    ap.add_argument("csv", help="ranking_timeaware_avgperf.csv")
    ap.add_argument("out", help="output PNG")

    ap.add_argument("--top-k", type=int, default=30, help="number of items to plot (default: 30)")
    ap.add_argument("--horizontal", action="store_true", help="use horizontal bars (default)")
    ap.add_argument("--vertical", action="store_true", help="use vertical bars")

    ap.add_argument("--wrap", type=int, default=22, help="wrap labels every N characters (0 disables)")
    ap.add_argument("--strip-prefix", type=str, default=None, help="strip this prefix from labels if present")

    ap.add_argument("--base-font", type=float, default=8.0, help="global base font size")
    ap.add_argument("--tick-font", type=float, default=8.0, help="tick label font size")
    ap.add_argument("--label-font", type=float, default=9.0, help="axis label font size")
    ap.add_argument("--title-font", type=float, default=10.0, help="title font size")
    ap.add_argument("--legend-font", type=float, default=8.0, help="legend font size")

    ap.add_argument("--dpi", type=int, default=300, help="output DPI")
    ap.add_argument("--fig-w", type=float, default=None, help="figure width (inches)")
    ap.add_argument("--fig-h", type=float, default=None, help="figure height (inches)")
    ap.add_argument("--title", type=str, default=None, help="figure title")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    horizontal = True
    if args.vertical:
        horizontal = False
    elif args.horizontal:
        horizontal = True

    plot_avgperf(
        args.csv,
        args.out,
        top_k=args.top_k,
        horizontal=horizontal,
        title=args.title,
        wrap_chars=args.wrap,
        base_font=args.base_font,
        tick_font=args.tick_font,
        label_font=args.label_font,
        title_font=args.title_font,
        legend_font=args.legend_font,
        figsize_w=args.fig_w,
        figsize_h=args.fig_h,
        dpi=args.dpi,
        strip_prefix=args.strip_prefix,
    )


if __name__ == "__main__":
    main()
