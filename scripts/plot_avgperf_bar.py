import csv, os, sys
import matplotlib.pyplot as plt

def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = []
        for row in rdr:
            try:
                rows.append({
                    "model": row["model"],
                    "avg_perf": float(row["avg_perf"]),
                    "robust_perf": float(row["robust_perf"]) if "robust_perf" in row and row["robust_perf"] != "" else None,
                    "time_s": float(row["time_s"]) if "time_s" in row and row["time_s"] != "" else None,
                })
            except:
                # skip malformed/unparsable rows
                pass
    rows.sort(key=lambda r: r["avg_perf"], reverse=True)
    return rows

def plot_avgperf(csv_path, out_png, top_k=15, title=None):
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No data rows in CSV.")
    rows = rows[:top_k]

    labels = [r["model"] for r in rows]
    avgp   = [r["avg_perf"] for r in rows]
    robp   = [r["robust_perf"] for r in rows]

    x = list(range(len(rows)))

    plt.figure()
    plt.bar(x, avgp, label="avg_perf")
    # overlay robust_perf as points, if available
    if any(v is not None for v in robp):
        xs = [i for i, v in enumerate(robp) if v is not None]
        ys = [v for v in robp if v is not None]
        plt.plot(xs, ys, marker="o", linestyle="none", label="robust_perf")

    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("performance per time (Acc/time)")
    if title:
        plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=2000)
    print("Saved:", out_png)

if __name__ == "__main__":
    # Usage:
    # python scripts/plot_avgperf_bar.py <ranking_timeaware_avgperf.csv> <out.png> [top_k]
    if len(sys.argv) < 3:
        print("Usage: python plot_avgperf_bar.py <ranking_timeaware_avgperf.csv> <out.png> [top_k]")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_png  = sys.argv[2]
    top_k    = int(sys.argv[3]) if len(sys.argv) >= 4 else 15
    # optional title derived from path (left as None by default)
    title = None
    plot_avgperf(csv_path, out_png, top_k=top_k, title=title)
