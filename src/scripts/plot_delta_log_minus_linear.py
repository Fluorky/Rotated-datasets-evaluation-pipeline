import csv, os, sys
import matplotlib.pyplot as plt

# --- helpers ---------------------------------------------------------------

def read_rows(path):
    """Read CSV rows into a list of dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def get_float(row, candidates):
    """Try to parse a float from one of several possible column names."""
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            try:
                return float(row[c])
            except:
                pass
    return None

def extract_metrics(row):
    """Extract key metrics from a row (adjusted to match your CSV column names)."""
    return {
        "avg":      get_float(row, ["avg","mean","avg_mean","median_mean"]),
        "auc":      get_float(row, ["auc_theta","auc_theta_norm","auc","auc_theta_mean"]),
        "worst":    get_float(row, ["acc_worst","worst","worst_acc","acc_worst_mean"]),
        "avg_perf": get_float(row, ["avg_perf","perf_per_time","avg_perf_mean"]),
    }

def find_variant(rows, family_key, variant_tag):
    """
    Works with a CSV where column 'family' contains combined names, e.g.:
    'CyResNet56-linear' and 'CyResNet56-log'.

    variant_tag: 'linear' or 'log'
    """
    fam_key = family_key.lower()
    tag = variant_tag.lower()
    for r in rows:
        fam = (r.get("family","") or "").strip().lower()
        # match both the family name and variant in a single 'family' field
        # e.g. 'cyresnet56-linear' contains both 'cyresnet56' and 'linear'
        if fam_key in fam and tag in fam:
            return (r.get("family",""), extract_metrics(r))
    return (None, None)

def format_line(metric, lin_val, log_val):
    """Format a delta line: linear → log with Δ difference."""
    if lin_val is None or log_val is None:
        return f"{metric}: n/a"
    delta = log_val - lin_val
    return f"{metric} {lin_val:.3f} → {log_val:.3f} (Δ {delta:+.3f})"

def plot_delta_bars(out_png, deltas_dict, title=None):
    """Create a bar plot showing deltas (log − linear) for metrics."""
    keys = ["avg","auc","worst","avg_perf"]
    labels = ["avg","AUC_theta","worst","avg_perf"]
    vals = [deltas_dict.get(k, 0.0) for k in keys]

    x = list(range(len(labels)))
    plt.figure()
    plt.bar(x, vals)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Δ (log − linear)")
    if title:
        plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved:", out_png)

# --- main ------------------------------------------------------------------

def main():
    if len(sys.argv) < 5:
        print("Usage:")
        print("  python scripts/plot_delta_log_minus_linear.py "
              "<summary_csv> <dataset> <family_key> <out_png>")
        print("Example:")
        print("  python scripts/plot_delta_log_minus_linear.py "
              "results/exports_family/MNIST/micro/family_summary_MNIST_micro.csv "
              "MNIST CyResNet56 results/fig/delta_log_minus_linear_MNIST_CyResNet56.png")
        sys.exit(1)

    summary_csv = sys.argv[1]
    dataset     = sys.argv[2]
    family_key  = sys.argv[3]
    out_png     = sys.argv[4]

    rows = read_rows(summary_csv)

    # match entries like 'CyResNet56-linear' / 'CyResNet56-log' in the 'family' column
    name_lin, m_lin = find_variant(rows, family_key, "linear")
    name_log, m_log = find_variant(rows, family_key, "log")

    if m_lin is None or m_log is None:
        print(f"Could not find linear/log pair for '{family_key}'. "
              f"Check names in {summary_csv}")
        # for convenience — print available 'family' values
        fam_vals = sorted({(r.get('family') or '') for r in rows})
        print("Available 'family' values:", fam_vals)
        sys.exit(2)

    # compute deltas (log - linear)
    deltas = {}
    for k in ["avg","auc","worst","avg_perf"]:
        v_lin = m_lin.get(k)
        v_log = m_log.get(k)
        deltas[k] = (v_log - v_lin) if (v_lin is not None and v_log is not None) else 0.0

    # print summary lines for thesis/report
    print(f"[{dataset}] {family_key} — log vs linear:")
    print(format_line("avg",       m_lin.get("avg"),      m_log.get("avg")))
    print(format_line("AUC_theta", m_lin.get("auc"),      m_log.get("auc")))
    print(format_line("worst",     m_lin.get("worst"),    m_log.get("worst")))
    print(format_line("avg_perf",  m_lin.get("avg_perf"), m_log.get("avg_perf")))

    # plot
    ttl = f"{dataset}: Δ(log−linear) — {family_key}"
    plot_delta_bars(out_png, deltas, title=ttl)

if __name__ == "__main__":
    main()
