import csv, os, sys
import matplotlib.pyplot as plt

def read_curve(filepath):
    """
    Return (xs, ys) from a CSV in one of two layouts:
    - wide:   columns = ["model","d0","d15",...,"d180"], a single data row
    - long:   columns include ["delta_deg"/"delta", "accuracy"]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    header = [h.strip().lower() for h in reader[0]]

    # Long format?
    if ("delta_deg" in header or "delta" in header) and "accuracy" in header:
        di = header.index("delta_deg") if "delta_deg" in header else header.index("delta")
        ai = header.index("accuracy")
        xs, ys = [], []
        for row in reader[1:]:
            if not row or len(row) <= ai:
                continue
            try:
                d = float(row[di])
                a = float(row[ai])
            except:
                continue
            xs.append(d); ys.append(a)
        # sort by Δθ
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs  = [p[0] for p in pairs]
        ys  = [p[1] for p in pairs]
        return xs, ys

    # Wide format?
    # Look for columns starting with 'd' followed by degrees: d0, d15, ..., d180
    dcols = []
    for h in header:
        if h.startswith("d"):
            try:
                deg = int(h[1:])
                if 0 <= deg <= 180:
                    dcols.append((deg, h))
            except:
                pass
    dcols = sorted(dcols, key=lambda t: t[0])
    if dcols and len(reader) >= 2:
        data = reader[1]  # first data row
        xs = [deg for deg, _ in dcols]
        ys = []
        # map column name -> index
        idx = {header[i]: i for i in range(len(header))}
        for _, col in dcols:
            val = data[idx[col]]
            try:
                ys.append(float(val))
            except:
                ys.append(None)
        # simple forward-fill for missing values
        last = None
        for i, v in enumerate(ys):
            if v is None and last is not None:
                ys[i] = last
            elif v is not None:
                last = v
        if ys[0] is None:
            first_known = next((v for v in ys if v is not None), 0.0)
            ys = [first_known if v is None else v for v in ys]
        return xs, ys

    raise ValueError(f"Unrecognized CSV shape: {filepath}")

def plot_dual(csv_a, label_a, csv_b, label_b, title, out_png):
    xs_a, ys_a = read_curve(csv_a)
    xs_b, ys_b = read_curve(csv_b)

    plt.figure()
    plt.plot(xs_a, ys_a, label=label_a)
    plt.plot(xs_b, ys_b, label=label_b)
    plt.xlabel("Δθ [deg]")
    plt.ylabel("Acc(Δθ)")
    plt.title(title)
    plt.xlim(0, 180)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out_png}")

if __name__ == "__main__":
    # Example CLI:
    # python scripts/plot_acc_vs_delta_dual.py A.csv "ResNet56-linear" B.csv "CyResNet56-linear" "LEGO — Acc(Δθ)" out.png
    if len(sys.argv) != 7:
        print("Usage: python plot_acc_vs_delta_dual.py <csv_a> <label_a> <csv_b> <label_b> <title> <out_png>")
        sys.exit(1)
    _, ca, la, cb, lb, ti, outp = sys.argv
    plot_dual(ca, la, cb, lb, ti, outp)
