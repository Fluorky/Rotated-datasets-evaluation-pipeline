# scripts/family_leaders_cy_vs_cnn.py
import sys
import csv
import os
from pathlib import Path
from typing import List, Dict, Optional


def read_family_summary(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def to_float(row: Dict[str, str], col: str) -> Optional[float]:
    """Safely convert a value from a given column to float."""
    v = (row.get(col) or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def split_cy_vs_cnn(rows: List[Dict[str, str]]):
    """
    Split rows into two groups:
    - Cy* models (cyclic CNNs)
    - regular CNN models
    """
    cy, cnn = [], []
    for r in rows:
        fam = (r.get("family") or "").strip()
        if not fam:
            continue
        # "Cy..." = cyclic models; others = standard CNNs
        if fam.lower().startswith("cy"):
            cy.append(r)
        else:
            cnn.append(r)
    return cy, cnn


def best_by_metric(rows: List[Dict[str, str]], metric_col: str) -> Optional[Dict[str, str]]:
    """Return the row with the highest value in the given metric column."""
    best_row = None
    best_val = None
    for r in rows:
        v = to_float(r, metric_col)
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_row = r
    return best_row


def process_dataset(exports_root: Path, dataset: str) -> List:
    """
    Returns one row for the output table:
    [dataset,
     cy_avg_family,  cy_avg,  cnn_avg_family,  cnn_avg,  d_avg,
     cy_auc_family,  cy_auc,  cnn_auc_family,  cnn_auc,  d_auc,
     cy_pt_family,   cy_pt,   cnn_pt_family,   cnn_pt,   d_pt]
    """
    summary_path = (
        exports_root
        / dataset
        / "micro"
        / f"family_summary_{dataset}_micro.csv"
    )

    if not summary_path.exists():
        print(f"WARNING: missing {summary_path}")
        return [dataset] + [""] * 15

    rows = read_family_summary(summary_path)
    cy_rows, cnn_rows = split_cy_vs_cnn(rows)

    if not cy_rows or not cnn_rows:
        print(f"WARNING: dataset={dataset} has cy={len(cy_rows)} cnn={len(cnn_rows)}")
        return [dataset] + [""] * 15

    # Column names as used in matrix_analyzer/family_aggregator
    col_avg = "avg_mean"
    col_auc = "auc_theta_mean"
    col_pt  = "avg_perf_mean"

    cy_avg_row  = best_by_metric(cy_rows,  col_avg)
    cnn_avg_row = best_by_metric(cnn_rows, col_avg)

    cy_auc_row  = best_by_metric(cy_rows,  col_auc)
    cnn_auc_row = best_by_metric(cnn_rows, col_auc)

    cy_pt_row   = best_by_metric(cy_rows,  col_pt)
    cnn_pt_row  = best_by_metric(cnn_rows, col_pt)

    def fam(row: Optional[Dict[str, str]]) -> str:
        return (row.get("family") or "") if row else ""

    def val(row: Optional[Dict[str, str]], col: str) -> str:
        v = to_float(row or {}, col)
        return f"{v:.6f}" if v is not None else ""

    # Numeric values for differences (kept as float)
    cy_avg_val  = to_float(cy_avg_row,  col_avg) if cy_avg_row  else None
    cnn_avg_val = to_float(cnn_avg_row, col_avg) if cnn_avg_row else None
    cy_auc_val  = to_float(cy_auc_row,  col_auc) if cy_auc_row  else None
    cnn_auc_val = to_float(cnn_auc_row, col_auc) if cnn_auc_row else None
    cy_pt_val   = to_float(cy_pt_row,   col_pt)  if cy_pt_row   else None
    cnn_pt_val  = to_float(cnn_pt_row,  col_pt)  if cnn_pt_row  else None

    d_avg = (cy_avg_val - cnn_avg_val) if (cy_avg_val is not None and cnn_avg_val is not None) else ""
    d_auc = (cy_auc_val - cnn_auc_val) if (cy_auc_val is not None and cnn_auc_val is not None) else ""
    d_pt  = (cy_pt_val  - cnn_pt_val)  if (cy_pt_val  is not None and cnn_pt_val  is not None) else ""

    return [
        dataset,
        fam(cy_avg_row),  val(cy_avg_row,  col_avg),
        fam(cnn_avg_row), val(cnn_avg_row, col_avg),
        f"{d_avg:.6f}" if d_avg != "" else "",
        fam(cy_auc_row),  val(cy_auc_row,  col_auc),
        fam(cnn_auc_row), val(cnn_auc_row, col_auc),
        f"{d_auc:.6f}" if d_auc != "" else "",
        fam(cy_pt_row),   val(cy_pt_row,   col_pt),
        fam(cnn_pt_row),  val(cnn_pt_row,  col_pt),
        f"{d_pt:.6f}" if d_pt != "" else "",
    ]


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python scripts/family_leaders_cy_vs_cnn.py "
              "<exports_root> <out_csv> <DATASET> [<DATASET2> ...]")
        print()
        print("Example:")
        print("  python scripts/family_leaders_cy_vs_cnn.py "
              "results/exports_family results/fig/cy_vs_cnn_leaders_micro.csv "
              "MNIST GTSRB GTSRB_RGB LEGO")
        sys.exit(1)

    exports_root = Path(sys.argv[1])
    out_csv      = Path(sys.argv[2])
    datasets     = sys.argv[3:]

    os.makedirs(out_csv.parent, exist_ok=True)

    header = [
        "dataset",
        "cy_avg_family",  "cy_avg",
        "cnn_avg_family", "cnn_avg",
        "d_avg",
        "cy_auc_family",  "cy_auc",
        "cnn_auc_family", "cnn_auc",
        "d_auc",
        "cy_pt_family",   "cy_avg_perf",
        "cnn_pt_family",  "cnn_avg_perf",
        "d_avg_perf",
    ]

    rows_out = []
    for ds in datasets:
        row = process_dataset(exports_root, ds)
        rows_out.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows_out:
            w.writerow(r)

    print(f"Saved: {out_csv}  rows={len(rows_out)}")


if __name__ == "__main__":
    main()
