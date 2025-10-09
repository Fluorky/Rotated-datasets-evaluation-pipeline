#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Markdown tables from all CSVs in media/csvs, with portable paths.

Usage:
  python csv_to_md.py                # print combined Markdown to stdout
  python csv_to_md.py --out tables.md  # write combined Markdown to a file
  python csv_to_md.py --precision 3    # control float precision (default 3)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

def find_csv_dir(start_file: Path) -> Path:
    """
    Walk upwards from the script location and return the first existing
    media/csvs directory. Supports both layouts:
      <root>/media/csvs
      <root>/MasterThesis/media/csvs
    """
    start = start_file.resolve().parent
    for ancestor in [start] + list(start.parents):
        cand1 = ancestor / "media" / "csvs"
        cand2 = ancestor / "MasterThesis" / "media" / "csvs"
        if cand1.is_dir():
            return cand1
        if cand2.is_dir():
            return cand2
    raise FileNotFoundError("Could not locate 'media/csvs' upward from script path.")

def format_value(val, precision: int):
    # render floats with fixed precision; other types as str
    try:
        if isinstance(val, float):
            return f"{val:.{precision}f}"
        return str(val)
    except Exception:
        return str(val)

def df_to_markdown(df: pd.DataFrame, precision: int = 3) -> str:
    """
    Minimal, Pandoc-friendly Markdown table (no external deps).
    """
    # ensure column order is stable
    cols = list(df.columns)
    # header
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "|" + "|".join("---" for _ in cols) + "|"
    # rows
    lines = [header, separator]
    for _, row in df.iterrows():
        cells = [format_value(row[c], precision) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def main(argv=None):
    parser = argparse.ArgumentParser(description="CSV → Markdown tables")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output .md file (if omitted, prints to stdout)")
    parser.add_argument("--precision", type=int, default=3,
                        help="Float precision (default: 3)")
    args = parser.parse_args(argv)

    # Locate media/csvs relative to this script
    script_path = Path(__file__).resolve()
    csv_dir = find_csv_dir(script_path)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}", file=sys.stderr)
        sys.exit(1)

    chunks = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Skipping {csv_path.name}: {e}", file=sys.stderr)
            continue
        title = f"#### {csv_path.stem}"
        table = df_to_markdown(df, precision=args.precision)
        chunks.append(f"{title}\n\n{table}\n")

    output = "\n".join(chunks).rstrip() + "\n"

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"[OK] Wrote Markdown tables to {args.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    main()
