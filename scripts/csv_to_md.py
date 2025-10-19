#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV -> Markdown tables + optional 'quick lines' with compact layout.

Why this fixes your PDF:
- Drops the ultra-wide `run` column by default (--compact), keeping just (arch, act).
- Shortens and wraps long tokens (underscores/hyphens) so LaTeX can break lines.
- Lets you split a wide table into two: metrics and deltas (--split).
- Rounds floats and renames headers to short labels.

Usage examples:
  python MasterThesis/scripts/csv_to_md.py --out tables.md --precision 3 --compact --wrap-cells latex
  python MasterThesis/scripts/csv_to_md.py --out tables.md --split
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re
import pandas as pd

# ---------- quick-lines config (optional) ----------

QUICK_PAIRS = {
    "GTSRB":     [("resnet56",   "linearpolar")],
    "GTSRB_RGB": [("cyresnet56", "logpolar")],
    "LEGO":      [("cyvgg19",    "logpolar")],
    "MNIST":     [("resnet56",   "linearpolar")],
}

ARCH_LABEL = {"resnet56": "ResNet", "cyresnet56": "CyResNet", "vgg19": "VGG", "cyvgg19": "CyVGG"}
ACT_LABEL  = {"linearpolar": "linear", "logpolar": "log"}

# ---------- helpers ----------

RUN_PREFIX_RE = re.compile(r'(?i)^custom_dataset_[a-z0-9_]+_non_rotated_')
def shorten_run(s: str) -> str:
    s = RUN_PREFIX_RE.sub("", s)
    return s.removesuffix(".log")

def find_csv_dir(start_file: Path) -> Path:
    here = start_file.resolve().parent
    for anc in [here] + list(here.parents):
        cand1 = anc / "media" / "csvs"
        cand2 = anc / "MasterThesis" / "media" / "csvs"
        if cand1.is_dir(): return cand1
        if cand2.is_dir(): return cand2
    raise FileNotFoundError("Could not locate 'media/csvs' upward from script path.")

def dataset_from_stem(stem: str) -> str | None:
    if "GTSRB_RGB" in stem: return "GTSRB_RGB"
    if "GTSRB" in stem and "RGB" not in stem: return "GTSRB"
    if "LEGO" in stem: return "LEGO"
    if "MNIST" in stem: return "MNIST"
    return None

def fmt(x, p: int):
    try:
        if isinstance(x, float):
            return f"{x:.{p}f}"
        return str(x)
    except Exception:
        return str(x)

def wrap_cell(s: str, mode: str) -> str:
    """Insert soft wrap opportunities for long tokens."""
    if not isinstance(s, str):
        return s
    if mode == "none":
        return s
    if mode == "zwsp":
        # zero-width space after '_' and '-'
        return s.replace("_", "_\u200b").replace("-", "-\u200b")
    if mode == "latex":
        # escape '_' for LaTeX and allow breaking after it / hyphens
        # works with pandoc --from=markdown+raw_tex
        s = s.replace("_", r"\_\allowbreak{}")
        s = s.replace("-", r"-\allowbreak{}")
        return s
    return s

def df_to_markdown(df: pd.DataFrame, precision: int = 3) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [fmt(row[c], precision) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def quick_line_block(dataset: str, df: pd.DataFrame, precision: int = 3) -> str:
    pairs = QUICK_PAIRS.get(dataset, [])
    if not pairs: return ""
    out = []
    for arch, act in pairs:
        sub = df[(df.get("arch") == arch) & (df.get("act") == act)]
        if sub.empty: continue
        r = sub.iloc[0]
        # accept either *_opt or *_optuna naming
        a_b = fmt(float(r.get("avg_base",  r.get("base_avg"))), precision)
        a_o = fmt(float(r.get("avg_opt",   r.get("opt_avg",  r.get("avg")))), precision)
        u_b = fmt(float(r.get("AUC_base",  r.get("base_auc"))), precision)
        u_o = fmt(float(r.get("AUC_opt",   r.get("opt_auc",  r.get("AUC_theta")))), precision)
        w_b = fmt(float(r.get("worst_base",r.get("base_worst"))), precision)
        w_o = fmt(float(r.get("worst_opt", r.get("opt_worst", r.get("worst")))), precision)
        d_avg = fmt(float(r.get("d_avg",   r.get("Δavg",   0.0))), precision)
        d_auc = fmt(float(r.get("d_AUC",   r.get("ΔAUC",   0.0))), precision)
        d_wst = fmt(float(r.get("d_worst", r.get("Δworst", 0.0))), precision)

        label_arch = ARCH_LABEL.get(arch, arch)
        label_act  = ACT_LABEL.get(act, act)

        block = []
        block.append(f"> **{dataset} - {label_arch}-{label_act}**  ")
        block.append(f"> `avg {a_b} → {a_o}` (**$\\Delta$avg {d_avg}**)  ")
        block.append(f"> `AUC {u_b} → {u_o}` (**$\\Delta$AUC {d_auc}**)  ")
        block.append(f"> `worst {w_b} → {w_o}` (**$\\Delta$worst {d_wst}**)")
        out.append("\n".join(block))
    return ("\n\n".join(out) + "\n") if out else ""

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a bunch of possible column variants to a compact, consistent set."""
    ren = {
        # metrics (optuna side)
        "avg": "avg_opt", "opt_avg": "avg_opt",
        "AUC_theta": "AUC_opt", "opt_auc": "AUC_opt", "auc_theta": "AUC_opt",
        "worst": "worst_opt", "opt_worst": "worst_opt",
        # base side
        "base_avg": "avg_base", "avg_base": "avg_base",
        "base_auc": "AUC_base", "auc_base": "AUC_base",
        "base_worst": "worst_base", "worst_base": "worst_base",
        # deltas
        "Δavg": "d_avg", "ΔAUC": "d_AUC", "Δworst": "d_worst",
    }
    for k, v in ren.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    return df

def make_compact(df: pd.DataFrame, wrap_mode: str, precision: int, drop_run: bool = True) -> pd.DataFrame:
    df = normalize_columns(df.copy())

    # shorten/wrap 'run' if kept
    if "run" in df.columns and not drop_run:
        df["run"] = df["run"].astype(str).map(shorten_run).map(lambda s: wrap_cell(s, wrap_mode))

    # prefer compact column order
    cols_pref = [
        "arch","act",
        "avg_opt","AUC_opt","worst_opt",
        "avg_base","AUC_base","worst_base",
        "d_avg","d_AUC","d_worst",
    ]
    present = [c for c in cols_pref if c in df.columns]
    # include 'run' at the end if requested
    if "run" in df.columns and not drop_run:
        present = ["run"] + present
    df = df[present]

    # round floats eagerly (keeps Markdown lean)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(precision)

    return df

# ---------- main ----------

def main(argv=None):
    ap = argparse.ArgumentParser(description="CSV → Markdown tables (compact & PDF-friendly)")
    ap.add_argument("--out", type=Path, default=None, help="Write result to a .md file (default: stdout)")
    ap.add_argument("--precision", type=int, default=3, help="Float precision (default: 3)")
    ap.add_argument("--compact", action="store_true", help="Use compact column set and drop the long 'run' column")
    ap.add_argument("--keep-run", action="store_true", help="Keep 'run' column (overrides --compact dropping)")
    ap.add_argument("--wrap-cells", choices=["none","zwsp","latex"], default="zwsp",
                    help="Insert soft-wrap hints inside long tokens (default: zwsp)")
    ap.add_argument("--split", action="store_true",
                    help="Split each table into two: (metrics) and (deltas) to reduce width")
    args = ap.parse_args(argv)

    csv_dir = find_csv_dir(Path(__file__))
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        print(f"[ERR] No CSV files found in {csv_dir}", file=sys.stderr)
        sys.exit(1)

    chunks: list[str] = []
    for csv_path in files:
        try:
            raw = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Skipping {csv_path.name}: {e}", file=sys.stderr)
            continue

        df = raw.copy()
        # optional compacting
        if args.compact:
            df = make_compact(df, wrap_mode=args.wrap_cells, precision=args.precision, drop_run=not args.keep_run)
        else:
            # still wrap long string-like columns to avoid overflow
            for c in df.columns:
                if pd.api.types.is_object_dtype(df[c]):
                    df[c] = df[c].astype(str).map(lambda s: wrap_cell(s, args.wrap_cells))
            # also shorten run if present
            if "run" in df.columns:
                df["run"] = df["run"].astype(str).map(shorten_run)

        stem = csv_path.stem
        ds = dataset_from_stem(stem) or stem

        chunks.append(f"#### {stem}\n")

        if args.split:
            left_cols  = [c for c in ["arch","act","avg_opt","AUC_opt","worst_opt","avg_base","AUC_base","worst_base"] if c in df.columns]
            right_cols = [c for c in ["arch","act","d_avg","d_AUC","d_worst"] if c in df.columns]
            if "run" in df.columns and args.keep_run:
                # keep run only in the first table
                left_cols  = (["run"] + left_cols) if "run" in df.columns else left_cols
            if left_cols:
                chunks.append(df_to_markdown(df[left_cols], precision=args.precision))
                chunks.append("")
            if right_cols:
                chunks.append(df_to_markdown(df[right_cols], precision=args.precision))
                chunks.append("")
        else:
            chunks.append(df_to_markdown(df, precision=args.precision))
            chunks.append("")

        # quick lines (optional, if columns match)
        try:
            ql = quick_line_block(ds, df, precision=args.precision)
            if ql:
                chunks.append(ql)
        except Exception:
            pass

    output = "\n".join(chunks).rstrip() + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"[OK] Wrote Markdown to {args.out}")
    else:
        sys.stdout.write(output)

if __name__ == "__main__":
    main()
