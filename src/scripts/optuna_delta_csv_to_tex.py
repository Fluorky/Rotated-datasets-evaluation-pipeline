#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → Markdown/LaTeX tables + optional 'quick lines'.

Examples
--------
# Markdown (as before)
python scripts/csv_to_md.py --out tables.md --precision 3

# LaTeX: longtable with wrapping and compact headers
python scripts/csv_to_md.py --format latex --out tables.tex --precision 3

# Extra controls
python scripts/csv_to_md.py --format latex --truncate-run 60
python scripts/csv_to_md.py --format latex --drop-run
python scripts/csv_to_md.py --format latex --no-compact
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re
import pandas as pd

# -------- quick lines config (as before) ------------------------------------

QUICK_PAIRS = {
    "GTSRB":     [("resnet56", "linearpolar")],
    "GTSRB_RGB": [("cyresnet56", "logpolar")],
    "LEGO":      [("cyvgg19", "logpolar")],
    "MNIST":     [("resnet56", "linearpolar")],
}
ARCH_LABEL = {"resnet56":"ResNet","cyresnet56":"CyResNet","vgg19":"VGG","cyvgg19":"CyVGG"}
ACT_LABEL  = {"linearpolar":"linear","logpolar":"log"}

# -------- paths & helpers ----------------------------------------------------

def find_csv_dir(start_file: Path) -> Path:
    """Search for 'media/csvs' walking upward from the script path (macOS/Linux/Windows)."""
    here = start_file.resolve().parent
    for anc in [here] + list(here.parents):
        cand1 = anc / "media" / "csvs"
        cand2 = anc / "MasterThesis" / "media" / "csvs"
        if cand1.is_dir(): return cand1
        if cand2.is_dir(): return cand2
    raise FileNotFoundError("Could not locate 'media/csvs' upward from the script path.")

def dataset_from_stem(stem: str) -> str | None:
    """Guess dataset name from a CSV filename stem."""
    if "GTSRB_RGB" in stem: return "GTSRB_RGB"
    if "GTSRB" in stem and "RGB" not in stem: return "GTSRB"
    if "LEGO" in stem: return "LEGO"
    if "MNIST" in stem: return "MNIST"
    return None

def fmt(x, p: int):
    """Format numbers: floats to p decimals, everything else as str."""
    try:
        if isinstance(x, float): return f"{x:.{p}f}"
        return str(x)
    except Exception:
        return str(x)

# -------- label utilities ----------------------------------------------------

_LATEX_SPECIALS = re.compile(r'([#$%&{}~^\\])')

def _latex_escape(s: str) -> str:
    """Escape LaTeX special chars; keep underscores as \_ and insert allowbreaks for wrapping."""
    s = _LATEX_SPECIALS.sub(r'\\\1', s)
    # escape underscore and add allowbreak after separators to enable wrapping
    s = s.replace('_', r'\_\allowbreak{}').replace('-', r'\-\allowbreak{}')
    return s

def shorten_headers(cols: list[str]) -> list[str]:
    """Compact, human-friendly headers for PDF width."""
    mapping = {
        "avg_base":"avg(b)","avg_opt":"avg(o)","d_avg":"Δavg",
        "AUC_base":"AUC(b)","AUC_opt":"AUC(o)","d_AUC":"ΔAUC",
        "worst_base":"worst(b)","worst_opt":"worst(o)","d_worst":"Δworst",
        "arch":"arch","act":"act","run":"run",
        # fallback casings
        "avg":"avg","auc_theta":"AUCθ","worst":"worst",
    }
    return [mapping.get(c, c) for c in cols]

def default_col_order(cols: list[str]) -> list[str]:
    """Desired column order for known CSV shapes."""
    preferred = [
        "run","arch","act",
        "avg_base","avg_opt","d_avg",
        "AUC_base","AUC_opt","d_AUC",
        "worst_base","worst_opt","d_worst",
    ]
    # keep only those present, append any extras at the end
    ordered = [c for c in preferred if c in cols]
    extras  = [c for c in cols if c not in ordered]
    return ordered + extras

# -------- table renderers ----------------------------------------------------

def df_to_markdown(df: pd.DataFrame, precision: int = 3) -> str:
    """Minimal, Pandoc-friendly Markdown table."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = [fmt(row[c], precision) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def df_to_latex_longtable(
    df: pd.DataFrame,
    precision: int = 3,
    truncate_run: int | None = 80,
    compact_headers: bool = True,
    drop_run: bool = False,
) -> str:
    """Render a LaTeX longtable with wrapping in the first (run) column."""
    dfx = df.copy()

    # reorder + rename headers
    dfx = dfx[default_col_order(list(dfx.columns))]
    if drop_run and "run" in dfx.columns:
        dfx = dfx.drop(columns=["run"])

    headers = list(dfx.columns)
    if compact_headers:
        headers = shorten_headers(headers)

    # truncate very long run names (preserve suffix like '.log')
    if "run" in dfx.columns and not drop_run and truncate_run:
        def _trunc(s: str) -> str:
            s = str(s)
            if len(s) <= truncate_run: return s
            # keep extension if present
            if '.' in s:
                base, dot, ext = s.rpartition('.')
                keep = truncate_run - len(ext) - 5  # room for "…." and ext
                return (base[:max(0, keep)] + "…." + ext) if keep > 0 else ("…" + ext)
            return s[:max(0, truncate_run-1)] + "…"
        dfx["run"] = dfx["run"].map(_trunc)

    # LaTeX escaping + allowbreaks
    for c in dfx.columns:
        if dfx[c].dtype == object:
            dfx[c] = dfx[c].map(lambda v: _latex_escape(str(v)))
        else:
            dfx[c] = dfx[c].map(lambda v: fmt(v, precision))

    # Column specs: first (run) as p{.46\textwidth} to wrap, others centered
    n = dfx.shape[1]
    if not drop_run and "run" in dfx.columns:
        specs = [r"p{.46\textwidth}"] + ["c"]*(n-1)
    else:
        specs = ["c"]*n

    # Build longtable
    head_line = " & ".join(headers) + r" \\"
    lines = []
    lines.append(r"\begingroup")
    lines.append(r"\setlength{\tabcolsep}{4pt}% tighter columns")
    lines.append(r"\renewcommand{\arraystretch}{1.05}% a bit taller rows")
    lines.append(r"\small")
    lines.append(r"\begin{longtable}{" + " ".join(specs) + r"}")
    lines.append(r"\toprule")
    lines.append(head_line)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(head_line)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")

    for _, row in dfx.iterrows():
        row_cells = [str(row[c]) for c in dfx.columns]
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)

# -------- quick lines (still available for MD/LaTeX) ------------------------

def quick_line_block(dataset: str, df: pd.DataFrame, precision: int = 3, latex: bool = False) -> str:
    """Emit a compact per-dataset summary line block for selected (arch, act) pairs."""
    pairs = QUICK_PAIRS.get(dataset, [])
    if not pairs:
        return ""
    out = []
    for arch, act in pairs:
        sub = df[(df["arch"] == arch) & (df["act"] == act)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        # accept both naming variants
        def pick(*names):
            for n in names:
                if n in r: return r[n]
            return None
        a_b = pick("avg_base");   a_o = pick("avg_opt");    d_avg = pick("d_avg")
        u_b = pick("AUC_base");   u_o = pick("AUC_opt");    d_auc = pick("d_AUC")
        w_b = pick("worst_base"); w_o = pick("worst_opt");  d_wst = pick("d_worst")

        label_arch = ARCH_LABEL.get(arch, arch)
        label_act  = ACT_LABEL.get(act, act)

        if latex:
            mk = lambda x: fmt(float(x), precision) if x is not None else "--"
            block = []
            block.append(r"\begin{quote}")
            block.append(rf"\textbf{{{dataset} - {label_arch}-{label_act}}}\\")
            block.append(rf"\texttt{{avg}} {mk(a_b)} $\rightarrow$ {mk(a_o)} (\textbf{{$\Delta$avg {mk(d_avg)}}})\\")
            block.append(rf"\texttt{{AUC}} {mk(u_b)} $\rightarrow$ {mk(u_o)} (\textbf{{$\Delta$AUC {mk(d_auc)}}})\\")
            block.append(rf"\texttt{{worst}} {mk(w_b)} $\rightarrow$ {mk(w_o)} (\textbf{{$\Delta$worst {mk(d_wst)}}})")
            block.append(r"\end{quote}")
            out.append("\n".join(block))
        else:
            mk = lambda x: fmt(float(x), precision) if x is not None else "--"
            block = []
            block.append(f"> **{dataset} - {label_arch}-{label_act}**  ")
            block.append(f"> `avg {mk(a_b)} → {mk(a_o)}` (**$\\Delta$avg {mk(d_avg)}**)  ")
            block.append(f"> `AUC {mk(u_b)} → {mk(u_o)}` (**$\\Delta$AUC {mk(d_auc)}**)  ")
            block.append(f"> `worst {mk(w_b)} → {mk(w_o)}` (**$\\Delta$worst {mk(d_wst)}**)")
            out.append("\n".join(block))
    return ("\n\n".join(out) + "\n") if out else ""

# -------- main ---------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="CSV → Markdown/LaTeX tables + quick lines")
    ap.add_argument("--out", type=Path, default=None, help="Output file (.md or .tex). Default: stdout")
    ap.add_argument("--precision", type=int, default=3, help="Float precision (default: 3)")
    ap.add_argument("--format", choices=["md","latex"], default="md", help="Output format (default: md)")
    ap.add_argument("--truncate-run", type=int, default=80, help="Max run label length (LaTeX only). Use 0 to disable.")
    ap.add_argument("--drop-run", action="store_true", help="Drop the 'run' column (LaTeX only).")
    ap.add_argument("--no-compact", action="store_true", help="Do not compact headers (LaTeX only).")
    args = ap.parse_args(argv)

    csv_dir = find_csv_dir(Path(__file__))
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        print(f"[ERR] No CSV files found in {csv_dir}", file=sys.stderr)
        sys.exit(1)

    chunks: list[str] = []
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Skipping {csv_path.name}: {e}", file=sys.stderr)
            continue

        stem = csv_path.stem
        ds = dataset_from_stem(stem) or stem

        # section header (comment in LaTeX, heading in MD)
        chunks.append(f"%% {stem}\n" if args.format=="latex" else f"#### {stem}\n")

        if args.format == "latex":
            # reorder columns for readability if known shape
            want_order = default_col_order(list(df.columns))
            df = df[want_order]
            table_tex = df_to_latex_longtable(
                df,
                precision=args.precision,
                truncate_run=(None if args.truncate_run<=0 else args.truncate_run),
                compact_headers=not args.no_compact,
                drop_run=args.drop_run,
            )
            chunks.append(table_tex)
            chunks.append("")  # blank line
            chunks.append(quick_line_block(ds, df, precision=args.precision, latex=True))
        else:
            # Markdown
            chunks.append(df_to_markdown(df, precision=args.precision))
            chunks.append("")
            chunks.append(quick_line_block(ds, df, precision=args.precision, latex=False))

    output = "\n".join(chunks).rstrip() + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"[OK] Wrote to {args.out}")
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
