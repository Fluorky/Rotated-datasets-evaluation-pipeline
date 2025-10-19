# src/analysis/best_models.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ------------------------- helpers -------------------------

def _shorten_label(s: str) -> str:
    """Trim long prefixes like 'arch-activation_...' from training labels."""
    return re.sub(r'^.+?-[^-]+-[^_]+_', '', s)

def _category(name: str) -> str:
    """Bucket a test column into dataset / merged / rotated / other."""
    n = name.lower()
    if n.startswith("dataset"):
        return "dataset"
    if n.startswith("merged"):
        return "merged"
    if n.startswith("rotated"):
        return "rotated"
    return "other"

def _load_group_csvs(results_root: Path, dataset_name: str) -> Dict[str, pd.DataFrame]:
    """
    Read all accuracy matrices created by learning_matrix.py:
      <results_root>/<dataset>/accuracy_matrix_<group_key>.csv
    Return: { group_key -> DataFrame }
    """
    base = Path(results_root) / dataset_name
    out: Dict[str, pd.DataFrame] = {}
    for csv in sorted(base.glob("accuracy_matrix_*.csv")):
        group_key = csv.stem.replace("accuracy_matrix_", "")
        try:
            df = pd.read_csv(csv, index_col=0)
        except Exception:
            continue
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        out[group_key] = df
    return out

def _to_long(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """Matrix -> long format: columns [group, train_label, test_case, acc]."""
    if df.empty:
        return pd.DataFrame(columns=["group", "train_label", "test_case", "acc"])
    long = df.reset_index().melt(id_vars=df.index.name or "index",
                                 var_name="test_case", value_name="acc")
    long.rename(columns={df.index.name or "index": "train_label"}, inplace=True)
    long["group"] = group_key
    return long[["group", "train_label", "test_case", "acc"]]


# ------------------------- main API -------------------------

def summarize_best_models(dataset_name: str, results_root: Path, out_dir: Optional[Path] = None) -> None:
    """
    Rank models for a dataset using the accuracy matrices already generated.

    Produces three CSVs (and prints handy Top-5 to console):
    - winners_by_test_case.csv   → for each test column, the single best (model, train) and its accuracy
    - overall_top_models.csv     → averages across ALL columns per (model=arch+transform, train_label-short)
    - overall_top_by_model.csv   → averages across ALL columns per model only (arch+transform)
    - category_top_models.csv    → averages per category (dataset / merged / rotated / other) per (model, train)

    Parameters
    ----------
    dataset_name : str
        The dataset folder name used by learning_matrix.py (e.g., "GTSRB_RGB").
    results_root : Path
        The parent folder containing <dataset>/accuracy_matrix_*.csv.
    out_dir : Optional[Path]
        Where to write the summary CSVs. Defaults to <results_root>/<dataset>/summary.
    """
    group_dfs = _load_group_csvs(results_root, dataset_name)
    if not group_dfs:
        print(f"❌ No accuracy_matrix_*.csv found under: {results_root}/{dataset_name}")
        return

    # Concatenate all groups into one long dataframe
    long_parts: List[pd.DataFrame] = []
    for gk, df in group_dfs.items():
        long_parts.append(_to_long(df, gk))
    all_long = pd.concat(long_parts, ignore_index=True)

    all_long["acc"] = pd.to_numeric(all_long["acc"], errors="coerce")
    all_long.dropna(subset=["acc"], inplace=True)

    # Convenience columns
    all_long["model"] = all_long["group"]                # arch + transform
    all_long["train_short"] = all_long["train_label"].map(_shorten_label)
    all_long["category"] = all_long["test_case"].map(_category)

    # 1) Winner per test column
    winners = (
        all_long.sort_values(["test_case", "acc", "model", "train_short"],
                             ascending=[True, False, True, True])
                .groupby("test_case", as_index=False)
                .first()
                .loc[:, ["test_case", "acc", "model", "train_label", "train_short"]]
                .rename(columns={"acc": "best_acc",
                                 "model": "best_model",
                                 "train_label": "best_train_label"})
    )

    # 2) Overall average by (model, train_short)
    overall_by_run = (
        all_long.groupby(["model", "train_short"], as_index=False)["acc"]
                .mean()
                .rename(columns={"acc": "mean_acc"})
                .sort_values("mean_acc", ascending=False)
    )

    # 3) Overall average by model only (ignore train label)
    overall_by_model = (
        all_long.groupby(["model"], as_index=False)["acc"]
                .mean()
                .rename(columns={"acc": "mean_acc"})
                .sort_values("mean_acc", ascending=False)
    )

    # 4) Category-wise averages per (model, train_short)
    cat_frames: List[pd.DataFrame] = []
    for cat in ["dataset", "merged", "rotated", "other"]:
        sub = all_long[all_long["category"] == cat]
        if sub.empty:
            continue
        agg = (sub.groupby(["model", "train_short"], as_index=False)["acc"]
                    .mean()
                    .rename(columns={"acc": f"mean_acc_{cat}"}))
        agg["category"] = cat
        cat_frames.append(agg.sort_values(f"mean_acc_{cat}", ascending=False))
    cat_summary = pd.concat(cat_frames, ignore_index=True) if cat_frames else pd.DataFrame()

    # Output paths
    if out_dir is None:
        out_dir = Path(results_root) / dataset_name / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    winners_csv = out_dir / "winners_by_test_case.csv"
    overall_runs_csv = out_dir / "overall_top_models.csv"
    overall_models_csv = out_dir / "overall_top_by_model.csv"
    cat_csv = out_dir / "category_top_models.csv"

    winners.to_csv(winners_csv, index=False, float_format="%.4f")
    overall_by_run.to_csv(overall_runs_csv, index=False, float_format="%.4f")
    overall_by_model.to_csv(overall_models_csv, index=False, float_format="%.4f")
    if not cat_summary.empty:
        cat_summary.to_csv(cat_csv, index=False, float_format="%.4f")

    print(f"✅ Saved: {winners_csv}")
    print(f"✅ Saved: {overall_runs_csv}")
    print(f"✅ Saved: {overall_models_csv}")
    if not cat_summary.empty:
        print(f"✅ Saved: {cat_csv}")

    # Console Top-5 quick view
    def top5(df: pd.DataFrame, col: str) -> pd.Series:
        return df.set_index(list(df.columns[:-1]))[col].sort_values(ascending=False).head(5)

    print("\n🏆 Overall Top-5 (by model only):")
    print(overall_by_model.set_index("model")["mean_acc"].head(5).to_string())

    print("\n🏆 Overall Top-5 (by model + train label):")
    print(overall_by_run.set_index(["model", "train_short"])["mean_acc"].head(5).to_string())

    for cat in ["dataset", "merged", "rotated"]:
        part = cat_summary[cat_summary["category"] == cat]
        if not part.empty:
            col = f"mean_acc_{cat}"
            print(f"\n🏆 Top-5 in category '{cat}':")
            print(part.set_index(["model", "train_short"])[col].head(5).to_string())
