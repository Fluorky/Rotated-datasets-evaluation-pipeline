# -*- coding: utf-8 -*-
"""
Matrix analyzer (per-dataset), with time-aware & rotation-aware rankings.

What it does
------------
1) Reads confusion matrices from <cm_root>/<MODEL>/<TEST_CASE>/confusion_matrix.npy
2) Computes accuracy (micro or macro) per (model, test_case) and inserts into SQLite table 'evaluations'
3) Aggregates training time from 'training_logs' → 'training_runs' (compute_training_times)
4) Prints per-dataset reports:
   - Quality-only ranking (avg, median, min, max, std) + CSV
   - Time-aware ranking: avg_perf (=avg/time), robust_perf (=min/time) + CSV
   - Balanced ranking: score = alpha·norm(avg) + (1-alpha)·norm(avg_perf) + CSV
   - Balanced (per-time only): score = alpha·norm(avg_perf) + (1-alpha)·norm(robust_perf) + CSV
   - Train–test gap (train-like vs OOD) per model
   - Acc(Δθ) curves and AUCθ (theta stability) + CSV per model
5) Mirrors console output to an optional log file (tee).

Dataset scoping (strict)
------------------------
Avoids mixing datasets (e.g., GTSRB vs GTSRB_RGB):
- MNIST      → model name contains 'MNIST'
- GTSRB      → contains 'GTSRB' but NOT 'GTSRB_RGB' or 'GTSRB-RGB'
- GTSRB_RGB  → contains 'GTSRB_RGB' or 'GTSRB-RGB'
- LEGO       → contains 'LEGO'

CSV exports go to: results/exports/<DATASET>/*
"""

from __future__ import annotations

import os
import sys
import re
import csv
import math
import sqlite3
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from statistics import mean, median, stdev
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

# put this right after the existing imports
_CURRENT_METRIC: Optional[str] = None  # optional hint set by CLI or env


def _normalize_metric(metric: Optional[str] = None) -> str:
    """
    Normalize the metric used for storing and querying evaluation rows.

    This keeps micro and macro runs separated in the database and in exports.
    """
    value = (metric or _CURRENT_METRIC or os.environ.get("MATRIX_ANALYZER_METRIC") or "micro").lower().strip()
    if value not in {"micro", "macro"}:
        raise ValueError(f"Unsupported metric: {metric!r}. Expected 'micro' or 'macro'.")
    return value


def _export_root(dataset: str, metric: Optional[str]) -> str:
    """
    Build and create an absolute export directory:
      <cwd>/results/exports/<dataset>/<metric>/
    Falls back to _CURRENT_METRIC or env MATRIX_ANALYZER_METRIC, then 'micro'.
    """
    m = _normalize_metric(metric)
    root = os.path.join(os.getcwd(), "results", "exports", dataset, m)
    _ensure_dir(root)
    print(f"\n📁 Export directory: {root}")
    return root


# =============================== tee / logging ===============================

@contextmanager
def tee_to_file(log_file: Optional[str]):
    """
    Mirror stdout to a file if provided.
    Usage:
        with tee_to_file("path/to.log"):
            print("...")  # goes to console and file
    """
    if not log_file:
        yield
        return
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    class TeeWriter:
        def __init__(self, a, b):
            self.a, self.b = a, b

        def write(self, s):
            self.a.write(s);
            self.b.write(s)

        def flush(self):
            self.a.flush();
            self.b.flush()

    old = sys.stdout
    f = open(log_file, "w", encoding="utf-8")
    sys.stdout = TeeWriter(old, f)
    try:
        yield
    finally:
        sys.stdout = old
        f.close()


def _log_header(dataset: str, alpha: float, metric: str, theta_step: int, top_n: int):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("═" * 70)
    print(f" Run:           {now}")
    print(f" Dataset:       {dataset}")
    print(f" Metric:        {metric}  (micro|macro)")
    print(f" Theta step:    {theta_step}°")
    print(f" Alpha balance: {alpha:.2f}")
    print(f" Top-N:         {top_n}")
    print("═" * 70)


# =============================== utils / io =================================

def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Tuple], header: List[str]):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)


def _safe_load_cm(path: str) -> Optional[np.ndarray]:
    """
    Load and validate a confusion matrix.

    Invalid matrices are skipped with a concrete reason instead of hiding the
    original error behind a generic "failed to load" message.
    """
    try:
        cm = np.load(path)
    except Exception as exc:
        print(f"⚠️ Failed to load confusion matrix {path}: {exc}")
        return None

    if cm.ndim != 2:
        print(f"⚠️ Invalid confusion matrix shape in {path}: expected 2D, got {cm.shape}")
        return None

    if cm.shape[0] != cm.shape[1]:
        print(f"⚠️ Invalid confusion matrix shape in {path}: expected square matrix, got {cm.shape}")
        return None

    if not np.issubdtype(cm.dtype, np.number):
        print(f"⚠️ Invalid confusion matrix dtype in {path}: expected numeric, got {cm.dtype}")
        return None

    if np.any(~np.isfinite(cm)):
        print(f"⚠️ Invalid confusion matrix values in {path}: matrix contains NaN or infinite values")
        return None

    if np.any(cm < 0):
        print(f"⚠️ Invalid confusion matrix values in {path}: matrix contains negative values")
        return None

    return cm


def _upper(s: str) -> str:
    return s.upper().replace("-", "_")


# ============================ dataset scoping ===============================

def _model_belongs_to_dataset(model_name: str, dataset: str) -> bool:
    """
    Strict dataset guard to avoid mixing GTSRB with GTSRB_RGB.
    """
    m = _upper(model_name)
    ds = _upper(dataset)
    if ds == "GTSRB_RGB":
        return ("GTSRB_RGB" in m) or ("GTSRB-RGB" in m)
    if ds == "GTSRB":
        return ("GTSRB" in m) and ("GTSRB_RGB" not in m) and ("GTSRB-RGB" not in m)
    return (ds in m)


# ============================ metrics: micro/macro ===========================

def _micro_acc_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm)) / float(total) if total > 0 else 0.0


def _macro_acc_from_cm(cm: np.ndarray) -> float:
    # Per-class accuracy: diagonal(row) / row sum.
    # Classes with zero support are ignored instead of counted as 0.0.
    diag = np.diag(cm).astype(np.float64)
    rows = cm.sum(axis=1).astype(np.float64)
    valid = rows > 0

    if not np.any(valid):
        return 0.0

    per_class = diag[valid] / rows[valid]
    return float(np.mean(per_class))
# ============================ train-like vs OOD ==============================

def _is_train_like(test_case: str) -> bool:
    s = test_case.lower()
    return ("dataset_" in s and "non_rotated" in s) or ("plus_non_rotated" in s)


# =============================== DB schema ==================================
def _initialize_db(db_path: str) -> sqlite3.Connection:
    """
    Ensure 'evaluations' exists and always contains the newer columns:
    dataset, metric. Works for both fresh and legacy DBs.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Base table (legacy-compatible). If it already exists, this won't alter it.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            test_case TEXT NOT NULL,
            accuracy REAL NOT NULL,
            avg REAL,
            median REAL,
            min REAL,
            max REAL,
            std REAL,
            train_time REAL,
            perf_per_time REAL
        )
    """)
    conn.commit()

    # ---- MIGRATIONS (idempotent) ----
    # Add new columns if they don't exist yet.
    def _safe_alter_add(sql: str):
        try:
            cur.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            # Column probably already exists; ignore.
            pass

    _safe_alter_add("ALTER TABLE evaluations ADD COLUMN dataset TEXT")
    _safe_alter_add("ALTER TABLE evaluations ADD COLUMN metric TEXT")

    return conn


def _safe_alter_add(conn: sqlite3.Connection, sql: str):
    try:
        conn.execute(sql);
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column probably exists


def _ensure_eval_extra_cols(db_path: str):
    conn = sqlite3.connect(db_path)
    _safe_alter_add(conn, "ALTER TABLE evaluations ADD COLUMN dataset TEXT")
    _safe_alter_add(conn, "ALTER TABLE evaluations ADD COLUMN metric TEXT")
    conn.close()


def ensure_indices(db_path: str):
    conn = sqlite3.connect(db_path);
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_model   ON evaluations(model)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_dataset ON evaluations(dataset)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_metric  ON evaluations(metric)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_m_t     ON evaluations(model, test_case)")
    conn.commit();
    conn.close()


def create_training_runs_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS training_runs")
    cur.execute("""
        CREATE TABLE training_runs (
            model TEXT PRIMARY KEY,
            log_file TEXT,
            total_train_time REAL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Created table: training_runs")
    ensure_indices(db_path)


def compute_training_times(db_path: str) -> None:
    """
    Sum 'elapsed_time' from 'training_logs' and insert into 'training_runs'.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='training_logs'
    """)
    if not cur.fetchone():
        print("⚠️ Table 'training_logs' not found. Training times will be NULL.")
        conn.close()
        return

    cur.execute("SELECT log_file, SUM(elapsed_time) FROM training_logs GROUP BY log_file")
    rows = cur.fetchall()

    entries = []
    for log_file, total_time in rows:
        model = log_file[:-10] if isinstance(log_file, str) and log_file.endswith("_train.txt") else log_file
        t = float(total_time or 0.0)
        entries.append((model, log_file, round(t, 2)))

    if entries:
        cur.executemany(
            "INSERT OR REPLACE INTO training_runs (model, log_file, total_train_time) VALUES (?, ?, ?)",
            entries
        )
        conn.commit()
    conn.close()
    print(f"✅ Inserted {len(entries)} rows into training_runs")
    ensure_indices(db_path)


# ========================== ingestion from confusion matrices ===============
def collect_and_store_results(
        cm_root: str,
        db_path: str,
        dataset: Optional[str] = None,
        metric: str = "micro",
        clear_dataset: bool = False,
) -> None:
    """
    Scan confusion matrices and insert enriched rows into 'evaluations'.

    Args:
        cm_root: base dir with <MODEL>/<TEST_CASE>/confusion_matrix.npy
        db_path: SQLite file
        dataset: if provided, accept only models belonging to this dataset
        metric: 'micro' (default) or 'macro'
        clear_dataset: if True, delete previous rows for this dataset and metric

    Notes:
        ``perf_per_time`` is stored per evaluation row as ``accuracy / train_time``.
        Model-level time-aware rankings are computed explicitly later as
        ``mean(accuracy) / train_time`` and ``min(accuracy) / train_time``.
        Since train time is constant per model, averaging ``accuracy / train_time``
        is numerically equivalent for the mean, but keeping the ranking calculation
        explicit makes the intended semantics clearer.
    """
    metric = _normalize_metric(metric)

    _ensure_eval_extra_cols(db_path)
    conn = _initialize_db(db_path)
    cur = conn.cursor()

    # Optional cleanup: wipe only the selected dataset+metric, not all metrics.
    if clear_dataset and dataset:
        cur.execute(
            "DELETE FROM evaluations WHERE dataset = ? AND metric = ?",
            (dataset, metric),
        )
        conn.commit()
        print(f"🧹 Cleared existing rows for dataset={dataset}, metric={metric}")

    if not os.path.isdir(cm_root):
        print(f"❌ Not a directory: {cm_root}")
        conn.close()
        return

    use_macro = metric == "macro"

    raw: List[Tuple[str, str, float]] = []
    skipped_models: List[str] = []

    for model_dir in tqdm(sorted(os.listdir(cm_root)), desc="📁 Scanning models"):
        model_path = os.path.join(cm_root, model_dir)
        if not os.path.isdir(model_path):
            continue

        # dataset guard (avoid e.g. GTSRB vs GTSRB_RGB mixing)
        if dataset and not _model_belongs_to_dataset(model_dir, dataset):
            skipped_models.append(model_dir)
            continue

        for test_subdir in sorted(os.listdir(model_path)):
            cm_file = os.path.join(model_path, test_subdir, "confusion_matrix.npy")
            if not os.path.exists(cm_file):
                continue

            cm = _safe_load_cm(cm_file)
            if cm is None:
                continue

            acc = _macro_acc_from_cm(cm) if use_macro else _micro_acc_from_cm(cm)
            raw.append((model_dir, test_subdir, float(acc)))

    if skipped_models:
        print(f"⚠️ Skipped {len(skipped_models)} model directories not matching dataset={dataset}:")
        for model_name in skipped_models[:20]:
            print(f"   - {model_name}")
        if len(skipped_models) > 20:
            print(f"   ... and {len(skipped_models) - 20} more")

    # Group accuracies by model.
    acc_by_model: Dict[str, List[float]] = defaultdict(list)
    for model, _, accuracy in raw:
        acc_by_model[model].append(accuracy)

    # Training time map.
    cur.execute("SELECT model, total_train_time FROM training_runs")
    time_map = {
        model: (float(total_time) if total_time is not None else None)
        for model, total_time in cur.fetchall()
    }

    # Build enriched rows.
    rows_to_insert = []
    for model, test_case, accuracy in raw:
        accs = acc_by_model[model]
        avg_ = mean(accs)
        med_ = median(accs)
        min_ = min(accs)
        max_ = max(accs)
        std_ = stdev(accs) if len(accs) > 1 else 0.0

        train_time = time_map.get(model)
        perf_per_time = (
            accuracy / train_time
            if train_time is not None and train_time > 0
            else None
        )

        rows_to_insert.append(
            (
                model,
                test_case,
                accuracy,
                avg_,
                med_,
                min_,
                max_,
                std_,
                train_time,
                perf_per_time,
                dataset,
                metric,
            )
        )

    if rows_to_insert:
        cur.executemany(
            """
            INSERT INTO evaluations (
                model, test_case, accuracy,
                avg, median, min, max, std,
                train_time, perf_per_time,
                dataset, metric
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()

    conn.close()
    print(
        f"✅ Saved {len(rows_to_insert)} rows into evaluations "
        f"(dataset={dataset}, metric={metric}, db: {db_path})"
    )


# def collect_and_store_results(
#     cm_root: str,
#     db_path: str,
#     dataset: Optional[str] = None,
#     metric: str = "micro",
#     clear_dataset: bool = False,
# ) -> None:
#     """
#     Scan confusion matrices and insert enriched rows into 'evaluations'.
#
#     Args:
#         cm_root: base dir with <MODEL>/<TEST_CASE>/confusion_matrix.npy
#         db_path: SQLite file
#         dataset: if provided, accept only models belonging to this dataset
#         metric: 'micro' (default) or 'macro'
#         clear_dataset: if True, delete previous rows for this dataset (heuristic via model name)
#     """
#     conn = _initialize_db(db_path)
#     cur  = conn.cursor()
#
#     # optional cleanup
#     if clear_dataset and dataset:
#         cur.execute("SELECT id, model FROM evaluations")
#         rows_all = cur.fetchall()
#         ids_to_del = [rid for (rid, m) in rows_all if _model_belongs_to_dataset(m, dataset)]
#         if ids_to_del:
#             cur.executemany("DELETE FROM evaluations WHERE id = ?", [(i,) for i in ids_to_del])
#             conn.commit()
#             print(f"🧹 Cleared {len(ids_to_del)} existing rows for dataset={dataset}")
#
#     if not os.path.isdir(cm_root):
#         print(f"❌ Not a directory: {cm_root}")
#         conn.close()
#         return
#
#     use_macro = (metric.lower() == "macro")
#
#     raw: List[Tuple[str, str, float]] = []
#     for model_dir in tqdm(sorted(os.listdir(cm_root)), desc="📁 Scanning models"):
#         model_path = os.path.join(cm_root, model_dir)
#         if not os.path.isdir(model_path):
#             continue
#
#         # dataset guard
#         if dataset and not _model_belongs_to_dataset(model_dir, dataset):
#             continue
#
#         for test_subdir in sorted(os.listdir(model_path)):
#             cm_file = os.path.join(model_path, test_subdir, "confusion_matrix.npy")
#             if not os.path.exists(cm_file):
#                 continue
#             cm = _safe_load_cm(cm_file)
#             if cm is None:
#                 print(f"⚠️ Failed to load {cm_file}")
#                 continue
#             acc = _macro_acc_from_cm(cm) if use_macro else _micro_acc_from_cm(cm)
#             raw.append((model_dir, test_subdir, float(acc)))
#
#     # group accuracies by model
#     acc_by_model: Dict[str, List[float]] = defaultdict(list)
#     for m, _, a in raw:
#         acc_by_model[m].append(a)
#
#     # training time map
#     cur.execute("SELECT model, total_train_time FROM training_runs")
#     time_map = {r[0]: (float(r[1]) if r[1] is not None else None) for r in cur.fetchall()}
#
#     # insert enriched rows
#     rows_to_insert = []
#     for model, test_case, acc in raw:
#         accs = acc_by_model[model]
#         avg_ = mean(accs)
#         med_ = median(accs)
#         min_ = min(accs)
#         max_ = max(accs)
#         std_ = stdev(accs) if len(accs) > 1 else 0.0
#
#         ttime = time_map.get(model)
#         perf  = (acc / ttime) if (ttime and ttime > 0) else None
#
#         rows_to_insert.append((
#             model, test_case, acc,
#             avg_, med_, min_, max_, std_,
#             ttime if ttime is not None else None,
#             perf if perf is not None else None
#         ))
#     # after executemany insert:
#     if dataset or metric:
#         conn = sqlite3.connect(db_path);
#         cur = conn.cursor()
#         if dataset:
#             cur.execute("UPDATE evaluations SET dataset = ? WHERE dataset IS NULL", (dataset,))
#         if metric:
#             cur.execute("UPDATE evaluations SET metric = ? WHERE metric IS NULL", (metric,))
#         conn.commit();
#         conn.close()
#
#     if rows_to_insert:
#         cur.executemany("""
#             INSERT INTO evaluations (
#                 model, test_case, accuracy,
#                 avg, median, min, max, std,
#                 train_time, perf_per_time
#             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """, rows_to_insert)
#         conn.commit()
#
#     conn.close()
#     print(f"✅ Saved {len(rows_to_insert)} rows into evaluations (db: {db_path})")


# ========================== queries & helpers (report) ======================

def _table_has_column(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _fetch_eval_rows(
        db_path: str,
        dataset: str,
        metric: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Fetch evaluation rows for exactly one dataset and metric.

    Falls back to the legacy model-name dataset guard if older DB rows do not
    contain dataset/metric metadata.
    """
    metric = _normalize_metric(metric)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    has_dataset = _table_has_column(cur, "evaluations", "dataset")
    has_metric = _table_has_column(cur, "evaluations", "metric")

    if has_dataset and has_metric:
        cur.execute(
            """
            SELECT model, accuracy
            FROM evaluations
            WHERE dataset = ? AND metric = ?
            """,
            (dataset, metric),
        )
        rows = cur.fetchall()
    else:
        cur.execute("SELECT model, accuracy FROM evaluations")
        rows = [
            (model, accuracy)
            for model, accuracy in cur.fetchall()
            if _model_belongs_to_dataset(model, dataset)
        ]

    conn.close()
    return rows


def _fetch_eval_rows_full(
        db_path: str,
        dataset: str,
        metric: Optional[str] = None,
) -> List[Tuple[str, str, float]]:
    """
    Fetch full evaluation rows for exactly one dataset and metric.

    Falls back to the legacy model-name dataset guard if older DB rows do not
    contain dataset/metric metadata.
    """
    metric = _normalize_metric(metric)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    has_dataset = _table_has_column(cur, "evaluations", "dataset")
    has_metric = _table_has_column(cur, "evaluations", "metric")

    if has_dataset and has_metric:
        cur.execute(
            """
            SELECT model, test_case, accuracy
            FROM evaluations
            WHERE dataset = ? AND metric = ?
            """,
            (dataset, metric),
        )
        rows = cur.fetchall()
    else:
        cur.execute("SELECT model, test_case, accuracy FROM evaluations")
        rows = [
            (model, test_case, accuracy)
            for model, test_case, accuracy in cur.fetchall()
            if _model_belongs_to_dataset(model, dataset)
        ]

    conn.close()
    return rows


def _fetch_train_times(db_path: str) -> Dict[str, float]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT model, total_train_time FROM training_runs")
    d = {m: float(t) for (m, t) in cur.fetchall() if t is not None}
    conn.close()
    return d


# def _extended_stats_table(acc_by_model: Dict[str, List[float]]) -> List[Dict[str, float]]:
#     tab = []
#     for model, accs in acc_by_model.items():
#         accs_clean = [a for a in accs if a is not None]
#         if not accs_clean:
#             continue
#         tab.append({
#             "model": model,
#             "avg": mean(accs_clean),
#             "median": median(accs_clean),
#             "min": min(accs_clean),
#             "max": max(accs_clean),
#             "std": stdev(accs_clean) if len(accs_clean) > 1 else 0.0
#         })
#     # Order: avg desc, std asc, min desc, median desc, max desc
#     tab.sort(key=lambda s: (-s["avg"], s["std"], -s["min"], -s["median"], -s["max"]))
#     return tab


# def _print_extended_stats(acc_by_model: Dict[str, List[float]], export_dir: Optional[str] = None):
#     tab = _extended_stats_table(acc_by_model)
#     print("\n📈 Extended stats per model (quality):")
#     for s in tab:
#         print(f"🧪 {s['model']}: avg={s['avg']:.4f}, median={s['median']:.4f}, "
#               f"min={s['min']:.4f}, max={s['max']:.4f}, std={s['std']:.4f}")
#     if export_dir:
#         _write_csv(
#             os.path.join(export_dir, "ranking_quality.csv"),
#             [(s["model"], s["avg"], s["median"], s["min"], s["max"], s["std"]) for s in tab],
#             ["model","avg","median","min","max","std"]
#         )
def _export_gap_train_test(db_path: str, dataset: str, export_dir: Optional[str], metric: Optional[str] = None):
    if not export_dir:
        return
    rows = _fetch_eval_rows_full(db_path, dataset, metric)
    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"trainlike": [], "ood": []})
    for m, t, a in rows:
        (per_model[m]["trainlike"] if _is_train_like(t) else per_model[m]["ood"]).append(a)
    out = []
    for m, parts in per_model.items():
        tl, od = parts["trainlike"], parts["ood"]
        if tl and od:
            gap = mean(tl) - mean(od)
            out.append((m, gap, mean(tl), mean(od)))
    out.sort(key=lambda r: -r[1])  # biggest gap first
    _write_csv(os.path.join(export_dir, "train_test_gap.csv"), out, ["model", "gap", "avg_trainlike", "avg_ood"])


def _print_extended_stats(acc_by_model: Dict[str, List[float]], export_dir: Optional[str] = None):
    tab = _extended_stats_table(acc_by_model)
    print("\n📈 Extended stats per model (quality):")
    for s in tab:
        print(
            f"🧪 {s['model']}: avg={s['avg']:.4f}, median={s['median']:.4f}, "
            f"min={s['min']:.4f}, max={s['max']:.4f}, std={s['std']:.4f}, "
            f"robust_mean={s['robust_mean']:.4f}, iqr={s['iqr']:.4f}"
        )
    if export_dir:
        _write_csv(
            os.path.join(export_dir, "ranking_quality.csv"),
            [(s["model"], s["avg"], s["median"], s["min"], s["max"], s["std"], s["robust_mean"], s["iqr"]) for s in
             tab],
            ["model", "avg", "median", "min", "max", "std", "robust_mean", "iqr"]
        )


def _print_gap_train_test(db_path: str, dataset: str, metric: Optional[str] = None):
    rows = _fetch_eval_rows_full(db_path, dataset, metric)
    if not rows:
        return
    per_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"trainlike": [], "ood": []})
    for m, t, a in rows:
        (per_model[m]["trainlike"] if _is_train_like(t) else per_model[m]["ood"]).append(a)
    print("\n🔀 Train–test gap (avg_trainlike - avg_ood):")
    for m, parts in per_model.items():
        tl = parts["trainlike"];
        od = parts["ood"]
        if tl and od:
            gap = mean(tl) - mean(od)
            print(f"🪢 {m}: gap={gap:.4f} (trainlike={mean(tl):.4f}, ood={mean(od):.4f})")


# ============================= time-aware stats ==============================

def _norm_minmax_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vmin = min(d.values());
    vmax = max(d.values())
    span = (vmax - vmin) if (vmax > vmin) else 1.0
    return {k: (v - vmin) / span for k, v in d.items()}


def _trimmed_mean(values: List[float], trim: float = 0.10) -> float:
    if not values:
        return 0.0
    n = len(values);
    k = int(n * trim)
    vals = sorted(values)
    core = vals[k:n - k] if n - 2 * k > 0 else vals
    return mean(core) if core else mean(vals)


def _print_extended_stats_per_time(
        acc_by_model: Dict[str, List[float]],
        train_times: Dict[str, float],
        export_dir: Optional[str] = None,
) -> None:
    rows = []
    for model, accs in acc_by_model.items():
        t = train_times.get(model)
        if not t or t <= 0:
            continue
        perfs = [a / t for a in accs if a is not None]
        if not perfs:
            continue
        rows.append({
            "model": model,
            "avg_perf": mean(perfs),
            "median_perf": median(perfs),
            "min_perf": min(perfs),
            "max_perf": max(perfs),
            "std_perf": stdev(perfs) if len(perfs) > 1 else 0.0,
            "time": t,
        })

    rows.sort(key=lambda s: (-s["avg_perf"], -s["min_perf"], -s["median_perf"], -s["max_perf"], s["std_perf"]))

    print("\n📈 Extended stats per model (per-time):")
    for s in rows:
        print(
            f"🧪 {s['model']}: "
            f"avg_perf={s['avg_perf']:.6f}, median_perf={s['median_perf']:.6f}, "
            f"min_perf={s['min_perf']:.6f}, max_perf={s['max_perf']:.6f}, "
            f"std_perf={s['std_perf']:.6f}, time={s['time']:.2f}s"
        )
    if export_dir:
        _write_csv(
            os.path.join(export_dir, "ranking_per_time.csv"),
            [(s["model"], s["avg_perf"], s["median_perf"], s["min_perf"], s["max_perf"], s["std_perf"], s["time"]) for s
             in rows],
            ["model", "avg_perf", "median_perf", "min_perf", "max_perf", "std_perf", "time_s"]
        )


def _print_time_aware_stats(
        acc_by_model: Dict[str, List[float]],
        train_times: Dict[str, float],
        alpha: float = 0.70,
        top_n: Optional[int] = None,
        export_dir: Optional[str] = None,
) -> None:
    avg_map: Dict[str, float] = {}
    min_map: Dict[str, float] = {}
    avg_perf: Dict[str, float] = {}
    robust_perf: Dict[str, float] = {}

    for model, accs in acc_by_model.items():
        if not accs:
            continue
        a = mean(accs)
        m = min(accs)
        avg_map[model] = a
        min_map[model] = m
        t = train_times.get(model)
        if t and t > 0:
            avg_perf[model] = a / t
            robust_perf[model] = m / t

    norm_avg = _norm_minmax_dict(avg_map)
    norm_avg_perf = _norm_minmax_dict(avg_perf) if avg_perf else {}
    score: Dict[str, float] = {}
    for model in avg_map.keys():
        s1 = norm_avg.get(model, 0.0)
        s2 = norm_avg_perf.get(model, 0.0)
        score[model] = alpha * s1 + (1.0 - alpha) * s2

    def _sort_key_avgperf(m):
        return (-avg_perf.get(m, -math.inf), -robust_perf.get(m, -math.inf))

    def _sort_key_score(m):
        return (-score.get(m, -math.inf), -avg_map.get(m, -math.inf))

    models = list(avg_map.keys())

    print("\n⚡ Time-aware ranking (average performance = avg/time):")
    top_avgperf = []
    for m in sorted(models, key=_sort_key_avgperf)[:top_n]:
        t = train_times.get(m);
        ap = avg_perf.get(m);
        rp = robust_perf.get(m)
        if t is None or ap is None:
            continue
        top_avgperf.append((m, t, ap, rp))
        print(f"🚀 {m}: train_time={t:.2f}s, avg_perf={ap:.6f}, robust_perf={rp:.6f}")

    print("\n🏅 Balanced ranking (alpha-weighted quality + efficiency):")
    print(f"    score = {alpha:.2f}·norm(avg) + {1 - alpha:.2f}·norm(avg_perf)")
    top_balanced = []
    for m in sorted(models, key=_sort_key_score)[:top_n]:
        t = train_times.get(m);
        ap = avg_perf.get(m);
        sc = score.get(m)
        if t is None or ap is None or sc is None:
            continue
        top_balanced.append((m, sc, avg_map[m], ap, t))
        print(f"🥇 {m}: score={sc:.4f}, avg={avg_map[m]:.4f}, avg_perf={ap:.6f}, time={t:.2f}s")

    if export_dir:
        _write_csv(
            os.path.join(export_dir, "ranking_timeaware_avgperf.csv"),
            [(m, ap, rp, t) for (m, t, ap, rp) in top_avgperf],
            ["model", "avg_perf", "robust_perf", "time_s"]
        )
        _write_csv(
            os.path.join(export_dir, "ranking_timeaware_balanced.csv"),
            [(m, sc, avg, ap, t) for (m, sc, avg, ap, t) in top_balanced],
            ["model", "score", "avg", "avg_perf", "time_s"]
        )


def _print_balanced_ranking_per_time(
        acc_by_model: Dict[str, List[float]],
        train_times: Dict[str, float],
        alpha: float = 0.70,
        top_n: int = 20,
        export_dir: Optional[str] = None,
) -> None:
    """
    Balanced ranking based ONLY on per-time metrics:
      score = alpha·norm(avg_perf) + (1-alpha)·norm(robust_perf)
    where:
      avg_perf    = mean(accuracy_i / time)
      robust_perf = 10% symmetric trimmed mean of (accuracy_i / time)
    """
    rows = []
    for model, accs in acc_by_model.items():
        t = train_times.get(model)
        if not t or t <= 0:
            continue
        perfs = [a / t for a in accs if a is not None]
        if not perfs:
            continue
        avgp = mean(perfs)
        robp = _trimmed_mean(perfs, trim=0.10)
        rows.append({"model": model, "avg_perf": avgp, "robust_perf": robp, "time": t})

    if not rows:
        print("\n⚠️ No per-time data available for balanced per-time ranking.")
        return

    min_avg = min(r["avg_perf"] for r in rows)
    max_avg = max(r["avg_perf"] for r in rows)
    min_rob = min(r["robust_perf"] for r in rows)
    max_rob = max(r["robust_perf"] for r in rows)
    span_avg = max(max_avg - min_avg, 1e-12)
    span_rob = max(max_rob - min_rob, 1e-12)

    for r in rows:
        r["norm_avg_perf"] = (r["avg_perf"] - min_avg) / span_avg
        r["norm_robust_perf"] = (r["robust_perf"] - min_rob) / span_rob
        r["score"] = alpha * r["norm_avg_perf"] + (1.0 - alpha) * r["norm_robust_perf"]

    rows.sort(key=lambda x: (-x["score"], -x["avg_perf"], -x["robust_perf"]))

    print("\n🏅 Balanced ranking (per-time only):")
    print(f"    score = {alpha:.2f}·norm(avg_perf) + {1 - alpha:.2f}·norm(robust_perf)")
    for r in rows[:top_n]:
        print(
            "🥇 {model}: score={score:.4f}, "
            "avg_perf={avg_perf:.6f}, robust_perf={robust_perf:.6f}, time={time:.2f}s"
            .format(**r)
        )

    if export_dir:
        _write_csv(
            os.path.join(export_dir, "ranking_balanced_per_time.csv"),
            [(r["model"], r["score"], r["avg_perf"], r["robust_perf"], r["time"]) for r in rows],
            ["model", "score", "avg_perf", "robust_perf", "time_s"]
        )


# ======================= rotation: Δθ, Acc(Δθ), AUCθ ========================

ANGLE_TOKEN = re.compile(
    r'(?i)(?:rotated-(\d+)(?:-(\d+))?)|(?:range[_-](\d+)[_-](\d+))|(?:full[_-]0[_-]360)'
)


def _interval_from_token(name: str) -> Optional[Tuple[float, float]]:
    s = name.lower()
    m = ANGLE_TOKEN.search(s)
    if not m:
        if "non_rotated" in s:
            return (0.0, 0.0)
        return None
    if m.group(1):  # rotated-a or rotated-a-b
        a = float(m.group(1));
        b = float(m.group(2)) if m.group(2) else float(m.group(1))
        return (a, b)
    if m.group(3):  # range_a_b
        a = float(m.group(3));
        b = float(m.group(4))
        return (a, b)
    return (0.0, 360.0)  # full_0_360


def _center_deg(iv: Tuple[float, float]) -> float:
    a, b = iv
    if b >= a:
        return (a + b) / 2.0
    # wrap (e.g., 330..30)
    return (a + ((b + 360.0 - a) / 2.0)) % 360.0


def _delta_deg(train_name: str, test_name: str) -> Optional[float]:
    it = _interval_from_token(train_name)
    ie = _interval_from_token(test_name)
    if not it or not ie:
        return None
    ct = _center_deg(it);
    ce = _center_deg(ie)
    d = abs(ct - ce) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return d


def _bin_delta(d: float, step: int = 15) -> int:
    b = int(round(d / step) * step)
    return min(b, 180)


def _acc_vs_delta_and_auc(db_path: str, dataset: str, theta_step: int = 15,
                          out_dir: Optional[str] = None, metric: Optional[str] = None):
    rows = _fetch_eval_rows_full(db_path, dataset, metric)
    if not rows:
        return
    by_model: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for m, t, a in rows:
        d = _delta_deg(m, t)
        if d is None:
            continue
        b = _bin_delta(d, theta_step)
        by_model[m][b].append(a)

    xgrid = list(range(0, 181, theta_step))
    exports = []
    for m, bins in by_model.items():
        ys = []
        for x in xgrid:
            bucket = bins.get(x, [])
            ys.append(mean(bucket) if bucket else None)
        # simple forward fill, then fill initial None with first known
        last = None
        for i, v in enumerate(ys):
            if v is None:
                ys[i] = last
            else:
                last = v
        first_known = next((v for v in ys if v is not None), 0.0)
        ys = [first_known if v is None else v for v in ys]

        # AUCθ (trapezoidal), normalized by 180°
        auc = 0.0
        for i in range(1, len(xgrid)):
            auc += (ys[i - 1] + ys[i]) * (xgrid[i] - xgrid[i - 1]) / 2.0
        auc_norm = auc / 180.0
        acc_worst = min(ys) if ys else 0.0
        sd_theta = float(np.std(ys)) if ys else 0.0
        exports.append((m, auc_norm, acc_worst, sd_theta))

        if out_dir:
            _ensure_dir(out_dir)
            _write_csv(
                os.path.join(out_dir, f"acc_vs_delta_{m}.csv"),
                [[m] + ys],
                ["model"] + [f"d{x}" for x in xgrid]
            )

    exports.sort(key=lambda r: (-r[1], -r[2], r[3]))  # AUCθ desc, worst desc, SDθ asc

    if out_dir:
        _write_csv(
            os.path.join(out_dir, "auc_theta_ranking.csv"),
            [(m, a, w, s) for (m, a, w, s) in exports],
            ["model", "auc_theta_norm", "acc_worst", "sd_theta"]
        )

    print("\n🧭 AUCθ ranking (normalized):")
    for m, a, w, s in exports[:20]:
        print(f"⦿ {m}: AUCθ={a:.4f}, worst={w:.4f}, SDθ={s:.4f}")


# ============================= legacy summary table =========================

def create_model_summary_table(db_path: str):
    """
    Optional legacy table for summaries, kept for CLI compatibility.
    """
    conn = sqlite3.connect(db_path);
    cur = conn.cursor()
    cur.execute('DROP TABLE IF EXISTS model_summary')
    cur.execute('''
        CREATE TABLE model_summary (
            model TEXT PRIMARY KEY,
            avg REAL,
            median REAL,
            min REAL,
            max REAL,
            std REAL,
            train_time REAL,
            perf_per_time REAL
        )
    ''')
    conn.commit();
    conn.close()
    print("✅ Created table: model_summary")


def compute_and_insert_model_summaries(db_path: str):
    """
    Populate 'model_summary' using 'evaluations' + 'training_runs'.
    """
    conn = sqlite3.connect(db_path);
    cur = conn.cursor()
    cur.execute('SELECT model, accuracy FROM evaluations')
    rows = cur.fetchall()

    acc_by_model: Dict[str, List[float]] = defaultdict(list)
    for model, acc in rows:
        acc_by_model[model].append(acc)

    cur.execute('SELECT model, total_train_time FROM training_runs')
    time_map = {m: t for (m, t) in cur.fetchall()}

    inserts = []
    for model, accs in acc_by_model.items():
        if not accs: continue
        avg_ = mean(accs);
        med_ = median(accs);
        min_ = min(accs);
        max_ = max(accs)
        std_ = stdev(accs) if len(accs) > 1 else 0.0
        tt = time_map.get(model)
        ppt = (avg_ / tt) if tt else None
        inserts.append((model, avg_, med_, min_, max_, std_, tt, ppt))

    if inserts:
        cur.executemany('''
            INSERT INTO model_summary (
                model, avg, median, min, max, std, train_time, perf_per_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', inserts)
        conn.commit()
    conn.close()
    print(f"✅ Inserted {len(inserts)} model summary rows")


# ================================ main report ===============================
#
# def query_best_models(
#     db_path: str,
#     dataset: str,
#     alpha: float = 0.70,
#     top_n: int = 50,
#     theta_step: int = 15,
# ) -> None:
#     """
#     Per-dataset report:
#       - average accuracy per model
#       - extended stats (quality) + CSV
#       - train–test gap
#       - per-time extended + time-aware + balanced per-time + CSV
#       - Acc(Δθ) / AUCθ + CSV
#       Exports go to results/exports/<dataset>/
#     """
#     rows = _fetch_eval_rows(db_path, dataset)
#     if not rows:
#         print(f"⚠️ No evaluation rows for dataset '{dataset}'.")
#         return
#
#     export_root = os.path.join("results", "exports", dataset)
#     _ensure_dir(export_root)
#
#     # group
#     acc_by_model: Dict[str, List[float]] = defaultdict(list)
#     for m, a in rows:
#         acc_by_model[m].append(a)
#
#     # avg per model
#     avg_list = [(m, mean(v)) for m, v in acc_by_model.items()]
#     avg_list.sort(key=lambda x: -x[1])
#
#     print("\n📊 Average accuracy per model (per-dataset):")
#     for m, a in avg_list:
#         print(f"🧠 {m}: {a:.4f}")
#     if avg_list:
#         print(f"\n🏆 Best model (by accuracy): {avg_list[0][0]} with avg accuracy: {avg_list[0][1]:.4f}")
#
#     # quality-only extended + CSV
#     _print_extended_stats(acc_by_model, export_dir=export_root)
#
#     # train–test gap
#     _print_gap_train_test(db_path, dataset, metric)
#
#     # time-aware views
#     train_times = _fetch_train_times(db_path)
#     _print_extended_stats_per_time(acc_by_model, train_times, export_dir=export_root)
#     _print_time_aware_stats(acc_by_model, train_times, alpha=alpha, top_n=top_n, export_dir=export_root)
#     _print_balanced_ranking_per_time(acc_by_model, train_times, alpha=alpha, top_n=top_n, export_dir=export_root)
#
#     # rotation stability
#     delta_dir = os.path.join(export_root, "delta_curves")
#     _acc_vs_delta_and_auc(db_path, dataset, theta_step=theta_step, out_dir=delta_dir, metric=metric)
#     _export_gap_train_test(db_path, dataset, export_root, metric)

def query_best_models(
        db_path: str,
        dataset: str,
        alpha: float = 0.70,
        top_n: int = 500,
        theta_step: int = 15,
        metric: Optional[str] = None,  # NEW
) -> None:
    """
    Per-dataset report:
      - average accuracy per model
      - extended stats (quality) + CSV
      - train–test gap
      - per-time extended + time-aware + balanced per-time + CSV
      - Acc(Δθ) / AUCθ + CSV
    Exports go to: <cwd>/results/exports/<dataset>/<metric>/
    """
    metric = _normalize_metric(metric)
    rows = _fetch_eval_rows(db_path, dataset, metric)
    if not rows:
        print(f"⚠️ No evaluation rows for dataset '{dataset}'.")
        return

    export_root = _export_root(dataset, metric)  # ABS path + printed

    # group
    acc_by_model: Dict[str, List[float]] = defaultdict(list)
    for m, a in rows:
        acc_by_model[m].append(a)

    # avg per model
    avg_list = [(m, mean(v)) for m, v in acc_by_model.items()]
    avg_list.sort(key=lambda x: -x[1])

    print("\n📊 Average accuracy per model (per-dataset):")
    for m, a in avg_list:
        print(f"🧠 {m}: {a:.4f}")
    if avg_list:
        print(f"\n🏆 Best model (by accuracy): {avg_list[0][0]} with avg accuracy: {avg_list[0][1]:.4f}")

    # quality-only extended + CSV
    _print_extended_stats(acc_by_model, export_dir=export_root)

    # train–test gap
    _print_gap_train_test(db_path, dataset, metric)

    # time-aware views
    train_times = _fetch_train_times(db_path)
    _print_extended_stats_per_time(acc_by_model, train_times, export_dir=export_root)
    _print_time_aware_stats(acc_by_model, train_times, alpha=alpha, top_n=top_n, export_dir=export_root)
    _print_balanced_ranking_per_time(acc_by_model, train_times, alpha=alpha, top_n=top_n, export_dir=export_root)

    # rotation stability
    delta_dir = os.path.join(export_root, "delta_curves")
    _acc_vs_delta_and_auc(db_path, dataset, theta_step=theta_step, out_dir=delta_dir, metric=metric)
    _export_gap_train_test(db_path, dataset, export_root, metric)


def _iqr(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    q75 = float(np.percentile(arr, 75))
    q25 = float(np.percentile(arr, 25))
    return q75 - q25


def _extended_stats_table(acc_by_model: Dict[str, List[float]]) -> List[Dict[str, float]]:
    tab = []
    for model, accs in acc_by_model.items():
        accs_clean = [a for a in accs if a is not None]
        if not accs_clean:
            continue
        tab.append({
            "model": model,
            "avg": mean(accs_clean),
            "median": median(accs_clean),
            "min": min(accs_clean),
            "max": max(accs_clean),
            "std": stdev(accs_clean) if len(accs_clean) > 1 else 0.0,
            "robust_mean": _trimmed_mean(accs_clean, trim=0.10),
            "iqr": _iqr(accs_clean),
        })
    # Order: avg desc, std asc, min desc, median desc, max desc, robust_mean desc, iqr asc
    tab.sort(key=lambda s: (-s["avg"], s["std"], -s["min"], -s["median"], -s["max"], -s["robust_mean"], s["iqr"]))
    return tab


# ======== Per-class vs angle (by test-angle and by Δθ) ======================
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _per_class_acc_from_cm(cm: np.ndarray) -> np.ndarray:
    """Return per-class accuracy vector (len = C)."""
    cm = cm.astype(np.float64)
    rows = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / rows
        acc[~np.isfinite(acc)] = 0.0
    return acc


def _angle_center_from_name(name: str) -> Optional[float]:
    iv = _interval_from_token(name)
    return _center_deg(iv) if iv else None


def _export_per_class_matrix(
        model: str,
        mat: np.ndarray,
        angle_bins: List[int],
        out_dir: str,
        title: str
):
    """
    mat: shape [num_classes, num_bins] in [0,1]
    angle_bins: ascending list (e.g., [0,15,...,180])
    """
    _ensure_dir(out_dir)
    # CSV (wide)
    csv_path = os.path.join(out_dir, f"{model}_per_class.csv")
    header = ["class"] + [f"{b}" for b in angle_bins]
    rows = []
    for c in range(mat.shape[0]):
        rows.append(tuple([c] + [float(x) for x in mat[c, :]]))
    _write_csv(csv_path, rows, header)

    # PNG heatmap
    png_path = os.path.join(out_dir, f"{model}_per_class.png")
    plt.figure(figsize=(14, 9), dpi=200)
    im = plt.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Per-class accuracy")
    plt.yticks(np.arange(0, mat.shape[0], max(1, mat.shape[0] // 20)))
    plt.xticks(ticks=np.arange(len(angle_bins)), labels=[str(b) for b in angle_bins], rotation=90)
    plt.xlabel("Angle bin [deg]")
    plt.ylabel("Class id")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def export_per_class_vs_angle(
        cm_root: str,
        dataset: str,
        theta_step: int = 15,
        by_delta: bool = False,
):
    """
    Build per-class accuracy heatmaps per model:
      - by_test  : bins by test-angle center
      - by_delta : bins by Δθ(train, test)
    Exports:
      results/exports/<DATASET>/per_class_by_test/<model>_per_class.{csv,png}
      results/exports/<DATASET>/per_class_by_delta/<model>_per_class.{csv,png}
    """
    if not os.path.isdir(cm_root):
        print(f"❌ Not a directory: {cm_root}")
        return

    mode_name = "per_class_by_delta" if by_delta else "per_class_by_test"
    base_out = os.path.join("results", "exports", dataset, mode_name)
    _ensure_dir(base_out)
    print(f"🖨️  Exporting {mode_name} to: {base_out}")

    # Angle grid (0..180)
    angle_bins = list(range(0, 181, theta_step))

    for model_dir in tqdm(sorted(os.listdir(cm_root)), desc="🎯 Per-class pass"):
        if not _model_belongs_to_dataset(model_dir, dataset):
            continue
        model_path = os.path.join(cm_root, model_dir)
        if not os.path.isdir(model_path):
            continue

        # First pass to infer number of classes
        num_classes = None
        for test_subdir in os.listdir(model_path):
            cmf = os.path.join(model_path, test_subdir, "confusion_matrix.npy")
            if os.path.exists(cmf):
                cm = _safe_load_cm(cmf)
                if cm is not None:
                    num_classes = cm.shape[0]
                    break
        if num_classes is None:
            continue

        # buckets: angle_bin -> list of per-class vectors
        buckets: Dict[int, List[np.ndarray]] = defaultdict(list)

        for test_subdir in sorted(os.listdir(model_path)):
            cmf = os.path.join(model_path, test_subdir, "confusion_matrix.npy")
            if not os.path.exists(cmf):
                continue
            cm = _safe_load_cm(cmf)
            if cm is None:
                continue

            if by_delta:
                d = _delta_deg(model_dir, test_subdir)
            else:
                d = _angle_center_from_name(test_subdir)
            if d is None:
                continue

            b = _bin_delta(float(d), theta_step)
            acc_vec = _per_class_acc_from_cm(cm)
            if acc_vec.shape[0] != num_classes:
                # skip malformed
                continue
            buckets[b].append(acc_vec)

        if not buckets:
            continue

        # aggregate to matrix [C, #bins]
        mat = np.full((num_classes, len(angle_bins)), np.nan, dtype=np.float64)
        for j, b in enumerate(angle_bins):
            vecs = buckets.get(b, [])
            if not vecs:
                continue
            V = np.stack(vecs, axis=0)  # [K, C]
            mat[:, j] = V.mean(axis=0)

        # fill NaNs (forward/backward fill along angle axis)
        # left->right
        for j in range(1, mat.shape[1]):
            nan_rows = np.isnan(mat[:, j])
            mat[nan_rows, j] = mat[nan_rows, j - 1]
        # right->left
        for j in range(mat.shape[1] - 2, -1, -1):
            nan_rows = np.isnan(mat[:, j])
            mat[nan_rows, j] = mat[nan_rows, j + 1]

        out_dir = os.path.join(base_out)
        title = f"{model_dir} – per-class accuracy vs {'Δθ' if by_delta else 'test angle'}"
        export_name_model = model_dir.replace("/", "_").replace("\\", "_")
        _export_per_class_matrix(
            model=export_name_model,
            mat=mat,
            angle_bins=angle_bins,
            out_dir=out_dir,
            title=title
        )
