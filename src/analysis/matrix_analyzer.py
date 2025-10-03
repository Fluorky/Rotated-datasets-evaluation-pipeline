# -*- coding: utf-8 -*-
"""
Matrix analyzer (per-dataset), with time-aware rankings.

What it does
------------
1) Reads confusion matrices under <cm_root>/<MODEL>/<TEST_CASE>/confusion_matrix.npy
2) Computes accuracy per (model, test_case) and stores to SQLite 'evaluations'
3) Pulls training time from 'training_logs' (via compute_training_times -> training_runs)
4) Prints per-dataset rankings:
   - Quality-only (avg, min, median, max, std)
   - Time-aware: avg_perf (=avg/time), robust_perf (=min/time), balanced score
5) Writes all console output to an optional log file (tee).

Notes
-----
- Dataset scoping is strict:
    MNIST      -> model name contains 'MNIST'
    GTSRB      -> contains 'GTSRB' but NOT 'GTSRB_RGB' / 'GTSRB-RGB'
    GTSRB_RGB  -> contains 'GTSRB_RGB' or 'GTSRB-RGB'
    LEGO       -> contains 'LEGO'
- Extended stats are printed in the exact style you requested.

"""

import os
import sys
import math
import sqlite3
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from statistics import mean, median, stdev
from collections import defaultdict, OrderedDict
from contextlib import contextmanager


# ----------------------------- tee / logging --------------------------------

@contextmanager
def tee_to_file(log_file: Optional[str]):
    """
    Mirror stdout to a file if log_file is provided.
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
            self.a.write(s)
            self.b.write(s)
        def flush(self):
            self.a.flush()
            self.b.flush()
    old = sys.stdout
    f = open(log_file, "w", encoding="utf-8")
    sys.stdout = TeeWriter(old, f)
    try:
        yield
    finally:
        sys.stdout = old
        f.close()


# ----------------------------- helpers --------------------------------------

def _safe_load_cm(path: str) -> Optional[np.ndarray]:
    try:
        return np.load(path)
    except Exception:
        return None

def _accuracy_from_cm(cm: np.ndarray) -> float:
    total = cm.sum()
    if total <= 0:
        return 0.0
    return float(np.trace(cm)) / float(total)

def _upper(s: str) -> str:
    return s.upper().replace("-", "_")

def _model_belongs_to_dataset(model_name: str, dataset: str) -> bool:
    """
    Strict dataset matching based on model folder name.
    Avoids mixing GTSRB with GTSRB_RGB.
    """
    m = _upper(model_name)
    ds = _upper(dataset)
    if ds == "GTSRB_RGB":
        return ("GTSRB_RGB" in m)
    if ds == "GTSRB":
        return ("GTSRB" in m) and ("GTSRB_RGB" not in m)
    return (ds in m)

def _norm_minmax(values: List[float]) -> Dict[str, float]:
    """
    Return mapping id->normalized value in [0,1].
    Here we accept a dict later; this helper works on simple list.
    """
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    span = (vmax - vmin) if (vmax > vmin) else 1.0
    return {i: (val - vmin) / span for i, val in enumerate(values)}

def _norm_minmax_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vmin = min(d.values())
    vmax = max(d.values())
    span = (vmax - vmin) if (vmax > vmin) else 1.0
    return {k: (v - vmin) / span for k, v in d.items()}


# ----------------------------- DB schema ------------------------------------

def _initialize_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
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
    return conn

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

def compute_training_times(db_path: str) -> None:
    """
    Aggregate total training time per log_file from 'training_logs'
    (populated by log_ingestor), and insert into 'training_runs'.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Might not exist if user didn't ingest training logs; guard it.
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
        model = log_file[:-10] if log_file.endswith("_train.txt") else log_file
        entries.append((model, log_file, round(float(total_time or 0.0), 2)))

    if entries:
        cur.executemany(
            "INSERT OR REPLACE INTO training_runs (model, log_file, total_train_time) VALUES (?, ?, ?)",
            entries
        )
        conn.commit()
    conn.close()
    print(f"✅ Inserted {len(entries)} rows into training_runs")


# ---------------------- main ingestion from confusion matrices --------------

def collect_and_store_results(cm_root: str, db_path: str, dataset: Optional[str] = None) -> None:
    """
    Scan confusion matrices and (re)fill 'evaluations' with enriched rows.
    If 'dataset' is provided, only models that belong to that dataset are accepted.
    """
    conn = _initialize_db(db_path)
    cur  = conn.cursor()

    raw: List[Tuple[str, str, float]] = []

    if not os.path.isdir(cm_root):
        print(f"❌ Not a directory: {cm_root}")
        conn.close()
        return

    for model_dir in tqdm(sorted(os.listdir(cm_root)), desc="📁 Scanning models"):
        model_path = os.path.join(cm_root, model_dir)
        if not os.path.isdir(model_path):
            continue

        # dataset filter
        if dataset and not _model_belongs_to_dataset(model_dir, dataset):
            continue

        for test_subdir in sorted(os.listdir(model_path)):
            test_path = os.path.join(model_path, test_subdir)
            cm_file = os.path.join(test_path, "confusion_matrix.npy")
            if not os.path.exists(cm_file):
                continue
            cm = _safe_load_cm(cm_file)
            if cm is None:
                print(f"⚠️ Failed to load {cm_file}")
                continue
            acc = _accuracy_from_cm(cm)
            raw.append((model_dir, test_subdir, acc))

    # Group accuracies by model
    acc_by_model: Dict[str, List[float]] = defaultdict(list)
    for m, _, a in raw:
        acc_by_model[m].append(a)

    # Insert enriched rows to 'evaluations'
    # (We don't truncate the table, so multiple runs can co-exist — up to you.)
    rows_to_insert = []
    # get training time map once
    cur.execute("""
        SELECT model, total_train_time FROM training_runs
    """)
    time_map = {r[0]: (r[1] or None) for r in cur.fetchall()}

    for model, test_case, acc in raw:
        m_accs = acc_by_model[model]
        avg_   = mean(m_accs)
        med_   = median(m_accs)
        min_   = min(m_accs)
        max_   = max(m_accs)
        std_   = stdev(m_accs) if len(m_accs) > 1 else 0.0

        ttime = time_map.get(model)
        perf  = (acc / ttime) if (ttime and ttime > 0) else None

        rows_to_insert.append((
            model, test_case, float(acc),
            float(avg_), float(med_), float(min_), float(max_), float(std_),
            float(ttime) if ttime else None,
            float(perf) if perf else None
        ))

    if rows_to_insert:
        cur.executemany("""
            INSERT INTO evaluations (
                model, test_case, accuracy,
                avg, median, min, max, std,
                train_time, perf_per_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        conn.commit()

    conn.close()
    print(f"✅ Saved {len(rows_to_insert)} rows into evaluations (db: {db_path})")


# ---------------------- reporting / rankings (per dataset) ------------------

def _fetch_eval_rows(db_path: str, dataset: str) -> List[Tuple[str, float]]:
    """
    Return [(model, accuracy), ...] filtered by dataset using Python-side guard.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT model, accuracy FROM evaluations")
    rows = cur.fetchall()
    conn.close()
    # strict filter to avoid GTSRB vs GTSRB_RGB mixing
    return [(m, a) for (m, a) in rows if _model_belongs_to_dataset(m, dataset)]

def _fetch_train_times(db_path: str) -> Dict[str, float]:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT model, total_train_time FROM training_runs")
    d = {m: float(t) for (m, t) in cur.fetchall() if t is not None}
    conn.close()
    return d

def _print_extended_stats(acc_by_model: Dict[str, List[float]]) -> None:
    """
    Classic extended stats (without time). Sorted by avg desc, then min desc,
    then median desc, then max desc, then std asc.
    """
    summary = []
    for model, accs in acc_by_model.items():
        accs_clean = [a for a in accs if a is not None]
        if not accs_clean:
            continue
        summary.append({
            "model":  model,
            "avg":    mean(accs_clean),
            "median": median(accs_clean),
            "min":    min(accs_clean),
            "max":    max(accs_clean),
            "std":    stdev(accs_clean) if len(accs_clean) > 1 else 0.0
        })

    summary.sort(key=lambda s: (-s["avg"], -s["min"], -s["median"], -s["max"], s["std"]))

    print("\n📈 Extended stats per model:")
    for s in summary:
        print(f"🧪 {s['model']}: avg={s['avg']:.4f}, median={s['median']:.4f}, "
              f"min={s['min']:.4f}, max={s['max']:.4f}, std={s['std']:.4f}")

def _print_time_aware_stats(
    acc_by_model: Dict[str, List[float]],
    train_times: Dict[str, float],
    alpha: float = 0.70,
    top_n: Optional[int] = None
) -> None:
    """
    Time-aware view: avg_perf, robust_perf, and a balanced score
    (alpha*norm(avg) + (1-alpha)*norm(avg_perf)).
    """
    # Build metrics
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

    # Balanced score
    norm_avg       = _norm_minmax_dict(avg_map)
    norm_avg_perf  = _norm_minmax_dict(avg_perf) if avg_perf else {}
    score: Dict[str, float] = {}
    for model in avg_map.keys():
        s1 = norm_avg.get(model, 0.0)
        s2 = norm_avg_perf.get(model, 0.0)
        score[model] = alpha * s1 + (1.0 - alpha) * s2

    # Sorting
    def _sort_key_quality(m):   # same as in classic
        return (-avg_map.get(m, -1), -min_map.get(m, -1))
    def _sort_key_avgperf(m):
        return (-avg_perf.get(m, -math.inf), -robust_perf.get(m, -math.inf))
    def _sort_key_score(m):
        return (-score.get(m, -math.inf), -avg_map.get(m, -math.inf))

    models = list(acc_by_model.keys())

    print("\n⚡ Time-aware ranking (average performance = avg/time):")
    for m in sorted(models, key=_sort_key_avgperf)[:top_n]:
        t = train_times.get(m)
        ap = avg_perf.get(m)
        rp = robust_perf.get(m)
        if t is None or ap is None:
            continue
        print(f"🚀 {m}: train_time={t:.2f}s, avg_perf={ap:.6f}, robust_perf={rp:.6f}")

    print("\n🏅 Balanced ranking (alpha-weighted quality + efficiency):")
    print(f"    score = {alpha:.2f}·norm(avg) + {1-alpha:.2f}·norm(avg_perf)")
    for m in sorted(models, key=_sort_key_score)[:top_n]:
        t = train_times.get(m)
        ap = avg_perf.get(m)
        sc = score.get(m)
        if t is None or ap is None or sc is None:
            continue
        print(f"🥇 {m}: score={sc:.4f}, avg={avg_map[m]:.4f}, avg_perf={ap:.6f}, time={t:.2f}s")

def query_best_models(db_path: str, dataset: str, alpha: float = 0.70, top_n: int = 500) -> None:
    """
    High-level per-dataset report:
      - Average accuracy per model (SQL-style)
      - Classic "Extended stats per model"
      - Time-aware rankings (avg_perf, robust_perf, balanced score)
    """
    rows = _fetch_eval_rows(db_path, dataset)
    if not rows:
        print(f"⚠️ No evaluation rows for dataset '{dataset}'.")
        return

    # group
    acc_by_model: Dict[str, List[float]] = defaultdict(list)
    for model, acc in rows:
        acc_by_model[model].append(acc)

    # SQL-like avg (Python)
    avg_list = [(m, mean(v)) for m, v in acc_by_model.items()]
    avg_list.sort(key=lambda x: -x[1])

    print("\n📊 Average accuracy per model (per-dataset):")
    for m, a in avg_list:
        print(f"🧠 {m}: {a:.4f}")

    if avg_list:
        print(f"\n🏆 Best model (by accuracy): {avg_list[0][0]} with avg accuracy: {avg_list[0][1]:.4f}")

    # Extended stats (classic)
    _print_extended_stats(acc_by_model)

    # # Time-aware
    # train_times = _fetch_train_times(db_path)
    # _print_time_aware_stats(acc_by_model, train_times, alpha=alpha, top_n=top_n)

    train_times = _fetch_train_times(db_path)

    # NEW: extended stats per-time (accuracy / train_time)
    _print_extended_stats_per_time(acc_by_model, train_times)

    # Existing time-aware sections
    _print_time_aware_stats(acc_by_model, train_times, alpha=alpha, top_n=top_n)

    # Balanced ranking based ONLY on per-time metrics (avg_perf & robust_perf)
    _print_balanced_ranking_per_time(acc_by_model, train_times, alpha=0.70, top_n=top_n)


def _trimmed_mean(values, trim=0.10):
    """Simple symmetric trimmed mean: drop trim% lowest & highest."""
    if not values:
        return 0.0
    n = len(values)
    k = int(n * trim)
    vals = sorted(values)
    core = vals[k:n-k] if n - 2*k > 0 else vals
    return mean(core) if core else mean(vals)


def _print_balanced_ranking_per_time(
    acc_by_model: Dict[str, List[float]],
    train_times: Dict[str, float],
    alpha: float = 0.70,
    top_n: int = 10,
) -> None:
    """
    Balanced ranking based ONLY on efficiency metrics (accuracy per time).
    score = alpha·norm(avg_perf) + (1-alpha)·norm(robust_perf)

    Where:
      - avg_perf   = mean(accuracy_i / train_time)
      - robust_perf = trimmed mean 10% of the per-time list
    """
    rows = []
    for model, accs in acc_by_model.items():
        t = train_times.get(model)
        if not t or t <= 0:
            continue
        perfs = [a / t for a in accs if a is not None]
        if not perfs:
            continue
        avg_perf = mean(perfs)
        robust_perf = _trimmed_mean(perfs, trim=0.10)
        rows.append({
            "model": model,
            "avg_perf": avg_perf,
            "robust_perf": robust_perf,
            "time": t,
        })

    if not rows:
        print("\n⚠️ No per-time data available for balanced per-time ranking.")
        return

    # Min-max normalization
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
    print(f"    score = {alpha:.2f}·norm(avg_perf) + {1-alpha:.2f}·norm(robust_perf)")
    for r in rows[:top_n]:
        print(
            "🥇 {model}: score={score:.4f}, "
            "avg_perf={avg_perf:.6f}, robust_perf={robust_perf:.6f}, time={time:.2f}s"
            .format(**r)
        )

def _print_extended_stats_per_time(
    acc_by_model: Dict[str, List[float]],
    train_times: Dict[str, float],
) -> None:
    """
    Extended stats per model, but for performance-per-time (acc / train_time).
    Sort: avg_perf desc, then min_perf desc, median_perf desc, max_perf desc, std_perf asc.
    """
    rows = []
    for model, accs in acc_by_model.items():
        t = train_times.get(model)
        if not t or t <= 0:
            continue
        perfs = [a / t for a in accs]
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
