# src/analysis/matrix_analyzer.py
import os
import sqlite3
from statistics import mean, median, stdev
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------- utils ----------

def _dataset_tokens(dataset: str) -> List[str]:
    """Tokens używane do dopasowania modeli w SQL (case-insensitive)."""
    ds = dataset.strip().lower()
    if ds == "gtsrb_rgb":
        return ["gtsrb-rgb-"]
    if ds == "gtsrb":
        return ["gtsrb-"]
    if ds == "mnist":
        return ["mnist-"]
    if ds == "lego":
        return ["lego-"]
    # fallback — dopasuj początek nazwy jak podano
    return [ds + "-"]

def _dataset_where_sql(alias: str, dataset: str) -> Tuple[str, List[str]]:
    """
    Buduje fragment WHERE dla SQL: (LOWER(alias) LIKE ? OR ...)
    alias – nazwa kolumny z modelem, np. 'model' albo 'm.model'
    """
    toks = _dataset_tokens(dataset)
    parts = [f"LOWER({alias}) LIKE ?" for _ in toks]
    params = [t + "%" for t in toks]
    return "(" + " OR ".join(parts) + ")", params

def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = {row[1].lower() for row in cur.fetchall()}
    if column.lower() not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        conn.commit()

def _tee_print(s: str, log_fp) -> None:
    print(s, flush=True)
    if log_fp:
        log_fp.write(s + "\n")
        log_fp.flush()

# ---------- accuracy helper ----------

def calculate_accuracy(conf_matrix: np.ndarray) -> float:
    correct = np.trace(conf_matrix)
    total = conf_matrix.sum()
    return float(correct) / float(total) if total > 0 else 0.0

# ---------- schema ----------

def initialize_db(db_path: str) -> sqlite3.Connection:
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
            perf_per_time REAL,
            dataset TEXT
        )
    """)
    conn.commit()
    return conn

def create_training_runs_table(db_path: str, log_fp=None) -> None:
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
    _tee_print("✅ Created table: training_runs", log_fp)

def compute_training_times(db_path: str, log_fp=None) -> None:
    """
    Sumuje czasy na podstawie tabeli training_logs (powstaje przy ingescie).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT log_file, SUM(elapsed_time) AS t
        FROM training_logs
        GROUP BY log_file
    """)
    rows = cur.fetchall()

    entries = []
    for log_file, total_time in rows:
        model = log_file
        if model.endswith("_train.txt"):
            model = model[:-10]  # strip "_train.txt"
        entries.append((model, log_file, round(total_time or 0.0, 2)))

    cur.executemany(
        "INSERT INTO training_runs (model, log_file, total_train_time) VALUES (?, ?, ?)",
        entries
    )
    conn.commit()
    conn.close()
    _tee_print(f"✅ Inserted/updated {len(entries)} rows into training_runs", log_fp)

def create_model_summary_table(db_path: str, log_fp=None) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS model_summary")
    cur.execute("""
        CREATE TABLE model_summary (
            dataset TEXT NOT NULL,
            model   TEXT NOT NULL,
            avg     REAL,
            median  REAL,
            min     REAL,
            max     REAL,
            std     REAL,
            train_time REAL,
            perf_per_time REAL,
            PRIMARY KEY (dataset, model)
        )
    """)
    conn.commit()
    conn.close()
    _tee_print("✅ Created table: model_summary", log_fp)

# ---------- core: collect, enrich, summarize (scoped by dataset) ----------

def collect_and_store_results(confusion_matrices_root_dir: str, db_path: str, dataset: str, log_fp=None) -> None:
    conn = initialize_db(db_path)
    # ew. migracja kolumny dataset jeśli starsza baza
    _ensure_column(conn, "evaluations", "dataset", "TEXT")

    raw_results: List[Tuple[str, str, float]] = []

    if not os.path.isdir(confusion_matrices_root_dir):
        _tee_print(f"⚠️ Not a directory: {confusion_matrices_root_dir}", log_fp)
        conn.close()
        return

    model_dirs = [d for d in os.listdir(confusion_matrices_root_dir)
                  if os.path.isdir(os.path.join(confusion_matrices_root_dir, d))]

    for model_dir in model_dirs:
        model_path = os.path.join(confusion_matrices_root_dir, model_dir)
        for test_subdir in os.listdir(model_path):
            test_path = os.path.join(model_path, test_subdir)
            cm_file = os.path.join(test_path, "confusion_matrix.npy")
            if not os.path.exists(cm_file):
                continue
            try:
                cm = np.load(cm_file)
                acc = calculate_accuracy(cm)
                raw_results.append((model_dir, test_subdir, acc))
            except Exception as e:
                _tee_print(f"⚠️ Failed to load {cm_file}: {e}", log_fp)

    # grupowanie do obliczeń statystyk po modelu (w obrębie tego datasetu)
    grouped: Dict[str, List[float]] = defaultdict(list)
    for model, _, acc in raw_results:
        grouped[model].append(acc)

    cur = conn.cursor()

    # enrich + insert (dataset scoped)
    to_insert = []
    for model, test_case, acc in raw_results:
        acc_list = grouped[model]
        avg_ = round(mean(acc_list), 6)
        med_ = round(median(acc_list), 6)
        min_ = round(min(acc_list), 6)
        max_ = round(max(acc_list), 6)
        std_ = round(stdev(acc_list), 6) if len(acc_list) > 1 else 0.0

        # training time (jeśli istnieje w training_runs)
        cur.execute("SELECT total_train_time FROM training_runs WHERE model = ?", (model,))
        row = cur.fetchone()
        train_time = float(row[0]) if row else None
        perf = (acc / train_time) if (train_time and train_time > 0) else None

        to_insert.append((
            model, test_case, acc,
            avg_, med_, min_, max_, std_,
            train_time, perf, dataset
        ))

    cur.executemany("""
        INSERT INTO evaluations (
            model, test_case, accuracy,
            avg, median, min, max, std,
            train_time, perf_per_time, dataset
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, to_insert)
    conn.commit()
    conn.close()

    _tee_print(f"✅ Saved {len(to_insert)} entries to {db_path} (dataset={dataset})", log_fp)

def compute_and_insert_model_summaries(db_path: str, dataset: str, log_fp=None) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Accuracy rows strictly for this dataset
    cur.execute("""
        SELECT model, accuracy
        FROM evaluations
        WHERE dataset = ?
    """, (dataset,))
    rows = cur.fetchall()

    acc_by_model = defaultdict(list)
    for model, acc in rows:
        if acc is not None:
            acc_by_model[model].append(float(acc))

    data_to_insert = []
    for model, accs in acc_by_model.items():
        if not accs:
            continue
        avg_acc = mean(accs)
        med_acc = median(accs)
        min_acc = min(accs)
        max_acc = max(accs)
        std_acc = stdev(accs) if len(accs) > 1 else 0.0

        # training time (global per model)
        cur.execute("SELECT total_train_time FROM training_runs WHERE model = ?", (model,))
        r = cur.fetchone()
        train_time = float(r[0]) if r else None
        perf_ratio = (avg_acc / train_time) if (train_time and train_time > 0) else None

        data_to_insert.append((dataset, model, avg_acc, med_acc, min_acc, max_acc, std_acc, train_time, perf_ratio))

    cur.executemany("""
        INSERT OR REPLACE INTO model_summary
        (dataset, model, avg, median, min, max, std, train_time, perf_per_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    conn.commit()
    conn.close()

    _tee_print(f"✅ Inserted {len(data_to_insert)} model summary rows (dataset={dataset})", log_fp)
#
# # ---------- reporting (scoped) ----------
#
# def _print_top(cur: sqlite3.Cursor, dataset: str, log_fp=None, top: int = 150) -> None:
#     _tee_print("\n📊 Average accuracy per model (scoped):", log_fp)
#     where_sql, params = _dataset_where_sql("model", dataset)
#     cur.execute(f"""
#         SELECT model, ROUND(AVG(accuracy), 6) AS avg_accuracy
#         FROM evaluations
#         WHERE {where_sql} OR dataset = ?
#         GROUP BY model
#         ORDER BY avg_accuracy DESC
#         LIMIT {top}
#     """, params + [dataset])
#     for model, avg_acc in cur.fetchall():
#         _tee_print(f"🧠 {model}: {avg_acc}", log_fp)
#
# def _print_top_speed(cur: sqlite3.Cursor, dataset: str, log_fp=None, top: int = 150) -> None:
#     _tee_print("\n⚡ Performance per training time (acc/sec) — scoped:", log_fp)
#     where_sql, params = _dataset_where_sql("model", dataset)
#     cur.execute(f"""
#         SELECT model, ROUND(AVG(perf_per_time), 6) AS avg_perf
#         FROM evaluations
#         WHERE ( {where_sql} OR dataset = ? ) AND perf_per_time IS NOT NULL
#         GROUP BY model
#         ORDER BY avg_perf DESC
#         LIMIT {top}
#     """, params + [dataset])
#     for model, avg_perf in cur.fetchall():
#         _tee_print(f"🚀 {model}: {avg_perf} acc/sec", log_fp)
#
# def _print_arch_winners(cur: sqlite3.Cursor, dataset: str, log_fp=None) -> None:
#     _tee_print("\n🔎 Best by architecture (avg accuracy):", log_fp)
#     arches = ["cyresnet56", "resnet56", "cyvgg19", "vgg19"]
#     where_sql_ds, params_ds = _dataset_where_sql("model", dataset)
#     for arch in arches:
#         cur.execute(f"""
#             SELECT model, ROUND(AVG(accuracy), 6) AS avg_acc
#             FROM evaluations
#             WHERE ({where_sql_ds} OR dataset = ?)
#               AND LOWER(model) LIKE ?
#             GROUP BY model
#             ORDER BY avg_acc DESC
#             LIMIT 1
#         """, params_ds + [dataset, f"%{arch}%"])
#         row = cur.fetchone()
#         if row:
#             _tee_print(f"🏆 {arch}: {row[0]}  → {row[1]}", log_fp)
#         else:
#             _tee_print(f"🏆 {arch}: (no data for {dataset})", log_fp)
#
# def query_best_models(db_path: str, dataset: str, log_fp=None) -> None:
#     conn = sqlite3.connect(db_path)
#     cur = conn.cursor()
#
#     _print_top(cur, dataset, log_fp)
#     _print_top_speed(cur, dataset, log_fp)
#     _print_arch_winners(cur, dataset, log_fp)
#
#     conn.close()
from statistics import mean, median, stdev
from collections import defaultdict

def _print_extended_plain(cur: sqlite3.Cursor, dataset: str, log_fp=None, top: int = 50) -> None:
    """
    Extended stats per model (scoped to dataset) – accuracy only:
      avg, median, min, max, std
    """
    # Pull all accuracies for this dataset
    cur.execute("""
        SELECT model, accuracy
        FROM evaluations
        WHERE dataset = ?
        ORDER BY model
    """, (dataset,))
    rows = cur.fetchall()

    acc_map = defaultdict(list)
    for model, acc in rows:
        if acc is not None:
            acc_map[model].append(float(acc))

    # Compute per-model stats
    summary = []
    for model, accs in acc_map.items():
        if not accs:
            continue
        avg_acc = mean(accs)
        med_acc = median(accs)
        min_acc = min(accs)
        max_acc = max(accs)
        std_acc = stdev(accs) if len(accs) > 1 else 0.0
        summary.append({
            "model": model,
            "avg": avg_acc,
            "median": med_acc,
            "min": min_acc,
            "max": max_acc,
            "std": std_acc,
        })

    summary.sort(key=lambda x: x["avg"], reverse=True)

    _tee_print("\n📈 Extended stats per model (scoped, accuracy only):", log_fp)
    for s in summary[:top]:
        _tee_print(
            f"🧪 {s['model']}: "
            f"avg={s['avg']:.4f}, median={s['median']:.4f}, "
            f"min={s['min']:.4f}, max={s['max']:.4f}, std={s['std']:.4f}",
            log_fp
        )


def _print_top(cur: sqlite3.Cursor, dataset: str, log_fp=None, top: int = 500) -> None:
    _tee_print("\n📊 Average accuracy per model (scoped):", log_fp)
    cur.execute("""
        SELECT model, ROUND(AVG(accuracy), 6) AS avg_accuracy
        FROM evaluations
        WHERE dataset = ?
        GROUP BY model
        ORDER BY avg_accuracy DESC
        LIMIT ?
    """, (dataset, top))
    for model, avg_acc in cur.fetchall():
        _tee_print(f"🧠 {model}: {avg_acc}", log_fp)

def _print_top_speed(cur: sqlite3.Cursor, dataset: str, log_fp=None, top: int = 500) -> None:
    _tee_print("\n⚡ Performance per training time (acc/sec) — scoped:", log_fp)
    cur.execute("""
        SELECT model, ROUND(AVG(perf_per_time), 6) AS avg_perf
        FROM evaluations
        WHERE dataset = ? AND perf_per_time IS NOT NULL
        GROUP BY model
        ORDER BY avg_perf DESC
        LIMIT ?
    """, (dataset, top))
    for model, avg_perf in cur.fetchall():
        _tee_print(f"🚀 {model}: {avg_perf} acc/sec", log_fp)

def _print_arch_winners(cur: sqlite3.Cursor, dataset: str, log_fp=None) -> None:
    _tee_print("\n🔎 Best by architecture (avg accuracy):", log_fp)
    arches = ["cyresnet56", "resnet56", "cyvgg19", "vgg19"]
    for arch in arches:
        cur.execute("""
            SELECT model, ROUND(AVG(accuracy), 6) AS avg_acc
            FROM evaluations
            WHERE dataset = ? AND LOWER(model) LIKE ?
            GROUP BY model
            ORDER BY avg_acc DESC
            LIMIT 1
        """, (dataset, f"%{arch}%"))
        row = cur.fetchone()
        if row:
            _tee_print(f"🏆 {arch}: {row[0]}  → {row[1]}", log_fp)
        else:
            _tee_print(f"🏆 {arch}: (no data for {dataset})", log_fp)

# def _print_extended(cur: sqlite3.Cursor, dataset: str, log_fp=None, top: int = 50) -> None:
#     """
#     Extended stats per model (scoped): avg/median/min/max/std + train_time + avg perf_per_time.
#     """
#     _tee_print("\n📈 Extended stats per model (scoped):", log_fp)
#     cur.execute("""
#         SELECT model,
#                AVG(accuracy)           AS avg_acc,
#                MIN(accuracy)           AS min_acc,
#                MAX(accuracy)           AS max_acc,
#                AVG(perf_per_time)      AS avg_perf
#         FROM evaluations
#         WHERE dataset = ?
#         GROUP BY model
#     """, (dataset,))
#     rows = cur.fetchall()
#
#     # compute median/std in Python (SQLite doesn’t have them)
#     # fetch all accuracies per model for this dataset
#     cur.execute("""
#         SELECT model, accuracy
#         FROM evaluations
#         WHERE dataset = ?
#         ORDER BY model
#     """, (dataset,))
#     acc_rows = cur.fetchall()
#     acc_map = defaultdict(list)
#     for m, a in acc_rows:
#         if a is not None:
#             acc_map[m].append(float(a))
#
#     # fetch train_time per model
#     cur.execute("SELECT model, total_train_time FROM training_runs")
#     tt_map = dict(cur.fetchall())
#
#     summary = []
#     for model, avg_acc, min_acc, max_acc, avg_perf in rows:
#         accs = acc_map.get(model, [])
#         if not accs:
#             continue
#         med = median(accs)
#         sd  = stdev(accs) if len(accs) > 1 else 0.0
#         tt  = tt_map.get(model)
#         summary.append({
#             "model": model,
#             "avg": round(avg_acc or 0.0, 6),
#             "median": round(med, 6),
#             "min": round(min_acc or 0.0, 6),
#             "max": round(max_acc or 0.0, 6),
#             "std": round(sd, 6),
#             "train_time": round(tt, 2) if tt else None,
#             "avg_perf_per_time": round(avg_perf, 6) if avg_perf is not None else None,
#         })
#
#     # show best by avg accuracy
#     summary.sort(key=lambda x: x["avg"], reverse=True)
#     for s in summary[:top]:
#         _tee_print(
#             f"🧪 {s['model']}: avg={s['avg']}, median={s['median']}, "
#             f"min={s['min']}, max={s['max']}, std={s['std']}, "
#             f"train_time={s['train_time']}, perf/time(avg)={s['avg_perf_per_time']}",
#             log_fp
#         )
#     if summary:
#         best = summary[0]
#         _tee_print(
#             f"\n🏅 Best model (Python, by avg acc): {best['model']} "
#             f"with avg={best['avg']} (perf/time={best['avg_perf_per_time']})",
#             log_fp
#         )

def _print_extended_with_perf(cur: sqlite3.Cursor, dataset: str, log_fp=None, top: int = 500) -> None:
    """
    Extended stats per model (scoped) – accuracy stats + train_time + avg perf/time.
    """
    # Base aggregates from SQL (avg/min/max and avg perf/time)
    cur.execute("""
        SELECT model,
               AVG(accuracy)           AS avg_acc,
               MIN(accuracy)           AS min_acc,
               MAX(accuracy)           AS max_acc,
               AVG(perf_per_time)      AS avg_perf
        FROM evaluations
        WHERE dataset = ?
        GROUP BY model
    """, (dataset,))
    rows = cur.fetchall()

    # Pull all accuracies to compute median/std in Python
    cur.execute("""
        SELECT model, accuracy
        FROM evaluations
        WHERE dataset = ?
        ORDER BY model
    """, (dataset,))
    acc_rows = cur.fetchall()
    acc_map = defaultdict(list)
    for m, a in acc_rows:
        if a is not None:
            acc_map[m].append(float(a))

    # Training time per model
    cur.execute("SELECT model, total_train_time FROM training_runs")
    tt_map = dict(cur.fetchall())

    summary = []
    for model, avg_acc, min_acc, max_acc, avg_perf in rows:
        accs = acc_map.get(model, [])
        if not accs:
            continue
        med = median(accs)
        sd  = stdev(accs) if len(accs) > 1 else 0.0
        tt  = tt_map.get(model)
        summary.append({
            "model": model,
            "avg": round((avg_acc or 0.0), 6),
            "median": round(med, 6),
            "min": round((min_acc or 0.0), 6),
            "max": round((max_acc or 0.0), 6),
            "std": round(sd, 6),
            "train_time": round(tt, 2) if tt else None,
            "avg_perf_per_time": round(avg_perf, 6) if avg_perf is not None else None,
        })

    summary.sort(key=lambda x: x["avg"], reverse=True)

    _tee_print("\n📈 Extended stats per model (scoped, incl. perf/time):", log_fp)
    for s in summary[:top]:
        _tee_print(
            f"🧪 {s['model']}: "
            f"avg={s['avg']:.4f}, median={s['median']:.4f}, "
            f"min={s['min']:.4f}, max={s['max']:.4f}, std={s['std']:.4f}, "
            f"train_time={s['train_time']}, perf/time(avg)={s['avg_perf_per_time']}",
            log_fp
        )

#
# def query_best_models(db_path: str, dataset: str, log_fp=None) -> None:
#     conn = sqlite3.connect(db_path)
#     cur = conn.cursor()
#
#     _tee_print(f"\n=== DATASET: {dataset} ===", log_fp)
#     _print_top(cur, dataset, log_fp)
#     _print_top_speed(cur, dataset, log_fp)
#     _print_arch_winners(cur, dataset, log_fp)
#     _print_extended(cur, dataset, log_fp)
#
#     conn.close()

def query_best_models(db_path: str, dataset: str, log_fp=None) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    _tee_print(f"\n=== DATASET: {dataset} ===", log_fp)

    # Top by avg accuracy (scoped)
    _tee_print("\n📊 Average accuracy per model (scoped):", log_fp)
    cur.execute("""
        SELECT model, ROUND(AVG(accuracy), 6) as avg_accuracy
        FROM evaluations
        WHERE dataset = ?
        GROUP BY model
        ORDER BY avg_accuracy DESC
        LIMIT 500
    """, (dataset,))
    for model, avg_acc in cur.fetchall():
        _tee_print(f"🧠 {model}: {avg_acc}", log_fp)

    # Top by perf per time (scoped)
    _tee_print("\n⚡ Performance per training time (acc/sec) — scoped:", log_fp)
    cur.execute("""
        SELECT model, ROUND(AVG(perf_per_time), 6) as avg_perf
        FROM evaluations
        WHERE dataset = ? AND perf_per_time IS NOT NULL
        GROUP BY model
        ORDER BY avg_perf DESC
        LIMIT 500
    """, (dataset,))
    for model, avg_perf in cur.fetchall():
        _tee_print(f"🚀 {model}: {avg_perf} acc/sec", log_fp)

    # Best per architecture (scoped)
    _tee_print("\n🔎 Best by architecture (avg accuracy):", log_fp)
    for arch in ["cyresnet56", "resnet56", "cyvgg19", "vgg19"]:
        cur.execute("""
            SELECT model, ROUND(AVG(accuracy), 6) AS avg_acc
            FROM evaluations
            WHERE dataset = ? AND LOWER(model) LIKE ?
            GROUP BY model
            ORDER BY avg_acc DESC
            LIMIT 1
        """, (dataset, f"%{arch}%"))
        row = cur.fetchone()
        msg = f"🏆 {arch}: {row[0]}  → {row[1]}" if row else f"🏆 {arch}: (no data for {dataset})"
        _tee_print(msg, log_fp)

    # >>> BOTH extended blocks <<<
    _print_extended_plain(cur, dataset, log_fp, top=500)         # accuracy-only (your old style)
    _print_extended_with_perf(cur, dataset, log_fp, top=500)     # with train_time + perf/time

    conn.close()
