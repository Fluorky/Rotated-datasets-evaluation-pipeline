import os
import numpy as np
import sqlite3
from tqdm import tqdm
from statistics import mean, median, stdev
from collections import defaultdict


def calculate_accuracy(conf_matrix):
    correct = np.trace(conf_matrix)
    total = conf_matrix.sum()
    return correct / total if total > 0 else 0.0


def initialize_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    return conn


def create_training_runs_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS training_runs')
    cursor.execute('''
        CREATE TABLE training_runs (
            model TEXT PRIMARY KEY,
            log_file TEXT,
            total_train_time REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Created table: training_runs")


def insert_results(conn, results):
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT INTO evaluations (
            model, test_case, accuracy,
            avg, median, min, max, std,
            train_time, perf_per_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', results)
    conn.commit()


def collect_and_store_results(confusion_matrices_root_dir, db_path):
    conn = initialize_db(db_path)
    raw_results = []

    for model_dir in tqdm(os.listdir(confusion_matrices_root_dir), desc="📁 Scanning models"):
        model_path = os.path.join(confusion_matrices_root_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

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
                print(f"⚠️ Failed to load {cm_file}: {e}")

    grouped = defaultdict(list)
    for model, _, acc in raw_results:
        grouped[model].append(acc)

    full_rows = []
    conn_check = sqlite3.connect(db_path)
    cursor = conn_check.cursor()

    for model, test_case, acc in raw_results:
        acc_list = grouped[model]
        avg_ = round(mean(acc_list), 4)
        median_ = round(median(acc_list), 4)
        min_ = round(min(acc_list), 4)
        max_ = round(max(acc_list), 4)
        std_ = round(stdev(acc_list), 4) if len(acc_list) > 1 else 0.0

        cursor.execute("SELECT total_train_time FROM training_runs WHERE model = ?", (model,))
        result = cursor.fetchone()
        train_time = result[0] if result else None
        perf_per_time = acc / train_time if train_time and train_time > 0 else None

        full_rows.append((
            model, test_case, acc,
            avg_, median_, min_, max_, std_,
            train_time, perf_per_time
        ))

    conn_check.close()
    insert_results(conn, full_rows)
    conn.close()
    print(f"✅ Saved {len(full_rows)} enriched entries to {db_path}")


def compute_training_times(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT log_file, SUM(elapsed_time) FROM training_logs GROUP BY log_file')
    rows = cursor.fetchall()

    entries = []
    for log_file, total_time in rows:
        if log_file.endswith("_train.txt"):
            model = log_file.replace("_train.txt", "")
        else:
            model = log_file
        entries.append((model, log_file, round(total_time, 2)))

    cursor.executemany(
        'INSERT INTO training_runs (model, log_file, total_train_time) VALUES (?, ?, ?)',
        entries
    )
    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(entries)} rows into training_runs")


def create_model_summary_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS model_summary')
    cursor.execute('''
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
    conn.commit()
    conn.close()
    print("✅ Created table: model_summary")


def compute_and_insert_model_summaries(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT model, accuracy FROM evaluations')
    rows = cursor.fetchall()

    acc_by_model = defaultdict(list)
    for model, acc in rows:
        acc_by_model[model].append(acc)

    data_to_insert = []

    for model, accs in acc_by_model.items():
        accs_clean = [a for a in accs if a is not None]
        if not accs_clean:
            continue
        avg_acc = mean(accs_clean)
        med_acc = median(accs_clean)
        min_acc = min(accs_clean)
        max_acc = max(accs_clean)
        std_acc = stdev(accs_clean) if len(accs_clean) > 1 else 0.0

        cursor.execute('SELECT total_train_time FROM training_runs WHERE model = ?', (model,))
        result = cursor.fetchone()
        if not result:
            print(f"❌ No training time found for model: {model}")
            continue
        train_time = result[0]
        perf_ratio = avg_acc / train_time if train_time else None

        data_to_insert.append((model, avg_acc, med_acc, min_acc, max_acc, std_acc, train_time, perf_ratio))

    cursor.executemany('''
        INSERT INTO model_summary (
            model, avg, median, min, max, std, train_time, perf_per_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    cursor.execute('DROP TABLE IF EXISTS training_runs')
    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(data_to_insert)} model summary rows")


def query_best_models(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT model, ROUND(AVG(accuracy), 4) as avg_accuracy
        FROM evaluations
        GROUP BY model
        ORDER BY avg_accuracy DESC
    ''')

    print("\n📊 Average accuracy per model (SQL):")
    for row in cursor.fetchall():
        print(f"🧠 {row[0]}: {row[1]}")

    cursor.execute('''
        SELECT model, ROUND(AVG(accuracy), 4) as avg_accuracy
        FROM evaluations
        GROUP BY model
        ORDER BY avg_accuracy DESC
        LIMIT 1
    ''')
    best = cursor.fetchone()
    print(f"\n🏆 Best model (by accuracy): {best[0]} with avg accuracy: {best[1]}")

    # === EXTENDED ===
    cursor.execute('SELECT model, accuracy FROM evaluations')
    all_rows = cursor.fetchall()

    model_stats = defaultdict(list)
    for model, acc in all_rows:
        model_stats[model].append(acc)

    print("\n📈 Extended stats per model:")
    summary = []
    for model, acc_list in model_stats.items():
        stats = {
            'model': model,
            'avg': round(mean(acc_list), 4),
            'median': round(median(acc_list), 4),
            'min': round(min(acc_list), 4),
            'max': round(max(acc_list), 4),
            'std': round(stdev(acc_list), 4) if len(acc_list) > 1 else 0.0
        }
        summary.append(stats)

    summary.sort(key=lambda x: x['avg'], reverse=True)

    for s in summary:
        print(f"🧪 {s['model']}: avg={s['avg']}, median={s['median']}, min={s['min']}, max={s['max']}, std={s['std']}")

    best_model = summary[0]
    print(f"\n🏅 Best model (Python): {best_model['model']} with avg accuracy = {best_model['avg']}")

    cursor.execute('''
        SELECT model, ROUND(AVG(perf_per_time), 6) as avg_perf
        FROM evaluations
        WHERE perf_per_time IS NOT NULL
        GROUP BY model
        ORDER BY avg_perf DESC
    ''')
    ranked = cursor.fetchall()

    print("\n⚡ Performance per training time (accuracy/sec):")
    for row in ranked:
        print(f"🚀 {row[0]}: {row[1]} acc/sec")

    if ranked:
        print(f"\n🥇 Best model (by speed-efficiency): {ranked[0][0]} with {ranked[0][1]} acc/sec")

    conn.close()
