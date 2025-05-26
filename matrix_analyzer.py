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
            accuracy REAL NOT NULL
        )
    ''')
    conn.commit()
    return conn

def insert_results(conn, results):
    cursor = conn.cursor()
    cursor.executemany('''
        INSERT INTO evaluations (model, test_case, accuracy)
        VALUES (?, ?, ?)
    ''', results)
    conn.commit()

def collect_and_store_results(confusion_matrices_root_dir, db_path):
    conn = initialize_db(db_path)
    results = []

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
                results.append((model_dir, test_subdir, acc))
            except Exception as e:
                print(f"⚠️ Failed to load {cm_file}: {e}")

    insert_results(conn, results)
    conn.close()
    print(f"✅ Saved {len(results)} entries to {db_path}")

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
    print(f"\n🏆 Best model (SQL): {best[0]} with avg accuracy: {best[1]}")

    # 🧠 Extended Python-side statistics
    cursor.execute('SELECT model, accuracy FROM evaluations')
    all_rows = cursor.fetchall()
    conn.close()

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

# === USAGE ===

confusion_matrices_root_dir = r'\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs\json_4_copy\confusion_matrices'
db_path = "confusion_results.db"

collect_and_store_results(confusion_matrices_root_dir, db_path)
query_best_models(db_path)
