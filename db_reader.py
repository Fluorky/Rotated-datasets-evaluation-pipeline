import pandas as pd
import sqlite3


def add_test_logs_table(db_path='mnist_logs.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            log_file TEXT,
            dataset TEXT,
            augmentation_info TEXT,
            transform TEXT,
            batch_size INTEGER,
            lr REAL,
            test_loss REAL,
            accuracy REAL,
            correct INTEGER,
            total INTEGER
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Table `test_logs` added or already existed.")


def add_training_logs_table(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            log_file TEXT,
            dataset TEXT,
            augmentation_info TEXT,
            transform TEXT,
            batch_size INTEGER,
            lr REAL,
            epoch INTEGER,
            train_loss REAL,
            val_loss REAL,
            accuracy REAL,
            elapsed_time REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Table `training_logs` added or already existed.")


def drop_table(table_name, db_path='mnist_logs.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
    conn.commit()
    conn.close()
    print(f"🗑️ Dropped table `{table_name}` (if it existed).")

def show_table(table_name='training_logs', db_path='mnist_logs.db'):
    try:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 0)
        pd.set_option('display.max_colwidth', None)

        conn = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()

        print(f"\n📋 Showing table: {table_name}")
        print(df)

    except Exception as e:
        print(f"❌ Error reading table `{table_name}`: {e}")

show_table()
show_table("training_logs")
show_table("test_logs")