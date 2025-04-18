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


def init_database(db_path='mnist_logs.db'):
    add_test_logs_table(db_path)
    add_training_logs_table(db_path)


def insert_training_logs(data, db_path='mnist_logs.db', overwrite=False):
    if not data:
        print("⚠️ No training data to insert.")
        return

    log_file = data[0]['log_file']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM training_logs WHERE log_file = ?', (log_file,))
    existing_count = cursor.fetchone()[0]

    if existing_count > 0:
        if overwrite:
            print(f"Overwriting existing training logs: {log_file} ({existing_count} rows)")
            cursor.execute('DELETE FROM training_logs WHERE log_file = ?', (log_file,))
        else:
            print(f"⏩ Skipped training logs: '{log_file}' already in database")
            conn.close()
            return

    for row in data:
        cursor.execute('''
            INSERT INTO training_logs (
                model_id, log_file, dataset, augmentation_info,
                transform, batch_size, lr, epoch,
                train_loss, val_loss, accuracy, elapsed_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['model_id'], row['log_file'], row['dataset'], row['augmentation_info'],
            row['transform'], row['batch_size'], row['lr'], row['epoch'],
            row['train_loss'], row['val_loss'], row['accuracy'], row['elapsed_time']
        ))

    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(data)} row(s) to training_logs for: {log_file}")


def insert_test_logs(data, db_path='mnist_logs.db', overwrite=False):
    if not data:
        print("⚠️ No test data to insert.")
        return

    log_file = data[0]['log_file']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM test_logs WHERE log_file = ?', (log_file,))
    exists = cursor.fetchone()[0]

    if exists > 0:
        if overwrite:
            print(f"Overwriting test log: {log_file}")
            cursor.execute('DELETE FROM test_logs WHERE log_file = ?', (log_file,))
        else:
            print(f"⏩ Skipped test log: {log_file}")
            conn.close()
            return

    for row in data:
        cursor.execute('''
            INSERT INTO test_logs (
                model_id, log_file, dataset, augmentation_info, transform,
                batch_size, lr, test_loss, accuracy, correct, total
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['model_id'], row['log_file'], row['dataset'], row['augmentation_info'], row['transform'],
            row['batch_size'], row['lr'], row['test_loss'], row['accuracy'], row['correct'], row['total']
        ))

    conn.commit()
    conn.close()
    print(f"✅ Inserted test log for: {log_file}")


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
