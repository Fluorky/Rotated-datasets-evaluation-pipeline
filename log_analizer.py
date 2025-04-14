import matplotlib.pyplot as plt
import sqlite3
import shutil
import os
import re
from wsl_handler import sync_wsl_logs

def extract_config(line):
    match = re.search(r"configuration:\s+({.*})", line)
    return eval(match.group(1)) if match else {}


def parse_filename(filename):
    name = os.path.basename(filename).replace('_train.txt', '').replace('.txt', '')
    parts = name.split('-')
    dataset = parts[0] if parts else ''
    augmentation = '-'.join(parts[1:]) if len(parts) > 1 else ''
    return dataset, augmentation


def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    config_line = next((line for line in lines if line.startswith("configuration:")), None)
    config = extract_config(config_line)
    dataset, augmentation = parse_filename(filepath)

    model_id = config.get('model', '')
    transform = config.get('polar_transform', '')
    batch_size = config.get('batch_size', None)
    lr = config.get('lr', None)

    rows = []
    for i, line in enumerate(lines):
        match = re.match(r'\[Epoch (\d+)\] Train Loss: ([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))

            val_line = lines[i + 1]
            val_match = re.search(r'Validation loss: ([\d.]+), Accuracy: (\d+)/(\d+) \(([\d.]+)%\)', val_line)
            val_loss = float(val_match.group(1))
            accuracy = float(val_match.group(4))

            time_line = lines[i + 2] if (i + 2) < len(lines) else ''
            elapsed_match = re.search(r'Elapsed time: ([\d.]+) sec', time_line)
            elapsed = float(elapsed_match.group(1)) if elapsed_match else None

            rows.append({
                'model_id': model_id,
                'log_file': os.path.basename(filepath),
                'dataset': dataset,
                'augmentation_info': augmentation,
                'transform': transform,
                'batch_size': batch_size,
                'lr': lr,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'elapsed_time': elapsed
            })

    return rows


def plot_metrics(data, file_path=None):
    epochs = [d['epoch'] for d in data]
    train_loss = [d['train_loss'] for d in data]
    val_loss = [d['val_loss'] for d in data]
    accuracy = [d['accuracy'] for d in data]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, accuracy, label='Accuracy (%)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Validation Accuracy over Epochs')

    plt.tight_layout()

    # === Save to file if file_path is given ===
    if file_path:
        os.makedirs("plots", exist_ok=True)
        filename = os.path.basename(file_path).replace(".txt", ".png")
        save_path = os.path.join("plots", filename)
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

    plt.close()  # close plot to avoid displaying inline if not needed


def init_db(db_path='mnist_logs.db'):
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

def parse_test_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    config_line = next((line for line in lines if line.startswith("configuration:")), None)
    config = extract_config(config_line)
    dataset, augmentation = parse_filename(filepath)

    model_id = config.get('model', '')
    transform = config.get('polar_transform', '')
    batch_size = config.get('batch_size', None)
    lr = config.get('lr', None)

    test_line = next((line for line in lines if line.startswith("Test loss:")), None)
    if not test_line:
        return []  # nie ma testu

    match = re.search(r'Test loss: ([\d.]+), Accuracy: (\d+)/(\d+) \(([\d.]+)%\)', test_line)
    if not match:
        return []

    test_loss = float(match.group(1))
    correct = int(match.group(2))
    total = int(match.group(3))
    accuracy = float(match.group(4))

    return [{
        'model_id': model_id,
        'log_file': os.path.basename(filepath),
        'dataset': dataset,
        'augmentation_info': augmentation,
        'transform': transform,
        'batch_size': batch_size,
        'lr': lr,
        'test_loss': test_loss,
        'accuracy': accuracy,
        'total': total,
        'correct': correct
    }]


def save_to_sqlite(data, db_path='mnist_logs.db', overwrite=False):
    if not data:
        print("⚠️ No data to insert.")
        return

    log_file = data[0]['log_file']

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM training_logs WHERE log_file = ?', (log_file,))
    existing_count = cursor.fetchone()[0]

    if existing_count > 0:
        if overwrite:
            print(f"Overwriting existing entries for log file: {log_file} ({existing_count} rows)")
            cursor.execute('DELETE FROM training_logs WHERE log_file = ?', (log_file,))
        else:
            print(f"Skipped: Log file '{log_file}' already exists in database ({existing_count} rows)")
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
    print(f"Inserted {len(data)} row(s) for log file: {log_file}")

def save_test_to_sqlite(data, db_path='mnist_logs.db', overwrite=False):
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
            print(f"Skipped test log: {log_file}")
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
    print(f"Inserted test log for: {log_file}")


def collect_log_files(log_path):
    if os.path.isfile(log_path):
        return [log_path]

    log_files = []
    for root, _, files in os.walk(log_path):
        for file in files:
            if file.endswith('.txt'):
                log_files.append(os.path.join(root, file))
    return log_files


# ===  ===

# WSL logs source and local logs destination

wsl_logs_source = r'\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs'
local_logs_folder = 'log_files_from_slave/logs'
overwrite_logs = False  # Set to True if you want to overwrite existing files

print("Syncing WSL logs...")

sync_wsl_logs(wsl_logs_source, local_logs_folder, overwrite=overwrite_logs)

db_file = 'mnist_logs.db'
overwrite_existing = False

if not os.path.exists(db_file):
    print("Creating database...")
    init_db(db_file)

log_files = collect_log_files(local_logs_folder)

for file_path in log_files:
    print(f"\nProcessing: {file_path}")
    parsed_data = parse_log_file(file_path)
    if parsed_data:
        plot_metrics(parsed_data, file_path)
        save_to_sqlite(parsed_data, db_file, overwrite=overwrite_existing)
    else:
        test_data = parse_test_log_file(file_path)
        if test_data:
            save_test_to_sqlite(test_data, db_file, overwrite=overwrite_existing)
        else:
            print("⚠️ Nothing to insert from this file.")
