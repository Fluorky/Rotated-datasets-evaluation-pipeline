import matplotlib.pyplot as plt
import sqlite3
import os
import re


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


def plot_metrics(data):
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
    plt.show()


def init_db(db_path='training_logs.db'):
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


def save_to_sqlite(data, db_path='training_logs.db', overwrite=False):
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


def collect_log_files(log_path):
    if os.path.isfile(log_path):
        return [log_path]

    log_files = []
    for root, _, files in os.walk(log_path):
        for file in files:
            if file.endswith('.txt'):
                log_files.append(os.path.join(root, file))
    return log_files


# === USAGE ===

log_path = 'log_files_from_slave/logs'  # File or folder
db_file = 'training_logs.db'
overwrite_existing = False

if not os.path.exists(db_file):
    print("Creating database...")
    init_db(db_file)

log_files = collect_log_files(log_path)

for file_path in log_files:
    print(f"\nProcessing: {file_path}")
    parsed_data = parse_log_file(file_path)
    plot_metrics(parsed_data)
    save_to_sqlite(parsed_data, db_file, overwrite=overwrite_existing)
