import os
import re

import matplotlib.pyplot as plt

from db_handler import init_database, insert_training_logs, insert_test_logs
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
            val_loss = float(val_match.group(1)) if val_match else None
            accuracy = float(val_match.group(4)) if val_match else None

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
        return []

    match = re.search(r'Test loss: ([\d.]+), Accuracy: (\d+)/(\d+) \(([\d.]+)%\)', test_line)
    if not match:
        return []

    return [{
        'model_id': model_id,
        'log_file': os.path.basename(filepath),
        'dataset': dataset,
        'augmentation_info': augmentation,
        'transform': transform,
        'batch_size': batch_size,
        'lr': lr,
        'test_loss': float(match.group(1)),
        'accuracy': float(match.group(4)),
        'total': int(match.group(3)),
        'correct': int(match.group(2))
    }]


def collect_log_files(log_path):
    log_files = []
    for root, _, files in os.walk(log_path):
        if "launcher_" in root:
            continue  # Skip directories with "launcher_"

        for file in files:
            if file.startswith("launcher_") or not file.endswith('.txt'):
                continue
            log_files.append(os.path.join(root, file))
    return log_files


def main():
    print("🔄 Syncing logs from WSL...")
    sync_wsl_logs(wsl_logs_source, local_logs_folder, overwrite=overwrite_logs)

    if not os.path.exists(db_file):
        print("🗂️ Creating database...")
        init_database(db_file)

    log_files = collect_log_files(local_logs_folder)

    for file_path in log_files:
        print(f"\n📂 Processing: {file_path}")
        parsed_data = parse_log_file(file_path)

        if parsed_data:
            plot_metrics(parsed_data, file_path)
            insert_training_logs(parsed_data, db_file, overwrite=overwrite_existing)
        else:
            test_data = parse_test_log_file(file_path)
            if test_data:
                insert_test_logs(test_data, db_file, overwrite=overwrite_existing)
            else:
                print("⚠️ No data found in file.")


if __name__ == "__main__":
    # === Config ===
    wsl_logs_source = r'\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\logs'
    local_logs_folder = 'log_files_from_slave/logs'
    db_file = 'mnist_logs.db'

    overwrite_logs = False  # Whether to overwrite files during WSL sync
    overwrite_existing = False  # Whether to overwrite existing DB entries

    main()
