from pathlib import Path
import os
import re
import matplotlib.pyplot as plt

from src.utils.db_handler import init_database, insert_training_logs, insert_test_logs, insert_training_run
from src.utils.wsl_handler import sync_wsl_logs

import ast


_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TRAIN_RE = re.compile(rf"\[Epoch\s+(\d+)\]\s+Train Loss:\s+({_FLOAT_RE})")
_VAL_RE = re.compile(rf"Validation loss:\s+({_FLOAT_RE}),\s+Accuracy:\s+(\d+)/(\d+)\s+\(({_FLOAT_RE})%\)")
_TEST_RE = re.compile(rf"Test loss:\s+({_FLOAT_RE}),\s+Accuracy:\s+(\d+)/(\d+)\s+\(({_FLOAT_RE})%\)")
_ELAPSED_RE = re.compile(rf"Elapsed time:\s+({_FLOAT_RE})\s+sec")


def extract_config(line):
    if not isinstance(line, str):
        return {}
    match = re.search(r"configuration:\s+({.*})", line)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return {}


def parse_filename(filename):
    name = os.path.basename(filename).replace('_train.txt', '').replace('.txt', '')
    if "_test_on_" in name:
        train_part, test_part = name.split("_test_on_", maxsplit=1)
        return train_part, test_part
    return name, ''


def _read_log_lines(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def _base_log_metadata(filepath, lines):
    config_line = next((line for line in lines if line.startswith("configuration:")), None)
    config = extract_config(config_line)
    dataset, augmentation = parse_filename(filepath)

    return {
        'model_id': config.get('model', ''),
        'log_file': os.path.basename(filepath),
        'dataset': dataset,
        'augmentation_info': augmentation,
        'transform': config.get('polar_transform', ''),
        'batch_size': config.get('batch_size', None),
        'lr': config.get('lr', None),
    }


def parse_log_file(filepath):
    """
    Parse a training log without assuming that validation and elapsed-time lines
    are located at fixed offsets after the epoch line.

    Expected fragments can appear with extra logging between them, for example:
      [Epoch 1] Train Loss: ...
      ... optional extra lines ...
      Validation loss: ..., Accuracy: ...
      ... optional extra lines ...
      Elapsed time: ... sec
    """
    lines = _read_log_lines(filepath)

    config_line = next((line for line in lines if line.startswith("configuration:")), None)
    if not config_line:
        return []

    metadata = _base_log_metadata(filepath, lines)

    total_elapsed_time = 0.0
    rows = []
    current_row = None

    def flush_current_row():
        if current_row is not None:
            rows.append(current_row)

    for line in lines:
        train_match = _TRAIN_RE.search(line)
        if train_match:
            flush_current_row()
            current_row = {
                **metadata,
                'epoch': int(train_match.group(1)),
                'train_loss': float(train_match.group(2)),
                'val_loss': None,
                'accuracy': None,
                'elapsed_time': None,
            }
            continue

        if current_row is None:
            continue

        val_match = _VAL_RE.search(line)
        if val_match:
            current_row['val_loss'] = float(val_match.group(1))
            current_row['accuracy'] = float(val_match.group(4))
            continue

        elapsed_match = _ELAPSED_RE.search(line)
        if elapsed_match:
            elapsed = float(elapsed_match.group(1))
            current_row['elapsed_time'] = elapsed
            total_elapsed_time += elapsed
            continue

    flush_current_row()

    for row in rows:
        row['total_train_time'] = total_elapsed_time

    return rows


def parse_test_log_file(filepath):
    lines = _read_log_lines(filepath)

    metadata = _base_log_metadata(filepath, lines)

    test_line = next((line for line in lines if line.startswith("Test loss:")), None)
    if not test_line:
        return []

    match = _TEST_RE.search(test_line)
    if not match:
        return []

    return [{
        **metadata,
        'test_loss': float(match.group(1)),
        'accuracy': float(match.group(4)),
        'total': int(match.group(3)),
        'correct': int(match.group(2))
    }]


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

    if file_path:
        filename = Path(file_path).with_suffix(".png").name
        dataset_prefix = filename.split("-")[0].lower()  # e.g., "mnist" from "mnist-custom-..."
        out_dir = Path("results/plots") / dataset_prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / filename
        plt.savefig(save_path)
        print(f"🖼️ Saved plot to: {save_path}")

    plt.close()


def collect_log_files(log_path):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(log_path)
        if "launcher_" not in root
        for file in files
        if file.endswith(".txt") and not file.startswith("launcher_")
    ]


def ingest_logs(
        wsl_source: str,
        local_logs_path: str,
        db_path: str,
        overwrite_logs=False,
        overwrite_db=False
):
    print("🔄 Syncing logs from WSL...")
    sync_wsl_logs(wsl_source, local_logs_path, overwrite=overwrite_logs)

    if not os.path.exists(db_path):
        print("🗂️ Creating database...")
        init_database(db_path)

    log_files = collect_log_files(local_logs_path)
    log_files = [p for p in log_files if "robocopy" not in p.lower()]

    for file_path in log_files:
        print(f"\n📂 Processing: {file_path}")
        parsed_data = parse_log_file(file_path)

        if parsed_data:
            plot_metrics(parsed_data, file_path)
            insert_training_logs(parsed_data, db_path, overwrite=overwrite_db)
            insert_training_run(parsed_data, db_path, overwrite=overwrite_db)
        else:
            test_data = parse_test_log_file(file_path)
            if test_data:
                insert_test_logs(test_data, db_path, overwrite=overwrite_db)
            else:
                print("⚠️ No data found in file.")
