import re
import matplotlib.pyplot as plt


# Load and parse log
def parse_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    epoch_data = []

    for i, line in enumerate(lines):
        match = re.match(r'\[Epoch (\d+)\] Train Loss: ([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))

            # Expect the next line to contain validation metrics
            val_line = lines[i + 1]
            val_match = re.search(r'Validation loss: ([\d.]+), Accuracy: (\d+)/(\d+) \(([\d.]+)%\)', val_line)
            val_loss = float(val_match.group(1))
            correct = int(val_match.group(2))
            total = int(val_match.group(3))
            acc = float(val_match.group(4))

            # Look for elapsed time
            elapsed_match = re.search(r'Elapsed time: ([\d.]+) sec', lines[i + 2])
            elapsed = float(elapsed_match.group(1)) if elapsed_match else None

            epoch_data.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': acc,
                'elapsed_time': elapsed
            })

    return epoch_data


# Plotting utility
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


# Usage
log_file_path = 'log_files_from_slave/train_results.txt'
data = parse_log(log_file_path)
plot_metrics(data)
