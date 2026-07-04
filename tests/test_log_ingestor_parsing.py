from src.analysis.log_ingestor import parse_filename, parse_log_file, parse_test_log_file


def test_parse_filename_for_train_log():
    dataset, augmentation = parse_filename("/tmp/MNIST-CyCNN_train.txt")

    assert dataset == "MNIST-CyCNN"
    assert augmentation == ""


def test_parse_filename_for_test_log():
    dataset, augmentation = parse_filename("/tmp/MNIST-CyCNN_test_on_rotated-30.txt")

    assert dataset == "MNIST-CyCNN"
    assert augmentation == "rotated-30"


def test_parse_training_log_tolerates_extra_lines_between_metrics(tmp_path):
    log_file = tmp_path / "MNIST-CyCNN_train.txt"
    log_file.write_text(
        "configuration: {'model': 'CyCNN', 'polar_transform': 'cyclic', 'batch_size': 64, 'lr': 0.001}\n"
        "[Epoch 1] Train Loss: 0.5000\n"
        "debug line inserted by experiment runner\n"
        "Validation loss: 0.4000, Accuracy: 80/100 (80.00%)\n"
        "another extra line\n"
        "Elapsed time: 12.5 sec\n"
        "[Epoch 2] Train Loss: 0.3000\n"
        "Validation loss: 0.2500, Accuracy: 90/100 (90.00%)\n"
        "Elapsed time: 10.0 sec\n",
        encoding="utf-8",
    )

    rows = parse_log_file(str(log_file))

    assert len(rows) == 2
    assert rows[0]["epoch"] == 1
    assert rows[0]["train_loss"] == 0.5
    assert rows[0]["val_loss"] == 0.4
    assert rows[0]["accuracy"] == 80.0
    assert rows[0]["elapsed_time"] == 12.5
    assert rows[1]["epoch"] == 2
    assert rows[1]["accuracy"] == 90.0
    assert rows[0]["total_train_time"] == 22.5
    assert rows[1]["total_train_time"] == 22.5


def test_parse_test_log_file_reads_accuracy_and_metadata(tmp_path):
    log_file = tmp_path / "MNIST-CyCNN_test_on_rotated-30.txt"
    log_file.write_text(
        "configuration: {'model': 'CyCNN', 'polar_transform': 'cyclic', 'batch_size': 64, 'lr': 0.001}\n"
        "Test loss: 0.1234, Accuracy: 95/100 (95.00%)\n",
        encoding="utf-8",
    )

    rows = parse_test_log_file(str(log_file))

    assert rows == [
        {
            "model_id": "CyCNN",
            "log_file": "MNIST-CyCNN_test_on_rotated-30.txt",
            "dataset": "MNIST-CyCNN",
            "augmentation_info": "rotated-30",
            "transform": "cyclic",
            "batch_size": 64,
            "lr": 0.001,
            "test_loss": 0.1234,
            "accuracy": 95.0,
            "total": 100,
            "correct": 95,
        }
    ]
