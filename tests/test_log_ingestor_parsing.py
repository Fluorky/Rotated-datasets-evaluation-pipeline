import pytest

from src.analysis.log_ingestor import (
    extract_config,
    parse_filename,
    parse_log_file,
    parse_test_log_file,
)


def test_extract_config_returns_empty_dict_for_missing_or_invalid_config():
    assert extract_config(None) == {}
    assert extract_config("no config here") == {}
    assert extract_config("configuration: {'model':") == {}


def test_parse_filename_for_train_log():
    dataset, augmentation = parse_filename("/tmp/MNIST-CyCNN_train.txt")

    assert dataset == "MNIST-CyCNN"
    assert augmentation == ""


def test_parse_filename_for_test_log():
    dataset, augmentation = parse_filename("/tmp/MNIST-CyCNN_test_on_rotated-30.txt")

    assert dataset == "MNIST-CyCNN"
    assert augmentation == "rotated-30"


def test_parse_log_file_returns_empty_list_without_configuration(tmp_path):
    log_path = tmp_path / "MNIST-CNN_train.txt"
    log_path.write_text("[Epoch 1] Train Loss: 0.5\n", encoding="utf-8")

    assert parse_log_file(log_path) == []


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
    assert rows[0]["train_loss"] == pytest.approx(0.5)
    assert rows[0]["val_loss"] == pytest.approx(0.4)
    assert rows[0]["accuracy"] == pytest.approx(80.0)
    assert rows[0]["elapsed_time"] == pytest.approx(12.5)
    assert rows[1]["epoch"] == 2
    assert rows[1]["accuracy"] == pytest.approx(90.0)
    assert rows[0]["total_train_time"] == pytest.approx(22.5)
    assert rows[1]["total_train_time"] == pytest.approx(22.5)


def test_parse_log_file_handles_elapsed_before_validation_within_epoch(tmp_path):
    log_path = tmp_path / "MNIST-CNN_train.txt"
    log_path.write_text(
        "configuration: {'model': 'CNN', 'polar_transform': 'none', 'batch_size': 32, 'lr': 0.001}\n"
        "[Epoch 1] Train Loss: 0.5000\n"
        "debug line that should not break parser\n"
        "Validation loss: 0.4000, Accuracy: 80/100 (80.00%)\n"
        "Elapsed time: 1.25 sec\n"
        "[Epoch 2] Train Loss: 0.3000\n"
        "Elapsed time: 1.75 sec\n"
        "Validation loss: 0.2000, Accuracy: 90/100 (90.00%)\n",
        encoding="utf-8",
    )

    rows = parse_log_file(log_path)

    assert len(rows) == 2
    assert rows[0]["val_loss"] == pytest.approx(0.4)
    assert rows[0]["elapsed_time"] == pytest.approx(1.25)
    assert rows[1]["val_loss"] == pytest.approx(0.2)
    assert rows[1]["elapsed_time"] == pytest.approx(1.75)
    assert rows[0]["total_train_time"] == pytest.approx(3.0)
    assert rows[1]["total_train_time"] == pytest.approx(3.0)


def test_parse_test_log_file_returns_empty_list_without_test_loss(tmp_path):
    log_path = tmp_path / "MNIST-CNN_test_on_rotated-90.txt"
    log_path.write_text(
        "configuration: {'model': 'CNN', 'polar_transform': 'none'}\n"
        "no test metrics here\n",
        encoding="utf-8",
    )

    assert parse_test_log_file(log_path) == []


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
            "test_loss": pytest.approx(0.1234),
            "accuracy": pytest.approx(95.0),
            "total": 100,
            "correct": 95,
        }
    ]
