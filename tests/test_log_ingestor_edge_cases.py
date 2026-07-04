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


def test_parse_filename_handles_training_and_test_log_names():
    assert parse_filename("MNIST-CNN_train.txt") == ("MNIST-CNN", "")
    assert parse_filename("MNIST-CNN_test_on_rotated-90.txt") == ("MNIST-CNN", "rotated-90")


def test_parse_log_file_returns_empty_list_without_configuration(tmp_path):
    log_path = tmp_path / "MNIST-CNN_train.txt"
    log_path.write_text("[Epoch 1] Train Loss: 0.5\n", encoding="utf-8")

    assert parse_log_file(log_path) == []


def test_parse_log_file_handles_extra_lines_and_multiple_epochs(tmp_path):
    log_path = tmp_path / "MNIST-CNN_train.txt"
    log_path.write_text(
        "configuration: {'model': 'CNN', 'polar_transform': 'none', 'batch_size': 32, 'lr': 0.001}\n"
        "[Epoch 1] Train Loss: 0.5000\n"
        "debug line that should not break parser\n"
        "Validation loss: 0.4000, Accuracy: 80/100 (80.00%)\n"
        "another unrelated line\n"
        "Elapsed time: 1.25 sec\n"
        "[Epoch 2] Train Loss: 0.3000\n"
        "Elapsed time: 1.75 sec\n"
        "Validation loss: 0.2000, Accuracy: 90/100 (90.00%)\n",
        encoding="utf-8",
    )

    rows = parse_log_file(log_path)

    assert len(rows) == 2
    assert rows[0]["epoch"] == 1
    assert rows[0]["train_loss"] == pytest.approx(0.5)
    assert rows[0]["val_loss"] == pytest.approx(0.4)
    assert rows[0]["accuracy"] == pytest.approx(80.0)
    assert rows[0]["elapsed_time"] == pytest.approx(1.25)
    assert rows[1]["epoch"] == 2
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


def test_parse_test_log_file_extracts_metrics_and_metadata(tmp_path):
    log_path = tmp_path / "MNIST-CNN_test_on_rotated-90.txt"
    log_path.write_text(
        "configuration: {'model': 'CNN', 'polar_transform': 'none', 'batch_size': 64, 'lr': 0.01}\n"
        "Test loss: 0.1234, Accuracy: 42/50 (84.00%)\n",
        encoding="utf-8",
    )

    rows = parse_test_log_file(log_path)

    assert rows == [
        {
            "model_id": "CNN",
            "log_file": "MNIST-CNN_test_on_rotated-90.txt",
            "dataset": "MNIST-CNN",
            "augmentation_info": "rotated-90",
            "transform": "none",
            "batch_size": 64,
            "lr": 0.01,
            "test_loss": pytest.approx(0.1234),
            "accuracy": pytest.approx(84.0),
            "total": 50,
            "correct": 42,
        }
    ]
