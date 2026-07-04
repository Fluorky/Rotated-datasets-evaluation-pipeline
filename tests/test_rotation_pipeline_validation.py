from pathlib import Path

import numpy as np
import pytest

from src.pipelines.rotation_pipeline import (
    _guard_output,
    _output_state,
    run_pipeline,
    validate_pipeline_inputs,
)


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_validate_pipeline_inputs_rejects_missing_base_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_pipeline_inputs(
            base_dir=tmp_path / "missing",
            dataset_name="dataset_MNIST_non_rotated",
            file_format="ubyte",
        )


def test_validate_pipeline_inputs_rejects_invalid_file_format(tmp_path):
    base_dir = tmp_path / "dataset"
    dataset_dir = base_dir / "dataset_MNIST_non_rotated"
    dataset_dir.mkdir(parents=True)

    with pytest.raises(ValueError):
        validate_pipeline_inputs(base_dir, "dataset_MNIST_non_rotated", file_format="jpg")


def test_validate_pipeline_inputs_rejects_missing_dataset_dir(tmp_path):
    base_dir = tmp_path / "dataset"
    base_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        validate_pipeline_inputs(base_dir, "dataset_MNIST_non_rotated", file_format="ubyte")


def test_validate_pipeline_inputs_accepts_complete_ubyte_dataset_layout(tmp_path):
    base_dir = tmp_path / "dataset"
    dataset_dir = base_dir / "dataset_MNIST_non_rotated"

    for split in ["train", "test"]:
        _touch(dataset_dir / f"{split}-images-idx3-ubyte")
        _touch(dataset_dir / f"{split}-labels-idx1-ubyte")

    validate_pipeline_inputs(base_dir, "dataset_MNIST_non_rotated", file_format="ubyte")


def test_validate_pipeline_inputs_accepts_complete_npy_dataset_layout(tmp_path):
    base_dir = tmp_path / "dataset"
    dataset_dir = base_dir / "dataset_GTSRB_RGB_non_rotated"
    dataset_dir.mkdir(parents=True)

    for split in ["train", "test"]:
        np.save(dataset_dir / f"{split}_images.npy", np.zeros((1, 32, 32, 3), dtype=np.uint8))
        np.save(dataset_dir / f"{split}_labels.npy", np.array([0], dtype=np.uint8))

    validate_pipeline_inputs(base_dir, "dataset_GTSRB_RGB_non_rotated", file_format="npy")


def test_run_pipeline_applies_validation_before_heavy_work(tmp_path):
    base_dir = tmp_path / "dataset"
    dataset_dir = base_dir / "dataset_MNIST_non_rotated"
    dataset_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        run_pipeline(
            base_dir=str(base_dir),
            dataset_name="dataset_MNIST_non_rotated",
            dataset_key="MNIST",
            file_format="ubyte",
        )


def test_output_state_distinguishes_missing_partial_and_complete(tmp_path):
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    assert _output_state(files) == "missing"

    _touch(files[0])
    assert _output_state(files) == "partial"

    _touch(files[1])
    assert _output_state(files) == "complete"


def test_guard_output_skips_complete_output_when_not_overwriting(tmp_path):
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for file in files:
        _touch(file)

    assert _guard_output(files, label="existing output", overwrite=False) is False


def test_guard_output_rejects_partial_output_when_not_overwriting(tmp_path):
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    _touch(files[0])

    with pytest.raises(RuntimeError):
        _guard_output(files, label="partial output", overwrite=False)


def test_guard_output_allows_any_state_when_overwriting(tmp_path):
    files = [tmp_path / "a.txt", tmp_path / "b.txt"]
    _touch(files[0])

    assert _guard_output(files, label="partial output", overwrite=True) is True
