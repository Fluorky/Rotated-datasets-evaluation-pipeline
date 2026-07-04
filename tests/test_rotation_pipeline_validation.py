import pytest

from src.pipelines.rotation_pipeline import run_pipeline


def test_run_pipeline_rejects_missing_base_dir(tmp_path):
    missing_base_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        run_pipeline(
            base_dir=str(missing_base_dir),
            dataset_name="dataset_MNIST_non_rotated",
            dataset_key="MNIST",
            file_format="ubyte",
        )


def test_run_pipeline_rejects_invalid_file_format(tmp_path):
    base_dir = tmp_path / "dataset"
    base_dir.mkdir()

    with pytest.raises(ValueError):
        run_pipeline(
            base_dir=str(base_dir),
            dataset_name="dataset_MNIST_non_rotated",
            dataset_key="MNIST",
            file_format="jpg",
        )


def test_run_pipeline_rejects_missing_dataset_dir(tmp_path):
    base_dir = tmp_path / "dataset"
    base_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        run_pipeline(
            base_dir=str(base_dir),
            dataset_name="dataset_MNIST_non_rotated",
            dataset_key="MNIST",
            file_format="ubyte",
        )


def test_run_pipeline_rejects_missing_ubyte_split_files(tmp_path):
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


def test_run_pipeline_rejects_missing_npy_split_files(tmp_path):
    base_dir = tmp_path / "dataset"
    dataset_dir = base_dir / "dataset_GTSRB_RGB_non_rotated"
    dataset_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        run_pipeline(
            base_dir=str(base_dir),
            dataset_name="dataset_GTSRB_RGB_non_rotated",
            dataset_key="GTSRB_RGB",
            file_format="npy",
        )