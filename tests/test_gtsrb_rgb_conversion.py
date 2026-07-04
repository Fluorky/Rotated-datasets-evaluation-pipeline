import pytest

from src.datasets.gtsrb_rgb import save_as_idx


def test_save_as_idx_rejects_missing_image_dir(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        save_as_idx(str(missing_dir), str(tmp_path / "out" / "train"))


def test_save_as_idx_rejects_empty_image_dir(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with pytest.raises(ValueError):
        save_as_idx(str(image_dir), str(tmp_path / "out" / "train"))