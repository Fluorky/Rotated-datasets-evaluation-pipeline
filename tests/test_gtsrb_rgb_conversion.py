import numpy as np
import pytest
from PIL import Image

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


def test_save_as_idx_rejects_dir_without_valid_class_dirs(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "not_a_class").mkdir()

    with pytest.raises(ValueError):
        save_as_idx(str(image_dir), str(tmp_path / "out" / "train"))


def test_save_as_idx_writes_images_and_labels(tmp_path):
    image_dir = tmp_path / "images"
    class_dir = image_dir / "7"
    class_dir.mkdir(parents=True)

    img = Image.fromarray(np.full((40, 40, 3), 128, dtype=np.uint8))
    img.save(class_dir / "sample.png")

    output_prefix = tmp_path / "out" / "train"
    save_as_idx(str(image_dir), str(output_prefix))

    images = np.load(f"{output_prefix}_images.npy")
    labels = np.load(f"{output_prefix}_labels.npy")

    assert images.shape == (1, 32, 32, 3)
    assert images.dtype == np.uint8
    assert labels.tolist() == [7]
    assert labels.dtype == np.uint8
