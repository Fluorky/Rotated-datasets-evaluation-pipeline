from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets.gtsrb import (
    check_dataset,
    clean_up,
    download_dataset,
    prepare_gtsrb_dataset,
)


IMAGE_SIZE = (32, 32)
IMAGE_PATTERN = "*.png"


def _iter_labeled_pngs(image_dir: Path):
    """
    Yield (image_path, label) pairs from a GTSRB-style directory:
        image_dir/<class_id>/*.png
    """
    for class_dir in sorted(image_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        try:
            label = int(class_dir.name)
        except ValueError:
            print(f"⚠️ Skipping non-label directory: {class_dir}")
            continue

        for image_path in sorted(class_dir.glob(IMAGE_PATTERN)):
            yield image_path, label


def save_as_idx(image_dir, output_prefix):
    """
    Convert GTSRB RGB PNG images into .npy files used by the NPY pipeline.

    Expected input layout:
        image_dir/<class_id>/*.png

    Output files:
        <output_prefix>_images.npy
        <output_prefix>_labels.npy

    Raises:
        FileNotFoundError: if image_dir does not exist.
        NotADirectoryError: if image_dir is not a directory.
        ValueError: if no valid images are found or image/label counts mismatch.
    """
    image_dir = Path(image_dir)
    output_prefix = Path(output_prefix)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image path is not a directory: {image_dir}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for image_path, label in _iter_labeled_pngs(image_dir):
        try:
            with Image.open(image_path) as img:
                arr = np.array(img.convert("RGB").resize(IMAGE_SIZE), dtype=np.uint8)
        except Exception as exc:
            skipped += 1
            print(f"❌ Skipping {image_path}: {exc}")
            continue

        images.append(arr)
        labels.append(label)

    if not images:
        raise ValueError(f"No valid PNG images found in {image_dir}")

    if len(images) != len(labels):
        raise ValueError(
            f"Image/label count mismatch for {image_dir}: "
            f"{len(images)} images vs {len(labels)} labels"
        )

    images_array = np.stack(images, axis=0)
    labels_array = np.array(labels, dtype=np.uint8)

    if images_array.ndim != 4 or images_array.shape[1:] != (*IMAGE_SIZE, 3):
        raise ValueError(
            f"Unexpected image array shape for {image_dir}: {images_array.shape}. "
            f"Expected (N, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 3)."
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    images_path = output_prefix.with_name(f"{output_prefix.name}_images.npy")
    labels_path = output_prefix.with_name(f"{output_prefix.name}_labels.npy")

    np.save(images_path, images_array)
    np.save(labels_path, labels_array)

    print(
        f"✅ Saved {len(images_array)} images to {images_path} and labels to {labels_path}"
    )
    if skipped:
        print(f"⚠️ Skipped {skipped} unreadable image(s) from {image_dir}")


def main():
    download_dataset()
    prepare_gtsrb_dataset()
    check_dataset()
    save_as_idx(
        "dataset/GTSRB_32x32/train",
        "dataset/GTSRB_RGB/dataset_GTSRB_non_rotated/train",
    )
    save_as_idx(
        "dataset/GTSRB_32x32/test",
        "dataset/GTSRB_RGB/dataset_GTSRB_non_rotated/test",
    )
    clean_up(["dataset/GTSRB_32x32", "dataset/GTSRB_raw"])


if __name__ == "__main__":
    main()
