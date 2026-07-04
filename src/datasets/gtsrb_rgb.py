import os
import numpy as np
from PIL import Image
from pathlib import Path
from src.datasets.gtsrb import download_dataset, prepare_gtsrb_dataset, check_dataset, clean_up


def save_as_idx(image_dir, output_prefix):
    """Convert 32x32 images into IDX format used in MNIST."""
    images, labels = [], []

    for class_dir in sorted(Path(image_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            continue
        for img_file in class_dir.glob("*.png"):
            try:
                img = Image.open(img_file).convert("RGB").resize((32, 32))
                images.append(np.array(img, dtype=np.uint8))
                labels.append(label)
            except Exception as e:
                print(f"❌ Skipping {img_file}: {e}")

    images = np.stack(images)
    labels = np.array(labels, dtype=np.uint8)
    num_images, _, cols, channels = images.shape

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    np.save(f"{output_prefix}_images.npy", images)
    np.save(f"{output_prefix}_labels.npy", labels)
    print(f"✅ Saved .npy dataset to {output_prefix}_images.npy and _labels.npy")


def main():
    download_dataset()
    prepare_gtsrb_dataset()
    check_dataset()
    save_as_idx("dataset/GTSRB_32x32/train", "dataset/GTSRB_RGB/dataset_GTSRB_non_rotated/train")
    save_as_idx("dataset/GTSRB_32x32/test", "dataset/GTSRB_RGB/dataset_GTSRB_non_rotated/test")
    clean_up(["dataset/GTSRB_32x32", "dataset/GTSRB_raw"])
