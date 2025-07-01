import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image

import struct
import numpy as np


def download_lego_kaggle(output_path="dataset/LEGO_raw"):
    os.makedirs(output_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("📥 Downloading Lego from Kaggle...")
    api.dataset_download_files(
        "joosthazelzet/lego-brick-images",
        path=output_path,
        unzip=True
    )
    print(f"✅ Download complete → {output_path}")


def parse_validation_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    label_map = {}   # text class → int
    image_entries = []

    for line in lines:
        parts = line.strip().split()
        class_name = " ".join(parts[:-1])
        filename = line.strip()

        if class_name not in label_map:
            label_map[class_name] = len(label_map)

        label = label_map[class_name]
        image_entries.append((filename, label))

    return image_entries, label_map


def process_images_to_idx(image_entries, image_folder, output_prefix, img_size=(128, 128)):
    images_bin = b""
    labels_bin = b""

    count = 0

    for filename, label in image_entries:
        img_path = Path(image_folder) / filename
        if not img_path.exists():
            print(f"⚠️ Not found: {img_path}")
            continue

        img = Image.open(img_path).convert("L")  # grayscale
        img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
        img_bytes = np.array(img_resized, dtype=np.uint8).tobytes()

        images_bin += img_bytes
        labels_bin += bytes([label])
        count += 1

    rows, cols = img_size

    with open(f"{output_prefix}-images-idx3-ubyte", "wb") as f:
        f.write(struct.pack(">IIII", 2051, count, rows, cols))  # magic number for images
        f.write(images_bin)

    with open(f"{output_prefix}-labels-idx1-ubyte", "wb") as f:
        f.write(struct.pack(">II", 2049, count))  # magic number for labels
        f.write(labels_bin)

    print(f"✅ Saved {count} samples to {output_prefix}")


if __name__ == "__main__":
    val_txt = "dataset/LEGO_raw/validation.txt"
    img_dir = "dataset/LEGO_raw/dataset"

    entries, label_map = parse_validation_file(val_txt)
    print(len(entries))
    train_split = int(0.8 * len(entries))
    train_entries = entries[:train_split]
    test_entries = entries[train_split:]

    os.makedirs("dataset/LEGO_idx224x224", exist_ok=True)

    process_images_to_idx(train_entries, img_dir, "dataset/LEGO_idx224x224/train")
    process_images_to_idx(test_entries, img_dir, "dataset/LEGO_idx224x224/test")

    print(f"🔢 Number of classes: {len(label_map)}")
