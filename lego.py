import os
from pathlib import Path
from collections import defaultdict
import struct
import numpy as np
from PIL import Image
import random
import json
from tqdm import tqdm  # pip install tqdm

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def parse_validation_file(txt_path):
    """
    Each line is a *filename* (relative to the image folder).
    Class name = everything except the last token (the last token is e.g. "078L.png").
    Example:
        "14719 flat tile corner 2x2 078L.png"
        -> class = "14719 flat tile corner 2x2"
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    label_map = {}        # class_name -> int
    image_entries = []    # (relative_path, label_id)

    for line in lines:
        parts = line.split()
        class_name = " ".join(parts[:-1])  # everything except the last token
        if class_name not in label_map:
            label_map[class_name] = len(label_map)
        label = label_map[class_name]
        image_entries.append((line, label))

    return image_entries, label_map


def stratified_split(entries, train_ratio=0.8):
    """
    Stratified split per class so train and test contain the same set of classes.
    """
    by_label = defaultdict(list)
    for path, lab in entries:
        by_label[lab].append(path)

    train_entries, test_entries = [], []
    for lab, files in by_label.items():
        random.shuffle(files)
        k = int(len(files) * train_ratio)
        train_entries += [(f, lab) for f in files[:k]]
        test_entries  += [(f, lab) for f in files[k:]]

    # Not required, but nice for determinism/readability
    train_entries.sort(key=lambda x: (x[1], x[0]))
    test_entries.sort(key=lambda x: (x[1], x[0]))
    return train_entries, test_entries


def write_idx_streaming(entries, image_folder, out_prefix, img_size=(96, 96)):
    """
    Fast, streaming IDX writer (no bytes concatenation in a loop).
    """
    rows, cols = img_size
    n = len(entries)

    images_path = f"{out_prefix}-images-idx3-ubyte"
    labels_path = f"{out_prefix}-labels-idx1-ubyte"

    with open(images_path, "wb") as f_img, open(labels_path, "wb") as f_lab:
        # IDX headers (MNIST-like)
        f_img.write(struct.pack(">IIII", 2051, n, rows, cols))
        f_lab.write(struct.pack(">II", 2049, n))

        for rel, label in tqdm(entries, desc=f"Writing {out_prefix}", unit="img"):
            img_path = Path(image_folder) / rel
            if not img_path.exists():
                print(f"⚠️ Missing file: {img_path}")
                continue  # skip or raise, your choice

            img = Image.open(img_path).convert("L")
            img = img.resize((cols, rows), Image.Resampling.LANCZOS)
            img_np = np.array(img, dtype=np.uint8)

            f_img.write(img_np.tobytes())
            f_lab.write(struct.pack("B", label))

    print(f"✅ Wrote {n} samples to {out_prefix}")


def sanity_check_idx(images_path, labels_path):
    with open(images_path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
    with open(labels_path, "rb") as f:
        _, num_l = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    print(
        f"{labels_path}: num={num_l}, min={labels.min()}, max={labels.max()}, "
        f"unique={len(np.unique(labels))}"
    )


if __name__ == "__main__":
    validation_txt = "dataset/LEGO_raw/validation.txt"
    images_root    = "dataset/LEGO_raw/dataset"   # all PNGs live here (flat)
    out_root       = "dataset/LEGO_idx96x96/dataset_LEGO_non_rotated"

    os.makedirs(out_root, exist_ok=True)

    entries, label_map = parse_validation_file(validation_txt)
    print("🔢 Total number of classes:", len(label_map))

    train_entries, test_entries = stratified_split(entries, train_ratio=0.8)

    # Sanity: ensure both splits contain the same set of classes
    train_classes = sorted({l for _, l in train_entries})
    test_classes  = sorted({l for _, l in test_entries})
    assert train_classes == test_classes, "Classes mismatch between train and test!"
    print(f"✅ Same classes in train and test: {len(train_classes)} classes")

    # Write IDX
    write_idx_streaming(train_entries, images_root, f"{out_root}/train", img_size=(96, 96))
    write_idx_streaming(test_entries,  images_root, f"{out_root}/test",  img_size=(96, 96))

    # Save label map
    with open(f"{out_root}/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    # Final sanity check
    sanity_check_idx(f"{out_root}/train-images-idx3-ubyte", f"{out_root}/train-labels-idx1-ubyte")
    sanity_check_idx(f"{out_root}/test-images-idx3-ubyte",  f"{out_root}/test-labels-idx1-ubyte")
