import os
import struct
import shutil
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(destination="dataset/GTSRB_raw"):
    """Download the GTSRB dataset from Kaggle."""
    os.makedirs(destination, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print("📥 Downloading GTSRB dataset from Kaggle...")
    api.dataset_download_files(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        path=destination,
        unzip=True
    )
    print(f"✅ Downloaded to {destination}")


def extract_archives(raw_dir):
    """Extract training and test archives if needed."""
    train_zip = os.path.join(raw_dir, "Train.zip")
    test_zip = os.path.join(raw_dir, "Test.zip")

    if not os.path.exists(os.path.join(raw_dir, "Train")) and os.path.exists(train_zip):
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, "Train"))

    if not os.path.exists(os.path.join(raw_dir, "Test")) and os.path.exists(test_zip):
        with zipfile.ZipFile(test_zip, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, "Test"))


def prepare_split(csv_path, raw_dir, target_dir):
    """Convert dataset CSV into 32x32 grayscale images in class-labeled folders."""
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ClassId"])
    df["ClassId"] = pd.to_numeric(df["ClassId"], errors='coerce').astype(int)
    df = df[df["ClassId"] < 43]  # Valid classes

    os.makedirs(target_dir, exist_ok=True)
    for _, row in df.iterrows():
        img_path = Path(raw_dir) / row["Path"]
        class_id = str(row["ClassId"]).zfill(5)
        output_class_dir = Path(target_dir) / class_id
        output_class_dir.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path).resize((32, 32), Image.Resampling.LANCZOS)
            img.save(output_class_dir / (img_path.stem + ".png"))
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")


def prepare_gtsrb_dataset():
    raw_dir = "dataset/GTSRB_raw"
    output_dir = "dataset/GTSRB_32x32"

    extract_archives(raw_dir)

    print("🔄 Preparing TRAIN dataset...")
    prepare_split(os.path.join(raw_dir, "Train.csv"), raw_dir, os.path.join(output_dir, "train"))

    print("🔄 Preparing TEST dataset...")
    prepare_split(os.path.join(raw_dir, "Test.csv"), raw_dir, os.path.join(output_dir, "test"))


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
                img = Image.open(img_file).convert("L").resize((32, 32))
                images.append(np.array(img, dtype=np.uint8))
                labels.append(label)
            except Exception as e:
                print(f"❌ Skipping {img_file}: {e}")

    images = np.stack(images)
    labels = np.array(labels, dtype=np.uint8)
    num_images, rows, cols = images.shape

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    with open(f"{output_prefix}-images-idx3-ubyte", "wb") as f_img:
        f_img.write(struct.pack(">IIII", 2051, num_images, rows, cols))
        f_img.write(images.tobytes())

    with open(f"{output_prefix}-labels-idx1-ubyte", "wb") as f_lbl:
        f_lbl.write(struct.pack(">II", 2049, num_images))
        f_lbl.write(labels.tobytes())

    print(f"✅ IDX files saved to: {output_prefix}-images-idx3-ubyte / -labels-idx1-ubyte")


def clean_up(paths):
    for path in paths:
        try:
            shutil.rmtree(path)
            print(f"🧹 Removed {path}")
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"❌ Failed to remove {path}: {e}")


def check_dataset():
    image_root = Path("dataset/GTSRB_raw")

    # Stats
    min_width = float("inf")
    min_height = float("inf")
    max_width = float("-inf")
    max_height = float("-inf")
    total_width = 0
    total_height = 0
    image_count = 0
    small_count = 0

    # Keep track of which images are smallest and largest
    smallest_image = None
    largest_image = None

    # For histograms
    widths = []
    heights = []

    # Supported formats
    image_extensions = [".png", ".jpg", ".jpeg"]

    for img_path in image_root.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    image_count += 1
                    total_width += width
                    total_height += height
                    widths.append(width)
                    heights.append(height)

                    if width < 32 or height < 32:
                        small_count += 1

                    if width <= min_width and height <= min_height:
                        min_width = width
                        min_height = height
                        smallest_image = img_path

                    if width >= max_width and height >= max_height:
                        max_width = width
                        max_height = height
                        largest_image = img_path

            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")

    # Compute averages
    avg_width = total_width / image_count if image_count else 0
    avg_height = total_height / image_count if image_count else 0

    # Output stats
    print(f"📊 Total images processed: {image_count}")
    print(f"📏 Min size:     {min_width} x {min_height} px → {smallest_image}")
    print(f"📐 Max size:     {max_width} x {max_height} px → {largest_image}")
    print(f"📉 Avg. size:    {avg_width:.2f} x {avg_height:.2f} px")
    print(f"⚠️ Images smaller than 32x32: {small_count}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(widths, bins=30, alpha=0.6, label="Width")
    plt.hist(heights, bins=30, alpha=0.6, label="Height")
    plt.axvline(32, color='red', linestyle='--', label="32 px threshold")
    plt.legend()
    plt.title("Image Size Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    download_dataset()
    prepare_gtsrb_dataset()
    check_dataset()
    save_as_idx("dataset/GTSRB_32x32/train", "dataset/GTSRB/dataset_GTSRB_non_rotated/train")
    save_as_idx("dataset/GTSRB_32x32/test", "dataset/GTSRB/dataset_GTSRB_non_rotated/test")
    clean_up(["dataset/GTSRB_32x32", "dataset/GTSRB_raw"])
