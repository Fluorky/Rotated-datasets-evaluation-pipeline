import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from PIL import Image
import pandas as pd
import struct
import numpy as np
import shutil


def download_gtsrb_kaggle(output_path="dataset/GTSRB_raw"):
    os.makedirs(output_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print("📥 Downloading GTSRB from Kaggle...")
    api.dataset_download_files(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign",
        path=output_path,
        unzip=True
    )
    print(f"✅ Download complete → {output_path}")


def convert_train_csv_to_32x32():
    raw_dir = "dataset/GTSRB_raw"
    train_csv_path = os.path.join(raw_dir, "Train.csv")
    target_train_dir = "dataset/GTSRB_32x32/train"

    print("🔄 Preparing TRAIN set...")

    if not os.path.exists(train_csv_path):
        print("⚠️ No Train.csv found → skipping train set preparation!")
        return

    train_csv = pd.read_csv(train_csv_path)
    print(f"✅ Found Train.csv with {len(train_csv)} entries")

    train_csv["ClassId"] = pd.to_numeric(train_csv["ClassId"], errors='coerce')
    train_csv = train_csv.dropna(subset=["ClassId"])
    train_csv["ClassId"] = train_csv["ClassId"].astype(int)

    # 🔍 Show invalid ClassId rows (>= 43)
    out_of_bounds = train_csv[train_csv["ClassId"] >= 43]
    if not out_of_bounds.empty:
        print("⚠️ ClassId >= 43 detected:")
        print(out_of_bounds[["ClassId", "Path"]].head())

    # ✅ Filter valid ones
    train_csv = train_csv[train_csv["ClassId"] < 43]

    os.makedirs(target_train_dir, exist_ok=True)

    for _, row in train_csv.iterrows():
        # Use path relative to raw_dir
        img_path = Path(raw_dir) / row["Path"]
        class_id = str(row["ClassId"]).zfill(5)
        output_class_dir = Path(target_train_dir) / class_id
        os.makedirs(output_class_dir, exist_ok=True)

        try:
            img = Image.open(img_path)
            img_32 = img.resize((32, 32), Image.Resampling.LANCZOS)

            output_file = output_class_dir / (img_path.stem + ".png")
            img_32.save(output_file)
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    print("✅ TRAIN set prepared.")


def prepare_gtsrb_32x32():
    raw_dir = "dataset/GTSRB_raw"
    train_zip = os.path.join(raw_dir, "Train.zip")
    test_zip = os.path.join(raw_dir, "Test.zip")

    train_dir = os.path.join(raw_dir, "Train")
    test_dir = os.path.join(raw_dir, "Test")
    output_base = "dataset/GTSRB_32x32"

    if not os.path.exists(train_dir) and os.path.exists(train_zip):
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(train_dir)
    if not os.path.exists(test_dir) and os.path.exists(test_zip):
        with zipfile.ZipFile(test_zip, 'r') as zip_ref:
            zip_ref.extractall(test_dir)

    # Convert train (correct version)
    convert_train_csv_to_32x32()

    # Prepare test
    print("🔄 Preparing TEST set...")
    test_csv_path = os.path.join(raw_dir, "Test.csv")
    if not os.path.exists(test_csv_path):
        print("⚠️ No Test.csv found → skipping test set preparation!")
        return

    test_csv = pd.read_csv(test_csv_path)
    test_csv["ClassId"] = pd.to_numeric(test_csv["ClassId"], errors='coerce')
    test_csv = test_csv.dropna(subset=["ClassId"])
    test_csv["ClassId"] = test_csv["ClassId"].astype(int)

    print(f"✅ Found Test.csv with {len(test_csv)} entries")

    target_test_dir = os.path.join(output_base, "test")
    os.makedirs(target_test_dir, exist_ok=True)

    for _, row in test_csv.iterrows():
        img_path = Path(raw_dir) / row["Path"]
        class_id = str(row["ClassId"]).zfill(5)
        output_class_dir = Path(target_test_dir) / class_id
        os.makedirs(output_class_dir, exist_ok=True)

        try:
            img = Image.open(img_path)
            img_32 = img.resize((32, 32), Image.Resampling.LANCZOS)
            output_file = output_class_dir / (img_path.stem + ".png")
            img_32.save(output_file)
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    print("✅ TEST set prepared.")


def create_idx_files(image_dir: str, output_prefix: str):
    images = []
    labels = []

    class_dirs = sorted(Path(image_dir).iterdir())
    for class_dir in class_dirs:
        if class_dir.is_dir():
            try:
                label = int(class_dir.name)
            except ValueError:
                continue
            for img_path in class_dir.glob("*.png"):
                try:
                    img = Image.open(img_path).convert("L")
                    img = img.resize((32, 32))
                    images.append(np.array(img, dtype=np.uint8))
                    labels.append(label)
                except Exception as e:
                    print(f"❌ Skipping {img_path}: {e}")

    images = np.stack(images)
    labels = np.array(labels, dtype=np.uint8)

    num_images, rows, cols = images.shape

    # Ensure output directory exists
    output_dir = os.path.dirname(output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    # Save images (idx3-ubyte)
    with open(f"{output_prefix}-images-idx3-ubyte", "wb") as f:
        f.write(struct.pack(">IIII", 2051, num_images, rows, cols))
        f.write(images.tobytes())

    # Save labels (idx1-ubyte)
    with open(f"{output_prefix}-labels-idx1-ubyte", "wb") as f:
        f.write(struct.pack(">II", 2049, num_images))
        f.write(labels.tobytes())

    print(f"✅ Saved IDX files to {output_prefix}-images-idx3-ubyte and -labels-idx1-ubyte")


def cleanup_temp_dirs(paths):
    """Remove temporary directories after processing."""
    for folder in paths:
        try:
            shutil.rmtree(folder)
            print(f"🧹 Removed temporary folder: {folder}")
        except FileNotFoundError:
            print(f"⚠️ Folder not found (already removed?): {folder}")
        except Exception as e:
            print(f"❌ Failed to remove {folder}: {e}")


if __name__ == "__main__":
    # Step 1: download
    download_gtsrb_kaggle()

    # Step 2 & 3: prepare train/test split with 32x32 images
    prepare_gtsrb_32x32()
    print("✅ All preprocessing done.")

    # Step 4: Create IDX files
    create_idx_files("dataset/GTSRB_32x32/train", "dataset/GTSRB/dataset_GTSRB_non_rotated_32x32/train")
    create_idx_files("dataset/GTSRB_32x32/test", "dataset/GTSRB/dataset_GTSRB_non_rotated_32x32/test")

    # Step 5: Cleanup
    cleanup_temp_dirs(["dataset/GTSRB_raw", "dataset/GTSRB_32x32"])
