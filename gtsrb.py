import os
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from PIL import Image
import pandas as pd


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
    print(f"✅ Found Train.csv with {len(train_csv)} entries → processing...")

    os.makedirs(target_train_dir, exist_ok=True)

    for _, row in train_csv.iterrows():
        # Use path relative to raw_dir
        img_path = Path(raw_dir) / row["Path"]
        class_id = str(row["ClassId"]).zfill(5)
        output_class_dir = Path(target_train_dir) / class_id
        os.makedirs(output_class_dir, exist_ok=True)

        img = Image.open(img_path)
        img_32 = img.resize((32, 32), Image.Resampling.LANCZOS)

        output_file = output_class_dir / (img_path.stem + ".png")
        img_32.save(output_file)

    print("✅ TRAIN set prepared.")


def prepare_gtsrb_32x32():
    raw_dir = "dataset/GTSRB_raw"
    train_zip = os.path.join(raw_dir, "Train.zip")
    test_zip = os.path.join(raw_dir, "Test.zip")

    train_dir = os.path.join(raw_dir, "Train")
    test_dir = os.path.join(raw_dir, "Test")
    output_base = "dataset/GTSRB_32x32"

    # Unzip if not already
    if not os.path.exists(train_dir):
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(train_dir)
    if not os.path.exists(test_dir):
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
    print(f"✅ Found Test.csv with {len(test_csv)} entries → processing...")

    target_test_dir = os.path.join(output_base, "test")
    os.makedirs(target_test_dir, exist_ok=True)

    for _, row in test_csv.iterrows():
        img_path = Path(raw_dir) / row["Path"]
        class_id = str(row["ClassId"]).zfill(5)
        output_class_dir = Path(target_test_dir) / class_id
        os.makedirs(output_class_dir, exist_ok=True)

        img = Image.open(img_path)
        img_32 = img.resize((32, 32), Image.Resampling.LANCZOS)

        output_file = output_class_dir / (img_path.stem + ".png")
        img_32.save(output_file)

    print("✅ TEST set prepared.")


# MAIN flow
if __name__ == "__main__":
    # Step 1: download
    download_gtsrb_kaggle()

    # Step 2 & 3: prepare train/test split with 32x32 images
    prepare_gtsrb_32x32()

    print("🎉 All done → ready to use in your pipeline!")
