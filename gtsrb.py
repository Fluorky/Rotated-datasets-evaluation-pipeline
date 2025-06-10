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


def convert_images_to_32x32(source_dir, target_dir):
    print(f"🔄 Converting images from {source_dir} → {target_dir} ...")
    os.makedirs(target_dir, exist_ok=True)

    # Loop over class folders
    for class_dir in Path(source_dir).glob("*"):
        if class_dir.is_dir():
            class_id = class_dir.name
            output_class_dir = Path(target_dir) / class_id
            os.makedirs(output_class_dir, exist_ok=True)

            for img_file in class_dir.glob("*.ppm"):
                img = Image.open(img_file)
                img_32 = img.resize((32, 32), Image.ANTIALIAS)

                # Save as PNG for convenience
                output_file = output_class_dir / (img_file.stem + ".png")
                img_32.save(output_file)

    print("✅ Conversion to 32x32 done.")


def prepare_gtsrb_32x32():
    raw_dir = "dataset/GTSRB_raw"
    train_dir = os.path.join(raw_dir, "Train")
    test_dir = os.path.join(raw_dir, "Test")
    output_base = "dataset/GTSRB_32x32"

    # Train
    convert_images_to_32x32(
        source_dir=os.path.join(train_dir, "Images"),
        target_dir=os.path.join(output_base, "train")
    )

    # Test (test labels are in CSV)
    print("🔄 Preparing TEST set...")

    # Check if Test.csv exists
    test_csv_path = os.path.join(raw_dir, "Test.csv")
    if not os.path.exists(test_csv_path):
        print("⚠️ No Test.csv found in Test folder → skipping test set preparation!")
        return  # skip test preparation gracefully

    test_csv = pd.read_csv(test_csv_path)
    print(f"✅ Found Test.csv with {len(test_csv)} entries → processing...")

    test_images_dir = test_dir
    target_test_dir = os.path.join(output_base, "test")
    os.makedirs(target_test_dir, exist_ok=True)

    for _, row in test_csv.iterrows():
        img_path = Path(raw_dir) / row["Path"]
        class_id = str(row["ClassId"]).zfill(5)
        output_class_dir = Path(target_test_dir) / class_id
        os.makedirs(output_class_dir, exist_ok=True)

        img = Image.open(img_path)
        img_32 = img.resize((32, 32), Image.ANTIALIAS)

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
