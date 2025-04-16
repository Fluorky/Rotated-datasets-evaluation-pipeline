import shutil
import os
from pathlib import Path


def copy_to_wsl_ubuntu_raw(source_folder, wsl_raw_folder):
    if Path(wsl_raw_folder).exists():
        shutil.rmtree(wsl_raw_folder)
    shutil.copytree(source_folder, wsl_raw_folder)
    print(f"Copied merged dataset to WSL raw folder: {wsl_raw_folder}")


def sync_wsl_logs(source_folder, dest_folder, overwrite=False):
    os.makedirs(dest_folder, exist_ok=True)

    for file in os.listdir(source_folder):
        if file.endswith(".txt"):
            src = os.path.join(source_folder, file)
            dst = os.path.join(dest_folder, file)

            if os.path.exists(dst):
                if overwrite:
                    shutil.copy2(src, dst)
                    print(f"Overwritten existing log: {file}")
                else:
                    print(f"Skipped existing log: {file}")
            else:
                shutil.copy2(src, dst)
                print(f"Copied log: {file}")


copy_to_wsl = False
wsl_raw_path = Path(r"\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\data\MNIST\raw")

merged_output_folder = "merged_datasets/merged_nonrot_45_90_120"

if copy_to_wsl:
    copy_to_wsl_ubuntu_raw(merged_output_folder, wsl_raw_path)
