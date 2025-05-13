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

    for root, dirs, files in os.walk(source_folder):
        # Preserve the relative path structure
        relative_path = os.path.relpath(root, source_folder)
        target_dir = os.path.join(dest_folder, relative_path)

        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            # Skip any file that starts with "launcher_"
            if file.startswith("launcher_"):
                continue

            # Full path to the source and destination
            src = os.path.join(root, file)
            dst = os.path.join(target_dir, file)

            # Check if the file already exists
            if os.path.exists(dst):
                if overwrite:
                    shutil.copy2(src, dst)
                    print(f"Overwritten existing log: {dst}")
                else:
                    print(f"Skipped existing log: {dst}")
            else:
                shutil.copy2(src, dst)
                print(f"Copied log: {dst}")



copy_to_wsl = False
wsl_raw_path = Path(r"\\wsl.localhost\Ubuntu\home\testhub\CyCNN\CyCNN-master\cycnn\data\MNIST\raw")

merged_output_folder = "merged_datasets/merged_nonrot_45_90_120"

if copy_to_wsl:
    copy_to_wsl_ubuntu_raw(merged_output_folder, wsl_raw_path)
