import os
from pathlib import Path


def merge_ubyte_files(folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files_to_merge = [
        ("train-images-idx3-ubyte", "train-images-idx3-ubyte"),
        ("train-labels-idx1-ubyte", "train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte"),
        ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"),
    ]

    for filename, output_name in files_to_merge:
        merged_data = b""
        for folder in folders:
            file_path = Path(folder) / filename
            if not file_path.exists():
                print(f"Missing: {file_path}")
                continue
            with open(file_path, "rb") as f:
                data = f.read()
                if merged_data == b"":
                    merged_data = data
                else:
                    # Skip 16-byte header for images or 8-byte for labels
                    header_size = 16 if "images" in filename else 8
                    merged_data += data[header_size:]
        output_path = Path(output_folder) / output_name
        with open(output_path, "wb") as f:
            f.write(merged_data)
        print(f"Merged {filename} → {output_path}")


# === CONFIGURATION ===
folders_to_merge = [
    "dataset_mnist_non_rotated",
    "rotated-45",
    "rotated-90-120"
]

merged_output_folder = "merged_datasets/merged_nonrot_45_90_120"

# === RUN ===
merge_ubyte_files(folders_to_merge, merged_output_folder)
