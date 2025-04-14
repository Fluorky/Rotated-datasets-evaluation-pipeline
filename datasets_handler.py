import os
from pathlib import Path
import struct


def merge_ubyte_files(folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files_to_merge = [
        ("train-images-idx3-ubyte", "train-images-idx3-ubyte", 16, ">IIII"),  # magic, num, rows, cols
        ("train-labels-idx1-ubyte", "train-labels-idx1-ubyte", 8, ">II"),  # magic, num
        ("t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte", 16, ">IIII"),
        ("t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte", 8, ">II"),
    ]

    for filename, output_name, header_size, header_fmt in files_to_merge:
        merged_body = b""
        total_samples = 0
        header_data = None

        for folder in folders:
            file_path = Path(folder) / filename
            if not file_path.exists():
                print(f"Missing: {file_path}")
                continue

            with open(file_path, "rb") as f:
                header = f.read(header_size)
                body = f.read()

                if header_data is None:
                    header_data = list(struct.unpack(header_fmt, header))

                if "images" in filename:
                    rows, cols = header_data[-2], header_data[-1]
                    sample_size = rows * cols
                else:
                    sample_size = 1

                samples = len(body) // sample_size
                total_samples += samples
                merged_body += body

        if header_data is not None:
            header_data[1] = total_samples  # update sample count
            new_header = struct.pack(header_fmt, *header_data)
            output_path = Path(output_folder) / output_name
            with open(output_path, "wb") as f:
                f.write(new_header)
                f.write(merged_body)
            print(f"Merged {filename} → {output_path} (samples: {total_samples})")


# === CONFIGURATION ===
folders_to_merge = [
    "dataset/dataset_mnist_non_rotated",
    "dataset/rotated-45",
]

merged_output_folder = "merged_datasets/merged_nonrot_45"

# === RUN ===
merge_ubyte_files(folders_to_merge, merged_output_folder)
