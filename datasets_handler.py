import os
import random
import struct
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple


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


def load_mnist_images(filename: str) -> Tuple[np.ndarray, int, int, int]:
    """Loads MNIST images from an IDX3-UBYTE file."""
    with open(filename, 'rb') as f:
        # Read the header information
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    return images, num_images, rows, cols


def rotate_images_by_angle(images: np.ndarray, angle: float) -> np.ndarray:
    """Rotates each image in the dataset by a given angle."""
    rotated_images = []
    for img in images:
        pil_img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        rotated_img = pil_img.rotate(angle)  # Rotate the image
        rotated_images.append(np.array(rotated_img, dtype=np.uint8))  # Convert back to NumPy array

    return np.array(rotated_images)


def rotate_images(images: np.ndarray, angle_range: Tuple[float, float]) -> np.ndarray:
    rotated_images = []
    for img in images:
        angle = random.uniform(*angle_range)
        pil_img = Image.fromarray(img)
        rotated_img = pil_img.rotate(angle)
        rotated_images.append(np.array(rotated_img, dtype=np.uint8))
    return np.array(rotated_images)


def save_mnist_images(filename: str, images: np.ndarray, num_images: int, rows: int, cols: int) -> None:
    """Saves images into an IDX3-UBYTE file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        # Write header (magic nugitmber, number of images, rows, columns)
        f.write(struct.pack(">IIII", 2051, num_images, rows, cols))

        # Write image data
        f.write(images.tobytes())


def rotate_and_save_fixed_angle(input_path: str, output_path: str, angle: float):
    """
    Rotate an MNIST dataset by a fixed angle and save the result to a specified path.

    :param input_path: Path to the original MNIST image file.
    :type input_path: str
    :param output_path: Path where the rotated images will be saved.
    :type output_path: str
    :param angle: Angle in degrees by which to rotate each image.
    :type angle: float

    :return: None
    :rtype: None
    """
    images, num_images, rows, cols = load_mnist_images(input_path)
    rotated_images = rotate_images_by_angle(images, angle)
    save_mnist_images(output_path, rotated_images, num_images, rows, cols)

    # Optional preview
    # plt.imshow(rotated_images[0], cmap='gray')
    plt.title(f"Rotated Image (angle={angle}°)")
    plt.axis("off")
    # plt.show()

    print(f"Rotated {num_images} images by {angle}° and saved to '{output_path}'")


def rotate_and_save_ranges(input_path: str, angle_ranges: list[tuple[int, int]]):
    """
    Rotate an MNIST dataset using multiple angle ranges and save results to separate folders.

    :param input_path: Path to the original MNIST image file.
    :type input_path: str
    :param angle_ranges: List of angle ranges (min, max) in degrees.
                         Each range is applied randomly to images.
    :type angle_ranges: list[tuple[int, int]]

    :return: None
    :rtype: None
    """
    images, num_images, rows, cols = load_mnist_images(input_path)

    for angle_range in angle_ranges:
        range_str = f"{angle_range[0]}-{angle_range[1]}"
        output_path = f"dataset/rotated-{range_str}/{os.path.basename(input_path)}"

        rotated_images = rotate_images(images, angle_range)
        save_mnist_images(output_path, rotated_images, num_images, rows, cols)

        # Optional preview
        plt.imshow(rotated_images[0], cmap='gray')
        plt.title(f"Rotated (angle ∈ {angle_range}°)")
        plt.axis("off")
        plt.show()

        print(f"Saved {num_images} images rotated in range {angle_range}° to '{output_path}'")


# === CONFIGURATION ===
folders_to_merge = [
    "dataset/dataset_mnist_non_rotated",
    "dataset/rotated-45",
]

merged_output_folder = "merged_datasets/merged_nonrot_45"

# === RUN ===
merge_ubyte_files(folders_to_merge, merged_output_folder)
