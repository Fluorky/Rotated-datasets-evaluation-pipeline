import os
import json
import random
import shutil
import struct
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict, List

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


def rotate_and_save_ranges(input_path: str, output_path: str, angle_ranges: list[tuple[int, int]]):
    """
    Rotate an MNIST dataset using multiple angle ranges and save results to separate folders.

    :param input_path: Path to the original MNIST image file.
    :type input_path: str
    :type output_path: str
    :param angle_ranges: List of angle ranges (min, max) in degrees.
                         Each range is applied randomly to images.
    :type angle_ranges: list[tuple[int, int]]

    :return: None
    :rtype: None
    """
    images, num_images, rows, cols = load_mnist_images(input_path)

    for angle_range in angle_ranges:
        print(angle_range)
        range_str = f"{angle_range[0]}-{angle_range[1]}"
        output_result_path = f"{output_path}/rotated-{range_str}/{os.path.basename(input_path)}"

        rotated_images = rotate_images(images, angle_range)
        save_mnist_images(output_result_path, rotated_images, num_images, rows, cols)
        copy_labels_to_folders(Path(input_path).parent, Path(output_result_path).parent)

        # Optional preview
        plt.imshow(rotated_images[0], cmap='gray')
        plt.title(f"Rotated (angle ∈ {angle_range}°)")
        plt.axis("off")
        plt.show()

        print(f"Saved {num_images} images rotated in range {angle_range}° to '{output_result_path}'")


def copy_labels_to_folders(source_folder, folder):
    """
    Copy label files (train-labels and t10k-labels) from source_folder to each folder in target_folders.

    Args:
        source_folder (str): Path to the folder where label files are located.
        folder (str): List of target folder paths to copy labels into.
    """
    label_files = [
        "train-labels-idx1-ubyte",
        "t10k-labels-idx1-ubyte"
    ]

    for label_file in label_files:
        source_path = Path(source_folder) / label_file

        if not source_path.exists():
            print(f"⚠️ Label file not found: {source_path}")
            continue

        target_path = Path(folder) / label_file
        os.makedirs(folder, exist_ok=True)  # ensure target folder exists

        shutil.copy2(source_path, target_path)
        print(f"✅ Copied {label_file} to {target_path}")


# Function to make a safe folder name
def make_merge_name(folders):
    names = []
    for path in folders:
        last = os.path.basename(path)
        last = last.replace("dataset_mnist_", "")  # Remove common prefixes
        names.append(last)
    return "_".join(names)


# Function to create all combinations
def generate_merging_scenarios(all_folders):
    scenarios = []
    for r in range(2, len(all_folders) + 1):  # All combinations of 2 to all
        for combo in itertools.combinations(all_folders, r):
            scenarios.append(combo)
    return scenarios

def generate_train_test_scenarios(
    merged_datasets_dir: str,
    output_json_path: str,
    max_tests: int = 20,
    seed: int = 42
):
    """
    Generates a JSON with test scenarios per training set.
    Ensures 'dataset_mnist_non_rotated' and the training set itself are included in the test list.
    Adds random diverse test sets (up to `max_tests` in total).
    """
    random.seed(seed)

    # Recursively find all directories in the merged_datasets_dir
    all_sets = []
    for root, dirs, _ in os.walk(merged_datasets_dir):
        for d in dirs:
            full_path = os.path.join(root, d)
            if os.path.isdir(full_path):
                all_sets.append(full_path)

    # Sort for consistency
    all_sets = sorted(all_sets)

    result = {}

    for train_set in all_sets:
        # Normalize train set path for dictionary key
        train_set_name = os.path.relpath(train_set, merged_datasets_dir).replace("\\", "/")

        # Base tests always include the non-rotated dataset and the training set itself
        base_tests = {"dataset_mnist_non_rotated", train_set_name}

        # Generate a random selection of additional test sets
        candidates = [os.path.relpath(s, merged_datasets_dir).replace("\\", "/") for s in all_sets if s != train_set]
        random.shuffle(candidates)

        # Create the final list of test sets
        selected_tests = list(base_tests)
        for test in candidates:
            if len(selected_tests) >= max_tests:
                break
            selected_tests.append(test)

        # Sort for consistency and assign to result
        result[train_set_name] = sorted(selected_tests)

    # Save to JSON file
    with open(output_json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Train-test scenarios saved to {output_json_path}")
