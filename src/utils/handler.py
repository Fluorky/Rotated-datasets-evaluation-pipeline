import os
import json
import random
import shutil
import struct
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Tuple

import numpy as np
import cv2


def merge_ubyte_files(folders, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files_to_merge = [
        (["train-images-idx3-ubyte"], "train-images-idx3-ubyte", 16, ">IIII"),
        (["train-labels-idx1-ubyte"], "train-labels-idx1-ubyte", 8, ">II"),
        (["test-images-idx3-ubyte", "t10k-images-idx3-ubyte"], "test-images-idx3-ubyte", 16, ">IIII"),
        (["test-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"], "test-labels-idx1-ubyte", 8, ">II"),
    ]

    for candidate_names, output_name, header_size, header_fmt in files_to_merge:
        merged_body = b""
        total_samples = 0
        header_data = None

        for folder in folders:
            file_path = None
            for name in candidate_names:
                path = Path(folder) / name
                if path.exists():
                    file_path = path
                    break

            if not file_path:
                print(f"Missing: none of {candidate_names} in {folder}")
                continue

            with open(file_path, "rb") as f:
                header = f.read(header_size)
                body = f.read()

                if header_data is None:
                    header_data = list(struct.unpack(header_fmt, header))

                if "images" in output_name:
                    rows, cols = header_data[-2], header_data[-1]
                    sample_size = rows * cols
                else:
                    sample_size = 1

                samples = len(body) // sample_size
                total_samples += samples
                merged_body += body

        if header_data is not None:
            header_data[1] = total_samples
            new_header = struct.pack(header_fmt, *header_data)
            output_path = Path(output_folder) / output_name
            with open(output_path, "wb") as f:
                f.write(new_header)
                f.write(merged_body)
            print(f"Merged {output_name} → {output_path} (samples: {total_samples})")


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
    """
    images, num_images, rows, cols = load_mnist_images(input_path)
    rotated_images = rotate_images_by_angle(images, angle)

    # Compose full output path including angle folder and filename
    angle_folder = f"rotated-{angle}"
    full_output_path = os.path.join(os.path.dirname(os.path.dirname(output_path)), angle_folder,
                                    os.path.basename(output_path))

    save_mnist_images(full_output_path, rotated_images, num_images, rows, cols)

    # Copy corresponding label files
    copy_labels_to_folders(Path(input_path).parent, Path(full_output_path).parent)

    # Optional preview
    # plt.imshow(rotated_images[0], cmap='gray')
    plt.title(f"Rotated Image (angle={angle}°)")
    plt.axis("off")
    # plt.show()

    print(f"Rotated {num_images} images by {angle}° and saved to '{full_output_path}'")


def rotate_and_save_ranges(input_path: str, output_base: str, angle_ranges: list[tuple[int, int]], split: str):
    """
    Rotate an MNIST dataset using multiple angle ranges and save results to separate folders.

    :param input_path: Path to the original MNIST image file.
    :type input_path: str
    :type output_path: str
    :param output_base: Path to the output folder
    :param angle_ranges: List of angle ranges (min, max) in degrees.
                         Each range is applied randomly to images.
    :type angle_ranges: list[tuple[int, int]]
    :param split: Name of the dataset split
    :type split: str
    :return: None
    :rtype: None
    """
    images, num_images, rows, cols = load_mnist_images(input_path)

    for angle_range in angle_ranges:
        print(angle_range)
        range_str = f"{angle_range[0]}-{angle_range[1]}"
        output_result_path = f"{output_base}/rotated-{range_str}/{split}-images-idx3-ubyte"

        rotated_images = rotate_images(images, angle_range)
        save_mnist_images(output_result_path, rotated_images, num_images, rows, cols)
        copy_labels_to_folders(Path(input_path).parent, Path(output_result_path).parent)

        # Optional preview
        plt.imshow(rotated_images[0], cmap='gray')
        plt.title(f"Rotated (angle ∈ {angle_range}°)")
        plt.axis("off")
        # plt.show()

        print(f"Saved {num_images} images rotated in range {angle_range}° to '{output_result_path}'")


def copy_labels_to_folders(source_folder, target_folder):
    """
    Copy label files (train-labels and t10k-labels) from source_folder to each folder in target_folders.

    Args:
        source_folder (str): Path to the folder where label files are located.
        target_folder (str): List of target folder paths to copy labels into.
    """
    label_files = [
        ("test-labels-idx1-ubyte",),
        ("t10k-labels-idx1-ubyte",),
        ("train-labels-idx1-ubyte",),
    ]

    for filenames in label_files:
        for name in filenames:
            src = Path(source_folder) / name
            if src.exists():
                if "t10k" in name:
                    dst = Path(target_folder) / name.replace("t10k", "test")
                else:
                    dst = Path(target_folder) / name
                os.makedirs(target_folder, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"✅ Copied {src.name} → {dst.name}")
                break



# Function to make a safe folder name
def make_merge_name(folders):
    names = []
    for path in folders:
        last = os.path.basename(path)
        last = last.replace("dataset_mnist_", "")  # Remove common prefixes
        names.append(last)
    return "_".join(names)

def rename_t10k_to_test(dataset_dir: str):
    """Rename original MNIST t10k files to test equivalents."""
    mapping = {
        "t10k-images-idx3-ubyte": "test-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte": "test-labels-idx1-ubyte"
    }

    for old_name, new_name in mapping.items():
        old_path = Path(dataset_dir) / old_name
        new_path = Path(dataset_dir) / new_name
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            print(f"🔁 Renamed {old_name} → {new_name}")

def has_data_files(directory: str) -> bool:
    """
    Check if the directory contains required train/test data files in .npy or IDX format.
    """
    path = Path(directory)
    # Check for .npy files
    npy_files = [
        "train_images.npy", "train_labels.npy",
        "test_images.npy", "test_labels.npy",
    ]
    if all((path / f).exists() for f in npy_files):
        return True

    # Fallback: check for IDX format
    idx_files = [
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "test-images-idx3-ubyte", "test-labels-idx1-ubyte",
    ]
    return all((path / f).exists() for f in idx_files)


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
            if has_data_files(full_path):
                all_sets.append(full_path)

    # Sort for consistency
    all_sets = sorted(all_sets)

    result = {}

    for train_set in all_sets:
        # Normalize train set path for dictionary key
        train_set_name = os.path.relpath(train_set, merged_datasets_dir).replace("\\", "/")

        # Base tests always include the non-rotated dataset and the training set itself
        base_tests = {train_set_name}

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


def npy_paths(base_dir: str, dataset_name: str, split: str) -> Tuple[str, str]:
    """Return (images.npy, labels.npy) for a split."""
    images = os.path.join(base_dir, dataset_name, f"{split}_images.npy")
    labels = os.path.join(base_dir, dataset_name, f"{split}_labels.npy")
    return images, labels

def _npy_out_paths(root_out: str, split: str) -> Tuple[str, str]:
    """Return output (images.npy, labels.npy) under a given directory."""
    os.makedirs(root_out, exist_ok=True)
    return (os.path.join(root_out, f"{split}_images.npy"),
            os.path.join(root_out, f"{split}_labels.npy"))

def _load_npy_pair(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(images_path, mmap_mode=None)
    y = np.load(labels_path, mmap_mode=None)
    return X, y

def _save_npy_pair(images_out: str, labels_out: str, X: np.ndarray, y: np.ndarray) -> None:
    np.save(images_out, X)
    np.save(labels_out, y)

def _ensure_chw(X: np.ndarray) -> np.ndarray:
    """
    Accept HxW, HxWxC, or NxHxW(/C). Return NxCxHxW for rotation with OpenCV.
    """
    X = np.asarray(X)
    if X.ndim == 2:  # H, W -> 1 image
        X = X[None, None, :, :]
    elif X.ndim == 3:
        # Either N,H,W or H,W,C
        if X.shape[-1] in (1, 3):          # H,W,C  -> N=1
            X = np.transpose(X, (2, 0, 1))[None, ...]  # 1,C,H,W
        else:                                # N,H,W  (grayscale)
            X = X[:, None, :, :]             # N,1,H,W
    elif X.ndim == 4:
        # N,H,W,C or N,C,H,W
        if X.shape[-1] in (1, 3):
            X = np.transpose(X, (0, 3, 1, 2))  # N,C,H,W
        # else assume already N,C,H,W
    else:
        raise ValueError(f"Unsupported image array shape: {X.shape}")
    return X

def _rotate_image_cv(img_hw_or_hwc: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a single image (H,W) or (H,W,C) by angle (degrees, center rotation, keep size).
    Uses BORDER_CONSTANT (black) like common preprocessing.
    """
    if img_hw_or_hwc.ndim == 2:
        H, W = img_hw_or_hwc.shape
        C = None
    else:
        H, W, C = img_hw_or_hwc.shape

    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    border = cv2.BORDER_CONSTANT
    if C is None:
        return cv2.warpAffine(img_hw_or_hwc, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)
    else:
        # rotate each channel then stack
        chans = [cv2.warpAffine(img_hw_or_hwc[..., c], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=border)
                 for c in range(C)]
        return np.stack(chans, axis=-1)

def _rotate_batch_nchw(X: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a batch in NCHW and return NCHW. Internally converts per-image to HWC/HW.
    """
    N, C, H, W = X.shape
    out = np.empty_like(X)
    for i in range(N):
        xi = X[i]
        if C == 1:
            img = xi[0]                          # H,W
            rot = _rotate_image_cv(img, angle_deg)
            out[i, 0] = rot
        else:
            img = np.transpose(xi, (1, 2, 0))    # H,W,C
            rot = _rotate_image_cv(img, angle_deg)
            out[i] = np.transpose(rot, (2, 0, 1))
    return out

def rotate_and_save_fixed_angle_npy(
    images_in: str, labels_in: str, out_dir: str, split: str, angle: float
) -> None:
    """
    NPY: rotate all images by a fixed angle, write split_{images,labels}.npy.
    """
    images_out, labels_out = _npy_out_paths(out_dir, split)
    if os.path.exists(images_out) and os.path.exists(labels_out):
        print(f"⏩ Skipping existing: {out_dir}/{split} (angle {angle}°)")
        return
    X, y = _load_npy_pair(images_in, labels_in)
    X = _ensure_chw(X)  # N,C,H,W
    Xr = _rotate_batch_nchw(X, angle)
    # back to original “NPY convention”: default to NHW (grayscale) / NHWC (color)
    if Xr.shape[1] == 1:
        Xr_save = Xr[:, 0, :, :]                # N,H,W
    else:
        Xr_save = np.transpose(Xr, (0, 2, 3, 1)) # N,H,W,C
    images_out, labels_out = _npy_out_paths(out_dir, split)
    _save_npy_pair(images_out, labels_out, Xr_save, y)

def rotate_and_save_ranges_npy(
    images_in: str, labels_in: str, base_out: str, split: str, ranges: List[Tuple[int, int]], seed: int = 1337
) -> None:
    """
    NPY: for each range [a,b), create folder rotated-a-b and rotate each sample by
    one random angle uniformly drawn from that range. Deterministic via seed.
    """

    rng = np.random.default_rng(seed)
    X, y = _load_npy_pair(images_in, labels_in)
    X = _ensure_chw(X)  # N,C,H,W

    for (a, b) in ranges:
        out_dir = os.path.join(base_out, f"rotated-{a}-{b}")
        images_out, labels_out = _npy_out_paths(out_dir, split)

        if os.path.exists(images_out) and os.path.exists(labels_out):
            print(f"⏩ Skipping existing range {a}-{b}° for {split}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        # draw per-image angle
        angles = rng.uniform(low=a, high=b, size=(X.shape[0],)).astype(np.float32)
        # rotate one by one (keeps memory bounded)
        Xr = np.empty_like(X)
        for i in range(X.shape[0]):
            Xr[i] = _rotate_batch_nchw(X[i:i+1], float(angles[i]))[0]
        # save back in NHW/NHWC style
        if Xr.shape[1] == 1:
            Xr_save = Xr[:, 0, :, :]
        else:
            Xr_save = np.transpose(Xr, (0, 2, 3, 1))
        images_out, labels_out = _npy_out_paths(out_dir, split)
        _save_npy_pair(images_out, labels_out, Xr_save, y)

def merge_npy_dirs(input_dirs: List[str], out_dir: str, splits: List[str]) -> None:
    """
    Concatenate NPY (images, labels) across multiple source dirs for each split.
    Expects each dir to contain split_images.npy and split_labels.npy.
    """
    if os.path.exists(out_dir):
        print(f"⏩ Skipping merge: {out_dir} already exists.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        xs, ys = [], []
        for d in input_dirs:
            xi = os.path.join(d, f"{split}_images.npy")
            yi = os.path.join(d, f"{split}_labels.npy")
            if not (os.path.isfile(xi) and os.path.isfile(yi)):
                # silently skip missing dirs (mirrors IDX merging tolerance)
                continue
            X, y = _load_npy_pair(xi, yi)
            xs.append(X)
            ys.append(y)
        if not xs:
            continue
        X_cat = np.concatenate(xs, axis=0)
        y_cat = np.concatenate(ys, axis=0)
        np.save(os.path.join(out_dir, f"{split}_images.npy"), X_cat)
        np.save(os.path.join(out_dir, f"{split}_labels.npy"), y_cat)
