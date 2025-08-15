import argparse
import os
import json
from typing import List, Tuple

import numpy as np
import cv2

# --- Existing IDX utilities (unchanged) ---
# Keep these imports for IDX flow exactly as you have them.
from src.utils.handler import (
    rotate_and_save_ranges,          # IDX
    rotate_and_save_fixed_angle,     # IDX
    merge_ubyte_files,               # IDX
    generate_train_test_scenarios,   # agnostic – it walks dirs
    rename_t10k_to_test              # IDX/MNIST
)

# =========================
# NPY helpers (local)
# =========================

def _npy_paths(base_dir: str, dataset_name: str, split: str) -> Tuple[str, str]:
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

# =========================
# Format-agnostic wrappers
# =========================

def rotate_fixed_angles(base_dir, dataset_name, angles, splits, file_format="ubyte"):
    """Rotate dataset by fixed angles for all splits."""
    if file_format == "npy":
        # NPY flow: use split_images.npy + split_labels.npy
        for angle in angles:
            out_dir = os.path.join(base_dir, f"rotated-{angle}")
            os.makedirs(out_dir, exist_ok=True)
            for split in splits:
                images_in, labels_in = _npy_paths(base_dir, dataset_name, split)
                rotate_and_save_fixed_angle_npy(images_in, labels_in, out_dir, split, angle)
    else:
        # IDX flow (legacy)
        suffix = "-images-idx3-ubyte"
        for angle in angles:
            for split in splits:
                input_file = os.path.join(base_dir, dataset_name, f"{split}{suffix}")
                out_dir = os.path.join(base_dir, f"rotated-{angle}")
                os.makedirs(out_dir, exist_ok=True)
                output_file = os.path.join(out_dir, f"{split}{suffix}")
                rotate_and_save_fixed_angle(input_file, output_file, angle)

def rotate_angle_ranges(base_dir, dataset_name, ranges, splits, file_format="ubyte"):
    """Rotate dataset across a range of angles for all splits."""
    if file_format == "npy":
        for split in splits:
            images_in, labels_in = _npy_paths(base_dir, dataset_name, split)
            rotate_and_save_ranges_npy(images_in, labels_in, base_out=base_dir, split=split, ranges=ranges)
    else:
        suffix = "-images-idx3-ubyte"
        for split in splits:
            input_file = os.path.join(base_dir, dataset_name, f"{split}{suffix}")
            rotate_and_save_ranges(input_file, base_dir, ranges, split, suffix=suffix)

def predefined_merges(base_dir, dataset_name, output_dir, angle_ranges, file_format="ubyte"):
    """Merge rotated datasets into defined presets."""
    def d(angle): return os.path.join(base_dir, f"rotated-{angle}")
    def dr(a, b): return os.path.join(base_dir, f"rotated-{a}-{b}")
    def base(): return os.path.join(base_dir, dataset_name)

    fixed_30 = list(range(30, 360, 30))
    fixed_45 = list(range(45, 360, 45))
    range_paths = lambda rs: [dr(a, b) for a, b in rs]

    merge_configs = {
        "merged_fixed_30": [d(a) for a in fixed_30],
        "merged_fixed_45": [d(a) for a in fixed_45],
        "merged_fixed_all": [d(a) for a in sorted(set(fixed_30 + fixed_45))],
        "merged_range_0_90": range_paths(angle_ranges[:3]),
        "merged_range_90_180": range_paths(angle_ranges[3:6]),
        "merged_range_180_270": range_paths(angle_ranges[6:9]),
        "merged_range_270_360": range_paths(angle_ranges[9:]),
        "merged_range_0_180": range_paths(angle_ranges[:6]),
        "merged_range_180_360": range_paths(angle_ranges[6:]),
        "merged_range_full_0_360": range_paths(angle_ranges),
        "merged_range_full_0_360_plus_non_rotated": [base()] + range_paths(angle_ranges),
        "merged_fixed_30_plus_non_rotated": [base()] + [d(a) for a in fixed_30],
        "merged_fixed_45_plus_non_rotated": [base()] + [d(a) for a in fixed_45],
        "merged_range_0_180_plus_non_rotated": [base()] + range_paths(angle_ranges[:6]),
        "merged_range_180_360_plus_non_rotrotated": [base()] + range_paths(angle_ranges[6:]),
    }

    for name, paths in merge_configs.items():
        print(f"\n📁 Merging preset: {name}")
        for p in paths:
            print(f"  - {p}")
        out_dir = os.path.join(output_dir, name)
        if file_format == "npy":
            merge_npy_dirs(paths, out_dir, splits=["train", "test"])
        else:
            merge_ubyte_files(paths, out_dir)


def run_pipeline(base_dir: str, dataset_name: str, dataset_key: str,
                 merged_dir_name: str = "merged_datasets", max_tests: int = 2000,
                 file_format: str = "ubyte"):
    """Run the full rotation and merging pipeline for a given dataset and file format."""
    angle_ranges = [(i, i + 30) for i in range(0, 360, 30)]
    fixed_angles = sorted(set(range(30, 360, 30)).union(range(45, 360, 45)))
    splits = ["train", "test"]

    merged_dir = os.path.join(base_dir, merged_dir_name)
    os.makedirs(merged_dir, exist_ok=True)

    # Only rename MNIST t10k -> test for IDX flows
    if dataset_key.startswith("MNIST") and file_format != "npy":
        rename_t10k_to_test(os.path.join(base_dir, dataset_name))

    print("🌀 Rotating fixed angles...")
    rotate_fixed_angles(base_dir, dataset_name, fixed_angles, splits, file_format)

    print("🌀 Rotating angle ranges...")
    rotate_angle_ranges(base_dir, dataset_name, angle_ranges, splits, file_format)

    print("🔧 Running predefined merges...")
    predefined_merges(base_dir, dataset_name, merged_dir, angle_ranges, file_format)

    json_out_name = f"train_test_scenarios_{dataset_name.replace('dataset_', '')}.json"
    output_json_path = os.path.join("configs", "scenarios", json_out_name)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    print("🧪 Generating train-test JSON...")
    generate_train_test_scenarios(
        merged_datasets_dir=base_dir,
        output_json_path=output_json_path,
        max_tests=max_tests
    )

    print(f"✅ All preprocessing completed. JSON saved as {output_json_path}")
