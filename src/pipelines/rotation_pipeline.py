import os
from src.utils.handler import (
    rotate_and_save_ranges,          # IDX
    rotate_and_save_fixed_angle,     # IDX
    merge_ubyte_files,               # IDX
    generate_train_test_scenarios,   # agnostic – it walks dirs
    rename_t10k_to_test,              # IDX/MNIST
    npy_paths,
    rotate_and_save_fixed_angle_npy,
    rotate_and_save_ranges_npy,
    merge_npy_dirs
)


def rotate_fixed_angles(base_dir, dataset_name, angles, splits, file_format="ubyte"):
    """Rotate dataset by fixed angles for all splits."""
    if file_format == "npy":
        # NPY flow: use split_images.npy + split_labels.npy
        for angle in angles:
            out_dir = os.path.join(base_dir, f"rotated-{angle}")
            os.makedirs(out_dir, exist_ok=True)
            for split in splits:
                images_in, labels_in = npy_paths(base_dir, dataset_name, split)
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
            images_in, labels_in = npy_paths(base_dir, dataset_name, split)
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
