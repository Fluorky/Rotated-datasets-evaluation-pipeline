import argparse
import os

from src.utils.handler import (
    rotate_and_save_ranges,
    rotate_and_save_fixed_angle,
    merge_ubyte_files,
    generate_train_test_scenarios,
    rename_t10k_to_test
)

def rotate_fixed_angles(base_dir, dataset_name, angles, splits):
    """Rotate dataset by fixed angles for all splits."""
    for angle in angles:
        for split in splits:
            input_file = os.path.join(base_dir, dataset_name, f"{split}-images-idx3-ubyte")
            output_file = os.path.join(base_dir, f"rotated-{angle}", f"{split}-images-idx3-ubyte")
            rotate_and_save_fixed_angle(input_file, output_file, angle)


def rotate_angle_ranges(base_dir, dataset_name, ranges, splits):
    """Rotate dataset across a range of angles for all splits."""
    for split in splits:
        input_file = os.path.join(base_dir, dataset_name, f"{split}-images-idx3-ubyte")
        rotate_and_save_ranges(input_file, base_dir, ranges, split)


def predefined_merges(base_dir, dataset_name, output_dir, angle_ranges):
    """Merge rotated datasets into defined presets."""

    def d(angle): return os.path.join(base_dir, f"rotated-{angle}")
    def dr(a, b): return os.path.join(base_dir, f"rotated-{a}-{b}")
    def base(): return os.path.join(base_dir, dataset_name)

    fixed_30 = list(range(30, 360, 30))
    fixed_45 = list(range(45, 360, 45))
    range_paths = lambda rs: [dr(a, b) for a, b in rs]

    merge_configs = {
        # "merged_non_rotated": [base()],
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
        "merged_range_180_360_plus_non_rotated": [base()] + range_paths(angle_ranges[6:]),
    }

    for name, paths in merge_configs.items():
        print(f"\n📁 Merging preset: {name}")
        for p in paths:
            print(f"  - {p}")
        merge_ubyte_files(paths, os.path.join(output_dir, name))


def run_pipeline(base_dir: str, dataset_name: str, dataset_key: str,
                 merged_dir_name: str = "merged_datasets", max_tests: int = 2000):
    """Run the full rotation and merging pipeline for a given dataset."""
    angle_ranges = [(i, i + 30) for i in range(0, 360, 30)]
    fixed_angles = sorted(set(range(30, 360, 30)).union(range(45, 360, 45)))
    splits = ["train", "test"]

    merged_dir = os.path.join(base_dir, merged_dir_name)
    os.makedirs(merged_dir, exist_ok=True)

    if dataset_key == "MNIST":
        rename_t10k_to_test(os.path.join(base_dir, dataset_name))

    print("🌀 Rotating fixed angles...")
    rotate_fixed_angles(base_dir, dataset_name, fixed_angles, splits)

    print("🌀 Rotating angle ranges...")
    rotate_angle_ranges(base_dir, dataset_name, angle_ranges, splits)

    print("🔧 Running predefined merges...")
    predefined_merges(base_dir, dataset_name, merged_dir, angle_ranges)

    json_out_name = f"train_test_scenarios_{dataset_name.replace('dataset_', '')}.json"
    output_json_path = os.path.join("../..", json_out_name)

    print("🧪 Generating train-test JSON...")
    generate_train_test_scenarios(
        merged_datasets_dir=base_dir,
        output_json_path=output_json_path,
        max_tests=max_tests
    )

    print(f"✅ All preprocessing completed. JSON saved as {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate, merge, and prepare dataset splits.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["GTSRB", "MNIST", "LEGO"],
        required=True,
        help="Name of the dataset directory under ./dataset/"
    )
    args = parser.parse_args()

    dataset_config = {
        "MNIST": "dataset_mnist_non_rotated",
        "GTSRB": "dataset_GTSRB_non_rotated",
        "LEGO": "dataset_LEGO_non_rotated"
    }

    run_pipeline(
        base_dir=os.path.join("../../dataset", args.dataset),
        dataset_name=dataset_config[args.dataset],
        dataset_key=args.dataset
    )
