import os
import argparse
from pathlib import Path
from datasets_handler import (
    rotate_and_save_ranges,
    rotate_and_save_fixed_angle,
    merge_ubyte_files,
    make_merge_name,
    generate_merging_scenarios,
    generate_train_test_scenarios
)


def rotate_fixed_angles(base_dir, dataset_name, angles):
    for angle in angles:
        for split in ["train", "t10k"]:
            input_file = f"{base_dir}/{dataset_name}/{split}-images-idx3-ubyte"
            output_file = f"{base_dir}/rotated-{angle}/{split}-images-idx3-ubyte"
            rotate_and_save_fixed_angle(input_file, output_file, angle)


def rotate_angle_ranges(base_dir, dataset_name, ranges):
    input_files = [
        f"{base_dir}/{dataset_name}/train-images-idx3-ubyte",
        f"{base_dir}/{dataset_name}/t10k-images-idx3-ubyte"
    ]
    for file in input_files:
        rotate_and_save_ranges(file, base_dir, ranges)


def predefined_merges(base_dir, dataset_name, output_dir, angle_ranges):
    def d(angle): return f"{base_dir}/rotated-{angle}"
    def dr(a, b): return f"{base_dir}/rotated-{a}-{b}"
    def base(): return f"{base_dir}/{dataset_name}"

    fixed_30 = list(range(30, 360, 30))
    fixed_45 = list(range(45, 360, 45))
    range_paths = lambda rs: [dr(a, b) for a, b in rs]

    merge_configs = {
        "merged_non_rotated": [base()],
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


def run_pipeline(base_dir: str, dataset_name: str, merged_dir_name="merged_datasets", max_tests=2000):
    angle_ranges = [(i, i + 30) for i in range(0, 360, 30)]
    fixed_angles = sorted(set().union(range(30, 360, 30), range(45, 360, 45)))

    merged_dir = os.path.join(base_dir, merged_dir_name)
    os.makedirs(merged_dir, exist_ok=True)

    print("🌀 Rotating fixed angles...")
    rotate_fixed_angles(base_dir, dataset_name, fixed_angles)

    print("🌀 Rotating angle ranges...")
    rotate_angle_ranges(base_dir, dataset_name, angle_ranges)

    print("🔧 Running predefined merges...")
    predefined_merges(base_dir, dataset_name, merged_dir, angle_ranges)

    json_out_name = f"train_test_scenarios_{dataset_name.replace('dataset_', '')}.json"
    output_json_path = os.path.join(".", json_out_name)

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
        choices=["GTSRB", "MNIST_copy"],
        required=True,
        help="Name of the dataset directory under ./dataset/"
    )
    args = parser.parse_args()

    # Map dataset name to expected subdir (folder with IDX files)
    dataset_config = {
        "MNIST_copy": "dataset_mnist_non_rotated",
        "GTSRB": "dataset_GTSRB_non_rotated"
    }

    run_pipeline(
        base_dir=f"dataset/{args.dataset}",
        dataset_name=dataset_config[args.dataset]
    )
