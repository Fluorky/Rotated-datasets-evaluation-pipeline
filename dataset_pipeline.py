import os
from pathlib import Path
from datasets_handler import (
    rotate_and_save_ranges,
    rotate_and_save_fixed_angle,
    merge_ubyte_files,
    make_merge_name,
    generate_merging_scenarios,
    generate_train_test_scenarios
)


def rotate_fixed_angles(base_dir, angles):
    for angle in angles:
        for split in ["train", "t10k"]:
            input_file = f"{base_dir}/dataset_mnist_non_rotated/{split}-images-idx3-ubyte"
            output_file = f"{base_dir}/rotated-{angle}/{split}-images-idx3-ubyte"
            rotate_and_save_fixed_angle(input_file, output_file, angle)


def rotate_angle_ranges(base_dir, ranges):
    input_files = [
        f"{base_dir}/dataset_mnist_non_rotated/train-images-idx3-ubyte",
        f"{base_dir}/dataset_mnist_non_rotated/t10k-images-idx3-ubyte"
    ]
    for file in input_files:
        rotate_and_save_ranges(file, base_dir, ranges)


# def run_merge_scenarios(base_dir, output_dir, folders):
#     os.makedirs(output_dir, exist_ok=True)
#     scenarios = generate_merging_scenarios(folders)
#     print(f"📦 Generated {len(scenarios)} scenarios.")
#     for i, group in enumerate(scenarios):
#         merged_name = make_merge_name(group)
#         target_path = os.path.join(output_dir, f"merged_{merged_name}")
#         print(f"🔄 [{i}] Merging: {group} -> {target_path}")
#         merge_ubyte_files(group, target_path)


def predefined_merges(base_dir, output_dir, angle_ranges):
    def d(angle): return f"{base_dir}/rotated-{angle}"
    def dr(a, b): return f"{base_dir}/rotated-{a}-{b}"
    def base(): return f"{base_dir}/dataset_mnist_non_rotated"

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


def run_pipeline():
    base_dir = "dataset/MNIST_copy"
    merged_dir = os.path.join(base_dir, "merged_datasets")
    angle_ranges = [(i, i + 30) for i in range(0, 360, 30)]
    fixed_angles = sorted(set().union(range(30, 360, 30), range(45, 360, 45)))

    print("🌀 Rotating fixed angles...")
    rotate_fixed_angles(base_dir, fixed_angles)

    print("🌀 Rotating angle ranges...")
    rotate_angle_ranges(base_dir, angle_ranges)

    # print("📂 Running merge scenarios...")
    # merge_sources = [
    #     f"{base_dir}/dataset_mnist_non_rotated",
    #     f"{base_dir}/rotated-20-50",
    #     f"{base_dir}/rotated-45",
    #     f"{base_dir}/rotated-50-90",
    #     f"{base_dir}/rotated-90-120",
    #     f"{base_dir}/rotated-120-150",
    #     f"{base_dir}/rotated-120-180",
    #     f"{base_dir}/rotated-150-180"
    # ]
    # run_merge_scenarios(base_dir, merged_dir, merge_sources)

    print("🔧 Running predefined merges...")
    predefined_merges(base_dir, merged_dir, angle_ranges)

    print("🧪 Generating train-test JSON...")
    generate_train_test_scenarios(
        merged_datasets_dir="./dataset/MNIST_copy/",
        output_json_path="./train_test_scenariosv11.json",
        max_tests=2000
    )
    print("✅ All preprocessing completed.")


if __name__ == "__main__":
    run_pipeline()
