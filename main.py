import os
from datasets_handler import (
    rotate_and_save_ranges,
    rotate_and_save_fixed_angle,
    merge_ubyte_files,
    make_merge_name,
    generate_merging_scenarios,
    generate_train_test_scenarios
)

if __name__ == '__main__':
    # === Config ===
    fixed_angle = 45
    angle_ranges = [(0, 20), (20, 50), (50, 90), (90, 120), (120, 150), (150, 180)]

    base_dir = "dataset/MNIST"
    merged_dir = os.path.join(base_dir, "merged_datasets")

    os.makedirs(merged_dir, exist_ok=True)

    # === Fixed-angle rotation ===
    rotate_and_save_fixed_angle(
        input_path=f"{base_dir}/dataset_mnist_non_rotated/t10k-images-idx3-ubyte",
        output_path=f"{base_dir}/rotated-{fixed_angle}/t10k-images-idx3-ubyte",
        angle=fixed_angle
    )
    rotate_and_save_fixed_angle(
        input_path=f"{base_dir}/dataset_mnist_non_rotated/train-images-idx3-ubyte",
        output_path=f"{base_dir}/rotated-{fixed_angle}/train-images-idx3-ubyte",
        angle=fixed_angle
    )

    # === Multi-range rotation ===
    input_files = [
        f"{base_dir}/dataset_mnist_non_rotated/t10k-images-idx3-ubyte",
        f"{base_dir}/dataset_mnist_non_rotated/train-images-idx3-ubyte"
    ]
    for input_file in input_files:
        rotate_and_save_ranges(input_file, base_dir, angle_ranges)

    # === Folders to consider ===
    all_folders = [
        f"{base_dir}/dataset_mnist_non_rotated",
        f"{base_dir}/rotated-20-50",
        f"{base_dir}/rotated-45",
        f"{base_dir}/rotated-50-90",
        f"{base_dir}/rotated-90-120"
    ]

    # === Merge scenarios ===
    scenarios = generate_merging_scenarios(all_folders)
    print(f"Generated {len(scenarios)} scenarios.")

    for i, folders_to_merge in enumerate(scenarios):
        output_name = make_merge_name(folders_to_merge)
        merged_output_folder = os.path.join(merged_dir, f"merged_{output_name}")
        print(f"{i} Merging: {folders_to_merge} -> {merged_output_folder}")
        merge_ubyte_files(folders_to_merge, merged_output_folder)

    generate_train_test_scenarios(
        merged_datasets_dir="./dataset/MNIST/merged_datasets",
        output_json_path="./train_test_scenarios.json",
        max_tests=20
    )