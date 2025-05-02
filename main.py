import os
import itertools
from datasets_handler import rotate_and_save_ranges, rotate_and_save_fixed_angle, merge_ubyte_files, \
    merge_train_or_test, make_merge_name, generate_merging_scenarios

if __name__ == '__main__':
    # === Config ===
    fixed_angle = 45
    angle_ranges = [(0, 20), (20, 50), (50, 90), (90, 120), (120, 150), (150, 180)]

    # === Fixed-angle processing ===
    rotate_and_save_fixed_angle(
        input_path="dataset/MNIST/dataset_mnist_non_rotated/t10k-images-idx3-ubyte",
        output_path=f"dataset/MNIST/rotated-{fixed_angle}/t10k-images-idx3-ubyte",
        angle=fixed_angle
    )

    rotate_and_save_fixed_angle(
        input_path="dataset/MNIST/dataset_mnist_non_rotated/train-images-idx3-ubyte",
        output_path=f"dataset/MNIST/rotated-{fixed_angle}/train-images-idx3-ubyte",
        angle=fixed_angle
    )

    # === Multi-range processing ===
    input_files = ["dataset/MNIST/dataset_mnist_non_rotated/t10k-images-idx3-ubyte",
                   "dataset/MNIST/dataset_mnist_non_rotated/train-images-idx3-ubyte"]
    output_path = "dataset/MNIST"
    for input_file in input_files:
        rotate_and_save_ranges(input_file, output_path, angle_ranges)
        # copy_labels_to_folders("dataset/MNIST/dataset_mnist_non_rotated")

    # === MERGING ===
    folders_to_merge = [
        "dataset/MNIST/dataset_mnist_non_rotated",
        "dataset/MNIST/rotated-45",
    ]

    merged_output_folder = "dataset/MNIST/merged_datasets/merged_nonrot_45"

    # === RUN ===
    merge_ubyte_files(folders_to_merge, merged_output_folder)

    # List of folders
    all_folders = [
        "dataset/MNIST/dataset_mnist_non_rotated",
        # "dataset/MNIST/rotated-0-20",
        "dataset/MNIST/rotated-20-50",
        "dataset/MNIST/rotated-45",
        "dataset/MNIST/rotated-50-90",
        "dataset/MNIST/rotated-90-120",
        # "dataset/MNIST/rotated-120-150",
        # "dataset/MNIST/rotated-120-180",
        # "dataset/MNIST/rotated-150-180",
    ]

    # Destination base folder
    merged_base = "dataset/MNIST/merged_datasets"

    # Get all merging scenarios
    scenarios = generate_merging_scenarios(all_folders)

    print(f"Generated {len(scenarios)} scenarios.")

    # Example usage
    i = 0
    for folders_to_merge in scenarios:
        output_name = make_merge_name(folders_to_merge)
        merged_output_folder = os.path.join(merged_base, f"merged_{output_name}")

        print(f"{i}Merging: {folders_to_merge} -> {merged_output_folder}")
        i = i + 1
        merge_ubyte_files(folders_to_merge, merged_output_folder)

    merge_output_root = "dataset/MNIST/merged_sets"

    # For each single folder – test
    for test_folder in all_folders:
        test_name = os.path.basename(test_folder)
        out_test_path = os.path.join(merge_output_root, f"test_{test_name}")
        merge_train_or_test([test_folder], out_test_path, train=False)

    # For each 2+ folders combination – train
    from itertools import combinations

    for n in range(2, len(all_folders) + 1):
        for train_combo in combinations(all_folders, n):
            train_names = "_".join([os.path.basename(f) for f in train_combo])
            out_train_path = os.path.join(merge_output_root, f"train_{train_names}")
            merge_train_or_test(list(train_combo), out_train_path, train=True)
