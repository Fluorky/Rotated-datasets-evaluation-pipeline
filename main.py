from datasets_handler import rotate_and_save_ranges, rotate_and_save_fixed_angle, merge_ubyte_files

if __name__ == '__main__':
    # === Config ===
    fixed_angle = 45
    angle_ranges = [(0,20), (20, 50), (50, 90), (90, 120), (120, 150), (150, 180)]

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
    input_files = ["dataset/MNIST/dataset_mnist_non_rotated/t10k-images-idx3-ubyte", "dataset/MNIST/dataset_mnist_non_rotated/train-images-idx3-ubyte"]
    output_path = "dataset/MNIST"
    for input_file in input_files:
        rotate_and_save_ranges(input_file, output_path, angle_ranges)

    # === MERGING ===
    folders_to_merge = [
        "dataset/MNIST/dataset_mnist_non_rotated",
        "dataset/MNIST/rotated-45",
    ]

    merged_output_folder = "dataset/MNIST/merged_datasets/merged_nonrot_45"

    # === RUN ===
    merge_ubyte_files(folders_to_merge, merged_output_folder)
