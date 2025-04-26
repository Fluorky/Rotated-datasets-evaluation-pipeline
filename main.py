from datasets_handler import rotate_and_save_ranges, rotate_and_save_fixed_angle

if __name__ == '__main__':
    # === Config ===
    fixed_angle = 45
    angle_ranges = [(20, 50), (50, 90), (90, 120), (120,180)]

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
    for input_file in input_files:
        rotate_and_save_ranges(input_file, angle_ranges)
