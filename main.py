import os
import matplotlib.pyplot as plt
from rotate_dateset import load_mnist_images, rotate_images, rotate_images_by_angle, save_mnist_images


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
    :rtype: None
    """
    images, num_images, rows, cols = load_mnist_images(input_path)
    rotated_images = rotate_images_by_angle(images, angle)
    save_mnist_images(output_path, rotated_images, num_images, rows, cols)

    # Optional preview
    # plt.imshow(rotated_images[0], cmap='gray')
    plt.title(f"Rotated Image (angle={angle}°)")
    plt.axis("off")
    # plt.show()

    print(f"Rotated {num_images} images by {angle}° and saved to '{output_path}'")


def rotate_and_save_ranges(input_path: str, angle_ranges: list[tuple[int, int]]):
    """
    Rotate an MNIST dataset using multiple angle ranges and save results to separate folders.

    :param input_path: Path to the original MNIST image file.
    :type input_path: str
    :param angle_ranges: List of angle ranges (min, max) in degrees.
                         Each range is applied randomly to images.
    :type angle_ranges: list[tuple[int, int]]

    :return: None
    :rtype: None
    """
    images, num_images, rows, cols = load_mnist_images(input_path)

    for angle_range in angle_ranges:
        range_str = f"{angle_range[0]}-{angle_range[1]}"
        output_path = f"dataset/rotated-{range_str}/{os.path.basename(input_path)}"

        rotated_images = rotate_images(images, angle_range)
        save_mnist_images(output_path, rotated_images, num_images, rows, cols)

        # Optional preview
        plt.imshow(rotated_images[0], cmap='gray')
        plt.title(f"Rotated (angle ∈ {angle_range}°)")
        plt.axis("off")
        plt.show()

        print(f"Saved {num_images} images rotated in range {angle_range}° to '{output_path}'")


if __name__ == '__main__':
    # === Config ===
    fixed_angle = 45
    angle_ranges = [(20, 50), (50, 90), (90, 120), (120,180)]

    # === Fixed-angle processing ===
    rotate_and_save_fixed_angle(
        input_path="dataset/dataset_mnist_non_rotated/t10k-images-idx3-ubyte",
        output_path="dataset/rotated-45/t10k-images-idx3-ubyte",
        angle=fixed_angle
    )

    rotate_and_save_fixed_angle(
        input_path="dataset/dataset_mnist_non_rotated/train-images-idx3-ubyte",
        output_path="dataset/rotated-45/train-images-idx3-ubyte",
        angle=fixed_angle
    )

    # === Multi-range processing ===
    input_files = ["dataset/dataset_mnist_non_rotated/t10k-images-idx3-ubyte", "dataset/dataset_mnist_non_rotated/train-images-idx3-ubyte"]
    for input_file in input_files:
        rotate_and_save_ranges(input_file, angle_ranges)
