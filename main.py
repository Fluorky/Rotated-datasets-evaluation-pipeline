import os
import matplotlib.pyplot as plt
from rotate_dateset import load_mnist_images, rotate_images, rotate_images_by_angle, save_mnist_images


def rotate_and_save_fixed_angle(input_path: str, output_path: str, angle: float):
    """Rotate dataset by a fixed angle and save it."""
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
    """Rotate dataset by multiple angle ranges and save to separate folders."""
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
    angle_ranges = [(20, 50), (50, 90), (90, 120)]

    # === Fixed-angle processing ===
    rotate_and_save_fixed_angle(
        input_path="dataset/t10k-images-idx3-ubyte",
        output_path="dataset/rotated-45/t10k-images-idx3-ubyte",
        angle=fixed_angle
    )

    rotate_and_save_fixed_angle(
        input_path="dataset/train-images-idx3-ubyte",
        output_path="dataset/rotated-45/train-images-idx3-ubyte",
        angle=fixed_angle
    )

    # === Multi-range processing ===
    input_files = ["dataset/t10k-images-idx3-ubyte", "dataset/train-images-idx3-ubyte"]
    for input_file in input_files:
        rotate_and_save_ranges(input_file, angle_ranges)
