import os
import matplotlib.pyplot as plt
from rotate_dateset import load_mnist_images, rotate_images, rotate_images_by_angle, save_mnist_images

# Usage
if __name__ == '__main__':

    # File paths
    input_filename = "dataset/t10k-images-idx3-ubyte"  # Original MNIST file
    output_filename = "dataset/rotated-45/t10k-images-idx3-ubyte"  # Transformed file

    rotation_angle_ranges = [
        (20, 50),
        (50, 90),
        (90, 120)
    ]

    # Load images
    images, num_images, rows, cols = load_mnist_images(input_filename)

    # Rotate images by 45 degrees
    rotated_images = rotate_images_by_angle(images, 45)

    # Save the transformed images back into a new IDX3-UBYTE file
    save_mnist_images(output_filename, rotated_images, num_images, rows, cols)

    # Display the first rotated image
    # plt.imshow(rotated_images[0], cmap='gray')
    plt.title("First Rotated MNIST Test Image (45°)")
    plt.axis("off")
    # plt.show()

    print(f"Rotated MNIST  {num_images} images saved to {output_filename}")

    # === Configuration ===
    input_file = "dataset/train-images-idx3-ubyte"
    output_file = "dataset/rotated-45/train-images-idx3-ubyte"
    rotation_angle = 45  # degrees  #TODO: Change it to  [(20, 50), (50, 90), (90, 120)]

    # === Processing ===
    images, num_images, rows, cols = load_mnist_images(input_file)
    rotated_images = rotate_images_by_angle(images, rotation_angle)
    save_mnist_images(output_file, rotated_images, num_images, rows, cols)

    # === Preview ===
    # plt.imshow(rotated_images[0], cmap='gray')
    plt.title(f"Rotated Training Image (angle={rotation_angle}°)")
    plt.axis("off")
    # plt.show()

    print(f"Saved {num_images} rotated images to '{output_file}'")

    # === Processing Multiple Ranges===
    input_files = ["dataset/t10k-images-idx3-ubyte", "dataset/train-images-idx3-ubyte"]
    for input_file in input_files:
        images, num_images, rows, cols = load_mnist_images(input_file)

        for angle_range in rotation_angle_ranges:
            range_str = f"{angle_range[0]}-{angle_range[1]}"
            output_file = f"dataset/rotated-{range_str}/{os.path.basename(input_file)}"

            rotated_images = rotate_images(images, angle_range)
            save_mnist_images(output_file, rotated_images, num_images, rows, cols)

            # Preview
            plt.imshow(rotated_images[0], cmap='gray')
            plt.title(f"Rotated (angle ∈ {angle_range}°)")
            plt.axis("off")
            plt.show()

            print(f"Saved {num_images} images rotated in range {angle_range}° to '{output_file}'")
