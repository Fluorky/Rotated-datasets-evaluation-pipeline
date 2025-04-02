import os
import struct
import random
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_mnist_images(filename: str) -> Tuple[np.ndarray, int, int, int]:
    """Loads MNIST images from an IDX3-UBYTE file."""
    with open(filename, 'rb') as f:
        # Read the header information
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    return images, num_images, rows, cols

def rotate_images_by_angle(images: np.ndarray, angle: float) -> np.ndarray:
    """Rotates each image in the dataset by a given angle."""
    rotated_images = []
    for img in images:
        pil_img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        rotated_img = pil_img.rotate(angle)  # Rotate the image
        rotated_images.append(np.array(rotated_img, dtype=np.uint8))  # Convert back to NumPy array

    return np.array(rotated_images)

def rotate_images(images: np.ndarray, angle_range: Tuple[float, float]) -> np.ndarray:
    rotated_images = []
    for img in images:
        angle = random.uniform(*angle_range)
        pil_img = Image.fromarray(img)
        rotated_img = pil_img.rotate(angle)
        rotated_images.append(np.array(rotated_img, dtype=np.uint8))
    return np.array(rotated_images)

def save_mnist_images(filename: str, images: np.ndarray, num_images: int, rows: int, cols: int) -> None:
    """Saves images into an IDX3-UBYTE file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        # Write header (magic nugitmber, number of images, rows, columns)
        f.write(struct.pack(">IIII", 2051, num_images, rows, cols))

        # Write image data
        f.write(images.tobytes())


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