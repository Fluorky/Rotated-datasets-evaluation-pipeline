import os
import struct
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
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

    return images, num_images, rows, cols


def rotate_images(images: np.ndarray, angle: float) -> np.ndarray:
    """Rotates each image in the dataset by a given angle."""
    rotated_images = []
    for img in images:
        pil_img = Image.fromarray(img)  # Convert NumPy array to PIL Image
        rotated_img = pil_img.rotate(angle)  # Rotate the image
        rotated_images.append(np.array(rotated_img, dtype=np.uint8))  # Convert back to NumPy array

    return np.array(rotated_images)


def save_mnist_images(filename: str, images: np.ndarray, num_images: int, rows: int, cols: int) -> None:
    """Saves images into an IDX3-UBYTE file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        # Write header (magic number, number of images, rows, columns)
        f.write(struct.pack(">IIII", 2051, num_images, rows, cols))

        # Write image data
        f.write(images.tobytes())


# File paths
input_filename = "dataset/t10k-images-idx3-ubyte"  # Original MNIST file
output_filename = "dataset/rotated-45/t10k-images-idx3-ubyte"  # Transformed file

# Load images
images, num_images, rows, cols = load_mnist_images(input_filename)

# Rotate images by 45 degrees
rotated_images = rotate_images(images, 45)

# Save the transformed images back into a new IDX3-UBYTE file
save_mnist_images(output_filename, rotated_images, num_images, rows, cols)

# Display the first rotated image
plt.imshow(rotated_images[0], cmap='gray')
plt.title("First Rotated MNIST Test Image (45°)")
plt.axis("off")
plt.show()

print(f"Rotated MNIST  {num_images} images saved to {output_filename}")


# === Configuration ===
input_file = "dataset/train-images-idx3-ubyte"
output_file = "dataset/rotated-45/train-images-idx3-ubyte"
rotation_angle = 45  # degrees  #TODO: Change it to  [(20, 50), (50, 90), (90, 120)]

# === Processing ===
images, num_images, rows, cols = load_mnist_images(input_file)
rotated_images = rotate_images(images, rotation_angle)
save_mnist_images(output_file, rotated_images, num_images, rows, cols)

# === Preview ===
plt.imshow(rotated_images[0], cmap='gray')
plt.title(f"Rotated Training Image (angle={rotation_angle}°)")
plt.axis("off")
plt.show()

print(f"Saved {num_images} rotated images to '{output_file}'")