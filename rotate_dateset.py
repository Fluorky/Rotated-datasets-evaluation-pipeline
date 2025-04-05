import os
import struct
import random
from typing import Tuple
import numpy as np
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
