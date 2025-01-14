from PIL import Image
import random
import os
import numpy as np
from tensorflow.keras.datasets import mnist
from typing import List, Tuple


def rotate_image_object(angle: float, image: Image.Image) -> Image.Image:
    """
    Rotates a given Image object by a specified angle.

    :param angle: The angle (in degrees) to rotate the image.
    :param image: The Image object to be rotated.
    :return: Rotated Image object.
    """
    return image.rotate(angle, expand=True)


def convert_mnist_to_images(train_images: np.ndarray) -> List[Image.Image]:
    """
    Converts MNIST dataset images to PIL.Image objects.

    :param train_images: NumPy array of MNIST images.
    :return: List of PIL.Image objects.
    """
    return [Image.fromarray(image) for image in train_images]


def rotate_batch_images_from_train(train_images: List[Image.Image], angle_ranges: List[Tuple[int, int]]) -> List[Image.Image]:
    """
    Rotates a batch of train images by random angles within specified ranges.

    :param train_images: List of Image objects representing the training images.
    :param angle_ranges: List of tuples specifying angle ranges (min_angle, max_angle).
    :return: List of rotated Image objects.
    """
    rotated_images = []

    for image in train_images:
        # Randomly choose an angle range and calculate a random angle within it
        angle_range = random.choice(angle_ranges)
        random_angle = random.uniform(*angle_range)

        # Rotate the image by the random angle
        rotated_image = rotate_image_object(random_angle, image)
        rotated_images.append(rotated_image)

    return rotated_images


def save_batch_images(images: List[Image.Image], output_dir: str):
    """
    Saves a batch of images to a specified directory.

    :param images: List of Image objects to be saved.
    :param output_dir: Directory where the images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, image in enumerate(images):
        save_path = os.path.join(output_dir, f"rotated_image_{i + 1}.png")
        image.save(save_path)
        print(f"Saved rotated image to {save_path}")


# Example usage
if __name__ == '__main__':
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Convert train_images to PIL.Image objects
    train_images_pil = convert_mnist_to_images(train_images)

    # Define angle ranges as specified in the instructions
    angle_ranges = [(20, 50), (50, 90), (90, 120)]

    # Rotate images using specified angle ranges
    rotated_images = rotate_batch_images_from_train(train_images_pil[:10], angle_ranges)  # Rotate first 10 images as a sample
    # Print labels for the rotated images
    print("Labels for the rotated images:")
    for i, label in enumerate(train_labels[:10]):
        print(f"Image {i + 1}: Label {label}")

    # Save rotated images to the specified output directory
    output_directory = 'rotated_images_output'
    save_batch_images(rotated_images, output_directory)
