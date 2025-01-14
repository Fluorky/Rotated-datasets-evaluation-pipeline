from PIL import Image
from typing import Optional, Union
import os


def rotate_image(angle: Union[int, float], image_path: str, save_path: Optional[str] = None,
                 save_format: Optional[str] = None) -> Optional[Image.Image]:
    """
    Rotates an image by a given angle.

    :param angle: The angle (in degrees) to rotate the image. Can be positive or negative.
    :param image_path: Path to the image file to be rotated.
    :param save_path: (Optional) Path to save the rotated image. If not provided, the image will not be saved.
    :param save_format: (Optional) Format to save the image (e.g., 'JPEG', 'PNG'). If not provided, the original format will be used.
    :return: Rotated image object, or None if an error occurred.
    """
    try:
        if not os.path.isfile(image_path):
            print(f"Error: The file '{image_path}' does not exist.")
            return None

        # Open the original image
        img = Image.open(image_path)
        # Rotate the image by the given angle
        image_rotated = img.rotate(angle, expand=True)

        if save_path:
            # Determine save format
            format_to_save = save_format or img.format
            image_rotated.save(save_path, format=format_to_save)
            print(f"Rotated image saved at: {save_path} ({format_to_save})")

        # Return the rotated image object
        return image_rotated

    except Exception as e:
        print(f"Error while rotating the image: {e}")
        return None


# Example usage
if __name__ == '__main__':
    rot_angle: float = -10  # Angle in degrees
    img_path: str = 'pillow-rotate-earth.png'  # Path to the input image
    save_path: str = 'rotated-image.png'  # Path to save the rotated image

    rotated_image = rotate_image(rot_angle, img_path, save_path, save_format='PNG')
    if rotated_image:
        rotated_image.show()
