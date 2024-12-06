from PIL import Image


def rotate_image(angle, image_path):
    """
    Rotates an image by a given angle.

    :param angle: The angle (in degrees) to rotate the image.
    :param image_path: Path to the image file to be rotated.
    :return: Rotated image object.
    """
    try:
        # Open the original image
        img = Image.open(image_path)
        # Rotate the image by the given angle
        image_rotated = img.rotate(angle, expand=True)
        # Return the rotated image
        return image_rotated
    except Exception as e:
        print(f"Error while rotating the image: {e}")
        return None


# Example usage
if __name__ == '__main__':
    rot_angle = -10

    img_path = 'pillow-rotate-earth.png'
    rotated_image = rotate_image(rot_angle, img_path)
    if rotated_image:
        rotated_image.show()
