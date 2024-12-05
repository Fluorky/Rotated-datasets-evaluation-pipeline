from PIL import Image

# Opening the original image
img = Image.open('pillow-rotate-earth.png')

# Showcasing the original image
img.show()

# Rotating the image by 30 degrees
image_rotated = img.rotate(30)

# Showcasing the rotated image
image_rotated.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Showcasing the rotated image
    image_rotated.show()



