import struct
import numpy as np
import matplotlib.pyplot as plt

def load_mnist_images(filename):
    """Loads MNIST images from an IDX3-UBYTE file."""
    with open(filename, 'rb') as f:
        # Read the header information
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    
    return images

# Path to the file
filename = "dataset/t10k-images-idx3-ubyte"

# Load the images
images = load_mnist_images(filename)

print(len(images))
# Display the first image
plt.imshow(images[0], cmap='gray')
plt.title("First MNIST Test Image")
plt.axis("off")
plt.show()
