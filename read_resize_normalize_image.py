import numpy as np
from PIL import Image

def read_resize_normalize_image(image_path, target_size):
    """
    Pre: image_path is the path to the image file
    Post: returns the image as a numpy array
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32) / 255.0
    # Normalize
    image = (image - 0.5) / 0.25
    # Scale back to range [0, 1]
    image = 0.25 * image + 0.5
    return image
