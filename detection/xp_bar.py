from PIL import Image
import numpy as np

def calculate_purple_percentage(image_path):
    """
    Calculate the percentage of purple pixels in an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        float: Percentage of purple pixels in the image.
    """
    image = Image.open(image_path)
    image = image.convert("RGB")

    image_array = np.array(image)

    # Define the range of purple color in RGB
    purple_min = np.array([128, 0, 128])  
    purple_max = np.array([255, 128, 255])  

    purple_mask = np.all((image_array >= purple_min) & (image_array <= purple_max), axis=-1)

    purple_pixels = np.sum(purple_mask)
    total_pixels = image_array.shape[0] * image_array.shape[1]  
    purple_percentage = (purple_pixels / total_pixels) * 100

    return purple_percentage
