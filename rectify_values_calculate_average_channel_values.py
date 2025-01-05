import numpy as np
from read_resize_normalize_image import read_resize_normalize_image
from filter_image_with_filters import filter_image_with_filters
import os
import random

def rectify_values_calculate_average_channel_values(filtered_image):
    
    # Rectify the filtered image using ReLU
    rectified_image = np.maximum(0, filtered_image)
    
    # Calculate the average values for each channel
    average_channel_values = np.mean(rectified_image, axis=(1, 2))
    
    return average_channel_values
