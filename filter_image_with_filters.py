import numpy as np
import os

def read_filter_and_bias(filter_num):
    """
    Reads the filter weights and bias from CSV files for a given filter number.
    
    How:
        - Read the weights and bias for each filter.

    Pre: 
        - filter_num is an integer between 0 and 15 inclusive.
        - Corresponding CSV files 'weight<filter_num>.csv' and 'bias<filter_num>.csv' exist.
    
    Post: 
        - Returns a tuple (weights, bias) where:
            - weights is a numpy array of shape (3, 9, 9).
            - bias is a single float value.
    """
    # Read weights and reshape to 3x9x9
    weights = np.loadtxt(f'weights/weight{filter_num}.csv').reshape(3, 9, 9)
    
    # Read bias
    bias = np.loadtxt(f'biases/bias{filter_num}.csv')
    
    return weights, bias

def apply_filter(image, weights, bias):
    """
    Applies a single filter to the image with zero padding.
    
    Pre: 
        - image is a numpy array of shape (3, 64, 64).
        - weights is a numpy array of shape (3, 9, 9).
        - bias is a float.
    
    Post: 
        - Returns a numpy array of shape (64, 64) representing the filtered image.
    """
    # Initialize output
    output = np.zeros((64, 64))
    
    # Pad the image with zeros on each side
    padded_image = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode='constant', constant_values=0)
    
    # Perform the convolution operation
    for i in range(64):
        for j in range(64):
            for c in range(3):  # Iterate over channels
                output[i, j] += np.sum(padded_image[c, i:i+9, j:j+9] * weights[c])
            output[i, j] += bias
    
    return output

def filter_image_with_filters(image):
    """
    Applies 16 filters to the input image and returns the filtered images.
    
    Pre: 
        - image is a numpy array of shape (64, 64, 3).
    
    Post: 
        - Returns a numpy array of shape (16, 64, 64) where each slice along the first axis
          corresponds to the output of one filter.
    """
    # Transpose the image to shape (3, 64, 64) if necessary
    if image.shape == (64, 64, 3):
        image = image.transpose(2, 0, 1)
    
    # Initialize the filtered image array
    filtered_image = np.zeros((16, 64, 64))
    
    # Apply each filter
    for k in range(16):
        weights, bias = read_filter_and_bias(k)
        
        # Ensure the image is correctly padded to avoid broadcasting issues
        padded_image = np.pad(image, ((0, 0), (4, 4), (4, 4)), mode='constant', constant_values=0)
        
        # Apply the filter
        filtered_image[k] = apply_filter(padded_image, weights, bias)
    
    return filtered_image