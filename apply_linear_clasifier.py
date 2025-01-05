import numpy as np
import os

def apply_linear_classifier(rectified_image):
    """
    Applies a linear classifier to the rectified image and returns the probability score.

    Pre:
        - rectified_image is a numpy array of shape (16,).
    
    Post:
        - Returns a float value representing the probability score.
    """
    
    # Read the weights and bias for the linear classifier
    weights = np.loadtxt('fc_weight.csv')
    bias = np.loadtxt('fc_bias.csv')
    
    # Calculate the score
    score = bias + np.dot(rectified_image, weights)
    
    # Apply the sigmoid function to get the probability
    probability = 1 / (1 + np.exp(-score))
    
    return probability
