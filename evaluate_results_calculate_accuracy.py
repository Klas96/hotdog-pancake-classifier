from read_resize_normalize_image import read_resize_normalize_image
from filter_image_with_filters import filter_image_with_filters
from rectify_values_calculate_average_channel_values import rectify_values_calculate_average_channel_values
from apply_linear_clasifier import apply_linear_classifier
import os
import random

def evaluate_results_calculate_accuracy():
    threshold = 0.5
    correct_hotdogs = 0
    correct_pancakes = 0

    # Evaluate hotdogs
    hotdog_dir = './data/hotdog'
    if not os.path.exists(hotdog_dir):
        raise FileNotFoundError(f"No such file or directory: '{hotdog_dir}'")
    hotdog_images = random.sample(os.listdir(hotdog_dir), 50)
    for image_name in hotdog_images:
        image_path = os.path.join(hotdog_dir, image_name)
        image = read_resize_normalize_image(image_path, (64, 64))
        filtered_image = filter_image_with_filters(image)
        rectified_image = rectify_values_calculate_average_channel_values(filtered_image)
        probability = apply_linear_classifier(rectified_image)
        if probability < threshold:
            correct_hotdogs += 1

    # Evaluate pancakes
    pancake_dir = 'data/pancake'
    pancake_images = random.sample(os.listdir(pancake_dir), 50)
    for image_name in pancake_images:
        image_path = os.path.join(pancake_dir, image_name)
        image = read_resize_normalize_image(image_path, (64, 64))
        filtered_image = filter_image_with_filters(image)
        rectified_image = rectify_values_calculate_average_channel_values(filtered_image)
        probability = apply_linear_classifier(rectified_image)
        if probability >= threshold:
            correct_pancakes += 1

    # Calculate accuracy
    accuracy = (correct_hotdogs + correct_pancakes) / 100
    print(f'Accuracy: {accuracy}')
    print(f'# Correct hotdogs: {correct_hotdogs}')
    print(f'# Correct pancakes: {correct_pancakes}')

# Call the function to evaluate and calculate accuracy
evaluate_results_calculate_accuracy()
