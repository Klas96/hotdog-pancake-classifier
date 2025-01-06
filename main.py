from read_resize_normalize_image import read_resize_normalize_image
from filter_image_with_filters import filter_image_with_filters
from rectify_values_calculate_average_channel_values import rectify_values_calculate_average_channel_values
from apply_linear_clasifier import apply_linear_classifier
import os

def classify_images(image_dir, label, threshold=0.5):
    correct_count = 0
    
    # Check if the directory exists
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist.")
        return correct_count
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = read_resize_normalize_image(image_path, (64, 64))
        
        # Filter image
        filtered_image = filter_image_with_filters(image)

        # Rectify values
        rectified_image = rectify_values_calculate_average_channel_values(filtered_image)

        # Apply linear classifier
        probability = apply_linear_classifier(rectified_image)
        print(f"Image: {image_name}, Probability: {probability}")

        # Classify based on threshold
        if (probability < threshold and label == 'hotdog') or (probability >= threshold and label == 'pancake'):
            correct_count += 1

    return correct_count

if __name__ == "__main__":
    # Classify hotdogs
    hotdog_dir = './data/hotdog'
    correct_hotdogs = classify_images(hotdog_dir, 'hotdog')

    # Classify pancakes
    pancake_dir = './data/pancakes'
    correct_pancakes = classify_images(pancake_dir, 'pancake')

    # Calculate accuracy
    accuracy = (correct_hotdogs + correct_pancakes) / 100
    print(f"Accuracy: {accuracy}")
    print(f"# Correct hotdogs: {correct_hotdogs}")
    print(f"# Correct pancakes: {correct_pancakes}")