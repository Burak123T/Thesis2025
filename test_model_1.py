import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import sys

# ANSI escape codes for colors and bold text
class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# --- 1. Define paths and categories ---

if len(sys.argv) != 3:
    print("Usage: python test_load.py <model_filename> <image_filename>")
    sys.exit(1)

model_filename = sys.argv[1]
image_filename = sys.argv[2]
current_dir = os.getcwd()
model_path = os.path.join(current_dir, model_filename)
image_path = os.path.join(current_dir, image_filename)

CATEGORIES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]  # ADJUST THIS LIST IF CATEGORIES ARE DIFFERENT


# --- 2. Load the Keras Model ---
try:
    print(f"Attempting to load model from path: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"{bcolors.OKGREEN}Model successfully loaded from: {model_path}{bcolors.ENDC}")
except Exception as e:
    print(f"{bcolors.FAIL}Error loading model from {model_path}: {e}{bcolors.ENDC}")
    exit()  # Exit if model loading fails

# --- 3. Load and Preprocess the Test Image ---
try:
    img_array = cv2.imread(image_path)

    if img_array is None:
        raise FileNotFoundError(f"Error: Could not read image at path: {image_path}")

    resized_array = cv2.resize(img_array, (64, 64))

    normalized_array = resized_array / 255.0

    input_image = np.expand_dims(normalized_array, axis=0)

except FileNotFoundError as e:
    print(f"{bcolors.FAIL}File not found error: {e}{bcolors.ENDC}")
    exit()
except Exception as e:
    print(f"{bcolors.FAIL}Error loading and preprocessing image: {e}{bcolors.ENDC}")
    exit()


# --- 4. Make Prediction ---
try:
    predictions = model.predict(input_image)

    predicted_class_index = np.argmax(predictions[0])

    predicted_category_name = CATEGORIES[predicted_class_index]

    predicted_probability = predictions[0][predicted_class_index]

except Exception as e:
    print(f"{bcolors.FAIL}Error during prediction: {e}{bcolors.ENDC}")
    exit()


# --- 5. Print the Prediction Results ---
print("\n--- Prediction Results ---")
print(f"Test Image: {image_path}")
print(f"{bcolors.BOLD}Predicted Category: {predicted_category_name}{bcolors.ENDC}")
print(f"{bcolors.BOLD}Probability: {(predicted_probability * 100):.2f}%{bcolors.ENDC}")

# Optional: Print probabilities for all categories (for more detailed output)
print("\nProbabilities for all categories:")
for i, category_name in enumerate(CATEGORIES):
    probability = predictions[0][i]
    print(f"  {category_name}: {(probability * 100):.2f}%")

print("--- Prediction complete ---")
