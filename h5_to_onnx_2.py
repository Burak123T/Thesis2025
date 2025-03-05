import tensorflow as tf
from tensorflow import keras
import onnx
import tf2onnx.convert

import os
import sys

# ANSI escape codes for colors and bold text
class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

print("\nTensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__, "\n")

if len(sys.argv) != 2:
    print("Usage: python h5_to_onnx_2.py <model_filename>")
    sys.exit(1)

# Load the model
model_filename = sys.argv[1]
current_dir = os.getcwd()
model_path = os.path.join(current_dir, model_filename)

try:
    print(f"Attempting to load model from path: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"{bcolors.OKGREEN}Model successfully loaded from: {model_path}{bcolors.ENDC}")
except Exception as e:
    print(f"{bcolors.FAIL}Error loading model from {model_path}: {e}{bcolors.ENDC}")
    exit()  # Exit if model loading fails

# Check its architecture
model.summary()

# Save the model in SavedModel format
model.export(filepath='saved_model')
print(f"\n{bcolors.BOLD}Model saved in SavedModel format at: /saved_model{bcolors.ENDC}\n")

# Now time to convert to ONNX!
try:
    new_model, _ = tf2onnx.convert.from_saved_model('saved_model', opset=13)  # Use from_saved_model
    onnx.save(new_model, 'simple_eurosat_model.onnx')
    print(f"{bcolors.OKGREEN}Model successfully saved into ONNX{bcolors.ENDC}")
except Exception as e:
    print(f"{bcolors.FAIL}Error during ONNX conversion: {e}{bcolors.ENDC}")

