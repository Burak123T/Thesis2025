import tensorflow as tf
from tensorflow import keras
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python test_load.py <model_filename>")
    sys.exit(1)

model_filename = sys.argv[1]
current_dir = os.getcwd()
model_path = os.path.join(current_dir, model_filename)

try:
    print(f"Attempting to load model from path (absolute path, minimal): {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully in minimal script (absolute path)!")
except Exception as e:
    print(f"Error in minimal script (absolute path): {e}")
    print(f"Error details (absolute path): {e}") # Print full error
