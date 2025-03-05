import tensorflow as tf
import tf2onnx
import onnx
from tensorflow import keras
from keras.models import load_model

import os
import sys

if len(sys.argv) != 2:
    print("Usage: python h5_to_onnx.py <model_filename>")
    sys.exit(1)

model_filename = sys.argv[1]
current_dir = os.getcwd()
model_path = os.path.join(current_dir, model_filename)

model = tf.keras.models.load_model(model_path)
model.output_names=['output']

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'new_model_eursat.onnx') 
