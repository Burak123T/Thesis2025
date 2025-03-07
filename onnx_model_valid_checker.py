import onnx
import sys

if len(sys.argv) < 2:
    print ("Provide path for the .onnx-file as a cmd-argument.")
    sys.exit()

model_path = sys.argv[1]

try:
    model = onnx.load(model_path)  # Use the full path
    print(f"ONNX Opset Version: {model.opset_import[0].version}")
except onnx.checker.ValidationError as e:
    print(f"ONNX Validation Error: {e}")
    exit(1)
except Exception as e:
     print(f"Error loading ONNX model: {e}")
     exit(1)
