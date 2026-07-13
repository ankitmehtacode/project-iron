import numpy as np
import openvino as ov
import nncf
from nncf import Dataset
import os

# ----------------------------
# Paths
# ----------------------------

# Input: Depth Anything V2 OpenVINO IR model
MODEL_XML = "../models/ir/depth_anything_v2_small.xml"

# Output: INT8 quantized model
OUTPUT_XML = "../models/int8/depth_anything_v2_small_int8.xml"
OUTPUT_BIN = "../models/int8/depth_anything_v2_small_int8.bin"

os.makedirs("../models/int8", exist_ok=True)

# ----------------------------
# Load Model
# ----------------------------

core = ov.Core()
model = core.read_model(MODEL_XML)

# Retrieve model input name dynamically
input_name = model.inputs[0].get_any_name()
print(f"Model input name  : {input_name}")
print(f"Model input shape : {model.inputs[0].partial_shape}")

# ----------------------------
# Calibration Dataset
# ----------------------------


def transform_fn(data_item):
    """
    Wrap raw input into a dictionary matching the model input format.

    Args:
        data_item (np.ndarray): Input tensor of shape [1, 3, 224, 224].

    Returns:
        dict: Mapping of input name to tensor.
    """
    return {input_name: data_item}


# Generate synthetic calibration data
# Shape: [1, 3, 224, 224] — (batch, channels, height, width)
# NOTE: Using random data for calibration. For better quantization
# accuracy, replace with real representative input samples.
N_CALIB = 10
calibration_data = [
    np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(N_CALIB)
]

dataset = Dataset(calibration_data, transform_fn)

print("Starting INT8 post-training quantization (PTQ)...")

# ----------------------------
# Quantization
# ----------------------------

# Apply full post-training quantization using calibration dataset.
# This estimates activation ranges and quantizes both weights and activations.
quantized_model = nncf.quantize(model, dataset)

# ----------------------------
# Save Model
# ----------------------------

ov.save_model(quantized_model, OUTPUT_XML)

assert os.path.exists(OUTPUT_XML), "ERROR: .xml not written!"
assert os.path.exists(OUTPUT_BIN), "ERROR: .bin not written!"

orig_mb = os.path.getsize(MODEL_XML.replace(".xml", ".bin")) / 1e6
bin_mb = os.path.getsize(OUTPUT_BIN) / 1e6

print("Quantization complete!")
print(f"  Original : {orig_mb:.1f} MB")
print(f"  INT8     : {bin_mb:.1f} MB")
print(f"  Reduction: {100*(1 - bin_mb/orig_mb):.1f}%")
