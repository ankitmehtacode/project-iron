import os
import numpy as np  # noqa: F401
import openvino as ov
import nncf

# ----------------------------
# Paths
# ----------------------------

# Directory to store INT8 compressed model
os.makedirs("../models/int8", exist_ok=True)

# Load CoTracker3 IR model
core = ov.Core()
model = core.read_model("../models/ir/cotracker3.xml")

# ----------------------------
# Inspect Model Inputs
# ----------------------------

print("Model Inputs:")
for inp in model.inputs:
    print(f"  {inp.get_any_name()} : {inp.partial_shape}")

# ----------------------------
# Weight Compression
# ----------------------------

"""
CoTracker3 model is relatively small (~5 MB), so full quantization
is not necessary. Instead, we apply weight compression using
INT8 asymmetric quantization.

INT8_ASYM:
    - Uses asymmetric quantization (separate scale and zero-point)
    - Better suited for weights with non-zero-centered distributions
    - Provides improved accuracy compared to symmetric INT8 in many cases
"""

compressed = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)

# ----------------------------
# Save Model
# ----------------------------

ov.save_model(compressed, "../models/int8/cotracker3_int8.xml")

print("Compression complete!")
