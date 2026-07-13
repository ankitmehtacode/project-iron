"""
quantize_vjepa.py
=================

INT8 weight compression for V-JEPA2 using OpenVINO + NNCF.

------------------------------------------------------------------
OVERVIEW
------------------------------------------------------------------
This script applies weight-only INT8 compression using
nncf.compress_weights(), which reduces model size while maintaining
FP32 computation for activations.

------------------------------------------------------------------
WHY NOT nncf.quantize()?
------------------------------------------------------------------
Full post-training quantization (PTQ) requires collecting activation
statistics, which significantly increases memory usage.

For large models like V-JEPA2 (~580MB), PTQ can exceed available RAM
on constrained systems.

------------------------------------------------------------------
SOLUTION
------------------------------------------------------------------
Use nncf.compress_weights():
    • Compresses weights to INT8
    • Processes one layer at a time (low memory footprint)
    • Avoids calibration dataset requirement
    • Preserves FP32 computation for activations

This provides a good trade-off between memory efficiency and model size.

------------------------------------------------------------------
RESULT
------------------------------------------------------------------
Typical size reduction:
    ~580 MB → ~290 MB

Note:
    This is not full INT8 inference. Only weights are compressed.
    Computation remains in FP32.

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
Run in appropriate OpenVINO environment:

    conda activate openvino310
    cd ~/Internship/openvino_project/scripts
    python quantize_vjepa.py
"""

import os
import openvino as ov
import nncf

# ----------------------------
# Paths
# ----------------------------

MODEL_XML = "../models/ir/vjepa2_vitl.xml"
OUTPUT_XML = "../models/int8/vjepa2_vitl_int8.xml"
OUTPUT_BIN = "../models/int8/vjepa2_vitl_int8.bin"

os.makedirs("../models/int8", exist_ok=True)

# ----------------------------
# Load Model
# ----------------------------

print("Loading V-JEPA2 IR model...")
core = ov.Core()
model = core.read_model(MODEL_XML)

# Display input information for verification
print(f"Input : {model.inputs[0].get_any_name()}  " f"{model.inputs[0].partial_shape}")

# ----------------------------
# Weight Compression
# ----------------------------

print("\nCompressing weights to INT8...")

# Apply layer-wise INT8 weight compression
compressed = nncf.compress_weights(
    model,
    mode=nncf.CompressWeightsMode.INT8,
)

# ----------------------------
# Save Model
# ----------------------------

print("\nSaving compressed model...")
ov.save_model(compressed, OUTPUT_XML)

# Ensure files were written successfully
assert os.path.exists(OUTPUT_XML), "ERROR: .xml not written!"
assert os.path.exists(OUTPUT_BIN), "ERROR: .bin not written!"

# ----------------------------
# Report Size Reduction
# ----------------------------

orig_mb = os.path.getsize(MODEL_XML.replace(".xml", ".bin")) / 1e6
xml_mb = os.path.getsize(OUTPUT_XML) / 1e6
bin_mb = os.path.getsize(OUTPUT_BIN) / 1e6

print("\nCompression complete!")
print(f"Original  : {orig_mb:.1f} MB")
print(f"Compressed: {bin_mb:.1f} MB  (.bin)")
print(f"Reduction : {100*(1 - bin_mb/orig_mb):.1f}%")
