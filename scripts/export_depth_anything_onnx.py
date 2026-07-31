import torch
import os
from transformers import DepthAnythingForDepthEstimation

# ----------------------------
# Configuration
# ----------------------------

# Directory containing pretrained Depth Anything V2 model weights
MODEL_DIR = "../models/weights/depth_anything_v2_small"

# Directory to save exported ONNX model
ONNX_DIR = "../models/onnx"
os.makedirs(ONNX_DIR, exist_ok=True)

# Output ONNX file path
ONNX_PATH = os.path.join(ONNX_DIR, "depth_anything_v2_small.onnx")

print("Loading Depth Anything V2 model...")

# Load pretrained model for monocular depth estimation
model = DepthAnythingForDepthEstimation.from_pretrained(MODEL_DIR)
model.eval()

# Export is performed on CPU for compatibility
device = torch.device("cpu")
model.to(device)

# ----------------------------
# Dummy Input
# ----------------------------

# Input tensor shape: (B, C, H, W)
# B = batch size
# C = channels (RGB → 3)
# H, W = spatial resolution (224x224 expected by model)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

print("Exporting ONNX...")

# ----------------------------
# ONNX Export
# ----------------------------

# Export PyTorch model to ONNX format for deployment (e.g., OpenVINO)
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],  # Name of input tensor in ONNX graph
    output_names=["depth"],  # Output represents predicted depth map
    opset_version=17,  # Stable opset for OpenVINO compatibility
    dynamic_axes={
        "input": {0: "batch"},  # Allow variable batch size
        "depth": {0: "batch"},
    },
)

print("ONNX export complete!")
print("Saved at:", ONNX_PATH)
