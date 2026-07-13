import torch
import os
from transformers import AutoModel

# ----------------------------
# Configuration
# ----------------------------

# Directory containing pretrained V-JEPA2 weights
MODEL_DIR = "../models/weights/vjepa2_vitl"

# Output ONNX file path
ONNX_PATH = "../models/onnx/vjepa2_vitl.onnx"

# Ensure ONNX output directory exists
os.makedirs("../models/onnx", exist_ok=True)

# ----------------------------
# Load Model
# ----------------------------

print("Loading V-JEPA ViT-L model...")

# Load model with trust_remote_code=True since V-JEPA uses custom architecture
model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)

model.eval()

# ----------------------------
# Dummy Input
# ----------------------------

# Input tensor shape: (B, T, C, H, W)
# B = batch size
# T = number of frames
# C = channels (RGB → 3)
# H, W = spatial resolution (224x224)
dummy_video = torch.randn(1, 8, 3, 224, 224)

# ----------------------------
# Wrapper (required for ONNX export)
# ----------------------------


class Wrapper(torch.nn.Module):
    """
    Wrapper module to standardize V-JEPA model outputs for ONNX export.

    The Hugging Face AutoModel output can vary:
        - tuple
        - dict
        - tensor

    ONNX export requires a single tensor output. This wrapper ensures
    that regardless of the original output format, a consistent tensor
    is returned.
    """

    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): Loaded V-JEPA model.
        """
        super().__init__()
        self.model = model

    def forward(self, video):
        """
        Forward pass with output normalization.

        Args:
            video (torch.Tensor): Input tensor of shape (B, T, C, H, W)

        Returns:
            torch.Tensor: Primary feature tensor extracted from model output
        """
        output = self.model(video)

        # Normalize output to a single tensor
        if isinstance(output, tuple):
            return output[0]
        elif isinstance(output, dict):
            # Return first value (typically the main feature representation)
            return list(output.values())[0]
        else:
            return output


wrapped_model = Wrapper(model)
wrapped_model.eval()

# ----------------------------
# ONNX Export
# ----------------------------

print("Exporting ONNX...")

with torch.no_grad():
    torch.onnx.export(
        wrapped_model,
        dummy_video,
        ONNX_PATH,
        input_names=["video"],  # Input video tensor
        output_names=["features"],  # Output feature embeddings
        opset_version=18,  # Required for newer transformer ops
        dynamic_axes={
            "video": {0: "batch", 1: "time"},  # Variable batch size and frame count
            "features": {0: "batch"},
        },
        dynamo=False,  # Use legacy exporter for better stability with complex models
    )

print(f"ONNX export complete! Saved at: {ONNX_PATH}")
