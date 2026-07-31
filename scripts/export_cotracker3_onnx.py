"""
export_cotracker3_onnx.py
=========================

Export CoTracker3 model to ONNX with OpenVINO compatibility fixes.

------------------------------------------------------------------
PROBLEM
------------------------------------------------------------------
PyTorch's F.grid_sample supports both 4D and 5D inputs:
    4D: [B, C, H, W]
    5D: [B, C, D, H, W]

However, OpenVINO only supports 4D grid_sample.

CoTracker3 internally uses 5D grid_sample for spatiotemporal
sampling, which breaks ONNX → OpenVINO conversion.

------------------------------------------------------------------
SOLUTION
------------------------------------------------------------------
Override F.grid_sample with a custom implementation that:

    • Passes 4D inputs unchanged
    • Converts 5D inputs into multiple 4D operations by:
        - Iterating over temporal dimension (T)
        - Applying 4D grid_sample per frame
        - Combining results using time-based interpolation weights

This ensures the exported ONNX graph only contains 4D grid_sample
operations, making it compatible with OpenVINO.

------------------------------------------------------------------
KEY IDEA
------------------------------------------------------------------
For 5D input [B, C, T, H, W]:
    - Process each frame independently using 4D grid_sample
    - Use the temporal coordinate (t) from the sampling grid
      to compute interpolation weights
    - Aggregate results across time

This avoids unsupported 5D operations while preserving behavior.

------------------------------------------------------------------
USAGE
------------------------------------------------------------------
Run inside the OpenVINO environment, from the repository root:

    python scripts/export_cotracker3_onnx.py

After export:

    ovc models/onnx/cotracker3_ovfix.onnx --output_model models/ir/cotracker3.xml
"""

import os
import torch
import torch.nn.functional as F
import torch.nn as nn

# ----------------------------
# Paths
# ----------------------------

WEIGHTS_PATH = "../models/weights/cotracker3/scaled_offline.pth"
ONNX_PATH = "../models/onnx/cotracker3_ovfix.onnx"

os.makedirs("../models/onnx", exist_ok=True)

# Store original PyTorch implementation
_orig_grid_sample = F.grid_sample


def _grid_sample_ov(
    input, grid, mode="bilinear", padding_mode="border", align_corners=True
):
    """
    OpenVINO-compatible replacement for torch.nn.functional.grid_sample.

    Behavior:
        • For 4D input: calls original implementation
        • For 5D input: decomposes operation into multiple 4D calls

    Args:
        input: Tensor of shape [B,C,H,W] or [B,C,T,H,W]
        grid : Sampling grid
        mode, padding_mode, align_corners: Same as PyTorch API

    Returns:
        Sampled tensor with shape consistent with PyTorch behavior
    """

    # Case 1: Standard 4D input (supported by OpenVINO)
    if input.dim() == 4:
        return _orig_grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    # Case 2: 5D input (requires decomposition)
    if input.dim() == 5:
        B, C, T, H, W = input.shape

        # Grid shape: [B, Ho, Wo, To, 3] → (x, y, t)
        Ho, Wo, To = grid.shape[1], grid.shape[2], grid.shape[3]

        results = []

        for t_idx in range(T):
            # Extract frame at time t_idx → [B, C, H, W]
            frame = input[:, :, t_idx, :, :]

            # Extract spatial coordinates (x, y)
            xy = grid[..., :2]  # [B, Ho, Wo, To, 2]
            xy_4d = xy.reshape(B, Ho, Wo * To, 2)  # [B, Ho, Wo*To, 2]

            # Temporal coordinate in range [-1, 1]
            t_coord = grid[..., 2]

            # Convert normalized t → index space [0, T-1]
            t_float = (t_coord + 1) / 2 * max(T - 1, 1)

            # Compute interpolation weight for current frame
            weight = (1.0 - (t_float - t_idx).abs()).clamp(min=0)
            weight_4d = weight.reshape(B, Ho, Wo * To)

            # Perform 4D grid_sample
            out_frame = _orig_grid_sample(
                frame,
                xy_4d,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

            # Apply temporal weight
            weight_exp = weight_4d.unsqueeze(1)
            results.append(out_frame * weight_exp)

        # Combine contributions from all frames
        out = torch.stack(results, dim=0).sum(dim=0)

        # Reshape to expected output format
        out = out.reshape(B, C, Ho, Wo, To)

        return out

    raise ValueError(f"_grid_sample_ov: unsupported input.dim={input.dim()}")


# Patch grid_sample globally
F.grid_sample = _grid_sample_ov

# Patch inside CoTracker utilities as well
import cotracker.models.core.model_utils as mu  # noqa: E402

mu.F.grid_sample = _grid_sample_ov


# ----------------------------
# Load Model
# ----------------------------

print("Loading CoTracker3 model...")

from cotracker.predictor import CoTrackerPredictor  # noqa: E402

predictor = CoTrackerPredictor(checkpoint=WEIGHTS_PATH, offline=True, window_len=60)

cotracker = predictor.model
cotracker.eval()


class CoTrackerExportWrapper(nn.Module):
    """
    Wrapper to normalize CoTracker model outputs for ONNX export.

    Ensures that the forward method returns a single tensor, which
    is required for ONNX export compatibility.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video, queries):
        outputs = self.model(video, queries)

        if isinstance(outputs, (list, tuple)):
            return outputs[0]

        return outputs


wrapped = CoTrackerExportWrapper(cotracker)
wrapped.eval()


# ----------------------------
# Dummy Inputs
# ----------------------------

# Video shape: [B, T, C, H, W]
# Queries shape: [B, N, 3]
B, T, C, H, W = 1, 4, 3, 224, 224

video = torch.randn(B, T, C, H, W)
queries = torch.randn(B, 100, 3)

print(f"video  : {list(video.shape)}")
print(f"queries: {list(queries.shape)}")

print("\nRunning sanity forward pass...")

with torch.no_grad():
    out = wrapped(video, queries)

print(f"Sanity pass output shape: {out.shape}")


# ----------------------------
# ONNX Export
# ----------------------------

print("\nExporting to ONNX...")

with torch.no_grad():
    torch.onnx.export(
        wrapped,
        (video, queries),
        ONNX_PATH,
        input_names=["video", "queries"],
        output_names=["tracks"],
        opset_version=17,
        dynamic_axes={
            "video": {0: "batch"},
            "queries": {0: "batch"},
            "tracks": {0: "batch"},
        },
        dynamo=False,
    )

# Restore original implementation
F.grid_sample = _orig_grid_sample

size_mb = os.path.getsize(ONNX_PATH) / 1e6

print("\nONNX export complete!")
print(f"Saved: {ONNX_PATH}  ({size_mb:.1f} MB)")

print("\nNext step:")
print("ovc models/onnx/cotracker3_ovfix.onnx --output_model models/ir/cotracker3.xml")
