import torch
import cProfile
import pstats
from transformers import AutoModel


class VJEPAWrapper:
    """
    Lightweight wrapper for loading and running inference with the V-JEPA2 ViT-L model.

    This class:
        - Loads a pretrained V-JEPA2 model from a local directory
        - Moves the model to the appropriate device (GPU if available, else CPU)
        - Provides a simple forward() method for inference

    Expected input:
        video_tensor: torch.Tensor of shape (B, T, C, H, W)
            B = batch size
            T = number of frames
            C = number of channels (typically 3)
            H, W = spatial dimensions (e.g., 224x224)

    Output:
        Model-dependent output (typically embeddings or feature representations)
    """

    def __init__(self, model_path):
        """
        Initialize the V-JEPA model.

        Args:
            model_path (str): Path to the local directory containing model weights.
        """
        print("Loading V-JEPA2 ViT-L model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model from Hugging Face-compatible directory
        self.model = AutoModel.from_pretrained(model_path)

        # Move model to device and set evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def forward(self, video_tensor):
        """
        Run inference on a batch of video data.

        Args:
            video_tensor (torch.Tensor): Input tensor of shape (B, T, C, H, W).

        Returns:
            torch.Tensor or model-specific output:
                Output produced by the V-JEPA model.
        """
        with torch.no_grad():
            output = self.model(video_tensor)
        return output


def profile_inference(wrapper, video_tensor):
    """
    Profile the inference time of the V-JEPA model using Python's cProfile.

    This function runs a single forward pass and prints the top time-consuming
    function calls sorted by cumulative time.

    Args:
        wrapper (VJEPAWrapper): Initialized model wrapper.
        video_tensor (torch.Tensor): Input tensor for inference.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    wrapper.forward(video_tensor)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime").print_stats(20)


if __name__ == "__main__":
    # Path to locally downloaded V-JEPA2 weights
    model_dir = "../models/weights/vjepa2_vitl"

    # Initialize wrapper
    wrapper = VJEPAWrapper(model_dir)

    # Create dummy input tensor for testing
    # Shape: (batch, frames, channels, height, width)
    video_tensor = torch.randn(1, 8, 3, 224, 224).to(wrapper.device)

    print("Running profiling...")

    # Profile inference performance
    profile_inference(wrapper, video_tensor)
