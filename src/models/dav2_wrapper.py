from typing import Dict
import numpy as np
import torch
from model_wrapper import ModelWrapper


class DAv2Wrapper(ModelWrapper):
    """Wrapper for Depth Anything V2 model."""

    def __init__(self, model_path: str, encoder: str = "vitl", device: str = "CPU"):
        """
        Args:
            model_path: Path to .pth checkpoint
            encoder: Model size - 'vits', 'vitb', 'vitl', or 'vitg'
            device: 'CPU' or 'GPU'
        """
        super().__init__(model_path, device)
        self.encoder = encoder

        self.model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

    def load(self) -> None:
        from depth_anything_v2.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(**self.model_configs[self.encoder])
        self.model.load_state_dict(torch.load(str(self.model_path), map_location="cpu"))

        if torch.cuda.is_available() and self.device == "GPU":
            self.model = self.model.cuda()

        self.model.eval()

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Args:
            inputs: {"image": cv2_image (HxWx3 BGR numpy array)}

        Returns:
            {"depth": depth_map (HxW numpy array)}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        raw_img = inputs["image"]

        # Model returns HxW depth map directly
        with torch.no_grad():
            depth = self.model.infer_image(raw_img)

        return {"depth": depth}
