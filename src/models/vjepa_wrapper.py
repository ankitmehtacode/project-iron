from typing import Dict
import numpy as np
import torch
from model_wrapper import ModelWrapper


class VJEPAWrapper(ModelWrapper):
    """Wrapper for V-JEPA 2 model (video embeddings)."""

    def __init__(self, model_size: str = "vitg", device: str = "CPU"):
        """
        Args:
            model_size: 'vitl', 'vith', 'vitg', or 'vitg_384'
            device: 'CPU' or 'GPU'
        """
        # V-JEPA uses torch hub, not file paths
        super().__init__(model_path=model_size, device=device)
        self.model_size = model_size
        self.processor = None

    def _validate_path(self) -> None:
        # Override - no file validation needed for torch hub
        pass

    def load(self) -> None:
        """Load V-JEPA 2 model from PyTorch Hub."""

        # Load preprocessor
        self.processor = torch.hub.load(
            "facebookresearch/vjepa2", "vjepa2_preprocessor"
        )

        # Load model based on size
        model_map = {
            "vitl": "vjepa2_vit_large",
            "vith": "vjepa2_vit_huge",
            "vitg": "vjepa2_vit_giant",
            "vitg_384": "vjepa2_vit_giant_384",
        }

        if self.model_size not in model_map:
            raise ValueError(
                f"Invalid model size. Choose from: {list(model_map.keys())}"
            )

        self.model = torch.hub.load(
            "facebookresearch/vjepa2", model_map[self.model_size]
        )

        if torch.cuda.is_available() and self.device == "GPU":
            self.model = self.model.cuda()

        self.model.eval()

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Args:
            inputs: {"video": video_tensor (T, H, W, C) or (B, T, H, W, C)}

        Returns:
            {"embeddings": video_embeddings}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        video = torch.from_numpy(inputs["video"])

        # Preprocess video
        video = self.processor(video)

        if torch.cuda.is_available() and self.device == "GPU":
            video = video.cuda()

        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(video)

        return {"embeddings": embeddings.cpu().numpy()}
