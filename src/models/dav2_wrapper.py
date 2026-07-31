from typing import Dict
import numpy as np
import torch
from src.contracts import DepthField, FrameGeometry
from src.models.model_wrapper import ModelWrapper


def as_depth_field(depth: np.ndarray) -> DepthField:
    """Wrap a raw DA-V2 output as a typed, honestly-labelled DepthField.

    Args:
        depth: ``[H, W]`` relative inverse depth as the model emits it.

    Returns:
        A ``DepthField`` with ``units="disparity_rel"``, geometry taken from
        the array's own shape, and a validity mask marking non-finite entries
        unusable.

    The values are passed through untouched. The mask is the only judgement:
    a non-finite depth is not a far one, and letting it through as a number
    would put a NaN into whatever 3D point is computed from it.
    """
    array = np.asarray(depth, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected an [H, W] depth map, got shape {array.shape}")

    height, width = array.shape
    return DepthField(
        data=array,
        # NOT "meters". See DAv2Wrapper.predict for why this single word is
        # the whole point of the type.
        units="disparity_rel",
        geometry=FrameGeometry(width=width, height=height),
        valid_mask=np.isfinite(array),
    )


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

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, DepthField]:
        """
        Args:
            inputs: {"image": cv2_image (HxWx3 BGR numpy array)}

        Returns:
            ``{"depth": DepthField}`` with ``units="disparity_rel"``.

        The units are the point. Depth-Anything-V2 emits *relative inverse
        depth* on an arbitrary per-frame scale — unknown scale AND unknown
        shift, so no constant converts it to metres. This wrapper used to
        return a bare ndarray, and the value flowed through the projector into
        a Parquet column documented as "Depth (meters)". Nothing raised;
        every 3D coordinate downstream had arbitrary scale while claiming to be
        metric.

        Returning a :class:`~src.contracts.DepthField` makes that
        unrepresentable: the units travel with the array, and
        :func:`~src.contracts.geometry.unproject` refuses anything non-metric.
        Callers wanting the raw values ask for ``.data`` explicitly, which is a
        visible act rather than an assumption.

        The numerical values are unchanged — only the envelope around them.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        raw_img = inputs["image"]

        # Model returns HxW depth map directly
        with torch.no_grad():
            depth = self.model.infer_image(raw_img)

        return {"depth": as_depth_field(depth)}
