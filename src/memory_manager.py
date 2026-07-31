import gc
import torch
import numpy as np
from typing import Dict, Any

from utils.disk_cache import DiskCache


class MemoryManager:
    """
    Manages memory during sequential model execution.
    Ensures only one model is in RAM at a time.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/depth_maps",
        cache_size_gb: float = 10.0,
        device: str = "GPU",
    ):
        """
        Args:
            cache_dir: Where to store cached depth maps
            cache_size_gb: Max cache size
            device: 'CPU' or 'GPU'
        """
        self.cache = DiskCache(cache_dir=cache_dir, max_size_gb=cache_size_gb)
        self.device = device

        self.dav2_model = None
        self.cotracker_model = None

    def _free_gpu_memory(self) -> None:
        """Force GPU memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_dav2(self, model_path: str, encoder: str = "vitl") -> None:
        """Load Depth Anything V2 model."""
        from models.dav2_wrapper import DAv2Wrapper

        print("[MemMgr] Loading DA-v2...")
        self.dav2_model = DAv2Wrapper(
            model_path=model_path, encoder=encoder, device=self.device
        )
        self.dav2_model.load()
        print("[MemMgr] DA-v2 loaded")

    def predict_and_cache_depth(
        self, image: np.ndarray, cache_key: str, force_recompute: bool = False
    ) -> np.ndarray:
        """
        Predict depth and cache to disk.

        Args:
            image: Input RGB image (HxWx3)
            cache_key: Unique ID for this image (e.g., filename)
            force_recompute: Recompute even if cached

        Returns:
            Depth map (HxW)
        """
        # Check cache first
        if not force_recompute and self.cache.exists(cache_key):
            print(f"[MemMgr] Using cached depth for: {cache_key}")
            cached = self.cache.get(cache_key)
            return cached["depth"]

        # Compute depth
        if self.dav2_model is None:
            raise RuntimeError("DA-v2 not loaded. Call load_dav2() first.")

        print(f"[MemMgr] Computing depth for: {cache_key}")
        result = self.dav2_model.predict({"image": image})
        depth_map = result["depth"]

        # Cache to disk immediately
        self.cache.put(
            key=cache_key,
            depth_map=depth_map,
            metadata={"shape": image.shape, "encoder": self.dav2_model.encoder},
        )

        return depth_map

    def unload_dav2(self) -> None:
        """Unload DA-v2 and free memory."""
        print("[MemMgr] Unloading DA-v2...")
        self.dav2_model = None
        self._free_gpu_memory()
        print("[MemMgr] DA-v2 unloaded, memory freed")

    def load_cotracker(self, model_path: str) -> None:
        """Load CoTracker3 model."""
        from models.cotracker3_wrapper import CoTracker3Wrapper

        print("[MemMgr] Loading CoTracker3...")
        self.cotracker_model = CoTracker3Wrapper(
            model_path=model_path, device=self.device
        )
        self.cotracker_model.load()
        print("[MemMgr] CoTracker3 loaded")

    def predict_tracks(self, video: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Run CoTracker3 inference."""
        if self.cotracker_model is None:
            raise RuntimeError("CoTracker3 not loaded. Call load_cotracker() first.")

        inputs = {"video": video, **kwargs}
        return self.cotracker_model.predict(inputs)

    def unload_cotracker(self) -> None:
        """Unload CoTracker3 and free memory."""
        print("[MemMgr] Unloading CoTracker3...")
        self.cotracker_model = None
        self._free_gpu_memory()
        print("[MemMgr] CoTracker3 unloaded, memory freed")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        return self.cache.stats()
