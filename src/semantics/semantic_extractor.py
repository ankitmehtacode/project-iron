"""
semantic_extractor.py
======================
Task 7.1 — Semantic Feature Extraction (V-JEPA + CoTracker3 Integration)

Maps CoTracker3 pixel-coordinate tracks to V-JEPA2 patch embeddings,
extracting a 1024-dimensional semantic vector for each tracked point
at each frame.

─────────────────────────────────────────────────────────────────────
DATA FLOW
─────────────────────────────────────────────────────────────────────

  Video [B, T, C, H, W]
       │
       ├──► CoTracker3  →  tracks [B, T, N, 2]   (pixel x,y per point)
       │
       └──► V-JEPA2     →  features [B, (T//tubelet)×196, 1024]
                               ↑
                        14×14 patch grid
                        (224px / 16px patch = 14 patches per axis)

  Pixel → Patch mapping:
      patch_col = x // PATCH_SIZE          (0..13)
      patch_row = y // PATCH_SIZE          (0..13)
      patch_idx = patch_row * 14 + patch_col   (0..195)

  Per frame t, the V-JEPA feature for tracked point n:
      features[b, (t//tubelet)*196 + patch_idx, :]  → shape (1024,)

  Final output: semantic_tracks [B, T, N, 1024]

─────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────

    from src.semantics.semantic_extractor import SemanticExtractor

    extractor = SemanticExtractor(
        vjepa_xml="models/int8/vjepa2_vitl_int8.xml",
        cotracker_checkpoint="models/weights/cotracker3/scaled_offline.pth",
    )

    # video: numpy [B, T, C, H, W], float32, normalised to [0,1]
    result = extractor.extract(video)
    # result["tracks"]           : [B, T, N, 2]    pixel coords
    # result["semantic_tracks"]  : [B, T, N, 1024] embeddings
    # result["visibility"]       : [B, T, N]        bool

Author: Radhe Tare
Task  : 7.1 — Semantic Feature Extraction
"""

from pathlib import Path

import numpy as np
import openvino as ov
import torch
from cotracker.predictor import CoTrackerPredictor

from src.contracts import FrameGeometry, TemporalSpan
from src.models.preprocess import PreprocessSpec
from src.semantics.patch_mapping import (
    map_tracks_to_embeddings,
    tokens_from_encoder_output,
)

# ─────────────────────────────────────────────────────────────────────
# Constants — V-JEPA2 ViT-L patch configuration
# ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224  # input spatial resolution
PATCH_SIZE = 16  # ViT patch size (224 / 16 = 14 patches per axis)
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES = GRID_SIZE * GRID_SIZE  # 196  (14×14)
EMBED_DIM = 1024  # V-JEPA2 ViT-L embedding dimension


def pixel_to_patch_index(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Map pixel coordinates to a flat patch index in the 14×14 V-JEPA grid.

    V-JEPA2 divides each 224×224 frame into a 14×14 grid of 16×16 patches.
    Patches are indexed row-major: patch 0 is top-left, patch 195 is
    bottom-right.

    Args:
        x: pixel x-coordinates (horizontal), shape arbitrary, values in [0, 224)
        y: pixel y-coordinates (vertical),   shape arbitrary, values in [0, 224)

    Returns:
        patch_indices: flat patch index in [0, 195], same shape as input.
    """
    # Clamp to valid range to handle any edge-case floating point coords
    x = np.clip(x, 0, IMAGE_SIZE - 1)
    y = np.clip(y, 0, IMAGE_SIZE - 1)

    patch_col = (x // PATCH_SIZE).astype(int)  # 0..13
    patch_row = (y // PATCH_SIZE).astype(int)  # 0..13

    return patch_row * GRID_SIZE + patch_col  # 0..195


class SemanticExtractor:
    """
    Extracts 1024-d V-JEPA2 semantic embeddings at CoTracker3 track positions.

    Uses the INT8 OpenVINO IR for V-JEPA2 (fast CPU inference) and the
    original PyTorch CoTracker3 predictor for tracking.

    Pipeline:
        1. Run CoTracker3 → pixel tracks [B, T, N, 2]
        2. Run V-JEPA2 IR  → patch features [B, (T//tubelet)×196, 1024]
        3. Map each (t, x, y) → patch index → look up embedding
        4. Return semantic_tracks [B, T, N, 1024]
    """

    def __init__(
        self,
        vjepa_xml: str,
        cotracker_checkpoint: str,
        grid_size: int = 10,
        device: str = "CPU",
    ):
        """
        Args:
            vjepa_xml           : Path to V-JEPA2 OpenVINO IR (.xml).
                                  Use the INT8 version for efficiency.
            cotracker_checkpoint: Path to CoTracker3 .pth checkpoint.
            grid_size           : Number of points per axis for grid tracking
                                  (total tracks = grid_size²). Default 10 → 100 pts.
            device              : OpenVINO device string. Default "CPU".
        """
        self.grid_size = grid_size

        # ── Load the preprocessing spec that belongs to this model ─
        # Loaded BEFORE the model, and from beside the model file, so a wrapper
        # can never run without knowing how its input must be prepared. This is
        # the fix for the defect where this path fed the encoder raw [0,1]
        # pixels while the model expected channel-standardised input.
        self._preprocess = PreprocessSpec.load_for_model(Path(vjepa_xml))
        print(f"  preprocessing: {self._preprocess.describe()}")

        # ── Load V-JEPA2 OpenVINO IR ──────────────────────────────
        print(f"Loading V-JEPA2 IR from: {vjepa_xml}")
        core = ov.Core()
        compiled = core.compile_model(vjepa_xml, device)
        self._vjepa_infer = compiled.create_infer_request()

        # Confirm output shape at load time
        out_shape = compiled.output(0).partial_shape
        print(f"  V-JEPA2 output shape: {out_shape}")  # [?, T*196, 1024]

        # ── Load CoTracker3 ───────────────────────────────────────
        print(f"Loading CoTracker3 from: {cotracker_checkpoint}")
        self._tracker = CoTrackerPredictor(checkpoint=cotracker_checkpoint)
        self._tracker.eval()

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _run_cotracker(
        self, video_torch: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run CoTracker3 grid tracking on a video tensor.

        Args:
            video_torch: float32 tensor [B, T, C, H, W], values in [0, 1].

        Returns:
            tracks    : float32 ndarray [B, T, N, 2]  — pixel (x, y)
            visibility: bool    ndarray [B, T, N]
        """
        with torch.no_grad():
            tracks, visibility = self._tracker(
                video_torch,
                grid_size=self.grid_size,
                grid_query_frame=0,
                backward_tracking=False,
            )
        return tracks.cpu().numpy(), visibility.cpu().numpy().astype(bool)

    def _run_vjepa(self, video_np: np.ndarray) -> np.ndarray:
        """
        Run V-JEPA2 IR inference to extract patch embeddings.

        Applies the model's own PreprocessSpec first. Before this existed, the
        array went into the encoder as raw [0, 1] pixels while V-JEPA2 expects
        ImageNet-standardised input — every activation shifted, every embedding
        degraded, and nothing raised. Measured against the official reference
        the per-patch cosine p1 was 0.332; with standardisation applied it is
        1.000000.

        Args:
            video_np: float32 ndarray [B, T, C, H, W], values in [0, 1].

        Returns:
            features: float32 ndarray [B, n_temporal * n_spatial, 1024].
            Note that n_temporal is ``T // tubelet``, not ``T``.
        """
        standardised = self._preprocess.apply(video_np)
        self._vjepa_infer.infer({"video": standardised})
        return self._vjepa_infer.get_output_tensor(0).data.copy()

    @property
    def preprocess_sha(self) -> str:
        """Hash of the preprocessing in force.

        Every embedding this extractor produces is only comparable with others
        sharing this value; it belongs in the metadata of any artifact built
        from them.
        """
        return self._preprocess.preprocess_sha()

    def _map_tracks_to_embeddings(
        self,
        tracks: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Look up V-JEPA patch embeddings at each tracked pixel coordinate.

        Delegates to :mod:`src.semantics.patch_mapping`, which routes the
        encoder output through :class:`~src.contracts.PatchTokens`. That type's
        ``n_temporal == frames_covered // tubelet`` assertion is what makes the
        temporal misassignment this method used to contain structurally
        impossible rather than merely fixed.

        Args:
            tracks  : float32 [B, T, N, 2] — pixel (x, y) per track
            features: float32 [B, n_temporal * n_spatial, dim]

        Returns:
            semantic_tracks: float32 [B, T, N, dim]
        """
        batch, frames, _points, _ = tracks.shape
        spec = self._preprocess
        grid = (
            spec.resolution[0] // spec.patch_size,
            spec.resolution[1] // spec.patch_size,
        )
        geometry = FrameGeometry(width=spec.resolution[1], height=spec.resolution[0])

        # Nanosecond span synthesized from the frame count. The encoder carries
        # no wall-clock time; the span exists here to carry the tubelet into
        # the contract check.
        span = TemporalSpan(
            start_ts_ns=0,
            end_ts_ns=max(1, frames),
            frames_covered=frames,
            tubelet=spec.tubelet,
        )

        outputs = []
        for index in range(batch):
            tokens = tokens_from_encoder_output(
                features,
                batch_index=index,
                span=span,
                grid=grid,
                geometry=geometry,
                encoder_sha=self.preprocess_sha,
            )
            outputs.append(map_tracks_to_embeddings(tokens, tracks[index]))

        return np.stack(outputs, axis=0).astype(np.float32)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def extract(self, video: np.ndarray) -> dict:
        """
        Full pipeline: track → embed → map.

        Args:
            video: float32 ndarray [B, T, C, H, W]
                   Must be normalised to [0, 1].
                   Spatial resolution must be 224×224.

        Returns:
            dict with keys:
                "tracks"          : float32 [B, T, N, 2]     pixel coords
                "visibility"      : bool    [B, T, N]         track visibility
                "semantic_tracks" : float32 [B, T, N, 1024]  V-JEPA embeddings
        """
        assert video.ndim == 5, "Expected video shape [B, T, C, H, W]"
        assert video.shape[3] == IMAGE_SIZE and video.shape[4] == IMAGE_SIZE, (
            f"Expected spatial size {IMAGE_SIZE}x{IMAGE_SIZE}, "
            f"got {video.shape[3]}x{video.shape[4]}"
        )

        # Step 1 — CoTracker3: pixel tracks
        video_torch = torch.from_numpy(video)
        tracks, visibility = self._run_cotracker(video_torch)
        # tracks     : [B, T, N, 2]
        # visibility : [B, T, N]

        # Step 2 — V-JEPA2: patch embeddings
        features = self._run_vjepa(video)
        # features : [B, T*196, 1024]

        # Step 3 — Map pixel coords → patch embeddings
        semantic_tracks = self._map_tracks_to_embeddings(tracks, features)
        # semantic_tracks : [B, T, N, 1024]

        return {
            "tracks": tracks,  # [B, T, N, 2]
            "visibility": visibility,  # [B, T, N]
            "semantic_tracks": semantic_tracks,  # [B, T, N, 1024]
        }


# ─────────────────────────────────────────────────────────────────────
# Quick sanity test (no real model needed — uses dummy tensors)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Patch mapping sanity check ===")

    # Test pixel_to_patch_index
    test_cases = [
        (0, 0, 0),  # top-left     → patch 0
        (223, 223, 195),  # bottom-right → patch 195
        (16, 0, 1),  # second patch in row 0
        (0, 16, 14),  # first patch in row 1
        (112, 112, 7 * 14 + 7),  # centre
    ]
    all_pass = True
    for x, y, expected in test_cases:
        got = pixel_to_patch_index(np.array([x]), np.array([y]))[0]
        status = "✅" if got == expected else "❌"
        if got != expected:
            all_pass = False
        print(
            f"  {status}  pixel ({x:3d},{y:3d}) → patch {got:3d}  (expected {expected})"
        )

    print()
    if all_pass:
        print("All patch mapping tests passed ✅")
    else:
        print("Some tests FAILED ❌ — check patch mapping logic")

    # Test shape arithmetic
    print("\n=== Shape arithmetic check ===")
    B, T, N = 1, 4, 100
    dummy_tracks = np.random.rand(B, T, N, 2).astype(np.float32) * 224
    dummy_features = np.random.rand(B, T * NUM_PATCHES, EMBED_DIM).astype(np.float32)

    extractor_dummy = object.__new__(SemanticExtractor)
    result = SemanticExtractor._map_tracks_to_embeddings(
        extractor_dummy, dummy_tracks, dummy_features
    )
    print(f"  Input  tracks   : {dummy_tracks.shape}")
    print(f"  Input  features : {dummy_features.shape}")
    print(f"  Output semantic : {result.shape}  (expected ({B},{T},{N},{EMBED_DIM}))")
    assert result.shape == (B, T, N, EMBED_DIM), "Shape mismatch!"
    print("  Shape test passed ✅")
