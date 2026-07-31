"""
embedding_preparation.py

Complete pipeline to extract, reduce, and prepare semantic embeddings
for fusion graph integration.
This module handles the end-to-end workflow from raw video input to PyTorch tensors
ready for graph construction
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional  # noqa: F401

from src.semantics.semantic_extractor import SemanticExtractor
from src.semantics.pca_reducer import PCAReducer


def validate_embeddings(
    embeddings: torch.Tensor,
    num_objects: int,
    embed_dim: int = 64,
) -> bool:
    checks = {
        "Type is torch.Tensor": isinstance(embeddings, torch.Tensor),
        f"Shape is ({num_objects}, {embed_dim})": embeddings.shape
        == (num_objects, embed_dim),
        "Dtype is float32": embeddings.dtype == torch.float32,
        "No NaN values": not torch.isnan(embeddings).any(),
        "No Inf values": not torch.isinf(embeddings).any(),
        "Matches tracked objects": embeddings.shape[0] == num_objects,
    }

    all_pass = True
    print("\n" + "=" * 50)
    print("EMBEDDING VALIDATION")
    print("=" * 50)
    for check_name, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check_name}")
        if not passed:
            all_pass = False
    print("=" * 50 + "\n")

    if not all_pass:
        raise ValueError("Embedding validation failed!")

    return True


def print_embedding_diagnostics(embeddings: np.ndarray, stage: str = "") -> None:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    print(f"\n{'-'*50}")
    if stage:
        print(f"Stage: {stage}")
    print(f"{'-'*50}")
    print(f"Shape:              {embeddings.shape}")
    print(f"Dtype:              {embeddings.dtype}")
    print(f"Min/Max:            {embeddings.min():.6f} / {embeddings.max():.6f}")
    print(f"Mean:               {embeddings.mean():.6f}")
    print(f"Std:                {embeddings.std():.6f}")
    print(f"Memory (MB):        {embeddings.nbytes / 1e6:.2f}")
    print(f"NaN count:          {np.isnan(embeddings).sum()}")
    print(f"Inf count:          {np.isinf(embeddings).sum()}")
    print(f"{'-'*50}\n")


def prepare_embeddings_for_fusion_graph(
    video: np.ndarray,
    vjepa_xml: str,
    cotracker_checkpoint: str,
    pca_path: str,
    grid_size: int = 10,
    device: str = "cuda",
    verbose: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    if verbose:
        print("\n" + "=" * 60)
        print("EMBEDDING EXTRACTION PIPELINE")
        print("=" * 60)

    # Input validation: ensure video has correct shape (B, T, C, H, W)
    if video.ndim != 5:
        raise ValueError(
            f"Expected video shape (B, T, C, H, W) with 5 dimensions, "
            f"got {video.ndim} dimensions. Actual shape: {video.shape}"
        )
    B, T, C, H, W = video.shape
    if H != 224 or W != 224:
        raise ValueError(
            f"Expected spatial resolution 224x224, got {H}x{W}. "
            f"Video must be resized to 224x224 before extraction."
        )
    if verbose:
        print(f"\nInput validation passed: video shape {video.shape} is correct")

    # Step 1: Initialize SemanticExtractor
    if verbose:
        print("\n[Step 1/6] Initializing SemanticExtractor...")

    extractor = SemanticExtractor(
        vjepa_xml=vjepa_xml,
        cotracker_checkpoint=cotracker_checkpoint,
        grid_size=grid_size,
        device="CPU",  # OpenVINO uses CPU
    )

    if verbose:
        print("  [DONE] SemanticExtractor initialized")
        print(f"    - V-JEPA IR: {vjepa_xml}")
        print(f"    - CoTracker3: {cotracker_checkpoint}")
        print(f"    - Grid size: {grid_size}x{grid_size} = {grid_size**2} points")

    # Step 2: Extract semantic embeddings
    if verbose:
        print("\n[Step 2/6] Extracting embeddings from video...")
        print(f"  Input video shape: {video.shape}")

    result = extractor.extract(video)
    semantic_embeddings = result["semantic_tracks"]  # [B, T, N, 1024]

    if verbose:
        print("  [DONE] Extraction complete")
        print_embedding_diagnostics(semantic_embeddings, "After SemanticExtractor")

    # Step 3: Load or train PCA with error handling
    if verbose:
        print("\n[Step 3/6] Loading PCA reducer...")

    pca_path_obj = Path(pca_path)
    reducer = None

    # Try loading pre-trained PCA
    if pca_path_obj.exists():
        try:
            reducer = PCAReducer.load(str(pca_path_obj))
            if verbose:
                print(f"  [DONE] Loaded pre-trained PCA from: {pca_path}")
        except Exception as e:
            print(f"  [WARNING] Failed to load PCA from {pca_path}: {str(e)}")
            print("  Proceeding to train new PCA model...")
            reducer = None

    # Train new PCA if not loaded
    if reducer is None:
        try:
            if verbose and pca_path_obj.exists():
                print("  [WARNING] Training new PCA on current batch...")
            elif verbose:
                print("  [WARNING] PCA model not found. Training on current batch...")

            reducer = PCAReducer(n_components=64)
            embeddings_list = [semantic_embeddings]
            reducer.fit(embeddings_list)

            # Save for future use
            pca_path_obj.parent.mkdir(parents=True, exist_ok=True)
            reducer.save(str(pca_path_obj))

            if verbose:
                print(f"  [DONE] PCA trained on {len(embeddings_list)} clip(s)")
                print(f"  [DONE] PCA model saved to: {pca_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to train PCA model: {str(e)}. "
                f"Ensure semantic embeddings are valid and properly shaped."
            )

    # Step 4: Apply dimensionality reduction
    if verbose:
        print("\n[Step 4/6] Reducing embeddings: 1024 to 64...")

    compact_embeddings = reducer.transform(semantic_embeddings)  # [B, T, N, 64]

    if verbose:
        print("  [DONE] Dimensionality reduction complete")
        print(f"    Original size: {semantic_embeddings.nbytes / 1e6:.2f} MB")
        print(f"    Reduced size:  {compact_embeddings.nbytes / 1e6:.2f} MB")
        print(
            "    Compression:   "
            f"{semantic_embeddings.nbytes / compact_embeddings.nbytes:.1f}x"
        )
        print_embedding_diagnostics(compact_embeddings, "After PCA reduction")

    # Step 5: Compute identity embeddings
    if verbose:
        print("\n[Step 5/6] Computing identity embeddings (time average)...")

    # Average across time dimension: [B, T, N, 64] to [B, N, 64]
    identity_embedding = compact_embeddings.mean(axis=1)

    if verbose:
        print("  [DONE] Time averaging complete")
        print(f"    Shape after averaging: {identity_embedding.shape}")

    # Batch size validation: ensure we have at least one batch
    if identity_embedding.shape[0] < 1:
        raise ValueError(
            f"Expected at least 1 batch after averaging, "
            f"got shape {identity_embedding.shape}"
        )

    # Extract first batch (assumes B >= 1; for single video typically B=1)
    embeddings_np = identity_embedding[0]  # [N, 64]

    if verbose:
        print(f"  [DONE] Batch extracted: {embeddings_np.shape}")
        print_embedding_diagnostics(embeddings_np, "After batch extraction")

    # Step 6: Convert to PyTorch & align device
    if verbose:
        print("\n[Step 6/6] Converting to PyTorch and moving to device...")

    embeddings_torch = torch.from_numpy(embeddings_np).float()

    # Ensure device availability: validate and adjust device if needed
    if device == "cuda" and not torch.cuda.is_available():
        if verbose:
            print("  [WARNING] CUDA not available, falling back to CPU")
        device = "cpu"

    # Move embeddings to target device
    embeddings_torch = embeddings_torch.to(device)
    if verbose:
        print(f"  [DONE] Moved tensor to device: {device}")

    # Optional: L2 normalization for improved downstream operations
    if normalize:
        embeddings_torch = torch.nn.functional.normalize(embeddings_torch, p=2, dim=1)
        if verbose:
            print("  [DONE] Applied L2 normalization")

    if verbose:
        print("  [DONE] Tensor prepared for fusion graph")
        print(f"    - Shape: {embeddings_torch.shape}")
        print(f"    - Device: {embeddings_torch.device}")
        print(f"    - Dtype: {embeddings_torch.dtype}")
        print(f"    - Normalized: {normalize}")

    # Step 7: Validate embeddings shape and alignment
    if verbose:
        print("\n[Step 7/7] Final validation and shape alignment check...")

    num_objects = compact_embeddings.shape[2]  # N from [B, T, N, 64]

    # Safety assertion: ensure embeddings shape matches expected object count
    if embeddings_torch.shape[0] != num_objects:
        raise AssertionError(
            f"Embeddings shape mismatch: expected {num_objects} objects, "
            f"got {embeddings_torch.shape[0]}. This indicates an error in "
            f"the embedding preparation pipeline."
        )
    if embeddings_torch.shape[1] != 64:
        raise AssertionError(
            f"Embedding dimension mismatch: expected 64 dimensions, "
            f"got {embeddings_torch.shape[1]}"
        )

    if verbose:
        print(
            f"  [DONE] Shape alignment verified: {embeddings_torch.shape} "
            f"matches {num_objects} tracked objects x 64 dims"
        )

    # Run comprehensive validation checks
    validate_embeddings(embeddings_torch, num_objects=num_objects, embed_dim=64)

    if verbose:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Output: torch.Tensor of shape {embeddings_torch.shape}")
        print("Ready to pass to build_fusion_graph(...)")
        print("=" * 60 + "\n")

    # Return embeddings tensor [N, 64]
    # Can be passed directly to fusion graph construction
    # Note: If integrating with modules expecting embeddings on same device
    # as other tensors, move to that device using: embeddings.to(other_tensor.device)
    return embeddings_torch


def extract_embeddings_batch(
    videos: list[np.ndarray],
    vjepa_xml: str,
    cotracker_checkpoint: str,
    pca_path: str,
    grid_size: int = 10,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """
    Process multiple videos and return list of embedding tensors.

    Args:
        videos: List of numpy arrays, each [B, T, C, H, W]
        vjepa_xml: Path to V-JEPA2 model
        cotracker_checkpoint: Path to CoTracker3 checkpoint
        pca_path: Path to PCA model
        grid_size: Number of tracking points per axis
        device: "cuda" or "cpu"

    Returns:
        List of torch.Tensor, each [N, 64]
    """
    embeddings_list = []

    for i, video in enumerate(videos):
        print(f"\n--- Processing video {i+1}/{len(videos)} ---")
        emb = prepare_embeddings_for_fusion_graph(
            video=video,
            vjepa_xml=vjepa_xml,
            cotracker_checkpoint=cotracker_checkpoint,
            pca_path=pca_path,
            grid_size=grid_size,
            device=device,
            verbose=True,
        )
        embeddings_list.append(emb)

    return embeddings_list


if __name__ == "__main__":
    """Quick sanity test with dummy data."""
    print("=" * 60)
    print("Embedding Preparation Pipeline - Sanity Test")
    print("=" * 60)

    # Generate dummy video for testing
    np.random.seed(42)
    video_dummy = np.random.randn(1, 4, 3, 224, 224).astype(np.float32)
    video_dummy = np.clip(video_dummy, 0, 1)

    print(f"\nTest video shape: {video_dummy.shape}")
    print(f"Test video range: [{video_dummy.min():.4f}, {video_dummy.max():.4f}]")
