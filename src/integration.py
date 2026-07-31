"""
integration.py
==============
End-to-End Integration of the Project Iron Pipeline.

This script demonstrates the closed-loop flow:
Video -> DA-V2 -> CoTracker3 -> Projector -> V-JEPA -> Fusion Graph
(Ghost Nodes) -> Iron Agent

Author: AI Assistant
"""

import os
import sys

import numpy as np
import torch

# Adjust path to import from src so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 1. Component Imports
from models.dav2_wrapper import DAv2Wrapper  # noqa: E402
from semantics.semantic_extractor import SemanticExtractor  # noqa: E402
from geometry.projector_vectorized import compute_intrinsics  # noqa: E402
from geometry.projector_vectorized import project_points_to_3d  # noqa: E402
from graph.fusion_graph import build_fusion_graphs_batch  # noqa: E402
from rag_agent import RAGAgent  # noqa: E402


def run_pipeline(video_path: str, output_dir: str):
    print("=" * 60)
    print("Project Iron: End-to-End Pipeline Integration")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Video Input
    # ---------------------------------------------------------
    print("\n[1/7] Processing Video Input...")
    # For demonstration, we simulate a small video batch tensor
    # [Batch, Time, Channels, Height, Width]
    B, T, C, H, W = 1, 4, 3, 224, 224
    print(f"Simulating video input of shape: [{B}, {T}, {C}, {H}, {W}]")
    video = np.random.rand(B, T, C, H, W).astype(np.float32)

    # ---------------------------------------------------------
    # 2. DA-V2 (Depth anything V2)
    # ---------------------------------------------------------
    print("\n[2/7] Running DA-V2 for Depth Maps...")
    depth_maps = []

    # Try loading the actual model if path exists, else simulate
    dav2_path = "../models/weights/dav2/depth_anything_v2_vitl.pth"
    if os.path.exists(dav2_path):
        dav2 = DAv2Wrapper(
            model_path=dav2_path,
            encoder="vitl",
            device="CPU",
        )
        dav2.load()
        for t in range(T):
            # Model expects HxWx3 BGR format
            img_bgr = np.transpose(video[0, t], (1, 2, 0)) * 255.0
            img_bgr = img_bgr.astype(np.uint8)
            depth = dav2.predict({"image": img_bgr})["depth"]
            depth_maps.append(depth)
        depth_maps = np.stack(depth_maps)
    else:
        print("  -> DA-V2 checkpoint not found. Simulating depth maps.")
        depth_maps = np.random.rand(T, H, W).astype(np.float32) * 50.0

    print(f"Generated depth maps of shape: {depth_maps.shape}")

    # ---------------------------------------------------------
    # 3 & 5. CoTracker3 & V-JEPA (Semantic Extractor)
    # ---------------------------------------------------------
    print("\n[3&5/7] Extracting Points and Semantics (CoTracker3 + V-JEPA)...")
    vjepa_xml = "../models/int8/vjepa2_vitl_int8.xml"
    cotracker_pth = "../models/weights/cotracker3/scaled_offline.pth"

    if os.path.exists(vjepa_xml) and os.path.exists(cotracker_pth):
        extractor = SemanticExtractor(
            vjepa_xml=vjepa_xml,
            cotracker_checkpoint=cotracker_pth,
            grid_size=10,
        )
        result = extractor.extract(video)
        tracks = result["tracks"][0]  # Shape: [T, N, 2]
        semantic_tracks = result["semantic_tracks"][0]  # Shape: [T, N, 1024]
        N = tracks.shape[1]
    else:
        print(
            "  -> V-JEPA / CoTracker3 checkpoints not found. "
            "Simulating tracks and embeddings."
        )
        N = 100  # 10x10 grid
        tracks = np.random.rand(T, N, 2).astype(np.float32) * min(H, W)
        semantic_tracks = np.random.rand(T, N, 1024).astype(np.float32)

    print(f"Extracted 2D tracks: {tracks.shape}")
    print(f"Extracted semantic embeddings: {semantic_tracks.shape}")

    # ---------------------------------------------------------
    # 4. Projector
    # ---------------------------------------------------------
    print("\n[4/7] Projecting 2D Tracks to 3D Space...")
    fx, fy, cx, cy = compute_intrinsics(H, W)

    points_3d_per_frame = []
    for t in range(T):
        # Lift 2D pixels to 3D world coordinates using the DA-V2 depth map
        pts_3d, mask = project_points_to_3d(tracks[t], depth_maps[t], fx, fy, cx, cy)
        points_3d_per_frame.append(pts_3d)

    print(f"Projected points to 3D. Shape per frame: {points_3d_per_frame[0].shape}")

    # ---------------------------------------------------------
    # 6. Fusion Graph (Ghost Nodes)
    # ---------------------------------------------------------
    print("\n[6/7] Building Spatial-Semantic Fusion Graph...")
    # Convert our frame-wise data to torch tensors for PyTorch Geometric
    positions_list = [torch.from_numpy(pts) for pts in points_3d_per_frame]
    track_ids_list = [torch.arange(N) for _ in range(T)]
    embeddings_list = [torch.from_numpy(semantic_tracks[t]) for t in range(T)]

    fusion_batch = build_fusion_graphs_batch(
        positions_list=positions_list,
        track_ids_list=track_ids_list,
        embeddings_list=embeddings_list,
        radius=1.5,
    )
    print(
        "Constructed PyTorch Geometric Batch: "
        f"{fusion_batch.num_graphs} graphs, {fusion_batch.num_nodes} total nodes."
    )

    # ---------------------------------------------------------
    # 7. Iron Agent
    # ---------------------------------------------------------
    print("\n[7/7] Iron Agent (RAG Integration)...")
    # In a real run, node features (3D points + semantic vectors) are saved to FAISS
    # and structured metadata is saved to Parquet.
    # The Iron Agent uses these to answer human queries.
    try:
        _ = RAGAgent(
            vector_db_path=".vectordb/faiss_index",
            parquet_path="outputs/tracking_results.parquet",
        )
        print(
            "Iron Agent initialized. Ready to accept queries "
            "like: 'Show me the red car'."
        )
    except Exception as e:
        print(
            "Iron Agent initialization skipped or threw error " f"(needs db files): {e}"
        )

    print("\n" + "=" * 60)
    print("Pipeline Integration Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline("dummy_video.mp4", "outputs/")
