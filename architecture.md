# Project Iron: Architecture Documentation

## Closed-Loop Flow Overview

The pipeline in Project Iron represents an advanced, end-to-end closed-loop system for semantic object tracking, 3D projection, and interactive querying.

The flow progresses as follows:
**Video** -> **DA-V2** -> **CoTracker3** -> **Projector** -> **V-JEPA** -> **Fusion Graph (Ghost Nodes)** -> **Iron Agent**

---

### 1. Video (Input)
The system receives a sequence of video frames (typically an RGB array of shape `[B, T, C, H, W]`). This serves as the raw spatiotemporal input from which all physical and semantic properties will be extracted.

### 2. DA-V2 (Depth Anything V2)
The video frames are passed into the `DA-V2` wrapper (`dav2_wrapper.py`), which uses the Depth Anything V2 model.
* **Input:** RGB frames.
* **Output:** High-resolution Monocular Depth Maps (`[H, W]` per frame).
* **Role:** To provide a dense understanding of scene geometry, converting flat pixels into relative/absolute Z-distances.

### 3. CoTracker3
The `CoTracker3` predictor (`cotracker3_wrapper.py` & `semantic_extractor.py`) processes the video to track specific points (pixels) robustly across time.
* **Input:** RGB video frames and a defined grid size or specific query points.
* **Output:** 2D pixel coordinates `(x, y)` and `visibility` flags for `N` tracked points across `T` frames `[B, T, N, 2]`.
* **Role:** Establishes temporal continuity. It tracks physical points of interest over time, ensuring that an object in frame 1 is linked to its position in frame `T`.

### 4. Projector
The `Projector` (`projector_vectorized.py`) merges the 2D tracking data with the dense depth maps.
* **Input:** 2D tracks `[T, N, 2]`, depth maps `[T, H, W]`, and camera intrinsics (`fx, fy, cx, cy`).
* **Output:** 3D coordinates `(X, Y, Z)` for each tracked point across time.
* **Role:** Lifts 2D pixel tracks into the 3D world space. This establishes the physical geometry of the objects in the environment.

### 5. V-JEPA (Semantic Extraction)
The `V-JEPA` model (`vjepa_wrapper.py` & `semantic_extractor.py`) extracts rich, patch-level semantic embeddings from the video.
* **Input:** RGB video frames.
* **Output:** 1024-dimensional semantic vectors for each spatial patch.
* **Role:** These embeddings are mapped to the spatial coordinates of the CoTracker3 points. This attaches "meaning" (e.g., what the object looks like or represents) to the moving 2D/3D points. The result is a semantic track `[B, T, N, 1024]`.

### 6. Fusion Graph (Ghost Nodes)
The physical 3D coordinates and semantic embeddings are combined into a graph structure (`fusion_graph.py`).
* **Input:** 3D positions `(N, 3)`, track IDs, and Semantic Embeddings `(N, 1024)`.
* **Output:** A PyTorch Geometric `Data` or `Batch` object containing node features (positions + embeddings) and spatial edges.
* **Role:** Creates a spatial-semantic graph representation of the scene. The concept of "Ghost Nodes" allows for handling temporary occlusion or missing embeddings dynamically (falling back to placeholder vectors) while preserving the graph structure.

### 7. Iron Agent
The `Iron Agent` (`rag_agent.py` / `orchestrator.py`) acts as the interactive brain of the closed-loop system.
* **Input:** Natural language queries from the user, the Fusion Graph data, Vector Databases (FAISS), and Parquet tracking tables.
* **Output:** Structured responses containing tracking IDs, 3D locations, and object summaries.
* **Role:** Converts human queries (e.g., "Show me the red car") into vector searches against the graph/FAISS indices, retrieving the specific temporal and 3D locations of the object. This completes the loop by bridging raw pixel streams to semantic, actionable human queries.
