# Project Iron

AI inference system with memory-efficient model orchestration for robotics and spatial understanding.

---

## Team Structure

```
src/
├── geometry/          ← Priyanshu's team (3D spatial, depth, tracking)
├── semantics/         ← Radhe's team (language, embeddings, reasoning)
├── interface/         ← Rishi's team (API, user interaction)
├── models/            ← Model wrappers (CoTracker3, DA-v2, V-JEPA)
├── utils/             ← Utilities (disk cache, logging)
├── orchestrator.py    ← Sequential inference with OpenVINO mmap
└── memory_manager.py  ← Memory-safe model loading
```

---

## Directory Structure

```
project-iron/
├── Dockerfile
├── locking-requirements.txt
├── conda_environment.yaml
├── .pre-commit-config.yaml
├── .gitignore
├── setup.py
│
├── src/
│   ├── geometry/          ← Priyanshu (3D, depth, tracking)
│   │   ├── __init__.py
│   │   ├── enhanced_cotracker.py
│   │   └── projector_vectorized.py
│   ├── semantics/         ← Radhe (language, reasoning)
│   │   └── __init__.py
│   ├── interface/         ← Rishi (API, orchestration)
│   │   └── __init__.py
│   ├── models/            ← AI model wrappers
│   │   ├── __init__.py
│   │   ├── model_wrapper.py       (base class)
│   │   ├── cotracker3_wrapper.py  (point tracking)
│   │   ├── dav2_wrapper.py        (depth estimation)
│   │   └── vjepa_wrapper.py       (video understanding)
│   ├── utils/
│   │   ├── disk_cache.py          (SSD caching)
│   │   └── parquet_writer.py      (output format)
│   ├── orchestrator.py            (OpenVINO sequential loading)
│   ├── memory_manager.py          (RAM management)
│   └── main.py
│
├── models/                ← Saved checkpoints (.pt, .pth, .xml)
├── data/
│   ├── raw/              ← Input videos, images
│   └── processed/        ← Cached depth maps, outputs
├── output/               ← Parquet results
├── tests/
├── configs/
├── scripts/
└── docs/
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Dalbirsm03/project-iron.git
cd project-iron

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r locking-requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## Core Features

### 1. Memory-Efficient Model Loading

Uses OpenVINO's `ENABLE_MMAP` to avoid OOM:

```python
from src.orchestrator import InferenceOrchestrator

orch = InferenceOrchestrator(models_dir="models/")
orch.load_model("depth_estimation")
result = orch.infer({"input": image_tensor})
```

### 2. Disk Caching (Prevents RAM Overflow)

```python
from src.memory_manager import MemoryManager

mem_mgr = MemoryManager(cache_dir=".cache", cache_size_gb=5.0)

# Stage 1: Depth
mem_mgr.load_dav2("checkpoints/depth.pth")
depth = mem_mgr.predict_and_cache_depth(image, key="frame_001")
mem_mgr.unload_dav2()  # Frees RAM

# Stage 2: Tracking (DA-v2 already unloaded)
mem_mgr.load_cotracker("checkpoints/cotracker.pth")
tracks = mem_mgr.predict_tracks(video)
```

### 3. Model Wrappers

Standard interface for all models:

```python
from src.models.dav2_wrapper import DAv2Wrapper
from src.models.cotracker3_wrapper import CoTracker3Wrapper

# Depth estimation
depth_model = DAv2Wrapper("checkpoints/depth_anything_v2_vitl.pth", encoder="vitl")
depth_model.load()
result = depth_model.predict({"image": cv2.imread("input.jpg")})

# Point tracking
tracker = CoTracker3Wrapper("checkpoints/scaled_offline.pth")
tracker.load()
tracks = tracker.predict({"video": video_tensor, "grid_size": 30})
```

### 4. Parquet Output Format

```python
from src.utils.parquet_writer import ParquetWriter

writer = ParquetWriter("output/results.parquet")
writer.write_batch(
    track_ids=[0, 1, 2],
    frame_indices=[0, 0, 0],
    x_coords=[100, 200, 300],
    y_coords=[150, 250, 350],
    z_coords=[2.5, 3.0, 2.8],
    ocr_texts=["", "STOP", ""],
    confidences=[0.95, 0.98, 0.92]
)
writer.close()
```

**Schema:**
| Column | Type | Description |
|--------|------|-------------|
| track_id | int64 | Unique point ID |
| frame_idx | int64 | Frame number |
| x | float32 | Pixel x-coordinate |
| y | float32 | Pixel y-coordinate |
| z | float32 | Depth (meters) |
| ocr_text | string | Detected text |
| confidence | float32 | Tracking confidence |

---

## Running the Pipeline

See `scripts/pipeline_example.py`:

```bash
python scripts/pipeline_example.py \
  --video data/raw/test_video.mp4 \
  --output output/results.parquet
```

---

## Docker Deployment

```bash
docker build -t project-iron .
docker run --rm -it -v $(pwd)/data:/app/data project-iron
```

---

## Development Workflow

### Pre-commit Hooks

Auto-format and lint on every commit:

```bash
git add .
git commit -m "feat: add new feature"
# black + flake8 run automatically
```

### Team Responsibilities

| Team | Module | Focus |
|------|--------|-------|
| Priyanshu | `src/geometry/` | 3D spatial, depth, tracking |
| Radhe | `src/semantics/` | Language, embeddings, reasoning |
| Rishi | `src/interface/` | API, orchestration, deployment |

---

## Technical Stack

- **Python**: 3.10+
- **Inference**: OpenVINO 2024.6.0 (memory-mapped models)
- **ML Framework**: PyTorch 2.2.0 (CPU)
- **Models**: CoTracker3, Depth Anything V2, V-JEPA 2
- **Storage**: Parquet (PyArrow)
- **Vector Store**: FAISS-CPU
- **Tracking**: MLflow

---

## License

[Add license]

---

## Citation

If you use this project, please cite:
- [CoTracker3](https://co-tracker.github.io/)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [V-JEPA 2](https://github.com/facebookresearch/vjepa2)
