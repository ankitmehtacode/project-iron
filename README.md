# Project Iron

[![CI](https://github.com/ankitmehtacode/project-iron/actions/workflows/ci.yaml/badge.svg)](https://github.com/ankitmehtacode/project-iron/actions/workflows/ci.yaml)

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

<!-- BEGIN TREE -->
<!-- regenerate with: python scripts/gen_tree.py -->

```
project-iron/
├── .claude/
│   └── skills/
│       ├── iron-cascade-runtime/
│       │   └── SKILL.md
│       ├── iron-contracts/
│       │   └── SKILL.md
│       ├── iron-eval-discipline/
│       │   └── SKILL.md
│       ├── iron-events/
│       │   └── SKILL.md
│       ├── iron-model-export/
│       │   └── SKILL.md
│       ├── iron-privacy-security/
│       │   └── SKILL.md
│       ├── iron-provenance/
│       │   └── SKILL.md
│       └── iron-testing/
│           └── SKILL.md
├── .github/
│   └── workflows/
│       └── ci.yaml
├── configs/
│   ├── golden/
│   │   ├── v1-driving.golden.json
│   │   └── v2-indoor.golden.json
│   ├── preprocess/
│   │   └── vjepa2_vitl.preprocess.json
│   ├── datasets.yaml
│   └── default.yaml
├── scripts/
│   ├── build_calibration_set.py
│   ├── cascade_bench.py
│   ├── cvat_project.py
│   ├── demo_events.py
│   ├── env_gate.py
│   ├── eval_report.py
│   ├── export_cotracker3_onnx.py
│   ├── export_depth_anything_onnx.py
│   ├── export_vjepa_onnx.py
│   ├── export_vjepa_ov.py
│   ├── fetch_dataset.py
│   ├── fetch_weights.py
│   ├── gen_tree.py
│   ├── make_golden_vectors.py
│   ├── mock_pipeline.py
│   ├── quantize_cotracker3.py
│   ├── quantize_depth_anything.py
│   ├── quantize_vjepa.py
│   ├── rebuild_index.py
│   └── vjepa_wrapper.py
├── src/
│   ├── cascade/
│   │   ├── __init__.py
│   │   ├── motion.py
│   │   ├── runner.py
│   │   ├── stage.py
│   │   └── stages.py
│   ├── contracts/
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   ├── fields.py
│   │   ├── frames.py
│   │   ├── geometry.py
│   │   └── tokens.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── registry.py
│   ├── endurance/
│   │   ├── __init__.py
│   │   ├── gates.py
│   │   ├── memory.py
│   │   └── runner.py
│   ├── events/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── geometry/
│   │   ├── ocr/
│   │   │   ├── __init__.py
│   │   │   └── text_detector.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── metadata_fusion.py
│   │   │   └── performance_eval.py
│   │   ├── __init__.py
│   │   ├── enhanced_cotracker.py
│   │   └── projector_vectorized.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── embedding_preparation.py
│   │   └── fusion_graph.py
│   ├── interface/
│   │   ├── ui/
│   │   │   ├── components/
│   │   │   │   ├── Dashboard/
│   │   │   │   │   └── AnalyticsPanel.js
│   │   │   │   └── Visualizer/
│   │   │   │       └── GaussianSplat.js
│   │   │   ├── data/
│   │   │   │   └── raw/
│   │   │   │       └── test_video.mp4
│   │   │   ├── workers/
│   │   │   │   └── GaussianSorter.worker.js
│   │   │   ├── app.js
│   │   │   ├── index.html
│   │   │   ├── serve.py
│   │   │   ├── three.min.js
│   │   │   └── trajectory_data.json
│   │   ├── __init__.py
│   │   ├── data_converter.py
│   │   └── temporal_stitching.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cotracker3_wrapper.py
│   │   ├── dav2_wrapper.py
│   │   ├── model_wrapper.py
│   │   └── vjepa_wrapper.py
│   ├── semantics/
│   │   ├── __init__.py
│   │   ├── pca_reducer.py
│   │   └── semantic_extractor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── disk_cache.py
│   │   └── parquet_writer.py
│   ├── __init__.py
│   ├── artifacts.py
│   ├── config.py
│   ├── integration.py
│   ├── main.py
│   ├── memory_manager.py
│   ├── orchestrator.py
│   ├── provenance.py
│   ├── rag_agent.py
│   └── vector_database.py
├── tests/
│   ├── fixtures/
│   │   └── cvat_miniature.xml
│   ├── golden/
│   │   ├── clips/
│   │   │   ├── real_test_video.npz
│   │   │   ├── synthetic_seeded_noise.npz
│   │   │   ├── synthetic_spatial_gradient.npz
│   │   │   └── synthetic_tubelet_probe.npz
│   │   ├── reference/
│   │   │   ├── real_test_video.npz
│   │   │   ├── synthetic_seeded_noise.npz
│   │   │   ├── synthetic_spatial_gradient.npz
│   │   │   └── synthetic_tubelet_probe.npz
│   │   └── manifest.json
│   ├── test_calibration_set.py
│   ├── test_cascade.py
│   ├── test_config.py
│   ├── test_contracts_properties.py
│   ├── test_converters.py
│   ├── test_cvat_roundtrip.py
│   ├── test_data_registry.py
│   ├── test_depth_honesty.py
│   ├── test_endurance_gates.py
│   ├── test_env_gate.py
│   ├── test_events_schema.py
│   ├── test_fetch_dataset.py
│   ├── test_golden_sets.py
│   ├── test_golden_vectors.py
│   ├── test_known_bugs.py
│   ├── test_preprocess_spec.py
│   ├── test_provenance.py
│   └── test_temporal_stitching.py
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── architecture.md
├── conda_environment.yaml
├── Dockerfile
├── endurance_run.py
├── FOUNDATION_REPORT.md
├── locking-requirements.txt
├── Makefile
├── mypy.ini
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── setup.py
```
<!-- END TREE -->

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ankitmehtacode/project-iron.git
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
| disparity_rel | float32 | Relative inverse depth, arbitrary per-frame scale. **Not metres.** |
| depth_units | string | What `disparity_rel` means; `"disparity_rel"` today |
| ocr_text | string | Detected text |
| confidence | float32 | Tracking confidence |

> **This column used to be named `z` and documented as "Depth (meters)". It was
> never metres.** Depth-Anything-V2 emits relative inverse depth with unknown
> scale *and* unknown shift, so no constant converts it. Anything computed from
> the old `z` as a distance was wrong. Converting to metric depth needs an
> anchoring step and real camera calibration; until both exist, `depth_units`
> stays `"disparity_rel"`.

---

## Running the Pipeline

End-to-end integration demo (simulates input where weights are absent):

```bash
python -m src.integration
```

Parquet writer smoke test, no model weights required:

```bash
python scripts/mock_pipeline.py
```

Memory-stability soak test:

```bash
python endurance_run.py
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
