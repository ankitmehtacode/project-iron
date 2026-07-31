"""Generate the golden-vector fixtures: fixed clips and reference embeddings.

Why golden vectors exist
------------------------
An exported vision transformer that is wrong still runs, still returns
plausibly-shaped tensors, and still passes every smoke test. Position embeddings
not interpolated after a resolution change, the context encoder exported where
the EMA target encoder was meant, preprocessing that never got wired in — each
produces exactly that failure mode, and none of them raise. A golden vector is
the single test that catches all of them: a fixed input must produce a known
embedding, within tolerance, forever.

The reference is produced by the **official PyTorch implementation** with the
**official preprocessor** — the code path in ``src/models/vjepa_wrapper.py``,
which loads ``vjepa2_preprocessor`` from torch.hub. That path is currently
unused by production, which is precisely the defect this fixture measures: the
OpenVINO path in ``src/semantics/semantic_extractor.py`` applies no
standardisation at all.

Two phases, because they have different prerequisites
-----------------------------------------------------
Clip generation needs only numpy and (for the real sequence) OpenCV, so the
inputs are deterministic, committed, and reviewable without a GPU or a
checkpoint. Reference generation needs torch and the official weights.

    python scripts/make_golden_vectors.py --clips-only   # inputs, no weights
    python scripts/make_golden_vectors.py                # inputs + references

Running without weights exits 2 with a clean message rather than a traceback,
and never writes a partial or fabricated reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_ROOT = REPO_ROOT / "tests" / "golden"
CLIP_DIR = GOLDEN_ROOT / "clips"
REFERENCE_DIR = GOLDEN_ROOT / "reference"
MANIFEST_PATH = GOLDEN_ROOT / "manifest.json"

REAL_VIDEO = REPO_ROOT / "src" / "interface" / "ui" / "data" / "raw" / "test_video.mp4"

# Fixed for reproducibility. Changing any of these changes every fixture, so
# they are recorded in the manifest and asserted by the golden test.
SEED = 20260801
CLIP_FRAMES = 4
CLIP_H = CLIP_W = 224


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash an array by its exact bytes, shape and dtype.

    Shape and dtype are folded in because two arrays with identical bytes but
    different shapes are different fixtures, and a reshape that silently
    changed the token layout is one of the bugs this is meant to catch.
    """
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Clip construction. uint8 on disk: these are images, and float32 would
# quadruple the committed size for no added information.
# ---------------------------------------------------------------------------


def clip_spatial_gradient() -> np.ndarray:
    """A static horizontal/vertical gradient, identical on every frame.

    Static on purpose: with correct tubelet handling every temporal slot sees
    the same content, so the reference embedding's temporal slots should be
    near-identical. A reference where they differ points at temporal handling
    rather than at spatial encoding.
    """
    ramp_x = np.linspace(0, 255, CLIP_W, dtype=np.float32)
    ramp_y = np.linspace(0, 255, CLIP_H, dtype=np.float32)
    frame = (ramp_x[None, :] * 0.5 + ramp_y[:, None] * 0.5).astype(np.uint8)
    rgb = np.stack(
        [frame, np.roll(frame, 32, axis=1), np.roll(frame, 64, axis=0)], axis=0
    )
    return np.repeat(rgb[None, ...], CLIP_FRAMES, axis=0)[None, ...]


def clip_tubelet_probe() -> np.ndarray:
    """Black for the first half of the clip, white for the second.

    Straddles the tubelet boundary exactly. With tubelet 2 and 4 frames, slot 0
    sees only black and slot 1 only white, so the two reference slots must
    differ sharply. If they do not, the temporal grouping is not what the
    configuration claims — the same probe used by
    ``test_patch_mapper_temporal_alignment``.
    """
    clip = np.zeros((1, CLIP_FRAMES, 3, CLIP_H, CLIP_W), dtype=np.uint8)
    clip[:, CLIP_FRAMES // 2 :] = 255
    return clip


def clip_seeded_noise() -> np.ndarray:
    """Seeded uniform noise: an unstructured control.

    Exercises no data-dependent branch, which is the point — it isolates raw
    numerical response from any content the model might partially recognise.
    """
    rng = np.random.default_rng(SEED)
    return rng.integers(
        0, 256, size=(1, CLIP_FRAMES, 3, CLIP_H, CLIP_W), dtype=np.uint8
    )


def clip_from_real_video() -> np.ndarray | None:
    """First frames of the repository's test video, resized to the clip size.

    Real content matters: synthetic clips exercise no natural image statistics,
    and an export defect can be invisible on noise while obvious on a scene.
    Returns None when OpenCV or the video is unavailable, in which case the
    fixture set is simply smaller and the manifest records its absence.

    The source is 320x176, so reaching 224x224 stretches the aspect ratio. That
    is recorded as the resize policy rather than silently applied — a fixture
    whose preprocessing is undocumented cannot be reproduced.
    """
    if importlib.util.find_spec("cv2") is None or not REAL_VIDEO.exists():
        return None
    import cv2

    capture = cv2.VideoCapture(str(REAL_VIDEO))
    if not capture.isOpened():
        return None
    frames: list[np.ndarray] = []
    try:
        while len(frames) < CLIP_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (CLIP_W, CLIP_H), interpolation=cv2.INTER_AREA)
            # cv2 decodes BGR; the model expects RGB. Getting this backwards is
            # a silent quality loss that no shape check would catch.
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
    finally:
        capture.release()

    if len(frames) < CLIP_FRAMES:
        return None
    return np.stack(frames, axis=0)[None, ...].astype(np.uint8)


def build_clips() -> dict[str, np.ndarray]:
    clips: dict[str, np.ndarray] = {
        "synthetic_spatial_gradient": clip_spatial_gradient(),
        "synthetic_tubelet_probe": clip_tubelet_probe(),
        "synthetic_seeded_noise": clip_seeded_noise(),
    }
    real = clip_from_real_video()
    if real is not None:
        clips["real_test_video"] = real
    return clips


def write_clips(clips: dict[str, np.ndarray]) -> dict[str, str]:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    shas: dict[str, str] = {}
    for name, clip in clips.items():
        path = CLIP_DIR / f"{name}.npz"
        np.savez_compressed(path, clip=clip)
        shas[name] = sha256_array(clip)
        print(f"  {name:32} {clip.shape} uint8  {path.stat().st_size / 1024:7.1f} KiB")
    return shas


def load_clips() -> dict[str, np.ndarray]:
    """Load the committed clips. Used by both this script and the golden test."""
    if not CLIP_DIR.exists():
        return {}
    clips: dict[str, np.ndarray] = {}
    for path in sorted(CLIP_DIR.glob("*.npz")):
        with np.load(path) as data:
            clips[path.stem] = data["clip"]
    return clips


# ---------------------------------------------------------------------------
# Reference embeddings, from the official PyTorch path.
# ---------------------------------------------------------------------------


def missing_reference_prerequisites() -> list[str]:
    """Everything still needed to produce reference embeddings."""
    from src.config import IronConfig

    unmet: list[str] = []
    for module in ("torch", "transformers"):
        if importlib.util.find_spec(module) is None:
            unmet.append(f"module {module!r} is not installed")

    config = IronConfig.load()
    weights = config.paths.resolved_models_dir / "weights" / "vjepa2_vitl"
    if not (weights / "config.json").exists():
        unmet.append(
            f"official V-JEPA2 checkpoint not found at {weights} "
            "(needs model.safetensors and config.json)"
        )
    return unmet


def reference_spec() -> "Any":
    """The preprocessing the reference applies, read from the checkpoint.

    Not transcribed and not defaulted. This is the whole point of the exercise:
    the reference must be produced by the preprocessing the model was trained
    with, so that any gap against the production path is attributable to the
    production path rather than to a second guess.
    """
    from src.config import IronConfig
    from src.models.preprocess import PreprocessSpec

    config = IronConfig.load()
    checkpoint = config.paths.resolved_models_dir / "weights" / "vjepa2_vitl"

    # Prefer the spec written beside the exported artifact, which was generated
    # from this same checkpoint and pins the export geometry.
    sidecar = PreprocessSpec.sidecar_path_for(config.paths.resolved_current_ir)
    if sidecar.exists():
        return PreprocessSpec.load(sidecar)

    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from export_vjepa_ov import spec_from_checkpoint

    return spec_from_checkpoint(checkpoint, CLIP_FRAMES, CLIP_H)


def generate_references(clips: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run the official PyTorch V-JEPA2 path over each clip.

    The reference applies the checkpoint's own channel standardisation. The
    production OpenVINO path applies none, which is the defect these vectors
    exist to measure — so this function must not be "fixed" to match production.

    Raises:
        RuntimeError: if any embedding contains a non-finite value. One NaN
            vector corrupts an entire FAISS index, and a reference containing
            one would enshrine the corruption.
    """
    import torch
    from transformers import AutoModel

    from src.config import IronConfig

    config = IronConfig.load()
    checkpoint = config.paths.resolved_models_dir / "weights" / "vjepa2_vitl"
    spec = reference_spec()
    print(f"  preprocessing: {spec.describe()}")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    # eager attention to match the exported graph; see export_vjepa_ov.py.
    model = AutoModel.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    references: dict[str, Any] = {}
    for name, clip in sorted(clips.items()):
        # uint8 [0,255] -> [0,1] -> standardised. Both steps, in that order.
        scaled = clip.astype(np.float32) / 255.0
        standardised = spec.apply(scaled)

        with torch.no_grad():
            output = model.get_vision_features(torch.from_numpy(standardised))
        embeddings = np.asarray(output.numpy(), dtype=np.float32)

        if not np.all(np.isfinite(embeddings)):
            raise RuntimeError(
                f"reference embedding for {name!r} contains non-finite values; "
                "refusing to write it — one NaN vector corrupts every index "
                "built from this reference"
            )

        path = REFERENCE_DIR / f"{name}.npz"
        np.savez_compressed(path, embeddings=embeddings)
        references[name] = {
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "sha256": sha256_array(embeddings),
            "preprocess_sha": spec.preprocess_sha(),
        }
        print(f"  {name:32} -> {embeddings.shape}")
    return references


def write_manifest(
    clip_shas: dict[str, str],
    references: dict[str, Any] | None,
    reference_blockers: list[str],
) -> None:
    """Record what produced these fixtures, and what is still missing."""
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/make_golden_vectors.py",
        "seed": SEED,
        "clip_geometry": {
            "frames": CLIP_FRAMES,
            "height": CLIP_H,
            "width": CLIP_W,
            "channels": 3,
            "dtype": "uint8",
            "layout": "[B, T, C, H, W]",
            "value_range": "[0, 255]; divide by 255 before the preprocessor",
        },
        "real_clip_source": {
            "path": str(REAL_VIDEO.relative_to(REPO_ROOT)),
            "resize_policy": "cv2.INTER_AREA stretch to 224x224 (aspect NOT preserved)",
            "channel_order": "RGB (converted from OpenCV BGR at read time)",
        },
        "clip_sha256": clip_shas,
        "references": references or {},
        "reference_blockers": reference_blockers,
    }
    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips-only",
        action="store_true",
        help="regenerate the input clips only; needs no weights",
    )
    args = parser.parse_args(argv)

    print("Building golden input clips...")
    clips = build_clips()
    clip_shas = write_clips(clips)
    if "real_test_video" not in clips:
        print("  (real video clip unavailable; fixture set is synthetic only)")

    if args.clips_only:
        write_manifest(clip_shas, None, ["--clips-only requested"])
        print(f"\nWrote {len(clips)} clips and {MANIFEST_PATH.name} (clips only).")
        return 0

    blockers = missing_reference_prerequisites()
    if blockers:
        write_manifest(clip_shas, None, blockers)
        print("\nCannot generate reference embeddings:", file=sys.stderr)
        for blocker in blockers:
            print(f"  - {blocker}", file=sys.stderr)
        print(
            "\nInput clips and the manifest were still written. Rerun without "
            "--clips-only once the prerequisites above are met.",
            file=sys.stderr,
        )
        return 2

    print("\nGenerating reference embeddings via the official PyTorch path...")
    references = generate_references(clips)
    write_manifest(clip_shas, references, [])
    print(f"\nWrote {len(references)} references and {MANIFEST_PATH.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
