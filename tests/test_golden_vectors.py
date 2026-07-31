"""Golden-vector test: the OpenVINO path against the official PyTorch reference.

This is the single test that catches the silent export bug classes — position
embeddings not interpolated after a resolution change, the wrong encoder
exported, preprocessing never wired in. All of them produce a model that runs,
returns plausibly-shaped tensors, and passes every smoke test.

Gated on the **1st percentile** of per-patch cosine, not the mean. A mean over
196 patches stays comfortable while the worst 2% of patches are unusable, and
the worst patches are where objects are. A quality floor expressed as a mean is
not a floor.

This test was RED until 2026-07-31: the OpenVINO path applied no channel
standardisation while the reference applied the official one, scoring a
per-patch cosine p1 of 0.332. With ``_run_vjepa`` applying the model's
PreprocessSpec it scores 0.999987, and the ``known_bug`` marker was removed in
the same change that fixed it — so this now guards the behaviour permanently.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.config import IronConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_ROOT = REPO_ROOT / "tests" / "golden"
CLIP_DIR = GOLDEN_ROOT / "clips"
REFERENCE_DIR = GOLDEN_ROOT / "reference"
MANIFEST_PATH = GOLDEN_ROOT / "manifest.json"

# Floor for reference-vs-OpenVINO agreement once preprocessing is correct.
#
# Measured on the 2026-07-31 FP32 export: p1 = 0.999987 across all four clips.
# The floor sits at 0.98 to leave room for the INT8 quantization that comes
# later without being so loose that a real export defect passes — before the
# normalization fix this same comparison scored 0.332.
GOLDEN_COSINE_P1_MIN = 0.98


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        pytest.skip(
            f"{MANIFEST_PATH} is missing. Generate the fixtures with "
            "python scripts/make_golden_vectors.py --clips-only"
        )
    return json.loads(MANIFEST_PATH.read_text())


def load_clip(name: str) -> np.ndarray:
    path = CLIP_DIR / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"golden clip {name!r} not found at {path}")
    with np.load(path) as data:
        return np.asarray(data["clip"])


def clip_names() -> list[str]:
    if not CLIP_DIR.exists():
        return []
    return sorted(path.stem for path in CLIP_DIR.glob("*.npz"))


def per_patch_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per token between two ``[..., n_tokens, dim]`` blocks."""
    left = a.reshape(-1, a.shape[-1]).astype(np.float64)
    right = b.reshape(-1, b.shape[-1]).astype(np.float64)
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    usable = norms > 0
    similarity = np.zeros(left.shape[0], dtype=np.float64)
    similarity[usable] = np.sum(left[usable] * right[usable], axis=1) / norms[usable]
    return similarity


# ---------------------------------------------------------------------------
# Fixture integrity. Runs today, needs no weights.
# ---------------------------------------------------------------------------


def test_golden_clips_are_committed() -> None:
    """The inputs must be in the repository, not regenerated per run.

    A fixture regenerated on the fly is not a fixture: it drifts with whatever
    numpy or OpenCV version happens to be installed, and the reference it is
    compared against silently stops corresponding to it.
    """
    names = clip_names()
    assert names, "no golden clips committed under tests/golden/clips/"
    assert "synthetic_tubelet_probe" in names
    assert len(names) >= 3


@pytest.mark.parametrize("name", clip_names())
def test_clip_matches_its_recorded_sha(name: str) -> None:
    """A fixture that changed without its sha changing is not a fixture."""
    import hashlib

    manifest = load_manifest()
    clip = load_clip(name)
    digest = hashlib.sha256()
    digest.update(str(clip.shape).encode())
    digest.update(str(clip.dtype).encode())
    digest.update(np.ascontiguousarray(clip).tobytes())

    recorded = manifest["clip_sha256"].get(name)
    assert recorded is not None, f"{name} has no recorded sha in the manifest"
    assert digest.hexdigest() == recorded, (
        f"golden clip {name!r} does not match its manifest sha. Either the clip "
        "was edited without regenerating the manifest, or the generator is no "
        "longer deterministic — both invalidate every reference built from it."
    )


def test_clip_geometry_matches_the_pipeline_config() -> None:
    """Fixtures must describe the clip shape the pipeline actually uses.

    If the configured geometry changes, these fixtures stop being comparable
    and the references must be regenerated. Failing here is the signal to do
    that, rather than discovering it as an unexplained cosine drop.
    """
    manifest = load_manifest()
    pipeline = IronConfig.load().pipeline
    geometry = manifest["clip_geometry"]

    assert geometry["frames"] == pipeline.clip_frames
    assert geometry["height"] == pipeline.clip_h
    assert geometry["width"] == pipeline.clip_w
    assert geometry["channels"] == pipeline.clip_channels


def test_tubelet_probe_straddles_the_boundary() -> None:
    """The probe must actually be black-then-white, or it probes nothing."""
    clip = load_clip("synthetic_tubelet_probe")
    frames = clip.shape[1]
    first_half = clip[:, : frames // 2]
    second_half = clip[:, frames // 2 :]
    assert first_half.max() == 0, "first half of the tubelet probe is not black"
    assert second_half.min() == 255, "second half of the tubelet probe is not white"


# ---------------------------------------------------------------------------
# The golden comparison. Blocked on weights today.
# ---------------------------------------------------------------------------


def _require_openvino_and_reference(name: str) -> tuple[Any, np.ndarray]:
    """Return (openvino module, reference embeddings), or skip listing the gap."""
    config = IronConfig.load()
    unmet: list[str] = []

    if importlib.util.find_spec("openvino") is None:
        unmet.append("module 'openvino' is not installed")
    xml = config.paths.resolved_current_ir
    if not xml.exists():
        unmet.append(f"V-JEPA2 OpenVINO IR not found at {xml}")
    reference_path = REFERENCE_DIR / f"{name}.npz"
    if not reference_path.exists():
        unmet.append(
            f"reference embedding not found at {reference_path}; generate it "
            "with python scripts/make_golden_vectors.py"
        )

    if unmet:
        pytest.skip(f"{len(unmet)} unmet prerequisite(s): " + "; ".join(unmet))

    import openvino as ov

    with np.load(reference_path) as data:
        reference = np.asarray(data["embeddings"], dtype=np.float64)
    return ov, reference


@pytest.mark.requires_weights
@pytest.mark.parametrize("name", clip_names())
def test_openvino_matches_pytorch_reference(name: str) -> None:
    """Per-patch cosine between the production path and the official reference.

    The failure message reports the full distribution rather than a pass/fail,
    because the shape of the disagreement identifies the cause: a uniform
    moderate drop across all patches is a preprocessing problem, while a small
    number of catastrophic patches is a position-embedding or token-layout
    problem.
    """
    config = IronConfig.load()
    ov, reference = _require_openvino_and_reference(name)

    # Exactly what production does: scale to [0, 1], then apply the model's own
    # PreprocessSpec. Feeding raw [0, 1] here — which is what the pipeline did
    # until audit finding 3 was fixed — scores a per-patch cosine p1 of 0.332
    # against this same reference.
    from src.models.preprocess import PreprocessSpec

    spec = PreprocessSpec.load_for_model(config.paths.resolved_current_ir)
    clip = spec.apply(load_clip(name).astype(np.float32) / 255.0)

    core = ov.Core()
    compiled = core.compile_model(
        str(config.paths.resolved_current_ir),
        config.runtime.device,
        config.runtime.openvino_properties(),
    )
    request = compiled.create_infer_request()
    request.infer({"video": clip})
    produced = np.asarray(request.get_output_tensor(0).data, dtype=np.float64)

    assert produced.shape == reference.shape, (
        f"shape mismatch for {name!r}: OpenVINO produced {produced.shape}, "
        f"reference is {reference.shape}. This is a token-layout difference, "
        "not a numerical one — compare tubelet and patch grid before cosine."
    )
    assert np.all(np.isfinite(produced)), (
        f"OpenVINO output for {name!r} contains non-finite values; one NaN "
        "vector corrupts every index built from it"
    )

    cosine = per_patch_cosine(produced, reference)
    p1 = float(np.percentile(cosine, 1))
    report = (
        f"\n  clip            : {name}"
        f"\n  tokens compared : {cosine.size}"
        f"\n  cosine p1       : {p1:.6f}   <- gated (floor {GOLDEN_COSINE_P1_MIN})"
        f"\n  cosine p50      : {float(np.percentile(cosine, 50)):.6f}"
        f"\n  cosine mean     : {float(np.mean(cosine)):.6f}"
        f"\n  cosine min      : {float(np.min(cosine)):.6f}"
        f"\n  cosine max      : {float(np.max(cosine)):.6f}"
        f"\n  A uniform moderate drop across patches indicates preprocessing;"
        f"\n  a few catastrophic patches indicate token layout or pos-embeds."
    )
    assert p1 >= GOLDEN_COSINE_P1_MIN, report
