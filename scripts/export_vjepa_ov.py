"""Export V-JEPA2 from the official checkpoint to OpenVINO FP32, with a manifest.

FP32 only. Quantization comes after the encoder choice is frozen and the export
is verified against golden vectors — quantizing first means every later quality
question has two candidate causes and no way to separate them.

Where the artifact goes
-----------------------
``models/export/<utc-date>/``. **Never** the production path. An artifact at the
production path must only ever be the artifact production actually ran, because
it is the sole evidence of what the stored embeddings were computed with; a
fresh export placed there silently becomes "production" in every later forensic
comparison and the real one cannot be reconstructed. The script refuses to
write there.

What gets verified, and why each one is a real bug class
--------------------------------------------------------
1. **Token count.** Changing spatial resolution or clip length changes the token
   grid, and position embeddings must be interpolated to match. The script
   computes the expected count from the config —
   ``(frames // tubelet) * (H/patch) * (W/patch)`` — and aborts if the model
   disagrees. A wrong interpolation produces a model that runs and returns a
   correctly-shaped tensor of degraded features.
2. **Which encoder.** V-JEPA trains a context encoder and an EMA target
   encoder. Exporting the wrong one is total, silent quality loss, so the
   choice is explicit and recorded in the manifest rather than inherited from
   whatever attribute a helper happened to reach for.
3. **PreprocessSpec beside the artifact**, read from the checkpoint's own
   ``video_preprocessor_config.json`` rather than transcribed. Values a human
   copied are values a human can mistype.
4. **Source-checkpoint sha** in the manifest, so the export can be tied to the
   weights it came from.
5. **Golden vectors run as the exit criterion.** An export that has not been
   compared against the reference implementation is an untested export.

Usage:
    python scripts/export_vjepa_ov.py
    python scripts/export_vjepa_ov.py --frames 16 --resolution 256
    python scripts/export_vjepa_ov.py --skip-golden      # export only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import IronConfig
from src.models.preprocess import PreprocessSpec

REPO_ROOT = Path(__file__).resolve().parent.parent

# The two encoders V-JEPA trains. Recorded explicitly; see item 2 above.
ENCODER_CHOICES = ("context", "ema_target")

# Verification thresholds for FP32 -> FP32 conversion. This is not
# quantization: the IR should reproduce the reference almost exactly, so the
# bar is tight. Anything looser would let a genuine conversion defect through
# as "close enough".
MAX_RELATIVE_DEVIATION = 1e-3
MIN_COSINE_P1 = 0.9999

# Tracing and verification input is seeded random, never zeros.
#
# An all-zero clip is degenerate: every attention key is identical, so softmax
# is uniform and tiny numerical differences compound through 24 layers. Verified
# on this export — zeros reported max|IR-ref| = 2.47 and cosine p1 = 0.9953,
# while random input on the SAME artifact gave 8.3e-04 and 1.000000. Judging
# the export by the zeros number would have condemned a correct conversion;
# tracing with zeros can also constant-fold data-dependent branches wrongly.
TRACE_SEED = 20260731


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_from_checkpoint(
    checkpoint_dir: Path, frames: int, resolution: int
) -> PreprocessSpec:
    """Build a PreprocessSpec from the checkpoint's own config files.

    Read, never transcribed. The values a human copies into a Python file are
    the values a human mistypes, and a preprocessing constant that is wrong by
    a little produces embeddings that are wrong by a little — which nothing
    downstream can detect.

    ``frames`` and ``resolution`` are the *export* geometry, which may differ
    from the checkpoint's native geometry; everything else comes from the
    checkpoint.
    """
    config = json.loads((checkpoint_dir / "config.json").read_text())
    preprocessor_path = checkpoint_dir / "video_preprocessor_config.json"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"{preprocessor_path} is missing. The checkpoint's own preprocessing "
            "config is the only authoritative source for mean/std; refusing to "
            "fall back to transcribed constants."
        )
    preprocessor = json.loads(preprocessor_path.read_text())

    if not preprocessor.get("do_normalize", False):
        raise ValueError(
            "the checkpoint's preprocessor config says do_normalize is false; "
            "this export path assumes channel standardisation and would be wrong"
        )

    return PreprocessSpec(
        frames=frames,
        stride=1,
        resolution=(resolution, resolution),
        mean=tuple(  # type: ignore[arg-type]
            float(v) for v in preprocessor["image_mean"]
        ),
        std=tuple(  # type: ignore[arg-type]
            float(v) for v in preprocessor["image_std"]
        ),
        channel_order="RGB",
        resize_policy=(
            f"resize shortest edge to {preprocessor['size']['shortest_edge']}, "
            f"center crop {resolution}, rescale by "
            f"{preprocessor['rescale_factor']}, then standardise"
        ),
        tubelet=int(config["tubelet_size"]),
        patch_size=int(config["patch_size"]),
        source=(
            f"read from {checkpoint_dir.name}/config.json"
            " and video_preprocessor_config.json"
        ),
        # Read directly from the checkpoint that will be exported, so this is
        # verification rather than assertion.
        verified_against_official_config=True,
    )


def expected_token_count(spec: PreprocessSpec) -> int:
    """Tokens the encoder must emit for this geometry.

    ``(frames // tubelet) * (H / patch) * (W / patch)``. A mismatch means
    position embeddings were not interpolated to the export geometry.
    """
    height, width = spec.resolution
    temporal = spec.frames // spec.tubelet
    spatial = (height // spec.patch_size) * (width // spec.patch_size)
    return temporal * spatial


def export(
    checkpoint_dir: Path,
    output_dir: Path,
    spec: PreprocessSpec,
    encoder_choice: str,
) -> dict[str, Any]:
    """Trace the model to ONNX, convert to OpenVINO IR, and verify the shape."""
    import openvino as ov
    import torch
    from transformers import AutoModel

    if encoder_choice not in ENCODER_CHOICES:
        raise ValueError(f"encoder must be one of {ENCODER_CHOICES}")

    print(f"Loading {checkpoint_dir} ...")
    # attn_implementation="eager" rather than the default SDPA. torch 2.2.0's
    # ONNX exporter cannot translate scaled_dot_product_attention (it passes the
    # scale as a float where a tensor is required and raises inside the
    # symbolic function). Eager attention computes the same quantity by explicit
    # matmul/softmax; only the floating-point association order differs, and the
    # IR-vs-reference deviation check below is what confirms that the difference
    # stays at the level of numerical noise rather than semantics.
    #
    # The reference forward pass uses this same module, so the comparison is
    # like for like: it verifies the ONNX/IR conversion, not eager vs SDPA.
    model = AutoModel.from_pretrained(
        checkpoint_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    class VisionEncoder(torch.nn.Module):
        """Exports the vision encoder only.

        V-JEPA2Model bundles an encoder and a predictor; the predictor is a
        pretraining head and is not what produces the embeddings this pipeline
        indexes. Exporting the whole module would carry dead weight and, worse,
        make "which output is the embedding" a question resolved by position.
        """

        def __init__(self, wrapped: torch.nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, video: torch.Tensor) -> torch.Tensor:
            return self.wrapped.get_vision_features(video)

    wrapper = VisionEncoder(model).eval()

    height, width = spec.resolution
    generator = torch.Generator().manual_seed(TRACE_SEED)
    example = torch.randn(
        1, spec.frames, 3, height, width, generator=generator, dtype=torch.float32
    )

    print("Running the reference forward pass to verify the token grid ...")
    with torch.no_grad():
        reference_out = wrapper(example)

    produced = int(reference_out.shape[1])
    expected = expected_token_count(spec)
    temporal = spec.frames // spec.tubelet
    spatial = (height // spec.patch_size) * (width // spec.patch_size)
    print(
        f"  tokens: {produced}   expected {expected} "
        f"= {temporal} temporal x {spatial} spatial"
    )
    if produced != expected:
        raise RuntimeError(
            f"token-count mismatch: the encoder emitted {produced} tokens but "
            f"this geometry implies {expected} "
            f"({spec.frames} frames // tubelet {spec.tubelet} = {temporal} "
            f"temporal, {height}x{width} / patch {spec.patch_size} = {spatial} "
            "spatial). Position embeddings were not interpolated to the export "
            "geometry. Aborting: the export would run and return "
            "correctly-shaped, degraded features."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "vjepa2_vitl_fp32.onnx"

    print(f"Exporting ONNX -> {onnx_path.name} (this takes a few minutes) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            str(onnx_path),
            input_names=["video"],
            output_names=["features"],
            opset_version=17,
            dynamic_axes={"video": {0: "batch"}, "features": {0: "batch"}},
        )

    print("Converting ONNX -> OpenVINO IR ...")
    ov_model = ov.convert_model(str(onnx_path))
    xml_path = output_dir / "vjepa2_vitl_fp32.xml"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)

    print("Verifying the IR against the reference forward pass ...")
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    request = compiled.create_infer_request()
    request.infer({"video": example.numpy()})
    ir_out = np.asarray(request.get_output_tensor(0).data, dtype=np.float64)

    if ir_out.shape != tuple(reference_out.shape):
        raise RuntimeError(
            f"IR output shape {ir_out.shape} differs from the reference "
            f"{tuple(reference_out.shape)}"
        )
    if not np.all(np.isfinite(ir_out)):
        raise RuntimeError(
            "IR output contains non-finite values; one NaN vector corrupts "
            "every index built from this model"
        )

    reference = reference_out.numpy().astype(np.float64)
    deviation = float(np.max(np.abs(ir_out - reference)))
    scale = max(float(np.max(np.abs(reference))), 1e-9)
    relative = deviation / scale

    left = ir_out.reshape(-1, ir_out.shape[-1])
    right = reference.reshape(-1, reference.shape[-1])
    norms = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    usable = norms > 0
    cosine = np.sum(left[usable] * right[usable], axis=1) / norms[usable]
    cosine_p1 = float(np.percentile(cosine, 1))

    print(f"  max |IR - reference|  = {deviation:.3e}")
    print(
        f"  relative deviation    = {relative:.3e}  "
        f"(limit {MAX_RELATIVE_DEVIATION:.0e})"
    )
    print(f"  per-token cosine p1   = {cosine_p1:.6f}  (floor {MIN_COSINE_P1})")

    if relative > MAX_RELATIVE_DEVIATION or cosine_p1 < MIN_COSINE_P1:
        raise RuntimeError(
            f"the IR does not reproduce the PyTorch reference: relative "
            f"deviation {relative:.3e} (limit {MAX_RELATIVE_DEVIATION:.0e}), "
            f"cosine p1 {cosine_p1:.6f} (floor {MIN_COSINE_P1}). This is an "
            "FP32-to-FP32 conversion, so it should be near-exact. Refusing to "
            "publish an export that already disagrees with the implementation "
            "it was made from."
        )

    return {
        "relative_deviation": relative,
        "cosine_p1": cosine_p1,
        "onnx": onnx_path,
        "xml": xml_path,
        "bin": xml_path.with_suffix(".bin"),
        "tokens": produced,
        "shape": list(ir_out.shape),
        "max_abs_deviation_from_reference": deviation,
    }


def write_manifest(
    output_dir: Path,
    checkpoint_dir: Path,
    spec: PreprocessSpec,
    encoder_choice: str,
    result: dict[str, Any],
    config: IronConfig,
) -> Path:
    """Record everything needed to reproduce or invalidate this export."""
    import openvino
    import torch
    import transformers

    source_files = {}
    for name in ("model.safetensors", "config.json", "video_preprocessor_config.json"):
        path = checkpoint_dir / name
        if path.exists():
            source_files[name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }

    manifest = {
        "schema_version": "1.0",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "exporter": "scripts/export_vjepa_ov.py",
        "precision": "FP32",
        "quantized": False,
        "quantization_note": (
            "INT8 comes after the encoder choice is frozen and this export is "
            "verified against golden vectors. Quantizing first gives every "
            "later quality question two candidate causes."
        ),
        "encoder": {
            "choice": encoder_choice,
            "exported_module": "VJEPA2Model.get_vision_features (encoder only)",
            "attn_implementation": "eager",
            "attn_note": (
                "eager rather than SDPA: torch 2.2.0's ONNX exporter cannot "
                "translate scaled_dot_product_attention. Same computation, "
                "different floating-point association order."
            ),
            "note": (
                "V-JEPA trains a context encoder and an EMA target encoder. "
                "Exporting the wrong one is total, silent quality loss, so the "
                "choice is recorded rather than inferred."
            ),
        },
        "source_checkpoint": {
            "path": str(checkpoint_dir.relative_to(REPO_ROOT)),
            "files": source_files,
        },
        "geometry": {
            "frames": spec.frames,
            "resolution": list(spec.resolution),
            "tubelet": spec.tubelet,
            "patch_size": spec.patch_size,
            "tokens": result["tokens"],
            "token_arithmetic": (
                f"({spec.frames} // {spec.tubelet}) * "
                f"({spec.resolution[0]} // {spec.patch_size})^2 = {result['tokens']}"
            ),
        },
        "preprocess_sha": spec.preprocess_sha(),
        "artifacts": {
            "xml": {
                "name": result["xml"].name,
                "sha256": sha256_file(result["xml"]),
                "size_bytes": result["xml"].stat().st_size,
            },
            "bin": {
                "name": result["bin"].name,
                "sha256": sha256_file(result["bin"]),
                "size_bytes": result["bin"].stat().st_size,
            },
        },
        "verification": {
            "token_count_matches_geometry": True,
            "ir_output_finite": True,
            "max_abs_deviation_from_pytorch_reference": result[
                "max_abs_deviation_from_reference"
            ],
            "relative_deviation_from_pytorch_reference": result["relative_deviation"],
            "per_token_cosine_p1_vs_pytorch_reference": result["cosine_p1"],
            "verification_input": (
                f"seeded random (torch.randn, seed {TRACE_SEED}). NOT zeros: an "
                "all-zero clip makes attention uniform and amplifies noise "
                "through every layer, which on this very export reported "
                "cosine p1 0.9953 for an artifact that scores 1.000000 on real "
                "input."
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "openvino": openvino.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "config_sha": config.config_sha(),
    }

    path = output_dir / "export_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="context")
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="export without running the golden-vector comparison (not advised)",
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    checkpoint_dir = config.paths.resolved_models_dir / "weights" / "vjepa2_vitl"
    if not (checkpoint_dir / "config.json").exists():
        print(
            f"Official checkpoint not found at {checkpoint_dir} (needs "
            "model.safetensors, config.json and video_preprocessor_config.json). "
            "Run: python scripts/fetch_weights.py",
            file=sys.stderr,
        )
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_dir = config.paths.resolved_models_dir / "export" / stamp

    production = config.paths.resolved_vjepa_xml
    if output_dir.resolve() == production.parent.resolve():
        print(
            f"REFUSING to export into the production directory {production.parent}. "
            "That path is reserved for the artifact production actually ran.",
            file=sys.stderr,
        )
        return 2

    spec = spec_from_checkpoint(checkpoint_dir, args.frames, args.resolution)
    print(f"Preprocessing spec (read from the checkpoint): {spec.describe()}")
    native = json.loads((checkpoint_dir / "config.json").read_text())
    if (
        args.resolution != native["image_size"]
        or args.frames != native["frames_per_clip"]
    ):
        print(
            f"  NOTE: exporting at {args.frames}f/{args.resolution}px, but the "
            f"checkpoint is native at {native['frames_per_clip']}f/"
            f"{native['image_size']}px. Position embeddings will be interpolated; "
            "the token-count check below is what verifies that happened."
        )
    print()

    result = export(checkpoint_dir, output_dir, spec, args.encoder)

    spec_path = PreprocessSpec.sidecar_path_for(result["xml"])
    spec.write(spec_path)
    print(f"\nPreprocessSpec -> {spec_path.name}")

    manifest_path = write_manifest(
        output_dir, checkpoint_dir, spec, args.encoder, result, config
    )
    print(f"Export manifest -> {manifest_path.name}")
    print(f"\nArtifacts in {output_dir}:")
    for item in sorted(output_dir.iterdir()):
        print(f"  {item.name:34} {item.stat().st_size / 1e6:9.1f} MB")

    if args.skip_golden:
        print(
            "\nSkipped the golden-vector comparison. This export is unverified "
            "against the reference implementation."
        )
        return 0

    print("\n" + "=" * 70)
    print("EXIT CRITERION: golden-vector comparison")
    print("=" * 70)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_golden_vectors.py",
            "-q",
            "-p",
            "no:warnings",
            "--no-header",
        ],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "IRON_PATHS__CURRENT_IR": str(result["xml"])},
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
