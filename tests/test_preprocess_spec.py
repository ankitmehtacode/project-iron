"""Tests for PreprocessSpec and the derived-artifact compatibility guard.

Two things are under test: that preprocessing values live with the model rather
than in Python, and that an artifact built under one spec cannot be read under
another without an explicit, loud refusal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.artifacts import (
    ArtifactMetadata,
    ArtifactMismatch,
    read_metadata,
    require_compatible,
    sidecar_path_for,
    write_metadata,
)
from src.config import IronConfig
from src.models.preprocess import (
    CANONICAL_SPEC_DIR,
    PreprocessError,
    PreprocessSpec,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

ENCODER = "e" * 64
MANIFEST = "m" * 64


def make_spec(**overrides: object) -> PreprocessSpec:
    defaults: dict[str, object] = {
        "frames": 4,
        "stride": 1,
        "resolution": (224, 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "channel_order": "RGB",
        "resize_policy": "stretch_to_square",
        "tubelet": 2,
        "patch_size": 16,
    }
    defaults.update(overrides)
    return PreprocessSpec(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The constants live with the model, not in Python
# ---------------------------------------------------------------------------

# ImageNet statistics, the values most likely to be pasted into a wrapper.
_FORBIDDEN_LITERALS = (
    r"0\.485",
    r"0\.456",
    r"0\.406",
    r"0\.229",
    r"0\.224",
    r"0\.225",
)

# Files allowed to contain them: the canonical spec (which is data), and the
# tests and report that discuss the rule.
_ALLOWED = {
    "configs/preprocess/vjepa2_vitl.preprocess.json",
    "tests/test_preprocess_spec.py",
    "tests/test_known_bugs.py",
    "FOUNDATION_REPORT.md",
}


def test_no_hardcoded_normalization_constants_in_source() -> None:
    """Preprocessing constants must not appear in Python source.

    Hardcoding them is how the pipeline ended up feeding a 64-frame model four
    frames, and how the OpenVINO path came to apply no standardisation while an
    unused wrapper applied the right one. The values belong beside the weights
    they describe, where they can be checked against the model.
    """
    offenders: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith((".venv/", "build/", "dist/")):
            continue
        if relative in _ALLOWED:
            continue
        text = path.read_text()
        for number, line in enumerate(text.splitlines(), start=1):
            if any(re.search(pattern, line) for pattern in _FORBIDDEN_LITERALS):
                offenders.append(f"{relative}:{number}: {line.strip()}")

    assert not offenders, (
        "normalization constants found in Python source; they belong in a "
        "PreprocessSpec beside the model:\n  " + "\n  ".join(offenders)
    )


def test_canonical_template_exists_and_parses() -> None:
    path = CANONICAL_SPEC_DIR / "vjepa2_vitl.preprocess.json"
    assert path.exists(), f"canonical template missing at {path}"
    spec = PreprocessSpec.from_dict(json.loads(path.read_text()))
    assert spec.channel_order == "RGB"
    assert spec.tubelet == 2


def test_canonical_template_is_marked_unverified() -> None:
    """It was transcribed without the checkpoint present, so it says so.

    A spec asserting values nobody checked against the model is a guess wearing
    a manifest. When someone confirms them with the weights in hand, they flip
    the flag — and this test changes with it, deliberately.
    """
    path = CANONICAL_SPEC_DIR / "vjepa2_vitl.preprocess.json"
    spec = PreprocessSpec.from_dict(json.loads(path.read_text()))
    assert spec.verified_against_official_config is False
    with pytest.raises(PreprocessError, match="unverified"):
        spec.assert_ready_for_production()


# ---------------------------------------------------------------------------
# Spec behaviour
# ---------------------------------------------------------------------------


def test_apply_standardises() -> None:
    spec = make_spec()
    clip = np.full((1, 4, 3, 224, 224), 0.5, dtype=np.float32)
    out = spec.apply(clip)
    for channel in range(3):
        expected = (0.5 - spec.mean[channel]) / spec.std[channel]
        assert out[0, 0, channel].mean() == pytest.approx(expected, rel=1e-5)


def test_apply_changes_the_input() -> None:
    """The transform must not be a no-op; a no-op passes a source grep."""
    spec = make_spec()
    ones = np.ones((1, 4, 3, 224, 224), dtype=np.float32)
    assert not np.allclose(spec.apply(ones), ones)


def test_apply_rejects_unscaled_input() -> None:
    """0-255 data here would make activations ~255x too large, silently."""
    spec = make_spec()
    clip = np.full((1, 4, 3, 224, 224), 128.0, dtype=np.float32)
    with pytest.raises(PreprocessError, match=r"must be in \[0, 1\]"):
        spec.apply(clip)


def test_apply_rejects_wrong_resolution() -> None:
    spec = make_spec()
    with pytest.raises(PreprocessError, match="describes 224x224"):
        spec.apply(np.zeros((1, 4, 3, 128, 128), dtype=np.float32))


def test_apply_rejects_wrong_frame_count() -> None:
    spec = make_spec()
    with pytest.raises(PreprocessError, match="frames"):
        spec.apply(np.zeros((1, 8, 3, 224, 224), dtype=np.float32))


def test_apply_rejects_non_finite() -> None:
    spec = make_spec()
    clip = np.zeros((1, 4, 3, 224, 224), dtype=np.float32)
    clip[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(PreprocessError, match="non-finite"):
        spec.apply(clip)


def test_zero_std_is_rejected() -> None:
    with pytest.raises(PreprocessError, match="std must be positive"):
        make_spec(std=(0.229, 0.0, 0.225))


def test_frames_must_be_a_multiple_of_tubelet() -> None:
    """Otherwise the encoder silently drops the tail of every clip."""
    with pytest.raises(PreprocessError, match="not a multiple of tubelet"):
        make_spec(frames=5, tubelet=2)


def test_unknown_channel_order_rejected() -> None:
    with pytest.raises(PreprocessError, match="channel_order"):
        make_spec(channel_order="RBG")


# ---------------------------------------------------------------------------
# preprocess_sha identity
# ---------------------------------------------------------------------------


def test_sha_is_stable() -> None:
    assert make_spec().preprocess_sha() == make_spec().preprocess_sha()


@pytest.mark.parametrize(
    "override",
    [
        {"mean": (0.5, 0.456, 0.406)},
        {"std": (0.2, 0.224, 0.225)},
        {"resolution": (256, 256)},
        {"frames": 8},
        {"tubelet": 1},
        {"channel_order": "BGR"},
        {"patch_size": 14},
        {"stride": 2},
    ],
)
def test_sha_changes_when_the_transform_changes(override: dict[str, object]) -> None:
    """Any field that alters encoder input must alter the hash."""
    assert make_spec(**override).preprocess_sha() != make_spec().preprocess_sha()


def test_sha_ignores_documentation_fields() -> None:
    """Who wrote the spec down does not change what it computes."""
    a = make_spec(source="from the paper")
    b = make_spec(source="from the config", verified_against_official_config=True)
    assert a.preprocess_sha() == b.preprocess_sha()


def test_round_trip_through_json(tmp_path: Path) -> None:
    spec = make_spec()
    path = spec.write(tmp_path / "m.preprocess.json")
    assert PreprocessSpec.load(path) == spec


def test_sidecar_path_sits_beside_the_model() -> None:
    assert PreprocessSpec.sidecar_path_for(
        Path("/models/int8/vjepa2_vitl_int8.xml")
    ) == Path("/models/int8/vjepa2_vitl_int8.preprocess.json")


def test_missing_sidecar_refuses_rather_than_falling_back(tmp_path: Path) -> None:
    """A silent fallback would apply another model's preprocessing.

    That produces embeddings wrong in a way no shape or finiteness check
    detects, which is the whole failure this module exists to prevent.
    """
    with pytest.raises(PreprocessError, match="Refusing to guess"):
        PreprocessSpec.load_for_model(tmp_path / "some_model.xml")


# ---------------------------------------------------------------------------
# Agreement with the pipeline config
# ---------------------------------------------------------------------------


def test_spec_matches_the_current_pipeline_config() -> None:
    """The canonical template and configs/default.yaml must not disagree."""
    spec = PreprocessSpec.from_dict(
        json.loads((CANONICAL_SPEC_DIR / "vjepa2_vitl.preprocess.json").read_text())
    )
    spec.assert_matches_pipeline(IronConfig.load().pipeline)


def test_mismatched_pipeline_is_rejected() -> None:
    spec = make_spec(frames=16, resolution=(256, 256))
    with pytest.raises(PreprocessError, match="disagrees with the model"):
        spec.assert_matches_pipeline(IronConfig.load().pipeline)


# ---------------------------------------------------------------------------
# Artifact coupling
# ---------------------------------------------------------------------------


def artifact_metadata(**overrides: object) -> ArtifactMetadata:
    defaults: dict[str, object] = {
        "kind": "faiss_index",
        "encoder_sha": ENCODER,
        "preprocess_sha": make_spec().preprocess_sha(),
        "manifest_sha": MANIFEST,
        "vector_count": 100,
    }
    defaults.update(overrides)
    return ArtifactMetadata(**defaults)  # type: ignore[arg-type]


def test_metadata_round_trips(tmp_path: Path) -> None:
    artifact = tmp_path / "index.faiss"
    artifact.write_bytes(b"not really an index")
    write_metadata(artifact, artifact_metadata())
    assert read_metadata(artifact).encoder_sha == ENCODER


def test_sidecar_sits_beside_the_artifact() -> None:
    assert sidecar_path_for(Path("/out/index.faiss")) == Path(
        "/out/index.faiss.meta.json"
    )


def test_compatible_artifact_loads(tmp_path: Path) -> None:
    artifact = tmp_path / "index.faiss"
    artifact.write_bytes(b"x")
    spec = make_spec()
    write_metadata(artifact, artifact_metadata())
    metadata = require_compatible(
        artifact, encoder_sha=ENCODER, preprocess_sha=spec.preprocess_sha()
    )
    assert metadata.vector_count == 100


def test_preprocess_change_refuses_the_artifact(tmp_path: Path) -> None:
    """The core coupling rule: new preprocessing voids old vectors."""
    artifact = tmp_path / "index.faiss"
    artifact.write_bytes(b"x")
    write_metadata(artifact, artifact_metadata())

    changed = make_spec(mean=(0.5, 0.5, 0.5))
    with pytest.raises(ArtifactMismatch, match="preprocess_sha") as excinfo:
        require_compatible(
            artifact, encoder_sha=ENCODER, preprocess_sha=changed.preprocess_sha()
        )
    assert "scripts/rebuild_index.py" in str(excinfo.value)


def test_encoder_change_refuses_the_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "index.faiss"
    artifact.write_bytes(b"x")
    write_metadata(artifact, artifact_metadata())
    with pytest.raises(ArtifactMismatch, match="encoder_sha"):
        require_compatible(
            artifact,
            encoder_sha="f" * 64,
            preprocess_sha=make_spec().preprocess_sha(),
        )


def test_missing_sidecar_is_a_refusal_not_a_pass(tmp_path: Path) -> None:
    """Pre-day-2 artifacts are void; absence of metadata must not read as OK.

    An artifact with no sidecar was written by the pipeline that applied no
    channel standardisation. Treating absence as "probably fine" would readmit
    exactly the data this guard exists to exclude.
    """
    artifact = tmp_path / "legacy.parquet"
    artifact.write_bytes(b"x")
    with pytest.raises(ArtifactMismatch, match="void") as excinfo:
        require_compatible(
            artifact,
            encoder_sha=ENCODER,
            preprocess_sha=make_spec().preprocess_sha(),
        )
    assert "rebuild_index.py" in str(excinfo.value)


def test_metadata_requires_its_producer_shas() -> None:
    with pytest.raises(ValueError, match="preprocess_sha is required"):
        artifact_metadata(preprocess_sha="")


def test_foreign_metadata_schema_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "index.faiss"
    artifact.write_bytes(b"x")
    path = sidecar_path_for(artifact)
    path.write_text(json.dumps({"schema_version": "9.9"}))
    with pytest.raises(ArtifactMismatch, match="schema_version"):
        read_metadata(artifact)


# ---------------------------------------------------------------------------
# Manifest integration
# ---------------------------------------------------------------------------


def test_manifest_records_preprocess_sha() -> None:
    from src.provenance import RunManifest

    spec = make_spec()
    manifest = RunManifest.capture(
        IronConfig.load(), model_paths=[], preprocess_sha=spec.preprocess_sha()
    )
    assert manifest.preprocess_sha == spec.preprocess_sha()
    assert spec.preprocess_sha()[:12] in manifest.summary()


def test_manifest_sha_changes_with_preprocessing() -> None:
    """Two runs under different preprocessing are not comparable, and say so."""
    from src.provenance import RunManifest

    config = IronConfig.load()
    a = RunManifest.capture(config, model_paths=[], preprocess_sha="a" * 64)
    b = RunManifest.capture(config, model_paths=[], preprocess_sha="b" * 64)
    assert a.manifest_sha != b.manifest_sha


def test_manifest_states_when_no_spec_is_wired_in() -> None:
    """The current production state must be visible, not blank."""
    from src.provenance import RunManifest

    manifest = RunManifest.capture(IronConfig.load(), model_paths=[])
    assert manifest.preprocess_sha is None
    assert "NONE" in manifest.summary()
