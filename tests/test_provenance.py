"""Tests for run manifests.

The properties that matter: a manifest must identify a *setup* stably enough
that two runs of it can be compared, it must notice a dirty working tree, and a
run that cannot write one must refuse to start rather than produce results
nobody can later attribute.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.config import IronConfig
from src.endurance.runner import ExitCode, execute
from src.provenance import (
    ManifestError,
    RunManifest,
    git_state,
    hash_model_files,
    library_versions,
    require_export_manifest,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A throwaway git checkout with one committed file.

    Real git rather than a mock: the thing under test is whether we read git's
    actual output correctly, and a mock would only assert that we agree with
    our own assumptions about its format.
    """
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git is not available")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.txt").write_text("original contents\n")

    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["add", "-A"],
        ["commit", "-q", "-m", "initial"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return repo


# ---------------------------------------------------------------------------
# Git state
# ---------------------------------------------------------------------------


def test_clean_repo_is_not_dirty(git_repo: Path) -> None:
    state = git_state(git_repo)
    assert state.available
    assert state.dirty is False
    assert state.sha is not None and len(state.sha) == 40
    assert state.branch == "main"


def test_dirty_flag_flips_when_a_tracked_file_is_touched(git_repo: Path) -> None:
    """Modifying a tracked file must be visible in the manifest.

    A run against an edited working tree cannot be reproduced from its commit,
    and the manifest is the only place that fact is recorded.
    """
    assert git_state(git_repo).dirty is False
    (git_repo / "tracked.txt").write_text("edited\n")
    assert git_state(git_repo).dirty is True


def test_dirty_flag_flips_for_an_untracked_file(git_repo: Path) -> None:
    """An untracked file can change behaviour just as much as an edit."""
    (git_repo / "stray.py").write_text("print('side effect')\n")
    assert git_state(git_repo).dirty is True


def test_sha_is_unchanged_by_a_working_tree_edit(git_repo: Path) -> None:
    before = git_state(git_repo).sha
    (git_repo / "tracked.txt").write_text("edited\n")
    assert git_state(git_repo).sha == before


def test_outside_a_repo_is_recorded_not_raised(tmp_path: Path) -> None:
    """A source tarball is a valid place to run; it just has no commit."""
    state = git_state(tmp_path)
    assert state.available is False
    assert state.sha is None
    assert state.detail


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def test_sha256_matches_a_known_value(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"project-iron")
    import hashlib

    assert sha256_file(target) == hashlib.sha256(b"project-iron").hexdigest()


def test_openvino_ir_hashes_its_weights_too(tmp_path: Path) -> None:
    """Hashing only the .xml would miss a re-quantization entirely.

    Re-quantizing rewrites the .bin and frequently leaves the topology
    byte-identical, so an .xml-only manifest would report two materially
    different models as the same one.
    """
    xml = tmp_path / "model.xml"
    binary = tmp_path / "model.bin"
    xml.write_text("<net/>")
    binary.write_bytes(b"weights v1")

    first = hash_model_files([xml])
    assert set(first) == {str(xml), str(binary)}

    binary.write_bytes(b"weights v2")
    second = hash_model_files([xml])
    assert second[str(xml)] == first[str(xml)], "topology should be unchanged"
    assert second[str(binary)] != first[str(binary)], "weight change went unnoticed"


def test_missing_model_is_recorded_not_skipped(tmp_path: Path) -> None:
    """Omitting an absent file would make a broken run look like a smaller one."""
    hashes = hash_model_files([tmp_path / "absent.pth"])
    assert "MISSING" in hashes[str(tmp_path / "absent.pth")]


def test_library_versions_record_absence_explicitly() -> None:
    versions = library_versions()
    assert "python" in versions
    assert "numpy" in versions
    # torch is optional in this environment; either way it must be stated.
    assert versions["torch"] == "not installed" or versions["torch"]


# ---------------------------------------------------------------------------
# Manifest identity
# ---------------------------------------------------------------------------


def test_manifest_is_stable_across_two_captures() -> None:
    """Two captures of an unchanged setup must share a manifest_sha.

    This is the property that makes run comparison possible at all. If the
    hash moved on every capture it would identify nothing.
    """
    config = IronConfig.load()
    first = RunManifest.capture(config, model_paths=[])
    second = RunManifest.capture(config, model_paths=[])
    assert first.manifest_sha == second.manifest_sha


def test_manifest_sha_excludes_the_timestamp() -> None:
    """The hash answers 'same setup?', not 'same moment?'."""
    config = IronConfig.load()
    manifest = RunManifest.capture(config, model_paths=[])
    relabelled = manifest.model_copy(
        update={"started_at_utc": "1999-12-31T23:59:59+00:00"}
    )
    assert relabelled.compute_sha() == manifest.compute_sha()


def test_manifest_sha_changes_when_the_config_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = RunManifest.capture(IronConfig.load(), model_paths=[]).manifest_sha
    monkeypatch.setenv("IRON_PIPELINE__CLIP_FRAMES", "8")
    assert (
        RunManifest.capture(IronConfig.load(), model_paths=[]).manifest_sha != baseline
    )


def test_manifest_sha_changes_when_model_weights_change(tmp_path: Path) -> None:
    """The coupling this whole module exists for.

    Re-exporting an encoder invalidates every vector in every index built from
    it. The manifest is what makes that detectable.
    """
    weights = tmp_path / "encoder.pth"
    weights.write_bytes(b"version one")
    config = IronConfig.load()

    before = RunManifest.capture(config, model_paths=[weights]).manifest_sha
    weights.write_bytes(b"version two")
    after = RunManifest.capture(config, model_paths=[weights]).manifest_sha
    assert before != after


def test_manifest_records_thread_settings_and_host() -> None:
    manifest = RunManifest.capture(IronConfig.load(), model_paths=[])
    assert manifest.hostname
    assert manifest.cpu_model
    assert manifest.cpu_count_logical >= 1
    assert set(manifest.thread_settings) == {
        "ov_num_threads",
        "torch_num_threads",
        "seed",
    }


# ---------------------------------------------------------------------------
# Writing and reading back
# ---------------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    manifest = RunManifest.capture(IronConfig.load(), model_paths=[])
    path = manifest.write(tmp_path / "nested" / "manifest.json")
    assert path.exists()

    loaded = RunManifest.read(path)
    assert loaded.manifest_sha == manifest.manifest_sha
    assert loaded.config_sha == manifest.config_sha


def test_read_rejects_a_tampered_manifest(tmp_path: Path) -> None:
    """A record that can be silently edited is not evidence."""
    path = RunManifest.capture(IronConfig.load(), model_paths=[]).write(
        tmp_path / "manifest.json"
    )
    payload = json.loads(path.read_text())
    payload["config_sha"] = "0" * 64
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifestError, match="modified since it was written"):
        RunManifest.read(path)


def test_write_failure_raises_rather_than_warning(tmp_path: Path) -> None:
    """An unwritable manifest must stop the run, not be logged and ignored."""
    blocker = tmp_path / "blocker"
    blocker.write_text("I am a file, not a directory")

    manifest = RunManifest.capture(IronConfig.load(), model_paths=[])
    with pytest.raises(ManifestError, match="Refusing to start"):
        manifest.write(blocker / "manifest.json")


def test_summary_flags_a_dirty_tree(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (git_repo / "tracked.txt").write_text("edited\n")
    manifest = RunManifest.capture(IronConfig.load(), model_paths=[]).model_copy(
        update={"git": git_state(git_repo)}
    )
    assert "DIRTY WORKING TREE" in manifest.summary()


# ---------------------------------------------------------------------------
# Integration with the endurance run
# ---------------------------------------------------------------------------


class _StableExtractor:
    def extract(self, video: object) -> dict[str, object]:
        import numpy as np

        return {"semantic_tracks": np.zeros((1, 2, 4, 8), dtype=np.float64)}


@pytest.fixture()
def scoped_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IronConfig:
    monkeypatch.setenv("IRON_PATHS__LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("IRON_PATHS__OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("IRON_PIPELINE__CLIP_H", "16")
    monkeypatch.setenv("IRON_PIPELINE__CLIP_W", "16")
    return IronConfig.load()


def test_run_writes_a_manifest_before_processing(scoped_config: IronConfig) -> None:
    code = execute(scoped_config, mode="steady", iterations=4, factory=_StableExtractor)
    assert code == ExitCode.OK

    manifest_path = scoped_config.paths.resolved_output_dir / "manifest.json"
    assert manifest_path.exists(), "run produced results without a manifest"
    assert RunManifest.read(manifest_path).config_sha == scoped_config.config_sha()


def test_every_metrics_record_carries_the_manifest_sha(
    scoped_config: IronConfig,
) -> None:
    """Results lacking a manifest_sha are unattributable, so none may lack one."""
    execute(scoped_config, mode="steady", iterations=5, factory=_StableExtractor)

    manifest = RunManifest.read(
        scoped_config.paths.resolved_output_dir / "manifest.json"
    )
    metrics_path = (
        scoped_config.paths.resolved_log_dir / scoped_config.endurance.metrics_filename
    )
    records = [
        json.loads(line) for line in metrics_path.read_text().splitlines() if line
    ]

    assert len(records) == 5
    assert all(record["manifest_sha"] == manifest.manifest_sha for record in records)


def test_reinit_records_also_carry_the_manifest_sha(
    scoped_config: IronConfig,
) -> None:
    execute(scoped_config, mode="reinit", iterations=3, factory=_StableExtractor)
    metrics_path = (
        scoped_config.paths.resolved_log_dir / scoped_config.endurance.metrics_filename
    )
    records = [
        json.loads(line) for line in metrics_path.read_text().splitlines() if line
    ]
    assert records and all(record["manifest_sha"] for record in records)


def test_run_refuses_to_start_when_the_manifest_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No manifest, no run — and a distinct exit code so ops can tell why."""
    blocker = tmp_path / "outputs"
    blocker.write_text("a file where the output directory should be")

    monkeypatch.setenv("IRON_PATHS__LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("IRON_PATHS__OUTPUT_DIR", str(blocker))
    monkeypatch.setenv("IRON_PIPELINE__CLIP_H", "16")
    monkeypatch.setenv("IRON_PIPELINE__CLIP_W", "16")
    config = IronConfig.load()

    extractor = _StableExtractor()
    calls: list[int] = []

    def counting_factory() -> _StableExtractor:
        calls.append(1)
        return extractor

    code = execute(config, mode="steady", iterations=10, factory=counting_factory)
    assert code == ExitCode.MANIFEST_UNWRITABLE
    assert calls == [], "the pipeline was constructed despite having no manifest"


# ---------------------------------------------------------------------------
# require_export_manifest — ADR 0008 (Day 15): no artifact loads unmanifested
# ---------------------------------------------------------------------------


def test_require_export_manifest_raises_when_sidecar_is_absent(tmp_path: Path) -> None:
    model = tmp_path / "some_model.xml"
    model.write_text("<xml/>")
    with pytest.raises(ManifestError, match="no export_manifest.json"):
        require_export_manifest(model)


def test_require_export_manifest_raises_when_manifest_names_a_different_file(
    tmp_path: Path,
) -> None:
    model = tmp_path / "some_model.xml"
    model.write_text("<xml/>")
    (tmp_path / "export_manifest.json").write_text(
        json.dumps({"artifacts": {"xml": {"name": "a_different_model.xml"}}})
    )
    with pytest.raises(ManifestError, match="does not name"):
        require_export_manifest(model)


def test_require_export_manifest_raises_on_malformed_json(tmp_path: Path) -> None:
    model = tmp_path / "some_model.xml"
    model.write_text("<xml/>")
    (tmp_path / "export_manifest.json").write_text("{not valid json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        require_export_manifest(model)


def test_require_export_manifest_succeeds_and_returns_payload(tmp_path: Path) -> None:
    model = tmp_path / "some_model.xml"
    model.write_text("<xml/>")
    payload = {
        "artifacts": {"xml": {"name": "some_model.xml", "sha256": "abc"}},
        "source_checkpoint": {"path": "models/weights/vjepa2_vitl"},
    }
    (tmp_path / "export_manifest.json").write_text(json.dumps(payload))

    result = require_export_manifest(model)
    assert result == payload


def test_require_export_manifest_accepts_the_real_day3_export() -> None:
    """The one artifact this repo actually ships: the Day-3 scripted FP32
    export, still the only one with a real manifest and a verified
    forensic answer (see ADR 0008 and docs/adr/0008-production-provenance-lost.md).
    """
    real_xml = REPO_ROOT / "models" / "export" / "2026-07-31" / "vjepa2_vitl_fp32.xml"
    if not real_xml.exists():
        pytest.skip("Day-3 export artifact not present in this checkout")
    payload = require_export_manifest(real_xml)
    assert payload["source_checkpoint"]["path"] == "models/weights/vjepa2_vitl"


def test_rebuild_index_audit_reports_void_artifacts_as_unattributable(
    tmp_path: Path,
) -> None:
    """ADR 0008, Decision 2: a void artifact's registry entry reads
    unattributable, not pending — no future action resolves it.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import rebuild_index

    import numpy as np

    (tmp_path / "orphan.npy").write_bytes(np.zeros(4, dtype=np.float32).tobytes())

    void_count = rebuild_index.audit([tmp_path])
    assert void_count == 1


def test_semantic_extractor_checks_manifest_before_compiling_model() -> None:
    """Regression guard: the manifest check must run before
    ov.Core().compile_model, not after — an artifact that fails provenance
    must never reach the compiler. Asserted on the source order directly,
    the same pattern this repo uses for other structural invariants that
    a later edit could silently reorder.
    """
    import inspect

    from src.semantics import semantic_extractor

    source = inspect.getsource(semantic_extractor.SemanticExtractor.__init__)
    manifest_idx = source.index("require_export_manifest(")
    compile_idx = source.index("compile_model(")
    assert manifest_idx < compile_idx, (
        "require_export_manifest() must run before core.compile_model() in "
        "SemanticExtractor.__init__"
    )
