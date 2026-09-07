"""The first real verification of ADR 0001's central promise (Day 36,
Objective 5): withdrawal costs an adapter retrain, not a backbone retrain.

ADR 0001 §Constraint on Phase 3 (item 4) requires this drill to be
exercised on synthetic identities before Site Zero enrolment begins — "An
erasure path that has never been run is a claim, not a capability." This is
that run. If the backbone were ever touched, that invalidates the erasure
architecture and would be the project's finding of the day, not a footnote
— see the module docstring in src/identity/erasure_drill.py.
"""

from __future__ import annotations

from pathlib import Path

from src.identity.erasure_drill import run_erasure_drill
from src.identity.selftest import SELF_TEST_LABEL


def test_backbone_file_and_live_weights_are_byte_identical_throughout(
    tmp_path: Path,
) -> None:
    result = run_erasure_drill(
        backbone_weights_path=tmp_path / "backbone.bin",
        n_identities=6,
        samples_per_identity=8,
        input_dim=16,
        feature_dim=32,
        epochs=20,
    )

    assert result.backbone_untouched is True
    assert (
        result.backbone_file_hash_before
        == result.backbone_file_hash_after_first_train
        == result.backbone_file_hash_after_retrain
    )
    assert (
        result.backbone_live_weights_hash_after_first_train
        == result.backbone_file_hash_before
    )
    assert (
        result.backbone_live_weights_hash_after_retrain
        == result.backbone_file_hash_before
    )


def test_two_checkpoints_have_distinct_adapter_sha_and_manifests(
    tmp_path: Path,
) -> None:
    result = run_erasure_drill(
        backbone_weights_path=tmp_path / "backbone.bin",
        n_identities=6,
        samples_per_identity=8,
        input_dim=16,
        feature_dim=32,
        epochs=20,
    )

    assert result.adapter_shas_distinct is True
    assert result.first_manifest.adapter_sha != result.retrained_manifest.adapter_sha
    assert result.first_manifest != result.retrained_manifest
    # Same frozen backbone, both times.
    assert result.first_manifest.backbone_sha == result.retrained_manifest.backbone_sha


def test_both_checkpoints_are_self_test_and_never_promoted(tmp_path: Path) -> None:
    result = run_erasure_drill(
        backbone_weights_path=tmp_path / "backbone.bin",
        n_identities=6,
        samples_per_identity=8,
        input_dim=16,
        feature_dim=32,
        epochs=20,
    )
    for manifest in (result.first_manifest, result.retrained_manifest):
        assert manifest.self_test_label == SELF_TEST_LABEL
        assert manifest.promoted is False


def test_the_retrained_dataset_label_names_one_fewer_identity(tmp_path: Path) -> None:
    """Sanity check on the withdrawal itself, not just its downstream
    effects: the retrained manifest's own dataset label must say so."""
    result = run_erasure_drill(
        backbone_weights_path=tmp_path / "backbone.bin",
        n_identities=6,
        samples_per_identity=8,
        input_dim=16,
        feature_dim=32,
        epochs=20,
    )
    retrained_label = next(iter(result.retrained_manifest.dataset_shas_and_lanes))
    assert "5-identities-post-withdrawal" in retrained_label
    first_label = next(iter(result.first_manifest.dataset_shas_and_lanes))
    assert "6-identities" in first_label


def test_render_reports_pass_plainly(tmp_path: Path) -> None:
    result = run_erasure_drill(
        backbone_weights_path=tmp_path / "backbone.bin",
        n_identities=6,
        samples_per_identity=8,
        input_dim=16,
        feature_dim=32,
        epochs=20,
    )
    rendered = result.render()
    assert "PASS" in rendered
    assert "backbone untouched" in rendered.lower()
