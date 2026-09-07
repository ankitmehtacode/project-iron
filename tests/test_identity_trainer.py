"""Tests for the adapter trainer, checkpoint provenance, and promotion gate
(Day 36, Objective 3).

The load-bearing claims here: (1) training actually moves the loss, not
just runs without crashing; (2) a checkpoint manifest structurally cannot
record non-lane-C training data; (3) a SELF_TEST checkpoint structurally
cannot be marked promoted; (4) promotion is refused, not granted, when the
adapter fails to beat its own frozen backbone.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import torch

from src.data.registry import ConsentRecord, DatasetEntry, LicenseSnapshot
from src.identity.adapter import Adapter, AdapterConfig
from src.identity.checkpoint import (
    write_checkpoint_manifest,
    write_selftest_checkpoint_manifest,
)
from src.identity.promotion import evaluate_promotion
from src.identity.selftest import (
    SELF_TEST_LABEL,
    SyntheticStandInBackbone,
    make_synthetic_identity_gallery,
    make_triplet_sampler,
)
from src.identity.trainer import TrainingConfig, train_adapter
from src.provenance import ManifestError


def _snapshot() -> LicenseSnapshot:
    return LicenseSnapshot(
        url="https://example.org/license",
        verified_date=date(2026, 9, 7),
        text_sha256="0" * 64,
        verified_by="test",
        verified_class="ours",
    )


# -- TrainingConfig -----------------------------------------------------------


def test_training_config_sha_is_stable_and_sensitive() -> None:
    a = TrainingConfig(epochs=10, learning_rate=1e-3)
    b = TrainingConfig(epochs=10, learning_rate=1e-3)
    c = TrainingConfig(epochs=11, learning_rate=1e-3)
    assert a.config_sha() == b.config_sha()
    assert a.config_sha() != c.config_sha()


# -- train_adapter genuinely trains -------------------------------------------


def test_train_adapter_reduces_triplet_loss() -> None:
    """The central plumbing claim: gradients actually flow and reduce loss
    on a genuinely-learnable (not trivially-separable) synthetic problem."""
    backbone = SyntheticStandInBackbone(input_dim=32, feature_dim=64, seed=7)
    adapter = Adapter(
        AdapterConfig(input_dim=64, hidden_dim=64, output_dim=32, num_layers=2)
    )
    clips, labels = make_synthetic_identity_gallery(
        n_identities=8,
        samples_per_identity=12,
        input_dim=32,
        seed=7,
        cluster_scale=1.2,
        noise_scale=1.0,
    )
    sampler = make_triplet_sampler(clips, labels, batch_size=16, seed=7)
    config = TrainingConfig(epochs=60, learning_rate=1e-2, triplet_margin=0.3, seed=7)

    losses = train_adapter(
        backbone=backbone, adapter=adapter, triplets=sampler, config=config
    )

    assert len(losses) == 60
    assert losses[0] > 0.0, "test setup check: the problem must not start trivial"
    assert losses[-1] < losses[0] * 0.5, "loss did not meaningfully decrease"


def test_train_adapter_never_updates_backbone_parameters() -> None:
    backbone = SyntheticStandInBackbone(input_dim=16, feature_dim=32, seed=3)
    weight_before = backbone._projection.weight.detach().clone()
    adapter = Adapter(AdapterConfig(input_dim=32, output_dim=16))
    clips, labels = make_synthetic_identity_gallery(
        n_identities=4, samples_per_identity=8, input_dim=16, seed=3
    )
    sampler = make_triplet_sampler(clips, labels, batch_size=8, seed=3)
    train_adapter(
        backbone=backbone,
        adapter=adapter,
        triplets=sampler,
        config=TrainingConfig(epochs=10, seed=3),
    )
    assert torch.equal(backbone._projection.weight.detach(), weight_before)


# -- Checkpoint manifest: lane-C-only refusal ---------------------------------


def test_write_checkpoint_manifest_refuses_non_lane_c(tmp_path: Path) -> None:
    adapter = Adapter(AdapterConfig(input_dim=16, output_dim=8))
    bad_entry = DatasetEntry(name="MEVA", lane="R", license_snapshot=_snapshot())
    with pytest.raises(ManifestError, match="lane C"):
        write_checkpoint_manifest(
            tmp_path / "checkpoint.manifest.json",
            adapter=adapter,
            backbone_sha="backbone-x",
            dataset_entries={"MEVA": bad_entry},
            training_config_sha="cfg-sha",
        )
    assert not (tmp_path / "checkpoint.manifest.json").exists()


def test_write_checkpoint_manifest_accepts_lane_c(tmp_path: Path) -> None:
    adapter = Adapter(AdapterConfig(input_dim=16, output_dim=8))
    good_entry = DatasetEntry(
        name="site-zero-adapter-set",
        lane="C",
        license_snapshot=_snapshot(),
        content_sha="deadbeef",
    )
    path = tmp_path / "checkpoint.manifest.json"
    manifest = write_checkpoint_manifest(
        path,
        adapter=adapter,
        backbone_sha="backbone-x",
        dataset_entries={"site-zero-adapter-set": good_entry},
        training_config_sha="cfg-sha",
    )
    assert path.exists()
    assert manifest.self_test_label is None
    assert manifest.dataset_shas_and_lanes["site-zero-adapter-set"] == "C:deadbeef"
    assert manifest.adapter_sha == adapter.adapter_sha


def test_write_checkpoint_manifest_refuses_c_pending_consent_too() -> None:
    """C_pending_consent is not lane C — the manifest gate must not treat
    the two as equivalent just because both start with 'C'."""
    adapter = Adapter(AdapterConfig(input_dim=16, output_dim=8))
    pending = DatasetEntry(
        name="thinkwill-cctv-archive",
        lane="C_pending_consent",
        license_snapshot=_snapshot(),
        consent_record=ConsentRecord(
            path="x",
            sha="1" * 64,
            subjects=1,
            captured_on=date(2026, 1, 1),
        ),
    )
    with pytest.raises(ManifestError, match="lane C"):
        write_checkpoint_manifest(
            Path("/tmp/unused-checkpoint.manifest.json"),
            adapter=adapter,
            backbone_sha="b",
            dataset_entries={"thinkwill-cctv-archive": pending},
            training_config_sha="cfg-sha",
        )


# -- SELF_TEST checkpoint: structurally never promotable ---------------------


def test_selftest_checkpoint_manifest_is_never_promoted(tmp_path: Path) -> None:
    adapter = Adapter(AdapterConfig(input_dim=16, output_dim=8))
    manifest = write_selftest_checkpoint_manifest(
        tmp_path / "selftest.manifest.json",
        adapter=adapter,
        backbone_sha="synthetic-backbone-sha",
        synthetic_dataset_label="synthetic-gallery",
        training_config_sha="cfg-sha",
        promotion_evidence={"metric_name": "identity.retrieval_map", "value": 0.99},
    )
    assert manifest.promoted is False
    assert manifest.self_test_label == SELF_TEST_LABEL


def test_selftest_checkpoint_manifest_records_no_real_lane() -> None:
    """A reader of the raw JSON must be told this is not lane-C data,
    without having to already know to check self_test_label."""
    adapter = Adapter(AdapterConfig(input_dim=16, output_dim=8))
    manifest = write_selftest_checkpoint_manifest(
        Path("/tmp/unused-selftest.manifest.json"),
        adapter=adapter,
        backbone_sha="b",
        synthetic_dataset_label="synthetic-gallery",
        training_config_sha="cfg-sha",
    )
    lane_note = manifest.dataset_shas_and_lanes["synthetic-gallery"]
    assert "not lane C" in lane_note
    assert "SELF_TEST" in lane_note


# -- Promotion gate ------------------------------------------------------------


def test_promotion_result_carries_value_baseline_and_margin() -> None:
    """Structural requirement: 'value (baseline, margin)', not just a bare
    number."""
    backbone = SyntheticStandInBackbone(input_dim=16, feature_dim=32, seed=5)
    adapter = Adapter(AdapterConfig(input_dim=32, output_dim=16))
    clips, labels = make_synthetic_identity_gallery(
        n_identities=4, samples_per_identity=6, input_dim=16, seed=5
    )
    result = evaluate_promotion(
        backbone=backbone, adapter=adapter, clips=clips, identity_labels=labels
    )
    evidence = result.as_evidence()
    assert evidence["metric_name"] == "identity.retrieval_map"
    assert "value" in evidence
    assert len(evidence["baselines"]) >= 1
    assert any(b["name"] == "raw_backbone_cosine" for b in evidence["baselines"])
    assert "margin" in evidence
    assert "raw_backbone_cosine" in result.render()
    assert "margin" in result.render()


def test_promotion_result_self_labels_when_asked_and_stays_silent_otherwise() -> None:
    """Day 36's own audit: PromotionResult originally carried no
    self_test_label field at all, unlike BakeoffProbeResult — a result
    that only read as SELF_TEST because a CLI script's print statement
    happened to prefix it. Closed by making the label part of the result
    itself, so .render()/.as_evidence() are self-labelling wherever they
    end up, not just at that one call site."""
    backbone = SyntheticStandInBackbone(input_dim=16, feature_dim=32, seed=5)
    adapter = Adapter(AdapterConfig(input_dim=32, output_dim=16))
    clips, labels = make_synthetic_identity_gallery(
        n_identities=4, samples_per_identity=6, input_dim=16, seed=5
    )

    real_shaped = evaluate_promotion(
        backbone=backbone, adapter=adapter, clips=clips, identity_labels=labels
    )
    assert real_shaped.self_test_label is None
    assert real_shaped.as_evidence()["self_test_label"] is None
    assert "SELF_TEST" not in real_shaped.render()

    labelled = evaluate_promotion(
        backbone=backbone,
        adapter=adapter,
        clips=clips,
        identity_labels=labels,
        self_test=True,
    )
    assert labelled.self_test_label == SELF_TEST_LABEL
    assert labelled.as_evidence()["self_test_label"] == SELF_TEST_LABEL
    assert SELF_TEST_LABEL in labelled.render()


def test_promotion_refused_when_adapter_does_not_beat_raw_backbone() -> None:
    """An adapter that has learned nothing (random init, zero training)
    must not clear the gate just by existing."""
    backbone = SyntheticStandInBackbone(input_dim=16, feature_dim=64, seed=9)
    # A deliberately destructive adapter: collapse everything to a near-
    # constant vector, which cannot outrank the raw backbone's own
    # cosine geometry.
    adapter = Adapter(AdapterConfig(input_dim=64, hidden_dim=8, output_dim=4))
    with torch.no_grad():
        for p in adapter.parameters():
            p.zero_()
        # a zero-weight linear map produces a constant (zero) output for
        # every input; normalize() on an all-zero vector is defined (torch
        # leaves it at 0), so every embedding is identical and ranking is
        # uninformative -- the adversarial case the gate must catch.

    clips, labels = make_synthetic_identity_gallery(
        n_identities=6, samples_per_identity=8, input_dim=16, seed=9
    )
    result = evaluate_promotion(
        backbone=backbone, adapter=adapter, clips=clips, identity_labels=labels
    )
    assert result.promoted is False


def test_promotion_granted_when_training_genuinely_helps() -> None:
    backbone = SyntheticStandInBackbone(input_dim=32, feature_dim=64, seed=11)
    adapter = Adapter(
        AdapterConfig(input_dim=64, hidden_dim=64, output_dim=32, num_layers=2)
    )
    clips, labels = make_synthetic_identity_gallery(
        n_identities=8,
        samples_per_identity=12,
        input_dim=32,
        seed=11,
        cluster_scale=1.2,
        noise_scale=1.0,
    )
    sampler = make_triplet_sampler(clips, labels, batch_size=16, seed=11)
    train_adapter(
        backbone=backbone,
        adapter=adapter,
        triplets=sampler,
        config=TrainingConfig(
            epochs=60, learning_rate=1e-2, triplet_margin=0.3, seed=11
        ),
    )
    result = evaluate_promotion(
        backbone=backbone, adapter=adapter, clips=clips, identity_labels=labels
    )
    assert result.promoted is True
    assert result.margin > 0
