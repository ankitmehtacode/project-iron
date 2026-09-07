"""ADR 0001's withdrawal drill: evict, retrain, confirm.

``docs/adr/0001-identity-adapter-architecture.md``'s whole argument is that
withdrawal costs an adapter retrain, not a backbone retrain — and its own
§Constraint on Phase 3 (item 4) says plainly: "A withdrawal drill — evict,
retrain, confirm — is exercised on synthetic identities before Site Zero
enrolment begins. An erasure path that has never been run is a claim, not a
capability." This module is that drill, run for the first time (Day 36).

Structural verification only — explicitly NOT a timing claim:

1. The backbone artifact's file hash is byte-identical before either
   training run, after the first (full-gallery) run, and after the second
   (post-withdrawal) run — zero bytes of the frozen backbone were touched
   by either. Checked two ways: the on-disk file's own hash never changes
   (nothing rewrites it — the same guarantee a real frozen ``.xml``/``.bin``
   backbone artifact gets), AND a live re-hash of the backbone's in-memory
   weights still matches that file hash after both runs (catching an
   in-place mutation bug even if the file itself was never touched).
2. The two adapter checkpoints — trained before and after evicting one
   identity's contribution — have distinct ``adapter_sha`` and distinct
   manifests.

If either assertion fails, ADR 0001's erasure architecture is invalidated,
and that is this project's finding of the day, not a footnote — see
``FOUNDATION_REPORT.md``'s Day-36 section.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.identity.adapter import Adapter, AdapterConfig
from src.identity.checkpoint import CheckpointManifest, write_selftest_checkpoint_manifest
from src.identity.selftest import (
    SyntheticStandInBackbone,
    make_synthetic_identity_gallery,
    make_triplet_sampler,
)
from src.identity.trainer import TrainingConfig, train_adapter
from src.provenance import sha256_file


@dataclass(frozen=True)
class ErasureDrillResult:
    backbone_file_hash_before: str
    backbone_file_hash_after_first_train: str
    backbone_file_hash_after_retrain: str
    backbone_live_weights_hash_after_first_train: str
    backbone_live_weights_hash_after_retrain: str
    backbone_untouched: bool
    first_manifest: CheckpointManifest
    retrained_manifest: CheckpointManifest
    adapter_shas_distinct: bool
    withdrawn_identity: int

    def render(self) -> str:
        verdict = "PASS" if self.backbone_untouched and self.adapter_shas_distinct else "FAIL"
        return (
            f"erasure drill: {verdict}\n"
            f"  backbone file hash   before={self.backbone_file_hash_before[:16]}...\n"
            f"                       after 1st train={self.backbone_file_hash_after_first_train[:16]}...\n"
            f"                       after retrain={self.backbone_file_hash_after_retrain[:16]}...\n"
            f"  backbone untouched (file AND live weights, both runs): {self.backbone_untouched}\n"
            f"  adapter_sha before withdrawal : {self.first_manifest.adapter_sha[:16]}...\n"
            f"  adapter_sha after withdrawal  : {self.retrained_manifest.adapter_sha[:16]}...\n"
            f"  adapter checkpoints distinct  : {self.adapter_shas_distinct}\n"
            f"  withdrawn synthetic identity  : {self.withdrawn_identity}"
        )


def run_erasure_drill(
    *,
    backbone_weights_path: Path,
    n_identities: int = 8,
    samples_per_identity: int = 12,
    input_dim: int = 32,
    feature_dim: int = 64,
    seed: int = 20260907,
    epochs: int = 40,
) -> ErasureDrillResult:
    """Run the drill once and return its structural verdict.

    Args:
        backbone_weights_path: Where the stand-in "backbone artifact" file
            is written (see :meth:`SyntheticStandInBackbone.save_weights`).
            Written once, at the start, and never rewritten — the same
            guarantee a real frozen backbone export gets.
    """
    backbone = SyntheticStandInBackbone(
        input_dim=input_dim, feature_dim=feature_dim, seed=seed
    )
    backbone.save_weights(backbone_weights_path)
    file_hash_before = sha256_file(backbone_weights_path)
    assert backbone.current_weights_sha() == file_hash_before, (
        "test-setup invariant: the freshly-saved file must hash identically "
        "to the live tensor it was saved from"
    )

    clips, labels = make_synthetic_identity_gallery(
        n_identities=n_identities,
        samples_per_identity=samples_per_identity,
        input_dim=input_dim,
        seed=seed,
        cluster_scale=1.2,
        noise_scale=1.0,
    )
    adapter_config = AdapterConfig(
        input_dim=feature_dim, hidden_dim=64, output_dim=32, num_layers=2
    )
    training_config = TrainingConfig(
        epochs=epochs, learning_rate=1e-2, triplet_margin=0.3, seed=seed
    )

    # -- Tier 2, first pass: train on the FULL synthetic gallery ----------
    first_adapter = Adapter(adapter_config)
    first_sampler = make_triplet_sampler(clips, labels, batch_size=16, seed=seed)
    train_adapter(
        backbone=backbone,
        adapter=first_adapter,
        triplets=first_sampler,
        config=training_config,
    )
    file_hash_after_first = sha256_file(backbone_weights_path)
    live_hash_after_first = backbone.current_weights_sha()

    first_manifest = write_selftest_checkpoint_manifest(
        backbone_weights_path.parent / "checkpoint_before_withdrawal.manifest.json",
        adapter=first_adapter,
        backbone_sha=backbone.backbone_sha,
        synthetic_dataset_label=f"synthetic-gallery-{n_identities}-identities",
        training_config_sha=training_config.config_sha(),
    )

    # -- Withdrawal: evict one identity, retrain from scratch -------------
    withdrawn_identity = n_identities - 1
    keep = [i for i, label in enumerate(labels) if label != withdrawn_identity]
    retained_clips = clips[keep]
    retained_labels = [labels[i] for i in keep]

    retrained_adapter = Adapter(adapter_config)
    retrain_sampler = make_triplet_sampler(
        retained_clips, retained_labels, batch_size=16, seed=seed + 1
    )
    train_adapter(
        backbone=backbone,
        adapter=retrained_adapter,
        triplets=retrain_sampler,
        config=training_config,
    )
    file_hash_after_retrain = sha256_file(backbone_weights_path)
    live_hash_after_retrain = backbone.current_weights_sha()

    retrained_manifest = write_selftest_checkpoint_manifest(
        backbone_weights_path.parent / "checkpoint_after_withdrawal.manifest.json",
        adapter=retrained_adapter,
        backbone_sha=backbone.backbone_sha,
        synthetic_dataset_label=(
            f"synthetic-gallery-{n_identities - 1}-identities-post-withdrawal"
        ),
        training_config_sha=training_config.config_sha(),
    )

    backbone_untouched = (
        file_hash_before
        == file_hash_after_first
        == file_hash_after_retrain
        == live_hash_after_first
        == live_hash_after_retrain
    )

    return ErasureDrillResult(
        backbone_file_hash_before=file_hash_before,
        backbone_file_hash_after_first_train=file_hash_after_first,
        backbone_file_hash_after_retrain=file_hash_after_retrain,
        backbone_live_weights_hash_after_first_train=live_hash_after_first,
        backbone_live_weights_hash_after_retrain=live_hash_after_retrain,
        backbone_untouched=backbone_untouched,
        first_manifest=first_manifest,
        retrained_manifest=retrained_manifest,
        adapter_shas_distinct=(
            first_manifest.adapter_sha != retrained_manifest.adapter_sha
        ),
        withdrawn_identity=withdrawn_identity,
    )
