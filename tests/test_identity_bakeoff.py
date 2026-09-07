"""Tests for the backbone bake-off harness (Day 36, Objective 4).

Built, not run for real — these tests exercise the harness against the
synthetic stand-in only, and assert every result carries SELF_TEST_LABEL
when run in self-test mode. A backbone bake-off VERDICT is explicitly out
of scope: no test here asserts one backbone beats another.
"""

from __future__ import annotations

import pytest
import torch

from src.data.registry import DatasetEntry, DatasetRegistry, normalise
from src.identity.bakeoff import (
    clothing_change_robustness,
    open_bakeoff_eval_set,
    patch_boundary_discontinuity,
    same_object_retrieval_map,
    temporal_embedding_stability,
)
from src.identity.selftest import (
    SELF_TEST_LABEL,
    SyntheticStandInBackbone,
    apply_synthetic_clothing_change,
    make_synthetic_identity_gallery,
    make_synthetic_patch_positions,
)


def _backbone() -> SyntheticStandInBackbone:
    return SyntheticStandInBackbone(input_dim=16, feature_dim=32, seed=42)


def test_same_object_retrieval_map_self_test_labelled_and_evidenced() -> None:
    backbone = _backbone()
    clips, labels = make_synthetic_identity_gallery(
        n_identities=6, samples_per_identity=8, input_dim=16, seed=42
    )
    pos0, posk = make_synthetic_patch_positions(len(labels), grid_size=4, seed=42)

    result = same_object_retrieval_map(
        backbone=backbone,
        clips=clips,
        identity_labels=labels,
        pos0=pos0,
        posk=posk,
        self_test=True,
    )

    assert result.self_test_label == SELF_TEST_LABEL
    evidence = result.as_evidence()
    assert evidence["metric_name"] == "identity.bakeoff.retrieval_map"
    assert any(b["name"] == "position_only" for b in evidence["baselines"])
    assert any(b["name"] == "chance" for b in evidence["baselines"])
    assert SELF_TEST_LABEL in result.render()


def test_same_object_retrieval_map_refuses_when_no_query_crosses_a_boundary() -> None:
    backbone = _backbone()
    clips, labels = make_synthetic_identity_gallery(
        n_identities=4, samples_per_identity=4, input_dim=16, seed=1
    )
    same_position = [(0, 0)] * len(labels)
    with pytest.raises(ValueError, match="cross-boundary"):
        same_object_retrieval_map(
            backbone=backbone,
            clips=clips,
            identity_labels=labels,
            pos0=same_position,
            posk=same_position,
        )


def test_temporal_embedding_stability_high_for_a_smoothly_varying_sequence() -> None:
    backbone = _backbone()
    base = torch.randn(1, 16)
    # Small jitter around a fixed point -- the same "entity" over time.
    sequence = base + 0.01 * torch.randn(10, 16)
    result = temporal_embedding_stability(
        backbone=backbone, clip_sequence=sequence, self_test=True
    )
    assert result.value > 0.9
    assert result.self_test_label == SELF_TEST_LABEL


def test_temporal_embedding_stability_requires_at_least_two_timesteps() -> None:
    backbone = _backbone()
    with pytest.raises(ValueError, match="at least 2 timesteps"):
        temporal_embedding_stability(
            backbone=backbone, clip_sequence=torch.randn(1, 16)
        )


def test_patch_boundary_discontinuity_is_a_nonnegative_descriptor() -> None:
    backbone = _backbone()
    a = torch.randn(20, 16)
    b = a + 0.1 * torch.randn(20, 16)
    result = patch_boundary_discontinuity(
        backbone=backbone, clips_a=a, clips_b=b, self_test=True
    )
    assert result.value >= 0.0
    # No trivial ceiling is defined for this metric -- margin renders NaN.
    import math

    assert math.isnan(result.margin)


def test_clothing_change_robustness_beats_chance_when_shift_is_moderate() -> None:
    backbone = _backbone()
    clips, labels = make_synthetic_identity_gallery(
        n_identities=6, samples_per_identity=8, input_dim=16, seed=42
    )
    perturbed = apply_synthetic_clothing_change(clips, labels, seed=42, shift_scale=0.5)
    result = clothing_change_robustness(
        backbone=backbone,
        clean_clips=clips,
        perturbed_clips=perturbed,
        identity_labels=labels,
        self_test=True,
    )
    evidence = result.as_evidence()
    chance = next(b["value"] for b in evidence["baselines"] if b["name"] == "chance")
    assert result.value > chance
    clean_baseline = next(
        b for b in evidence["baselines"] if b["name"] == "clean_embedding"
    )
    assert clean_baseline["flag_worthy"] is False


# -- open_bakeoff_eval_set: lane R permitted, eval-only stack walk respected --


def test_open_bakeoff_eval_set_accepts_lane_r_from_a_neutral_context() -> None:
    from datetime import date

    from src.data.registry import LicenseSnapshot

    snapshot = LicenseSnapshot(
        url="https://example.org", verified_date=date(2026, 9, 7), text_sha256="0" * 64
    )
    entry = DatasetEntry(name="CHIRLA-like", lane="R", license_snapshot=snapshot)
    registry = DatasetRegistry({normalise(entry.name): entry})
    resolved = open_bakeoff_eval_set(registry, "CHIRLA-like")
    assert resolved.lane == "R"


def test_bakeoff_module_carries_no_training_path_marker() -> None:
    """This harness is an eval path. If someone ever adds
    IRON_TRAINING_PATH = True here by mistake (e.g. copy-pasting from
    lane_gate.py), DatasetRegistry.open_for_eval's stack walk would refuse
    lane-R data reached through this module even for legitimate eval use —
    this test pins the module's actual (absent) marker so that regression
    would be caught immediately."""
    import src.identity.bakeoff as bakeoff_module
    from src.data.registry import TRAINING_PATH_MARKER

    assert not getattr(bakeoff_module, TRAINING_PATH_MARKER, False)
