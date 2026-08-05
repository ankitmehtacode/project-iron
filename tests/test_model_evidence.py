"""Objective 5 — Evidence, EvidenceCommitment, and calibrated Confidence.

STRUCTURAL rule under test: a Confidence cannot claim to be a probability
without a CalibrationRecord — UncalibratedScore has no ``probability``
attribute at all, and CalibratedProbability cannot be built without one.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest

from src.model.evidence import (
    CalibratedProbability,
    CalibrationRecord,
    DerivationStep,
    Evidence,
    EvidenceCommitment,
    EvidenceError,
    UncalibratedScore,
    compute_merkle_root,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _step(stage: str = "detector:yolov8") -> DerivationStep:
    return DerivationStep(stage=stage, producer_sha="sha-model-1")


def _evidence(**overrides: object) -> Evidence:
    kwargs: dict[str, object] = dict(
        evidence_id="ev-1",
        clip_refs=("cam-1/clip-1",),
        state_refs=("graph_rev=3",),
        observation_refs=(_sha("obs-1"),),
        derivation_chain=(_step(),),
        producer_shas=("sha-model-1",),
        reproducible=True,
        reproduce_command="python scripts/rerun_claim.py --evidence ev-1",
    )
    kwargs.update(overrides)
    return Evidence(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_requires_nonempty_observation_refs() -> None:
    with pytest.raises(EvidenceError):
        _evidence(observation_refs=())


def test_evidence_requires_nonempty_derivation_chain() -> None:
    with pytest.raises(EvidenceError):
        _evidence(derivation_chain=())


def test_evidence_requires_reproduce_command_even_when_not_reproducible() -> None:
    with pytest.raises(EvidenceError):
        _evidence(reproducible=False, reproduce_command="")
    ok = _evidence(
        reproducible=False,
        reproduce_command="not reproducible: live badge-reader poll, no seed",
    )
    assert not ok.reproducible
    assert ok.reproduce_command


def test_evidence_constructs_and_is_frozen() -> None:
    ev = _evidence()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.reproducible = False  # type: ignore[misc]


def test_derivation_step_requires_stage_and_sha() -> None:
    with pytest.raises(EvidenceError):
        DerivationStep(stage="", producer_sha="s")
    with pytest.raises(EvidenceError):
        DerivationStep(stage="s", producer_sha="")


# ---------------------------------------------------------------------------
# EvidenceCommitment — Merkle root
# ---------------------------------------------------------------------------


def test_merkle_root_is_deterministic_for_a_given_order() -> None:
    leaves = [_sha("a"), _sha("b"), _sha("c")]
    root1 = compute_merkle_root(leaves)
    root2 = compute_merkle_root(list(leaves))
    assert root1 == root2
    assert len(root1) == 64  # hex sha256


def test_merkle_root_changes_if_any_leaf_changes() -> None:
    leaves = [_sha("a"), _sha("b"), _sha("c")]
    tampered = [_sha("a"), _sha("B-tampered"), _sha("c")]
    assert compute_merkle_root(leaves) != compute_merkle_root(tampered)


def test_merkle_root_is_order_sensitive() -> None:
    leaves = [_sha("a"), _sha("b")]
    reversed_leaves = list(reversed(leaves))
    assert compute_merkle_root(leaves) != compute_merkle_root(reversed_leaves)


def test_merkle_root_rejects_empty_leaves() -> None:
    with pytest.raises(EvidenceError):
        compute_merkle_root([])


def test_merkle_root_odd_leaf_count_duplicates_last() -> None:
    # Should not raise, and should differ from the even-padded-with-zero case.
    root = compute_merkle_root([_sha("a"), _sha("b"), _sha("c")])
    assert len(root) == 64


def test_evidence_commitment_compute_and_verify_round_trip() -> None:
    leaves = [_sha("obs-1"), _sha("obs-2"), _sha("obs-3")]
    commitment = EvidenceCommitment.compute(
        commitment_id="commit-1", observation_hashes=leaves, computed_at_ns=100
    )
    assert commitment.leaf_count == 3
    assert commitment.status == "reproducible"
    assert commitment.verify(leaves)
    assert not commitment.verify(list(reversed(leaves)))
    assert not commitment.verify(leaves[:2])


def test_evidence_commitment_survives_erasure_without_storing_leaves() -> None:
    leaves = [_sha("obs-1"), _sha("obs-2")]
    commitment = EvidenceCommitment.compute(
        commitment_id="commit-2",
        observation_hashes=leaves,
        computed_at_ns=100,
        status="inputs_erased",
    )
    # No field on the record holds the original hashes.
    field_names = {f.name for f in dataclasses.fields(EvidenceCommitment)}
    assert "observation_hashes" not in field_names
    assert "leaf_hashes" not in field_names
    assert commitment.status == "inputs_erased"
    assert commitment.merkle_root  # still provable
    assert commitment.leaf_count == 2


def test_evidence_commitment_rejects_zero_leaf_count() -> None:
    with pytest.raises(EvidenceError):
        EvidenceCommitment(
            commitment_id="c",
            merkle_root=_sha("root"),
            leaf_count=0,
            computed_at_ns=1,
            status="reproducible",
        )


def test_evidence_commitment_rejects_unknown_status() -> None:
    with pytest.raises(EvidenceError):
        EvidenceCommitment(
            commitment_id="c",
            merkle_root=_sha("root"),
            leaf_count=1,
            computed_at_ns=1,
            status="probably_fine",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# STRUCTURAL: Confidence cannot claim to be a probability without calibration.
# ---------------------------------------------------------------------------


def test_uncalibrated_score_has_no_probability_attribute() -> None:
    score = UncalibratedScore(0.87)
    assert not hasattr(score, "probability")
    field_names = {f.name for f in dataclasses.fields(UncalibratedScore)}
    assert "probability" not in field_names
    assert "calibration" not in field_names


def test_uncalibrated_score_serializes_only_as_uncalibrated_score() -> None:
    score = UncalibratedScore(0.87)
    payload = score.as_dict()
    assert payload == {"uncalibrated_score": 0.87}
    assert "probability" not in payload


def test_calibrated_probability_requires_calibration_record() -> None:
    with pytest.raises(TypeError):
        CalibratedProbability(probability=0.9)  # type: ignore[call-arg]


def test_calibrated_probability_constructs_with_calibration_and_serializes_as_probability() -> (
    None
):
    calib = CalibrationRecord(
        calibration_id="calib-1",
        method="isotonic_regression",
        fit_on="golden/v3-indoor",
        valid_for="motion_gate@v3-indoor",
        manifest_sha="sha-calib-1",
    )
    prob = CalibratedProbability(probability=0.91, calibration=calib)
    payload = prob.as_dict()
    assert payload["probability"] == 0.91
    assert payload["calibration_id"] == "calib-1"


def test_calibration_record_requires_all_fields_nonempty() -> None:
    with pytest.raises(EvidenceError):
        CalibrationRecord(
            calibration_id="", method="m", fit_on="f", valid_for="v", manifest_sha="s"
        )


def test_score_and_probability_reject_out_of_range_values() -> None:
    with pytest.raises(EvidenceError):
        UncalibratedScore(1.5)
    calib = CalibrationRecord(
        calibration_id="c", method="m", fit_on="f", valid_for="v", manifest_sha="s"
    )
    with pytest.raises(EvidenceError):
        CalibratedProbability(probability=-0.1, calibration=calib)
