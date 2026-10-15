"""The detection validity gate (Day 42, Objective B7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.data import validity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validity_matrix  # noqa: E402


def _textured(t: int = 4, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (t, size, size, 3), dtype=np.uint8)


def _flat(t: int = 4, size: int = 64) -> np.ndarray:
    return np.full((t, size, size, 3), 128, dtype=np.uint8)


def test_detection_appears_in_the_matrix_with_no_other_edit() -> None:
    assert "detection" in validity.GATES
    assert "detection" in validity_matrix.registered_capabilities()


def test_passes_with_texture_and_one_thick_cell() -> None:
    result = validity.evaluate(
        "detection",
        "fixture",
        frames=_textured(),
        instance_counts={"Person": {"[64,128)": 40, "[0,32)": 3}},
    )
    assert result.passed
    assert result.evidence["scoreable_cells"] == ["Person x [64,128)"]
    assert result.evidence["unscoreable_cells"] == ["Person x [0,32) (3)"]


def test_refuses_when_every_cell_is_thin() -> None:
    result = validity.evaluate(
        "detection",
        "fixture",
        frames=_textured(),
        instance_counts={"Person": {"[64,128)": 29}},
    )
    assert not result.passed and "30" in result.reason


def test_refuses_flat_frames_whatever_the_counts() -> None:
    result = validity.evaluate(
        "detection",
        "fixture",
        frames=_flat(),
        instance_counts={"Person": {"[64,128)": 400}},
    )
    assert not result.passed and "texture" in result.reason


def test_refuses_without_annotations() -> None:
    result = validity.evaluate("detection", "fixture", frames=_textured())
    assert not result.passed and "annotations" in result.reason


def test_the_texture_floor_is_the_appearance_gates_floor() -> None:
    a = validity.evaluate("appearance_semantics", "f", frames=_flat())
    d = validity.evaluate(
        "detection", "f", frames=_flat(), instance_counts={"P": {"b": 100}}
    )
    assert a.evidence["texture_energy"] == d.evidence["texture_energy"]
    assert d.evidence["texture_energy_floor"] == validity.MIN_TEXTURE_ENERGY
