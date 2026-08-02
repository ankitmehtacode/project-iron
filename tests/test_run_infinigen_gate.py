"""Sanity for the Infinigen gate runner, without needing Infinigen installed.

The runner is the mechanism that lets a real Infinigen render (day 11+) be
scored by the same gates that condemned v3-indoor. The gates themselves have
their own tests; what this file covers is the loader that stands between the
render outputs and the gates — the surface where an off-by-one on frame order,
a shape mismatch, or a silent unit conversion would let a valid gate report on
a fixture that never existed.

None of this needs bpy or infinigen: the fixtures are hand-written .npz files
in the pinned venv, matching the schema ``scripts/infinigen_generate.py``
promises to produce.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from run_infinigen_gate import load_sample


def _write_frame(path: Path, rgb: np.ndarray, depth: np.ndarray) -> None:
    np.savez_compressed(path, rgb=rgb.astype(np.uint8), depth_metres=depth.astype(np.float32))


def test_load_sample_stacks_frames_in_lexical_order(tmp_path: Path) -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    for index, filler in enumerate((10, 20, 30, 5), start=1):
        _write_frame(tmp_path / f"frame_{index:04d}.npz", np.full_like(rgb, filler), np.ones((4, 6)) * filler)
    frames, gt = load_sample(tmp_path)
    assert frames.shape == (4, 4, 6, 3)
    assert gt.shape == (4, 4, 6)
    # File 4 was named frame_0004 but had value 5 — lexical order stacks it last.
    assert frames[3, 0, 0, 0] == 5
    assert gt[3, 0, 0] == 5.0


def test_load_sample_refuses_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path)


def test_load_sample_refuses_missing_keys(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "frame_0001.npz", pixels=np.zeros((2, 2, 3)))
    with pytest.raises(KeyError):
        load_sample(tmp_path)


def test_load_sample_refuses_mixed_shapes(tmp_path: Path) -> None:
    _write_frame(tmp_path / "frame_0001.npz", np.zeros((4, 4, 3)), np.zeros((4, 4)))
    _write_frame(tmp_path / "frame_0002.npz", np.zeros((5, 5, 3)), np.zeros((5, 5)))
    with pytest.raises(ValueError, match="mixed frame shapes"):
        load_sample(tmp_path)
