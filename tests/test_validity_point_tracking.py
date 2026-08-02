"""The point-tracking validity gate, and why an empty scene starves it.

Same shape as every other test in this repo that verifies a gate against a
constructed fixture. The gate measures Shi-Tomasi corner density per pixel,
and its whole reason for existing is to refuse fixtures on which a tracker's
accuracy would describe extrapolation. Two fixtures make the case:

- A textureless field (constant grey with tiny noise) has almost no local
  gradient and therefore no corners; the gate must refuse.
- A high-contrast checkerboard has corners at every square boundary; the gate
  must pass.

Reads gates through :func:`src.data.validity.evaluate` rather than calling the
function directly, because that is the surface every caller uses and its
returning-a-refusal-instead-of-raising behaviour is part of the contract.
"""

from __future__ import annotations

import numpy as np

from src.data import validity


def _flat_field(t: int = 4, size: int = 64) -> np.ndarray:
    """A perfectly constant grey field — no gradient anywhere, no corners.

    NOT a "near-constant field with tiny jitter": Shi-Tomasi at
    qualityLevel=0.01 is sensitive enough that a 3-value uniform-integer
    jitter reads as ~0.04 corners/pixel, well above the floor. That is the
    kind of construction that lets a validity test pass vacuously — the
    fixture isn't refused, the fixture isn't textureless. A truly flat field
    is what the gate is meant to refuse."""
    return np.full((t, size, size, 3), 128, dtype=np.uint8)


def _checkerboard(t: int = 4, size: int = 64, tile: int = 8) -> np.ndarray:
    """High-contrast squares — corners at every tile boundary."""
    y, x = np.indices((size, size))
    pattern = (((y // tile) + (x // tile)) % 2 * 255).astype(np.uint8)
    frame = np.stack([pattern, pattern, pattern], axis=-1)
    return np.broadcast_to(frame, (t, size, size, 3)).copy()


def test_gate_refuses_textureless_field() -> None:
    result = validity.evaluate("point_tracking", "flat", frames=_flat_field())
    assert not result.passed
    assert "corner density" in result.reason
    assert result.evidence["corner_density_per_pixel"] < (
        validity.MIN_TRACKABLE_CORNER_DENSITY
    )


def test_gate_passes_checkerboard() -> None:
    result = validity.evaluate("point_tracking", "checker", frames=_checkerboard())
    assert result.passed, result.reason
    assert result.evidence["corner_density_per_pixel"] >= (
        validity.MIN_TRACKABLE_CORNER_DENSITY
    )


def test_gate_refuses_wrong_shape() -> None:
    # A tracker fed a single frame or a 2D array is not a tracker at all.
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    result = validity.evaluate("point_tracking", "single", frames=bad)
    assert not result.passed
    assert "expected [T, H, W, C]" in result.reason


def test_capability_registered_by_evaluate() -> None:
    # Regression: a capability not in GATES gets a refusal from evaluate(),
    # not a KeyError; the fact that point_tracking is now in GATES is the
    # point of adding it, so an unrelated typo should not silently drop it.
    assert "point_tracking" in validity.GATES
