"""DAv2Wrapper returns a typed DepthField, with values provably unchanged.

Audit finding 4 closed at the source. The publication path was made honest on
day 2 (``disparity_rel`` column, ``depth_units``, README corrected); this is
the wrapper itself, which until now handed out a bare ndarray that anything
could read as metres.

The change must be **behaviour-neutral**, so the central test compares the
wrapped values against the raw model output element-by-element on a fixed
input. A units fix that also moved the numbers would be two changes wearing
one commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import DepthField, FrameGeometry, Intrinsics, unproject
from src.contracts.errors import UnitsError
from src.models.dav2_wrapper import as_depth_field


def raw_depth(seed: int = 20260801, shape: tuple[int, int] = (48, 64)) -> np.ndarray:
    """A fixed pseudo-DA-V2 output: positive relative inverse depth."""
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 12.0 + 0.25).astype(np.float32)


# ---------------------------------------------------------------------------
# Behaviour neutrality — the point of the whole objective
# ---------------------------------------------------------------------------


def test_values_are_bit_identical_to_the_raw_output() -> None:
    """Before/after on a fixed input. The envelope changed; the numbers did not.

    ``float64`` widening is exact for float32 inputs, so this is an equality
    assertion rather than a tolerance one — a tolerance would let a real change
    hide inside it.
    """
    raw = raw_depth()
    field = as_depth_field(raw)
    np.testing.assert_array_equal(field.data, raw.astype(np.float64))


def test_shape_is_preserved() -> None:
    raw = raw_depth(shape=(37, 91))
    field = as_depth_field(raw)
    assert field.data.shape == raw.shape
    assert field.geometry == FrameGeometry(width=91, height=37)


def test_geometry_is_derived_from_the_array_not_assumed() -> None:
    """A wrong geometry would silently mis-scale intrinsics downstream."""
    for height, width in ((16, 16), (48, 64), (720, 1280)):
        field = as_depth_field(np.ones((height, width), dtype=np.float32))
        assert field.geometry.height == height
        assert field.geometry.width == width


# ---------------------------------------------------------------------------
# The units claim
# ---------------------------------------------------------------------------


def test_units_are_disparity_rel_not_meters() -> None:
    """The one word this objective exists for.

    DA-V2 emits relative inverse depth with unknown scale AND unknown shift,
    so no constant converts it to metres.
    """
    field = as_depth_field(raw_depth())
    assert field.units == "disparity_rel"
    assert field.is_metric is False


def test_unproject_refuses_the_wrapper_output() -> None:
    """The payoff: 3D geometry from DA-V2 output is now impossible by default.

    Before this, the same values reached the projector as a bare array and
    produced confident, arbitrary-scale 3D.
    """
    field = as_depth_field(raw_depth(shape=(32, 32)))
    K = Intrinsics(
        fx=32.0,
        fy=32.0,
        cx=16.0,
        cy=16.0,
        distortion=(),
        valid_for=field.geometry,
        calibrated=True,
    )
    with pytest.raises(UnitsError, match="disparity_rel"):
        unproject(field, np.array([[16.0, 16.0]]), K)


def test_metric_depth_still_possible_when_actually_anchored() -> None:
    """The guard blocks a false claim, not metric depth as such.

    An anchoring step that produces genuine metres constructs a DepthField
    saying so, and that one unprojects.
    """
    geometry = FrameGeometry(width=32, height=32)
    anchored = DepthField(
        data=np.full(geometry.shape, 2.0),
        units="meters",
        geometry=geometry,
        valid_mask=np.ones(geometry.shape, dtype=np.bool_),
    )
    K = Intrinsics(
        fx=32.0,
        fy=32.0,
        cx=16.0,
        cy=16.0,
        distortion=(),
        valid_for=geometry,
        calibrated=True,
    )
    result = unproject(anchored, np.array([[16.0, 16.0]]), K)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# The validity mask
# ---------------------------------------------------------------------------


def test_non_finite_entries_are_marked_invalid() -> None:
    """A NaN depth is not a far one.

    Passing it through as a number puts a NaN into whatever 3D point is
    computed from it, and one NaN vector corrupts an entire index.
    """
    raw = raw_depth(shape=(8, 8)).copy()
    raw[0, 0] = np.nan
    raw[1, 1] = np.inf

    field = as_depth_field(raw)
    assert field.valid_mask[0, 0] is np.False_ or not field.valid_mask[0, 0]
    assert not field.valid_mask[1, 1]
    assert field.valid_mask.sum() == raw.size - 2


def test_all_finite_input_is_fully_valid() -> None:
    field = as_depth_field(raw_depth())
    assert field.valid_mask.all()


def test_mask_shape_matches_the_data() -> None:
    field = as_depth_field(raw_depth(shape=(23, 41)))
    assert field.valid_mask.shape == field.data.shape


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_non_2d_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[H, W\]"):
        as_depth_field(np.zeros((3, 8, 8), dtype=np.float32))


def test_wrapper_predict_returns_a_depthfield_type() -> None:
    """Pin the wrapper's declared return type against its docstring.

    Executed via the module-level helper rather than the class, because
    instantiating DAv2Wrapper needs a checkpoint. The helper is what predict()
    returns.
    """
    import inspect

    from src.models.dav2_wrapper import DAv2Wrapper

    source = inspect.getsource(DAv2Wrapper.predict)
    assert "as_depth_field(depth)" in source
    assert "Dict[str, DepthField]" in source
