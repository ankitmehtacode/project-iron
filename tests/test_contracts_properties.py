"""Property-based tests for the boundary contracts.

A contract without a property test is a comment. Example-based tests confirm
the cases the author already thought of; these search for the ones they did not.

Numeric ranges are bounded deliberately. Transforms are constrained to scales
in [0.05, 20] and offsets within +/-5000 px, which spans every resize, crop,
and letterbox this pipeline performs. Admitting scales of 1e300 would only
prove that float64 has finite precision, which is not a property of this code.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.contracts import (
    ALL_UNITS,
    AffineTransform,
    DepthField,
    FrameGeometry,
    GeometryMismatch,
    Intrinsics,
    PatchTokens,
    TemporalSpan,
    Units,
    UnitsError,
    unproject,
)

EXAMPLES = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

scales = st.floats(
    min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False
)
signed_scales = st.one_of(scales, scales.map(lambda s: -s))
offsets = st.floats(
    min_value=-5000.0, max_value=5000.0, allow_nan=False, allow_infinity=False
)
pixel_coords = st.floats(
    min_value=-2000.0, max_value=4000.0, allow_nan=False, allow_infinity=False
)
dimensions = st.integers(min_value=1, max_value=8192)


@st.composite
def affine_transforms(draw: st.DrawFn) -> AffineTransform:
    return AffineTransform(
        scale_x=draw(signed_scales),
        scale_y=draw(signed_scales),
        offset_x=draw(offsets),
        offset_y=draw(offsets),
    )


@st.composite
def frame_geometries(draw: st.DrawFn) -> FrameGeometry:
    return FrameGeometry(width=draw(dimensions), height=draw(dimensions))


@st.composite
def point_arrays(draw: st.DrawFn) -> np.ndarray:
    count = draw(st.integers(min_value=1, max_value=32))
    values = draw(st.lists(pixel_coords, min_size=count * 2, max_size=count * 2))
    return np.array(values, dtype=np.float64).reshape(count, 2)


# ---------------------------------------------------------------------------
# (a) A transform composed with its inverse is the identity.
# ---------------------------------------------------------------------------


@EXAMPLES
@given(transform=affine_transforms())
def test_compose_with_inverse_is_identity(transform: AffineTransform) -> None:
    identity = transform.compose(transform.inverse())
    assert identity.scale_x == pytest.approx(1.0, rel=1e-9, abs=1e-12)
    assert identity.scale_y == pytest.approx(1.0, rel=1e-9, abs=1e-12)
    assert identity.offset_x == pytest.approx(0.0, abs=1e-6)
    assert identity.offset_y == pytest.approx(0.0, abs=1e-6)


@EXAMPLES
@given(transform=affine_transforms())
def test_inverse_of_inverse_is_original(transform: AffineTransform) -> None:
    restored = transform.inverse().inverse()
    assert restored.scale_x == pytest.approx(transform.scale_x, rel=1e-9)
    assert restored.scale_y == pytest.approx(transform.scale_y, rel=1e-9)
    assert restored.offset_x == pytest.approx(transform.offset_x, rel=1e-9, abs=1e-9)
    assert restored.offset_y == pytest.approx(transform.offset_y, rel=1e-9, abs=1e-9)


@EXAMPLES
@given(transform=affine_transforms(), points=point_arrays())
def test_apply_then_unapply_round_trips(
    transform: AffineTransform, points: np.ndarray
) -> None:
    restored = transform.inverse().apply(transform.apply(points))
    np.testing.assert_allclose(restored, points, rtol=1e-9, atol=1e-6)


# ---------------------------------------------------------------------------
# (b) Composition chains round-trip points to sub-hundredth-pixel accuracy.
# ---------------------------------------------------------------------------


@EXAMPLES
@given(
    chain=st.lists(affine_transforms(), min_size=1, max_size=6),
    points=point_arrays(),
)
def test_composition_chain_round_trips_within_a_hundredth_of_a_pixel(
    chain: list[AffineTransform], points: np.ndarray
) -> None:
    """A chain of stage transforms must be reversible to well under a pixel.

    This is the property that lets a coordinate travel through resize, crop,
    and pad stages and come home to the canonical frame. 0.01 px is two orders
    of magnitude below the half-pixel errors that a forgotten centre-offset
    correction produces, so this test distinguishes float noise from a real
    convention bug.
    """
    combined = chain[0]
    for transform in chain[1:]:
        combined = combined.compose(transform)

    forward = combined.apply(points)
    restored = combined.inverse().apply(forward)
    np.testing.assert_allclose(restored, points, rtol=0, atol=0.01)


@EXAMPLES
@given(outer=affine_transforms(), inner=affine_transforms(), points=point_arrays())
def test_compose_matches_sequential_application(
    outer: AffineTransform, inner: AffineTransform, points: np.ndarray
) -> None:
    """``a.compose(b)`` must equal "apply b, then a".

    Pins the documented order. Reversing it produces a transform that is still
    correct whenever the two scales happen to match, which is exactly when
    nobody notices.
    """
    np.testing.assert_allclose(
        outer.compose(inner).apply(points),
        outer.apply(inner.apply(points)),
        rtol=1e-9,
        atol=1e-9,
    )


@EXAMPLES
@given(source=frame_geometries(), target=frame_geometries(), points=point_arrays())
def test_resize_transform_round_trips(
    source: FrameGeometry, target: FrameGeometry, points: np.ndarray
) -> None:
    transform = AffineTransform.for_resize(source, target)
    restored = transform.inverse().apply(transform.apply(points))
    np.testing.assert_allclose(restored, points, rtol=1e-9, atol=1e-6)


# ---------------------------------------------------------------------------
# (c) Intrinsics survive a rescale round trip.
# ---------------------------------------------------------------------------


@st.composite
def intrinsics(draw: st.DrawFn) -> Intrinsics:
    geometry = draw(frame_geometries())
    return Intrinsics(
        fx=draw(st.floats(min_value=1.0, max_value=1e4, allow_nan=False)),
        fy=draw(st.floats(min_value=1.0, max_value=1e4, allow_nan=False)),
        cx=draw(st.floats(min_value=-1e3, max_value=1e4, allow_nan=False)),
        cy=draw(st.floats(min_value=-1e3, max_value=1e4, allow_nan=False)),
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        valid_for=geometry,
        calibrated=True,
    )


@EXAMPLES
@given(original=intrinsics(), other=frame_geometries())
def test_intrinsics_rescale_round_trips(
    original: Intrinsics, other: FrameGeometry
) -> None:
    restored = original.rescaled_to(other).rescaled_to(original.valid_for)
    assert restored.fx == pytest.approx(original.fx, rel=1e-9)
    assert restored.fy == pytest.approx(original.fy, rel=1e-9)
    assert restored.cx == pytest.approx(original.cx, rel=1e-9, abs=1e-9)
    assert restored.cy == pytest.approx(original.cy, rel=1e-9, abs=1e-9)
    assert restored.valid_for == original.valid_for


@EXAMPLES
@given(original=intrinsics(), other=frame_geometries())
def test_rescale_leaves_distortion_untouched(
    original: Intrinsics, other: FrameGeometry
) -> None:
    """Distortion coefficients are dimensionless and must not scale.

    They are functions of normalised image coordinates in the standard
    radial-tangential model. Scaling them with the raster would be precisely
    the units error this package exists to prevent.
    """
    assert original.rescaled_to(other).distortion == original.distortion


@EXAMPLES
@given(original=intrinsics(), other=frame_geometries())
def test_rescale_preserves_normalised_principal_point(
    original: Intrinsics, other: FrameGeometry
) -> None:
    """cx/width is invariant under rescale — the optical centre does not move."""
    rescaled = original.rescaled_to(other)
    assert rescaled.cx / other.width == pytest.approx(
        original.cx / original.valid_for.width, rel=1e-9, abs=1e-12
    )


# ---------------------------------------------------------------------------
# (d) unproject rejects every non-metric unit.
# ---------------------------------------------------------------------------


def _depth_field(units: Units, size: int = 8) -> DepthField:
    geometry = FrameGeometry(width=size, height=size)
    return DepthField(
        data=np.full(geometry.shape, 3.0, dtype=np.float64),
        units=units,
        geometry=geometry,
        valid_mask=np.ones(geometry.shape, dtype=np.bool_),
    )


def _intrinsics_for(geometry: FrameGeometry) -> Intrinsics:
    return Intrinsics(
        fx=float(geometry.width),
        fy=float(geometry.height),
        cx=geometry.width / 2.0,
        cy=geometry.height / 2.0,
        distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        valid_for=geometry,
        calibrated=True,
    )


NON_METRIC_UNITS = tuple(u for u in ALL_UNITS if u != "meters")


@pytest.mark.parametrize("units", NON_METRIC_UNITS)
def test_unproject_raises_on_every_non_metric_unit(units: Units) -> None:
    """Every non-metric unit must raise — no exceptions, no auto-conversion."""
    depth = _depth_field(units)
    points = np.array([[4.0, 4.0]], dtype=np.float64)
    with pytest.raises(UnitsError):
        unproject(depth, points, _intrinsics_for(depth.geometry))


def test_non_metric_units_list_is_not_accidentally_empty() -> None:
    """Guard the parametrised test above against silently testing nothing."""
    assert len(NON_METRIC_UNITS) >= 2
    assert "meters" not in NON_METRIC_UNITS


@EXAMPLES
@given(points=point_arrays())
def test_unproject_accepts_metric_depth(points: np.ndarray) -> None:
    depth = _depth_field("meters", size=64)
    result = unproject(depth, points, _intrinsics_for(depth.geometry))
    assert result.shape == (points.shape[0], 3)


def test_unproject_raises_on_geometry_mismatch() -> None:
    depth = _depth_field("meters", size=8)
    stale = _intrinsics_for(FrameGeometry(width=16, height=16))
    with pytest.raises(GeometryMismatch):
        unproject(depth, np.array([[4.0, 4.0]]), stale)


def test_unproject_is_correct_for_a_hand_checked_point() -> None:
    """Pin the pinhole arithmetic itself, not just the guard rails."""
    geometry = FrameGeometry(width=100, height=100)
    depth = DepthField(
        data=np.full(geometry.shape, 2.0, dtype=np.float64),
        units="meters",
        geometry=geometry,
        valid_mask=np.ones(geometry.shape, dtype=np.bool_),
    )
    K = Intrinsics(
        fx=50.0,
        fy=50.0,
        cx=50.0,
        cy=50.0,
        distortion=(0.0,),
        valid_for=geometry,
        calibrated=True,
    )
    # 10 px right and 20 px below the principal point, 2 m away.
    result = unproject(depth, np.array([[60.0, 70.0]]), K)
    np.testing.assert_allclose(result, [[0.4, 0.8, 2.0]], rtol=1e-12)


def test_unproject_marks_invalid_pixels_nan_not_zero() -> None:
    """Unusable depth must be NaN; zeros would masquerade as real points."""
    geometry = FrameGeometry(width=8, height=8)
    mask = np.ones(geometry.shape, dtype=np.bool_)
    mask[0, 0] = False
    depth = DepthField(
        data=np.full(geometry.shape, 3.0, dtype=np.float64),
        units="meters",
        geometry=geometry,
        valid_mask=mask,
    )
    result = unproject(
        depth, np.array([[0.5, 0.5], [4.5, 4.5]]), _intrinsics_for(geometry)
    )
    assert np.all(np.isnan(result[0]))
    assert np.all(np.isfinite(result[1]))


def test_unproject_marks_out_of_bounds_nan() -> None:
    """Out-of-frame track coordinates must not read a clamped border depth."""
    geometry = FrameGeometry(width=8, height=8)
    depth = _depth_field("meters", size=8)
    result = unproject(
        depth,
        np.array([[-5.0, 4.0], [400.0, 4.0], [4.0, 4.0]]),
        _intrinsics_for(geometry),
    )
    assert np.all(np.isnan(result[0]))
    assert np.all(np.isnan(result[1]))
    assert np.all(np.isfinite(result[2]))


# ---------------------------------------------------------------------------
# (e) PatchTokens rejects mismatched shapes.
# ---------------------------------------------------------------------------


@st.composite
def token_parameters(draw: st.DrawFn) -> tuple[int, int, int, int]:
    tubelet = draw(st.integers(min_value=1, max_value=8))
    slots = draw(st.integers(min_value=1, max_value=8))
    rows = draw(st.integers(min_value=1, max_value=16))
    cols = draw(st.integers(min_value=1, max_value=16))
    return tubelet, slots, rows, cols


@EXAMPLES
@given(params=token_parameters())
def test_patch_tokens_accepts_consistent_shapes(
    params: tuple[int, int, int, int],
) -> None:
    tubelet, slots, rows, cols = params
    frames = slots * tubelet
    span = TemporalSpan(
        start_ts_ns=0, end_ts_ns=1_000_000, frames_covered=frames, tubelet=tubelet
    )
    tokens = PatchTokens(
        data=np.zeros((slots, rows * cols, 8), dtype=np.float64),
        span=span,
        grid=(rows, cols),
        geometry=FrameGeometry(width=224, height=224),
        encoder_sha="a" * 8,
    )
    assert tokens.n_temporal == slots
    assert tokens.n_spatial == rows * cols
    assert tokens.dim == 8


@EXAMPLES
@given(params=token_parameters(), wrong_delta=st.integers(min_value=1, max_value=5))
def test_patch_tokens_rejects_wrong_temporal_count(
    params: tuple[int, int, int, int], wrong_delta: int
) -> None:
    """The assertion that permanently kills the temporal off-by-2 bug class."""
    tubelet, slots, rows, cols = params
    frames = slots * tubelet
    span = TemporalSpan(
        start_ts_ns=0, end_ts_ns=1_000_000, frames_covered=frames, tubelet=tubelet
    )
    with pytest.raises(ValueError, match="temporal mismatch"):
        PatchTokens(
            data=np.zeros((slots + wrong_delta, rows * cols, 8), dtype=np.float64),
            span=span,
            grid=(rows, cols),
            geometry=FrameGeometry(width=224, height=224),
            encoder_sha="a" * 8,
        )


@EXAMPLES
@given(params=token_parameters(), wrong_delta=st.integers(min_value=1, max_value=5))
def test_patch_tokens_rejects_wrong_spatial_count(
    params: tuple[int, int, int, int], wrong_delta: int
) -> None:
    tubelet, slots, rows, cols = params
    frames = slots * tubelet
    span = TemporalSpan(
        start_ts_ns=0, end_ts_ns=1_000_000, frames_covered=frames, tubelet=tubelet
    )
    with pytest.raises(ValueError, match="spatial mismatch"):
        PatchTokens(
            data=np.zeros((slots, rows * cols + wrong_delta, 8), dtype=np.float64),
            span=span,
            grid=(rows, cols),
            geometry=FrameGeometry(width=224, height=224),
            encoder_sha="a" * 8,
        )


def test_patch_tokens_rejects_the_repos_documented_shape() -> None:
    """The exact claim in semantic_extractor.py must be rejected.

    That module documents V-JEPA2 output as ``[B, T*196, 1024]`` for T=4, i.e.
    4 temporal slots for a 4-frame clip. With tubelet 2 there are 2. This is
    the concrete instance of the bug, pinned so the contract cannot be relaxed
    back into accepting it.
    """
    span = TemporalSpan(
        start_ts_ns=0, end_ts_ns=133_000_000, frames_covered=4, tubelet=2
    )
    assert span.n_temporal == 2
    with pytest.raises(ValueError, match="assumed tubelet=1"):
        PatchTokens(
            data=np.zeros((4, 196, 1024), dtype=np.float64),
            span=span,
            grid=(14, 14),
            geometry=FrameGeometry(width=224, height=224),
            encoder_sha="deadbeef",
        )


def test_patch_tokens_requires_an_encoder_sha() -> None:
    span = TemporalSpan(start_ts_ns=0, end_ts_ns=1000, frames_covered=2, tubelet=2)
    with pytest.raises(ValueError, match="encoder_sha"):
        PatchTokens(
            data=np.zeros((1, 4, 8), dtype=np.float64),
            span=span,
            grid=(2, 2),
            geometry=FrameGeometry(width=32, height=32),
            encoder_sha="",
        )


@EXAMPLES
@given(params=token_parameters())
def test_slot_and_frame_mappings_are_consistent(
    params: tuple[int, int, int, int],
) -> None:
    """Every represented frame maps into the slot that claims to cover it."""
    tubelet, slots, rows, cols = params
    frames = slots * tubelet
    span = TemporalSpan(
        start_ts_ns=0, end_ts_ns=1_000_000, frames_covered=frames, tubelet=tubelet
    )
    tokens = PatchTokens(
        data=np.zeros((slots, rows * cols, 4), dtype=np.float64),
        span=span,
        grid=(rows, cols),
        geometry=FrameGeometry(width=224, height=224),
        encoder_sha="a" * 8,
    )
    for slot in range(tokens.n_temporal):
        for frame in tokens.frames_for_slot(slot):
            assert tokens.slot_for_frame(frame) == slot


# ---------------------------------------------------------------------------
# DepthField shape agreement.
# ---------------------------------------------------------------------------


@EXAMPLES
@given(geometry=frame_geometries().filter(lambda g: g.width * g.height <= 4096))
def test_depth_field_accepts_matching_shapes(geometry: FrameGeometry) -> None:
    field = DepthField(
        data=np.zeros(geometry.shape, dtype=np.float64),
        units="disparity_rel",
        geometry=geometry,
        valid_mask=np.ones(geometry.shape, dtype=np.bool_),
    )
    assert field.is_metric is False


def test_depth_field_rejects_shape_disagreement() -> None:
    geometry = FrameGeometry(width=10, height=5)
    with pytest.raises(ValueError, match="does not match geometry"):
        DepthField(
            data=np.zeros((5, 11), dtype=np.float64),
            units="meters",
            geometry=geometry,
            valid_mask=np.ones((5, 11), dtype=np.bool_),
        )


def test_depth_field_rejects_mask_disagreement() -> None:
    geometry = FrameGeometry(width=10, height=5)
    with pytest.raises(ValueError, match="valid_mask shape"):
        DepthField(
            data=np.zeros(geometry.shape, dtype=np.float64),
            units="meters",
            geometry=geometry,
            valid_mask=np.ones((4, 10), dtype=np.bool_),
        )


def test_depth_field_rejects_unknown_units() -> None:
    geometry = FrameGeometry(width=4, height=4)
    with pytest.raises(UnitsError):
        DepthField(
            data=np.zeros(geometry.shape, dtype=np.float64),
            units="furlongs",  # type: ignore[arg-type]
            geometry=geometry,
            valid_mask=np.ones(geometry.shape, dtype=np.bool_),
        )


# ---------------------------------------------------------------------------
# Guard rails on the primitives themselves.
# ---------------------------------------------------------------------------


def test_affine_rejects_zero_scale() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        AffineTransform(scale_x=0.0, scale_y=1.0, offset_x=0.0, offset_y=0.0)


def test_frame_geometry_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        FrameGeometry(width=0, height=10)


def test_frame_geometry_equality_is_by_value() -> None:
    assert FrameGeometry(1920, 1080) == FrameGeometry(1920, 1080)
    assert FrameGeometry(1920, 1080) != FrameGeometry(1080, 1920)


def test_temporal_span_rejects_backwards_time() -> None:
    with pytest.raises(ValueError, match="advance in time"):
        TemporalSpan(start_ts_ns=100, end_ts_ns=100, frames_covered=2, tubelet=2)


def test_temporal_span_reports_unrepresented_tail() -> None:
    """A frame count that is not a multiple of the tubelet drops its tail."""
    span = TemporalSpan(start_ts_ns=0, end_ts_ns=1000, frames_covered=5, tubelet=2)
    assert span.n_temporal == 2
    assert span.frames_unrepresented == 1


def test_intrinsics_requires_an_explicit_calibration_claim() -> None:
    """Day 31, Objective 4: `calibrated` has no default. It used to
    default to True, so a caller who said nothing asserted a calibration
    it had not been shown to have -- the default itself was the
    unverified claim. There is no correct default for a fact about
    provenance, so there is none."""
    with pytest.raises(TypeError):
        Intrinsics(  # type: ignore[call-arg]
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            distortion=(),
            valid_for=FrameGeometry(4, 4),
        )


def test_intrinsics_rejects_nonpositive_focal_length() -> None:
    with pytest.raises(ValueError, match="Focal lengths"):
        Intrinsics(
            fx=0.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            distortion=(),
            valid_for=FrameGeometry(4, 4),
            calibrated=True,
        )
