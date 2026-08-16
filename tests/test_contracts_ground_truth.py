"""The GT kinematic contracts, and the Day-29 axis bug they close.

Two structural claims are under test here, and "structural" in this repo
means impossible to violate rather than caught by review:

1. A GT array with no declared axis convention is unconstructable.
2. Mixing two conventions in one operation raises.

The rest of the file is the ordinary behaviour — differentiation
convention, vertical-component selection, the world-frame bridge — plus
one regression test that reconstructs the actual Day-29 defect and
confirms the type makes it unreachable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import (
    GENERATOR_AXES,
    AxisConventionMismatch,
    GroundTruthAxes,
    GtAccelerationTrack,
    GtPositionClip,
    GtPositionTrack,
    GtVelocityTrack,
    gt_position_track,
)
from src.model.world import UNREGISTERED

DT_S = 1.0 / 12.0


def _straight_line(frames: int = 10) -> np.ndarray:
    """A walker at constant height 0.86 m moving along the depth axis, in
    the generator's own convention — the shape every golden clip has."""
    track = np.zeros((frames, 3), dtype=np.float64)
    track[:, 1] = 0.86
    track[:, 2] = np.linspace(10.0, 4.0, frames)
    return track


# --------------------------------------------------------------------------
# Structural claim 1: no convention, no track
# --------------------------------------------------------------------------


def test_axes_is_required_with_no_default() -> None:
    """Constructing a GT track without stating its convention is a
    TypeError, not a track in some assumed default frame."""
    with pytest.raises(TypeError):
        GtPositionTrack(_straight_line())  # type: ignore[call-arg]


def test_a_non_enum_axes_value_is_rejected() -> None:
    """A bare integer is the shape the bug took (`VERTICAL_AXIS = 2`), so
    passing one must not be accepted as a convention."""
    with pytest.raises(AxisConventionMismatch):
        GtPositionTrack(_straight_line(), 2)  # type: ignore[arg-type]


def test_shape_must_be_t_by_3() -> None:
    with pytest.raises(ValueError, match=r"\[T, 3\]"):
        GtPositionTrack(np.zeros((10, 2)), GENERATOR_AXES)


# --------------------------------------------------------------------------
# Structural claim 2: mixing conventions raises
# --------------------------------------------------------------------------


def test_combining_two_conventions_raises() -> None:
    generator_frame = GtPositionTrack(_straight_line(), GroundTruthAxes.X_Y_UP_Z)
    world_frame = GtPositionTrack(_straight_line(), GroundTruthAxes.X_Y_Z_UP)
    with pytest.raises(AxisConventionMismatch, match="values_in"):
        generator_frame.combine(world_frame)


def test_combining_the_same_convention_returns_both_arrays() -> None:
    a = GtPositionTrack(_straight_line(), GENERATOR_AXES)
    b = GtPositionTrack(_straight_line() + 1.0, GENERATOR_AXES)
    left, right = a.combine(b)
    assert np.allclose(left, a.values)
    assert np.allclose(right, b.values)


def test_combining_different_quantities_raises_even_on_matching_axes() -> None:
    """Units are carried by the TYPE. A position and a velocity agreeing on
    axes is not agreement — mypy rejects this statically and `combine`
    rejects it at runtime, so neither a typed nor an untyped caller gets
    a metres/metres-per-second mix."""
    position = GtPositionTrack(_straight_line(), GENERATOR_AXES)
    velocity = position.differentiate(DT_S)
    with pytest.raises(TypeError, match="different physical quantities"):
        position.combine(velocity)


# --------------------------------------------------------------------------
# The Day-29 regression
# --------------------------------------------------------------------------


def test_vertical_component_uses_the_declared_axis_not_index_2() -> None:
    """The bug, reconstructed. `agent_xyz` index 2 is depth; index 1 is
    height. A track whose depth changes abruptly and whose height is
    constant must report ZERO vertical motion."""
    track = _straight_line()
    positions = GtPositionTrack(track, GENERATOR_AXES)

    assert np.allclose(positions.vertical_component(), 0.86)
    # And the quantity the Day-29 script actually got wrong:
    acceleration = positions.differentiate(DT_S).differentiate(DT_S)
    assert np.allclose(acceleration.vertical_component(), 0.0)
    # ...while the depth axis genuinely carries the motion.
    assert not np.allclose(positions.values[:, 2], positions.values[0, 2])


def test_reading_the_same_array_under_the_wrong_convention_disagrees() -> None:
    """The two conventions are not interchangeable, which is why an
    undeclared array is dangerous: the SAME bytes give a constant vertical
    under one and a moving one under the other."""
    track = _straight_line()
    correct = GtPositionTrack(track, GroundTruthAxes.X_Y_UP_Z)
    mislabelled = GtPositionTrack(track, GroundTruthAxes.X_Y_Z_UP)
    assert np.allclose(correct.vertical_component(), 0.86)
    assert not np.allclose(
        mislabelled.vertical_component(), mislabelled.vertical_component()[0]
    )


# --------------------------------------------------------------------------
# Differentiation convention
# --------------------------------------------------------------------------


def test_velocity_matches_the_projects_standing_convention() -> None:
    """Forward difference with frame 0 mirroring frame 1 — the same
    convention `src.estimator.regime.classify_track` uses, so a regime
    partition and an acceleration distribution computed here describe the
    same frames."""
    track = _straight_line()
    velocity = GtPositionTrack(track, GENERATOR_AXES).differentiate(DT_S)
    expected = np.zeros_like(track)
    expected[1:] = (track[1:] - track[:-1]) / DT_S
    expected[0] = expected[1]
    assert np.allclose(velocity.values, expected)


def test_double_differentiation_makes_index_1_identically_zero() -> None:
    """Not a property of the data — a property of the mirror. Recorded as
    a test because Day 29's script counted index 1 as a measurement, which
    put one guaranteed zero into every track's distribution."""
    rng = np.random.default_rng(30)
    track = rng.normal(size=(20, 3))
    acceleration = (
        GtPositionTrack(track, GENERATOR_AXES).differentiate(DT_S).differentiate(DT_S)
    )
    assert acceleration.values[1] == pytest.approx(np.zeros(3))
    assert not np.allclose(acceleration.values[2], 0.0)


def test_meaningful_drops_exactly_the_convention_artifact_frames() -> None:
    track = _straight_line(frames=10)
    acceleration = (
        GtPositionTrack(track, GENERATOR_AXES).differentiate(DT_S).differentiate(DT_S)
    )
    assert acceleration.frames == 10
    assert acceleration.meaningful().frames == 8
    assert acceleration.meaningful().axes is GENERATOR_AXES


def test_meaningful_on_a_too_short_track_is_empty_not_an_error() -> None:
    """A 2-frame track cannot support an acceleration. That is a fact
    about the data; a caller counting frames should see zero rather than
    an exception it has to translate back into one."""
    short = GtPositionTrack(_straight_line(frames=2), GENERATOR_AXES)
    acceleration = short.differentiate(DT_S).differentiate(DT_S)
    assert acceleration.meaningful().frames == 0


def test_non_positive_dt_raises() -> None:
    positions = GtPositionTrack(_straight_line(), GENERATOR_AXES)
    with pytest.raises(ValueError, match="dt_s must be positive"):
        positions.differentiate(0.0)


# --------------------------------------------------------------------------
# Conversions and invariants
# --------------------------------------------------------------------------


def test_values_in_the_same_convention_is_the_identity() -> None:
    positions = GtPositionTrack(_straight_line(), GENERATOR_AXES)
    assert positions.values_in(GENERATOR_AXES) is positions.values


def test_permutation_is_its_own_inverse() -> None:
    track = _straight_line()
    positions = GtPositionTrack(track, GroundTruthAxes.X_Y_UP_Z)
    world = positions.values_in(GroundTruthAxes.X_Y_Z_UP)
    back = GtPositionTrack(world, GroundTruthAxes.X_Y_Z_UP).values_in(
        GroundTruthAxes.X_Y_UP_Z
    )
    assert np.allclose(back, track)


def test_magnitude_is_invariant_under_the_permutation() -> None:
    """The reason several existing consumers survived the Day-29 bug
    untouched: anything that only takes a norm cannot be wrong about the
    axis order. Worth pinning, because it is the criterion the Objective-4
    audit uses to decide which raw-array boundaries are actually at risk."""
    track = _straight_line()
    a = GtPositionTrack(track, GroundTruthAxes.X_Y_UP_Z)
    b = GtPositionTrack(a.values_in(GroundTruthAxes.X_Y_Z_UP), GroundTruthAxes.X_Y_Z_UP)
    assert np.allclose(a.magnitude(), b.magnitude())


def test_vertical_and_ground_plane_indices_partition_the_axes() -> None:
    for axes in GroundTruthAxes:
        indices = {axes.vertical_index, *axes.ground_plane_indices}
        assert indices == {0, 1, 2}


def test_world_position_array_bridge_permutes_into_the_world_frame() -> None:
    """The one place the generator-frame -> world-frame permutation is
    written, and it carries `twin_rev` because `WorldPositionArray` will
    not accept a position whose revision is unstated."""
    positions = GtPositionTrack(_straight_line(), GENERATOR_AXES)
    world = positions.as_world_position_array(UNREGISTERED)
    assert world.twin_rev == UNREGISTERED
    # Height moved from index 1 to index 2.
    assert np.allclose(world.xyz_m[:, 2], 0.86)


def test_world_position_array_bridge_requires_a_twin_rev() -> None:
    positions = GtPositionTrack(_straight_line(), GENERATOR_AXES)
    with pytest.raises(TypeError):
        positions.as_world_position_array()  # type: ignore[call-arg]


def test_units_are_class_level_and_distinct_per_quantity() -> None:
    assert GtPositionTrack.UNITS == "m"
    assert GtVelocityTrack.UNITS == "m/s"
    assert GtAccelerationTrack.UNITS == "m/s^2"


def test_convenience_constructor_casts_float32_clips_to_float64() -> None:
    """Clips are stored float32. Differencing them in float32 is how a
    'zero' acceleration set reports 1e-4 — the cast belongs in one place."""
    track = _straight_line().astype(np.float32)
    positions = gt_position_track(track, GENERATOR_AXES)
    assert positions.values.dtype == np.float64


# --------------------------------------------------------------------------
# GtPositionClip — the [T, A, 3] shape, and the false declaration it replaced
# --------------------------------------------------------------------------


def _clip(frames: int = 6, agents: int = 2) -> np.ndarray:
    rng = np.random.default_rng(4)
    clip = rng.normal(size=(frames, agents, 3))
    clip[:, :, 1] = 0.86
    return clip


def test_clip_requires_both_an_axis_convention_and_a_twin_rev() -> None:
    """Both facts, or no object. The scorecard and Inspector previously
    got twin_rev from WorldPositionArray and the axis convention from
    nowhere — while that type's own module documents a +z-up frame that
    `agent_xyz` is not in."""
    with pytest.raises(TypeError):
        GtPositionClip(_clip(), GENERATOR_AXES)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        GtPositionClip(_clip(), twin_rev=UNREGISTERED)  # type: ignore[call-arg]


def test_clip_rejects_a_track_shaped_array() -> None:
    with pytest.raises(ValueError, match=r"\[T, A, 3\]"):
        GtPositionClip(_straight_line(), GENERATOR_AXES, UNREGISTERED)


def test_clip_track_carries_the_clips_convention_forward() -> None:
    clip = GtPositionClip(_clip(), GENERATOR_AXES, UNREGISTERED)
    assert clip.frames == 6 and clip.agents == 2
    track = clip.track(1)
    assert isinstance(track, GtPositionTrack)
    assert track.axes is GENERATOR_AXES
    assert np.allclose(track.vertical_component(), 0.86)


def test_clip_values_are_unchanged_by_the_wrapper() -> None:
    """The wrapper declares; it must not transform. Both consumers pass
    `.values` straight to code that applies the generator-frame
    extrinsics, so a silent permutation here would break them."""
    raw = _clip()
    assert np.allclose(GtPositionClip(raw, GENERATOR_AXES, UNREGISTERED).values, raw)
