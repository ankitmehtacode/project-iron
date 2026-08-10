"""Day 21, Objective 2 — GT motion-regime classification.

Constructed tracks with a known-by-design regime sequence, so each
assertion checks the classifier against ground truth we wrote, not against
itself.
"""

from __future__ import annotations

import numpy as np

from src.estimator.regime import classify_track

FPS = 12.0
DT_S = 1.0 / FPS


def test_empty_track_returns_empty() -> None:
    assert classify_track(np.zeros((0, 3)), DT_S) == ()


def test_single_frame_is_static() -> None:
    assert classify_track(np.zeros((1, 3)), DT_S) == ("static",)


def test_fully_static_track() -> None:
    track = np.zeros((10, 3))
    labels = classify_track(track, DT_S)
    assert labels == ("static",) * 10


def test_immediate_constant_velocity_from_frame_zero_is_onset_then_sustained() -> None:
    """No prior static frame exists to define an onset boundary against --
    the track just starts moving. Frame 0 is treated as onset (can't rule
    out it just started); the trailing window after it is onset too, then
    sustained."""
    frames = 20
    track = np.zeros((frames, 3))
    track[:, 0] = np.arange(frames) * 0.5  # constant velocity, no stop
    labels = classify_track(track, DT_S)
    assert labels[0] == "onset"
    assert all(label == "onset" for label in labels[:6])
    assert all(label == "sustained" for label in labels[10:])


def test_static_then_walk_produces_onset_window() -> None:
    frames = 20
    static_frames = 8
    track = np.zeros((frames, 3))
    for t in range(static_frames, frames):
        track[t, 0] = (t - static_frames) * 0.5
    labels = classify_track(track, DT_S)
    # Position first changes at index static_frames, so the finite
    # difference first shows nonzero velocity at static_frames + 1.
    first_moving_index = static_frames + 1
    assert labels[:first_moving_index] == ("static",) * first_moving_index
    onset_region = labels[first_moving_index : first_moving_index + 5]
    assert all(label == "onset" for label in onset_region)


def test_walk_then_stop_produces_cessation_window() -> None:
    frames = 20
    stop_at = 12
    track = np.zeros((frames, 3))
    for t in range(frames):
        track[t, 0] = min(t, stop_at) * 0.5
    labels = classify_track(track, DT_S)
    # Frames well after stopping are static.
    assert labels[stop_at + 2 :] == ("static",) * len(labels[stop_at + 2 :])
    # Some frames just before stopping are cessation (not sustained/onset).
    assert any(label == "cessation" for label in labels[max(0, stop_at - 4) : stop_at])


def test_straight_line_constant_speed_has_no_maneuver() -> None:
    """v3-indoor's actual agent model: a straight line, one constant
    speed. Never a maneuver by construction."""
    frames = 30
    track = np.zeros((frames, 3))
    track[:, 0] = np.arange(frames) * 0.3
    labels = classify_track(track, DT_S)
    assert "maneuver" not in labels


def test_sharp_heading_change_mid_motion_is_a_maneuver() -> None:
    frames = 30
    track = np.zeros((frames, 3))
    # Move along +x for a while (well past onset), then turn 90 degrees
    # onto +y at constant speed.
    turn_at = 15
    for t in range(frames):
        if t < turn_at:
            track[t, 0] = t * 0.3
        else:
            track[t, 0] = turn_at * 0.3
            track[t, 1] = (t - turn_at) * 0.3
    labels = classify_track(track, DT_S)
    # The position "kink" is at turn_at; the finite-differenced velocity
    # first reflects the new heading one frame later.
    assert labels[turn_at + 1] == "maneuver"


def test_sudden_speed_change_mid_motion_is_a_maneuver() -> None:
    frames = 30
    track = np.zeros((frames, 3))
    speed_up_at = 15
    pos = 0.0
    for t in range(frames):
        track[t, 0] = pos
        step = 0.1 if t < speed_up_at else 2.0
        pos += step
    labels = classify_track(track, DT_S)
    # Same off-by-one as the heading-change case: the step size change
    # applied at speed_up_at first shows up in the t+1 finite difference.
    assert labels[speed_up_at + 1] == "maneuver"


def test_every_label_is_a_valid_regime() -> None:
    from src.estimator.regime import MOTION_REGIMES

    rng = np.random.default_rng(20260731)
    track = np.cumsum(rng.normal(0, 0.3, size=(40, 3)), axis=0)
    labels = classify_track(track, DT_S)
    assert all(label in MOTION_REGIMES for label in labels)
    assert len(labels) == 40
