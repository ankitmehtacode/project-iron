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
    # Some frames just before stopping are cessation (anticipatory half).
    assert any(label == "cessation" for label in labels[max(0, stop_at - 4) : stop_at])
    # Day 23: the recovery half -- frames right after the stop are ALSO
    # cessation, not immediately "static". This track is too short to run
    # past the recovery window (see the settling test below for that).
    assert labels[stop_at + 1] == "cessation"


def test_cessation_recovery_window_eventually_settles_to_static() -> None:
    """Day 23: the recovery half of cessation is a WINDOW, not forever --
    a track long enough to run past PEDESTRIAN_STOP_DURATION_S worth of
    frames after the stop must return to 'static'."""
    from src.estimator.regime import _cessation_recovery_window_frames

    recovery_frames = _cessation_recovery_window_frames(DT_S)
    stop_at = 10
    frames = stop_at + recovery_frames + 15
    track = np.zeros((frames, 3))
    for t in range(frames):
        track[t, 0] = min(t, stop_at) * 0.5
    labels = classify_track(track, DT_S)

    first_static_after_stop = stop_at + 1
    # Immediately after the stop, still cessation (recovery, not settled).
    assert labels[first_static_after_stop] == "cessation"
    # Well past the recovery window, back to static.
    settled_index = first_static_after_stop + recovery_frames + 5
    assert labels[settled_index] == "static"


def test_cessation_recovery_does_not_apply_to_a_track_static_from_the_start() -> None:
    """A track that is static from frame 0 carries no GT evidence it ever
    stopped -- it must stay 'static' throughout, never 'cessation'."""
    track = np.zeros((30, 3))
    labels = classify_track(track, DT_S)
    assert labels == ("static",) * 30
    assert "cessation" not in labels


def test_stop_then_restart_labels_both_transients() -> None:
    """A stop-then-restart sequence (Day 23, v5-cessation's own scenario
    shape): both the cessation recovery after the first stop and the onset
    after restarting must be labeled, not swallowed into one or the other."""
    walk1, stop_len, walk2 = 10, 20, 10
    frames = walk1 + stop_len + walk2
    track = np.zeros((frames, 3))
    pos = 0.0
    for t in range(frames):
        if t < walk1:
            pos = t * 0.5
        elif t < walk1 + stop_len:
            pos = walk1 * 0.5  # held
        else:
            pos = walk1 * 0.5 + (t - walk1 - stop_len) * 0.5
        track[t, 0] = pos
    labels = classify_track(track, DT_S)
    # Recovery cessation right after the first stop.
    assert labels[walk1 + 1] == "cessation"
    # Onset right after restarting (position changes again at walk1+stop_len,
    # finite difference shows it one frame later).
    restart_index = walk1 + stop_len + 1
    assert labels[restart_index] == "onset"


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
