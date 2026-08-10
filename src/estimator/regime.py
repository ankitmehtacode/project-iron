"""Motion-regime classification from exact GT (Day 21, redefined Day 23).

Partitions a track's frames into mutually exclusive kinematic regimes,
computed from ground truth position alone — never from the filter's own
estimate, so the partition used to evaluate the filter cannot be
contaminated by the thing it evaluates.

Day 23 — cessation covers the full transient, not just its onset
------------------------------------------------------------------
Day 21's own diagnosis (NEES climbing to ~800 immediately after a stop and
decaying back to nominal over ~10-14 frames) was traced from data that
this module was, until today, mislabeling: the pre-Day-23 ``cessation``
regime covered only the *anticipatory* frames — still GT-moving, with a
static frame imminent within :data:`ONSET_CESSATION_WINDOW_FRAMES` — and
the instant GT speed actually reached zero, every frame from then on was
immediately reclassified ``static``, including the ~10-14 frames where the
filter's covariance was demonstrably still catching up (Day 22 measured
those frames as the entire content of the "cessation" regime's near-empty
frame count: 0 on v3-indoor, 3 on v4.1-gate — one anticipatory frame per
track, nothing else). Splitting a stop event from its own recovery tail
hid the phenomenon from the metric built to see it. A cessation event now
means the FULL transient: the anticipatory window before the stop, plus a
recovery window after it, during which a track is GT-static but the event
that produced that stillness is still recent enough that a well-calibrated
filter's covariance has not necessarily caught up yet.

The recovery window's length is
:data:`~src.estimator.motion_model.PEDESTRIAN_STOP_DURATION_S` (Day 22's
already-declared ~1-second pedestrian-settling bound, reused rather than
fitted to Day 21's specific decay curve — see :func:`classify_track` for
the exact mechanics). At 12fps this is 12 frames, independently landing in
the same order of magnitude as Day 21's empirically observed 10-14 frame
decay — reported as corroboration of a constant chosen for an unrelated
reason, not as the basis for choosing it.

Five regimes, checked in two passes
-------------------------------------
A frame belongs to exactly one regime. First pass, by GT speed alone:

    static > onset > cessation (anticipatory) > maneuver > sustained

Second pass (Day 23): any frame the first pass called ``static`` is
reclassified ``cessation`` if it falls within the recovery window of the
moving-to-static transition that produced it — UNLESS that transition is
the track's own start (a track that is static from frame 0 carries no GT
evidence it ever stopped; see :func:`classify_track`'s docstring, the
same asymmetry :data:`ONSET_CESSATION_WINDOW_FRAMES`'s track-start
handling uses in the opposite direction for ``onset``).

``onset``/``cessation`` are checked before ``maneuver`` deliberately: the
transition from zero velocity to some heading is itself a heading "change"
by definition (there is no prior heading to compare against, or the prior
heading is undefined), so checking ``maneuver`` first would swallow every
onset frame into that bucket instead. ``maneuver`` is reserved for a
heading or speed change *while already moving*, which is a different
physical event from starting or stopping.

All thresholds below are declared, not measured — this project's synthetic
sets have exact, noiseless GT, so "static" only needs to guard against
floating-point jitter, not sensor noise. They are revisited once Site Zero
footage gives a real onset-duration / maneuver-frequency distribution to
fit against, exactly the caveat carried on every other declared-not-
measured constant in this codebase (e.g. ``PLACEHOLDER_DOWNSTREAM_COST_MS_PER_FRAME``).
"""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from src.estimator.motion_model import PEDESTRIAN_STOP_DURATION_S

FloatArray = npt.NDArray[np.float64]

MotionRegime = Literal["static", "onset", "sustained", "cessation", "maneuver"]
MOTION_REGIMES: tuple[MotionRegime, ...] = get_args(MotionRegime)

STATIC_SPEED_THRESHOLD_MPS = 0.02
"""Below this, GT speed is treated as zero. v3-indoor/v4.1-gate's GT is
exact (no sensor noise), so a genuinely static agent's finite-differenced
speed is exactly 0.0 — this floor exists for floating-point safety, not to
absorb noise."""

ONSET_CESSATION_WINDOW_FRAMES = 5
"""Frames on the MOVING side of a static<->moving boundary labeled onset/
cessation (anticipatory half) rather than sustained. ~0.4s at 12fps. Not
the same window as the cessation recovery half — see module docstring."""

MANEUVER_HEADING_THRESHOLD_DEG = 20.0
"""Per-frame heading change above this, while already moving (not onset/
cessation), is a maneuver."""

MANEUVER_SPEED_THRESHOLD_MPS_PER_FRAME = 0.5
"""Per-frame speed change above this, while already moving, is a
maneuver — roughly 6 m/s^2 at 12fps, well above a pedestrian's steady gait
variation."""


def _cessation_recovery_window_frames(dt_s: float) -> int:
    """Frames after a moving-to-static transition still counted as
    ``cessation`` rather than ``static`` (Day 23) — derived from
    :data:`~src.estimator.motion_model.PEDESTRIAN_STOP_DURATION_S`, the
    same already-declared pedestrian-settling bound Day 22's velocity
    floor used, not fitted to any specific decay curve. At least 1 frame
    regardless of ``dt_s`` so the recovery half is never a no-op."""
    return max(1, int(round(PEDESTRIAN_STOP_DURATION_S / dt_s)))


def classify_track(track_xyz: FloatArray, dt_s: float) -> tuple[MotionRegime, ...]:
    """One regime label per frame of ``track_xyz`` (``[T, 3]`` GT positions).

    Returns an empty tuple for an empty track. Frame 0's velocity is taken
    equal to frame 1's (there is no frame -1 to difference against) — the
    same convention :mod:`scripts.eval_estimator` already uses for GT
    velocity.
    """
    frames = track_xyz.shape[0]
    if frames == 0:
        return ()
    if frames == 1:
        return ("static",)

    velocity = np.zeros_like(track_xyz)
    velocity[1:] = (track_xyz[1:] - track_xyz[:-1]) / dt_s
    velocity[0] = velocity[1]
    speed = np.linalg.norm(velocity, axis=1)
    is_static = speed < STATIC_SPEED_THRESHOLD_MPS

    labels: list[MotionRegime] = ["sustained"] * frames
    for t in range(frames):
        if is_static[t]:
            labels[t] = "static"

    # onset: moving, and a static frame -- or the track's own start, which
    # carries no evidence the entity was NOT already moving before the clip
    # began, so it is treated the same as an unresolved static boundary --
    # lies within the trailing window.
    for t in range(frames):
        if labels[t] == "static":
            continue
        window_start = t - ONSET_CESSATION_WINDOW_FRAMES
        touches_track_start = window_start <= 0
        if touches_track_start or bool(np.any(is_static[max(0, window_start) : t])):
            labels[t] = "onset"

    # cessation, anticipatory half: moving, not onset, and a static frame
    # lies within the leading window (real GT evidence the agent is about
    # to stop).
    for t in range(frames):
        if labels[t] in ("static", "onset"):
            continue
        window_end = min(frames, t + ONSET_CESSATION_WINDOW_FRAMES + 1)
        if bool(np.any(is_static[t + 1 : window_end])):
            labels[t] = "cessation"

    # cessation, recovery half (Day 23): a frame the first pass called
    # "static" is reclassified "cessation" if it falls within the recovery
    # window of the moving-to-static transition that produced it. A static
    # RUN that begins at the track's own start (index 0) carries no GT
    # evidence the entity ever stopped -- there is no transition to recover
    # from, so it is left "static", the mirror image of onset's opposite
    # convention (which treats an unresolved track-start boundary as
    # evidence FOR a transition, because a track starting in motion could
    # only just have started; a track starting static could simply have
    # always been static).
    recovery_frames = _cessation_recovery_window_frames(dt_s)
    run_start: int | None = None
    for t in range(frames):
        if is_static[t]:
            if run_start is None:
                run_start = t
        else:
            run_start = None
            continue
        if (
            labels[t] == "static"
            and run_start is not None
            and run_start > 0
            and (t - run_start) < recovery_frames
        ):
            labels[t] = "cessation"

    # maneuver: moving, not onset/cessation, with a heading or speed jump
    # relative to the immediately preceding (also-moving) frame.
    for t in range(1, frames):
        if labels[t] != "sustained" or is_static[t - 1]:
            continue
        speed_delta = abs(float(speed[t] - speed[t - 1]))
        v_prev, v_curr = velocity[t - 1], velocity[t]
        denom = float(np.linalg.norm(v_prev) * np.linalg.norm(v_curr))
        cos_angle = float(np.dot(v_prev, v_curr) / denom) if denom > 1e-12 else 1.0
        heading_delta_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
        if (
            speed_delta > MANEUVER_SPEED_THRESHOLD_MPS_PER_FRAME
            or heading_delta_deg > MANEUVER_HEADING_THRESHOLD_DEG
        ):
            labels[t] = "maneuver"

    return tuple(labels)
