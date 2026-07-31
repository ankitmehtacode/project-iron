"""Stage 0: the motion gate.

This is the only stage that runs on every frame of every camera, so it is the
one place where cost is not negotiable. Everything downstream is gated on its
answer, which makes it both the cheapest stage and the most consequential: a
gate that wakes on noise destroys the idle budget, and a gate that sleeps
through a real event loses the event entirely. The asymmetry matters — waking
unnecessarily costs money, failing to wake costs evidence — so the tuning bias
is deliberately toward waking.

Hysteresis
----------
The gate stays awake for a configurable number of frames after motion stops.
Without it, a person pausing mid-stride flips the whole cascade off and on
several times a second. Each transition re-enters the detector, which pays
model warm-up repeatedly, so flapping costs more than simply staying awake —
and it fragments what should be one continuous track into several, which is a
correctness problem, not just a performance one.

Backends
--------
MOG2 background subtraction when OpenCV is available, adaptive frame
differencing otherwise. The fallback exists because ``cv2`` is a heavy optional
dependency and the gate is the piece most likely to be exercised on a minimal
install; it is genuinely less accurate against gradual lighting change, and
:attr:`MotionGate.backend` reports which one is running so a benchmark never
silently compares the two.

Measured cost, and the open problem
-----------------------------------
Measured 2026-07-31 with ``scripts/cascade_bench.py`` on the development
machine (12 logical cores), 12 fps, as a share of one core per camera:

======================  ==========  ==========  ========
Resolution / backend    p50 / frame  % of core  wake rate
======================  ==========  ==========  ========
1280x720 mog2              24.6 ms     29.5%       35%
1280x720 frame-diff        20.3 ms     24.4%      5.4%
640x360  mog2               6.0 ms      7.2%       35%
320x180  mog2               2.0 ms      2.3%       35%
320x180  frame-diff         1.1 ms      1.3%       35%
======================  ==========  ==========  ========

**At full resolution this stage misses the Tier-1 idle budget of 3% of one
core by roughly ten times.** Running it at 320x180 meets the budget and, on
this sequence, detects exactly the same frames — motion gating does not need
the resolution the detector needs. That change is not made here because it
alters gate behaviour and belongs in a measured change of its own, on real
footage rather than a synthetic blob.

Two things in that table are worth carrying forward. Frame differencing at
720p wakes on only 5.4% of frames against MOG2's 35%: it sees only the moving
*edges* of the blob, so the foreground fraction falls under the threshold.
:attr:`MotionGateConfig.min_foreground_fraction` is therefore not portable
across resolutions or backends, and re-tuning is required whenever either
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from src.cascade.stage import FrameBatch, StageContext, StageOutput

GrayFrame = npt.NDArray[np.uint8]
MaskFrame = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class MotionGateConfig:
    """Tuning for the motion gate.

    Attributes:
        min_foreground_fraction: Share of pixels that must be foreground before
            the frame counts as motion. Sensor noise moves a small fraction of
            pixels constantly; this is the floor that separates a scene from a
            grain pattern.
        stay_awake_frames: Frames to remain awake after motion stops. See the
            module docstring on why flapping is worse than staying on.
        diff_threshold: Per-pixel intensity change counted as movement by the
            frame-difference fallback.
        warmup_frames: Frames the background model absorbs before its output is
            trusted. Every pixel looks like foreground to an unbuilt model, so
            without this the gate wakes the whole cascade on startup.
    """

    min_foreground_fraction: float = 0.002
    stay_awake_frames: int = 12
    diff_threshold: int = 25
    warmup_frames: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_foreground_fraction <= 1.0:
            raise ValueError(
                "min_foreground_fraction is a fraction of the frame and must "
                f"lie in [0, 1], got {self.min_foreground_fraction}"
            )
        if self.stay_awake_frames < 0:
            raise ValueError("stay_awake_frames must not be negative")
        if not 0 <= self.diff_threshold <= 255:
            raise ValueError("diff_threshold must lie in [0, 255]")


class MotionBackend(Protocol):
    """A foreground estimator."""

    @property
    def name(self) -> str:
        ...

    def foreground_fraction(self, gray: GrayFrame) -> float:
        """Return the share of pixels judged to be foreground, in [0, 1]."""
        ...


class FrameDifferenceBackend:
    """Absolute difference against the previous frame.

    Deliberately simple. It cannot distinguish a slow global brightness change
    from real movement, which is exactly why MOG2 is preferred where available.
    """

    def __init__(self, threshold: int) -> None:
        self._threshold = threshold
        self._previous: GrayFrame | None = None

    @property
    def name(self) -> str:
        return "frame-difference"

    def foreground_fraction(self, gray: GrayFrame) -> float:
        previous = self._previous
        self._previous = gray
        if previous is None or previous.shape != gray.shape:
            # No baseline yet: report no motion rather than guessing. Claiming
            # motion here would wake the cascade on the first frame of every
            # stream, on every reconnect.
            return 0.0
        delta = np.abs(gray.astype(np.int16) - previous.astype(np.int16))
        return float(np.count_nonzero(delta > self._threshold) / delta.size)


class Mog2Backend:
    """OpenCV MOG2 background subtraction.

    Shadow pixels are excluded: MOG2 marks them 127 rather than 255, and
    counting them as foreground makes the gate fire on a cloud passing the
    window.
    """

    def __init__(self) -> None:
        import cv2

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=250, varThreshold=16, detectShadows=True
        )

    @property
    def name(self) -> str:
        return "mog2"

    def foreground_fraction(self, gray: GrayFrame) -> float:
        mask = self._subtractor.apply(gray)
        return float(np.count_nonzero(mask >= 255) / mask.size)


def _to_gray(frame: npt.NDArray[Any]) -> GrayFrame:
    """Reduce a frame to single-channel uint8.

    Uses a plain channel mean rather than luma weights. The gate only ever
    compares a frame against its own history, so the exact colour transform is
    irrelevant, and the mean is both cheaper and independent of channel order —
    which matters because BGR/RGB confusion is otherwise silent here.
    """
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array.mean(axis=2)
    return array.astype(np.uint8)


class MotionGate:
    """Stage 0. Decides whether anything downstream runs at all."""

    def __init__(
        self,
        config: MotionGateConfig | None = None,
        backend: MotionBackend | None = None,
    ) -> None:
        self._config = config or MotionGateConfig()
        self._backend = backend or self._default_backend()
        self._frames_seen = 0
        self._awake_remaining = 0
        self._last_fraction = 0.0

    def _default_backend(self) -> MotionBackend:
        try:
            return Mog2Backend()
        except (ImportError, AttributeError):
            return FrameDifferenceBackend(self._config.diff_threshold)

    @property
    def name(self) -> str:
        return "motion_gate"

    @property
    def backend(self) -> str:
        """Which estimator is running. Reported so benchmarks stay comparable."""
        return self._backend.name

    @property
    def awake_remaining(self) -> int:
        """Frames the gate will stay awake for even without further motion."""
        return self._awake_remaining

    def process(self, frame_batch: FrameBatch, ctx: StageContext) -> StageOutput:
        """Judge whether this batch contains motion.

        Args:
            frame_batch: ``[N, H, W, C]`` or ``[N, H, W]`` frames. Only the
                last frame is examined: the gate answers "is there motion
                now?", and scanning the whole batch would multiply the cost of
                the one stage that runs on everything.
            ctx: Pass context; the computed fraction is written to
                ``ctx.payload`` for the log.

        Returns:
            A :class:`StageOutput` whose ``wake_next`` reflects motion or
            hysteresis.
        """
        frames = np.asarray(frame_batch)
        if frames.ndim < 3:
            raise ValueError(
                f"MotionGate expects [N, H, W] or [N, H, W, C] frames, got "
                f"shape {frames.shape}"
            )

        gray = _to_gray(frames[-1])
        self._frames_seen += 1
        fraction = self._backend.foreground_fraction(gray)
        self._last_fraction = fraction

        warming_up = self._frames_seen <= self._config.warmup_frames
        moving = not warming_up and fraction >= self._config.min_foreground_fraction

        if moving:
            self._awake_remaining = self._config.stay_awake_frames
            held_awake = False
        elif self._awake_remaining > 0:
            self._awake_remaining -= 1
            held_awake = True
        else:
            held_awake = False

        wake_next = moving or held_awake
        ctx.payload["motion_fraction"] = fraction

        return StageOutput(
            woke=True,
            wake_next=wake_next,
            detail={
                "foreground_fraction": round(fraction, 6),
                "moving": moving,
                "held_awake_by_hysteresis": held_awake,
                "awake_remaining": self._awake_remaining,
                "warming_up": warming_up,
                "backend": self._backend.name,
            },
        )

    def should_wake_next(self, output: StageOutput) -> bool:
        return output.wake_next
