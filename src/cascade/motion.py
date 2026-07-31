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

Measured cost
-------------
Measured 2026-07-31 with ``scripts/cascade_bench.py``, 1280x720 source at
12 fps, MOG2, gating at 320x180, on the development machine:

==================  ==========  ==========  =========  ========
Scenario            p50 / frame  % of core  wake@full  wake@gate
==================  ==========  ==========  =========  ========
static (idle)          2.48 ms      2.97%       0.0%      0.0%
near target            2.48 ms      2.97%      33.3%     33.3%
SMALL target           2.45 ms      2.94%      33.3%     33.3%
real footage           2.18 ms      2.62%      84.8%     84.8%
==================  ==========  ==========  =========  ========

Wake decisions are **identical** at both resolutions on every scenario,
including the small-target case (a blob ~15 px tall at gate resolution, i.e. a
person about 60 px tall at 720p — someone across a car park). Cost is within
the 3% Tier-1 idle budget, with very little margin.

Getting there took three changes, in descending order of effect. Gating at
320x180 rather than 720p. Resizing the colour frame *before* the greyscale
reduction rather than after, which is mathematically equivalent for linear
operations but moves the bulk work into one OpenCV call. And computing the
channel mean with ``cv2.transform`` instead of ``numpy.mean``, which is the
same arithmetic 8x faster.

Two things worth knowing about the numbers. The margin is thin — 2.97% against
3.00% on a contended laptop — so the bench also asserts a machine-independent
floor of 3x cheaper than full-resolution gating, which is the assertion that
actually catches someone removing the downscale. And ``cv2.transform`` rounds
where ``numpy.mean`` truncated, a one-grey-level difference that changed
2 wake decisions out of 250 on real footage; it changed none on any synthetic
scenario. That is a real behaviour change, in the direction of correctness.

``min_foreground_fraction`` is not portable across resolutions or backends.
Frame differencing at 720p wakes on 5.4% of frames where MOG2 wakes on 35%,
because it sees only the moving *edges* of a target. Re-tune whenever either
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from src.cascade.stage import FrameBatch, StageContext, StageOutput

GrayFrame = npt.NDArray[np.uint8]

# Equal weights: a plain channel mean, not ITU-R luma. Luma weighting would be
# channel-order dependent, and BGR/RGB confusion here is silent.
_CHANNEL_MEAN_WEIGHTS = np.full((1, 3), 1.0 / 3.0, dtype=np.float64)
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

    gate_width: int = 320
    gate_height: int = 180
    """Resolution the gate actually runs at.

    Motion gating does not need the resolution a detector needs: the question
    is "did anything move", not "what is it". Running the background model on
    a 320x180 view of a 720p frame costs about a twelfth as much and, on every
    benchmark scenario here, reaches the same wake decisions.

    Both fields are compared against the incoming frame, so a source already at
    or below this size is left alone rather than upscaled. Set either to 0 to
    disable downscaling entirely and gate at full resolution.
    """

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
        if self.gate_width < 0 or self.gate_height < 0:
            raise ValueError(
                "gate_width and gate_height must not be negative; use 0 to "
                "disable downscaling"
            )

    @property
    def downscaling_enabled(self) -> bool:
        return self.gate_width > 0 and self.gate_height > 0


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


def _downscale(gray: GrayFrame, target_height: int, target_width: int) -> GrayFrame:
    """Reduce a single-channel frame to the gate's working resolution.

    Uses **area averaging**, not nearest-neighbour subsampling. Subsampling
    drops whole rows and columns, so a distant person 60 px tall at 720p can
    lose most of its pixels to the discard pattern and fall under the
    foreground threshold — the gate would then sleep through exactly the events
    it exists to catch. Area averaging preserves the energy of small moving
    targets, which is what makes wake parity with full-resolution gating
    achievable at all.

    Frames already at or below the target are returned untouched; upscaling
    would invent detail and cost time.
    """
    height, width = gray.shape[:2]
    if height <= target_height and width <= target_width:
        return gray

    try:
        import cv2
    except ImportError:
        pass
    else:
        return np.asarray(
            cv2.resize(
                gray, (target_width, target_height), interpolation=cv2.INTER_AREA
            ),
            dtype=np.uint8,
        )

    # Pure-numpy area average. Exact when the target divides the source;
    # otherwise it crops the remainder, which is closer to INTER_AREA than
    # striding would be.
    block_h = max(1, height // target_height)
    block_w = max(1, width // target_width)
    usable_h = (height // block_h) * block_h
    usable_w = (width // block_w) * block_w
    cropped = gray[:usable_h, :usable_w].astype(np.float32)
    pooled = cropped.reshape(
        usable_h // block_h, block_h, usable_w // block_w, block_w
    ).mean(axis=(1, 3))
    return pooled.astype(np.uint8)


def _downscale_colour(
    frame: npt.NDArray[Any], target_height: int, target_width: int
) -> npt.NDArray[Any]:
    """Area-average a colour frame down to the gate resolution.

    Done *before* the greyscale reduction rather than after. Both operations
    are linear — a mean across channels and a mean across a spatial block — so
    they commute, and the result is identical up to uint8 rounding. The order
    matters enormously for cost: reducing 1280x720x3 to greyscale first means a
    per-pixel reduction over 2.7 million values in numpy, whereas resizing
    first hands the bulk work to one OpenCV call and leaves the greyscale step
    operating on 320x180.

    That reordering is what took stage 0 from 22% of a core to under 3%. The
    benchmark verifies frame-by-frame that wake decisions are unchanged.
    """
    height, width = frame.shape[:2]
    if height <= target_height and width <= target_width:
        return frame
    try:
        import cv2
    except ImportError:
        return frame
    return np.asarray(
        cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    )


def _to_gray(frame: npt.NDArray[Any]) -> GrayFrame:
    """Reduce a frame to single-channel uint8.

    Uses a plain channel mean rather than luma weights. The gate only ever
    compares a frame against its own history, so the exact colour transform is
    irrelevant, and the mean is both cheaper and independent of channel order —
    which matters because BGR/RGB confusion is otherwise silent here.
    """
    array = np.asarray(frame)
    if array.ndim != 3:
        return array.astype(np.uint8)

    try:
        import cv2
    except ImportError:
        return array.mean(axis=2, dtype=np.float32).astype(np.uint8)

    # cv2.transform with equal weights computes exactly the channel mean, in C.
    # The numpy equivalent costs 1.0 ms per 320x180 frame against 0.13 ms here,
    # because reducing along the last axis of an interleaved uint8 image is
    # cache-hostile. Results differ by at most one grey level, from rounding
    # (OpenCV rounds half away from zero, numpy truncates on the uint8 cast);
    # tests/test_cascade.py pins that the difference changes no wake decision.
    return np.asarray(cv2.transform(array, _CHANNEL_MEAN_WEIGHTS), dtype=np.uint8)


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
        self._gate_shape: tuple[int, ...] = ()

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
    def gate_shape(self) -> tuple[int, ...]:
        """Resolution the last frame was actually gated at.

        Reported so a benchmark cannot compare two gates running at different
        resolutions without noticing.
        """
        return self._gate_shape

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

        latest = frames[-1]
        source_shape = latest.shape[:2]
        if self._config.downscaling_enabled and latest.ndim == 3:
            latest = _downscale_colour(
                latest, self._config.gate_height, self._config.gate_width
            )
        gray = _to_gray(latest)
        if self._config.downscaling_enabled:
            # No-op when the colour resize already got there; the fallback path
            # without OpenCV still needs it.
            gray = _downscale(gray, self._config.gate_height, self._config.gate_width)
        self._gate_shape = gray.shape

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
                "source_shape": list(source_shape),
                "gate_shape": list(gray.shape),
            },
        )

    def should_wake_next(self, output: StageOutput) -> bool:
        return output.wake_next
