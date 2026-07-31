"""Walks frames through the wake hierarchy and reports what each stage cost.

The short-circuit is the whole point: when a stage declines to wake the next
one, every remaining stage is skipped and recorded as seen-but-not-woken. The
cost of an idle camera is therefore the cost of stage 0 alone, which is what
makes the Tier-1 budget arithmetic work.

Statistics are emitted on a wall-clock interval rather than a frame count, so
the reporting cadence does not change with frame rate — the log stays readable
whether a camera is running at 12 fps or 30.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from src.cascade.stage import (
    FrameBatch,
    Stage,
    StageAccounting,
    StageContext,
    StageStats,
)

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class CascadeStats:
    """One reporting interval's worth of per-stage accounting."""

    interval_index: int
    frames_processed: int
    elapsed_s: float
    stages: tuple[StageStats, ...] = field(default_factory=tuple)

    @property
    def frames_per_second(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval": self.interval_index,
            "frames_processed": self.frames_processed,
            "elapsed_s": round(self.elapsed_s, 4),
            "fps": round(self.frames_per_second, 3),
            "stages": [stage.as_dict() for stage in self.stages],
        }

    def summary(self) -> str:
        lines = [
            f"interval {self.interval_index}: {self.frames_processed} frames in "
            f"{self.elapsed_s:.2f}s ({self.frames_per_second:.1f} fps)"
        ]
        lines.extend(f"  {stage.summary()}" for stage in self.stages)
        return "\n".join(lines)


class CascadeRunner:
    """Drives frames through an ordered list of stages.

    Args:
        stages: Stages in wake order. Stage 0 first.
        stats_interval_s: How often to emit a :class:`CascadeStats` record.
        sink: Called with each stats record. Defaults to discarding them, so
            the runner is usable in a test without a file.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        stats_interval_s: float = 10.0,
        sink: Callable[[CascadeStats], None] | None = None,
    ) -> None:
        if not stages:
            raise ValueError("CascadeRunner needs at least one stage")
        self._stages = list(stages)
        self._accounting = [StageAccounting(stage.name) for stage in self._stages]
        self._interval_s = stats_interval_s
        self._sink = sink or (lambda _stats: None)
        self._frames_processed = 0
        self._intervals_emitted = 0
        self._interval_started = time.perf_counter()
        self._interval_frames = 0

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    def process_frame(self, frame_batch: FrameBatch, ctx: StageContext) -> int:
        """Run one batch through the cascade.

        Returns:
            How many stages actually woke. One means the motion gate declined
            to wake anything, which is the cheap path and should be the common
            one on real footage.
        """
        woken = 0
        proceed = True

        for stage, accounting in zip(self._stages, self._accounting):
            if not proceed:
                # Seen but skipped. Recorded so wake fractions are computed
                # against every frame the stage was offered, not only the ones
                # it ran on — otherwise a stage that runs twice out of a
                # thousand frames reports a 100% wake rate.
                accounting.observe(woke=False, woke_next=False)
                continue

            with accounting.time_wake():
                output = stage.process(frame_batch, ctx)
            woken += 1
            proceed = stage.should_wake_next(output)
            accounting.observe(woke=True, woke_next=proceed)

        self._frames_processed += 1
        self._interval_frames += 1
        self._maybe_emit()
        return woken

    def run(self, frames: Iterable[tuple[FrameBatch, StageContext]]) -> CascadeStats:
        """Consume an iterator of ``(batch, context)`` pairs.

        Returns:
            A final :class:`CascadeStats` covering the whole run, emitted to
            the sink as well as returned.
        """
        for frame_batch, ctx in frames:
            self.process_frame(frame_batch, ctx)
        return self.flush()

    def _maybe_emit(self) -> None:
        elapsed = time.perf_counter() - self._interval_started
        if elapsed >= self._interval_s:
            self._emit(elapsed)

    def _emit(self, elapsed: float) -> CascadeStats:
        stats = CascadeStats(
            interval_index=self._intervals_emitted,
            frames_processed=self._interval_frames,
            elapsed_s=elapsed,
            stages=tuple(a.snapshot() for a in self._accounting),
        )
        self._sink(stats)
        self._intervals_emitted += 1
        self._interval_started = time.perf_counter()
        self._interval_frames = 0
        return stats

    def flush(self) -> CascadeStats:
        """Emit a final stats record covering the time since the last one."""
        return self._emit(time.perf_counter() - self._interval_started)

    def stats(self) -> tuple[StageStats, ...]:
        """Current per-stage statistics without emitting a record."""
        return tuple(a.snapshot() for a in self._accounting)


class JsonlStatsSink:
    """Writes each :class:`CascadeStats` record as one JSON line."""

    def __init__(self, handle: TextIO) -> None:
        self._handle = handle

    def __call__(self, stats: CascadeStats) -> None:
        self._handle.write(json.dumps(stats.as_dict(), sort_keys=True) + "\n")
        self._handle.flush()
