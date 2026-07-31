"""The stage protocol and per-stage accounting.

Why a cascade at all
--------------------
Tier 1 sells "a layer of intelligence on your existing hardware", which is
exactly as true as the idle-scene budget being met: under 3% of one core per
camera. A pipeline that runs a person detector on every frame of eight 720p
streams cannot meet that on NUC-class hardware, and no amount of optimising the
detector will change the arithmetic. The only way is to not run it — most
frames of most cameras contain nothing moving, and the cascade's entire job is
to spend nothing on them.

Stage 0 (motion) is always on and nearly free. Each later stage runs only when
the one before it says something is worth looking at::

    MotionGate -> DetectorStage -> TrackerStage -> SemanticsStage

Accounting is part of the protocol, not an add-on
-------------------------------------------------
Every stage records how often it woke and how long it took, at p50/p95/p99 —
never a mean, which hides exactly the tail that drops frames. A stage without
wake accounting is unfinished, because the only way to know whether the budget
holds is to measure the wake rate on real footage. A cascade that silently
wakes everything still produces correct output; it just costs eight times what
was sold.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FrameBatch = npt.NDArray[np.uint8]
"""A batch of frames, ``[N, H, W, C]``, uint8.

Kept as a plain array rather than a contract envelope because the cascade
operates entirely in one frame's own pixel space; nothing here resizes or
reprojects. The moment a stage does either, its output must carry an
``AffineTransform`` from :mod:`src.contracts`.
"""


@dataclass
class StageContext:
    """Mutable state carried along one pass through the cascade.

    Stages communicate through this rather than by returning ever-larger
    tuples, so adding a stage does not change the signature of the ones around
    it.
    """

    frame_index: int
    ts_ns: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageOutput:
    """What a stage produced, and whether it is worth continuing.

    Attributes:
        woke: Whether this stage did any work. ``False`` means it was skipped
            because an earlier stage short-circuited, and it must not count
            toward this stage's latency statistics.
        wake_next: Whether the next stage should run.
        detail: Stage-specific diagnostics, logged rather than acted upon.
    """

    woke: bool
    wake_next: bool
    detail: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Stage(Protocol):
    """One step of the wake hierarchy.

    A Protocol rather than a base class: stages are independently testable
    units with nothing to inherit, and structural typing keeps a test double
    from needing to import production machinery.
    """

    @property
    def name(self) -> str: ...

    def process(self, frame_batch: FrameBatch, ctx: StageContext) -> StageOutput:
        """Do this stage's work and report whether to wake the next one."""
        ...

    def should_wake_next(self, output: StageOutput) -> bool:
        """Decide whether the next stage runs, given this stage's output."""
        ...


class StageAccounting:
    """Wake counts and latency percentiles for one stage.

    Durations are recorded only for frames where the stage actually woke.
    Including skipped frames as zero-cost samples would drag every percentile
    toward zero and make an expensive stage look cheap precisely when it is
    rarely used.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._durations_ms: list[float] = []
        self._frames_seen = 0
        self._wakes = 0
        self._woke_next = 0

    @property
    def name(self) -> str:
        return self._name

    @contextmanager
    def time_wake(self) -> Iterator[None]:
        """Time one execution of the stage and count it as a wake."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._durations_ms.append((time.perf_counter() - start) * 1000.0)
            self._wakes += 1

    def observe(self, *, woke: bool, woke_next: bool) -> None:
        """Record that this stage saw a frame, whether or not it ran."""
        self._frames_seen += 1
        if woke and woke_next:
            self._woke_next += 1

    def snapshot(self) -> "StageStats":
        durations = np.asarray(self._durations_ms, dtype=np.float64)
        if durations.size == 0:
            return StageStats(
                name=self._name,
                frames_seen=self._frames_seen,
                wakes=0,
                woke_next=self._woke_next,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
            )
        return StageStats(
            name=self._name,
            frames_seen=self._frames_seen,
            wakes=self._wakes,
            woke_next=self._woke_next,
            p50_ms=float(np.percentile(durations, 50)),
            p95_ms=float(np.percentile(durations, 95)),
            p99_ms=float(np.percentile(durations, 99)),
            max_ms=float(np.max(durations)),
        )


@dataclass(frozen=True)
class StageStats:
    """Immutable per-stage statistics for a reporting interval."""

    name: str
    frames_seen: int
    wakes: int
    woke_next: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @property
    def wake_fraction(self) -> float:
        """Share of frames on which this stage ran. The budget lever."""
        if self.frames_seen == 0:
            return 0.0
        return self.wakes / self.frames_seen

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "frames_seen": self.frames_seen,
            "wakes": self.wakes,
            "wake_fraction": round(self.wake_fraction, 6),
            "woke_next": self.woke_next,
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "max_ms": round(self.max_ms, 4),
        }

    def summary(self) -> str:
        return (
            f"{self.name:<16} woke {self.wakes:>6}/{self.frames_seen:<6} "
            f"({self.wake_fraction * 100:5.1f}%)  "
            f"p50 {self.p50_ms:7.3f} ms  p99 {self.p99_ms:7.3f} ms  "
            f"max {self.max_ms:7.3f} ms"
        )
