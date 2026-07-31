"""Process memory sampling for the endurance harness.

Why RSS alone is not enough
---------------------------
RSS counts every resident page including shared library text, so it moves when
an unrelated library is paged in and stays flat when a leak lands in memory the
allocator has already reserved. Two additions make the picture honest:

- **USS/PSS** from ``/proc/self/smaps_rollup``. USS is the memory that would be
  freed if the process exited — the number that actually answers "is this
  process accumulating?" — and it is unaffected by shared-library noise.
- **``malloc_trim(0)`` before each reading.** glibc keeps freed memory in its
  arenas rather than returning it to the kernel, so RSS can plateau at a high
  water mark while the heap is genuinely healthy. Trimming first distinguishes
  "leaked" from "merely retained by the allocator", which is the difference
  between a bug and a non-event.

Both are Linux-specific and both degrade gracefully: on macOS or in a container
without ``/proc``, the fields come back ``None`` and the caller reports reduced
fidelity rather than pretending. A sampler that silently returns zeros for
unavailable metrics produces flat graphs that look like success.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import psutil

SMAPS_ROLLUP = Path("/proc/self/smaps_rollup")
BYTES_PER_MB = 1024.0 * 1024.0


@dataclass(frozen=True)
class MemorySample:
    """One point-in-time reading of process memory.

    ``uss_mb`` and ``pss_mb`` are ``None`` when the platform cannot supply
    them, never ``0.0``. A zero is a measurement; ``None`` is the absence of
    one, and conflating the two turns a blind spot into a clean result.
    """

    rss_mb: float
    uss_mb: float | None
    pss_mb: float | None
    gc_counts: tuple[int, ...]
    gc_tracked_objects: int | None
    trimmed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "rss_mb": round(self.rss_mb, 3),
            "uss_mb": None if self.uss_mb is None else round(self.uss_mb, 3),
            "pss_mb": None if self.pss_mb is None else round(self.pss_mb, 3),
            "gc_counts": list(self.gc_counts),
            "gc_tracked_objects": self.gc_tracked_objects,
            "malloc_trimmed": self.trimmed,
        }


def malloc_trim() -> bool:
    """Ask glibc to return free heap pages to the kernel.

    Returns:
        True if the call was made. False on non-Linux platforms, or where libc
        does not export ``malloc_trim`` (notably musl, as used by Alpine).
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.malloc_trim(ctypes.c_size_t(0))
    except (OSError, AttributeError):
        return False
    return True


def read_smaps_rollup() -> tuple[float | None, float | None]:
    """Read USS and PSS in MB from ``/proc/self/smaps_rollup``.

    USS is computed as ``Private_Clean + Private_Dirty``, which is the standard
    definition: pages mapped by no other process.

    Returns:
        ``(uss_mb, pss_mb)``, either element ``None`` if unavailable.
    """
    if not SMAPS_ROLLUP.exists():
        return (None, None)
    try:
        text = SMAPS_ROLLUP.read_text()
    except OSError:
        return (None, None)

    private_kb = 0.0
    pss_kb: float | None = None
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        if key in ("Private_Clean", "Private_Dirty"):
            private_kb += value
        elif key == "Pss":
            pss_kb = value

    uss_mb = private_kb / 1024.0 if private_kb else None
    return (uss_mb, None if pss_kb is None else pss_kb / 1024.0)


class MemorySampler:
    """Collects memory readings, optionally with tracemalloc allocation diffs.

    Args:
        trim_before_reading: Call ``malloc_trim(0)`` before each sample. On by
            default; see the module docstring for why it changes the meaning of
            the numbers.
        track_allocations: Enable ``tracemalloc``. Off by default because it
            roughly doubles allocation cost, which would distort the latency
            percentiles the same run is measuring.
        top_allocations: How many allocation sites to report per diff.
    """

    def __init__(
        self,
        *,
        trim_before_reading: bool = True,
        track_allocations: bool = False,
        top_allocations: int = 10,
    ) -> None:
        self._process = psutil.Process()
        self._trim = trim_before_reading
        self._top = top_allocations
        self._tracking = track_allocations
        self._previous_snapshot: tracemalloc.Snapshot | None = None
        if track_allocations and not tracemalloc.is_tracing():
            tracemalloc.start()

    @property
    def tracking_allocations(self) -> bool:
        return self._tracking

    def sample(self, *, count_objects: bool = False) -> MemorySample:
        """Take one reading.

        Args:
            count_objects: Also count every GC-tracked object. This walks the
                whole heap, so it is reserved for the periodic deep sample
                rather than every clip.
        """
        trimmed = malloc_trim() if self._trim else False
        uss_mb, pss_mb = read_smaps_rollup()
        rss_mb = float(self._process.memory_info().rss) / BYTES_PER_MB

        return MemorySample(
            rss_mb=rss_mb,
            uss_mb=uss_mb,
            pss_mb=pss_mb,
            gc_counts=tuple(gc.get_count()),
            gc_tracked_objects=len(gc.get_objects()) if count_objects else None,
            trimmed=trimmed,
        )

    def allocation_diff(self) -> list[str]:
        """Top allocation-site growth since the previous call.

        Returns:
            Human-readable lines, largest growth first. Empty when tracking is
            disabled, or on the first call, when there is no baseline to
            difference against.
        """
        if not self._tracking:
            return []

        snapshot = tracemalloc.take_snapshot()
        previous = self._previous_snapshot
        self._previous_snapshot = snapshot
        if previous is None:
            return []

        stats = snapshot.compare_to(previous, "lineno")[: self._top]
        return [
            f"{stat.size_diff / 1024.0:+9.1f} KiB  "
            f"{stat.count_diff:+6d} blocks  {stat.traceback[0]}"
            for stat in stats
        ]

    def stop(self) -> None:
        """Release tracemalloc if this sampler started it."""
        if self._tracking and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._tracking = False


def platform_fidelity_note() -> str:
    """Describe which memory metrics this platform can actually provide.

    Logged at run start so nobody later mistakes an unavailable metric for a
    healthy one.
    """
    uss, pss = read_smaps_rollup()
    parts = ["RSS via psutil"]
    if uss is None and pss is None:
        parts.append(
            f"USS/PSS unavailable ({SMAPS_ROLLUP} not readable on "
            f"{sys.platform}); leak attribution is coarser"
        )
    else:
        parts.append("USS/PSS via /proc/self/smaps_rollup")
    parts.append(
        "malloc_trim available"
        if sys.platform.startswith("linux")
        else f"malloc_trim unavailable on {sys.platform}; "
        "RSS may plateau at an allocator high-water mark"
    )
    return "; ".join(parts)
