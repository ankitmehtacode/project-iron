"""Statistical gates for memory stability and latency.

Why a regression and not a threshold
------------------------------------
The previous gate was::

    if total_growth > 150 and trend > 30:  # both magic numbers
        FAIL

That has three defects. It compares two hand-picked constants with no stated
basis. It uses ``and``, so a run had to breach *both* to fail — a genuine
100 MB/hour leak on a short run passed. And it is scale-dependent: the same
leak passes on a 50-clip run and fails on a 500-clip one, because total growth
depends on how long you happened to watch.

A leak is a *rate*, so the gate measures a rate. Fit RSS against clip index,
take the upper bound of the 95% confidence interval on the slope, convert to
MB/hour using the run's measured throughput, and compare that to one threshold
with units attached. Using the CI upper bound rather than the point estimate
means noise makes the gate stricter, never more permissive — a noisy run cannot
sneak a leak past by widening its own error bars.

Everything here is a pure function of sampled numbers, with no process state or
I/O, which is what lets ``tests/test_endurance_gates.py`` verify the gate on
synthetic series without model weights.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

SECONDS_PER_HOUR = 3600.0

# A regression needs three points before it has any residual degrees of
# freedom at all, but three is only the mathematical floor. Below roughly ten
# readings the confidence interval is so wide that the gate stops being
# informative in either direction, and a slope fitted through a handful of
# points is dominated by whatever the process happened to be doing during
# start-up. Failing on that produces false alarms, which is how leak gates end
# up switched off. Short runs report INCONCLUSIVE instead — honest, and still
# actionable ("run more clips").
MIN_SAMPLES_FOR_CONFIDENCE = 10


class Verdict(enum.Enum):
    """Outcome of a gate.

    ``INCONCLUSIVE`` exists so that a short smoke run cannot report either a
    clean bill of health it did not earn or a failure it did not demonstrate.
    Three clips is not a leak test; saying so is more useful than guessing.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class LeakVerdict:
    """Result of the memory-stability regression, with its own evidence.

    Every field the gate used to reach its decision is retained so that a
    failure in CI can be understood from the log alone, without a rerun.
    """

    verdict: Verdict
    reason: str
    n_samples: int
    slope_mb_per_clip: float
    slope_ci_upper_mb_per_clip: float
    mb_per_hour: float
    mb_per_hour_ci_upper: float
    threshold_mb_per_hour: float
    clips_per_hour: float
    r_squared: float

    @property
    def failed(self) -> bool:
        return self.verdict is Verdict.FAIL

    def summary(self) -> str:
        """One-line human summary for the log."""
        return (
            f"{self.verdict.value}: {self.mb_per_hour_ci_upper:+.1f} MB/hour "
            f"(95% CI upper bound) against a {self.threshold_mb_per_hour:.1f} "
            f"MB/hour limit; point estimate {self.mb_per_hour:+.1f} MB/hour, "
            f"slope {self.slope_mb_per_clip:+.4f} MB/clip over {self.n_samples} "
            f"clips, R^2={self.r_squared:.3f}"
        )


def evaluate_leak(
    rss_mb: Sequence[float],
    seconds_per_clip: float,
    threshold_mb_per_hour: float,
    confidence: float = 0.95,
) -> LeakVerdict:
    """Decide whether a series of RSS readings shows a memory leak.

    Args:
        rss_mb: Post-warm-up RSS readings in MB, one per clip, in order. The
            warm-up clip must already be excluded: the first inference call
            allocates framework buffers, and including it fits a step change as
            though it were a trend.
        seconds_per_clip: Measured mean wall-clock seconds per clip. Converts
            the fitted MB/clip slope into MB/hour, so the gate is expressed in
            the units an operator cares about ("how much will this grow
            overnight?") rather than in units of an implementation detail.
        threshold_mb_per_hour: Maximum tolerated growth rate.
        confidence: Confidence level for the interval. 0.95 by default.

    Returns:
        A :class:`LeakVerdict`. ``INCONCLUSIVE`` when there are too few
        readings for a usable interval, when throughput is unknown, or when a
        reading is non-finite. Perfectly constant RSS is a ``PASS``, not a
        degenerate case.
    """
    readings = np.asarray(list(rss_mb), dtype=np.float64)
    n = int(readings.size)

    def inconclusive(reason: str) -> LeakVerdict:
        return LeakVerdict(
            verdict=Verdict.INCONCLUSIVE,
            reason=reason,
            n_samples=n,
            slope_mb_per_clip=float("nan"),
            slope_ci_upper_mb_per_clip=float("nan"),
            mb_per_hour=float("nan"),
            mb_per_hour_ci_upper=float("nan"),
            threshold_mb_per_hour=threshold_mb_per_hour,
            clips_per_hour=float("nan"),
            r_squared=float("nan"),
        )

    if n < MIN_SAMPLES_FOR_CONFIDENCE:
        return inconclusive(
            f"need at least {MIN_SAMPLES_FOR_CONFIDENCE} post-warm-up readings "
            f"for a usable confidence interval, got {n}"
        )
    if not np.all(np.isfinite(readings)):
        return inconclusive("RSS readings contain non-finite values")
    if seconds_per_clip <= 0 or not np.isfinite(seconds_per_clip):
        return inconclusive(
            f"cannot convert MB/clip to MB/hour without a positive clip "
            f"duration, got {seconds_per_clip}"
        )

    clips_per_hour = SECONDS_PER_HOUR / seconds_per_clip

    # Perfectly constant RSS. scipy returns slope=0 with stderr=nan here
    # (the correlation coefficient is 0/0), which must not be mistaken for a
    # degenerate fit: it is the least ambiguous pass there is. Handled up front
    # so the nan never reaches the confidence-interval arithmetic.
    if float(np.ptp(readings)) == 0.0:
        return LeakVerdict(
            verdict=Verdict.PASS,
            reason=f"RSS is exactly constant at {readings[0]:.1f} MB",
            n_samples=n,
            slope_mb_per_clip=0.0,
            slope_ci_upper_mb_per_clip=0.0,
            mb_per_hour=0.0,
            mb_per_hour_ci_upper=0.0,
            threshold_mb_per_hour=threshold_mb_per_hour,
            clips_per_hour=clips_per_hour,
            r_squared=1.0,
        )

    clip_index = np.arange(n, dtype=np.float64)
    result = stats.linregress(clip_index, readings)

    slope = float(result.slope)
    stderr = float(result.stderr)
    r_squared = float(result.rvalue) ** 2

    if not np.isfinite(slope) or not np.isfinite(stderr):
        return inconclusive(
            "regression is degenerate; the fit produced a non-finite slope"
        )

    # Upper bound of the one-sided-equivalent two-sided CI on the slope. Only
    # the upper bound matters: memory falling is not a failure.
    degrees_of_freedom = n - 2
    t_critical = float(stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, degrees_of_freedom))
    slope_ci_upper = slope + t_critical * stderr

    mb_per_hour = slope * clips_per_hour
    mb_per_hour_ci_upper = slope_ci_upper * clips_per_hour

    if mb_per_hour_ci_upper > threshold_mb_per_hour:
        verdict = Verdict.FAIL
        reason = (
            f"growth of {mb_per_hour_ci_upper:.1f} MB/hour (95% CI upper bound) "
            f"exceeds the {threshold_mb_per_hour:.1f} MB/hour limit"
        )
    else:
        verdict = Verdict.PASS
        reason = (
            f"growth bounded above by {mb_per_hour_ci_upper:.1f} MB/hour, "
            f"within the {threshold_mb_per_hour:.1f} MB/hour limit"
        )

    return LeakVerdict(
        verdict=verdict,
        reason=reason,
        n_samples=n,
        slope_mb_per_clip=slope,
        slope_ci_upper_mb_per_clip=slope_ci_upper,
        mb_per_hour=mb_per_hour,
        mb_per_hour_ci_upper=mb_per_hour_ci_upper,
        threshold_mb_per_hour=threshold_mb_per_hour,
        clips_per_hour=clips_per_hour,
        r_squared=r_squared,
    )


@dataclass(frozen=True)
class LatencySummary:
    """Per-clip latency distribution.

    Percentiles rather than a mean because a mean hides exactly what matters
    operationally. A pipeline averaging 40 ms with a p99 of 900 ms drops frames
    once a second; its mean says it is comfortable.
    """

    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float

    def summary(self) -> str:
        return (
            f"p50 {self.p50_ms:.1f} ms | p95 {self.p95_ms:.1f} ms | "
            f"p99 {self.p99_ms:.1f} ms | max {self.max_ms:.1f} ms | "
            f"mean {self.mean_ms:.1f} ms over {self.count} clips"
        )


def summarize_latency(durations_s: Sequence[float]) -> LatencySummary:
    """Summarise per-clip wall-clock durations.

    Args:
        durations_s: Per-clip durations in seconds.

    Returns:
        A :class:`LatencySummary` in milliseconds. All-zero for an empty input,
        which the caller distinguishes via ``count``.

    Raises:
        ValueError: if any duration is negative, which means the clock went
            backwards and every derived number is untrustworthy.
    """
    values = np.asarray(list(durations_s), dtype=np.float64)
    if values.size == 0:
        return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if np.any(values < 0):
        raise ValueError(
            "negative clip duration recorded; the monotonic clock moved "
            "backwards and the latency summary cannot be trusted"
        )

    ms = values * 1000.0
    return LatencySummary(
        count=int(ms.size),
        p50_ms=float(np.percentile(ms, 50)),
        p95_ms=float(np.percentile(ms, 95)),
        p99_ms=float(np.percentile(ms, 99)),
        max_ms=float(np.max(ms)),
        mean_ms=float(np.mean(ms)),
    )
