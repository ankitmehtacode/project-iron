"""Confidence intervals for rates reported from counts.

A rate without its denominator reads as more certain than it is: 3 misses of
4 and 300 of 400 are both "75%". Every per-bin rate in the Day-42 size curves
is reported with its count and a Wilson score interval, which stays inside
[0, 1] and behaves at 0 and n (where the normal approximation collapses to a
zero-width interval and claims certainty from one sample).
"""

from __future__ import annotations

import math

Z95 = 1.959963984540054
"""Two-sided 95% standard-normal quantile."""


def wilson_interval(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``successes / trials``.

    Raises on ``trials == 0``: an interval over nothing is not ``(0, 1)``,
    it is undefined, and the caller must say so rather than print a range.
    """
    if trials <= 0:
        raise ValueError("a Wilson interval needs at least one trial")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside [0, {trials}]")
    p = successes / trials
    z2 = z * z
    centre = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    half = (z / (1 + z2 / trials)) * math.sqrt(
        p * (1 - p) / trials + z2 / (4 * trials * trials)
    )
    return (max(0.0, centre - half), min(1.0, centre + half))
