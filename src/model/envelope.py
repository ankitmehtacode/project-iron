"""Envelope: a capability's measured operating curve, not a scalar threshold.

Day 7 measured the motion gate's wake threshold varying 4.9x with subject
speed. A scalar "the gate wakes above 0.01" would have been wrong at every
speed but one. :class:`EnvelopeCurve` is therefore constructed from at
least two measured points and interpolated, never a single number —
enforced in ``__post_init__``, not by convention.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass


class EnvelopeError(ValueError):
    """Raised when an envelope curve or record violates its contract."""


@dataclass(frozen=True)
class EnvelopeCurve:
    """A measured capability threshold as a function of one independent variable.

    Attributes:
        independent_variable: What the curve is measured against (e.g.
            ``"speed_mps"``, ``"range_m"``, ``"illuminance_lux"``).
        points: ``(x, threshold)`` pairs, strictly increasing in ``x``, at
            least two of them. Two points is the minimum that is a curve
            rather than a constant; a single-point "curve" is exactly the
            scalar-threshold bug this type exists to prevent, so it is
            rejected at construction.
    """

    independent_variable: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.independent_variable:
            raise EnvelopeError("EnvelopeCurve.independent_variable must not be empty")
        if len(self.points) < 2:
            raise EnvelopeError(
                f"EnvelopeCurve needs at least 2 measured points to be a "
                f"curve rather than a scalar threshold, got {len(self.points)}. "
                "A single point cannot represent a threshold that varies "
                "with the independent variable — see Day 7's 4.9x "
                "speed-dependent wake threshold."
            )
        xs = [x for x, _ in self.points]
        if xs != sorted(xs) or len(set(xs)) != len(xs):
            raise EnvelopeError(
                f"EnvelopeCurve.points must be strictly increasing in x, "
                f"got x values {xs}"
            )

    def value_at(self, x: float) -> float:
        """Linearly interpolate the threshold at ``x``; clamp outside the range.

        Clamping (not extrapolating) is deliberate: a capability measured
        between 0.2 and 4.9 m/s says nothing about 20 m/s, and returning a
        linear extrapolation there would manufacture a number this project
        has never measured.
        """
        xs = [px for px, _ in self.points]
        ys = [py for _, py in self.points]
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect_left(xs, x)
        if xs[i] == x:
            return ys[i]
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)


@dataclass(frozen=True)
class Envelope:
    """One capability's measured operating envelope for one camera, one twin_rev.

    Attributes:
        capability: What is being characterised (e.g. ``"motion_gate"``,
            ``"reid"``, ``"interaction_detection"``).
        camera_id: Which sensor this measurement applies to. Envelopes do
            not generalise across cameras — different optics, mounting
            height, and field of view all shift the curve.
        twin_rev: The site-twin revision this measurement is valid for.
            A camera recalibration or remount bumps ``twin_rev`` and
            invalidates envelopes measured under the old one.
        curve: The measured threshold curve.
        sample_count: How many measurements the curve is fit from. Present
            so a consumer can tell a well-supported curve from one fit on
            two points, without re-deriving it from raw data.
        manifest_sha: The run that produced this measurement.
    """

    capability: str
    camera_id: str
    twin_rev: int
    curve: EnvelopeCurve
    sample_count: int
    manifest_sha: str

    def __post_init__(self) -> None:
        if not self.capability:
            raise EnvelopeError("Envelope.capability must not be empty")
        if not self.camera_id:
            raise EnvelopeError("Envelope.camera_id must not be empty")
        if self.twin_rev < 0:
            raise EnvelopeError(f"twin_rev must be non-negative, got {self.twin_rev}")
        if self.sample_count < 2:
            raise EnvelopeError(
                f"Envelope.sample_count must be >= 2 (a curve needs at least "
                f"as many measurements as it has points), got {self.sample_count}"
            )
        if not self.manifest_sha:
            raise EnvelopeError("Envelope.manifest_sha must not be empty")
