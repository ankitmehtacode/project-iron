"""Pixel frames, the transforms between them, and camera intrinsics.

The canonical frame
-------------------
There is exactly one canonical coordinate frame: **the original video's pixel
space**. Every stage that resizes, crops, letterboxes, or pads carries an
:class:`AffineTransform` back to it. A stage output without such a transform is
uninterpretable, because "x = 112" means nothing until you know which frame it
was measured in.

Pixel-centre convention
-----------------------
**Pixel centres sit at half-integer coordinates.** Pixel column ``i`` covers the
continuous interval ``[i, i + 1)``, so its centre is at ``i + 0.5``, and the
frame spans ``[0, width]``.

This is not an arbitrary preference. Under this convention a resize by factor
``s`` is exactly ``x' = s * x`` — a pure scale with no offset. Under the
alternative convention (centres at integers) the same resize is
``x' = s * (x + 0.5) - 0.5``, and the half-pixel correction term is forgotten
approximately every time it is written by hand. Choosing the convention that
makes the common operation a pure scale removes a whole family of half-pixel
drift bugs by construction, and is why :meth:`Intrinsics.rescaled_to` is a
plain multiplication.

Distortion is not affine
------------------------
Lens distortion is a nonlinear function of radius and must never enter the
transform algebra here. It is removed once, at ingest, and recorded in
provenance. :class:`Intrinsics` carries the coefficients so that a consumer can
tell whether undistortion has happened, not so that they can be composed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from src.contracts.errors import GeometryMismatch

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FrameGeometry:
    """The pixel dimensions of a frame.

    Equality is by value, which is the entire point: it lets any consumer that
    holds two arrays assert they describe the same raster before combining
    them, instead of discovering the mismatch as a silently wrong number.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"FrameGeometry must be positive, got {self.width}x{self.height}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Numpy row-major shape ``(height, width)``.

        Named explicitly because the (width, height) / (height, width) swap
        between image-space and array-space is a recurring source of silent
        transposition.
        """
        return (self.height, self.width)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class AffineTransform:
    """An axis-aligned scale-and-offset from a stage's pixel space to canonical.

    Applying it maps a coordinate measured in some stage's frame into the
    canonical frame::

        canonical_x = scale_x * stage_x + offset_x
        canonical_y = scale_y * stage_y + offset_y

    Rotation and shear are deliberately not representable. Every resize, crop,
    pad, and letterbox in this pipeline is axis-aligned, and a type that cannot
    express a rotation cannot silently acquire one.
    """

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    def __post_init__(self) -> None:
        for name in ("scale_x", "scale_y", "offset_x", "offset_y"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"AffineTransform.{name} must be finite, got {value}")
        if self.scale_x == 0.0 or self.scale_y == 0.0:
            raise ValueError(
                "AffineTransform scale must be nonzero; a zero scale collapses "
                "the frame and cannot be inverted "
                f"(scale_x={self.scale_x}, scale_y={self.scale_y})"
            )

    @classmethod
    def identity(cls) -> "AffineTransform":
        """The transform for a stage that operates in canonical space already."""
        return cls(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0)

    @classmethod
    def for_resize(
        cls, source: FrameGeometry, target: FrameGeometry
    ) -> "AffineTransform":
        """Transform mapping ``target`` coordinates back to ``source`` space.

        Use when a stage consumes a resized copy of a frame: the stage reports
        coordinates in ``target`` space, and this transform carries them home.
        Pure scale with no offset, which is what the half-integer pixel-centre
        convention buys.
        """
        return cls(
            scale_x=source.width / target.width,
            scale_y=source.height / target.height,
            offset_x=0.0,
            offset_y=0.0,
        )

    def apply(self, xy: FloatArray) -> FloatArray:
        """Map an ``[N, 2]`` array of ``(x, y)`` coordinates into canonical space.

        Args:
            xy: Coordinates in this transform's source frame, shape ``[N, 2]``.

        Returns:
            Coordinates in the canonical frame, shape ``[N, 2]``, float64.

        Raises:
            ValueError: if ``xy`` is not ``[N, 2]``.
        """
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"AffineTransform.apply expects [N, 2] coordinates, got {points.shape}"
            )
        scale = np.array([self.scale_x, self.scale_y], dtype=np.float64)
        offset = np.array([self.offset_x, self.offset_y], dtype=np.float64)
        return points * scale + offset

    def inverse(self) -> "AffineTransform":
        """The transform mapping canonical coordinates back to this frame."""
        return AffineTransform(
            scale_x=1.0 / self.scale_x,
            scale_y=1.0 / self.scale_y,
            offset_x=-self.offset_x / self.scale_x,
            offset_y=-self.offset_y / self.scale_y,
        )

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Return the transform applying ``other`` first, then ``self``.

        Standard right-to-left function composition, so that::

            a.compose(b).apply(xy) == a.apply(b.apply(xy))

        and therefore ``t.compose(t.inverse())`` is the identity. The order is
        stated here because "compose" is ambiguous in the wild and getting it
        backwards produces a transform that is wrong only when the scales
        differ — which is exactly when nobody is checking.
        """
        return AffineTransform(
            scale_x=self.scale_x * other.scale_x,
            scale_y=self.scale_y * other.scale_y,
            offset_x=self.scale_x * other.offset_x + self.offset_x,
            offset_y=self.scale_y * other.offset_y + self.offset_y,
        )


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics, bound to the frame geometry they describe.

    ``fx``, ``fy``, ``cx``, and ``cy`` are all measured in pixels, which makes
    them meaningless without the raster size they were measured against. That
    is what :attr:`valid_for` records, and why :func:`src.contracts.geometry.unproject`
    refuses to run when it disagrees with the depth field's geometry.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    valid_for: FrameGeometry
    calibrated: bool
    """Whether these numbers came from a calibration, or were invented.

    REQUIRED, with no default (Day 31, Objective 4). It used to default to
    ``True``, which meant a caller who said nothing asserted a calibration
    it had not been shown to have — the default itself was the unverified
    claim. That is the cheapest possible instance of the pattern Day 30
    found in ``WorldPositionArray``: a contract manufacturing assurance.
    The conservative default would have been ``False``, but there is no
    correct default for a fact about provenance, so there is none.

    ``False`` marks a placeholder — see :func:`placeholder_intrinsics`.
    Placeholder intrinsics are legitimate for experiments and visualisation,
    and are rejected by :func:`src.contracts.geometry.unproject`, because a
    guessed focal length turns "3D position in metres" into a number with no
    physical meaning that still plots convincingly.
    """

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"Focal lengths must be positive, got fx={self.fx}, fy={self.fy}"
            )
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"Intrinsics.{name} must be finite, got {value}")
        if any(not np.isfinite(k) for k in self.distortion):
            raise ValueError(
                f"Distortion coefficients must be finite, got {self.distortion}"
            )

    @property
    def is_undistorted(self) -> bool:
        """True when all distortion coefficients are zero.

        :func:`unproject` assumes a rectified frame. This is how a caller can
        check that assumption rather than hope.
        """
        return all(k == 0.0 for k in self.distortion)

    def rescaled_to(self, geometry: FrameGeometry) -> "Intrinsics":
        """Return these intrinsics expressed for a different frame size.

        A pure multiplication, because pixel centres are at half-integers — see
        the module docstring. Distortion coefficients are carried over
        unchanged: in the standard radial-tangential model they are
        dimensionless functions of normalised image coordinates, so they do not
        scale with the raster. Rescaling them would be a units error of exactly
        the kind this package exists to prevent.

        Args:
            geometry: The frame size to express the intrinsics for.

        Returns:
            New intrinsics with ``valid_for`` set to ``geometry``.
        """
        scale_x = geometry.width / self.valid_for.width
        scale_y = geometry.height / self.valid_for.height
        return replace(
            self,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            valid_for=geometry,
        )

    def require_geometry(self, geometry: FrameGeometry) -> None:
        """Raise unless these intrinsics describe ``geometry``.

        Raises:
            GeometryMismatch: if they do not.
        """
        if self.valid_for != geometry:
            raise GeometryMismatch(
                f"Intrinsics are valid for {self.valid_for} but were used with "
                f"{geometry}. Call intrinsics.rescaled_to(...) at the point of "
                "resize instead of reusing stale focal lengths."
            )


def placeholder_intrinsics(geometry: FrameGeometry) -> Intrinsics:
    """Invent plausible intrinsics for a frame, marked as uncalibrated.

    Uses ``fx = fy = max(width, height)`` with the principal point at the frame
    centre — roughly a 53-degree horizontal field of view. This is the guess
    ``projector_vectorized.compute_intrinsics`` was making silently; the only
    change is that the result now says so.

    The values are not measured, not calibrated, and not recorded anywhere, so
    any 3D point derived from them has arbitrary scale. That is fine for
    checking that a pipeline runs, and never fine for a distance, a zone
    boundary, or an event. :func:`src.contracts.geometry.unproject` rejects
    these unless ``IRON_ALLOW_UNCALIBRATED=1`` is set.

    Args:
        geometry: The frame these intrinsics will describe.

    Returns:
        Intrinsics with ``calibrated=False``.
    """
    focal = float(max(geometry.width, geometry.height))
    return Intrinsics(
        fx=focal,
        fy=focal,
        cx=geometry.width / 2.0,
        cy=geometry.height / 2.0,
        distortion=(),
        valid_for=geometry,
        calibrated=False,
    )
