"""Where an observation's coordinates live: geometry, transform, and twin version.

Wraps the existing :mod:`src.contracts.frames` primitives rather than
duplicating them — see the ``iron-contracts`` rule that a raw array or a
bare coordinate never crosses a module boundary without the metadata to
interpret it. :class:`FrameOfReference` adds the one thing the contracts
layer does not carry: which revision of the site's 3D twin (camera
extrinsics, zone polygons, floorplan) the transform was computed against.
A twin re-version (a camera recalibrated, a zone redrawn) invalidates any
observation whose ``twin_rev`` predates it for purposes that depend on
world coordinates, even though the pixel data itself has not changed.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.contracts.frames import AffineTransform, FrameGeometry


@dataclass(frozen=True)
class FrameOfReference:
    """The pixel geometry, canonicalizing transform, and twin revision.

    Attributes:
        geometry: The raster this observation's pixel coordinates are
            measured in.
        to_canonical: Transform from ``geometry`` back to the canonical
            frame (see :mod:`src.contracts.frames`).
        twin_rev: Monotonically increasing revision of the site's 3D twin
            (camera extrinsics, zone geometry) in effect when this
            observation's world-space fields, if any, were computed.
    """

    geometry: FrameGeometry
    to_canonical: AffineTransform
    twin_rev: int

    def __post_init__(self) -> None:
        if self.twin_rev < 0:
            raise ValueError(f"twin_rev must be non-negative, got {self.twin_rev}")

    def is_current_for(self, current_twin_rev: int) -> bool:
        """Whether this observation's world-space fields still hold.

        Used at query time, not at write time: a Day-13 twin re-version
        must not rewrite history, only mark which observations agree with
        the twin now in effect.
        """
        return self.twin_rev == current_twin_rev
