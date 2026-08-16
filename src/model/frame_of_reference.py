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
from src.model.world import UNREGISTERED


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
            observation's world-space fields, if any, were computed, or
            :data:`~src.model.world.UNREGISTERED` for data with no
            recorded revision.

    Day 31, Objective 4 — UNREGISTERED is accepted here too
    ---------------------------------------------------------
    ``twin_rev`` used to reject every negative value, so
    :data:`~src.model.world.UNREGISTERED` (-1) was unconstructable and a
    caller with no twin had to invent a revision. ``scripts/eval_estimator.py``
    did exactly that, passing ``twin_rev=1`` for synthetic clips that
    record no twin at all — while ``src/data/scorecard.py`` and
    ``src/inspector/artifacts.py`` wrapped the SAME clips as
    ``UNREGISTERED``. One codebase, one dataset, two contradictory
    revision claims, and the invented one was the more confident.

    Day 15 established the rule for world-frame data (``WorldPosition``,
    ``WorldPositionArray``): an absent revision is typed, not guessed.
    This type is now consistent with that, and :meth:`is_current_for`
    never answers ``True`` for an unregistered frame — unknown provenance
    is not assumed to match anything, not even another unknown, the same
    asymmetry ``WorldPosition.distance_to`` already enforces.
    """

    geometry: FrameGeometry
    to_canonical: AffineTransform
    twin_rev: int

    def __post_init__(self) -> None:
        if self.twin_rev < 0 and self.twin_rev != UNREGISTERED:
            raise ValueError(
                f"twin_rev must be non-negative, or the UNREGISTERED "
                f"sentinel ({UNREGISTERED}), got {self.twin_rev}"
            )

    @property
    def is_registered(self) -> bool:
        """Whether this frame records a real twin revision."""
        return self.twin_rev != UNREGISTERED

    def is_current_for(self, current_twin_rev: int) -> bool:
        """Whether this observation's world-space fields still hold.

        Used at query time, not at write time: a Day-13 twin re-version
        must not rewrite history, only mark which observations agree with
        the twin now in effect.

        Always ``False`` for an ``UNREGISTERED`` frame, including against
        ``UNREGISTERED`` itself: an observation whose twin revision was
        never recorded cannot be shown to agree with the current one, and
        "we do not know" must not read as "yes".
        """
        if not self.is_registered or current_twin_rev == UNREGISTERED:
            return False
        return self.twin_rev == current_twin_rev
