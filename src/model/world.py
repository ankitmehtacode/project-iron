"""World-frame positions, and the twin_rev they are meaningless without.

The bug this closes (Day-13 falsification test 3)
----------------------------------------------------
A world-frame position storable without its ``twin_rev`` means
re-versioning the site's 3D twin (a camera recalibration, a remount, a
zone redrawn) silently changes what every historical position means —
the coordinates on disk do not change, but the frame they were measured
in has moved out from under them. That is precisely the failure §0 of
the data model exists to prevent: a stored fact whose meaning drifts with
no record of the drift.

:class:`WorldPosition` makes ``twin_rev`` a required, undefaulted field —
the same shape as Day 13's bitemporal ``Relationship``, which made a
single-timestamp relationship unconstructable by requiring four
undefaulted fields instead of one. Here, three coordinates plus
``twin_rev``, all undefaulted, mean a position with no stated frame
cannot be built.

Cross-twin_rev operations require an explicit transform
-----------------------------------------------------------
:meth:`WorldPosition.distance_to` and :meth:`WorldPosition.reproject` both
raise :class:`TwinRevError` when the two revisions differ and no
:class:`TwinRevTransform` is supplied. There is deliberately no
``__sub__``/``__add__``/ordering operator defined on :class:`WorldPosition`
at all — arithmetic on a bare Python object with no override raises
``TypeError`` by construction, so "silently do the wrong arithmetic
across revs" was never representable in the first place; the two
explicit methods are the only sanctioned operations, and both check.

:class:`TwinRevTransformRegistry` is how a caller obtains a
:class:`TwinRevTransform` for a rev pair that is not the trivial
same-rev case. STRUCTURAL: :meth:`TwinRevTransformRegistry.resolve`
raises for an unregistered pair rather than defaulting to identity —
silent identity is the bug a re-versioned twin produces, and this
registry refuses to reproduce it. Same-rev resolution is the one case
that never needs registration, because no re-versioning happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class TwinRevError(ValueError):
    """Raised when a world-frame position or transform operation crosses
    twin_rev without an explicit, resolvable :class:`TwinRevTransform`.
    """


@dataclass(frozen=True)
class RigidTransform3D:
    """A rotation plus translation between two 3D coordinate frames.

    Attributes:
        rotation: Row-major 3x3 matrix as a flat 9-tuple
            ``(r00, r01, r02, r10, r11, r12, r20, r21, r22)``.
        translation: ``(tx, ty, tz)`` in metres.
    """

    rotation: tuple[float, float, float, float, float, float, float, float, float]
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.rotation) != 9:
            raise ValueError(
                f"RigidTransform3D.rotation needs 9 elements, got {len(self.rotation)}"
            )
        if len(self.translation) != 3:
            raise ValueError(
                f"RigidTransform3D.translation needs 3 elements, got "
                f"{len(self.translation)}"
            )
        if not all(math.isfinite(v) for v in self.rotation):
            raise ValueError(
                f"RigidTransform3D.rotation must be finite, got {self.rotation}"
            )
        if not all(math.isfinite(v) for v in self.translation):
            raise ValueError(
                f"RigidTransform3D.translation must be finite, got {self.translation}"
            )

    @classmethod
    def identity(cls) -> "RigidTransform3D":
        return cls(
            rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            translation=(0.0, 0.0, 0.0),
        )

    def apply(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        """Map ``point`` through this transform: ``R @ point + t``."""
        r = self.rotation
        x, y, z = point
        tx, ty, tz = self.translation
        return (
            r[0] * x + r[1] * y + r[2] * z + tx,
            r[3] * x + r[4] * y + r[5] * z + ty,
            r[6] * x + r[7] * y + r[8] * z + tz,
        )

    def inverse(self) -> "RigidTransform3D":
        """The transform mapping back to this transform's source frame.

        Assumes ``rotation`` is orthonormal (a true rigid rotation, no
        scale or shear), which is the only kind this type represents:
        ``R^-1 = R^T`` and ``t' = -R^T t``.
        """
        r = self.rotation
        rt = (r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8])
        tx, ty, tz = self.translation
        itx = -(rt[0] * tx + rt[1] * ty + rt[2] * tz)
        ity = -(rt[3] * tx + rt[4] * ty + rt[5] * tz)
        itz = -(rt[6] * tx + rt[7] * ty + rt[8] * tz)
        return RigidTransform3D(rotation=rt, translation=(itx, ity, itz))


@dataclass(frozen=True)
class WorldPosition:
    """A metric position in the site's world frame, at a specific twin_rev.

    STRUCTURAL: all four fields are required with no default. There is no
    way to construct a world-frame position without stating which twin
    revision its coordinates are meaningful under — see the module
    docstring.
    """

    x_m: float
    y_m: float
    z_m: float
    twin_rev: int

    def __post_init__(self) -> None:
        for name, value in (("x_m", self.x_m), ("y_m", self.y_m), ("z_m", self.z_m)):
            if not math.isfinite(value):
                raise ValueError(f"WorldPosition.{name} must be finite, got {value}")
        if self.twin_rev < 0:
            raise ValueError(
                f"WorldPosition.twin_rev must be non-negative, got {self.twin_rev}"
            )

    def distance_to(
        self, other: "WorldPosition", transform: "TwinRevTransform | None" = None
    ) -> float:
        """Euclidean distance in metres.

        Raises:
            TwinRevError: if ``self`` and ``other`` carry different
                ``twin_rev`` and ``transform`` is not supplied, or is
                supplied but does not connect the two revisions.
        """
        if self.twin_rev == other.twin_rev:
            dx = self.x_m - other.x_m
            dy = self.y_m - other.y_m
            dz = self.z_m - other.z_m
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        if transform is None:
            raise TwinRevError(
                f"cannot compute distance between a twin_rev={self.twin_rev} "
                f"position and a twin_rev={other.twin_rev} position without "
                "an explicit TwinRevTransform connecting them"
            )
        other_here = other.reproject(transform)
        return self.distance_to(other_here)

    def reproject(self, transform: "TwinRevTransform") -> "WorldPosition":
        """Re-express this position under a different twin_rev, explicitly.

        This is how a re-versioned twin keeps historical positions
        interpretable: never by mutating them (they are frozen) or by
        silently reinterpreting their coordinates, but by producing a new
        ``WorldPosition`` through a transform whose provenance (which rev
        pair) is explicit at the call site.

        Raises:
            TwinRevError: if ``transform.from_twin_rev`` does not match
                ``self.twin_rev``.
        """
        if transform.from_twin_rev != self.twin_rev:
            raise TwinRevError(
                f"transform is from twin_rev={transform.from_twin_rev} but "
                f"this position is twin_rev={self.twin_rev}"
            )
        x, y, z = transform.transform.apply((self.x_m, self.y_m, self.z_m))
        return WorldPosition(x_m=x, y_m=y, z_m=z, twin_rev=transform.to_twin_rev)


@dataclass(frozen=True)
class TwinRevTransform:
    """A rigid transform between two named twin revisions.

    Attributes:
        from_twin_rev: Source revision.
        to_twin_rev: Target revision.
        transform: The rigid transform. When ``from_twin_rev ==
            to_twin_rev`` this must be :meth:`RigidTransform3D.identity`
            — a same-rev transform that is not the identity is a
            contradiction, not a measurement.
    """

    from_twin_rev: int
    to_twin_rev: int
    transform: RigidTransform3D

    def __post_init__(self) -> None:
        if self.from_twin_rev < 0 or self.to_twin_rev < 0:
            raise TwinRevError("twin_rev values must be non-negative")
        if (
            self.from_twin_rev == self.to_twin_rev
            and self.transform != RigidTransform3D.identity()
        ):
            raise TwinRevError(
                "a TwinRevTransform between a twin_rev and itself must be "
                "the identity transform"
            )

    @classmethod
    def identity_for(cls, twin_rev: int) -> "TwinRevTransform":
        return cls(
            from_twin_rev=twin_rev,
            to_twin_rev=twin_rev,
            transform=RigidTransform3D.identity(),
        )


class TwinRevTransformRegistry:
    """Known rigid relationships between twin_rev pairs.

    Nothing here defaults to identity across differing revisions — that
    silent assumption is exactly the bug this module exists to close.
    Same-rev resolution is the sole exception, because no re-versioning
    happened and there is nothing to measure.
    """

    def __init__(self) -> None:
        self._transforms: dict[tuple[int, int], TwinRevTransform] = {}

    def register(self, transform: TwinRevTransform) -> None:
        """Register a measured relationship between two distinct revisions.

        The inverse direction is registered automatically — a measured
        rigid relationship is symmetric by construction.

        Raises:
            TwinRevError: if ``transform`` is a same-rev (identity)
                transform; those need no registration and registering one
                would suggest a re-versioning that did not occur.
        """
        if transform.from_twin_rev == transform.to_twin_rev:
            raise TwinRevError(
                "refusing to register a same-rev transform: identity is "
                "always available for a twin_rev compared with itself, and "
                "registering one here would misrepresent it as a measured "
                "re-versioning"
            )
        key = (transform.from_twin_rev, transform.to_twin_rev)
        self._transforms[key] = transform
        inverse_key = (transform.to_twin_rev, transform.from_twin_rev)
        self._transforms[inverse_key] = TwinRevTransform(
            from_twin_rev=transform.to_twin_rev,
            to_twin_rev=transform.from_twin_rev,
            transform=transform.transform.inverse(),
        )

    def resolve(self, from_twin_rev: int, to_twin_rev: int) -> TwinRevTransform:
        """Return the transform from ``from_twin_rev`` to ``to_twin_rev``.

        Raises:
            TwinRevError: if the two revisions differ and no relationship
                between them has been registered. Never falls back to
                identity.
        """
        if from_twin_rev == to_twin_rev:
            return TwinRevTransform.identity_for(from_twin_rev)
        key = (from_twin_rev, to_twin_rev)
        if key not in self._transforms:
            raise TwinRevError(
                f"no registered transform from twin_rev={from_twin_rev} to "
                f"twin_rev={to_twin_rev}; silent identity is not assumed "
                "across a twin re-version — register the measured "
                "relationship first"
            )
        return self._transforms[key]
