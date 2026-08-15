"""Ground-truth kinematic tracks that carry their own axis convention.

The bug this kills
------------------
Day 29's first GT constraint run reported 16 ``gravity_floor_transition``
violations at up to -36.58 m/s^2 — 3.7x free fall — and every one of them
was the measuring script reading ``agent_xyz`` index 2 as vertical. Index
2 is camera-facing depth; index 1 is height. The script cited
``src/model/world.py``'s ``+z``-up world frame, which is a real convention
in this codebase, just not the one that array is in.

Nothing raised. An abrupt depth-axis speed change reads exactly like an
impossible fall when the axis is mislabelled, so the wrong number was
correctly shaped, physically alarming, and completely fictitious.

Two conventions genuinely live in this repository, and both are correct
where they are declared:

===========================  ==============  ====================================
Convention                   Vertical axis   Declared by
===========================  ==============  ====================================
:attr:`GroundTruthAxes.X_Y_UP_Z`   index 1   ``scripts/gen_synthetic_indoor.py``,
                                             which builds every GT position as
                                             ``[x, height_m / 2.0, z]``
:attr:`GroundTruthAxes.X_Y_Z_UP`   index 2   ``src/model/world.py``'s site world
                                             frame
===========================  ==============  ====================================

The defect was never that one of them is wrong. It was that ``agent_xyz``
is a bare ``np.ndarray`` crossing a module boundary between the generator,
the scorer, and the estimator, carrying no statement of which one it is
in — so the two modules' shared assumption could diverge, and did,
silently. That is the same class as Day 24's registry/consumer drift.

What is structural here
-----------------------
**A GT array with no declared convention is unconstructable.**
:class:`GtPositionTrack`, :class:`GtVelocityTrack`, and
:class:`GtAccelerationTrack` all take ``axes`` as a required, undefaulted
field — the same treatment :class:`~src.model.world.WorldPosition` gives
``twin_rev``, and for the same reason: there is no sensible default, and a
default is exactly how an unstated convention becomes a silent one.

**Mixing conventions in one operation raises.** Every binary operation
goes through :meth:`_GtTrack.combine`, which raises
:class:`~src.contracts.errors.AxisConventionMismatch` on disagreeing
``axes``. There is no coercion path: an axis permutation is exact and
lossless, so it is tempting to apply one automatically, and that is
precisely the auto-conversion the contracts layer forbids — it would
recreate the original bug with more confidence attached. A caller that
genuinely wants the other convention names it, at the call site, via
:meth:`_GtTrack.values_in`.

**Units are carried by the type, not by a field.** Position, velocity,
and acceleration are three classes, not one class with a units string, so
handing a velocity to something expecting a position is a ``mypy`` error
rather than a value that is wrong by a factor of ``dt``.

Differentiation, and the two frames that mean nothing
-----------------------------------------------------
:meth:`GtPositionTrack.differentiate` implements this project's standing
finite-difference convention exactly as
:func:`src.estimator.regime.classify_track` and
``scripts/eval_estimator.py`` already do it: forward difference, with
frame 0 mirroring frame 1 because there is no frame -1.

That mirror has a consequence worth stating once here rather than
rediscovering per script: differentiating TWICE makes index 1 identically
zero, for every track, regardless of what the track does — ``v[0] == v[1]``
implies ``a[1] == 0``. Index 0 is likewise a mirror, not a measurement. So
a genuine acceleration distribution starts at index 2, and
:attr:`GtAccelerationTrack.first_meaningful_index` says so rather than
leaving each consumer to work it out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from src.contracts.errors import AxisConventionMismatch

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, see below
    from src.model.world import WorldPositionArray

FloatArray = npt.NDArray[np.float64]


class GroundTruthAxes(Enum):
    """Which index of a ``[..., 3]`` GT array is vertical.

    An enum rather than a bare int so that the vertical index cannot be
    passed where a horizontal one is expected, and so that the two
    conventions this repository actually contains are enumerable — a
    third one appearing means adding a member here, in view of both
    existing ones, rather than a magic ``2`` appearing in a new script.
    """

    X_Y_UP_Z = "x, y_up, z_depth"
    """The synthetic generator's convention: index 1 is height, index 2 is
    camera-facing depth. ``scripts/gen_synthetic_indoor.py`` writes every
    ``agent_xyz`` in this frame (``np.array([x, self.height_m / 2.0, z])``).
    """

    X_Y_Z_UP = "x, y, z_up"
    """The site world frame's convention: index 2 is height. Declared by
    :mod:`src.model.world`. Every ``WorldPosition``/``WorldPositionArray``
    is in this frame."""

    @property
    def vertical_index(self) -> int:
        """Index of the vertical (up-positive) component."""
        return 1 if self is GroundTruthAxes.X_Y_UP_Z else 2

    @property
    def ground_plane_indices(self) -> tuple[int, int]:
        """Indices of the two horizontal components, in order."""
        return (0, 2) if self is GroundTruthAxes.X_Y_UP_Z else (0, 1)

    def permutation_to(self, other: "GroundTruthAxes") -> tuple[int, int, int]:
        """Index order that re-expresses this convention's components in
        ``other``'s.

        The mapping is a pure index permutation: exact, lossless, and its
        own inverse. It is still never applied implicitly — see
        :meth:`_GtTrack.values_in`.

        It is a transposition of two axes, so it REFLECTS rather than
        rotates: handedness flips. That is harmless for positions,
        velocities, accelerations, and every magnitude taken from them —
        all component-wise or norm-based — and it would be wrong for a
        cross product or a torque. Nothing here computes one; a future
        caller that does must compose a proper rotation instead of reusing
        this.
        """
        if self is other:
            return (0, 1, 2)
        # The two members differ only by swapping the vertical axis with
        # the second horizontal one, in either direction.
        return (0, 2, 1)


GENERATOR_AXES = GroundTruthAxes.X_Y_UP_Z
"""The convention every ``.npz`` clip this project generates is written in.

Named so that a loader states the fact once, citing the generator, instead
of each consumer restating an integer. Verified against
``scripts/gen_synthetic_indoor.py``'s ``Agent.position_at`` /
``MultiSegmentAgent.position_at``, not assumed from any other module's
frame."""


@dataclass(frozen=True, eq=False)
class _GtTrack:
    """Shared machinery for a ``[T, 3]`` ground-truth kinematic track.

    ``eq=False`` for the same reason :class:`~src.contracts.fields.DepthField`
    disables it: dataclass equality would compare ``values`` with bare
    ``==`` and return an array, which is ambiguous in a boolean context.

    Not exported. Consumers hold one of the three concrete subclasses, so
    that the quantity's units are part of its type.
    """

    values: FloatArray
    axes: GroundTruthAxes

    UNITS: ClassVar[str] = "unspecified"
    """Units of ``values``, for error messages. Overridden by each concrete
    subclass. A ``ClassVar``, not a field: units are a property of the
    quantity's TYPE, and a per-instance units field would be a value a
    caller could set wrong — which is the whole failure mode
    :class:`~src.contracts.fields.DepthField` had to guard at runtime and
    this hierarchy gets to make impossible."""

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(
                f"{type(self).__name__}.values must be [T, 3], got shape "
                f"{array.shape}"
            )
        if not isinstance(self.axes, GroundTruthAxes):
            raise AxisConventionMismatch(
                f"{type(self).__name__}.axes must be a GroundTruthAxes, got "
                f"{self.axes!r}. There is no default: an array whose axis "
                "convention is unstated is exactly the boundary this type "
                "exists to close."
            )
        object.__setattr__(self, "values", array)

    @property
    def frames(self) -> int:
        """Number of frames ``T``."""
        return int(self.values.shape[0])

    def values_in(self, axes: GroundTruthAxes) -> FloatArray:
        """This track's values re-expressed in ``axes``, named explicitly.

        The one sanctioned way out of the envelope into a bare array. It
        takes the target convention as a required argument so the call
        site records which one it wanted — ``track.values_in(
        GroundTruthAxes.X_Y_UP_Z)`` is a statement a reviewer can check,
        where ``track.values`` would be a statement nobody made.
        """
        if axes is self.axes:
            return self.values
        return self.values[:, list(self.axes.permutation_to(axes))]

    def vertical_component(self) -> FloatArray:
        """``[T]`` of the vertical component, using the DECLARED convention.

        The operation Day 29 got wrong. There is no way to call it against
        the wrong axis without having constructed the track with the wrong
        ``axes`` in the first place, which is a statement, not an
        oversight.
        """
        return self.values[:, self.axes.vertical_index]

    def ground_plane(self) -> FloatArray:
        """``[T, 2]`` of the two horizontal components, in index order."""
        return self.values[:, list(self.axes.ground_plane_indices)]

    def magnitude(self) -> FloatArray:
        """``[T]`` Euclidean norm — invariant under any axis permutation,
        so this is the one quantity here that cannot be got wrong by
        mislabelling the vertical axis. Stated because that invariance is
        why several existing consumers were unaffected by the Day-29 bug,
        and a reader should be able to tell which those were."""
        return np.asarray(np.linalg.norm(self.values, axis=1), dtype=np.float64)

    def ground_plane_magnitude(self) -> FloatArray:
        """``[T]`` horizontal speed/acceleration magnitude."""
        return np.asarray(np.linalg.norm(self.ground_plane(), axis=1), dtype=np.float64)

    def combine(self, other: "_GtTrack") -> tuple[FloatArray, FloatArray]:
        """Return both raw arrays if the two tracks agree on convention.

        The single gate every binary operation goes through, mirroring
        :meth:`src.model.world.WorldPositionArray.combine`.

        Raises:
            AxisConventionMismatch: if the two tracks declare different
                ``axes``. No transform parameter, deliberately: the caller
                converts with :meth:`values_in` at a visible call site, or
                does not combine them.
            TypeError: if the two tracks are different quantities (a
                position and a velocity), which is a units error that
                ``mypy`` also catches statically.
        """
        if type(self) is not type(other):
            raise TypeError(
                f"cannot combine {type(self).__name__} ({self.UNITS}) with "
                f"{type(other).__name__} ({other.UNITS}): different physical "
                "quantities, not different views of one"
            )
        if self.axes is not other.axes:
            raise AxisConventionMismatch(
                f"cannot combine a {self.axes.name} track with a "
                f"{other.axes.name} track. Convert explicitly with "
                "values_in(...) at the call site; no implicit permutation "
                "is applied, because an automatic conversion is how the "
                "convention became implicit in the first place."
            )
        return self.values, other.values

    def _differentiated(self, dt_s: float) -> FloatArray:
        """Forward difference with frame 0 mirroring frame 1 — this
        project's standing convention. See the module docstring for what
        the mirror costs when applied twice."""
        if dt_s <= 0.0:
            raise ValueError(
                f"dt_s must be positive, got {dt_s}; a non-positive timestep "
                "makes the difference quotient meaningless rather than large"
            )
        derivative = np.zeros_like(self.values)
        if self.frames >= 2:
            derivative[1:] = (self.values[1:] - self.values[:-1]) / dt_s
            derivative[0] = derivative[1]
        return derivative


@dataclass(frozen=True, eq=False)
class GtPositionTrack(_GtTrack):
    """``[T, 3]`` ground-truth positions in metres."""

    UNITS: ClassVar[str] = "m"

    def differentiate(self, dt_s: float) -> "GtVelocityTrack":
        """Ground-truth velocity, in the same axis convention."""
        return GtVelocityTrack(self._differentiated(dt_s), self.axes)

    def as_world_position_array(self, twin_rev: int) -> "WorldPositionArray":
        """This track as a :class:`~src.model.world.WorldPositionArray`,
        permuted into the site world frame's ``+z``-up convention.

        The single place the generator-frame -> world-frame permutation is
        written. ``twin_rev`` is required and undefaulted for the same
        reason ``axes`` is: ``WorldPositionArray`` will not accept a
        position whose twin revision is unstated, and this bridge does not
        get to guess it on the caller's behalf.

        Imported lazily: :mod:`src.contracts` must not depend on
        :mod:`src.model` at import time — the contracts layer sits under
        the data model, not beside it.
        """
        from src.model.world import WorldPositionArray

        return WorldPositionArray(
            xyz_m=self.values_in(GroundTruthAxes.X_Y_Z_UP), twin_rev=twin_rev
        )


@dataclass(frozen=True, eq=False)
class GtVelocityTrack(_GtTrack):
    """``[T, 3]`` ground-truth velocities in metres per second.

    Index 0 is a mirror of index 1, not a measurement — see the module
    docstring.
    """

    UNITS: ClassVar[str] = "m/s"

    def differentiate(self, dt_s: float) -> "GtAccelerationTrack":
        """Ground-truth acceleration, in the same axis convention."""
        return GtAccelerationTrack(self._differentiated(dt_s), self.axes)

    def speed(self) -> FloatArray:
        """``[T]`` scalar speed. A magnitude, so axis-convention-invariant."""
        return self.magnitude()


@dataclass(frozen=True, eq=False)
class GtAccelerationTrack(_GtTrack):
    """``[T, 3]`` ground-truth accelerations in metres per second squared.

    Indices 0 AND 1 are artifacts of the differencing convention rather
    than measurements — see :attr:`FIRST_MEANINGFUL_INDEX` and the module
    docstring.
    """

    UNITS: ClassVar[str] = "m/s^2"

    FIRST_MEANINGFUL_INDEX: ClassVar[int] = 2
    """First index whose value is a measurement rather than a convention
    artifact. Two, because acceleration needs three positions and the
    frame-0 velocity mirror forces ``a[1] == 0`` for every track."""

    def meaningful(self) -> "GtAccelerationTrack":
        """This track with the leading convention-artifact frames dropped.

        Returns an empty track (``frames == 0``) rather than raising when
        the source is too short: "this track cannot support an
        acceleration" is a fact about the data, and a caller counting
        frames should see a zero, not an exception it has to translate
        back into one.
        """
        return GtAccelerationTrack(
            self.values[self.FIRST_MEANINGFUL_INDEX :], self.axes
        )


def gt_position_track(values: FloatArray, axes: GroundTruthAxes) -> GtPositionTrack:
    """Convenience constructor, for symmetry with the read path.

    Exists so that a caller reading ``agent_xyz`` out of an ``.npz`` has a
    single named function to reach for, rather than each script writing
    its own ``np.asarray(..., dtype=np.float64)`` and picking an axis
    convention by memory.
    """
    return GtPositionTrack(np.asarray(values, dtype=np.float64), axes)
