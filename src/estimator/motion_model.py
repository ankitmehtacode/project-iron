"""Motion models: the state-transition half of the estimator contract (§15).

A ``MotionModel`` answers exactly two questions for one ``entity_kind``: how
does the state propagate over ``dt`` (:meth:`f`, via the linear transition
matrix :meth:`F`), and how much process noise enters while it does
(:meth:`Q`). Both are swappable per entity kind, both are versioned by a
content ``sha`` so two state records can be compared only when they agree on
which model produced them (§15's ``motion_model_sha``), and neither model
ever sees an observation — that is the measurement model's job
(:mod:`src.estimator.measurement_model`).

State vector, fixed for every kind so :class:`~src.estimator.filter` never
has to branch on dimensionality::

    [x_m, y_m, z_m, vx_mps, vy_mps, vz_mps]

world-frame metres and metres/second, matching v3-indoor's ``agent_xyz`` GT
and :class:`~src.model.world.WorldPosition`.

Four kinds, one shared mechanism
---------------------------------
Every kind below is a constant-velocity (CV) transition with a different
process-noise density, except ``fixture``, which has no kinematics at all.
That is a deliberate simplification, not an oversight:

- ``person``: CV with a pedestrian acceleration bound (~1.5 m/s^2) — people
  start, stop, and turn, and the process noise says how much unmodeled
  acceleration the filter should tolerate between updates.
- ``asset_static``: CV with a very small acceleration bound. An asset that
  should not be moving at all is not modeled as "zero velocity" (a state
  the filter could never leave once wrong) but as "extremely reluctant to
  acquire velocity", which keeps the state high-confidence without making
  the estimator structurally unable to notice the asset actually moved.
- ``asset_carried``: CV with an inflated acceleration bound relative to
  ``person``. §15's real design is rigid coupling to a carrier entity's own
  trajectory plus slip noise, which is multi-entity work (a carried asset's
  true motion source is another entity's state, not its own). Today's scope
  is single-entity: the interface carries ``carrier_entity_id`` as the hook
  a later day wires up, and until then a carried asset behaves as CV with
  more process noise than a person, reflecting the extra, currently-
  unmodeled dynamics of being carried by someone whose own motion this
  model cannot see.
- ``fixture``: identity transition, (near-)zero process noise. A zone, a
  door, a camera does not move; the state estimate should never drift.

Positive-definite Q, deliberately regularized
------------------------------------------------
The textbook discretized white-noise-acceleration block for one axis,
``q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]``, has determinant exactly zero —
it is positive *semi*-definite, not positive *definite*, because it is the
discretization of a single scalar noise source driving two states. Stacked
block-diagonally across three axes the full 6x6 matrix is still only rank 3.
:data:`_Q_REGULARIZATION` adds a small floor to every diagonal entry so
:meth:`Q` is strictly PD, which downstream Cholesky/inverse operations in
the filter require and which "Q is positive-definite" (Day 20's own test
requirement) means literally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

STATE_DIM = 6
"""[x, y, z, vx, vy, vz] — fixed across every entity kind."""

MotionEntityKind = Literal["person", "asset_static", "asset_carried", "fixture"]
"""The estimator's own kinematic-behaviour taxonomy.

Deliberately NOT :data:`src.model.entity.EntityKind` (``person``, ``asset``,
``vehicle``, ``door``, ``zone``, ``camera``, ``container``) or
:data:`src.model.entity.IdentityClass`. Those classify *what an entity is*
for identity and persistence purposes; this classifies *how it moves* for
prediction purposes, and the two taxonomies do not align — ``EntityKind``
has no static/carried distinction for ``asset`` at all, because that
distinction has never mattered until there was a filter to configure with
it. A caller maps its own entity kind to this one at the estimator's
boundary; that mapping is a product-configuration decision, not something
this module should silently assume.
"""
MOTION_ENTITY_KINDS: tuple[MotionEntityKind, ...] = get_args(MotionEntityKind)

# Acceleration process-noise densities (m/s^2), the one free parameter per
# kind. Set from the kind's PURPOSE (see the module docstring), not fitted
# to any observed trajectory — there is no real trajectory data yet to fit
# to, and a parameter chosen to match a specific run would stop meaning what
# its name says the moment a different run arrived.
PERSON_SIGMA_A_MPS2 = 1.5
ASSET_STATIC_SIGMA_A_MPS2 = 0.02
ASSET_CARRIED_SIGMA_A_MPS2 = 4.0

_Q_REGULARIZATION = 1e-6
"""Diagonal floor added to every Q so it is strictly PD. See module docstring."""


class MotionModelError(ValueError):
    """Raised when a motion model or its parameters are malformed."""


def _sha_for(kind: str, params: dict[str, float | str | None]) -> str:
    canonical = json.dumps({"kind": kind, "params": params}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cv_transition(dt_s: float) -> FloatArray:
    """The constant-velocity F(dt): position += velocity * dt, velocity held."""
    f = np.eye(STATE_DIM, dtype=np.float64)
    for axis in range(3):
        f[axis, axis + 3] = dt_s
    return f


def _cv_process_noise(dt_s: float, sigma_a_mps2: float) -> FloatArray:
    """Discretized white-noise-acceleration Q, block-diagonal per axis, regularized."""
    q = sigma_a_mps2**2
    dt2, dt3, dt4 = dt_s**2, dt_s**3, dt_s**4
    Q = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    for axis in range(3):
        pos, vel = axis, axis + 3
        Q[pos, pos] = q * dt4 / 4.0
        Q[pos, vel] = Q[vel, pos] = q * dt3 / 2.0
        Q[vel, vel] = q * dt2
    Q += _Q_REGULARIZATION * np.eye(STATE_DIM, dtype=np.float64)
    return Q


@dataclass(frozen=True)
class ConstantVelocityMotionModel:
    """CV transition shared by ``person``, ``asset_static``, ``asset_carried``,
    and (Day 21) IMM's ``maneuvering`` mode.

    Attributes:
        kind: Which entity kind — or, for ``"maneuvering"``, which IMM
            kinematic regime — this instance was configured for.
            ``"maneuvering"`` is distinct from :data:`MotionEntityKind`
            entirely: it is IMM's own high-process-noise mode
            (:mod:`src.estimator.imm`), never reachable through
            :func:`motion_model_for`'s entity-kind dispatch — a kinematic
            *regime* label, not an entity-kind label, sharing this class
            only because the underlying CV math is identical.
        sigma_a_mps2: Acceleration process-noise density (m/s^2).
        carrier_entity_id: §15's rigid-coupling hook for ``asset_carried``.
            Unused today — this is the interface the multi-entity day wires
            up, not a functioning coupling. ``None`` for every kind except
            ``asset_carried``, where it is *allowed* to be set (recorded for
            forward compatibility) but has no effect on :meth:`F` or
            :meth:`Q` yet; a carried asset with no known carrier and one
            with a known carrier compute identically until coupling lands.
    """

    kind: Literal["person", "asset_static", "asset_carried", "maneuvering"]
    sigma_a_mps2: float
    carrier_entity_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("person", "asset_static", "asset_carried", "maneuvering"):
            raise MotionModelError(
                f"ConstantVelocityMotionModel.kind must be one of person/"
                f"asset_static/asset_carried/maneuvering, got {self.kind!r}"
            )
        if not (self.sigma_a_mps2 > 0.0) or not np.isfinite(self.sigma_a_mps2):
            raise MotionModelError(
                f"sigma_a_mps2 must be a finite positive number, got "
                f"{self.sigma_a_mps2}"
            )
        if self.carrier_entity_id is not None and self.kind != "asset_carried":
            raise MotionModelError(
                f"carrier_entity_id is only meaningful for kind='asset_carried', "
                f"got kind={self.kind!r} with carrier_entity_id set"
            )

    @property
    def sha(self) -> str:
        return _sha_for(
            self.kind,
            {
                "sigma_a_mps2": self.sigma_a_mps2,
                "carrier_entity_id": self.carrier_entity_id,
            },
        )

    def F(self, dt_s: float) -> FloatArray:
        if dt_s < 0:
            raise MotionModelError(f"dt_s must be non-negative, got {dt_s}")
        return _cv_transition(dt_s)

    def Q(self, dt_s: float) -> FloatArray:
        if dt_s < 0:
            raise MotionModelError(f"dt_s must be non-negative, got {dt_s}")
        return _cv_process_noise(dt_s, self.sigma_a_mps2)

    def f(self, state: FloatArray, dt_s: float) -> FloatArray:
        result: FloatArray = self.F(dt_s) @ state
        return result


@dataclass(frozen=True)
class FixtureMotionModel:
    """No kinematics: the state never changes, and process noise stays at
    the regularization floor rather than genuine zero (see module docstring
    on why Q must be strictly PD)."""

    kind: Literal["fixture"] = "fixture"

    @property
    def sha(self) -> str:
        return _sha_for(self.kind, {})

    def F(self, dt_s: float) -> FloatArray:
        if dt_s < 0:
            raise MotionModelError(f"dt_s must be non-negative, got {dt_s}")
        return np.eye(STATE_DIM, dtype=np.float64)

    def Q(self, dt_s: float) -> FloatArray:
        if dt_s < 0:
            raise MotionModelError(f"dt_s must be non-negative, got {dt_s}")
        return _Q_REGULARIZATION * np.eye(STATE_DIM, dtype=np.float64)

    def f(self, state: FloatArray, dt_s: float) -> FloatArray:
        result: FloatArray = self.F(dt_s) @ state
        return result


@dataclass(frozen=True)
class NearlyConstantPositionMotionModel:
    """Position holds still; velocity is not used to predict it (Day 21).

    Distinct from a :class:`ConstantVelocityMotionModel` tuned with a tiny
    ``sigma_a_mps2``: that class shares the same F as every other CV
    instance — position += velocity * dt — so a nonzero velocity, however
    it entered the state (e.g. mixed in from another IMM mode), keeps
    getting propagated forward, just with a tighter, more confident
    covariance around doing so. Built for IMM's "static" mode
    (:mod:`src.estimator.imm`) after Day 21 found exactly that failure: a
    tight-Q CV "static" mode does not resist predicting motion, it just
    claims more confidence while doing it, and wins the mode-probability
    contest against a genuinely moving target purely on covariance width
    — a race the mode was never supposed to be in.

    This model's F has **no velocity-to-position coupling at all**:
    predicted position next step equals current position next step,
    regardless of whatever velocity is sitting in the state vector. An
    entity in this mode does not move, however confident or not that
    claim is — which is what "static" needs to mean for the mode contest
    to be decided by evidence (the actual innovation) rather than by
    which mode happens to claim the tightest prior.
    """

    kind: Literal["static_position"] = "static_position"
    sigma_position_mps_sqrt_s: float = 0.05
    """Position process-noise density (m / sqrt(s)) — small sway, not zero
    (a person standing is not a fixture; see :class:`FixtureMotionModel`
    for the truly-immobile case)."""
    sigma_velocity_mps2: float = 0.05
    """Velocity process-noise density. Kept small and nonzero rather than
    frozen at whatever mixing handed this mode, so a genuine transition
    OUT of this mode (the entity starts moving) is not permanently
    fighting a velocity state stuck at a stale value."""

    def __post_init__(self) -> None:
        if not (self.sigma_position_mps_sqrt_s > 0.0) or not np.isfinite(
            self.sigma_position_mps_sqrt_s
        ):
            raise MotionModelError(
                f"sigma_position_mps_sqrt_s must be finite and positive, got "
                f"{self.sigma_position_mps_sqrt_s}"
            )
        if not (self.sigma_velocity_mps2 > 0.0) or not np.isfinite(
            self.sigma_velocity_mps2
        ):
            raise MotionModelError(
                f"sigma_velocity_mps2 must be finite and positive, got "
                f"{self.sigma_velocity_mps2}"
            )

    @property
    def sha(self) -> str:
        return _sha_for(
            self.kind,
            {
                "sigma_position_mps_sqrt_s": self.sigma_position_mps_sqrt_s,
                "sigma_velocity_mps2": self.sigma_velocity_mps2,
            },
        )

    def F(self, dt_s: float) -> FloatArray:
        if dt_s < 0:
            raise MotionModelError(f"dt_s must be non-negative, got {dt_s}")
        # Identity: position and velocity both hold at their prior value,
        # and critically, position does NOT advance by velocity * dt.
        return np.eye(STATE_DIM, dtype=np.float64)

    def Q(self, dt_s: float) -> FloatArray:
        if dt_s < 0:
            raise MotionModelError(f"dt_s must be non-negative, got {dt_s}")
        Q = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        position_var = (self.sigma_position_mps_sqrt_s**2) * dt_s
        velocity_var = (self.sigma_velocity_mps2**2) * dt_s
        for axis in range(3):
            Q[axis, axis] = position_var
            Q[axis + 3, axis + 3] = velocity_var
        Q += _Q_REGULARIZATION * np.eye(STATE_DIM, dtype=np.float64)
        return Q

    def f(self, state: FloatArray, dt_s: float) -> FloatArray:
        result: FloatArray = self.F(dt_s) @ state
        return result


MotionModel = (
    ConstantVelocityMotionModel | FixtureMotionModel | NearlyConstantPositionMotionModel
)


def motion_model_for(
    entity_kind: MotionEntityKind, *, carrier_entity_id: str | None = None
) -> MotionModel:
    """The swappable-component factory: one call site per entity kind.

    Args:
        entity_kind: Which of :data:`MOTION_ENTITY_KINDS` to build.
        carrier_entity_id: Only accepted for ``asset_carried`` — see
            :class:`ConstantVelocityMotionModel`.
    """
    if entity_kind == "person":
        return ConstantVelocityMotionModel(
            kind="person", sigma_a_mps2=PERSON_SIGMA_A_MPS2
        )
    if entity_kind == "asset_static":
        return ConstantVelocityMotionModel(
            kind="asset_static", sigma_a_mps2=ASSET_STATIC_SIGMA_A_MPS2
        )
    if entity_kind == "asset_carried":
        return ConstantVelocityMotionModel(
            kind="asset_carried",
            sigma_a_mps2=ASSET_CARRIED_SIGMA_A_MPS2,
            carrier_entity_id=carrier_entity_id,
        )
    if entity_kind == "fixture":
        return FixtureMotionModel()
    raise MotionModelError(
        f"unknown entity_kind {entity_kind!r}; expected one of {MOTION_ENTITY_KINDS}"
    )
