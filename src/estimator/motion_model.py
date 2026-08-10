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

The velocity covariance floor (Day 22, Objective 2)
------------------------------------------------------
Day 21 diagnosed the cessation failure's mechanism directly: NEES up to
~800 in the frames right after a walker stops, decaying back to nominal
over ~10-14 frames as the filter's *posterior* velocity covariance —
tight from a long preceding walk, where repeated updates against an
accurate position sensor pull steady-state variance well below what any
single predict step's ``Q`` alone would inject — catches up to the fact
that velocity has actually gone to zero. Nothing in a constant-velocity
filter unlearns quickly; a stop is, by construction, an event the
filter's own acquired confidence says should not happen.

:func:`pedestrian_velocity_covariance_floor_mps2` is a **constraint**, not
a tuned parameter: it clamps a mode's posterior velocity-diagonal entries
so they may never fall below what :data:`PERSON_SIGMA_A_MPS2` — this
module's own already-declared pedestrian deceleration bound (see
``person``'s entry above) — implies is physically possible over one
timestep. It is derived once, from that one already-existing constant,
and applied verbatim; see the function's own docstring for the full
derivation. It is never fit or swept against ``scripts/eval_estimator.py``'s
output — see that function's docstring for what a fitted version of this
same number would mean.
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

PEDESTRIAN_STOP_DURATION_S = 1.0
"""A comfortable, voluntary pedestrian stop takes roughly a second,
essentially independent of how long or how steadily the person was
walking beforehand — stopping is a local decision, not something that
takes longer the more confidently a filter (or an observer) has been
tracking a steady gait. Declared, same convention as
:data:`PERSON_SIGMA_A_MPS2` itself (not fitted to any observed
trajectory) — used only as the narrative cross-check below, not as a
free parameter in the floor formula itself."""


def pedestrian_velocity_covariance_floor_mps2(dt_s: float) -> float:
    """The physically-derived floor under a person-kind (or asset_carried,
    which inherits it) mode's posterior velocity variance, per axis, at
    one timestep ``dt_s`` (Day 22, Objective 2).

    Derivation
    ----------
    :data:`PERSON_SIGMA_A_MPS2` (``1.5 m/s^2``) is already this module's
    declared pedestrian acceleration bound (see the module docstring's
    ``person`` entry) — cross-checked here by the same physical fact Day
    22 was asked to reason from: a commonly-cited comfortable adult
    walking pace is on the order of ~1-1.5 m/s (this project's own
    v3-indoor/v4.1-gate synthetic walkers move slower, ~0.5 m/s — see the
    Day-21 report — which only makes the bound below more conservative,
    not less), and :data:`PEDESTRIAN_STOP_DURATION_S` (~1s) is roughly how
    long a voluntary stop from that pace takes, so ``v_typical /
    PEDESTRIAN_STOP_DURATION_S`` lands in the same ~1-1.5 m/s^2 order of
    magnitude already declared. This is a cross-check that the existing
    constant is the right order of magnitude for "how fast can a person's
    velocity legitimately change", not a new, independently-fitted number
    — the formula below uses ``PERSON_SIGMA_A_MPS2`` directly, not a
    freshly-derived value.

    ``person``'s own ``Q`` (:func:`_cv_process_noise`) already asserts,
    every single predict step, that velocity uncertainty of up to
    ``(PERSON_SIGMA_A_MPS2 * dt_s)^2`` enters regardless of what the prior
    believed — that is what "process noise density" means. It is
    physically incoherent for the filter's *posterior* (after however
    many updates have narrowed it) to ever claim LESS velocity uncertainty
    than its own model already asserts is possible for one single step:
    doing so is exactly Day 21's diagnosed mechanism, a filter so
    confident in a converged velocity that a genuine stop becomes an
    ~800-sigma-squared event. The floor is therefore the same quantity
    ``Q``'s own velocity-block diagonal entry already computes::

        sigma_v_floor^2 = (PERSON_SIGMA_A_MPS2 * dt_s)^2

    Args:
        dt_s: The timestep this floor applies to (the same ``dt_s`` used
            to compute ``F``/``Q`` for that step).

    Raises:
        MotionModelError: if ``dt_s`` is negative.
    """
    if dt_s < 0:
        raise MotionModelError(f"dt_s must be non-negative, got {dt_s}")
    return (PERSON_SIGMA_A_MPS2 * dt_s) ** 2


def apply_velocity_covariance_floor(
    cov: FloatArray, floor_mps2: float | None
) -> FloatArray:
    """Clamp ``cov``'s three velocity-diagonal entries up to ``floor_mps2``;
    a no-op (returns ``cov`` unchanged) when ``floor_mps2`` is ``None`` --
    the model in question declares no floor for its kind.

    Only the diagonal is touched. Raising a PD matrix's diagonal entry by
    itself (holding every other entry fixed) is equivalent to adding a
    rank-1 PSD perturbation (``delta * e_i @ e_i.T``, ``delta >= 0``), so
    this can only preserve or improve positive-definiteness, never break
    it — the STRUCTURAL PD requirement every ``Q``/posterior covariance in
    this module already carries (see module docstring) is not at risk
    from this operation.
    """
    if floor_mps2 is None:
        return cov
    floored = cov.copy()
    for axis in range(3):
        idx = 3 + axis
        if floored[idx, idx] < floor_mps2:
            floored[idx, idx] = floor_mps2
    return floored


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
        velocity_variance_floor_enabled: Day 22, Objective 2. Only
            meaningful for ``kind in ("person", "asset_carried")`` — an
            ``asset_carried`` instance with this set inherits the SAME
            pedestrian-derived floor value a ``person`` instance would
            (:func:`pedestrian_velocity_covariance_floor_mps2`, always
            computed from :data:`PERSON_SIGMA_A_MPS2`, never from this
            instance's own — possibly inflated — ``sigma_a_mps2``): the
            floor represents the carrier's own body's physical stopping
            bound, not the carried object's process-noise budget. Rejected
            at construction for every other kind (§ ``__post_init__``) —
            "static/asset_static/fixture/maneuvering do not need it" is
            enforced, not left to the caller to remember.
    """

    kind: Literal["person", "asset_static", "asset_carried", "maneuvering"]
    sigma_a_mps2: float
    carrier_entity_id: str | None = None
    velocity_variance_floor_enabled: bool = False

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
        if self.velocity_variance_floor_enabled and self.kind not in (
            "person",
            "asset_carried",
        ):
            raise MotionModelError(
                "velocity_variance_floor_enabled is only meaningful for "
                f"kind='person' or 'asset_carried' (it inherits the carrier's "
                f"bound), got kind={self.kind!r} -- asset_static/fixture/"
                "maneuvering do not represent a pedestrian's own stopping "
                "physics and are not floored"
            )

    @property
    def sha(self) -> str:
        return _sha_for(
            self.kind,
            {
                "sigma_a_mps2": self.sigma_a_mps2,
                "carrier_entity_id": self.carrier_entity_id,
                "velocity_variance_floor_enabled": self.velocity_variance_floor_enabled,
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

    def velocity_covariance_floor_mps2(self, dt_s: float) -> float | None:
        """The pedestrian-derived floor for this timestep, or ``None`` if
        this instance does not carry one (Day 22, Objective 2). Always
        computed from :data:`PERSON_SIGMA_A_MPS2` — never from this
        instance's own ``sigma_a_mps2`` — see
        :attr:`velocity_variance_floor_enabled`'s docstring for why."""
        if not self.velocity_variance_floor_enabled:
            return None
        return pedestrian_velocity_covariance_floor_mps2(dt_s)


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

    def velocity_covariance_floor_mps2(self, dt_s: float) -> float | None:
        """A fixture never moves; it carries no velocity floor. See
        :meth:`ConstantVelocityMotionModel.velocity_covariance_floor_mps2`."""
        return None


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

    def velocity_covariance_floor_mps2(self, dt_s: float) -> float | None:
        """IMM's own ``static`` mode is not a :data:`MotionEntityKind` and
        Objective 2's per-entity-kind directive does not name it; it
        carries no velocity floor. See
        :meth:`ConstantVelocityMotionModel.velocity_covariance_floor_mps2`."""
        return None


MotionModel = (
    ConstantVelocityMotionModel | FixtureMotionModel | NearlyConstantPositionMotionModel
)


def motion_model_for(
    entity_kind: MotionEntityKind,
    *,
    carrier_entity_id: str | None = None,
    velocity_covariance_floor: bool = False,
) -> MotionModel:
    """The swappable-component factory: one call site per entity kind.

    Args:
        entity_kind: Which of :data:`MOTION_ENTITY_KINDS` to build.
        carrier_entity_id: Only accepted for ``asset_carried`` — see
            :class:`ConstantVelocityMotionModel`.
        velocity_covariance_floor: Day 22, Objective 2. Only valid for
            ``entity_kind in ("person", "asset_carried")`` — raises for
            ``asset_static``/``fixture`` rather than silently ignoring the
            request, since those kinds do not represent a pedestrian's own
            stopping physics (see :class:`ConstantVelocityMotionModel`'s
            docstring).
    """
    if entity_kind == "person":
        return ConstantVelocityMotionModel(
            kind="person",
            sigma_a_mps2=PERSON_SIGMA_A_MPS2,
            velocity_variance_floor_enabled=velocity_covariance_floor,
        )
    if entity_kind == "asset_static":
        if velocity_covariance_floor:
            raise MotionModelError(
                "velocity_covariance_floor=True is not valid for "
                "entity_kind='asset_static' -- an asset that should not be "
                "moving at all has no pedestrian stopping physics to floor "
                "against"
            )
        return ConstantVelocityMotionModel(
            kind="asset_static", sigma_a_mps2=ASSET_STATIC_SIGMA_A_MPS2
        )
    if entity_kind == "asset_carried":
        return ConstantVelocityMotionModel(
            kind="asset_carried",
            sigma_a_mps2=ASSET_CARRIED_SIGMA_A_MPS2,
            carrier_entity_id=carrier_entity_id,
            velocity_variance_floor_enabled=velocity_covariance_floor,
        )
    if entity_kind == "fixture":
        if velocity_covariance_floor:
            raise MotionModelError(
                "velocity_covariance_floor=True is not valid for "
                "entity_kind='fixture' -- a fixture does not move at all"
            )
        return FixtureMotionModel()
    raise MotionModelError(
        f"unknown entity_kind {entity_kind!r}; expected one of {MOTION_ENTITY_KINDS}"
    )
