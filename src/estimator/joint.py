"""Multi-entity joint estimation over a Component (Day 26, Objective 3).

Fills the gap left explicitly open since Day 13 (see
:mod:`src.model.episode`'s module docstring) and the interface Day 20 built
for it (:attr:`~src.estimator.motion_model.ConstantVelocityMotionModel.
carrier_entity_id`, unused until today). Conditional on Day 26 Objective 2's
own gate: that objective measured a narrow, data-bounded pass ("p95
component size <= 6 on the one golden set with multi-agent clips") and
carried the caveat forward explicitly rather than treating it as settled
physics — see ``docs/adr/0011-multi-entity-factor-graph.md``.

Components as the unit of inference
--------------------------------------
A :class:`Component` is one carrier plus zero or more entities it carries,
declared statically by the caller (see "Not implemented today" below for
why this is a real scope limit, not an oversight). A component with no
carried entities is not "joint" at all: :func:`run_joint_filter` delegates
such a component's ENTIRE run to
:func:`~src.estimator.filter.run_single_entity_filter` verbatim — not a
reimplementation that happens to agree, the actual function call — so a
size-1 component reproduces Day 20/25's single-entity behaviour
bit-for-bit by construction, never by coincidence. Components stay
independent of each other: nothing in this module ever solves two
unrelated components jointly, and there is no cross-component state.

Rigid coupling plus slip, not independent CV (the motivating case)
----------------------------------------------------------------------
A carried entity's own kinematics are not tracked. Its state is a 3D
``offset`` from its carrier's position (``carried_absolute_position =
carrier_position + offset``) that is nearly constant between updates ("rigid")
with a small process-noise density representing how much the object shifts
relative to the carrier's body while held ("slip") — exactly the design
:mod:`src.estimator.motion_model`'s ``asset_carried`` docstring named since
Day 20 ("rigid coupling to a carrier entity's own trajectory plus slip
noise") and left unimplemented pending this day. This is why "if A carries
laptop 7 and A moves, laptop 7's posterior must move": every PREDICT step
advances the carrier's position through its own motion model, and the
carried entity's implied absolute position (:meth:`JointStateEstimate.
carried_position_m`) moves with it automatically — even across steps with
no new observation of the carried entity at all, which is the entire
product value of coupling: the carrier's motion informs the carried
entity's estimate between the object's own, possibly sparse, detections.

State layout, per component: ``[carrier_x, carrier_y, carrier_z,
carrier_vx, carrier_vy, carrier_vz, offset_1_x, offset_1_y, offset_1_z,
...]`` — 6 + 3*k dimensions for k carried entities, in
``component.carried_entity_ids`` order. Predict: the carrier's 6x6 block
uses its own motion model's F/Q verbatim; each offset's 3x3 block is
identity in F (rigid) with a declared, small process-noise density in Q
(slip); all cross blocks are zero in F and Q (correlation between blocks
emerges through the UPDATE step's Kalman gain, not through synthetic
process-noise coupling). Update: same Joseph-form math as the
single-entity filter, generalized to the joint dimension, applied
sequentially for however many entities are observed at one timestep
(mathematically equivalent to a stacked batch update under independent
per-entity measurement noise). :func:`~src.estimator.motion_model.
apply_velocity_covariance_floor` is reused unchanged on the joint
covariance — it only ever touches indices 3:6 (the carrier's velocity
block) regardless of how many carried-entity dimensions follow, so
Day 25's adopted floor (config B) applies to a joint carrier exactly as
it does to a single-entity one.

STRUCTURAL, re-tested against the single-entity precedent
--------------------------------------------------------------
- **Prior firewall (§17).** :func:`run_joint_filter` takes no parameter
  that could carry a behavioral prior — checked directly via
  ``inspect.signature`` (``tests/test_estimator_joint.py``), same pattern
  as ``run_single_entity_filter``'s own test.
- **Consistency residuals required on every estimate.**
  :class:`JointStateEstimate` enforces the same non-empty-residuals-with-
  an-nis-entry rule as :class:`~src.estimator.state.StateEstimate`,
  independently re-checked at construction, not inherited by assumption.
- **graph_rev reproducibility.** :func:`resolve_joint_state` replays
  ``graph.factors_as_of(query.graph_rev)`` exactly like
  :func:`~src.estimator.filter.resolve_state` — re-solving a component at
  an earlier revision after more factors have been appended returns a
  bit-identical result, tested directly.

Not implemented today (skeletons, not silent gaps)
--------------------------------------------------------
- **Hypothesis management** (discrete data-association uncertainty) --
  :func:`resolve_data_association` raises ``NotImplementedError`` naming
  what it would resolve: which carrier a carried entity belongs to, or
  whether an ambiguous nearby entity is the same tracked identity. A
  :class:`Component` here is declared by the caller, never inferred.
- **Smoothing across the joint graph** -- :func:`resolve_joint_state`
  raises ``NotImplementedError`` for ``horizon_kind="smoothed"``, same
  convention as :data:`~src.model.episode.StateQuery.horizon_kind`.
- **The discrete/continuous hybrid** -- :class:`HybridDiscreteContinuousState`
  raises ``NotImplementedError`` at construction, naming what it would do:
  let a component's own MEMBERSHIP change mid-track (a "picked up"/"set
  down" event splitting or merging components) without restarting the
  filter. Today's :class:`Component` composition is fixed for the life of
  one :func:`run_joint_filter` call -- see that function's docstring on
  why every member must be observed at the component's bootstrap step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from src.estimator import consistency
from src.estimator.filter import LARGE_VELOCITY_VARIANCE_MPS2, run_single_entity_filter
from src.estimator.filter import FilterError
from src.estimator.filter import resolve_state as _resolve_single_entity_state
from src.estimator.measurement_model import MeasurementModel
from src.estimator.motion_model import (
    STATE_DIM,
    MotionModel,
    apply_velocity_covariance_floor,
)
from src.estimator.state import ConsistencyResidual, StateEstimate
from src.model.episode import Factor, StateGraph, StateQuery
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.ulid import generate_ulid

FloatArray = npt.NDArray[np.float64]

OFFSET_DIM = 3
"""Each carried entity contributes a 3D offset -- no independent velocity
state (see module docstring: a carried entity's own kinematics are not
tracked, only its rigid-plus-slip displacement from its carrier)."""

OFFSET_SLIP_SIGMA_MPS_SQRT_S = 0.05
"""Declared, not fitted: how much a held object shifts relative to the
carrier's body per unit time, in the same small-sway convention
:class:`~src.estimator.motion_model.NearlyConstantPositionMotionModel`
already uses for IMM's ``static`` mode (``sigma_position_mps_sqrt_s =
0.05``) -- a carried object held against or near the body is at least as
rigid as a standing person's own sway, so the same order-of-magnitude
value is the natural default rather than a new, separately-tuned one."""

_JOINT_Q_REGULARIZATION = 1e-6
"""Same convention and same value as
:data:`~src.estimator.motion_model._Q_REGULARIZATION` -- keeps the joint Q
strictly positive-definite for the same reason (a rank-deficient
discretized-white-noise block stacked with an identity-F offset block is
not guaranteed PD on its own)."""

JOINT_UPDATE_RULE_VERSION = "kalman-joseph-form-joint-v1"
JOINT_UPDATE_RULE_SHA = hashlib.sha256(
    JOINT_UPDATE_RULE_VERSION.encode("utf-8")
).hexdigest()


class JointFilterError(RuntimeError):
    """Raised when a Component, its observations, or a joint solve are
    malformed. The joint-estimation package's own vocabulary -- mirrors
    :class:`~src.estimator.filter.FilterError`'s role for the single-entity
    path."""


@dataclass(frozen=True)
class Component:
    """The unit of joint inference: one carrier plus the entities it
    carries, declared statically. See the module docstring for why
    membership is fixed for the life of a run, not inferred or grown."""

    carrier_entity_id: str
    carried_entity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.carrier_entity_id:
            raise JointFilterError("Component.carrier_entity_id must not be empty")
        if len(set(self.carried_entity_ids)) != len(self.carried_entity_ids):
            raise JointFilterError(
                f"Component.carried_entity_ids must not repeat an id, got "
                f"{self.carried_entity_ids}"
            )
        if self.carrier_entity_id in self.carried_entity_ids:
            raise JointFilterError(
                f"Component.carrier_entity_id {self.carrier_entity_id!r} cannot "
                "also appear in carried_entity_ids -- an entity cannot carry itself"
            )
        if not all(self.carried_entity_ids):
            raise JointFilterError(
                "Component.carried_entity_ids must not be empty strings"
            )

    @property
    def size(self) -> int:
        return 1 + len(self.carried_entity_ids)

    @property
    def state_dim(self) -> int:
        return STATE_DIM + OFFSET_DIM * len(self.carried_entity_ids)

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Carrier first, then carried entities in declared order -- the
        canonical processing order used to break same-timestep ties."""
        return (self.carrier_entity_id, *self.carried_entity_ids)

    def offset_slice(self, carried_entity_id: str) -> slice:
        if carried_entity_id not in self.carried_entity_ids:
            raise JointFilterError(
                f"{carried_entity_id!r} is not a carried entity of this component "
                f"({self.carried_entity_ids})"
            )
        idx = self.carried_entity_ids.index(carried_entity_id)
        start = STATE_DIM + OFFSET_DIM * idx
        return slice(start, start + OFFSET_DIM)


@dataclass(frozen=True)
class JointObservation:
    """One entity's observation, tagged with which member of the component
    it belongs to -- the joint analogue of a single ``Observation`` in
    :func:`~src.estimator.filter.run_single_entity_filter`'s input
    sequence."""

    entity_id: str
    observation: Observation


@dataclass(frozen=True)
class JointStateEstimate:
    """One resolved joint state -- a marginal mean/cov over an entire
    :class:`Component`, at ``ts_ns``. Same shape rules and STRUCTURAL
    requirements as :class:`~src.estimator.state.StateEstimate`, sized to
    ``component.state_dim`` instead of the fixed single-entity
    :data:`~src.estimator.motion_model.STATE_DIM`."""

    ts_ns: int
    component: Component
    mean: tuple[float, ...]
    cov: tuple[tuple[float, ...], ...]
    observed: bool
    motion_model_sha: str
    measurement_model_sha: str
    update_rule_sha: str
    graph_rev: int
    residuals: tuple[ConsistencyResidual, ...]

    def __post_init__(self) -> None:
        dim = self.component.state_dim
        if len(self.mean) != dim:
            raise JointFilterError(
                f"JointStateEstimate.mean must have {dim} entries "
                f"(component.state_dim), got {len(self.mean)}"
            )
        if len(self.cov) != dim or any(len(row) != dim for row in self.cov):
            raise JointFilterError(
                f"JointStateEstimate.cov must be {dim}x{dim}, got shape "
                f"({len(self.cov)}, {[len(r) for r in self.cov]})"
            )
        for name, value in (
            ("motion_model_sha", self.motion_model_sha),
            ("measurement_model_sha", self.measurement_model_sha),
            ("update_rule_sha", self.update_rule_sha),
        ):
            if not value:
                raise JointFilterError(f"JointStateEstimate.{name} must not be empty")
        if self.graph_rev < 0:
            raise JointFilterError(
                f"JointStateEstimate.graph_rev must be >= 0, got {self.graph_rev}"
            )
        if not self.residuals:
            raise JointFilterError(
                "JointStateEstimate.residuals must not be empty -- stage 4 "
                "(consistency) always runs, same rule as StateEstimate."
            )
        if not any(r.kind == "nis" for r in self.residuals):
            raise JointFilterError(
                "JointStateEstimate.residuals must include an 'nis'-kind entry"
            )

    def mean_array(self) -> FloatArray:
        return np.array(self.mean, dtype=np.float64)

    def cov_array(self) -> FloatArray:
        return np.array(self.cov, dtype=np.float64)

    def carrier_position_m(self) -> FloatArray:
        return self.mean_array()[:3]

    def carrier_velocity_mps(self) -> FloatArray:
        return self.mean_array()[3:6]

    def carried_offset_m(self, carried_entity_id: str) -> FloatArray:
        sl = self.component.offset_slice(carried_entity_id)
        return self.mean_array()[sl]

    def carried_position_m(self, carried_entity_id: str) -> FloatArray:
        """``carrier_position + offset`` -- the rigid-coupling identity
        this whole module exists to maintain."""
        return self.carrier_position_m() + self.carried_offset_m(carried_entity_id)

    def carried_position_cov_m2(self, carried_entity_id: str) -> FloatArray:
        """Covariance of ``carried_position_m`` -- NOT
        ``Cov(carrier_pos) + Cov(offset)``, which silently drops the
        cross-covariance term ``2*Cov(carrier_pos, offset)`` that the
        Joseph-form update actually builds up between the two blocks (a
        carried-entity observation updates offset via a gain that also
        touches carrier_pos, and vice versa -- see ``_H_for``). Computed
        as ``H @ cov @ H.T`` with the SAME 3 x state_dim projection matrix
        :func:`_H_for` builds for scoring a carried-entity observation, so
        the mean and the covariance this method returns are projections
        of the joint posterior under the identical linear map -- they
        cannot silently drift apart the way two independently-derived
        formulas could."""
        H = _H_for(self.component, carried_entity_id)
        result: FloatArray = H @ self.cov_array() @ H.T
        return result


@dataclass(frozen=True)
class _JointAppendedState:
    """What ``StateGraph``'s payload slot holds for a joint run -- mirrors
    ``src.estimator.filter._AppendedState``."""

    estimate: JointStateEstimate
    carrier_motion_model: MotionModel
    measurement_model: MeasurementModel
    offset_slip_sigma_mps_sqrt_s: float


def _distance_m(position_m: FloatArray, sensor_origin_m: FloatArray) -> float:
    return float(np.linalg.norm(position_m - sensor_origin_m))


def _joint_transition(
    component: Component, dt_s: float, carrier_motion_model: MotionModel
) -> FloatArray:
    dim = component.state_dim
    F = np.eye(dim, dtype=np.float64)
    F[:STATE_DIM, :STATE_DIM] = carrier_motion_model.F(dt_s)
    return F


def _joint_process_noise(
    component: Component,
    dt_s: float,
    carrier_motion_model: MotionModel,
    offset_slip_sigma_mps_sqrt_s: float,
) -> FloatArray:
    dim = component.state_dim
    Q = np.zeros((dim, dim), dtype=np.float64)
    Q[:STATE_DIM, :STATE_DIM] = carrier_motion_model.Q(dt_s)
    slip_var = (offset_slip_sigma_mps_sqrt_s**2) * dt_s
    for i in range(len(component.carried_entity_ids)):
        start = STATE_DIM + OFFSET_DIM * i
        Q[start : start + OFFSET_DIM, start : start + OFFSET_DIM] = slip_var * np.eye(
            OFFSET_DIM, dtype=np.float64
        )
    Q += _JOINT_Q_REGULARIZATION * np.eye(dim, dtype=np.float64)
    return Q


def _H_for(component: Component, entity_id: str) -> FloatArray:
    """3 x state_dim measurement matrix for an observation of ``entity_id``
    (the carrier, or one of its carried entities)."""
    H = np.zeros((3, component.state_dim), dtype=np.float64)
    H[:, :3] = np.eye(3, dtype=np.float64)
    if entity_id != component.carrier_entity_id:
        sl = component.offset_slice(entity_id)
        H[:, sl] = np.eye(3, dtype=np.float64)
    return H


def _require_position(observation: Observation) -> WorldPositionMeasurement:
    if not isinstance(observation.measurement, WorldPositionMeasurement):
        raise JointFilterError(
            f"observation {observation.observation_id} carries a "
            f"{type(observation.measurement).__name__}, not "
            "WorldPositionMeasurement -- the joint filter only consumes "
            "unprojected 3D position readings, same restriction as "
            "run_single_entity_filter"
        )
    return observation.measurement


def _xyz(measurement: WorldPositionMeasurement) -> FloatArray:
    return np.array(
        [measurement.x_m, measurement.y_m, measurement.z_m], dtype=np.float64
    )


def run_joint_filter(
    graph: StateGraph,
    component: Component,
    observations: Sequence[JointObservation],
    carrier_motion_model: MotionModel,
    measurement_model: MeasurementModel,
    manifest_sha: str,
    offset_slip_sigma_mps_sqrt_s: float = OFFSET_SLIP_SIGMA_MPS_SQRT_S,
    sensor_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> StateGraph:
    """Run predict-then-update jointly over ``component``'s entities,
    appending one factor per distinct timestamp to ``graph``. Mutates and
    returns ``graph``.

    A component with no carried entities (``component.size == 1``)
    delegates its ENTIRE run to
    :func:`~src.estimator.filter.run_single_entity_filter` -- the actual
    function call, so it reproduces Day 20/25's single-entity behaviour
    (including config B's adopted velocity floor) bit-for-bit.

    Bootstrap requires every declared component member (carrier and every
    carried entity) to have an observation at the FIRST timestamp in
    ``observations`` -- dynamic mid-track membership (a carried entity
    first noticed after the component has already been tracking) is not
    handled today; see :class:`HybridDiscreteContinuousState`'s docstring
    for what would be needed and why it is out of scope.

    Args:
        graph: The append-only store to write into.
        component: Which entities are jointly estimated.
        observations: Every entity's observations, tagged by
            :class:`JointObservation`. Must be non-empty. Multiple
            entities observed at the identical ``ts_ns`` are processed as
            one predict step followed by sequential updates, in
            ``component.entity_ids`` order (carrier, then carried entities
            in declared order) for determinism.
        carrier_motion_model, measurement_model: The swappable components,
            same convention as ``run_single_entity_filter`` -- one shared
            measurement model across every entity in the component (a
            deliberate simplification: this project has one declared
            camera envelope per sensor, not per detected object class).
        manifest_sha: The run that produced these observations.
        offset_slip_sigma_mps_sqrt_s: See :data:`OFFSET_SLIP_SIGMA_MPS_SQRT_S`.
        sensor_origin_m: Same meaning as ``run_single_entity_filter``'s.

    Raises:
        JointFilterError: on empty input, an observation for an entity not
            in ``component``, a non-positive dt between timestamps, or a
            component member missing from the bootstrap timestamp.
    """
    if not component.carried_entity_ids:
        for jo in observations:
            if jo.entity_id != component.carrier_entity_id:
                raise JointFilterError(
                    f"observation entity_id {jo.entity_id!r} does not match "
                    f"this size-1 component's carrier "
                    f"{component.carrier_entity_id!r}"
                )
        try:
            return run_single_entity_filter(
                graph,
                [jo.observation for jo in observations],
                carrier_motion_model,
                measurement_model,
                manifest_sha,
                sensor_origin_m,
            )
        except FilterError as exc:
            raise JointFilterError(str(exc)) from exc

    if not observations:
        raise JointFilterError("run_joint_filter needs at least one observation")

    entity_order = component.entity_ids
    by_ts: dict[int, list[JointObservation]] = {}
    for jo in observations:
        if jo.entity_id not in entity_order:
            raise JointFilterError(
                f"observation entity_id {jo.entity_id!r} is not part of this "
                f"component {entity_order}"
            )
        by_ts.setdefault(jo.observation.ts_ns, []).append(jo)
    for ts in by_ts:
        by_ts[ts].sort(key=lambda jo: entity_order.index(jo.entity_id))

    timestamps = sorted(by_ts)
    first_ts = timestamps[0]
    first_group = by_ts[first_ts]
    first_entities = {jo.entity_id for jo in first_group}
    missing = set(entity_order) - first_entities
    if missing:
        raise JointFilterError(
            f"component bootstrap at ts_ns={first_ts} is missing observations "
            f"for {sorted(missing)} -- every declared member must be observed "
            "at the first timestep (dynamic mid-track component membership is "
            "not supported today; see HybridDiscreteContinuousState's docstring)"
        )

    origin = np.array(sensor_origin_m, dtype=np.float64)
    dim = component.state_dim
    mean = np.zeros(dim, dtype=np.float64)
    cov = np.zeros((dim, dim), dtype=np.float64)

    carrier_obs = next(
        jo.observation
        for jo in first_group
        if jo.entity_id == component.carrier_entity_id
    )
    carrier_z = _xyz(_require_position(carrier_obs))
    carrier_R = measurement_model.R(_distance_m(carrier_z, origin))
    mean[:3] = carrier_z
    cov[:3, :3] = carrier_R
    cov[3:6, 3:6] = LARGE_VELOCITY_VARIANCE_MPS2 * np.eye(3, dtype=np.float64)

    for i, carried_id in enumerate(component.carried_entity_ids):
        carried_obs = next(
            jo.observation for jo in first_group if jo.entity_id == carried_id
        )
        carried_z = _xyz(_require_position(carried_obs))
        carried_R = measurement_model.R(_distance_m(carried_z, origin))
        offset = carried_z - carrier_z
        start = STATE_DIM + OFFSET_DIM * i
        mean[start : start + 3] = offset
        # offset = carried_z - carrier_z, two independent measurement
        # errors -> Var(offset) = carrier_R + carried_R; Cov(carrier_pos,
        # offset) = Cov(carrier_z, carried_z - carrier_z) = -Var(carrier_z)
        # = -carrier_R (carried_z's error is independent of carrier_z's).
        cov[start : start + 3, start : start + 3] = carrier_R + carried_R
        cov[:3, start : start + 3] = -carrier_R
        cov[start : start + 3, :3] = -carrier_R

    inputs = tuple(str(jo.observation.observation_id) for jo in first_group)
    estimate = JointStateEstimate(
        ts_ns=first_ts,
        component=component,
        mean=tuple(mean.tolist()),
        cov=tuple(tuple(row) for row in cov.tolist()),
        observed=True,
        motion_model_sha=carrier_motion_model.sha,
        measurement_model_sha=measurement_model.sha,
        update_rule_sha=JOINT_UPDATE_RULE_SHA,
        graph_rev=graph.graph_rev + 1,
        residuals=(
            consistency.no_observation_nis_stub(),
            *consistency.stub_residuals(),
        ),
    )
    payload = _JointAppendedState(
        estimate=estimate,
        carrier_motion_model=carrier_motion_model,
        measurement_model=measurement_model,
        offset_slip_sigma_mps_sqrt_s=offset_slip_sigma_mps_sqrt_s,
    )
    factor = graph.append_factor(
        str(generate_ulid()), "bootstrap", inputs, manifest_sha, payload=payload
    )
    previous = payload
    previous_factor_id = factor.factor_id

    for ts in timestamps[1:]:
        group = by_ts[ts]
        dt_s = (ts - previous.estimate.ts_ns) / 1e9
        if dt_s <= 0:
            raise JointFilterError(
                f"non-positive dt ({dt_s}s) between joint observations at "
                f"{previous.estimate.ts_ns} and {ts}"
            )
        F = _joint_transition(component, dt_s, carrier_motion_model)
        Q = _joint_process_noise(
            component, dt_s, carrier_motion_model, offset_slip_sigma_mps_sqrt_s
        )
        current_mean = F @ previous.estimate.mean_array()
        current_cov = F @ previous.estimate.cov_array() @ F.T + Q

        first_nis: ConsistencyResidual | None = None
        eye_dim = np.eye(dim, dtype=np.float64)
        for jo in group:
            z = _xyz(_require_position(jo.observation))
            R = measurement_model.R(_distance_m(z, origin))
            H = _H_for(component, jo.entity_id)
            innovation = z - H @ current_mean
            innovation_cov = H @ current_cov @ H.T + R
            kalman_gain = current_cov @ H.T @ np.linalg.inv(innovation_cov)
            current_mean = current_mean + kalman_gain @ innovation
            i_kh = eye_dim - kalman_gain @ H
            current_cov = i_kh @ current_cov @ i_kh.T + kalman_gain @ R @ kalman_gain.T
            if first_nis is None:
                first_nis = consistency.compute_nis(innovation, innovation_cov)

        current_cov = apply_velocity_covariance_floor(
            current_cov, carrier_motion_model.velocity_covariance_floor_mps2(dt_s)
        )
        assert first_nis is not None  # group is never empty (built from by_ts)

        estimate = JointStateEstimate(
            ts_ns=ts,
            component=component,
            mean=tuple(current_mean.tolist()),
            cov=tuple(tuple(row) for row in current_cov.tolist()),
            observed=True,
            motion_model_sha=carrier_motion_model.sha,
            measurement_model_sha=measurement_model.sha,
            update_rule_sha=JOINT_UPDATE_RULE_SHA,
            graph_rev=graph.graph_rev + 1,
            residuals=(first_nis, *consistency.stub_residuals()),
        )
        payload = _JointAppendedState(
            estimate=estimate,
            carrier_motion_model=carrier_motion_model,
            measurement_model=measurement_model,
            offset_slip_sigma_mps_sqrt_s=offset_slip_sigma_mps_sqrt_s,
        )
        factor = graph.append_factor(
            str(generate_ulid()),
            "measurement_update",
            (previous_factor_id, *(str(jo.observation.observation_id) for jo in group)),
            manifest_sha,
            payload=payload,
        )
        previous = payload
        previous_factor_id = factor.factor_id

    return graph


def _joint_payload_factors(
    graph: StateGraph, graph_rev: int
) -> list[tuple[Factor, _JointAppendedState]]:
    out = []
    for f in graph.factors_as_of(graph_rev):
        payload = graph.payload_for(f.factor_id)
        if isinstance(payload, _JointAppendedState):
            out.append((f, payload))
    return out


def _is_joint_graph(graph: StateGraph, graph_rev: int) -> bool:
    for f in graph.factors_as_of(graph_rev):
        payload = graph.payload_for(f.factor_id)
        if payload is not None:
            return isinstance(payload, _JointAppendedState)
    return False


def resolve_joint_state(
    query: StateQuery, graph: StateGraph
) -> StateEstimate | JointStateEstimate:
    """The joint-estimation entry point, usable uniformly regardless of
    component size: a graph built by a size-1 ``run_joint_filter`` call
    (delegated to ``run_single_entity_filter`` internally) resolves via
    :func:`~src.estimator.filter.resolve_state` and returns a
    :class:`~src.estimator.state.StateEstimate`; a genuinely joint graph
    (size >= 2) resolves here directly and returns a
    :class:`JointStateEstimate`. Both expose ``mean_array()``/
    ``cov_array()``, so a caller computing NEES does not need to branch on
    which type came back.

    Raises:
        NotImplementedError: if ``query.horizon_kind == "smoothed"``.
        JointFilterError: if the graph has no resolvable state before
            ``query.at_ts_ns``, or resolving it needs more extrapolation
            than ``query.horizon_ns`` allows.
    """
    if query.horizon_kind == "smoothed":
        raise NotImplementedError(
            "smoothed-horizon resolution for a joint/multi-entity graph is "
            "out of Day 26's scope -- see "
            "src.model.episode.StateQuery.horizon_kind's docstring for the "
            "single-entity precedent this follows"
        )

    if not _is_joint_graph(graph, query.graph_rev):
        try:
            return _resolve_single_entity_state(query, graph)
        except FilterError as exc:
            raise JointFilterError(str(exc)) from exc

    candidates = _joint_payload_factors(graph, query.graph_rev)
    if not candidates:
        raise JointFilterError(
            f"no resolvable joint state in this graph at "
            f"graph_rev={query.graph_rev}"
        )
    at_or_before = [(f, p) for f, p in candidates if p.estimate.ts_ns <= query.at_ts_ns]
    if not at_or_before:
        earliest = candidates[0][1].estimate.ts_ns
        raise JointFilterError(
            f"query.at_ts_ns={query.at_ts_ns} predates this graph's earliest "
            f"resolvable joint state (ts_ns={earliest})"
        )

    _, latest = at_or_before[-1]
    dt_ns = query.at_ts_ns - latest.estimate.ts_ns
    if dt_ns == 0:
        return latest.estimate
    if dt_ns > query.horizon_ns:
        raise JointFilterError(
            f"resolving at_ts_ns={query.at_ts_ns} needs {dt_ns}ns of "
            f"extrapolation beyond the last resolvable factor "
            f"(ts_ns={latest.estimate.ts_ns}), exceeding "
            f"horizon_ns={query.horizon_ns}"
        )

    dt_s = dt_ns / 1e9
    component = latest.estimate.component
    F = _joint_transition(component, dt_s, latest.carrier_motion_model)
    Q = _joint_process_noise(
        component,
        dt_s,
        latest.carrier_motion_model,
        latest.offset_slip_sigma_mps_sqrt_s,
    )
    predicted_mean = F @ latest.estimate.mean_array()
    predicted_cov = F @ latest.estimate.cov_array() @ F.T + Q
    predicted_cov = apply_velocity_covariance_floor(
        predicted_cov, latest.carrier_motion_model.velocity_covariance_floor_mps2(dt_s)
    )
    return JointStateEstimate(
        ts_ns=query.at_ts_ns,
        component=component,
        mean=tuple(predicted_mean.tolist()),
        cov=tuple(tuple(row) for row in predicted_cov.tolist()),
        observed=False,
        motion_model_sha=latest.estimate.motion_model_sha,
        measurement_model_sha=latest.estimate.measurement_model_sha,
        update_rule_sha=latest.estimate.update_rule_sha,
        graph_rev=query.graph_rev,
        residuals=(
            consistency.no_observation_nis_stub(),
            *consistency.stub_residuals(),
        ),
    )


def resolve_data_association(*args: object, **kwargs: object) -> None:
    """Hypothesis management: discrete uncertainty over WHICH carrier a
    carried entity belongs to, or whether an ambiguous nearby entity is the
    same tracked identity already in a component. NOT implemented today --
    :class:`Component` membership is declared by the caller, never
    inferred or resolved from competing hypotheses. See
    ``FOUNDATION_REPORT.md``'s long-carried "hypothesis management"
    punch-list item, unchanged in scope by today's work.
    """
    raise NotImplementedError(
        "hypothesis management (discrete data-association uncertainty) is "
        "not implemented -- Component membership is declared, not inferred"
    )


class HybridDiscreteContinuousState:
    """The discrete/continuous hybrid: jointly reasoning about a discrete
    mode (e.g. "entity X is currently carried" vs "set down and now
    independent") together with the continuous kinematic state, so a
    component's own membership can change mid-track without restarting the
    filter. NOT implemented today -- :func:`run_joint_filter` fixes
    ``Component`` composition for the life of one call (every member must
    be observed at bootstrap; see that function's docstring). A "picked
    up"/"set down" event that should split or merge components needs
    exactly this hybrid machinery.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "discrete/continuous hybrid state (dynamic component membership "
            "driven by a discrete pick-up/set-down mode) is not implemented"
        )
