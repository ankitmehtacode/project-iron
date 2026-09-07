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
  bit-identical result, tested directly. Unaffected by today's
  :func:`resolve_data_association`, which does not touch
  :class:`~src.model.episode.StateGraph` at all (see below) -- the
  claim is that this still holds, re-verified, not that it now covers a
  new case.

Implemented today (Day 32, Objective 4) -- hypothesis management, started
---------------------------------------------------------------------------
:func:`resolve_data_association` proposes one :class:`~src.model.
hypothesis.Hypothesis` per :class:`AssociationCandidate` in the shared
:class:`~src.model.hypothesis.HypothesisStore`, then prunes in two
passes, in this fixed order: hard-constraint violations first (via
:func:`~src.estimator.constraints.prune_for_hard_violation`, the same
function :func:`check_hard_constraints` uses), THEN a per-component
:class:`AssociationBudgetConfig` cut by ``log_likelihood`` over whatever
survives the first pass. Hard-before-budget is not incidental ordering:
a hard-impossible hypothesis must never occupy a budget slot a
physically-possible competitor could have used. Budget cuts are recorded
as :class:`~src.model.hypothesis.PrunedByBudget` -- "lost a resource
competition," never "was evaluated and found worse" -- so
``store.considered_alternatives()`` keeps the two answers to "was this
considered?" distinguishable, per that store's own reason for existing.

**Scope, stated tightly, because this is a start:** ``log_likelihood``
is caller-supplied, not computed here -- no re-estimation, no motion or
measurement model touches this function. Nothing here feeds back into
:func:`run_joint_filter`'s ``Component`` composition, which stays
caller-declared exactly as before; a resolved association does not yet
grow, shrink, or merge a running component. ``MergedInto`` and
``ExpiredHorizon`` (:mod:`src.model.hypothesis`'s other two death
causes) have no caller in this module -- there is no merge or
horizon-expiry logic yet, only propose / hard-prune / budget-prune.

**Day 33 additions.** Objective 2 stress-tested the above fresh and
found no architectural violation (hard-constraint pruning genuinely
happens before any likelihood ranking; the prior firewall holds; a
determinism test replaces the graph_rev-reproducibility test that does
not yet apply, since this function still does not touch
:class:`~src.model.episode.StateGraph`). It also found a real, named
gap: :class:`AssociationResolution` exposed no ambiguity/margin signal
on its primary surface (``.surviving``) -- a near-tie and a landslide
looked identical there. **Fixed Day 34, Objective 2 (see below) -- the
gap did not survive a second day**, unlike this project's usual pattern
of a `known_bug`/`xfail` living on as a regression guard until a design
decision lands; here the design decision (:class:`AssociationVerdict`)
landed the next session. Objective 3 connects the Day-26 component-size
cap to a membership CHANGE, which construction-time enforcement alone
cannot see: :func:`resolve_component_membership` applies one accepted
carrier assignment to an existing :class:`Component` and returns either
the grown ``Component`` (at or under
:data:`DEFAULT_COMPONENT_CAP_CONFIG`) or a :class:`ComponentCapRefusal`
(over cap, ``degradation_action`` required, existing membership
unchanged) -- same structural-impossibility pattern as
:class:`DegradedComponentEstimate`. It is deliberately NOT wired to
:func:`resolve_data_association`'s output automatically: that would
require :class:`AssociationCandidate` to carry structured carrier
identity rather than an opaque ``proposition`` string, which is a real
design change or scope creep on the store's own "does not interpret"
boundary and is not decided here.

**Day 34 additions.** Objective 2: :func:`resolve_data_association` now
returns an :data:`AssociationVerdict` (:class:`Decisive` or
:class:`Ambiguous`) alongside the survivor set -- see
:data:`DECISIVE_LOG_BAYES_FACTOR` for the cited threshold and
:func:`compute_association_verdict` for the margin computation, which
runs on hard-constraint survivors BEFORE the budget touches the
ranking (whether the evidence itself is decisive is not a function of
how many hypotheses a budget can afford to carry). Objective 3: budget
cuts are no longer unconditionally :class:`~src.model.hypothesis.
PrunedByBudget` -- :func:`dominate_by_likelihood` is now the second
(and only other) function permitted to assign a death cause based on
likelihood, structurally requiring a supra-threshold margin to
construct a :class:`~src.model.hypothesis.DominatedByLikelihood`, so a
resource decision can no longer be relabeled as an evidentiary one, nor
the reverse. Objective 3 also checked the interaction with
:func:`resolve_component_membership` directly (not assumed): there is
none yet, because that function never touches a hypothesis, a
candidate, or a death cause -- confirmed by inspecting its signature.

Not implemented today (skeletons, not silent gaps)
--------------------------------------------------------
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
import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Callable, Literal, NoReturn, Sequence

import numpy as np
import numpy.typing as npt

from src.estimator import consistency
from src.estimator.constraints import prune_for_hard_violation
from src.estimator.filter import LARGE_VELOCITY_VARIANCE_MPS2, run_single_entity_filter
from src.estimator.filter import FilterError
from src.estimator.filter import resolve_state as _resolve_single_entity_state
from src.estimator.measurement_model import MeasurementModel
from src.estimator.motion_model import (
    PERSON_SIGMA_A_MPS2,
    STATE_DIM,
    MotionModel,
    apply_velocity_covariance_floor,
    motion_model_for,
)
from src.estimator.state import ConsistencyResidual, StateEstimate
from src.events.schema import EntityRef, Verb
from src.model.constraint import HardConstraintViolation
from src.model.episode import Factor, StateGraph, StateQuery, solve_state
from src.model.events import InferredEvent, ObservedEvent
from src.model.hypothesis import (
    DeadHypothesis,
    DeathCause,
    DominatedByLikelihood,
    Hypothesis,
    HypothesisStore,
    PrunedByBudget,
    RefutedByHardConstraint,
)
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
value is the natural default rather than a new, separately-tuned one.
This is the CONSTANT slip model's value -- see :data:`OffsetSlipModel`
for the Day 27 alternative that scales this baseline by relative
acceleration instead of applying it uniformly."""

OffsetSlipModel = Literal["constant", "acceleration_scaled", "two_term"]
"""Which physical model governs the offset's process noise.

``"constant"`` (Day 26) applies :data:`OFFSET_SLIP_SIGMA_MPS_SQRT_S`
uniformly at every step, regardless of what the carrier is doing.

``"acceleration_scaled"`` (Day 27) is the physically-derived
alternative -- a grip slips when motion CHANGES, not when it is steady,
so the effective slip sigma at a step is
:data:`OFFSET_SLIP_SIGMA_MPS_SQRT_S` scaled by the carrier's own
estimated acceleration relative to :data:`~src.estimator.motion_model.
PERSON_SIGMA_A_MPS2` (this module's already-declared pedestrian
acceleration bound, reused rather than a new fitted reference):

    effective_sigma = OFFSET_SLIP_SIGMA_MPS_SQRT_S * (|a_hat| / PERSON_SIGMA_A_MPS2)

Measured (Day 27): fixes the carrier's cessation-regime overconfidence
(unjustified information gain +19.3% -> +0.9%) but the effective sigma
collapses toward the Q-regularization floor whenever estimated
acceleration is near zero -- most of a walking track -- which also
collapses the Kalman gain the filter needs to keep averaging in new
asset observations, destroying the asset's own RMSE margin over
independent filtering (+0.0425m -> -0.0051m on v5-cessation).

``"two_term"`` (Day 28) is ``"constant"`` PLUS ``"acceleration_scaled"``,
combined as VARIANCES (the physically correct combination for two
independent noise sources -- their variances add; their sigmas do not):

    offset_variance = OFFSET_SLIP_SIGMA_MPS_SQRT_S^2
                     + (OFFSET_SLIP_SIGMA_MPS_SQRT_S * |a_hat|/PERSON_SIGMA_A_MPS2)^2

Physical reading: the constant term is baseline grip compliance under
STEADY carry (a held object is never perfectly rigid even when nothing
is accelerating -- the same physical fact Day 26's constant model was
already built on). The acceleration term is slip specifically INDUCED
by a change in motion (Day 27's own physical reading, unchanged). Both
terms use the SAME already-declared baseline
(:data:`OFFSET_SLIP_SIGMA_MPS_SQRT_S`) and the SAME already-declared
acceleration reference (:data:`~src.estimator.motion_model.PERSON_SIGMA_A_MPS2`)
-- no new, separately-tuned ratio between the two terms was introduced
to combine them. The ratio between what each term contributes at any
given moment is therefore not a free parameter: it falls out entirely
from the instantaneous acceleration ratio |a_hat|/PERSON_SIGMA_A_MPS2
already established Day 27, which is exactly 1 (the two terms
contribute equally) at the nominal acceleration bound, below 1 during
steady motion (floor term dominates -- Day 26's regime, where the
asset's benefit was measured), and above 1 during a sharp transient
(acceleration term dominates -- Day 27's regime, where the carrier's
overconfidence was measured). See ADR 0011's "Day 28" section for
whether this recovers both Day 26's asset margin and Day 27's carrier
calibration fix simultaneously, measured, not assumed from the
derivation being clean.

Carrier acceleration is estimated causally from the two most recent
carrier velocity states already in the factor chain (no new state
dimension) -- see :func:`run_joint_filter`'s own loop for where it is
computed, one step behind the predict it informs (the acceleration
during step k-1->k sets the slip noise for step k->k+1), since
acceleration during the CURRENT step cannot be known before its own
update completes."""


def _effective_offset_slip_sigma(
    base_slip_sigma_mps_sqrt_s: float,
    offset_slip_model: Literal["constant", "acceleration_scaled"],
    carrier_acceleration_mps2: float | None,
) -> float:
    """Day 26/27's original single-term dispatch, unchanged -- reused by
    :func:`_offset_slip_variance_mps2` for both of its non-two-term cases
    and directly by "two_term"'s acceleration component, so neither
    existing model's numeric behavior can silently drift when a third
    model is added alongside them."""
    if offset_slip_model == "constant" or carrier_acceleration_mps2 is None:
        return base_slip_sigma_mps_sqrt_s
    ratio = carrier_acceleration_mps2 / PERSON_SIGMA_A_MPS2
    return base_slip_sigma_mps_sqrt_s * ratio


def _offset_slip_variance_mps2(
    base_slip_sigma_mps_sqrt_s: float,
    offset_slip_model: OffsetSlipModel,
    carrier_acceleration_mps2: float | None,
) -> float:
    """The offset's process-noise VARIANCE for one predict step (before
    multiplying by ``dt_s`` -- see :func:`_joint_process_noise`). Variance
    is the unit two independent noise sources combine at ("two_term");
    sigma is not."""
    if offset_slip_model != "two_term":
        sigma = _effective_offset_slip_sigma(
            base_slip_sigma_mps_sqrt_s, offset_slip_model, carrier_acceleration_mps2
        )
        return sigma**2
    floor_variance = base_slip_sigma_mps_sqrt_s**2
    if carrier_acceleration_mps2 is None:
        # No acceleration estimate yet (first post-bootstrap step) --
        # nothing to add a transient term FOR, so this is the floor alone,
        # not a second copy of it (see acceleration_scaled's own
        # None-handling for why that mode instead falls back to the
        # baseline itself: the two models answer a different question
        # when the estimate is unavailable).
        return floor_variance
    acceleration_sigma = _effective_offset_slip_sigma(
        base_slip_sigma_mps_sqrt_s, "acceleration_scaled", carrier_acceleration_mps2
    )
    return floor_variance + acceleration_sigma**2


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


DegradationAction = Literal["independent_fallback"]
"""Day 27, Objective 2: what the solver does when a component exceeds its
configured cap. Only one action is implemented today (``"split_weakest_
coupling"`` was the named alternative and was NOT built -- see
:data:`DEFAULT_COMPONENT_CAP_CONFIG`'s docstring for why). A ``Literal``
with a single member is intentional, not a placeholder: it keeps the
field's type honest about what can actually happen today, and adding a
second action later is a type-checker-visible change everywhere this is
matched, not a silent string comparison someone forgot to update."""


@dataclass(frozen=True)
class ComponentCapConfig:
    """Config-driven, versioned bound on joint-component size -- see
    :data:`DEFAULT_COMPONENT_CAP_CONFIG` for the declared default and its
    justification. Passed explicitly to :func:`run_joint_filter` (never a
    module-global mutated in place), same convention as
    :class:`~src.estimator.imm.ImmConfig`."""

    max_component_size: int
    degradation_action: DegradationAction

    def __post_init__(self) -> None:
        if self.max_component_size < 1:
            raise JointFilterError(
                f"ComponentCapConfig.max_component_size must be >= 1, got "
                f"{self.max_component_size}"
            )

    @property
    def sha(self) -> str:
        payload = {
            "max_component_size": self.max_component_size,
            "degradation_action": self.degradation_action,
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


DEFAULT_COMPONENT_CAP_CONFIG = ComponentCapConfig(
    max_component_size=6, degradation_action="independent_fallback"
)
"""6, not a round number picked for convenience: Day 26 Objective 2's own
coupling-density measurement (``scripts/measure_component_sparsity.py``)
found this project's data supports component sizes up to 6 -- the
largest authored scene's agent count -- and its own decision gate
("proceed to Objective 3 only if p95 component size <= 6") used exactly
this bound to authorize building any solver at all. The solver was
designed against, and evaluated against
(``scripts/eval_joint_estimator.py``), data topping out at 6 entities.
Capping the solver's own operation at that SAME measured bound means it
never runs in a component-size regime nothing has validated it against —
exceeding 6 degrades rather than extrapolating joint-estimation behaviour
into untested territory. See ``docs/adr/0011-multi-entity-factor-graph.md``'s
Day 26 "Decision gate" section for the full derivation.

``degradation_action="independent_fallback"``, not ``"split_weakest_
coupling"``: independent fallback regresses to Day 20/25's single-entity
filter, already well-tested and well-understood, with a known accuracy/
calibration profile (``scripts/eval_estimator.py``). "Split weakest
coupling" would need a measured notion of relationship information
strength to rank factors by -- this project has never measured that (Day
26 Objective 2 measured coupling DENSITY, not coupling INFORMATIVENESS),
so building a ranking heuristic today would be exactly the kind of
unmeasured analytical claim Day 25/26's own rule warns against."""


@dataclass(frozen=True)
class DegradedComponentEstimate:
    """What :func:`run_joint_filter` emits INSTEAD of a
    :class:`JointStateEstimate` when ``component.size`` exceeds
    ``cap_config.max_component_size``. STRUCTURAL: this is a distinct
    TYPE, not a flag bolted onto ``JointStateEstimate`` -- a caller
    holding one of these cannot construct it without
    ``degradation_action`` and ``cap_config_sha`` (both required fields,
    no default), and cannot mistake it for a genuine joint solve the way
    an optional flag on the same type could be overlooked. Every entity's
    own independently-filtered :class:`~src.estimator.state.StateEstimate`
    is preserved (not discarded) -- a degraded component still resolves
    to a real per-entity answer, just not a jointly-coupled one.
    """

    ts_ns: int
    component: Component
    degradation_action: DegradationAction
    cap_config_sha: str
    entity_estimates: tuple[tuple[str, StateEstimate], ...]

    def __post_init__(self) -> None:
        if not self.degradation_action:
            raise JointFilterError(
                "DegradedComponentEstimate.degradation_action must not be empty"
            )
        if not self.cap_config_sha:
            raise JointFilterError(
                "DegradedComponentEstimate.cap_config_sha must not be empty"
            )
        recorded_ids = {entity_id for entity_id, _ in self.entity_estimates}
        expected_ids = set(self.component.entity_ids)
        if recorded_ids != expected_ids:
            raise JointFilterError(
                f"DegradedComponentEstimate.entity_estimates covers "
                f"{sorted(recorded_ids)}, expected exactly "
                f"{sorted(expected_ids)} (component.entity_ids)"
            )

    def estimate_for(self, entity_id: str) -> StateEstimate:
        for recorded_id, estimate in self.entity_estimates:
            if recorded_id == entity_id:
                return estimate
        raise JointFilterError(
            f"{entity_id!r} is not a member of this degraded component "
            f"({self.component.entity_ids})"
        )


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
    offset_slip_model: OffsetSlipModel = "constant"


@dataclass(frozen=True)
class _DegradedAppendedState:
    """What ``StateGraph``'s payload slot holds for a capped (degraded)
    component step -- the degraded-path analogue of ``_JointAppendedState``."""

    estimate: DegradedComponentEstimate


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
    offset_slip_model: OffsetSlipModel = "constant",
    carrier_acceleration_mps2: float | None = None,
) -> FloatArray:
    dim = component.state_dim
    Q = np.zeros((dim, dim), dtype=np.float64)
    Q[:STATE_DIM, :STATE_DIM] = carrier_motion_model.Q(dt_s)
    offset_variance = _offset_slip_variance_mps2(
        offset_slip_sigma_mps_sqrt_s, offset_slip_model, carrier_acceleration_mps2
    )
    slip_var = offset_variance * dt_s
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


def _run_degraded_fallback(
    graph: StateGraph,
    component: Component,
    observations: Sequence[JointObservation],
    carrier_motion_model: MotionModel,
    measurement_model: MeasurementModel,
    manifest_sha: str,
    cap_config: ComponentCapConfig,
    sensor_origin_m: tuple[float, float, float],
) -> StateGraph:
    """``component.size > cap_config.max_component_size``: run every
    entity INDEPENDENTLY (the carrier under its own motion model, each
    carried entity under the un-coupled ``asset_carried`` model -- Day
    20's original, pre-coupling default) and bundle the results into one
    :class:`DegradedComponentEstimate` per shared timestamp, appended to
    ``graph`` as the returned type :func:`resolve_joint_state` will hand
    back for this graph.

    Requires every entity to be observed at every timestamp ANY entity in
    the component is observed at (full synchronization) -- every caller
    of :func:`run_joint_filter` today (``scripts/eval_joint_estimator.py``)
    already constructs fully synchronized per-entity observation streams;
    tolerating a genuinely asynchronous, per-entity-sparse stream in the
    degraded path is not needed by anything that exists today and is not
    built speculatively.
    """
    if not observations:
        raise JointFilterError("run_joint_filter needs at least one observation")

    entity_ids = component.entity_ids
    per_entity_obs: dict[str, list[Observation]] = {eid: [] for eid in entity_ids}
    for jo in observations:
        if jo.entity_id not in per_entity_obs:
            raise JointFilterError(
                f"observation entity_id {jo.entity_id!r} is not part of this "
                f"component {entity_ids}"
            )
        per_entity_obs[jo.entity_id].append(jo.observation)

    all_ts = sorted({o.ts_ns for obs_list in per_entity_obs.values() for o in obs_list})
    for eid in entity_ids:
        observed_ts = {o.ts_ns for o in per_entity_obs[eid]}
        missing = [ts for ts in all_ts if ts not in observed_ts]
        if missing:
            raise JointFilterError(
                f"degraded fallback for component {entity_ids} requires every "
                f"entity observed at every shared timestamp; entity {eid!r} is "
                f"missing {len(missing)} of {len(all_ts)} timestamps"
            )

    entity_motion_models: dict[str, MotionModel] = {
        component.carrier_entity_id: carrier_motion_model
    }
    for cid in component.carried_entity_ids:
        entity_motion_models[cid] = motion_model_for("asset_carried")

    entity_graphs: dict[str, StateGraph] = {}
    for eid in entity_ids:
        obs_sorted = sorted(per_entity_obs[eid], key=lambda o: o.ts_ns)
        sub_graph = StateGraph()
        run_single_entity_filter(
            sub_graph,
            obs_sorted,
            entity_motion_models[eid],
            measurement_model,
            manifest_sha,
            sensor_origin_m,
        )
        entity_graphs[eid] = sub_graph

    cap_sha = cap_config.sha
    previous_factor_id: str | None = None
    for ts in all_ts:
        entity_estimates = []
        for eid in entity_ids:
            sub_graph = entity_graphs[eid]
            query = StateQuery(at_ts_ns=ts, horizon_ns=0, graph_rev=sub_graph.graph_rev)
            entity_estimates.append((eid, solve_state(query, sub_graph)))

        degraded = DegradedComponentEstimate(
            ts_ns=ts,
            component=component,
            degradation_action=cap_config.degradation_action,
            cap_config_sha=cap_sha,
            entity_estimates=tuple(entity_estimates),
        )
        payload = _DegradedAppendedState(estimate=degraded)
        obs_ids_at_ts = tuple(
            str(jo.observation.observation_id)
            for jo in observations
            if jo.observation.ts_ns == ts
        )
        factor_inputs = (
            (previous_factor_id, *obs_ids_at_ts)
            if previous_factor_id
            else obs_ids_at_ts
        )
        factor_kind = (
            "degraded_bootstrap" if previous_factor_id is None else "degraded_update"
        )
        factor = graph.append_factor(
            str(generate_ulid()),
            factor_kind,
            factor_inputs,
            manifest_sha,
            payload=payload,
        )
        previous_factor_id = factor.factor_id

    return graph


def run_joint_filter(
    graph: StateGraph,
    component: Component,
    observations: Sequence[JointObservation],
    carrier_motion_model: MotionModel,
    measurement_model: MeasurementModel,
    manifest_sha: str,
    offset_slip_sigma_mps_sqrt_s: float = OFFSET_SLIP_SIGMA_MPS_SQRT_S,
    offset_slip_model: OffsetSlipModel = "constant",
    sensor_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cap_config: ComponentCapConfig = DEFAULT_COMPONENT_CAP_CONFIG,
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
        offset_slip_model: See :data:`OffsetSlipModel`. Default
            ``"constant"`` preserves Day 26's exact behaviour;
            ``"acceleration_scaled"`` is the Day 27 physically-derived
            alternative; ``"two_term"`` is Day 28's, adding the two
            rather than choosing between them. Ignored for a size-1
            component (no offset exists to apply any of them to).
        sensor_origin_m: Same meaning as ``run_single_entity_filter``'s.
        cap_config: See :data:`DEFAULT_COMPONENT_CAP_CONFIG`. When
            ``component.size`` exceeds ``cap_config.max_component_size``,
            this function does NOT run the joint update at all -- it
            appends :class:`DegradedComponentEstimate` payloads instead
            (independent per-entity filtering, degradation recorded on
            every one), and the returned ``StateGraph`` must be resolved
            via :func:`resolve_joint_state`, which returns that type
            rather than :class:`JointStateEstimate` for such a graph.

    Raises:
        JointFilterError: on empty input, an observation for an entity not
            in ``component``, a non-positive dt between timestamps, or a
            component member missing from the bootstrap timestamp (or, in
            the degraded-fallback path, missing from any shared timestamp
            -- see :func:`_run_degraded_fallback`'s own docstring for that
            path's synchronization requirement).
    """
    if component.size > cap_config.max_component_size:
        return _run_degraded_fallback(
            graph,
            component,
            observations,
            carrier_motion_model,
            measurement_model,
            manifest_sha,
            cap_config,
            sensor_origin_m,
        )

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
        offset_slip_model=offset_slip_model,
    )
    factor = graph.append_factor(
        str(generate_ulid()), "bootstrap", inputs, manifest_sha, payload=payload
    )
    previous = payload
    previous_factor_id = factor.factor_id
    # Causal acceleration estimate for "acceleration_scaled": the velocity
    # BEFORE `previous`'s own velocity, so step k's slip noise is set from
    # the (k-2 -> k-1) acceleration, one step behind the dynamics it
    # informs -- see OffsetSlipModel's docstring for why this is causal by
    # construction, not a lag introduced as a shortcut.
    prior_carrier_velocity: FloatArray = mean[3:6].copy()

    for ts in timestamps[1:]:
        group = by_ts[ts]
        dt_s = (ts - previous.estimate.ts_ns) / 1e9
        if dt_s <= 0:
            raise JointFilterError(
                f"non-positive dt ({dt_s}s) between joint observations at "
                f"{previous.estimate.ts_ns} and {ts}"
            )
        carrier_acceleration_mps2: float | None = None
        if offset_slip_model in ("acceleration_scaled", "two_term"):
            carrier_acceleration_mps2 = float(
                np.linalg.norm(
                    previous.estimate.carrier_velocity_mps() - prior_carrier_velocity
                )
                / dt_s
            )
        F = _joint_transition(component, dt_s, carrier_motion_model)
        Q = _joint_process_noise(
            component,
            dt_s,
            carrier_motion_model,
            offset_slip_sigma_mps_sqrt_s,
            offset_slip_model,
            carrier_acceleration_mps2,
        )
        prior_carrier_velocity = previous.estimate.carrier_velocity_mps()
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
            offset_slip_model=offset_slip_model,
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


def _degraded_payload_factors(
    graph: StateGraph, graph_rev: int
) -> list[tuple[Factor, _DegradedAppendedState]]:
    out = []
    for f in graph.factors_as_of(graph_rev):
        payload = graph.payload_for(f.factor_id)
        if isinstance(payload, _DegradedAppendedState):
            out.append((f, payload))
    return out


PayloadKind = Literal["single", "joint", "degraded"]


def _payload_kind(graph: StateGraph, graph_rev: int) -> PayloadKind:
    for f in graph.factors_as_of(graph_rev):
        payload = graph.payload_for(f.factor_id)
        if isinstance(payload, _JointAppendedState):
            return "joint"
        if isinstance(payload, _DegradedAppendedState):
            return "degraded"
        if payload is not None:
            return "single"
    return "single"


def resolve_joint_state(
    query: StateQuery, graph: StateGraph
) -> StateEstimate | JointStateEstimate | DegradedComponentEstimate:
    """The joint-estimation entry point, usable uniformly regardless of
    component size or cap status: a graph built by a size-1
    ``run_joint_filter`` call (delegated to ``run_single_entity_filter``
    internally) resolves via
    :func:`~src.estimator.filter.resolve_state` and returns a
    :class:`~src.estimator.state.StateEstimate`; a genuinely joint graph
    (size >= 2, within the cap) resolves here directly and returns a
    :class:`JointStateEstimate`; a graph built by a component that
    exceeded :data:`DEFAULT_COMPONENT_CAP_CONFIG`'s (or a caller-supplied)
    cap returns a :class:`DegradedComponentEstimate`.

    ``StateEstimate`` and ``JointStateEstimate`` both expose
    ``mean_array()``/``cov_array()``, so NEES scoring does not need to
    branch between them. ``DegradedComponentEstimate`` deliberately does
    NOT -- there is no single joint mean/cov for a degraded component,
    only per-entity ones via ``estimate_for`` -- so a caller must branch
    (``isinstance``) on this return type specifically. That asymmetry is
    intentional: silently treating a degraded result as if it were
    jointly-coupled would hide exactly the provenance this cap exists to
    make visible.

    Degraded-path resolution supports EXACT timestamp matches only
    (``dt_ns == 0`` against a recorded factor) -- extrapolating a degraded
    component beyond its last recorded step would need the per-entity sub
    -graphs :func:`_run_degraded_fallback` builds internally, which are
    not persisted (the degraded path is a structural safety net, not a
    primary code path); this raises :class:`JointFilterError` rather than
    silently falling back to something un-degraded.

    Raises:
        NotImplementedError: if ``query.horizon_kind == "smoothed"``.
        JointFilterError: if the graph has no resolvable state before
            ``query.at_ts_ns``, if resolving it needs more extrapolation
            than ``query.horizon_ns`` allows, or (degraded path only) any
            extrapolation at all.
    """
    if query.horizon_kind == "smoothed":
        raise NotImplementedError(
            "smoothed-horizon resolution for a joint/multi-entity graph is "
            "out of Day 26's scope -- see "
            "src.model.episode.StateQuery.horizon_kind's docstring for the "
            "single-entity precedent this follows"
        )

    kind = _payload_kind(graph, query.graph_rev)

    if kind == "single":
        try:
            return _resolve_single_entity_state(query, graph)
        except FilterError as exc:
            raise JointFilterError(str(exc)) from exc

    if kind == "degraded":
        degraded_candidates = _degraded_payload_factors(graph, query.graph_rev)
        if not degraded_candidates:
            raise JointFilterError(
                f"no resolvable degraded component state in this graph at "
                f"graph_rev={query.graph_rev}"
            )
        degraded_at_or_before = [
            (f, p) for f, p in degraded_candidates if p.estimate.ts_ns <= query.at_ts_ns
        ]
        if not degraded_at_or_before:
            earliest = degraded_candidates[0][1].estimate.ts_ns
            raise JointFilterError(
                f"query.at_ts_ns={query.at_ts_ns} predates this graph's "
                f"earliest resolvable degraded state (ts_ns={earliest})"
            )
        _, latest_degraded = degraded_at_or_before[-1]
        if latest_degraded.estimate.ts_ns != query.at_ts_ns:
            raise JointFilterError(
                f"query.at_ts_ns={query.at_ts_ns} does not exactly match a "
                "recorded degraded-component factor "
                f"(nearest at or before: ts_ns={latest_degraded.estimate.ts_ns}) "
                "-- extrapolating a degraded component is not supported, see "
                "resolve_joint_state's own docstring"
            )
        return latest_degraded.estimate

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
    # Extrapolation beyond the last factor has no live velocity history to
    # estimate acceleration from -- falls back to the constant baseline
    # regardless of offset_slip_model (see _effective_offset_slip_sigma:
    # carrier_acceleration_mps2=None always returns the base sigma).
    Q = _joint_process_noise(
        component,
        dt_s,
        latest.carrier_motion_model,
        latest.offset_slip_sigma_mps_sqrt_s,
        latest.offset_slip_model,
        carrier_acceleration_mps2=None,
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


@dataclass(frozen=True)
class AssociationCandidate:
    """One proposed answer to a data-association question for a component
    -- e.g. "carried entity 7 belongs to carrier A" versus "...to carrier
    B". ``hypothesis_id`` need only be unique within the component;
    :func:`resolve_data_association` namespaces it under the component id
    before it ever reaches the shared :class:`~src.model.hypothesis.
    HypothesisStore`. ``log_likelihood`` is whatever scoring the caller's
    evaluation logic assigns -- this module does not compute one itself,
    the same "store, does not interpret" boundary
    :class:`~src.model.hypothesis.Hypothesis` already draws for
    ``support``."""

    hypothesis_id: str
    proposition: str
    log_likelihood: float


@dataclass(frozen=True)
class AssociationBudgetConfig:
    """Config-driven, versioned per-component cap on live association
    hypotheses -- same convention as :class:`ComponentCapConfig`
    (explicit, never a module-global mutated in place; a ``sha`` so a
    change to the bound is a recorded config change, not a silent
    behavior shift)."""

    per_component_budget: int

    def __post_init__(self) -> None:
        if self.per_component_budget < 1:
            raise JointFilterError(
                "AssociationBudgetConfig.per_component_budget must be >= 1, "
                f"got {self.per_component_budget}"
            )

    @property
    def sha(self) -> str:
        payload = {"per_component_budget": self.per_component_budget}
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


DEFAULT_ASSOCIATION_BUDGET_CONFIG = AssociationBudgetConfig(per_component_budget=3)
"""3, UNLIKE :data:`DEFAULT_COMPONENT_CAP_CONFIG`'s 6, is NOT measured --
no equivalent of Day 26's coupling-density study exists yet for how many
genuinely competing association hypotheses this project's data produces
per component. This is a placeholder pending exactly that measurement,
named as one rather than presented as a derived bound. Do not cite this
number as evidence of anything; it exists so the budget is a config
value from day one, not a hypothesis management is not implemented today
away from being one later."""


@dataclass(frozen=True)
class PruneDecision:
    """One line of the pruning log :func:`resolve_data_association`
    returns for every candidate it was given -- kept or not, and why.
    Logged for kept candidates too (``cause=None``), so a decision record
    is complete rather than only naming what died."""

    component_id: str
    hypothesis_id: str
    kept: bool
    cause: DeathCause | None
    log_likelihood: float


DECISIVE_LOG_BAYES_FACTOR = math.log(3)
"""~1.0986 nats. Kass & Raftery (1995, "Bayes Factors", JASA 90(430),
p.777, Table 4) and the Jeffreys (1961) scale it formalizes: a Bayes
factor (here, a log-likelihood ratio between two hypotheses under equal
priors -- the two coincide exactly in that case) below 3 (log 3 ~=
1.0986) is "not worth more than a bare mention" -- the weakest, most
permissive band on the standard scale, chosen deliberately as the FLOOR
for calling a result decisive rather than a stricter band (Kass &
Raftery's next band starts at a Bayes factor of 20, "positive"
evidence): calling something decisive at the loosest defensible boundary
is the conservative direction for a `PRUNED_BY_BUDGET`-vs-
`dominated_by_likelihood` decision (Objective 3) -- a HIGHER threshold
would only relabel MORE budget cuts as ambiguous, never fewer, so
starting at the loosest cited band and not tightening it is itself the
cautious choice. This is a cited constant, not a value fitted against
this project's own test data -- see the module docstring's warning about
exactly that failure mode (Day 22, Day 24)."""


@dataclass(frozen=True)
class Decisive:
    """The top-ranked hypothesis's log-likelihood margin over the runner-
    up meets or exceeds :data:`DECISIVE_LOG_BAYES_FACTOR` -- there IS a
    designated winner, evidenced, not just ranked highest by default."""

    winner: Hypothesis
    margin: float
    competitors: tuple[Hypothesis, ...]
    kind: Literal["decisive"] = field(default="decisive", init=False)


@dataclass(frozen=True)
class Ambiguous:
    """The margin between the top two is below
    :data:`DECISIVE_LOG_BAYES_FACTOR` -- there is NO designated winner.
    ``competitors`` holds every candidate under consideration, including
    whichever ranked first; nothing here is named ``winner``, so a caller
    cannot access one by mistake the way an ``Optional`` field invites
    (Day 34, Objective 2's own structural requirement)."""

    competitors: tuple[Hypothesis, ...]
    margin: float
    kind: Literal["ambiguous"] = field(default="ambiguous", init=False)


AssociationVerdict = Decisive | Ambiguous
"""No shared ``winner: Hypothesis | None`` field exists anywhere on
either variant -- ``if verdict.winner is not None`` is not writable
because ``Ambiguous`` has no ``winner`` attribute at all (an
``AttributeError``/``mypy`` error, not a runtime maybe). A caller MUST
``match``/``isinstance``-narrow both variants, the same closed-world-
dispatch discipline :class:`~src.model.events.PredictedEvent`'s alert-
ineligibility and :func:`~src.model.events._admit_as_evidence` already
use elsewhere in this codebase."""


def assert_verdict_never(verdict: NoReturn) -> NoReturn:
    """Exhaustiveness marker for a ``match`` over :data:`AssociationVerdict`
    -- same idiom as :func:`~src.model.constraint.assert_outcome_never`
    (Python 3.10 target predates ``typing.assert_never``). ``mypy``
    accepts the call only where ``verdict`` has narrowed to ``Never``, so
    a third verdict variant added later is a type error here, not a
    silently-unhandled branch."""
    raise AssertionError(f"unreachable: unhandled AssociationVerdict {verdict!r}")


def compute_association_verdict(
    ranked: Sequence[Hypothesis], threshold: float = DECISIVE_LOG_BAYES_FACTOR
) -> AssociationVerdict:
    """The margin is ALWAYS computed and ALWAYS reported -- there is no
    return path that omits it, mirroring the co-emission discipline
    :mod:`src.estimator.informativeness` established for calibration
    (Day 31): a verdict without its margin cannot be constructed, because
    neither :class:`Decisive` nor :class:`Ambiguous` has a default for
    ``margin``.

    ``ranked`` must already be sorted by ``support`` descending -- this
    function does not re-sort, so callers that already have a ranked
    list (:func:`resolve_data_association` does) do not pay for a second
    sort. A single candidate (no competitor to compare against at all)
    is treated as infinitely decisive (``margin = math.inf``) -- the
    strongest, not the weakest, case: nothing competes with it.
    """
    if not ranked:
        raise JointFilterError(
            "compute_association_verdict: ranked must not be empty -- "
            "there is no association question to resolve"
        )
    if len(ranked) == 1:
        return Decisive(winner=ranked[0], margin=math.inf, competitors=())

    top, runner_up = ranked[0], ranked[1]
    margin = top.support - runner_up.support
    if margin >= threshold:
        return Decisive(winner=top, margin=margin, competitors=tuple(ranked[1:]))
    return Ambiguous(competitors=tuple(ranked), margin=margin)


def dominate_by_likelihood(
    store: HypothesisStore,
    dominant_hypothesis_id: str,
    dominated_hypothesis_id: str,
    margin: float,
    threshold: float = DECISIVE_LOG_BAYES_FACTOR,
) -> DeadHypothesis:
    """The only function in this codebase permitted to record
    :class:`~src.model.hypothesis.DominatedByLikelihood` -- mirrors
    :func:`~src.estimator.constraints.prune_for_hard_violation`'s role
    for hard-constraint refutation. Requires ``margin`` as the evidence
    for the claim it is about to make: below ``threshold``, this raises
    rather than constructing the death record, because a margin that
    thin does not support "evaluated and found decisively worse" (Day
    34, Objective 3) -- a resource decision (:class:`~src.model.
    hypothesis.PrunedByBudget`) must never be relabeled as an
    evidentiary one by a caller reaching for the more confident-sounding
    cause. There is no way to construct a ``DominatedByLikelihood`` with
    an insufficient margin anywhere in this codebase; this function is it.
    """
    if margin < threshold:
        raise JointFilterError(
            f"dominate_by_likelihood({dominated_hypothesis_id!r}): margin "
            f"{margin} is below the decisiveness threshold {threshold} -- "
            "this is PRUNED_BY_BUDGET territory, not DOMINATED_BY_LIKELIHOOD; "
            "a resource decision must not be relabeled as an evidentiary one"
        )
    return store.kill(
        dominated_hypothesis_id,
        DominatedByLikelihood(dominant_hypothesis_id=dominant_hypothesis_id),
    )


@dataclass(frozen=True)
class AssociationResolution:
    """The result of :func:`resolve_data_association`: which hypotheses
    survived, the full decision log for every candidate considered
    (survivors included, ``cause=None``) -- not just the ones that died
    -- and the :class:`AssociationVerdict` over the hard-constraint
    survivors (computed independently of the budget, before it: whether
    there is a decisive winner is a property of the EVIDENCE, not of how
    many hypotheses a budget can afford to carry)."""

    component_id: str
    surviving: tuple[Hypothesis, ...]
    decisions: tuple[PruneDecision, ...]
    verdict: AssociationVerdict


def _namespaced_id(component_id: str, hypothesis_id: str) -> str:
    return f"{component_id}::{hypothesis_id}"


def resolve_data_association(
    store: HypothesisStore,
    component_id: str,
    candidates: Sequence[AssociationCandidate],
    hard_violation_check: Callable[
        [AssociationCandidate], HardConstraintViolation | None
    ],
    budget_config: AssociationBudgetConfig = DEFAULT_ASSOCIATION_BUDGET_CONFIG,
) -> AssociationResolution:
    """Hypothesis management: discrete uncertainty over WHICH carrier a
    carried entity belongs to, or whether an ambiguous nearby entity is
    the same tracked identity already in a component -- started today,
    scoped tightly (see the module docstring's "Implemented today"
    section for exactly what this does and does not do).

    Two pruning passes, in this order, because the order is the point:

    1. **Hard constraints, first.** Every candidate is proposed in
       ``store``, then checked via ``hard_violation_check`` (the
       caller's own physical-consistency evaluation -- this function
       does not invent one). A violation is pruned immediately through
       :func:`~src.estimator.constraints.prune_for_hard_violation`, the
       one function in this codebase permitted to record a
       ``RefutedByHardConstraint`` death. This happens BEFORE any
       likelihood ranking, per Day 29's own reason for building
       constraint typing first: a hard-impossible hypothesis must never
       occupy a budget slot a physically-possible competitor could have
       used, and it must never be carried into whatever continuous
       update would follow (out of scope here, but the ordering this
       function establishes is what makes that safe later).
    2. **The verdict, over hard-constraint survivors, before budget.**
       :func:`compute_association_verdict` runs on the ranked survivors
       -- :class:`Decisive` or :class:`Ambiguous` is a property of the
       EVIDENCE, computed before the budget ever touches the ranking
       (Day 34, Objective 2).
    3. **Budget, over hard-constraint survivors only.** Ranked by
       ``log_likelihood`` descending; every rank at or beyond
       ``budget_config.per_component_budget`` is killed. The death cause
       is NOT always :class:`~src.model.hypothesis.PrunedByBudget`: a cut
       candidate's margin from the weakest SURVIVING hypothesis is
       checked against :data:`DECISIVE_LOG_BAYES_FACTOR` via
       :func:`dominate_by_likelihood`. Below threshold (a near-tie the
       budget happened to break) it is ``PrunedByBudget`` -- a resource
       decision, not a rejection (see ``src/model/hypothesis.py``'s
       module docstring). At or above threshold (the weakest survivor
       decisively outranks it -- the evidence itself already excluded
       this candidate, the budget just didn't have to arbitrate) it is
       :class:`~src.model.hypothesis.DominatedByLikelihood` (Day 34,
       Objective 3: a resource decision must never be relabeled as an
       evidentiary one, and the converse -- an evidentiary exclusion
       hiding behind the more modest-sounding budget label -- is
       avoided too).

    Every candidate gets a :class:`PruneDecision` in the returned log,
    survivors included -- "what happened to every candidate considered"
    is answerable from this one return value without re-querying the
    store.
    """
    if not candidates:
        raise JointFilterError(
            f"resolve_data_association({component_id!r}): candidates must not "
            "be empty -- there is no association question to resolve"
        )

    decisions: list[PruneDecision] = []
    survivors: list[tuple[AssociationCandidate, str, Hypothesis]] = []

    for candidate in candidates:
        full_id = _namespaced_id(component_id, candidate.hypothesis_id)
        hypothesis = store.propose(
            full_id, candidate.proposition, candidate.log_likelihood
        )
        violation = hard_violation_check(candidate)
        if violation is not None:
            prune_for_hard_violation(store, full_id, violation)
            decisions.append(
                PruneDecision(
                    component_id=component_id,
                    hypothesis_id=candidate.hypothesis_id,
                    kept=False,
                    cause=RefutedByHardConstraint(
                        constraint_name=violation.constraint_name
                    ),
                    log_likelihood=candidate.log_likelihood,
                )
            )
            continue
        survivors.append((candidate, full_id, hypothesis))

    ranked = sorted(
        survivors, key=lambda triple: triple[0].log_likelihood, reverse=True
    )
    verdict = compute_association_verdict([hyp for _, _, hyp in ranked])

    budget = budget_config.per_component_budget
    kept = ranked[:budget]
    cut = ranked[budget:]
    kept_ids = {full_id for _, full_id, _ in kept}
    weakest_survivor = kept[-1] if kept else None

    for candidate, full_id, _hypothesis in kept:
        decisions.append(
            PruneDecision(
                component_id=component_id,
                hypothesis_id=candidate.hypothesis_id,
                kept=True,
                cause=None,
                log_likelihood=candidate.log_likelihood,
            )
        )

    for candidate, full_id, hypothesis in cut:
        assert (
            weakest_survivor is not None
        )  # budget >= 1 (enforced by AssociationBudgetConfig)
        _, weakest_id, weakest_hypothesis = weakest_survivor
        boundary_margin = weakest_hypothesis.support - hypothesis.support
        if boundary_margin >= DECISIVE_LOG_BAYES_FACTOR:
            dead = dominate_by_likelihood(store, weakest_id, full_id, boundary_margin)
        else:
            dead = store.kill(full_id, PrunedByBudget(budget=budget))
        decisions.append(
            PruneDecision(
                component_id=component_id,
                hypothesis_id=candidate.hypothesis_id,
                kept=False,
                cause=dead.cause,
                log_likelihood=candidate.log_likelihood,
            )
        )

    surviving = tuple(hyp for hyp in store.alive() if hyp.id in kept_ids)
    # Preserve the input candidates' relative order in the decision log,
    # not the rank order used for pruning -- a caller reading the log
    # against its own candidate list should not have to re-sort it.
    order = {c.hypothesis_id: i for i, c in enumerate(candidates)}
    decisions.sort(key=lambda d: order[d.hypothesis_id])

    return AssociationResolution(
        component_id=component_id,
        surviving=surviving,
        decisions=tuple(decisions),
        verdict=verdict,
    )


def _decisive_confidence_ceiling(margin: float) -> float:
    """Posterior probability of the top hypothesis, given exactly TWO
    competing hypotheses with equal priors: a direct consequence of
    Bayes' theorem, not a fitted constant. Under equal priors, posterior
    odds equal the likelihood ratio, so posterior probability = odds /
    (1 + odds) = sigmoid(log-odds) = sigmoid(margin), since ``margin`` IS
    the log-likelihood ratio (:func:`compute_association_verdict`).

    With more than two competitors this OVER-states the true posterior
    (a third candidate takes probability mass neither of the top two
    formulas accounts for) -- an upper bound, not the exact multi-way
    posterior, stated as an approximation rather than presented as exact.
    Still strictly useful as a CEILING: capping a caller's own confidence
    estimate at this value can only ever reduce it, never inflate it.
    """
    return 1.0 / (1.0 + math.exp(-margin))


def event_confidence_for_verdict(
    verdict: AssociationVerdict, caller_confidence: float
) -> tuple[float, Hypothesis]:
    """The subject hypothesis and the confidence an event about it may
    carry, given the verdict that resolved it.

    ``Decisive``: the caller's own confidence passes through unchanged --
    the association layer found a clear winner; it does not know enough
    about the caller's own detector/model posterior to second-guess it.

    ``Ambiguous``: a provisional pick (Day 34, Objective 4's chosen
    branch -- see :func:`build_identity_event`'s docstring for why the
    alternative, withholding identity assignment entirely, is deferred)
    is the strongest-supported competitor, and the caller's confidence is
    CAPPED at :func:`_decisive_confidence_ceiling` of the margin -- never
    raised, only ever reduced relative to what the caller supplied.
    """
    match verdict:
        case Decisive():
            return caller_confidence, verdict.winner
        case Ambiguous():
            provisional = max(verdict.competitors, key=lambda h: h.support)
            ceiling = _decisive_confidence_ceiling(verdict.margin)
            return min(caller_confidence, ceiling), provisional
        case _:
            assert_verdict_never(verdict)


def build_identity_event(
    verdict: AssociationVerdict,
    subject_for: Callable[[Hypothesis], EntityRef],
    *,
    event_id: uuid.UUID,
    site_id: str,
    ts_ns: int,
    verb: Verb,
    manifest_sha: str,
    importance: float,
    caller_confidence: float,
) -> ObservedEvent | InferredEvent:
    """Wires :data:`AssociationVerdict` into event emission (Day 34,
    Objective 4), bounded scope: full multi-modal state propagation --
    both competing hypotheses coexisting as a genuine mixture in the
    joint state until later evidence resolves them -- is explicitly
    DEFERRED to its own day. Today's scope is narrower: an event whose
    identity resolution came from an ``Ambiguous`` verdict must not carry
    confidence indistinguishable from one that came from a ``Decisive``
    verdict.

    **Chosen branch: (b), a provisional pick with reduced confidence and
    a recorded competitor reference -- not (a), withholding identity
    assignment entirely.** Stated why, not left implicit: ``subject`` is
    a REQUIRED, non-optional field on every :mod:`~src.model.events`
    class (``_EventCommon.subject: EntityRef``) -- there is no "unknown
    subject" representation in the current event schema at all. Building
    one would mean making ``subject`` optional across every event class
    and consumer, a schema change of the same flavor and scope as the
    deferred mixture-state question, not a small addition -- so (b) is
    chosen because it is what today's schema can actually represent, and
    the schema-change question for (a) is named here as deferred rather
    than silently avoided.

    STRUCTURAL: there is no branch of the ``match`` below that reaches
    ``ObservedEvent`` from an ``Ambiguous`` verdict -- the two verdict
    variants map to exactly the two return-type union members, checked
    exhaustively via :func:`assert_verdict_never`, not by a conditional a
    future edit could invert.
    """
    confidence, subject_hypothesis = event_confidence_for_verdict(
        verdict, caller_confidence
    )
    subject = subject_for(subject_hypothesis)

    match verdict:
        case Decisive():
            return ObservedEvent(
                event_id=event_id,
                site_id=site_id,
                ts_ns=ts_ns,
                subject=subject,
                verb=verb,
                confidence=confidence,
                importance=importance,
                manifest_sha=manifest_sha,
            )
        case Ambiguous(competitors=competitors, margin=margin):
            basis = (
                f"provisional identity pick among {len(competitors)} competing "
                f"association hypotheses, margin {margin:.4f} nats below the "
                f"{DECISIVE_LOG_BAYES_FACTOR:.4f}-nat decisiveness threshold; "
                f"competitor ids: {[c.id for c in competitors]}"
            )
            return InferredEvent(
                event_id=event_id,
                site_id=site_id,
                ts_ns=ts_ns,
                subject=subject,
                verb=verb,
                confidence=confidence,
                importance=importance,
                manifest_sha=manifest_sha,
                basis=basis,
            )
        case _:
            assert_verdict_never(verdict)


@dataclass(frozen=True)
class ComponentCapRefusal:
    """What :func:`resolve_component_membership` returns INSTEAD of a
    grown :class:`Component` when accepting an association's winning
    hypothesis would push component size past
    ``cap_config.max_component_size`` -- the same cap
    :data:`DEFAULT_COMPONENT_CAP_CONFIG` enforces at
    :func:`run_joint_filter` construction (Day 26), re-enforced here on
    the membership-CHANGE path (Day 33, Objective 3), which construction-
    time enforcement alone cannot see: a component can start within cap
    and still be pushed over it by a later association decision.

    STRUCTURAL, same pattern as :class:`DegradedComponentEstimate`:
    ``degradation_action`` and ``cap_config_sha`` are required fields,
    no default -- there is no way to represent "the cap was exceeded"
    without naming what was done about it.
    ``existing_component`` is what the caller already had, UNCHANGED:
    growth is refused, not silently truncated some other way (e.g.
    dropping an arbitrary existing member to make room)."""

    existing_component: Component
    refused_carried_entity_id: str
    degradation_action: DegradationAction
    cap_config_sha: str


def resolve_component_membership(
    existing: Component,
    carried_entity_id: str,
    cap_config: ComponentCapConfig = DEFAULT_COMPONENT_CAP_CONFIG,
) -> Component | ComponentCapRefusal:
    """Apply one association decision -- "``carried_entity_id`` belongs to
    ``existing.carrier_entity_id``" -- to a component's membership,
    subject to the cap. Not a replacement for
    :func:`resolve_data_association`: this function does not decide WHICH
    carrier wins (that is what produced ``carried_entity_id`` as an
    input, e.g. the ``hypothesis_id`` of an
    :class:`AssociationResolution`'s single surviving carrier-assignment
    hypothesis) -- it only decides whether accepting that win is safe to
    apply to ``existing``'s membership.

    A no-op (returns ``existing`` unchanged) when ``carried_entity_id``
    is already a member -- re-applying an already-accepted association
    must not be distinguishable from applying it the first time.

    STRUCTURAL: there is no return path that both exceeds the cap and
    lacks a recorded ``degradation_action`` -- the return type is
    ``Component`` (accepted, at or under cap) or
    :class:`ComponentCapRefusal` (refused, degradation_action required),
    exhaustively, never a silently-truncated ``Component`` or a bare
    exception with no typed trace of what happened.
    """
    if carried_entity_id in existing.carried_entity_ids:
        return existing
    candidate = Component(
        carrier_entity_id=existing.carrier_entity_id,
        carried_entity_ids=existing.carried_entity_ids + (carried_entity_id,),
    )
    if candidate.size <= cap_config.max_component_size:
        return candidate
    return ComponentCapRefusal(
        existing_component=existing,
        refused_carried_entity_id=carried_entity_id,
        degradation_action=cap_config.degradation_action,
        cap_config_sha=cap_config.sha,
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
