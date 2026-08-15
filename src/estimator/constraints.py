"""Concrete track-hypothesis constraints, and their estimator wiring.

Day 28, Objective 2: wires :mod:`src.model.constraint`'s typed
hard/twin-dependent distinction into the estimator as hypothesis-pruning
factors. This is the mechanism that keeps multi-hypothesis tracking
computationally survivable once it exists (Day-N, still ahead): a hard
violation prunes a hypothesis immediately, before it ever competes on
likelihood against its siblings, rather than every hard-impossible
hypothesis being carried forward at full cost until evidence eventually
outweighs it.

The pruning path, and why it cannot silently take a twin-dependent violation
------------------------------------------------------------------------------
:func:`prune_for_hard_violation` is the only function in this codebase
that calls :meth:`~src.model.hypothesis.HypothesisStore.kill`. Its
``violation`` parameter is typed ``HardConstraintViolation`` — not the
``HardConstraintViolation | TwinRevisionHypothesis`` union
:func:`~src.model.constraint.evaluate_constraint` can return in general.
Passing a ``TwinRevisionHypothesis`` there is a ``mypy`` error at the
call site, not a runtime possibility a branch has to guard against.

:func:`raise_twin_revision` is the twin-dependent counterpart, and its
signature has no :class:`~src.model.hypothesis.HypothesisStore`
parameter at all — there is no way to reach ``store.kill`` from inside
it, structurally, regardless of what the violation says. That is the
"closed-world dispatch, not an if" the objective asked for: the two
violation kinds are routed to two functions with disjoint capabilities
(one can reach a store, one cannot) before either function body ever
runs, rather than one shared function inspecting ``violation.kind`` and
branching on it.

Day 29, Objective 3 — the seven predicates, and the third outcome
--------------------------------------------------------------------
Day 28 typed seven constraints and left every predicate raising
``NotImplementedError``. Day 29 fills all seven, and filling them changed
two things about the shape above.

**A constraint that cannot be evaluated may not contribute a PASS.** All
four twin-dependent predicates need site geometry this project does not
have at any ``twin_rev`` (:class:`TwinGeometry` names exactly what is
missing). They return :class:`~src.model.constraint.Unevaluable`, a
third outcome the type system keeps separate from
:class:`~src.model.constraint.Satisfied` — see ``src/model/constraint.py``'s
own Day-29 section for why ``None`` had to be removed rather than
extended. :class:`HardConstraintCheck` and :class:`TwinDependentCheck`
carry the unevaluable set explicitly, so "nothing objected" can no longer
be read as "everything passed." A vacuous pass is worse than an absent
constraint: it manufactures assurance that something was checked.

**Two of Day 28's three "hard" constraints were described in
twin-dependent terms.** ``gravity_floor_transition`` was described as
requiring a "modeled transition (stairs, lift, ramp)" — that is
:func:`stair_or_lift_for_floor_change`, a twin-dependent constraint that
already existed; one physical claim was filed under both kinds.
``mass_conservation`` was described in terms of "a modeled portal or
occlusion boundary" — also twin geometry. Both keep their names and both
are genuinely hard once reduced to their twin-free core (free-fall
bound; reachability bound), and each predicate's docstring records what
was moved out of its scope and where it went. A description that names
the twin is the tell that a constraint is mis-typed, and it is worth
grepping for before adding the next one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from src.estimator.motion_model import PEDESTRIAN_MAX_SPEED_MPS
from src.model.constraint import (
    HardConstraint,
    HardConstraintViolation,
    Satisfied,
    TwinDependentConstraint,
    TwinRevisionHypothesis,
    Unevaluable,
    assert_outcome_never,
    evaluate_constraint,
)
from src.model.hypothesis import HypothesisStore, RefutedByHardConstraint


def prune_for_hard_violation(
    store: HypothesisStore, hypothesis_id: str, violation: HardConstraintViolation
) -> None:
    """Kill ``hypothesis_id`` in ``store`` for a hard-constraint violation.

    The only function that may prune a hypothesis for a constraint
    violation. See the module docstring for why its parameter type, not
    a runtime check, is what keeps a twin-dependent violation out of
    this path.
    """
    store.kill(
        hypothesis_id,
        RefutedByHardConstraint(constraint_name=violation.constraint_name),
    )


def raise_twin_revision(violation: TwinRevisionHypothesis) -> TwinRevisionHypothesis:
    """Surface a twin-dependent violation without touching any hypothesis
    store — this function does not take one as a parameter, so it is
    structurally incapable of pruning anything. Returns ``violation``
    unchanged: Day 28 establishes this as a real, typed value with no
    consumer yet (see ADR 0011 and :class:`TwinRevisionHypothesis`'s own
    docstring for the twin-drift-detector capability this would enable
    if repeated violations at the same location were tracked over time —
    not built today).
    """
    return violation


@dataclass(frozen=True)
class HardConstraintCheck:
    """The result of checking a hypothesis against hard constraints.

    Three fields, not a ``violation | None``, because Day 29's
    structural rule needs "everything passed" and "nothing objected"
    to be different answers. A check that evaluated one constraint,
    could not evaluate three, and found no violation has NOT cleared the
    hypothesis — and under the old ``None`` return it was indistinguishable
    from one that cleared all four. :attr:`all_satisfied` is the only
    honest "cleared" reading, and it is false whenever anything was
    :class:`~src.model.constraint.Unevaluable`.
    """

    satisfied: tuple[Satisfied, ...]
    unevaluable: tuple[Unevaluable, ...]
    violation: HardConstraintViolation | None

    @property
    def pruned(self) -> bool:
        """The hypothesis was killed. Only a hard violation does this."""
        return self.violation is not None

    @property
    def all_satisfied(self) -> bool:
        """Every constraint ran AND held. An unevaluable constraint makes
        this false even with no violation — that is the whole point of
        the type."""
        return self.violation is None and not self.unevaluable


@dataclass(frozen=True)
class TwinDependentCheck:
    """The twin-dependent counterpart. Carries no
    :class:`~src.model.hypothesis.HypothesisStore` and no
    :class:`~src.model.constraint.HardConstraintViolation` — nothing
    reachable from this value can prune anything."""

    satisfied: tuple[Satisfied, ...]
    unevaluable: tuple[Unevaluable, ...]
    hypotheses: tuple[TwinRevisionHypothesis, ...]

    @property
    def all_satisfied(self) -> bool:
        return not self.hypotheses and not self.unevaluable


def check_hard_constraints(
    store: HypothesisStore,
    hypothesis_id: str,
    constraints: Sequence[HardConstraint],
    *args: object,
    **kwargs: object,
) -> HardConstraintCheck:
    """Evaluate ``constraints`` in order against ``*args, **kwargs``; on
    the first violation, prune ``hypothesis_id`` in ``store`` and stop.

    An :class:`~src.model.constraint.Unevaluable` outcome is recorded and
    evaluation CONTINUES — it is not a violation, so it must not prune,
    and it is not a pass, so it must not be dropped. Only a
    :class:`~src.model.constraint.HardConstraintViolation` reaches
    :func:`prune_for_hard_violation`; the ``match`` below is exhaustive
    over :data:`~src.model.constraint.HardConstraintOutcome`, so adding a
    fourth outcome type later is a ``mypy`` error here rather than a
    silently-skipped case.

    The hypothesis-pruning factor the objective asks for: a real
    multi-hypothesis tracker would call this once per candidate
    hypothesis per step, once such a tracker exists (Day 28 keeps
    multi-hypothesis management itself skeleton — see
    ``src/model/hypothesis.py``'s module docstring).
    """
    satisfied: list[Satisfied] = []
    unevaluable: list[Unevaluable] = []
    for constraint in constraints:
        outcome = evaluate_constraint(constraint, *args, **kwargs)
        match outcome:
            case Satisfied():
                satisfied.append(outcome)
            case Unevaluable():
                unevaluable.append(outcome)
            case HardConstraintViolation():
                prune_for_hard_violation(store, hypothesis_id, outcome)
                return HardConstraintCheck(
                    satisfied=tuple(satisfied),
                    unevaluable=tuple(unevaluable),
                    violation=outcome,
                )
            case _:
                assert_outcome_never(outcome)
    return HardConstraintCheck(
        satisfied=tuple(satisfied), unevaluable=tuple(unevaluable), violation=None
    )


def check_twin_dependent_constraints(
    constraints: Sequence[TwinDependentConstraint],
    *args: object,
    **kwargs: object,
) -> TwinDependentCheck:
    """Evaluate ``constraints`` against ``*args, **kwargs``; return every
    resulting :class:`TwinRevisionHypothesis`, unresolved, alongside what
    was satisfied and what could not be evaluated. No ``HypothesisStore``
    parameter exists on this function or on :func:`raise_twin_revision` —
    see the module docstring for why that absence, not a conditional, is
    what keeps these violations from pruning anything.

    Unlike the hard path this does not stop early: a twin-dependent
    violation refutes nothing, so there is no decision to short-circuit,
    and every ambiguity is worth recording.
    """
    satisfied: list[Satisfied] = []
    unevaluable: list[Unevaluable] = []
    hypotheses: list[TwinRevisionHypothesis] = []
    for constraint in constraints:
        outcome = evaluate_constraint(constraint, *args, **kwargs)
        match outcome:
            case Satisfied():
                satisfied.append(outcome)
            case Unevaluable():
                unevaluable.append(outcome)
            case TwinRevisionHypothesis():
                hypotheses.append(raise_twin_revision(outcome))
            case _:
                assert_outcome_never(outcome)
    return TwinDependentCheck(
        satisfied=tuple(satisfied),
        unevaluable=tuple(unevaluable),
        hypotheses=tuple(hypotheses),
    )


# ---------------------------------------------------------------------------
# Concrete hard constraints
# ---------------------------------------------------------------------------

ONE_BODY_ONE_PLACE_EPSILON_M = 1e-6
"""Two distinct physical bodies cannot occupy the same point — true by
definition, no calibration involved. This epsilon exists only to give an
otherwise-exact floating point coincidence check a tolerance; it does
not encode a body-size assumption the way, e.g., a minimum personal-space
radius would."""


class ConstraintInputError(ValueError):
    """A predicate was handed a malformed input (non-finite, or negative
    where physics forbids it). Distinct from every constraint OUTCOME:
    this says the question was ill-posed, not that the answer is
    unknown (:class:`~src.model.constraint.Unevaluable`) or bad
    (:class:`~src.model.constraint.HardConstraintViolation`)."""


def _require_finite(**values: float) -> None:
    """Reject NaN/inf inputs by raising, never by returning a verdict.

    A malformed input is a defect in the caller, not a statement about
    the world, and the two must not be conflated. Returning ``False``
    would prune a hypothesis for a bug; returning ``True`` would pass it
    for one; returning :class:`~src.model.constraint.Unevaluable` would
    file a bug under "missing world data" and lose it. Note
    ``not (nan <= x)`` is ``True`` in Python, so every threshold
    comparison below would silently report a VIOLATION on a NaN without
    this guard.
    """
    for name, value in values.items():
        if not math.isfinite(value):
            raise ConstraintInputError(
                f"{name}={value!r} is not finite; a constraint predicate "
                "cannot return a verdict about a malformed input"
            )


def _one_body_one_place_satisfied(
    position_a_m: tuple[float, float, float], position_b_m: tuple[float, float, float]
) -> bool:
    # Day 29: same NaN hazard as every other predicate here, and worse in
    # this one -- a NaN coordinate makes the separation NaN, `nan > eps` is
    # False, and the constraint reports a coincident-body VIOLATION, which
    # prunes. See _require_finite.
    _require_finite(
        **{f"position_a_m[{i}]": v for i, v in enumerate(position_a_m)},
        **{f"position_b_m[{i}]": v for i, v in enumerate(position_b_m)},
    )
    dx = position_a_m[0] - position_b_m[0]
    dy = position_a_m[1] - position_b_m[1]
    dz = position_a_m[2] - position_b_m[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) > ONE_BODY_ONE_PLACE_EPSILON_M


ONE_BODY_ONE_PLACE = HardConstraint(
    name="one_body_one_place",
    description=(
        "two distinct tracked entities cannot occupy the same point "
        "simultaneously; violation means at least one hypothesis "
        "mis-identifies which physical body it is tracking"
    ),
    predicate=_one_body_one_place_satisfied,
)
"""Fully implemented: no site geometry, no twin, no fitted constant
beyond a floating-point tolerance — true regardless of twin correctness,
as every hard constraint must be."""


def _max_pedestrian_velocity_satisfied(speed_mps: float) -> bool:
    """``speed_mps`` within the declared impossibility bound.

    Threshold is :data:`~src.estimator.motion_model.
    PEDESTRIAN_MAX_SPEED_MPS` — see that constant's own docstring for
    why the typical-scale pedestrian constants this module's motion
    model already uses could NOT serve here, and why using them would
    have passed Day 29's acceptance measurement while being wrong.

    Negative speed is malformed, not slow: speed is a magnitude.
    """
    _require_finite(speed_mps=speed_mps)
    if speed_mps < 0.0:
        raise ConstraintInputError(
            f"speed_mps={speed_mps} is negative; speed is a magnitude, so "
            "a negative value is a sign error at the call site, not a "
            "constraint verdict"
        )
    return speed_mps <= PEDESTRIAN_MAX_SPEED_MPS


MAX_PEDESTRIAN_VELOCITY = HardConstraint(
    name="max_pedestrian_velocity",
    description=(
        f"an entity's speed cannot exceed {PEDESTRIAN_MAX_SPEED_MPS} m/s, "
        "the fastest a human body is known to travel on foot; violation "
        "means the hypothesis has associated observations of two "
        "different bodies, not that someone moved very fast"
    ),
    predicate=_max_pedestrian_velocity_satisfied,
)


def _mass_conservation_satisfied(
    gap_displacement_m: float, gap_duration_s: float
) -> bool:
    """Continuity of existence: a body that leaves observation and
    returns must have been able to TRAVEL between the two places.

    A physical body cannot cease to exist and re-form elsewhere. So a
    re-acquisition at distance ``gap_displacement_m`` after an
    unobserved gap of ``gap_duration_s`` is admissible only if the gap
    was crossable at :data:`~src.estimator.motion_model.
    PEDESTRIAN_MAX_SPEED_MPS`. At ``gap_duration_s == 0`` the bound is
    ``0``: same instant, two places, is teleportation.

    Scope, and what was moved out of it
    -------------------------------------
    Day 28 described this constraint as "an entity cannot appear or
    vanish outside a modeled portal or occlusion boundary." That
    description is not hard — "modeled portal" and "occlusion boundary"
    are both twin geometry, and a constraint depending on the twin is
    twin-dependent by this module's own typing rule, whichever list Day
    28 filed it under. The hard, twin-free core is the reachability
    bound above: it needs no map, and it is true even if every wall in
    the building moved yesterday. Explaining an unreachable gap (a
    portal, a lift, an occluder) is the twin-dependent half and lives in
    :func:`portal_required`.
    """
    _require_finite(
        gap_displacement_m=gap_displacement_m, gap_duration_s=gap_duration_s
    )
    if gap_displacement_m < 0.0 or gap_duration_s < 0.0:
        raise ConstraintInputError(
            f"gap_displacement_m={gap_displacement_m}, "
            f"gap_duration_s={gap_duration_s}: both are magnitudes and "
            "neither may be negative"
        )
    return gap_displacement_m <= PEDESTRIAN_MAX_SPEED_MPS * gap_duration_s


MASS_CONSERVATION = HardConstraint(
    name="mass_conservation",
    description=(
        "an entity that leaves observation and returns must have been "
        "able to travel between the two places in the elapsed time; a "
        "body cannot cease to exist and re-form elsewhere"
    ),
    predicate=_mass_conservation_satisfied,
)


STANDARD_GRAVITY_MPS2 = 9.80665
"""Standard gravitational acceleration, the CGPM-defined value. A
physical constant, not a parameter of this project — cited, not chosen,
and not tunable."""


def _gravity_floor_transition_satisfied(
    delta_vertical_velocity_mps: float, dt_s: float
) -> bool:
    """Vertical dynamics: a body cannot accelerate DOWNWARD faster than
    free fall.

    ``a_z >= -g``, i.e. ``delta_vertical_velocity_mps >= -g * dt_s``,
    with sign convention ``+z`` up. Downward is bounded because gravity
    is the only force available to a body with nothing to push against;
    UPWARD acceleration is deliberately unbounded here, because a person
    pushing off the ground briefly exceeds ``g`` and a hard constraint
    may not refute that.

    Scope, and the typing error this corrects
    -------------------------------------------
    Day 28 described this constraint as "an entity's vertical position
    cannot change floor level without passing through a MODELED
    transition (stairs, lift, ramp)" — which is, word for word, the
    claim :func:`stair_or_lift_for_floor_change` already makes as a
    twin-dependent constraint. Day 28 filed one physical claim under
    both kinds. It cannot be both: "a modeled transition" is a fact
    about the twin, so that half is twin-dependent, and it stays where
    it already was. What remains here is the half that genuinely needs
    no twin — gravity itself, which holds whether or not the staircase
    is on the map.
    """
    _require_finite(
        delta_vertical_velocity_mps=delta_vertical_velocity_mps, dt_s=dt_s
    )
    if dt_s < 0.0:
        raise ConstraintInputError(
            f"dt_s={dt_s} is negative; a constraint cannot be evaluated "
            "backwards in time"
        )
    return delta_vertical_velocity_mps >= -STANDARD_GRAVITY_MPS2 * dt_s


GRAVITY_FLOOR_TRANSITION = HardConstraint(
    name="gravity_floor_transition",
    description=(
        "an entity cannot accelerate downward faster than free fall "
        f"({STANDARD_GRAVITY_MPS2} m/s^2); upward acceleration is "
        "unbounded here because a person pushing off the ground exceeds "
        "g and a hard constraint may not refute that"
    ),
    predicate=_gravity_floor_transition_satisfied,
)


HARD_CONSTRAINTS: tuple[HardConstraint, ...] = (
    ONE_BODY_ONE_PLACE,
    MAX_PEDESTRIAN_VELOCITY,
    MASS_CONSERVATION,
    GRAVITY_FLOOR_TRANSITION,
)
"""Every hard constraint, all four with running predicates as of Day 29.
Not a registry with lookup or registration — a tuple, because the only
thing anything needs from it is to iterate the complete set (see
``scripts/measure_gt_constraint_violations.py``, which measures exactly
these against GT). Note each takes DIFFERENT arguments, so this cannot
be passed to :func:`check_hard_constraints` as one batch; that is a
property of the constraints, not a defect in the collection."""


# ---------------------------------------------------------------------------
# Concrete twin-dependent constraints
#
# Each is a factory, not a module-level instance: unlike a hard
# constraint, a twin-dependent one is meaningless without stating which
# twin_rev it was checked against, so there is no single "the" instance
# to declare at import time — see TwinDependentConstraint's own
# undefaulted twin_rev field.
# ---------------------------------------------------------------------------


NO_TWIN_GEOMETRY = "no twin geometry"
"""The single reason string every twin-dependent predicate returns today.
One constant, not four copies: a caller counting how many constraints are
blocked on the same missing input should be able to group by an identity
comparison, and four hand-written strings drift."""


class TwinGeometry(Protocol):
    """What a twin-dependent predicate needs and this project does not
    have: queryable site geometry at a stated ``twin_rev``.

    No implementation of this Protocol exists anywhere in this codebase,
    and that absence is the point — it is why all four twin-dependent
    predicates return :class:`~src.model.constraint.Unevaluable` today.
    The Protocol is declared rather than left implicit so the gap is
    NAMED: each method below is a specific capability the twin pipeline
    must provide before the matching constraint can produce a verdict,
    and a test double implementing it (see
    ``tests/test_estimator_constraints.py``) proves the predicates do
    return real verdicts once geometry is supplied — rather than the
    unevaluable path being untestable and therefore permanent.

    Every method answers about a straight-line segment between two
    world-frame points, which is what a track step is. ``True`` means
    the segment does the thing the method names.
    """

    @property
    def twin_rev(self) -> int:
        """Which twin revision this geometry describes. Checked against
        the constraint's own ``twin_rev`` before any query runs — see
        :func:`_twin_geometry_for`."""

    def segment_crosses_solid_wall(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        ...

    def segment_crosses_portal_required_boundary(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        """``True`` if the segment crosses a zone boundary requiring a
        portal WITHOUT passing through one."""

    def segment_passes_stair_or_lift(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        ...

    def segment_is_reachable(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        """``True`` if the twin's visibility/accessibility graph admits
        the segment."""


def _twin_geometry_for(
    twin_geometry: TwinGeometry | None, constraint_twin_rev: int
) -> TwinGeometry | Unevaluable:
    """The gate every twin-dependent predicate passes through.

    Two ways to have no answer, both :class:`Unevaluable`, with
    different reasons:

    - No geometry at all. Today's permanent case.
    - Geometry at a DIFFERENT ``twin_rev`` than the constraint was built
      for. Not usable, and not silently usable either: this is the same
      rule ``src/model/world.py`` already enforces for a cross-revision
      position (never fall back to identity across a re-version). A
      constraint checked against the wrong revision's walls would
      produce a confident verdict about a building that has since
      changed — worse than no verdict, because it looks like one.
    """
    if twin_geometry is None:
        return Unevaluable(reason=NO_TWIN_GEOMETRY)
    if twin_geometry.twin_rev != constraint_twin_rev:
        return Unevaluable(
            reason=(
                f"twin geometry is at twin_rev={twin_geometry.twin_rev} but "
                f"the constraint was built for twin_rev="
                f"{constraint_twin_rev}; a verdict from the wrong "
                "revision's geometry is not a verdict"
            )
        )
    return twin_geometry


def wall_impermeability(twin_rev: int) -> TwinDependentConstraint:
    """A track step cannot pass through a wall the twin records as solid.

    Unevaluable without twin geometry (:class:`TwinGeometry`), which
    this project has none of at any revision.
    """

    def _predicate(
        start_m: tuple[float, float, float],
        end_m: tuple[float, float, float],
        twin_geometry: TwinGeometry | None = None,
    ) -> bool | Unevaluable:
        geometry = _twin_geometry_for(twin_geometry, twin_rev)
        if isinstance(geometry, Unevaluable):
            return geometry
        return not geometry.segment_crosses_solid_wall(start_m, end_m)

    return TwinDependentConstraint(
        name="wall_impermeability",
        description=(
            "an entity's track cannot cross a wall the twin records as "
            "solid at twin_rev"
        ),
        predicate=_predicate,
        twin_rev=twin_rev,
    )


def portal_required(twin_rev: int) -> TwinDependentConstraint:
    """A zone boundary the twin marks portal-required must be crossed
    through a portal. Also the twin-dependent half of Day 28's
    ``mass_conservation`` description — see
    :func:`_mass_conservation_satisfied`'s scope note.

    Unevaluable without twin geometry.
    """

    def _predicate(
        start_m: tuple[float, float, float],
        end_m: tuple[float, float, float],
        twin_geometry: TwinGeometry | None = None,
    ) -> bool | Unevaluable:
        geometry = _twin_geometry_for(twin_geometry, twin_rev)
        if isinstance(geometry, Unevaluable):
            return geometry
        return not geometry.segment_crosses_portal_required_boundary(start_m, end_m)

    return TwinDependentConstraint(
        name="portal_required",
        description=(
            "an entity cannot cross a zone boundary the twin records as "
            "requiring a portal at twin_rev"
        ),
        predicate=_predicate,
        twin_rev=twin_rev,
    )


def stair_or_lift_for_floor_change(twin_rev: int) -> TwinDependentConstraint:
    """A change of floor level must pass through a stair or lift the twin
    records. Holds the "modeled transition" half of Day 28's
    ``gravity_floor_transition`` description, which was typed hard by
    mistake — see :func:`_gravity_floor_transition_satisfied`'s scope
    note.

    Unevaluable without twin geometry.
    """

    def _predicate(
        start_m: tuple[float, float, float],
        end_m: tuple[float, float, float],
        twin_geometry: TwinGeometry | None = None,
    ) -> bool | Unevaluable:
        geometry = _twin_geometry_for(twin_geometry, twin_rev)
        if isinstance(geometry, Unevaluable):
            return geometry
        return geometry.segment_passes_stair_or_lift(start_m, end_m)

    return TwinDependentConstraint(
        name="stair_or_lift_for_floor_change",
        description=(
            "a floor change must pass through a stair or lift the twin "
            "records at twin_rev"
        ),
        predicate=_predicate,
        twin_rev=twin_rev,
    )


def visibility_and_accessibility(twin_rev: int) -> TwinDependentConstraint:
    """A track cannot pass through a region the twin's
    visibility/accessibility graph marks unreachable.

    Unevaluable without twin geometry.
    """

    def _predicate(
        start_m: tuple[float, float, float],
        end_m: tuple[float, float, float],
        twin_geometry: TwinGeometry | None = None,
    ) -> bool | Unevaluable:
        geometry = _twin_geometry_for(twin_geometry, twin_rev)
        if isinstance(geometry, Unevaluable):
            return geometry
        return geometry.segment_is_reachable(start_m, end_m)

    return TwinDependentConstraint(
        name="visibility_and_accessibility",
        description=(
            "an entity's track cannot pass through a region the twin's "
            "visibility/accessibility graph marks unreachable at twin_rev"
        ),
        predicate=_predicate,
        twin_rev=twin_rev,
    )


TWIN_DEPENDENT_CONSTRAINT_FACTORIES: tuple[
    Callable[[int], TwinDependentConstraint], ...
] = (
    wall_impermeability,
    portal_required,
    stair_or_lift_for_floor_change,
    visibility_and_accessibility,
)
"""Every twin-dependent constraint, as factories — see the section
comment above for why there is no module-level instance to collect. All
four are :class:`~src.model.constraint.Unevaluable` at every ``twin_rev``
today, for the same reason (:data:`NO_TWIN_GEOMETRY`)."""
