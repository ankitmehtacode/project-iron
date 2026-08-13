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
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from src.model.constraint import (
    HardConstraint,
    HardConstraintViolation,
    TwinDependentConstraint,
    TwinRevisionHypothesis,
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


def check_hard_constraints(
    store: HypothesisStore,
    hypothesis_id: str,
    constraints: Sequence[HardConstraint],
    *args: object,
    **kwargs: object,
) -> HardConstraintViolation | None:
    """Evaluate ``constraints`` in order against ``*args, **kwargs``; on
    the first violation, prune ``hypothesis_id`` in ``store`` and return
    the violation. ``None`` if every constraint is satisfied.

    The hypothesis-pruning factor the objective asks for: a real
    multi-hypothesis tracker would call this once per candidate
    hypothesis per step, once such a tracker exists (Day 28 keeps
    multi-hypothesis management itself skeleton — see
    ``src/model/hypothesis.py``'s module docstring).
    """
    for constraint in constraints:
        violation = evaluate_constraint(constraint, *args, **kwargs)
        if violation is not None:
            prune_for_hard_violation(store, hypothesis_id, violation)
            return violation
    return None


def check_twin_dependent_constraints(
    constraints: Sequence[TwinDependentConstraint],
    *args: object,
    **kwargs: object,
) -> tuple[TwinRevisionHypothesis, ...]:
    """Evaluate ``constraints`` against ``*args, **kwargs``; return every
    resulting :class:`TwinRevisionHypothesis`, unresolved. No
    ``HypothesisStore`` parameter exists on this function or on
    :func:`raise_twin_revision` — see the module docstring for why that
    absence, not a conditional, is what keeps these violations from
    pruning anything.
    """
    return tuple(
        raise_twin_revision(violation)
        for constraint in constraints
        if (violation := evaluate_constraint(constraint, *args, **kwargs)) is not None
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


def _one_body_one_place_satisfied(
    position_a_m: tuple[float, float, float], position_b_m: tuple[float, float, float]
) -> bool:
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


def _not_yet_derived(name: str) -> Callable[..., bool]:
    """A predicate that raises rather than guessing. Same convention as
    ``src/model/__init__.py``'s own placeholder rule: a function that
    raises ``NotImplementedError`` naming what will fill it, never a stub
    that silently returns a plausible answer."""

    def _predicate(*args: object, **kwargs: object) -> bool:
        raise NotImplementedError(
            f"{name}'s predicate is not implemented — Day 28 establishes "
            "its typed place in the hard/twin-dependent registry, not its "
            "physics. Deriving it (not fitting it — see Day 28 Objective "
            "1's own warning about that trap) is future work."
        )

    return _predicate


GRAVITY_FLOOR_TRANSITION = HardConstraint(
    name="gravity_floor_transition",
    description=(
        "an entity's vertical position cannot change floor level without "
        "passing through a modeled transition (stairs, lift, ramp)"
    ),
    predicate=_not_yet_derived("gravity_floor_transition"),
)

MAX_PEDESTRIAN_VELOCITY = HardConstraint(
    name="max_pedestrian_velocity",
    description=(
        "an entity's estimated speed cannot exceed a physically "
        "plausible pedestrian bound"
    ),
    predicate=_not_yet_derived("max_pedestrian_velocity"),
)

MASS_CONSERVATION = HardConstraint(
    name="mass_conservation",
    description=(
        "an entity cannot appear or vanish outside a modeled portal or "
        "occlusion boundary"
    ),
    predicate=_not_yet_derived("mass_conservation"),
)


# ---------------------------------------------------------------------------
# Concrete twin-dependent constraints
#
# Each is a factory, not a module-level instance: unlike a hard
# constraint, a twin-dependent one is meaningless without stating which
# twin_rev it was checked against, so there is no single "the" instance
# to declare at import time — see TwinDependentConstraint's own
# undefaulted twin_rev field.
# ---------------------------------------------------------------------------


def wall_impermeability(twin_rev: int) -> TwinDependentConstraint:
    return TwinDependentConstraint(
        name="wall_impermeability",
        description=(
            "an entity's track cannot cross a wall the twin records as "
            "solid at twin_rev"
        ),
        predicate=_not_yet_derived("wall_impermeability"),
        twin_rev=twin_rev,
    )


def portal_required(twin_rev: int) -> TwinDependentConstraint:
    return TwinDependentConstraint(
        name="portal_required",
        description=(
            "an entity cannot cross a zone boundary the twin records as "
            "requiring a portal at twin_rev"
        ),
        predicate=_not_yet_derived("portal_required"),
        twin_rev=twin_rev,
    )


def stair_or_lift_for_floor_change(twin_rev: int) -> TwinDependentConstraint:
    return TwinDependentConstraint(
        name="stair_or_lift_for_floor_change",
        description=(
            "a floor change must pass through a stair or lift the twin "
            "records at twin_rev"
        ),
        predicate=_not_yet_derived("stair_or_lift_for_floor_change"),
        twin_rev=twin_rev,
    )


def visibility_and_accessibility(twin_rev: int) -> TwinDependentConstraint:
    return TwinDependentConstraint(
        name="visibility_and_accessibility",
        description=(
            "an entity's track cannot pass through a region the twin's "
            "visibility/accessibility graph marks unreachable at twin_rev"
        ),
        predicate=_not_yet_derived("visibility_and_accessibility"),
        twin_rev=twin_rev,
    )
