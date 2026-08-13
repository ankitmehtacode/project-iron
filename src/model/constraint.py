"""Track-hypothesis constraints, typed by what they depend on.

Closes ``docs/data_model/v0.3.md``'s "Constraint typing — hard vs.
twin-dependent" finding: cited as settled architecture since at least
Day 20 (``src/estimator/consistency.py``'s ``_CONSTRAINT_CONSUMER``
stub, "twin-revision hypothesis (not yet implemented)"), with no
``ConstraintKind`` type, no hard/twin-dependent distinction, and no code
path that ever raised anything resembling a twin-revision hypothesis
object, anywhere, before today (Day 28).

Why the distinction is physical, not stylistic
-----------------------------------------------
A track HYPOTHESIS is a claim about where a physical body is and how it
got there — it is not ground truth, and the 3D twin it is checked
against is not ground truth either (a twin can be stale: an unmapped
staircase, a propped-open fire door, a moved partition).

``kind="hard"`` — true regardless of whether the twin is current: one
body cannot occupy two places, gravity does not suspend, a pedestrian's
speed is bounded, mass does not appear or vanish outside a modeled
portal or occlusion boundary. Violating one means the HYPOTHESIS is
wrong — nothing about the twin's own correctness is in question, so the
only sound action is pruning that hypothesis.

``kind="twin_dependent"`` — true only as far as the twin's own
``twin_rev`` is current: wall impermeability, portal-required movement,
stair-or-lift-for-floor-change, visibility/accessibility. Violating one
is AMBIGUOUS between "the hypothesis is wrong" and "the twin is stale" —
the world may have changed since the twin was last surveyed. Pruning the
hypothesis here would silently treat a possibly-current, possibly-right
track as false because a possibly-stale map said so. The sound action is
recording the ambiguity as a :class:`TwinRevisionHypothesis`, not
resolving it.

STRUCTURAL: a :class:`TwinDependentConstraint` cannot be constructed
without a ``twin_rev`` — same shape as :class:`~src.model.world.
WorldPosition`'s undefaulted ``twin_rev`` field (§0), for the identical
reason: a claim that depends on the twin's state is meaningless without
saying which twin state it was checked against.

STRUCTURAL: the pruning path accepts only :class:`HardConstraintViolation`
-- see :func:`evaluate_constraint`'s ``@overload`` pair. Passing a
:class:`TwinDependentConstraint` through the ``HardConstraint`` overload
is a ``mypy`` error, not a runtime branch a caller could get wrong; there
is no function anywhere that takes a :class:`TwinRevisionHypothesis` and
a hypothesis store together (see ``src/estimator/constraints.py``). The
kinds are routed apart by TYPE, before either violation object exists,
which is the "closed-world dispatch, not an if" the pruning rule needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Union, overload

ConstraintKind = Literal["hard", "twin_dependent"]


@dataclass(frozen=True)
class HardConstraint:
    """A constraint true regardless of twin correctness.

    ``predicate`` returns ``True`` when the constraint is satisfied.
    Violation means the track hypothesis itself is wrong.
    """

    name: str
    description: str
    predicate: Callable[..., bool]
    kind: Literal["hard"] = field(default="hard", init=False)


@dataclass(frozen=True)
class TwinDependentConstraint:
    """A constraint only as true as ``twin_rev``.

    ``twin_rev`` is undefaulted: there is no way to build a
    twin-dependent constraint without stating which twin revision it was
    checked against, the same rule :class:`~src.model.world.WorldPosition`
    already enforces for a raw position (§0).
    """

    name: str
    description: str
    predicate: Callable[..., bool]
    twin_rev: int
    kind: Literal["twin_dependent"] = field(default="twin_dependent", init=False)


Constraint = Union[HardConstraint, TwinDependentConstraint]


@dataclass(frozen=True)
class HardConstraintViolation:
    """A hard constraint failed. The hypothesis is refuted; see
    ``src/estimator/constraints.py::prune_for_hard_violation``, the only
    function that consumes this type."""

    constraint_name: str
    description: str


@dataclass(frozen=True)
class TwinRevisionHypothesis:
    """A twin-dependent constraint failed at ``twin_rev``.

    Not a refutation of the track hypothesis — an unresolved ambiguity
    between "the hypothesis is wrong" and "the twin is stale at this
    revision." Recorded as a real, typed value (Day 28); it has NO
    consumer yet anywhere in this codebase — nothing reads a
    ``TwinRevisionHypothesis`` once it is raised. Repeated violations of
    the same constraint at the same location, tracked over time, are the
    "twin drift detector" this project has wanted since early on — that
    capability falls out of typing constraints correctly rather than
    needing separate engineering, but the tracking itself is not built
    today; only the typed value it would consume is.
    """

    constraint_name: str
    description: str
    twin_rev: int


@overload
def evaluate_constraint(
    constraint: HardConstraint, *args: object, **kwargs: object
) -> HardConstraintViolation | None:
    ...


@overload
def evaluate_constraint(
    constraint: TwinDependentConstraint, *args: object, **kwargs: object
) -> TwinRevisionHypothesis | None:
    ...


def evaluate_constraint(
    constraint: Constraint, *args: object, **kwargs: object
) -> HardConstraintViolation | TwinRevisionHypothesis | None:
    """Run ``constraint.predicate(*args, **kwargs)``; ``None`` if satisfied.

    The two ``@overload`` signatures above are the actual typing
    contract: a caller holding a ``HardConstraint`` gets back
    ``HardConstraintViolation | None`` with no narrowing needed: a
    caller holding a ``TwinDependentConstraint`` gets back
    ``TwinRevisionHypothesis | None``. The ``isinstance`` check in this
    implementation exists to pick the right violation type to
    construct, not to decide whether a violation may prune anything —
    that decision is made by which function a caller passes the result
    to next (see the module docstring).
    """
    if constraint.predicate(*args, **kwargs):
        return None
    if isinstance(constraint, HardConstraint):
        return HardConstraintViolation(
            constraint_name=constraint.name, description=constraint.description
        )
    return TwinRevisionHypothesis(
        constraint_name=constraint.name,
        description=constraint.description,
        twin_rev=constraint.twin_rev,
    )
