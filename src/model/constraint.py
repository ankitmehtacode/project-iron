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
body cannot occupy two places, nothing falls faster than gravity, a
pedestrian's speed is bounded, and a body cannot cease to exist and
re-form somewhere it could not have walked to. Violating one means the
HYPOTHESIS is wrong — nothing about the twin's own correctness is in
question, so the only sound action is pruning that hypothesis.

(Day 29: this paragraph previously read "mass does not appear or vanish
outside a modeled PORTAL or OCCLUSION BOUNDARY," which named twin
geometry inside the definition of the kind that must not depend on it.
Naming the twin is the tell that a constraint — or a description of one
— is mis-typed; see ``src/estimator/constraints.py``'s Day-29 section
for the two concrete constraints this caught.)

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

Day 29 — the third outcome, and why ``None`` could not stay
--------------------------------------------------------------
Day 28's :func:`evaluate_constraint` returned ``None`` for "satisfied"
and a violation object otherwise. That is a TWO-outcome world, and it is
only sound while every predicate can actually run. Day 29 fills the seven
predicates Day 28 left stubbed, and four of them (the twin-dependent
ones) cannot run at all: this project has no twin geometry — no wall
set, no portal graph, no stair/lift inventory, no
visibility/accessibility graph — at any ``twin_rev``. A predicate with
nothing to check against has three honest options, and only one of them
is acceptable:

1. Return ``True``. This is the manufactured-assurance failure: the
   hypothesis decision records a PASS for a constraint that was never
   evaluated, and the pass is indistinguishable downstream from one
   earned against real geometry. Prohibited.
2. Raise. Correct for Day 28, when the predicates were placeholders and
   raising named what would fill them. Wrong now: "no twin geometry
   exists" is an expected, permanent-until-surveyed state of the world,
   not a bug in the caller, and a caller checking four constraints
   should not have to catch an exception per constraint to learn that
   three of them are unanswerable here.
3. Return :class:`Unevaluable`, a value the type system keeps separate
   from :class:`Satisfied`. Adopted.

So the outcome set is now closed and three-valued per kind
(:data:`HardConstraintOutcome` / :data:`TwinDependentOutcome`), and
``None`` is gone entirely. Removing ``None`` rather than adding
``Unevaluable`` alongside it is the point: a caller writing the natural
``if evaluate_constraint(...) is None: # satisfied`` would have silently
classified ``Unevaluable`` as a violation, and the inverse spelling
``is not None`` would have classified it as one too — one of the two
readings is wrong whichever way the caller guesses, and neither spelling
is visibly wrong at the call site. With ``None`` removed there is no
``is None`` idiom left to get backwards; a caller must name which of the
three outcomes it means, and ``mypy`` checks that it handled the rest.

:class:`Satisfied` is a value, not a sentinel, for the same reason: it
carries the ``constraint_name`` that actually passed, so "this decision
was made against these constraints, all evaluated" is recordable rather
than inferred from an absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal, NoReturn, Union, overload

ConstraintKind = Literal["hard", "twin_dependent"]


@dataclass(frozen=True)
class Unevaluable:
    """The predicate could not run — no verdict, in either direction.

    NOT a pass and NOT a violation. A hypothesis decision may record
    that this constraint was unevaluable; it may never count it as
    evidence the hypothesis is sound. ``reason`` states what was missing
    (e.g. ``"no twin geometry"``) so an unevaluable constraint is
    actionable — it names what would have to exist for the check to
    become real.

    A predicate constructs this with ``reason`` alone;
    :func:`evaluate_constraint` stamps ``constraint_name`` from the
    constraint itself, which is the only authority on its own name (a
    predicate that set the name would be free to set a different one).
    """

    reason: str
    constraint_name: str = ""


ConstraintPredicate = Callable[..., Union[bool, Unevaluable]]
"""``True`` satisfied, ``False`` violated, :class:`Unevaluable` could not
run. Nothing else: :func:`evaluate_constraint` raises
:class:`ConstraintPredicateError` rather than truthy-testing an
unexpected return value."""


@dataclass(frozen=True)
class HardConstraint:
    """A constraint true regardless of twin correctness.

    ``predicate`` returns ``True`` when the constraint is satisfied,
    ``False`` when it is violated, or :class:`Unevaluable` when it could
    not run at all. Violation means the track hypothesis itself is wrong.
    """

    name: str
    description: str
    predicate: ConstraintPredicate
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
    predicate: ConstraintPredicate
    twin_rev: int
    kind: Literal["twin_dependent"] = field(default="twin_dependent", init=False)


Constraint = Union[HardConstraint, TwinDependentConstraint]


@dataclass(frozen=True)
class Satisfied:
    """The predicate RAN and the constraint holds.

    Distinct from :class:`Unevaluable` by type, which is the whole point
    (see the module docstring's Day-29 section): only this outcome may
    be read as a constraint contributing a PASS to a hypothesis
    decision.
    """

    constraint_name: str


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


HardConstraintOutcome = Union[Satisfied, Unevaluable, HardConstraintViolation]
"""Closed: evaluating a hard constraint yields exactly one of these three."""

TwinDependentOutcome = Union[Satisfied, Unevaluable, TwinRevisionHypothesis]
"""Closed: evaluating a twin-dependent constraint yields exactly one of
these three. :class:`HardConstraintViolation` is not a member, so no
twin-dependent outcome can reach the pruning path."""

ConstraintOutcome = Union[HardConstraintOutcome, TwinDependentOutcome]


def assert_outcome_never(outcome: NoReturn) -> NoReturn:
    """Exhaustiveness marker for a ``match`` over a constraint outcome.

    ``typing.assert_never`` is Python 3.11+; this project targets 3.10
    (``setup.py``) and does not otherwise depend on ``typing_extensions``,
    so the pre-3.11 idiom is spelled out here rather than taking a
    dependency for one line. It does the same job: ``mypy`` accepts the
    call only where the matched value has narrowed to ``Never``, so a
    fourth outcome type added to :data:`HardConstraintOutcome` or
    :data:`TwinDependentOutcome` becomes a type error at every
    ``match`` that forgot it — which is the property that makes
    :class:`Unevaluable` impossible to silently skip.
    """
    raise AssertionError(f"unhandled constraint outcome: {outcome!r}")


class ConstraintPredicateError(TypeError):
    """A predicate returned something that is neither ``bool`` nor
    :class:`Unevaluable`.

    Raised, never coerced. A predicate returning ``None``, ``0``, or an
    arbitrary object is a defect in that predicate, and the one thing
    this module must not do with an ambiguous verdict is pick a plausible
    reading of it — silently truthy-testing the return value is how a
    ``None`` from a half-written predicate becomes a hard-constraint
    violation, and a non-empty error string becomes a PASS.
    """


@overload
def evaluate_constraint(
    constraint: HardConstraint, *args: object, **kwargs: object
) -> HardConstraintOutcome:
    ...


@overload
def evaluate_constraint(
    constraint: TwinDependentConstraint, *args: object, **kwargs: object
) -> TwinDependentOutcome:
    ...


def evaluate_constraint(
    constraint: Constraint, *args: object, **kwargs: object
) -> ConstraintOutcome:
    """Run ``constraint.predicate(*args, **kwargs)`` and name the outcome.

    Returns :class:`Satisfied`, :class:`Unevaluable`, or the violation
    type matching ``constraint``'s kind. There is no ``None`` return —
    see the module docstring's Day-29 section for why removing it was
    the fix rather than adding a fourth case beside it.

    The two ``@overload`` signatures above are the actual typing
    contract: a caller holding a ``HardConstraint`` gets back
    ``HardConstraintOutcome`` with no narrowing needed; a caller holding
    a ``TwinDependentConstraint`` gets back ``TwinDependentOutcome``,
    which cannot contain a :class:`HardConstraintViolation` at all. The
    ``isinstance`` check in this implementation exists to pick the right
    violation type to construct, not to decide whether a violation may
    prune anything — that decision is made by which function a caller
    passes the result to next (see the module docstring).

    Raises:
        ConstraintPredicateError: if the predicate returns a value that
            is neither ``bool`` nor :class:`Unevaluable`.
    """
    verdict = constraint.predicate(*args, **kwargs)

    if isinstance(verdict, Unevaluable):
        return replace(verdict, constraint_name=constraint.name)
    if not isinstance(verdict, bool):
        raise ConstraintPredicateError(
            f"{constraint.name}'s predicate returned "
            f"{type(verdict).__name__} ({verdict!r}); a predicate must "
            "return bool (ran, and here is the verdict) or Unevaluable "
            "(could not run). Neither reading of an ambiguous value is "
            "safe to guess at."
        )
    if verdict:
        return Satisfied(constraint_name=constraint.name)
    if isinstance(constraint, HardConstraint):
        return HardConstraintViolation(
            constraint_name=constraint.name, description=constraint.description
        )
    return TwinRevisionHypothesis(
        constraint_name=constraint.name,
        description=constraint.description,
        twin_rev=constraint.twin_rev,
    )
