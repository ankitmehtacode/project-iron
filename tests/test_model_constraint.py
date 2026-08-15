"""Day 28, Objective 2 — constraint typing: hard vs. twin-dependent.
Day 29, Objective 3 — the third outcome, Unevaluable.

STRUCTURAL rules under test:
  - TwinDependentConstraint is unconstructable without twin_rev (same
    shape as WorldPosition's undefaulted twin_rev, §0).
  - HardConstraint needs no twin_rev at all.
  - evaluate_constraint dispatches a HardConstraint violation to
    HardConstraintViolation and a TwinDependentConstraint violation to
    TwinRevisionHypothesis — never the other way around.
  - (Day 29) The outcome set is three-valued and closed: Satisfied,
    Unevaluable, or the kind's violation type. No None, so no `is None`
    idiom exists for a caller to get backwards.
  - (Day 29) A predicate returning anything outside {bool, Unevaluable}
    raises rather than being truthy-tested into a verdict.
"""

from __future__ import annotations

import pytest

from src.model.constraint import (
    ConstraintPredicateError,
    HardConstraint,
    HardConstraintViolation,
    Satisfied,
    TwinDependentConstraint,
    TwinRevisionHypothesis,
    Unevaluable,
    evaluate_constraint,
)


def test_hard_constraint_needs_no_twin_rev() -> None:
    constraint = HardConstraint(
        name="always_true", description="trivially satisfied", predicate=lambda: True
    )
    assert constraint.kind == "hard"
    assert not hasattr(constraint, "twin_rev")


def test_twin_dependent_constraint_unconstructable_without_twin_rev() -> None:
    with pytest.raises(TypeError):
        TwinDependentConstraint(  # type: ignore[call-arg]
            name="wall_impermeability",
            description="cannot cross a solid wall",
            predicate=lambda: True,
        )


def test_twin_dependent_constraint_constructs_with_twin_rev() -> None:
    constraint = TwinDependentConstraint(
        name="wall_impermeability",
        description="cannot cross a solid wall",
        predicate=lambda: True,
        twin_rev=3,
    )
    assert constraint.kind == "twin_dependent"
    assert constraint.twin_rev == 3


def test_hard_constraint_kind_is_immutable() -> None:
    constraint = HardConstraint(name="x", description="y", predicate=lambda: True)
    with pytest.raises(AttributeError):
        constraint.kind = "twin_dependent"  # type: ignore[misc]


def test_evaluate_constraint_satisfied_hard_returns_satisfied() -> None:
    constraint = HardConstraint(name="x", description="y", predicate=lambda: True)
    outcome = evaluate_constraint(constraint)
    assert isinstance(outcome, Satisfied)
    assert outcome.constraint_name == "x"


def test_evaluate_constraint_violated_hard_returns_hard_constraint_violation() -> None:
    constraint = HardConstraint(
        name="one_body_one_place",
        description="no two bodies at one point",
        predicate=lambda: False,
    )
    result = evaluate_constraint(constraint)
    assert isinstance(result, HardConstraintViolation)
    assert result.constraint_name == "one_body_one_place"


def test_evaluate_constraint_satisfied_twin_dependent_returns_satisfied() -> None:
    constraint = TwinDependentConstraint(
        name="wall_impermeability", description="y", predicate=lambda: True, twin_rev=5
    )
    outcome = evaluate_constraint(constraint)
    assert isinstance(outcome, Satisfied)
    assert outcome.constraint_name == "wall_impermeability"


def test_evaluate_constraint_violated_twin_dependent_returns_revision_hypothesis() -> (
    None
):
    constraint = TwinDependentConstraint(
        name="wall_impermeability", description="y", predicate=lambda: False, twin_rev=5
    )
    result = evaluate_constraint(constraint)
    assert isinstance(result, TwinRevisionHypothesis)
    assert result.constraint_name == "wall_impermeability"
    assert result.twin_rev == 5
    assert not isinstance(result, HardConstraintViolation)


def test_evaluate_constraint_passes_args_and_kwargs_through_to_predicate() -> None:
    seen: dict[str, object] = {}

    def predicate(a: int, *, b: int) -> bool:
        seen["a"] = a
        seen["b"] = b
        return a == b

    constraint = HardConstraint(
        name="equal", description="a equals b", predicate=predicate
    )
    assert isinstance(evaluate_constraint(constraint, 1, b=1), Satisfied)
    assert seen == {"a": 1, "b": 1}
    violation = evaluate_constraint(constraint, 1, b=2)
    assert isinstance(violation, HardConstraintViolation)


# ---------------------------------------------------------------------------
# Day 29 — Unevaluable, the third outcome
# ---------------------------------------------------------------------------


def test_unevaluable_is_returned_intact_and_stamped_with_the_constraint_name() -> None:
    constraint = HardConstraint(
        name="needs_geometry",
        description="y",
        predicate=lambda: Unevaluable(reason="no twin geometry"),
    )
    outcome = evaluate_constraint(constraint)
    assert isinstance(outcome, Unevaluable)
    assert outcome.reason == "no twin geometry"
    assert outcome.constraint_name == "needs_geometry"


def test_the_constraint_not_the_predicate_is_the_authority_on_its_name() -> None:
    """A predicate that names itself something else is overruled: the
    constraint's own name is the one recorded."""
    constraint = TwinDependentConstraint(
        name="wall_impermeability",
        description="y",
        predicate=lambda: Unevaluable(reason="r", constraint_name="something_else"),
        twin_rev=1,
    )
    outcome = evaluate_constraint(constraint)
    assert isinstance(outcome, Unevaluable)
    assert outcome.constraint_name == "wall_impermeability"


def test_unevaluable_is_neither_satisfied_nor_a_violation_of_either_kind() -> None:
    """The type-level statement of the rule "an unevaluable constraint
    may not contribute a PASS to any hypothesis decision"."""
    for constraint in (
        HardConstraint(
            name="h", description="y", predicate=lambda: Unevaluable(reason="r")
        ),
        TwinDependentConstraint(
            name="t", description="y", predicate=lambda: Unevaluable(reason="r"),
            twin_rev=1,
        ),
    ):
        outcome = evaluate_constraint(constraint)
        assert isinstance(outcome, Unevaluable)
        assert not isinstance(outcome, Satisfied)
        assert not isinstance(outcome, HardConstraintViolation)
        assert not isinstance(outcome, TwinRevisionHypothesis)


def test_evaluate_constraint_never_returns_none_for_any_outcome() -> None:
    """None is gone from the contract entirely — that removal, not the
    addition of Unevaluable, is what makes the `is None` misreading
    impossible to write."""
    predicates = (
        lambda: True,
        lambda: False,
        lambda: Unevaluable(reason="r"),
    )
    for predicate in predicates:
        constraint = HardConstraint(name="x", description="y", predicate=predicate)
        assert evaluate_constraint(constraint) is not None


@pytest.mark.parametrize("bad_return", [None, 0, 1, "", "no", [], object()])
def test_a_predicate_returning_a_non_verdict_raises(bad_return: object) -> None:
    """Neither reading of an ambiguous return is safe to guess at: a
    truthy test would turn a half-written predicate's None into a
    violation, and a non-empty error string into a PASS."""
    constraint = HardConstraint(
        name="x", description="y", predicate=lambda: bad_return
    )
    with pytest.raises(ConstraintPredicateError):
        evaluate_constraint(constraint)
