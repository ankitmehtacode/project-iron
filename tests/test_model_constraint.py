"""Day 28, Objective 2 — constraint typing: hard vs. twin-dependent.

STRUCTURAL rules under test:
  - TwinDependentConstraint is unconstructable without twin_rev (same
    shape as WorldPosition's undefaulted twin_rev, §0).
  - HardConstraint needs no twin_rev at all.
  - evaluate_constraint dispatches a HardConstraint violation to
    HardConstraintViolation and a TwinDependentConstraint violation to
    TwinRevisionHypothesis — never the other way around.
"""

from __future__ import annotations

import pytest

from src.model.constraint import (
    HardConstraint,
    HardConstraintViolation,
    TwinDependentConstraint,
    TwinRevisionHypothesis,
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


def test_evaluate_constraint_satisfied_hard_returns_none() -> None:
    constraint = HardConstraint(name="x", description="y", predicate=lambda: True)
    assert evaluate_constraint(constraint) is None


def test_evaluate_constraint_violated_hard_returns_hard_constraint_violation() -> None:
    constraint = HardConstraint(
        name="one_body_one_place",
        description="no two bodies at one point",
        predicate=lambda: False,
    )
    result = evaluate_constraint(constraint)
    assert isinstance(result, HardConstraintViolation)
    assert result.constraint_name == "one_body_one_place"


def test_evaluate_constraint_satisfied_twin_dependent_returns_none() -> None:
    constraint = TwinDependentConstraint(
        name="wall_impermeability", description="y", predicate=lambda: True, twin_rev=5
    )
    assert evaluate_constraint(constraint) is None


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
    assert evaluate_constraint(constraint, 1, b=1) is None
    assert seen == {"a": 1, "b": 1}
    violation = evaluate_constraint(constraint, 1, b=2)
    assert isinstance(violation, HardConstraintViolation)
