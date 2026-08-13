"""Day 28, Objective 2 — hard constraints wired as hypothesis-pruning
factors; twin-dependent constraints routed away from pruning entirely.

STRUCTURAL rules under test:
  - A hard-constraint violation prunes the hypothesis in the store, with
    a RefutedByHardConstraint death cause.
  - A twin-dependent-constraint violation does NOT prune anything, and
    instead produces a TwinRevisionHypothesis.
  - The function that can prune (prune_for_hard_violation) has no
    counterpart reachable from a twin-dependent violation:
    raise_twin_revision takes no HypothesisStore parameter at all.
"""

from __future__ import annotations

import inspect

from src.estimator.constraints import (
    GRAVITY_FLOOR_TRANSITION,
    MASS_CONSERVATION,
    MAX_PEDESTRIAN_VELOCITY,
    ONE_BODY_ONE_PLACE,
    check_hard_constraints,
    check_twin_dependent_constraints,
    portal_required,
    prune_for_hard_violation,
    raise_twin_revision,
    stair_or_lift_for_floor_change,
    visibility_and_accessibility,
    wall_impermeability,
)
from src.model.constraint import HardConstraintViolation, TwinRevisionHypothesis
from src.model.hypothesis import HypothesisStore, RefutedByHardConstraint


def _store_with_one_hypothesis() -> tuple[HypothesisStore, str]:
    store = HypothesisStore()
    store.propose("h1", "candidate track hypothesis", 0.8)
    return store, "h1"


def test_one_body_one_place_satisfied_when_positions_differ() -> None:
    store, hid = _store_with_one_hypothesis()
    violation = check_hard_constraints(
        store, hid, [ONE_BODY_ONE_PLACE], (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    )
    assert violation is None
    assert [h.id for h in store.alive()] == [hid]


def test_one_body_one_place_violated_prunes_the_hypothesis() -> None:
    store, hid = _store_with_one_hypothesis()
    violation = check_hard_constraints(
        store, hid, [ONE_BODY_ONE_PLACE], (2.0, 3.0, 0.5), (2.0, 3.0, 0.5)
    )
    assert isinstance(violation, HardConstraintViolation)
    assert violation.constraint_name == "one_body_one_place"
    assert store.alive() == ()
    record = store.considered_alternatives()[0]
    assert isinstance(record.cause, RefutedByHardConstraint)
    assert record.cause.constraint_name == "one_body_one_place"


def test_hard_constraint_violation_routes_to_prune_for_hard_violation_directly() -> (
    None
):
    store, hid = _store_with_one_hypothesis()
    violation = HardConstraintViolation(
        constraint_name="max_pedestrian_velocity", description="too fast"
    )
    prune_for_hard_violation(store, hid, violation)
    assert store.alive() == ()
    assert isinstance(store.considered_alternatives()[0].cause, RefutedByHardConstraint)


def test_twin_dependent_violation_does_not_prune_and_emits_a_revision_hypothesis() -> (
    None
):
    store, hid = _store_with_one_hypothesis()
    constraint = wall_impermeability(twin_rev=7)
    # Force a violation directly, bypassing the not-yet-derived predicate,
    # so this test exercises the routing, not the (deliberately
    # unimplemented) physics.
    from dataclasses import replace

    violating_constraint = replace(constraint, predicate=lambda *a, **k: False)
    results = check_twin_dependent_constraints(
        [violating_constraint], (0.0, 0.0, 0.0), (5.0, 0.0, 0.0)
    )
    assert len(results) == 1
    hypothesis = results[0]
    assert isinstance(hypothesis, TwinRevisionHypothesis)
    assert hypothesis.constraint_name == "wall_impermeability"
    assert hypothesis.twin_rev == 7
    # Nothing was pruned -- the hypothesis proposed above is still alive.
    assert [h.id for h in store.alive()] == [hid]
    assert store.considered_alternatives() == ()


def test_raise_twin_revision_has_no_hypothesis_store_parameter() -> None:
    """Structural guarantee, not a runtime check: this function cannot
    reach a HypothesisStore because none is ever passed to it."""
    params = inspect.signature(raise_twin_revision).parameters
    assert "store" not in params
    assert not any(
        p.annotation is HypothesisStore
        for p in params.values()
        if p.annotation is not p.empty
    )


def test_check_hard_constraints_stops_at_the_first_violation() -> None:
    store, hid = _store_with_one_hypothesis()
    calls: list[str] = []

    from src.model.constraint import HardConstraint

    def _record_and_pass(name: str) -> bool:
        calls.append(name)
        return True

    first_fails = HardConstraint(
        name="first", description="d", predicate=lambda: calls.append("first") or False
    )
    second = HardConstraint(
        name="second", description="d", predicate=lambda: _record_and_pass("second")
    )
    violation = check_hard_constraints(store, hid, [first_fails, second])
    assert violation is not None
    assert violation.constraint_name == "first"
    assert calls == ["first"]
    assert store.alive() == ()


def test_named_hard_constraints_cover_the_four_examples_from_the_objective() -> None:
    names = {
        ONE_BODY_ONE_PLACE.name,
        GRAVITY_FLOOR_TRANSITION.name,
        MAX_PEDESTRIAN_VELOCITY.name,
        MASS_CONSERVATION.name,
    }
    assert names == {
        "one_body_one_place",
        "gravity_floor_transition",
        "max_pedestrian_velocity",
        "mass_conservation",
    }
    assert all(
        c.kind == "hard"
        for c in (
            ONE_BODY_ONE_PLACE,
            GRAVITY_FLOOR_TRANSITION,
            MAX_PEDESTRIAN_VELOCITY,
            MASS_CONSERVATION,
        )
    )


def test_named_twin_dependent_constraints_require_twin_rev() -> None:
    for factory in (
        wall_impermeability,
        portal_required,
        stair_or_lift_for_floor_change,
        visibility_and_accessibility,
    ):
        constraint = factory(twin_rev=9)
        assert constraint.kind == "twin_dependent"
        assert constraint.twin_rev == 9


def test_not_yet_derived_predicates_raise_rather_than_guess() -> None:
    import pytest

    with pytest.raises(NotImplementedError):
        GRAVITY_FLOOR_TRANSITION.predicate()
    with pytest.raises(NotImplementedError):
        MAX_PEDESTRIAN_VELOCITY.predicate()
    with pytest.raises(NotImplementedError):
        MASS_CONSERVATION.predicate()
    with pytest.raises(NotImplementedError):
        wall_impermeability(twin_rev=1).predicate()
