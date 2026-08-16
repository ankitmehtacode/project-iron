"""Day 28, Objective 2 — hard constraints wired as hypothesis-pruning
factors; twin-dependent constraints routed away from pruning entirely.
Day 29, Objective 3 — the seven predicates filled, and the unevaluable
path that keeps four of them from passing vacuously.

STRUCTURAL rules under test:
  - A hard-constraint violation prunes the hypothesis in the store, with
    a RefutedByHardConstraint death cause.
  - A twin-dependent-constraint violation does NOT prune anything, and
    instead produces a TwinRevisionHypothesis.
  - The function that can prune (prune_for_hard_violation) has no
    counterpart reachable from a twin-dependent violation:
    raise_twin_revision takes no HypothesisStore parameter at all.
  - (Day 29) An Unevaluable outcome never prunes, never counts as a
    pass, and is never silently dropped: all_satisfied is False whenever
    anything was unevaluable, even with zero violations.
  - (Day 29) The unevaluable path is not a permanent dead end — supplied
    with a TwinGeometry test double, the same predicates return real
    verdicts, in both directions.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace

import pytest

from src.estimator.constraints import (
    GRAVITY_FLOOR_TRANSITION,
    HARD_CONSTRAINTS,
    MASS_CONSERVATION,
    MAX_PEDESTRIAN_ACCELERATION,
    MAX_PEDESTRIAN_VELOCITY,
    NO_TWIN_GEOMETRY,
    ONE_BODY_ONE_PLACE,
    STANDARD_GRAVITY_MPS2,
    TWIN_DEPENDENT_CONSTRAINT_FACTORIES,
    ConstraintInputError,
    check_hard_constraints,
    check_twin_dependent_constraints,
    portal_required,
    prune_for_hard_violation,
    raise_twin_revision,
    stair_or_lift_for_floor_change,
    visibility_and_accessibility,
    wall_impermeability,
)
from src.estimator.motion_model import (
    PEDESTRIAN_MAX_ACCELERATION_MPS2,
    PEDESTRIAN_MAX_DECELERATION_MPS2,
    PEDESTRIAN_MAX_JERK_MPS3,
    PEDESTRIAN_MAX_SPEED_MPS,
    PERSON_SIGMA_A_MPS2,
)
from src.model.constraint import (
    HardConstraint,
    HardConstraintViolation,
    Satisfied,
    TwinRevisionHypothesis,
    Unevaluable,
    evaluate_constraint,
)
from src.model.hypothesis import HypothesisStore, RefutedByHardConstraint

ORIGIN = (0.0, 0.0, 0.0)
ONE_METRE_EAST = (1.0, 0.0, 0.0)


def _store_with_one_hypothesis() -> tuple[HypothesisStore, str]:
    store = HypothesisStore()
    store.propose("h1", "candidate track hypothesis", 0.8)
    return store, "h1"


@dataclass(frozen=True)
class FakeTwinGeometry:
    """A test double for :class:`~src.estimator.constraints.TwinGeometry`.

    Deliberately a fixture and not an implementation: every answer is a
    constructor argument, because the point is to prove the predicates
    consume a geometry correctly, not to build the geometry pipeline
    (which does not exist — see the Protocol's own docstring). Defaults
    are the permissive answers, so each test sets only the one fact it
    is about.
    """

    twin_rev: int
    crosses_wall: bool = False
    crosses_portal_boundary: bool = False
    passes_stair_or_lift: bool = True
    reachable: bool = True

    def segment_crosses_solid_wall(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        return self.crosses_wall

    def segment_crosses_portal_required_boundary(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        return self.crosses_portal_boundary

    def segment_passes_stair_or_lift(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        return self.passes_stair_or_lift

    def segment_is_reachable(
        self, start_m: tuple[float, float, float], end_m: tuple[float, float, float]
    ) -> bool:
        return self.reachable


# ---------------------------------------------------------------------------
# Day 28 — routing (hard prunes, twin-dependent cannot)
# ---------------------------------------------------------------------------


def test_one_body_one_place_satisfied_when_positions_differ() -> None:
    store, hid = _store_with_one_hypothesis()
    check = check_hard_constraints(
        store, hid, [ONE_BODY_ONE_PLACE], ORIGIN, ONE_METRE_EAST
    )
    assert check.violation is None
    assert check.all_satisfied
    assert [s.constraint_name for s in check.satisfied] == ["one_body_one_place"]
    assert [h.id for h in store.alive()] == [hid]


def test_one_body_one_place_violated_prunes_the_hypothesis() -> None:
    store, hid = _store_with_one_hypothesis()
    check = check_hard_constraints(
        store, hid, [ONE_BODY_ONE_PLACE], (2.0, 3.0, 0.5), (2.0, 3.0, 0.5)
    )
    assert isinstance(check.violation, HardConstraintViolation)
    assert check.violation.constraint_name == "one_body_one_place"
    assert check.pruned
    assert not check.all_satisfied
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
    check = check_twin_dependent_constraints(
        [constraint],
        ORIGIN,
        (5.0, 0.0, 0.0),
        twin_geometry=FakeTwinGeometry(twin_rev=7, crosses_wall=True),
    )
    assert len(check.hypotheses) == 1
    hypothesis = check.hypotheses[0]
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

    def _record_and_pass(name: str) -> bool:
        calls.append(name)
        return True

    first_fails = HardConstraint(
        name="first", description="d", predicate=lambda: calls.append("first") or False
    )
    second = HardConstraint(
        name="second", description="d", predicate=lambda: _record_and_pass("second")
    )
    check = check_hard_constraints(store, hid, [first_fails, second])
    assert check.violation is not None
    assert check.violation.constraint_name == "first"
    assert calls == ["first"]
    assert store.alive() == ()


def test_named_hard_constraints_cover_the_four_examples_from_the_objective() -> None:
    """Day 28's four, plus Day 30's `max_pedestrian_acceleration` — added
    because Day 29 measured GT at 3.7g that none of the original four
    could refute (a walking SPEED arrived at impossibly fast, entirely
    horizontal, so neither the speed bound nor the free-fall bound
    applies)."""
    assert {c.name for c in HARD_CONSTRAINTS} == {
        "one_body_one_place",
        "gravity_floor_transition",
        "max_pedestrian_velocity",
        "max_pedestrian_acceleration",
        "mass_conservation",
    }
    assert all(c.kind == "hard" for c in HARD_CONSTRAINTS)


def test_named_twin_dependent_constraints_require_twin_rev() -> None:
    for factory in TWIN_DEPENDENT_CONSTRAINT_FACTORIES:
        constraint = factory(9)
        assert constraint.kind == "twin_dependent"
        assert constraint.twin_rev == 9
    assert {f.__name__ for f in TWIN_DEPENDENT_CONSTRAINT_FACTORIES} == {
        "wall_impermeability",
        "portal_required",
        "stair_or_lift_for_floor_change",
        "visibility_and_accessibility",
    }


# ---------------------------------------------------------------------------
# Day 29 — the three hard predicates, filled
# ---------------------------------------------------------------------------


def test_max_pedestrian_velocity_holds_at_and_below_the_declared_bound() -> None:
    for speed in (0.0, 0.5, 1.4, PEDESTRIAN_MAX_SPEED_MPS):
        assert isinstance(
            evaluate_constraint(MAX_PEDESTRIAN_VELOCITY, speed), Satisfied
        ), f"{speed} m/s should be admissible"


def test_max_pedestrian_velocity_does_not_refute_a_jogger() -> None:
    """The mis-derivation guard. A threshold taken from the motion
    model's TYPICAL-scale constants (PERSON_SIGMA_A_MPS2 *
    PEDESTRIAN_STOP_DURATION_S = 1.5 m/s, a comfortable walking pace)
    would prune anyone who jogs — and, measured, violates on 46.0% of
    v5-cessation's GT frames. See PEDESTRIAN_MAX_SPEED_MPS's docstring
    and scripts/measure_gt_constraint_violations.py."""
    typical_scale_threshold = 1.5
    jogging_mps = 3.0
    assert jogging_mps > typical_scale_threshold
    assert isinstance(evaluate_constraint(MAX_PEDESTRIAN_VELOCITY, jogging_mps), Satisfied)


def test_max_pedestrian_velocity_violated_above_the_bound() -> None:
    outcome = evaluate_constraint(MAX_PEDESTRIAN_VELOCITY, PEDESTRIAN_MAX_SPEED_MPS + 0.1)
    assert isinstance(outcome, HardConstraintViolation)
    assert outcome.constraint_name == "max_pedestrian_velocity"


# ---------------------------------------------------------------------------
# Day 30 — max_pedestrian_acceleration, and the asymmetry it encodes
# ---------------------------------------------------------------------------


def test_acceleration_bound_is_asymmetric_and_braking_is_the_looser_side() -> None:
    """Humans stop faster than they start — different mechanisms (leg
    propulsion vs friction and eccentric load), different ceilings. The
    ordering is the physical claim; the exact values are declared bounds."""
    assert PEDESTRIAN_MAX_DECELERATION_MPS2 > PEDESTRIAN_MAX_ACCELERATION_MPS2


def test_acceleration_bound_admits_ordinary_gait_and_a_sprint_start() -> None:
    for magnitude in (0.0, 1.0, 1.5, 9.80665):
        assert isinstance(
            evaluate_constraint(MAX_PEDESTRIAN_ACCELERATION, magnitude, False),
            Satisfied,
        ), f"{magnitude} m/s^2 accelerating should be admissible"


def test_acceleration_bound_refutes_the_generator_defect_it_was_added_for() -> None:
    """v5-cessation's worst measured GT frame: 36.58 m/s^2, 3.7g. Refuted
    as braking (the direction it occurred in) AND as acceleration, so the
    verdict does not depend on getting `is_braking` right for this one."""
    v5_worst_mps2 = 36.58
    for is_braking in (True, False):
        outcome = evaluate_constraint(
            MAX_PEDESTRIAN_ACCELERATION, v5_worst_mps2, is_braking
        )
        assert isinstance(outcome, HardConstraintViolation)
        assert outcome.constraint_name == "max_pedestrian_acceleration"


def test_a_value_between_the_two_bounds_depends_on_direction() -> None:
    """The whole reason `is_braking` is a required argument: 15 m/s^2 is
    an impossible way to speed up and a survivable way to stop."""
    between = (PEDESTRIAN_MAX_ACCELERATION_MPS2 + PEDESTRIAN_MAX_DECELERATION_MPS2) / 2
    assert isinstance(
        evaluate_constraint(MAX_PEDESTRIAN_ACCELERATION, between, True), Satisfied
    )
    assert isinstance(
        evaluate_constraint(MAX_PEDESTRIAN_ACCELERATION, between, False),
        HardConstraintViolation,
    )


def test_acceleration_bound_is_not_the_process_noise_density() -> None:
    """Objective 5's rule, pinned as a test. PERSON_SIGMA_A_MPS2 (1.5) is
    a noise density with unbounded support; using it as a hard bound would
    refute ordinary gait initiation. The two must stay far apart."""
    assert PEDESTRIAN_MAX_ACCELERATION_MPS2 > PERSON_SIGMA_A_MPS2 * 4
    ordinary_gait_initiation_mps2 = 2.0
    assert ordinary_gait_initiation_mps2 > PERSON_SIGMA_A_MPS2
    assert isinstance(
        evaluate_constraint(
            MAX_PEDESTRIAN_ACCELERATION, ordinary_gait_initiation_mps2, False
        ),
        Satisfied,
    )


def test_acceleration_bound_rejects_a_negative_magnitude_as_malformed() -> None:
    with pytest.raises(ConstraintInputError, match="MAGNITUDE"):
        evaluate_constraint(MAX_PEDESTRIAN_ACCELERATION, -1.0, False)


def test_acceleration_bound_rejects_nan_rather_than_reporting_a_violation() -> None:
    """`not (nan <= x)` is True, so without the finiteness guard every
    NaN frame would silently PRUNE. Same hazard as every other predicate
    here."""
    with pytest.raises(ConstraintInputError):
        evaluate_constraint(MAX_PEDESTRIAN_ACCELERATION, float("nan"), True)


def test_jerk_bound_is_declared_and_admits_the_deceleration_bound() -> None:
    """A jerk bound below `max deceleration / 0.1 s` would make the
    deceleration bound unreachable, which would mean one of the two is
    wrong. They are declared independently, so this consistency is worth
    asserting rather than assuming."""
    assert PEDESTRIAN_MAX_JERK_MPS3 >= PEDESTRIAN_MAX_DECELERATION_MPS2 / 0.1


def test_mass_conservation_admits_a_reachable_gap_and_refutes_teleportation() -> None:
    # 1 m in 1 s: trivially reachable.
    assert isinstance(evaluate_constraint(MASS_CONSERVATION, 1.0, 1.0), Satisfied)
    # Exactly at the bound.
    assert isinstance(
        evaluate_constraint(MASS_CONSERVATION, PEDESTRIAN_MAX_SPEED_MPS, 1.0), Satisfied
    )
    # 100 m in 1 s.
    assert isinstance(
        evaluate_constraint(MASS_CONSERVATION, 100.0, 1.0), HardConstraintViolation
    )
    # Same instant, two places.
    assert isinstance(
        evaluate_constraint(MASS_CONSERVATION, 0.5, 0.0), HardConstraintViolation
    )
    assert isinstance(evaluate_constraint(MASS_CONSERVATION, 0.0, 0.0), Satisfied)


def test_gravity_bounds_downward_acceleration_but_not_upward() -> None:
    dt_s = 0.5
    free_fall_delta_v = -STANDARD_GRAVITY_MPS2 * dt_s
    # At free fall exactly, and slower: admissible.
    assert isinstance(
        evaluate_constraint(GRAVITY_FLOOR_TRANSITION, free_fall_delta_v, dt_s), Satisfied
    )
    assert isinstance(
        evaluate_constraint(GRAVITY_FLOOR_TRANSITION, -0.1, dt_s), Satisfied
    )
    # Faster than free fall: impossible.
    assert isinstance(
        evaluate_constraint(GRAVITY_FLOOR_TRANSITION, free_fall_delta_v - 0.1, dt_s),
        HardConstraintViolation,
    )


def test_gravity_leaves_upward_acceleration_unbounded() -> None:
    """A person pushing off the ground briefly exceeds g upward. A hard
    constraint may not refute that — being loose is the correct failure
    direction for a predicate on an irreversible pruning path."""
    dt_s = 0.1
    jump_delta_v = 3.0 * STANDARD_GRAVITY_MPS2 * dt_s
    assert isinstance(
        evaluate_constraint(GRAVITY_FLOOR_TRANSITION, jump_delta_v, dt_s), Satisfied
    )


@pytest.mark.parametrize(
    "constraint,args",
    [
        (MAX_PEDESTRIAN_VELOCITY, (float("nan"),)),
        (MAX_PEDESTRIAN_VELOCITY, (float("inf"),)),
        (MAX_PEDESTRIAN_VELOCITY, (-1.0,)),
        (MASS_CONSERVATION, (float("nan"), 1.0)),
        (MASS_CONSERVATION, (1.0, -1.0)),
        (GRAVITY_FLOOR_TRANSITION, (float("nan"), 1.0)),
        (GRAVITY_FLOOR_TRANSITION, (1.0, -1.0)),
        (ONE_BODY_ONE_PLACE, ((float("nan"), 0.0, 0.0), ORIGIN)),
        (ONE_BODY_ONE_PLACE, (ORIGIN, (0.0, float("inf"), 0.0))),
    ],
)
def test_malformed_input_raises_rather_than_returning_any_verdict(
    constraint: HardConstraint, args: tuple[float, ...]
) -> None:
    """A NaN must not become a violation by accident: ``not (nan <= x)``
    is True, so every threshold comparison would silently report a
    violation without the guard. Nor may it become an Unevaluable — that
    would file a caller bug under "missing world data" and lose it."""
    with pytest.raises(ConstraintInputError):
        evaluate_constraint(constraint, *args)


# ---------------------------------------------------------------------------
# Day 29 — the unevaluable path (the four twin-dependent predicates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", TWIN_DEPENDENT_CONSTRAINT_FACTORIES)
def test_every_twin_dependent_predicate_is_unevaluable_without_geometry(
    factory: object,
) -> None:
    constraint = factory(4)  # type: ignore[operator]
    outcome = evaluate_constraint(constraint, ORIGIN, ONE_METRE_EAST)
    assert isinstance(outcome, Unevaluable)
    assert outcome.reason == NO_TWIN_GEOMETRY
    assert outcome.constraint_name == constraint.name


@pytest.mark.parametrize("factory", TWIN_DEPENDENT_CONSTRAINT_FACTORIES)
def test_unevaluable_is_not_satisfied_and_not_a_violation(factory: object) -> None:
    """The whole structural point: three outcomes, not two-plus-a-guess."""
    outcome = evaluate_constraint(factory(4), ORIGIN, ONE_METRE_EAST)  # type: ignore[operator]
    assert not isinstance(outcome, Satisfied)
    assert not isinstance(outcome, TwinRevisionHypothesis)
    assert not isinstance(outcome, HardConstraintViolation)


def test_unevaluable_constraints_do_not_make_a_check_all_satisfied() -> None:
    """A check with zero violations has NOT cleared the hypothesis if
    anything went unevaluated. Under Day 28's ``violation | None`` return
    these two states were indistinguishable."""
    check = check_twin_dependent_constraints(
        [f(4) for f in TWIN_DEPENDENT_CONSTRAINT_FACTORIES], ORIGIN, ONE_METRE_EAST
    )
    assert check.hypotheses == ()
    assert len(check.unevaluable) == 4
    assert check.satisfied == ()
    assert not check.all_satisfied


def test_unevaluable_hard_constraint_neither_prunes_nor_counts_as_a_pass() -> None:
    store, hid = _store_with_one_hypothesis()
    unevaluable_constraint = HardConstraint(
        name="needs_something_absent",
        description="d",
        predicate=lambda *a, **k: Unevaluable(reason="no twin geometry"),
    )
    check = check_hard_constraints(
        store, hid, [unevaluable_constraint, ONE_BODY_ONE_PLACE], ORIGIN, ONE_METRE_EAST
    )
    assert check.violation is None
    assert not check.pruned
    assert not check.all_satisfied
    assert [u.constraint_name for u in check.unevaluable] == ["needs_something_absent"]
    assert [s.constraint_name for s in check.satisfied] == ["one_body_one_place"]
    # Not pruned: an unevaluable constraint refutes nothing.
    assert [h.id for h in store.alive()] == [hid]


def test_unevaluable_does_not_stop_evaluation_of_later_constraints() -> None:
    """Unlike a violation, an unevaluable outcome decides nothing, so
    there is nothing to short-circuit — the constraints after it must
    still run."""
    store, hid = _store_with_one_hypothesis()
    ran: list[str] = []

    def _record(name: str) -> bool:
        ran.append(name)
        return True

    check = check_hard_constraints(
        store,
        hid,
        [
            HardConstraint(
                name="a", description="d", predicate=lambda: Unevaluable(reason="none")
            ),
            HardConstraint(name="b", description="d", predicate=lambda: _record("b")),
        ],
    )
    assert ran == ["b"]
    assert len(check.unevaluable) == 1
    assert len(check.satisfied) == 1


def test_geometry_at_a_different_twin_rev_is_unevaluable_not_a_verdict() -> None:
    """Same rule src/model/world.py already enforces for a cross-revision
    position: never silently use one revision's data for another."""
    constraint = wall_impermeability(twin_rev=7)
    outcome = evaluate_constraint(
        constraint,
        ORIGIN,
        ONE_METRE_EAST,
        twin_geometry=FakeTwinGeometry(twin_rev=8, crosses_wall=True),
    )
    assert isinstance(outcome, Unevaluable)
    assert "twin_rev=8" in outcome.reason and "twin_rev=7" in outcome.reason


# ---------------------------------------------------------------------------
# Day 29 — the unevaluable path is not a dead end: same predicates, with
# geometry supplied, return real verdicts in both directions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,satisfying,violating",
    [
        (wall_impermeability, {"crosses_wall": False}, {"crosses_wall": True}),
        (
            portal_required,
            {"crosses_portal_boundary": False},
            {"crosses_portal_boundary": True},
        ),
        (
            stair_or_lift_for_floor_change,
            {"passes_stair_or_lift": True},
            {"passes_stair_or_lift": False},
        ),
        (visibility_and_accessibility, {"reachable": True}, {"reachable": False}),
    ],
)
def test_twin_dependent_predicates_return_verdicts_once_geometry_exists(
    factory: object, satisfying: dict[str, bool], violating: dict[str, bool]
) -> None:
    constraint = factory(2)  # type: ignore[operator]

    satisfied = evaluate_constraint(
        constraint,
        ORIGIN,
        ONE_METRE_EAST,
        twin_geometry=FakeTwinGeometry(twin_rev=2, **satisfying),
    )
    assert isinstance(satisfied, Satisfied)

    violated = evaluate_constraint(
        constraint,
        ORIGIN,
        ONE_METRE_EAST,
        twin_geometry=FakeTwinGeometry(twin_rev=2, **violating),
    )
    assert isinstance(violated, TwinRevisionHypothesis)
    assert violated.twin_rev == 2
    # Still not a refutation, even with real geometry behind it.
    assert not isinstance(violated, HardConstraintViolation)


def test_no_hard_constraint_predicate_is_unevaluable_today() -> None:
    """Every hard constraint runs. This is the Day-29 completion check:
    a hard constraint that cannot be evaluated would be a hard
    constraint in name only, since nothing could ever prune from it."""
    probes: dict[str, tuple[object, ...]] = {
        "one_body_one_place": (ORIGIN, ONE_METRE_EAST),
        "max_pedestrian_velocity": (1.0,),
        "mass_conservation": (1.0, 1.0),
        "gravity_floor_transition": (0.0, 1.0),
        "max_pedestrian_acceleration": (1.0, False),
    }
    for constraint in HARD_CONSTRAINTS:
        outcome = evaluate_constraint(constraint, *probes[constraint.name])
        assert not isinstance(outcome, Unevaluable), constraint.name


def test_replaced_predicate_still_routes_by_constraint_kind() -> None:
    """Guards the dispatch itself against the fixtures above: swapping a
    predicate cannot change which violation type a kind produces."""
    twin = replace(wall_impermeability(1), predicate=lambda: False)
    hard = replace(ONE_BODY_ONE_PLACE, predicate=lambda: False)
    assert isinstance(evaluate_constraint(twin), TwinRevisionHypothesis)
    assert isinstance(evaluate_constraint(hard), HardConstraintViolation)
