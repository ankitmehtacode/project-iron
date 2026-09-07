"""Day 26, Objective 3 — the multi-entity joint filter.

Structural properties under test, each isolated so a regression in one
doesn't hide behind another passing (mirrors tests/test_estimator_filter.py's
own organization for the single-entity precedent this extends):

- Size-1 components reproduce Day 20/25's single-entity filter bit-for-bit
  (mean, cov, and NEES dof), by delegation, not by a second implementation
  that happens to agree.
- The motivating case: a carrier's own motion moves a carried entity's
  posterior even with no further observation of the carried entity at all.
- NEES dof is correct at component sizes 1, 2, and 3 -- 6, 9, and 12
  respectively -- getting this wrong is exactly the silent-mis-report risk
  Day 26's objective named explicitly.
- The prior firewall (no behavioral-prior parameter) and graph_rev
  reproducibility, re-tested against the joint path specifically.
"""

from __future__ import annotations

import inspect
import math
from typing import Callable

import numpy as np
import pytest

from src.contracts.frames import AffineTransform, FrameGeometry
from src.estimator import consistency
from src.estimator.joint import (
    Ambiguous,
    AssociationBudgetConfig,
    AssociationCandidate,
    Component,
    ComponentCapConfig,
    ComponentCapRefusal,
    DECISIVE_LOG_BAYES_FACTOR,
    DEFAULT_COMPONENT_CAP_CONFIG,
    Decisive,
    DegradedComponentEstimate,
    HybridDiscreteContinuousState,
    JointFilterError,
    JointObservation,
    JointStateEstimate,
    compute_association_verdict,
    dominate_by_likelihood,
    resolve_component_membership,
    resolve_data_association,
    resolve_joint_state,
    run_joint_filter,
)
from src.estimator.measurement_model import measurement_model_for
from src.estimator.motion_model import motion_model_for
from src.model.constraint import HardConstraintViolation
from src.model.episode import StateGraph, StateQuery
from src.model.frame_of_reference import FrameOfReference
from src.model.hypothesis import (
    DominatedByLikelihood,
    Hypothesis,
    HypothesisStore,
    PrunedByBudget,
    RefutedByHardConstraint,
)
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.uncertainty import Uncertainty
from src.model.ulid import generate_ulid

SECOND_NS = 1_000_000_000
BASE_TS = 1_785_000_000 * SECOND_NS


def _frame_of_reference() -> FrameOfReference:
    return FrameOfReference(
        geometry=FrameGeometry(1920, 1080),
        to_canonical=AffineTransform.identity(),
        twin_rev=1,
    )


def _obs(
    x_m: float, y_m: float, z_m: float, ts_ns: int, sensor_id: str = "cam-1"
) -> Observation:
    return Observation(
        observation_id=generate_ulid(now_ns=ts_ns),
        sensor_id=sensor_id,
        ts_ns=ts_ns,
        frame_ref=f"{sensor_id}/frame-{ts_ns}",
        measurement=WorldPositionMeasurement(x_m=x_m, y_m=y_m, z_m=z_m),
        uncertainty=Uncertainty(kind="gaussian_3d", params=(("sigma_m", 0.05),)),
        frame_of_reference=_frame_of_reference(),
        producer_shas=("test",),
        envelope_status="within_envelope",
    )


def _walking_track(n: int, step_m: float = 0.5, dt_s: float = 1.0) -> list[Observation]:
    return [
        _obs(
            x_m=step_m * i, y_m=0.0, z_m=0.0, ts_ns=BASE_TS + int(i * dt_s * SECOND_NS)
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Component validation
# ---------------------------------------------------------------------------


def test_component_rejects_carrier_carrying_itself() -> None:
    with pytest.raises(JointFilterError):
        Component(carrier_entity_id="A", carried_entity_ids=("A",))


def test_component_rejects_duplicate_carried_ids() -> None:
    with pytest.raises(JointFilterError):
        Component(carrier_entity_id="A", carried_entity_ids=("laptop-7", "laptop-7"))


def test_component_size_and_state_dim() -> None:
    solo = Component(carrier_entity_id="A")
    assert solo.size == 1 and solo.state_dim == 6
    pair = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    assert pair.size == 2 and pair.state_dim == 9
    triple = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7", "bag-3"))
    assert triple.size == 3 and triple.state_dim == 12


# ---------------------------------------------------------------------------
# Prior firewall
# ---------------------------------------------------------------------------


def test_run_joint_filter_has_no_prior_parameter() -> None:
    sig = inspect.signature(run_joint_filter)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


def test_resolve_joint_state_has_no_prior_parameter() -> None:
    sig = inspect.signature(resolve_joint_state)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


# ---------------------------------------------------------------------------
# Size-1 reproduces Day 20/25's single-entity filter bit-for-bit
# ---------------------------------------------------------------------------


def test_size_one_component_reproduces_single_entity_filter_exactly() -> None:
    from src.estimator.filter import run_single_entity_filter
    from src.model.episode import solve_state

    observations = _walking_track(10)
    component = Component(carrier_entity_id="A")

    single_graph = StateGraph()
    run_single_entity_filter(
        single_graph,
        observations,
        motion_model_for("person", velocity_covariance_floor=True),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=single_graph.graph_rev
    )
    single_estimate = solve_state(query, single_graph)

    joint_graph = StateGraph()
    run_joint_filter(
        joint_graph,
        component,
        [JointObservation(entity_id="A", observation=o) for o in observations],
        motion_model_for("person", velocity_covariance_floor=True),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    joint_query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=joint_graph.graph_rev
    )
    joint_estimate = resolve_joint_state(joint_query, joint_graph)

    assert joint_estimate.mean == single_estimate.mean
    assert joint_estimate.cov == single_estimate.cov

    gt = np.array([4.5, 0.0, 0.0, 0.5, 0.0, 0.0])
    single_nees = consistency.compute_nees(
        single_estimate.mean_array() - gt, single_estimate.cov_array()
    )
    joint_nees = consistency.compute_nees(
        joint_estimate.mean_array() - gt, joint_estimate.cov_array()
    )
    assert single_nees.value == joint_nees.value
    assert single_nees.dof == joint_nees.dof == 6


# ---------------------------------------------------------------------------
# The motivating case
# ---------------------------------------------------------------------------


def test_carried_entity_posterior_moves_with_carrier_without_further_observation() -> (
    None
):
    """If A carries laptop-7 and A moves, laptop-7's posterior must move --
    even across steps with NO new observation of laptop-7 at all."""
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    bootstrap_offset = np.array([0.3, 0.0, 0.0])

    carrier_obs = _walking_track(6, step_m=0.5)
    carrier_start_x = carrier_obs[0].measurement.x_m  # type: ignore[union-attr]
    carried_bootstrap = _obs(
        x_m=carrier_start_x + bootstrap_offset[0],
        y_m=0.0,
        z_m=0.0,
        ts_ns=BASE_TS,
    )
    joint_obs = [JointObservation("A", o) for o in carrier_obs]
    joint_obs.append(JointObservation("laptop-7", carried_bootstrap))

    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        joint_obs,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    first_query = StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=graph.graph_rev)
    first_estimate = resolve_joint_state(first_query, graph)
    assert isinstance(first_estimate, JointStateEstimate)
    carried_start = first_estimate.carried_position_m("laptop-7")

    last_ts = carrier_obs[-1].ts_ns
    last_query = StateQuery(at_ts_ns=last_ts, horizon_ns=0, graph_rev=graph.graph_rev)
    last_estimate = resolve_joint_state(last_query, graph)
    assert isinstance(last_estimate, JointStateEstimate)
    carried_end = last_estimate.carried_position_m("laptop-7")
    carrier_end = last_estimate.carrier_position_m()
    carrier_start = first_estimate.carrier_position_m()

    # The carried entity moved by (approximately) the same displacement as
    # the carrier -- the rigid-coupling identity -- despite receiving no
    # observation of its own after bootstrap.
    np.testing.assert_allclose(
        carried_end - carried_start, carrier_end - carrier_start, atol=0.3
    )


# ---------------------------------------------------------------------------
# NEES dof correctness at component sizes 1, 2, 3
# ---------------------------------------------------------------------------


def _bootstrap_component(carried_ids: tuple[str, ...]) -> tuple[StateGraph, Component]:
    component = Component(carrier_entity_id="A", carried_entity_ids=carried_ids)
    carrier_obs = _walking_track(6, step_m=0.5)
    joint_obs = [JointObservation("A", o) for o in carrier_obs]
    for i, cid in enumerate(carried_ids):
        joint_obs.append(
            JointObservation(
                cid, _obs(x_m=0.1 * (i + 1), y_m=0.0, z_m=0.0, ts_ns=BASE_TS)
            )
        )
    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        joint_obs,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    return graph, component


@pytest.mark.parametrize(
    "carried_ids,expected_dof",
    [((), 6), (("laptop-7",), 9), (("laptop-7", "bag-3"), 12)],
)
def test_nees_dof_matches_component_size(
    carried_ids: tuple[str, ...], expected_dof: int
) -> None:
    graph, component = _bootstrap_component(carried_ids)
    query = StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=graph.graph_rev)
    estimate = resolve_joint_state(query, graph)
    gt = np.zeros(component.state_dim)
    nees = consistency.compute_nees(estimate.mean_array() - gt, estimate.cov_array())
    assert nees.dof == expected_dof


# ---------------------------------------------------------------------------
# Bootstrap correctness and error handling
# ---------------------------------------------------------------------------


def test_bootstrap_offset_is_carried_minus_carrier() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    carrier_obs = _obs(x_m=1.0, y_m=2.0, z_m=0.0, ts_ns=BASE_TS)
    carried_obs = _obs(x_m=1.3, y_m=2.0, z_m=0.0, ts_ns=BASE_TS)
    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        [JointObservation("A", carrier_obs), JointObservation("laptop-7", carried_obs)],
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=graph.graph_rev)
    estimate = resolve_joint_state(query, graph)
    assert isinstance(estimate, JointStateEstimate)
    np.testing.assert_allclose(estimate.carried_offset_m("laptop-7"), [0.3, 0.0, 0.0])
    np.testing.assert_allclose(estimate.carried_position_m("laptop-7"), [1.3, 2.0, 0.0])


def test_bootstrap_missing_a_declared_member_raises() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    graph = StateGraph()
    with pytest.raises(JointFilterError, match="missing observations"):
        run_joint_filter(
            graph,
            component,
            [JointObservation("A", _obs(0.0, 0.0, 0.0, BASE_TS))],
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_observation_for_unrelated_entity_raises() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    graph = StateGraph()
    with pytest.raises(JointFilterError, match="not part of this component"):
        run_joint_filter(
            graph,
            component,
            [
                JointObservation("A", _obs(0.0, 0.0, 0.0, BASE_TS)),
                JointObservation("laptop-7", _obs(0.1, 0.0, 0.0, BASE_TS)),
                JointObservation("intruder", _obs(5.0, 5.0, 0.0, BASE_TS + SECOND_NS)),
            ],
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_empty_observations_raises() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    graph = StateGraph()
    with pytest.raises(JointFilterError):
        run_joint_filter(
            graph,
            component,
            [],
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


# ---------------------------------------------------------------------------
# Velocity floor applies to the carrier block in a joint solve
# ---------------------------------------------------------------------------


def test_velocity_floor_binds_on_the_carrier_block_of_a_joint_solve() -> None:
    from src.estimator.motion_model import pedestrian_velocity_covariance_floor_mps2

    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    carrier_obs = _walking_track(10, step_m=0.5, dt_s=1.0 / 12.0)
    carried_obs = _obs(
        x_m=carrier_obs[0].measurement.x_m + 0.3,  # type: ignore[union-attr]
        y_m=0.0,
        z_m=0.0,
        ts_ns=BASE_TS,
    )
    joint_obs = [JointObservation("A", o) for o in carrier_obs]
    joint_obs.append(JointObservation("laptop-7", carried_obs))

    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        joint_obs,
        motion_model_for("person", velocity_covariance_floor=True),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(
        at_ts_ns=carrier_obs[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = resolve_joint_state(query, graph)
    floor = pedestrian_velocity_covariance_floor_mps2(1.0 / 12.0)
    cov = estimate.cov_array()
    for axis in range(3):
        assert cov[3 + axis, 3 + axis] >= floor - 1e-9


# ---------------------------------------------------------------------------
# STRUCTURAL — reproducibility: re-solve a joint component at an earlier rev
# ---------------------------------------------------------------------------


def test_resolving_a_joint_component_at_an_earlier_rev_is_bit_identical() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    first_batch = _walking_track(4)
    carried_bootstrap = _obs(
        x_m=first_batch[0].measurement.x_m + 0.3,  # type: ignore[union-attr]
        y_m=0.0,
        z_m=0.0,
        ts_ns=BASE_TS,
    )
    joint_obs = [JointObservation("A", o) for o in first_batch]
    joint_obs.append(JointObservation("laptop-7", carried_bootstrap))

    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        joint_obs,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    rev_after_first_batch = graph.graph_rev
    query = StateQuery(
        at_ts_ns=first_batch[-1].ts_ns, horizon_ns=0, graph_rev=rev_after_first_batch
    )
    before = resolve_joint_state(query, graph)

    # Append factors for a SECOND, unrelated component onto the SAME graph
    # (mirrors test_estimator_filter.py's own reproducibility test, which
    # appends an unrelated second walker) -- this must not perturb the
    # first component's already-resolved earlier state.
    other_component = Component(carrier_entity_id="B", carried_entity_ids=("bag-3",))
    other_obs = [
        JointObservation(
            "B",
            _obs(
                x_m=100.0 + i,
                y_m=0.0,
                z_m=0.0,
                ts_ns=first_batch[-1].ts_ns + (i + 1) * SECOND_NS,
            ),
        )
        for i in range(3)
    ]
    other_obs.append(
        JointObservation(
            "bag-3",
            _obs(x_m=100.3, y_m=0.0, z_m=0.0, ts_ns=first_batch[-1].ts_ns + SECOND_NS),
        )
    )
    run_joint_filter(
        graph,
        other_component,
        other_obs,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    assert graph.graph_rev > rev_after_first_batch

    after = resolve_joint_state(query, graph)

    assert before.mean == after.mean
    assert before.cov == after.cov
    assert before.graph_rev == after.graph_rev == rev_after_first_batch


# ---------------------------------------------------------------------------
# Skeletons: NotImplementedError, not a silent gap
# ---------------------------------------------------------------------------


def test_hybrid_discrete_continuous_state_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        HybridDiscreteContinuousState()


def test_resolve_joint_state_smoothed_horizon_raises_not_implemented() -> None:
    component = Component(carrier_entity_id="A")
    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        [JointObservation("A", _obs(0.0, 0.0, 0.0, BASE_TS))],
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(
        at_ts_ns=BASE_TS,
        horizon_ns=0,
        graph_rev=graph.graph_rev,
        horizon_kind="smoothed",
    )
    with pytest.raises(NotImplementedError):
        resolve_joint_state(query, graph)


# ---------------------------------------------------------------------------
# Day 27, Objective 1 -- acceleration-scaled slip model
# ---------------------------------------------------------------------------


def test_constant_slip_model_is_the_default_and_unchanged_from_day_26() -> None:
    import inspect

    sig = inspect.signature(run_joint_filter)
    assert sig.parameters["offset_slip_model"].default == "constant"


def test_effective_offset_slip_sigma_constant_model_ignores_acceleration() -> None:
    from src.estimator.joint import _effective_offset_slip_sigma

    base = 0.05
    assert _effective_offset_slip_sigma(base, "constant", None) == base
    assert _effective_offset_slip_sigma(base, "constant", 10.0) == base


def test_effective_offset_slip_sigma_acceleration_scaled_at_reference_equals_base() -> (
    None
):
    from src.estimator.joint import _effective_offset_slip_sigma
    from src.estimator.motion_model import PERSON_SIGMA_A_MPS2

    base = 0.05
    at_reference = _effective_offset_slip_sigma(
        base, "acceleration_scaled", PERSON_SIGMA_A_MPS2
    )
    assert at_reference == pytest.approx(base)


def test_effective_offset_slip_sigma_acceleration_scaled_shrinks_when_steady() -> None:
    from src.estimator.joint import _effective_offset_slip_sigma

    steady = _effective_offset_slip_sigma(0.05, "acceleration_scaled", 0.01)
    assert steady < 0.05


def test_effective_offset_slip_sigma_grows_during_a_sharp_stop() -> None:
    from src.estimator.joint import _effective_offset_slip_sigma
    from src.estimator.motion_model import PERSON_SIGMA_A_MPS2

    sharp_stop = _effective_offset_slip_sigma(
        0.05, "acceleration_scaled", 4.0 * PERSON_SIGMA_A_MPS2
    )
    assert sharp_stop == pytest.approx(0.05 * 4.0)


def test_acceleration_scaled_model_runs_end_to_end_and_stays_positive_definite() -> (
    None
):
    """A stop-then-restart carrier track (real acceleration, not just
    steady walking) under the acceleration_scaled model must still
    produce a valid, positive-definite joint posterior at every step."""
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    walk = _walking_track(6, step_m=0.5, dt_s=1.0 / 12.0)
    stop = [
        _obs(
            x_m=walk[-1].measurement.x_m,  # type: ignore[union-attr]
            y_m=0.0,
            z_m=0.0,
            ts_ns=walk[-1].ts_ns + (i + 1) * int(SECOND_NS / 12),
        )
        for i in range(6)
    ]
    carrier_obs = walk + stop
    carried_bootstrap = _obs(
        x_m=carrier_obs[0].measurement.x_m + 0.3,  # type: ignore[union-attr]
        y_m=0.0,
        z_m=0.0,
        ts_ns=BASE_TS,
    )
    joint_obs = [JointObservation("A", o) for o in carrier_obs]
    joint_obs.append(JointObservation("laptop-7", carried_bootstrap))

    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        joint_obs,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
        offset_slip_model="acceleration_scaled",
    )
    query = StateQuery(
        at_ts_ns=carrier_obs[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = resolve_joint_state(query, graph)
    cov = estimate.cov_array()
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(
        eigenvalues > 0
    ), f"joint covariance not PD: eigenvalues={eigenvalues}"


# ---------------------------------------------------------------------------
# Day 27, Objective 2 -- component-size cap and the degradation path
# ---------------------------------------------------------------------------


def test_component_cap_config_rejects_nonpositive_size() -> None:
    with pytest.raises(JointFilterError):
        ComponentCapConfig(
            max_component_size=0, degradation_action="independent_fallback"
        )


def test_default_component_cap_is_six_with_independent_fallback() -> None:
    assert DEFAULT_COMPONENT_CAP_CONFIG.max_component_size == 6
    assert DEFAULT_COMPONENT_CAP_CONFIG.degradation_action == "independent_fallback"


def _synchronized_component_observations(
    carrier_id: str, carried_ids: tuple[str, ...], n_frames: int = 5
) -> list[JointObservation]:
    obs: list[JointObservation] = []
    for t in range(n_frames):
        ts = BASE_TS + t * SECOND_NS
        obs.append(JointObservation(carrier_id, _obs(0.5 * t, 0.0, 0.0, ts)))
        for i, cid in enumerate(carried_ids):
            obs.append(
                JointObservation(cid, _obs(0.5 * t + 0.1 * (i + 1), 0.0, 0.0, ts))
            )
    return obs


def test_component_within_cap_is_not_degraded() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    small_cap = ComponentCapConfig(
        max_component_size=6, degradation_action="independent_fallback"
    )
    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        _synchronized_component_observations("A", ("laptop-7",)),
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
        cap_config=small_cap,
    )
    query = StateQuery(
        at_ts_ns=BASE_TS + 4 * SECOND_NS, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = resolve_joint_state(query, graph)
    assert isinstance(estimate, JointStateEstimate)


def test_component_exceeding_cap_degrades_and_records_provenance() -> None:
    """STRUCTURAL: the overflow path, tested explicitly. A component
    larger than the cap must never silently solve jointly -- it must
    return a DegradedComponentEstimate with the degradation recorded."""
    carried_ids = ("laptop-7", "bag-3", "phone-1")
    component = Component(carrier_entity_id="A", carried_entity_ids=carried_ids)
    assert component.size == 4
    tight_cap = ComponentCapConfig(
        max_component_size=2, degradation_action="independent_fallback"
    )
    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        _synchronized_component_observations("A", carried_ids),
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
        cap_config=tight_cap,
    )
    query = StateQuery(
        at_ts_ns=BASE_TS + 4 * SECOND_NS, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = resolve_joint_state(query, graph)
    assert isinstance(estimate, DegradedComponentEstimate)
    assert estimate.degradation_action == "independent_fallback"
    assert estimate.cap_config_sha == tight_cap.sha
    for entity_id in component.entity_ids:
        per_entity = estimate.estimate_for(entity_id)
        assert per_entity.observed is True


def test_degraded_component_estimate_requires_all_fields() -> None:
    """A DegradedComponentEstimate cannot be constructed without a
    degradation_action or a cap_config_sha -- no default exists for
    either, so "a capped solve with no recorded degradation" is a
    TypeError, not a runtime possibility."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(DegradedComponentEstimate)}
    assert "degradation_action" in fields
    assert "cap_config_sha" in fields
    field_defaults = {
        f.name: f.default is dataclasses.MISSING
        for f in dataclasses.fields(DegradedComponentEstimate)
    }
    assert field_defaults["degradation_action"] is True  # no default -> required
    assert field_defaults["cap_config_sha"] is True


def test_degraded_component_estimate_rejects_incomplete_entity_coverage() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    with pytest.raises(JointFilterError):
        DegradedComponentEstimate(
            ts_ns=BASE_TS,
            component=component,
            degradation_action="independent_fallback",
            cap_config_sha="deadbeef",
            entity_estimates=(),  # missing both entities
        )


def test_degraded_fallback_requires_full_synchronization() -> None:
    carried_ids = ("laptop-7", "bag-3", "phone-1")
    component = Component(carrier_entity_id="A", carried_entity_ids=carried_ids)
    tight_cap = ComponentCapConfig(
        max_component_size=2, degradation_action="independent_fallback"
    )
    obs = _synchronized_component_observations("A", carried_ids)
    # Drop one carried entity's observation at the last timestamp.
    obs = [
        jo
        for jo in obs
        if not (
            jo.entity_id == "phone-1"
            and jo.observation.ts_ns == BASE_TS + 4 * SECOND_NS
        )
    ]
    graph = StateGraph()
    with pytest.raises(JointFilterError, match="missing"):
        run_joint_filter(
            graph,
            component,
            obs,
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
            cap_config=tight_cap,
        )


def test_degraded_resolution_rejects_extrapolation() -> None:
    carried_ids = ("laptop-7", "bag-3", "phone-1")
    component = Component(carrier_entity_id="A", carried_entity_ids=carried_ids)
    tight_cap = ComponentCapConfig(
        max_component_size=2, degradation_action="independent_fallback"
    )
    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        _synchronized_component_observations("A", carried_ids),
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
        cap_config=tight_cap,
    )
    query = StateQuery(
        at_ts_ns=BASE_TS + 4 * SECOND_NS + SECOND_NS // 2,
        horizon_ns=int(SECOND_NS),
        graph_rev=graph.graph_rev,
    )
    with pytest.raises(JointFilterError, match="extrapolat"):
        resolve_joint_state(query, graph)


def test_degraded_graph_rev_reproducibility() -> None:
    """Same STRUCTURAL guarantee as the joint path, re-tested for the
    degraded path specifically."""
    carried_ids = ("laptop-7", "bag-3", "phone-1")
    component = Component(carrier_entity_id="A", carried_entity_ids=carried_ids)
    tight_cap = ComponentCapConfig(
        max_component_size=2, degradation_action="independent_fallback"
    )
    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        _synchronized_component_observations("A", carried_ids, n_frames=4),
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
        cap_config=tight_cap,
    )
    rev_after_first = graph.graph_rev
    query = StateQuery(
        at_ts_ns=BASE_TS + 3 * SECOND_NS, horizon_ns=0, graph_rev=rev_after_first
    )
    before = resolve_joint_state(query, graph)

    other_component = Component(carrier_entity_id="B")
    run_joint_filter(
        graph,
        other_component,
        [JointObservation("B", _obs(100.0, 0.0, 0.0, BASE_TS + 4 * SECOND_NS))],
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
        cap_config=tight_cap,
    )
    assert graph.graph_rev > rev_after_first

    after = resolve_joint_state(query, graph)
    assert isinstance(before, DegradedComponentEstimate)
    assert isinstance(after, DegradedComponentEstimate)
    assert before.entity_estimates == after.entity_estimates
    assert before.ts_ns == after.ts_ns == BASE_TS + 3 * SECOND_NS


# ---------------------------------------------------------------------------
# Day 28, Objective 1 -- the two-term slip model
# ---------------------------------------------------------------------------


def test_offset_slip_model_literal_includes_two_term() -> None:
    from src.estimator.joint import OffsetSlipModel

    assert "two_term" in OffsetSlipModel.__args__  # type: ignore[attr-defined]


def test_two_term_variance_is_floor_alone_with_no_acceleration_estimate() -> None:
    from src.estimator.joint import _offset_slip_variance_mps2

    base = 0.05
    variance = _offset_slip_variance_mps2(base, "two_term", None)
    assert variance == pytest.approx(base**2)


def test_two_term_variance_at_reference_acceleration_is_double_the_floor() -> None:
    """At |a_hat| == PERSON_SIGMA_A_MPS2, the acceleration term equals the
    floor term exactly (Day 27's own "at the nominal bound, effective_sigma
    equals the baseline" finding) -- so the combined VARIANCE (not sigma)
    is exactly twice the floor variance, with no separate ratio chosen."""
    from src.estimator.joint import _offset_slip_variance_mps2
    from src.estimator.motion_model import PERSON_SIGMA_A_MPS2

    base = 0.05
    variance = _offset_slip_variance_mps2(base, "two_term", PERSON_SIGMA_A_MPS2)
    assert variance == pytest.approx(2 * base**2)


def test_two_term_variance_reduces_toward_the_floor_when_steady() -> None:
    from src.estimator.joint import _offset_slip_variance_mps2

    base = 0.05
    steady = _offset_slip_variance_mps2(base, "two_term", 0.01)
    assert steady == pytest.approx(base**2, rel=0.05)


def test_two_term_variance_exceeds_the_floor_during_a_sharp_stop() -> None:
    from src.estimator.joint import _offset_slip_variance_mps2
    from src.estimator.motion_model import PERSON_SIGMA_A_MPS2

    base = 0.05
    sharp_stop = _offset_slip_variance_mps2(base, "two_term", 4.0 * PERSON_SIGMA_A_MPS2)
    floor_alone = base**2
    assert sharp_stop > floor_alone
    # Never LESS than either single-term model would give alone at the
    # same acceleration -- the whole point of adding rather than choosing.
    from src.estimator.joint import _effective_offset_slip_sigma

    accel_only = (
        _effective_offset_slip_sigma(
            base, "acceleration_scaled", 4.0 * PERSON_SIGMA_A_MPS2
        )
        ** 2
    )
    assert sharp_stop > floor_alone
    assert sharp_stop > accel_only


def test_two_term_model_never_produces_less_variance_than_the_floor() -> None:
    """The two-term model's whole physical point: unlike acceleration_scaled
    alone, it never collapses below the constant model's own floor,
    regardless of estimated acceleration."""
    from src.estimator.joint import _offset_slip_variance_mps2

    base = 0.05
    floor_alone = base**2
    for accel in (0.0, 0.001, 0.5, 1.5, 3.0, 10.0):
        assert _offset_slip_variance_mps2(base, "two_term", accel) >= floor_alone


def test_two_term_model_runs_end_to_end_and_stays_positive_definite() -> None:
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    walk = _walking_track(6, step_m=0.5, dt_s=1.0 / 12.0)
    stop = [
        _obs(
            x_m=walk[-1].measurement.x_m,  # type: ignore[union-attr]
            y_m=0.0,
            z_m=0.0,
            ts_ns=walk[-1].ts_ns + (i + 1) * int(SECOND_NS / 12),
        )
        for i in range(6)
    ]
    carrier_obs = walk + stop
    carried_bootstrap = _obs(
        x_m=carrier_obs[0].measurement.x_m + 0.3,  # type: ignore[union-attr]
        y_m=0.0,
        z_m=0.0,
        ts_ns=BASE_TS,
    )
    joint_obs = [JointObservation("A", o) for o in carrier_obs]
    joint_obs.append(JointObservation("laptop-7", carried_bootstrap))

    graph = StateGraph()
    run_joint_filter(
        graph,
        component,
        joint_obs,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
        offset_slip_model="two_term",
    )
    query = StateQuery(
        at_ts_ns=carrier_obs[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = resolve_joint_state(query, graph)
    cov = estimate.cov_array()
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(
        eigenvalues > 0
    ), f"joint covariance not PD: eigenvalues={eigenvalues}"


def test_two_term_model_actually_differs_from_constant_during_real_deceleration() -> (
    None
):
    """Regression guard for a real Day 28 bug: the causal acceleration
    estimate was only ever computed when ``offset_slip_model ==
    "acceleration_scaled"`` -- "two_term" silently ran with
    ``carrier_acceleration_mps2=None`` at every step and collapsed to
    bit-identical output with "constant" for an entire evaluation run
    before this was caught. A track with a real, abrupt deceleration
    must produce a LARGER carrier-offset covariance under "two_term"
    than under "constant" at the step right after the stop -- if it does
    not, the acceleration term is not being fed at all."""
    component = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    walk = _walking_track(6, step_m=0.5, dt_s=1.0 / 12.0)
    stop = [
        _obs(
            x_m=walk[-1].measurement.x_m,  # type: ignore[union-attr]
            y_m=0.0,
            z_m=0.0,
            ts_ns=walk[-1].ts_ns + (i + 1) * int(SECOND_NS / 12),
        )
        for i in range(4)
    ]
    carrier_obs = walk + stop
    carried_bootstrap = _obs(
        x_m=carrier_obs[0].measurement.x_m + 0.3,  # type: ignore[union-attr]
        y_m=0.0,
        z_m=0.0,
        ts_ns=BASE_TS,
    )
    joint_obs = [JointObservation("A", o) for o in carrier_obs]
    joint_obs.append(JointObservation("laptop-7", carried_bootstrap))

    def _final_offset_variance(offset_slip_model: str) -> float:
        graph = StateGraph()
        run_joint_filter(
            graph,
            component,
            joint_obs,
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
            offset_slip_model=offset_slip_model,  # type: ignore[arg-type]
        )
        query = StateQuery(
            at_ts_ns=carrier_obs[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
        )
        estimate = resolve_joint_state(query, graph)
        assert isinstance(estimate, JointStateEstimate)
        offset_slice = component.offset_slice("laptop-7")
        return float(estimate.cov_array()[offset_slice, offset_slice].trace())

    constant_variance = _final_offset_variance("constant")
    two_term_variance = _final_offset_variance("two_term")
    assert two_term_variance > constant_variance, (
        f"two_term ({two_term_variance}) did not exceed constant "
        f"({constant_variance}) after a real stop -- the acceleration "
        "term is not being fed"
    )


# ---------------------------------------------------------------------------
# Day 32, Objective 4 -- hypothesis management, started
# ---------------------------------------------------------------------------


def _never_violates(candidate: AssociationCandidate) -> HardConstraintViolation | None:
    return None


def _violate_named(
    name: str,
) -> "Callable[[AssociationCandidate], HardConstraintViolation | None]":
    def check(candidate: AssociationCandidate) -> HardConstraintViolation | None:
        if candidate.hypothesis_id == name:
            return HardConstraintViolation(
                constraint_name="test_constraint", description="fixture violation"
            )
        return None

    return check


def test_resolve_data_association_rejects_empty_candidates() -> None:
    store = HypothesisStore()
    with pytest.raises(JointFilterError):
        resolve_data_association(store, "comp-1", [], _never_violates)


def test_resolve_data_association_keeps_the_best_within_budget() -> None:
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("a", "carried-7 -> carrier A", log_likelihood=-1.0),
        AssociationCandidate("b", "carried-7 -> carrier B", log_likelihood=-5.0),
        AssociationCandidate("c", "carried-7 -> carrier C", log_likelihood=-0.5),
    ]
    resolution = resolve_data_association(
        store,
        "comp-1",
        candidates,
        _never_violates,
        AssociationBudgetConfig(per_component_budget=2),
    )
    surviving_ids = {h.id for h in resolution.surviving}
    assert surviving_ids == {"comp-1::a", "comp-1::c"}, (
        "the two highest log_likelihood candidates (a, c) should survive a "
        "budget of 2; b (the worst) should not"
    )
    decision_by_id = {d.hypothesis_id: d for d in resolution.decisions}
    assert decision_by_id["b"].kept is False
    # b's margin from the weakest survivor (a, -1.0) is -1.0 - (-5.0) = 4.0
    # nats, well above DECISIVE_LOG_BAYES_FACTOR (~1.0986) -- Day 34,
    # Objective 3: a margin this large means the evidence itself excludes
    # b, so the cause is DominatedByLikelihood, not PrunedByBudget.
    assert decision_by_id["b"].cause == DominatedByLikelihood(
        dominant_hypothesis_id="comp-1::a"
    )
    assert decision_by_id["a"].kept and decision_by_id["a"].cause is None
    assert decision_by_id["c"].kept and decision_by_id["c"].cause is None


def test_resolve_data_association_decision_log_covers_every_candidate_in_order() -> (
    None
):
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("x", "p1", log_likelihood=-2.0),
        AssociationCandidate("y", "p2", log_likelihood=-1.0),
    ]
    resolution = resolve_data_association(
        store, "comp-2", candidates, _never_violates, AssociationBudgetConfig(1)
    )
    assert [d.hypothesis_id for d in resolution.decisions] == [
        "x",
        "y",
    ], "the decision log must be in the CANDIDATES' order, not rank order"


def test_hard_constraint_violation_prunes_before_budget_ranking() -> None:
    """The load-bearing ordering claim: a candidate that would have WON
    the budget ranking on log_likelihood alone must still die with
    RefutedByHardConstraint, not survive, if it violates a hard
    constraint -- hard constraints are checked first and unconditionally,
    regardless of how good the candidate looks on likelihood."""
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("best-but-impossible", "p", log_likelihood=100.0),
        AssociationCandidate("worst-but-possible", "p", log_likelihood=-100.0),
    ]
    resolution = resolve_data_association(
        store,
        "comp-3",
        candidates,
        _violate_named("best-but-impossible"),
        AssociationBudgetConfig(per_component_budget=2),
    )
    surviving_ids = {h.id for h in resolution.surviving}
    assert surviving_ids == {"comp-3::worst-but-possible"}
    decision_by_id = {d.hypothesis_id: d for d in resolution.decisions}
    assert decision_by_id["best-but-impossible"].cause == RefutedByHardConstraint(
        constraint_name="test_constraint"
    )


def test_budget_pruned_hypotheses_are_forensically_distinguishable_from_refuted() -> (
    None
):
    """PRUNED_BY_BUDGET is the load-bearing case (Day 28's module
    docstring, re-tested here against a real caller): a forensic query
    over what was considered must be able to tell "ruled out" apart from
    "never evaluated," for both a hard-refuted and a budget-pruned
    candidate produced by the SAME resolve_data_association call.
    "budget-cut" is a genuine near-tie with "kept" (margin 0.5 nats,
    below DECISIVE_LOG_BAYES_FACTOR ~1.0986) so it is excluded by the
    budget rather than by decisive evidence -- Day 34, Objective 3's
    PRUNED_BY_BUDGET-vs-DOMINATED_BY_LIKELIHOOD split."""
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("refuted", "p", log_likelihood=50.0),
        AssociationCandidate("budget-cut", "p", log_likelihood=9.5),
        AssociationCandidate("kept", "p", log_likelihood=10.0),
    ]
    resolve_data_association(
        store,
        "comp-4",
        candidates,
        _violate_named("refuted"),
        AssociationBudgetConfig(per_component_budget=1),
    )
    alternatives = {d.id: d for d in store.considered_alternatives()}
    assert isinstance(alternatives["comp-4::refuted"].cause, RefutedByHardConstraint)
    assert isinstance(alternatives["comp-4::budget-cut"].cause, PrunedByBudget)
    assert alternatives["comp-4::refuted"].proposition == "p"
    assert alternatives["comp-4::budget-cut"].support_at_death == 9.5
    assert "comp-4::kept" not in alternatives, "the survivor must not appear as dead"


def test_resolve_data_association_has_no_prior_parameter() -> None:
    sig = inspect.signature(resolve_data_association)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


def test_association_budget_config_rejects_a_non_positive_budget() -> None:
    with pytest.raises(JointFilterError):
        AssociationBudgetConfig(per_component_budget=0)


def test_association_budget_config_sha_changes_with_the_budget() -> None:
    assert AssociationBudgetConfig(2).sha != AssociationBudgetConfig(3).sha
    assert AssociationBudgetConfig(2).sha == AssociationBudgetConfig(2).sha


# ---------------------------------------------------------------------------
# Day 33, Objective 2 -- stress-testing Day 32's late-session code, fresh
# ---------------------------------------------------------------------------


def test_resolve_data_association_is_deterministic_given_identical_inputs() -> None:
    """The reproducibility property actually available today: this
    function takes no StateGraph/graph_rev at all (see the module
    docstring's "Implemented today" section) -- association decisions are
    not yet part of the graph, so a graph_rev replay test does not apply
    until Objective 4's punch-list item ("wire resolve_data_association's
    output into run_joint_filter's Component selection") lands. What DOES
    apply today: this function must be a pure function of its arguments,
    with no hidden state that could make two calls with identical inputs
    diverge. Two independent HypothesisStore instances, same candidates,
    same budget, same hard_violation_check -> bit-identical resolutions."""
    candidates = [
        AssociationCandidate("a", "carried-7 -> carrier A", log_likelihood=-1.0),
        AssociationCandidate("b", "carried-7 -> carrier B", log_likelihood=-5.0),
        AssociationCandidate("c", "carried-7 -> carrier C", log_likelihood=-0.5),
    ]
    budget = AssociationBudgetConfig(per_component_budget=2)

    store_1 = HypothesisStore()
    resolution_1 = resolve_data_association(
        store_1, "comp-x", candidates, _never_violates, budget
    )
    store_2 = HypothesisStore()
    resolution_2 = resolve_data_association(
        store_2, "comp-x", candidates, _never_violates, budget
    )

    assert {h.id for h in resolution_1.surviving} == {
        h.id for h in resolution_2.surviving
    }
    assert type(resolution_1.verdict) is type(resolution_2.verdict)
    assert resolution_1.verdict.margin == resolution_2.verdict.margin
    assert [h.id for h in resolution_1.verdict.competitors] == [
        h.id for h in resolution_2.verdict.competitors
    ], "verdict variant, margin, and competitor ranking (Day 34) must match"
    assert resolution_1.decisions == resolution_2.decisions, (
        "identical inputs against independent stores must produce "
        "bit-identical decision logs"
    )


def test_near_tie_hypotheses_produce_an_ambiguous_verdict_not_a_silent_winner() -> None:
    """FIXED Day 34, Objective 2 (was a known_bug/xfail Day-33 finding --
    marker and xfail removed in this same change, per this project's own
    known_bug convention). Two candidates within 0.001 nats of each other
    -- the resolution still designates a `surviving` hypothesis (budget=1
    forces a pick for pruning purposes), but `resolution.verdict` now
    reports Ambiguous, not Decisive: a caller checking the verdict, not
    just `.surviving`, cannot mistake this near-tie for a landslide."""
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("near-tie-winner", "p", log_likelihood=-1.000),
        AssociationCandidate("near-tie-loser", "p", log_likelihood=-1.001),
    ]
    resolution = resolve_data_association(
        store, "comp-tie", candidates, _never_violates, AssociationBudgetConfig(1)
    )
    assert isinstance(resolution.verdict, Ambiguous)
    assert resolution.verdict.margin == pytest.approx(0.001)
    assert {h.id for h in resolution.verdict.competitors} == {
        "comp-tie::near-tie-winner",
        "comp-tie::near-tie-loser",
    }


# ---------------------------------------------------------------------------
# Day 33, Objective 3 -- component partitioning meets hypothesis management
# ---------------------------------------------------------------------------


def test_resolve_component_membership_accepts_within_cap() -> None:
    existing = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    result = resolve_component_membership(
        existing,
        "bag-3",
        ComponentCapConfig(
            max_component_size=6, degradation_action="independent_fallback"
        ),
    )
    assert isinstance(result, Component)
    assert result.carried_entity_ids == ("laptop-7", "bag-3")


def test_resolve_component_membership_is_a_noop_for_an_existing_member() -> None:
    existing = Component(carrier_entity_id="A", carried_entity_ids=("laptop-7",))
    result = resolve_component_membership(
        existing, "laptop-7", DEFAULT_COMPONENT_CAP_CONFIG
    )
    assert result is existing


def test_resolve_component_membership_refuses_past_the_cap() -> None:
    tight_cap = ComponentCapConfig(
        max_component_size=2, degradation_action="independent_fallback"
    )
    existing = Component(carrier_entity_id="A", carried_entity_ids=("item-1", "item-2"))
    result = resolve_component_membership(existing, "item-3", tight_cap)
    assert isinstance(result, ComponentCapRefusal)
    assert (
        result.existing_component == existing
    ), "refused growth must not mutate the existing component"
    assert result.refused_carried_entity_id == "item-3"
    assert result.degradation_action == "independent_fallback"
    assert result.cap_config_sha == tight_cap.sha


def test_resolve_component_membership_at_exactly_the_cap_is_accepted() -> None:
    """Off-by-one boundary: a component landing EXACTLY at
    max_component_size (which counts the carrier itself: 1 + len(carried))
    is accepted, not refused -- the cap is <=, not <."""
    at_cap = ComponentCapConfig(
        max_component_size=3, degradation_action="independent_fallback"
    )
    existing = Component(carrier_entity_id="A", carried_entity_ids=("item-1",))
    result = resolve_component_membership(existing, "item-2", at_cap)
    assert isinstance(result, Component)
    assert result.size == 3

    one_over = resolve_component_membership(result, "item-3", at_cap)
    assert isinstance(one_over, ComponentCapRefusal), (
        "the very next entity, pushing size to 4 against a cap of 3, must "
        "be refused -- confirms the boundary is tight in both directions"
    )


def test_resolve_component_membership_return_type_is_exhaustive() -> None:
    """STRUCTURAL: every path through resolve_component_membership returns
    either Component or ComponentCapRefusal -- inspected directly on the
    return annotation rather than trusted from reading the source, same
    discipline as the prior-firewall signature checks."""
    import typing

    hints = typing.get_type_hints(resolve_component_membership)
    ret = hints["return"]
    args = typing.get_args(ret)
    assert set(args) == {Component, ComponentCapRefusal}, (
        f"resolve_component_membership's return type is {ret!r}, expected "
        "Component | ComponentCapRefusal exactly"
    )


# ---------------------------------------------------------------------------
# Day 34, Objective 2 -- Decisive vs Ambiguous, and its derived threshold
# ---------------------------------------------------------------------------


def _ranked(*supports: float) -> list[Hypothesis]:
    return [
        Hypothesis(id=f"h{i}", proposition="p", support=s)
        for i, s in enumerate(supports)
    ]


def test_compute_association_verdict_exact_tie_is_ambiguous() -> None:
    verdict = compute_association_verdict(_ranked(1.0, 1.0))
    assert isinstance(verdict, Ambiguous)
    assert verdict.margin == 0.0


def test_compute_association_verdict_near_tie_is_ambiguous() -> None:
    just_under = DECISIVE_LOG_BAYES_FACTOR - 0.001
    verdict = compute_association_verdict(_ranked(just_under, 0.0))
    assert isinstance(verdict, Ambiguous)
    assert verdict.margin == pytest.approx(just_under)


def test_compute_association_verdict_landslide_is_decisive() -> None:
    verdict = compute_association_verdict(_ranked(50.0, 0.0, -10.0))
    assert isinstance(verdict, Decisive)
    assert verdict.winner.id == "h0"
    assert verdict.margin == 50.0
    assert {c.id for c in verdict.competitors} == {"h1", "h2"}


def test_compute_association_verdict_exactly_at_threshold_is_decisive() -> None:
    """The threshold is a lower bound, not a strict one: margin ==
    threshold is decisive, matching the >= in the implementation and in
    Kass & Raftery's own band boundaries (each band is closed on its
    lower edge)."""
    verdict = compute_association_verdict(_ranked(DECISIVE_LOG_BAYES_FACTOR, 0.0))
    assert isinstance(verdict, Decisive)


def test_compute_association_verdict_single_candidate_is_infinitely_decisive() -> None:
    verdict = compute_association_verdict(_ranked(3.0))
    assert isinstance(verdict, Decisive)
    assert verdict.margin == math.inf
    assert verdict.competitors == ()


def test_compute_association_verdict_rejects_empty_input() -> None:
    with pytest.raises(JointFilterError):
        compute_association_verdict([])


def test_compute_association_verdict_has_no_prior_parameter() -> None:
    sig = inspect.signature(compute_association_verdict)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


def test_ambiguous_has_no_winner_attribute() -> None:
    """STRUCTURAL: `if verdict.winner is not None` is unwritable as a way
    to skip handling Ambiguous -- there is no `winner` attribute on it at
    all, an AttributeError, not a None to silently pass through."""
    verdict = compute_association_verdict(_ranked(1.0, 1.0))
    assert isinstance(verdict, Ambiguous)
    assert not hasattr(verdict, "winner")


def test_association_verdict_match_is_exhaustive_over_both_variants() -> None:
    """A caller MUST handle both variants -- exercised directly via
    `match`, the same closed-world-dispatch discipline used elsewhere in
    this codebase (assert_outcome_never, _admit_as_evidence)."""
    from src.estimator.joint import assert_verdict_never

    for verdict in (
        compute_association_verdict(_ranked(50.0, 0.0)),
        compute_association_verdict(_ranked(1.0, 1.0)),
    ):
        match verdict:
            case Decisive():
                handled = "decisive"
            case Ambiguous():
                handled = "ambiguous"
            case _:
                assert_verdict_never(verdict)
        assert handled in ("decisive", "ambiguous")


# ---------------------------------------------------------------------------
# Day 34, Objective 3 -- dominate_by_likelihood, and the death-cause fix
# ---------------------------------------------------------------------------


def test_dominate_by_likelihood_refuses_a_sub_threshold_margin() -> None:
    store = HypothesisStore()
    store.propose("winner", "p", 1.0)
    store.propose("loser", "p", 0.9)
    with pytest.raises(JointFilterError):
        dominate_by_likelihood(store, "winner", "loser", margin=0.1)
    # Refused, not silently downgraded -- the hypothesis is still alive.
    assert {h.id for h in store.alive()} == {"winner", "loser"}


def test_dominate_by_likelihood_accepts_a_supra_threshold_margin() -> None:
    store = HypothesisStore()
    store.propose("winner", "p", 10.0)
    store.propose("loser", "p", 0.0)
    dead = dominate_by_likelihood(store, "winner", "loser", margin=10.0)
    assert isinstance(dead.cause, DominatedByLikelihood)
    assert dead.cause.dominant_hypothesis_id == "winner"


def test_forced_prune_below_threshold_is_pruned_by_budget() -> None:
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("kept", "p", log_likelihood=1.0),
        AssociationCandidate(
            "cut", "p", log_likelihood=0.9
        ),  # margin 0.1, sub-threshold
    ]
    resolution = resolve_data_association(
        store, "comp-margin", candidates, _never_violates, AssociationBudgetConfig(1)
    )
    decision = {d.hypothesis_id: d for d in resolution.decisions}["cut"]
    assert decision.cause == PrunedByBudget(budget=1)


def test_forced_prune_above_threshold_is_dominated_by_likelihood() -> None:
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("kept", "p", log_likelihood=10.0),
        AssociationCandidate(
            "cut", "p", log_likelihood=0.0
        ),  # margin 10.0, supra-threshold
    ]
    resolution = resolve_data_association(
        store, "comp-margin2", candidates, _never_violates, AssociationBudgetConfig(1)
    )
    decision = {d.hypothesis_id: d for d in resolution.decisions}["cut"]
    assert decision.cause == DominatedByLikelihood(
        dominant_hypothesis_id="comp-margin2::kept"
    )


def test_three_way_forensic_distinction_refuted_dominated_budget_pruned() -> None:
    """A forensic query distinguishes all three outcomes without
    collapsing any pair -- refuted (hard constraint), dominated (decisive
    margin), and budget-pruned-while-ambiguous (sub-threshold margin),
    all from one resolve_data_association call with budget=1 among three
    hard-constraint survivors."""
    store = HypothesisStore()
    candidates = [
        AssociationCandidate("refuted", "p", log_likelihood=999.0),
        AssociationCandidate("winner", "p", log_likelihood=10.0),
        AssociationCandidate(
            "near-tie", "p", log_likelihood=9.9
        ),  # margin 0.1 from winner
        AssociationCandidate(
            "far-behind", "p", log_likelihood=-50.0
        ),  # margin 60 from winner
    ]
    resolve_data_association(
        store,
        "comp-three",
        candidates,
        _violate_named("refuted"),
        AssociationBudgetConfig(per_component_budget=1),
    )
    alternatives = {d.id: d for d in store.considered_alternatives()}
    assert isinstance(
        alternatives["comp-three::refuted"].cause, RefutedByHardConstraint
    )
    assert isinstance(
        alternatives["comp-three::near-tie"].cause, PrunedByBudget
    ), "near-tie's margin from the winner (0.1) is sub-threshold"
    assert isinstance(
        alternatives["comp-three::far-behind"].cause, DominatedByLikelihood
    ), "far-behind's margin from the winner (60) is decisively supra-threshold"
    causes = {type(a.cause) for a in alternatives.values()}
    assert len(causes) == 3, "all three cause types must be distinct, none collapsed"


def test_component_cap_and_death_cause_do_not_yet_interact() -> None:
    """Day 34, Objective 3 asked to verify the death-cause fix interacts
    correctly with the Day-26 component cap. Verified: it does not
    interact AT ALL yet, and that is the correct, honest answer, not a
    gap silently left unchecked. resolve_component_membership takes an
    already-decided carried_entity_id (a plain string) and never touches
    a HypothesisStore, AssociationCandidate, or death cause of any kind
    -- confirmed by inspecting its signature directly. The two mechanisms
    cannot tell two different stories about the same decision because
    they do not share any state or decision point today; that changes
    only once Day-33's punch-list item (structured carrier identity
    wiring resolve_data_association's output into component membership)
    lands, at which point this test's assumption should be revisited."""
    sig = inspect.signature(resolve_component_membership)
    param_types = {name: p.annotation for name, p in sig.parameters.items()}
    assert "HypothesisStore" not in str(param_types.values())
    assert "AssociationCandidate" not in str(param_types.values())
