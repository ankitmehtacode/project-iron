"""Day 20, Objective 2 — the single-entity filter and solve_state.

Structural properties under test, each isolated so a regression in one
doesn't hide behind another passing:

- §17 the prior firewall: no parameter that could carry a behavioral prior.
- Reproducibility: re-solving at an earlier graph_rev after more factors
  have been appended reproduces the earlier result bit-identically.
- Occlusion: with no observations, predicted covariance grows monotonically
  and observed=False.
- StateQuery's horizon_ns and horizon_kind are both genuinely exercised.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.contracts.frames import AffineTransform, FrameGeometry
from src.estimator.filter import (
    FilterError,
    resolve_state,
    run_single_entity_filter,
)
from src.estimator.measurement_model import measurement_model_for
from src.estimator.motion_model import motion_model_for
from src.model.episode import EpisodeError, StateGraph, StateQuery, solve_state
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import CameraFrameMeasurement, WorldPositionMeasurement
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
    """A straight-line walker at step_m/dt_s m/s, along x."""
    return [
        _obs(
            x_m=step_m * i, y_m=0.0, z_m=0.0, ts_ns=BASE_TS + int(i * dt_s * SECOND_NS)
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# §17 — the prior firewall
# ---------------------------------------------------------------------------


def test_run_single_entity_filter_has_no_prior_parameter() -> None:
    sig = inspect.signature(run_single_entity_filter)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


def test_resolve_state_has_no_prior_parameter() -> None:
    sig = inspect.signature(resolve_state)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


def test_bootstrap_uses_only_the_first_observation() -> None:
    """The very first state comes from the first observation alone -- no
    external position/velocity guess enters anywhere."""
    graph = StateGraph()
    obs = _obs(x_m=7.0, y_m=-3.0, z_m=1.0, ts_ns=BASE_TS)
    run_single_entity_filter(
        graph,
        [obs],
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=graph.graph_rev)
    estimate = solve_state(query, graph)
    np.testing.assert_allclose(estimate.position_m(), [7.0, -3.0, 1.0])
    np.testing.assert_allclose(estimate.velocity_mps(), [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Basic end-to-end correctness
# ---------------------------------------------------------------------------


def test_filter_tracks_a_straight_line_walker() -> None:
    graph = StateGraph()
    observations = _walking_track(10)
    run_single_entity_filter(
        graph,
        observations,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    assert estimate.observed is True
    np.testing.assert_allclose(estimate.position_m(), [4.5, 0.0, 0.0], atol=0.5)
    np.testing.assert_allclose(estimate.velocity_mps(), [0.5, 0.0, 0.0], atol=0.5)


def test_every_update_carries_an_nis_residual() -> None:
    graph = StateGraph()
    observations = _walking_track(5)
    run_single_entity_filter(
        graph,
        observations,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    for factor in graph.factors_as_of(graph.graph_rev):
        payload = graph.payload_for(factor.factor_id)
        assert payload is not None
        nis_entries = [r for r in payload.estimate.residuals if r.kind == "nis"]
        assert len(nis_entries) == 1
    # Bootstrap has no innovation to score; every subsequent update does.
    payloads = [
        graph.payload_for(f.factor_id) for f in graph.factors_as_of(graph.graph_rev)
    ]
    bootstrap_nis = next(r for r in payloads[0].estimate.residuals if r.kind == "nis")
    assert bootstrap_nis.value is None
    for payload in payloads[1:]:
        nis = next(r for r in payload.estimate.residuals if r.kind == "nis")
        assert nis.value is not None
        assert nis.within_bound is not None


def test_empty_observations_raises() -> None:
    graph = StateGraph()
    with pytest.raises(FilterError):
        run_single_entity_filter(
            graph,
            [],
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_out_of_order_observations_raises() -> None:
    graph = StateGraph()
    observations = [
        _obs(0.0, 0.0, 0.0, BASE_TS + SECOND_NS),
        _obs(1.0, 0.0, 0.0, BASE_TS),
    ]
    with pytest.raises(FilterError, match="strictly increasing"):
        run_single_entity_filter(
            graph,
            observations,
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_non_position_measurement_raises() -> None:
    graph = StateGraph()
    bad = Observation(
        observation_id=generate_ulid(now_ns=BASE_TS),
        sensor_id="cam-1",
        ts_ns=BASE_TS,
        frame_ref="cam-1/frame",
        measurement=CameraFrameMeasurement(bbox_px=(0.0, 0.0, 10.0, 10.0)),
        uncertainty=Uncertainty.gaussian_px(1.0, 1.0),
        frame_of_reference=_frame_of_reference(),
        producer_shas=("test",),
        envelope_status="within_envelope",
    )
    with pytest.raises(FilterError, match="WorldPositionMeasurement"):
        run_single_entity_filter(
            graph,
            [bad],
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_non_positive_dt_raises() -> None:
    graph = StateGraph()
    observations = [_obs(0.0, 0.0, 0.0, BASE_TS), _obs(1.0, 0.0, 0.0, BASE_TS)]
    with pytest.raises(FilterError, match="non-positive dt"):
        run_single_entity_filter(
            graph,
            observations,
            motion_model_for("person"),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


# ---------------------------------------------------------------------------
# STRUCTURAL — reproducibility: re-solve at an earlier graph_rev
# ---------------------------------------------------------------------------


def test_resolving_at_an_earlier_graph_rev_is_bit_identical_after_more_appends() -> (
    None
):
    graph = StateGraph()
    first_batch = _walking_track(4)
    run_single_entity_filter(
        graph,
        first_batch,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    rev_after_first_batch = graph.graph_rev
    query = StateQuery(
        at_ts_ns=first_batch[-1].ts_ns, horizon_ns=0, graph_rev=rev_after_first_batch
    )
    before = solve_state(query, graph)

    # Append MORE factors -- a second, independent walker's observations,
    # continuing in time, onto the SAME graph.
    second_batch = [
        _obs(
            x_m=100.0 + i,
            y_m=0.0,
            z_m=0.0,
            ts_ns=first_batch[-1].ts_ns + (i + 1) * SECOND_NS,
        )
        for i in range(3)
    ]
    run_single_entity_filter(
        graph,
        second_batch,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    assert graph.graph_rev > rev_after_first_batch

    # Re-solve at the SAME (earlier) graph_rev and timestamp.
    after = solve_state(query, graph)

    assert before.mean == after.mean
    assert before.cov == after.cov
    assert before.graph_rev == after.graph_rev == rev_after_first_batch
    assert before.motion_model_sha == after.motion_model_sha
    assert before.measurement_model_sha == after.measurement_model_sha


# ---------------------------------------------------------------------------
# Occlusion: covariance grows monotonically, observed=False
# ---------------------------------------------------------------------------


def test_covariance_grows_monotonically_through_an_observation_gap() -> None:
    graph = StateGraph()
    observations = _walking_track(5)
    run_single_entity_filter(
        graph,
        observations,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    last_ts = observations[-1].ts_ns
    horizon_ns = 20 * SECOND_NS

    variances = []
    observed_flags = []
    for gap_s in (1, 2, 4, 8, 16):
        query = StateQuery(
            at_ts_ns=last_ts + gap_s * SECOND_NS,
            horizon_ns=horizon_ns,
            graph_rev=graph.graph_rev,
        )
        estimate = solve_state(query, graph)
        variances.append(estimate.cov[0][0])  # x-position variance
        observed_flags.append(estimate.observed)

    assert variances == sorted(variances)
    assert len(set(variances)) == len(variances), "variance must be STRICTLY increasing"
    assert all(flag is False for flag in observed_flags)


def test_extrapolation_beyond_horizon_raises() -> None:
    graph = StateGraph()
    observations = _walking_track(3)
    run_single_entity_filter(
        graph,
        observations,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns + 100 * SECOND_NS,
        horizon_ns=5 * SECOND_NS,
        graph_rev=graph.graph_rev,
    )
    with pytest.raises(EpisodeError, match="horizon_ns"):
        solve_state(query, graph)


def test_query_before_earliest_state_raises() -> None:
    graph = StateGraph()
    observations = _walking_track(3)
    run_single_entity_filter(
        graph,
        observations,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(
        at_ts_ns=observations[0].ts_ns - SECOND_NS,
        horizon_ns=0,
        graph_rev=graph.graph_rev,
    )
    with pytest.raises(EpisodeError, match="predates"):
        solve_state(query, graph)


def test_smoothed_horizon_kind_raises_not_implemented_via_filter() -> None:
    graph = StateGraph()
    observations = _walking_track(2)
    run_single_entity_filter(
        graph,
        observations,
        motion_model_for("person"),
        measurement_model_for("cam-1"),
        manifest_sha="m",
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns,
        horizon_ns=0,
        graph_rev=graph.graph_rev,
        horizon_kind="smoothed",
    )
    with pytest.raises(NotImplementedError):
        resolve_state(query, graph)
