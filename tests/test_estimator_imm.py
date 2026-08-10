"""Day 21, Objective 3 — IMM.

Structural properties, mirroring tests/test_estimator_filter.py's coverage
for the single-model case: the prior firewall, reproducibility, and
occlusion. Plus IMM-specific behaviour: mode probabilities actually
respond to what the data shows, config validation, and dispatch through
the same solve_state entry point the single-model filter uses.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.contracts.frames import AffineTransform, FrameGeometry
from src.estimator.imm import (
    ImmConfig,
    ImmError,
    default_imm_config,
    resolve_imm_state,
    run_imm_filter,
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
        uncertainty=Uncertainty(kind="gaussian_3d", params=(("sigma_m", 0.03),)),
        frame_of_reference=_frame_of_reference(),
        producer_shas=("test",),
        envelope_status="within_envelope",
    )


def _walk_then_stop(
    n_walk: int, n_static: int, step_m: float = 0.5
) -> list[Observation]:
    observations = []
    pos = 0.0
    for i in range(n_walk):
        observations.append(_obs(pos, 0.0, 0.0, BASE_TS + i * SECOND_NS))
        pos += step_m
    for i in range(n_static):
        observations.append(_obs(pos, 0.0, 0.0, BASE_TS + (n_walk + i) * SECOND_NS))
    return observations


# ---------------------------------------------------------------------------
# §17 -- the prior firewall, a new code path, tested again
# ---------------------------------------------------------------------------


def test_run_imm_filter_has_no_prior_parameter() -> None:
    sig = inspect.signature(run_imm_filter)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


def test_resolve_imm_state_has_no_prior_parameter() -> None:
    sig = inspect.signature(resolve_imm_state)
    for name in sig.parameters:
        assert "prior" not in name.lower(), f"found a prior-shaped parameter: {name!r}"


def test_bootstrap_mode_probabilities_are_uniform() -> None:
    graph = StateGraph()
    config = default_imm_config()
    obs = _obs(1.0, 2.0, 0.0, BASE_TS)
    run_imm_filter(
        graph, [obs], config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(at_ts_ns=BASE_TS, horizon_ns=0, graph_rev=graph.graph_rev)
    estimate = solve_state(query, graph)
    assert estimate.mode_probabilities is not None
    probs = dict(estimate.mode_probabilities)
    assert set(probs) == set(config.mode_names)
    for p in probs.values():
        assert p == pytest.approx(1.0 / len(config.mode_names))


# ---------------------------------------------------------------------------
# ImmConfig validation
# ---------------------------------------------------------------------------


def test_default_imm_config_has_three_modes() -> None:
    config = default_imm_config()
    assert set(config.mode_names) == {"static", "constant_velocity", "maneuvering"}


def test_imm_config_rejects_too_few_modes() -> None:
    model = motion_model_for("person")
    with pytest.raises(ImmError):
        ImmConfig(modes=(("only", model),), transition_matrix=((1.0,),))


def test_imm_config_rejects_duplicate_mode_names() -> None:
    model = motion_model_for("person")
    with pytest.raises(ImmError):
        ImmConfig(
            modes=(("a", model), ("a", model)),
            transition_matrix=((0.5, 0.5), (0.5, 0.5)),
        )


def test_imm_config_rejects_wrong_shaped_matrix() -> None:
    model = motion_model_for("person")
    with pytest.raises(ImmError):
        ImmConfig(modes=(("a", model), ("b", model)), transition_matrix=((1.0,),))


def test_imm_config_rejects_row_not_summing_to_one() -> None:
    model = motion_model_for("person")
    with pytest.raises(ImmError):
        ImmConfig(
            modes=(("a", model), ("b", model)),
            transition_matrix=((0.9, 0.05), (0.5, 0.5)),
        )


def test_imm_config_sha_deterministic() -> None:
    a = default_imm_config()
    b = default_imm_config()
    assert a.sha == b.sha


def test_imm_config_sha_changes_with_persistence() -> None:
    a = default_imm_config(persistence_probability=0.95)
    b = default_imm_config(persistence_probability=0.80)
    assert a.sha != b.sha


def test_default_imm_config_rejects_bad_persistence() -> None:
    with pytest.raises(ImmError):
        default_imm_config(persistence_probability=1.5)


# ---------------------------------------------------------------------------
# Basic filtering behaviour and mode-probability response
# ---------------------------------------------------------------------------


def test_empty_observations_raises() -> None:
    with pytest.raises(ImmError):
        run_imm_filter(
            StateGraph(),
            [],
            default_imm_config(),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_out_of_order_observations_raises() -> None:
    observations = [_obs(1.0, 0, 0, BASE_TS + SECOND_NS), _obs(0.0, 0, 0, BASE_TS)]
    with pytest.raises(ImmError, match="strictly increasing"):
        run_imm_filter(
            StateGraph(),
            observations,
            default_imm_config(),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_non_position_measurement_raises() -> None:
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
    with pytest.raises(ImmError, match="WorldPositionMeasurement"):
        run_imm_filter(
            StateGraph(),
            [bad],
            default_imm_config(),
            measurement_model_for("cam-1"),
            manifest_sha="m",
        )


def test_static_track_favors_static_mode() -> None:
    graph = StateGraph()
    config = default_imm_config()
    rng = np.random.default_rng(1)
    observations = [
        _obs(
            float(rng.normal(0, 0.01)),
            float(rng.normal(0, 0.01)),
            0.0,
            BASE_TS + i * SECOND_NS,
        )
        for i in range(15)
    ]
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    probs = dict(estimate.mode_probabilities)  # type: ignore[arg-type]
    assert probs["static"] > probs["constant_velocity"]
    assert probs["static"] > probs["maneuvering"]


def test_steady_walk_favors_constant_velocity_mode() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=15, n_static=0, step_m=0.5)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    probs = dict(estimate.mode_probabilities)  # type: ignore[arg-type]
    assert probs["constant_velocity"] > probs["static"]


def test_stop_after_walk_shifts_probability_away_from_constant_velocity() -> None:
    """The Day-21 finding, directly: right after a stop, the mode mix
    should move away from constant_velocity (which was winning during the
    walk) toward static/maneuvering."""
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=10, n_static=8, step_m=0.5)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )

    query_during_walk = StateQuery(
        at_ts_ns=observations[8].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    query_after_stop = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    during_walk_probs = solve_state(query_during_walk, graph).mode_probabilities
    after_stop_probs = solve_state(query_after_stop, graph).mode_probabilities
    during_walk = dict(during_walk_probs)  # type: ignore[arg-type]
    after_stop = dict(after_stop_probs)  # type: ignore[arg-type]

    assert during_walk["constant_velocity"] > after_stop["constant_velocity"]


def test_estimate_carries_imm_config_sha_and_mode_probabilities() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=5, n_static=0)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    assert estimate.imm_config_sha == config.sha
    assert estimate.mode_probabilities is not None


def test_dominant_mode_matches_the_highest_probability() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=15, n_static=0, step_m=1.5)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    dominant = estimate.dominant_mode()
    assert dominant is not None
    name, prob = dominant
    probs = dict(estimate.mode_probabilities)  # type: ignore[arg-type]
    assert prob == max(probs.values())
    assert name == "constant_velocity"  # a clear, fast walk


def test_dominant_mode_is_none_for_single_model_estimate() -> None:
    from src.estimator.filter import run_single_entity_filter

    graph = StateGraph()
    observations = _walk_then_stop(n_walk=5, n_static=0)
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
    assert estimate.dominant_mode() is None


def test_residuals_include_combined_and_per_mode_nis() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=5, n_static=0)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    nis_entries = [r for r in estimate.residuals if r.kind == "nis"]
    # combined + one per mode
    assert len(nis_entries) == 1 + len(config.mode_names)
    per_mode_notes = {r.note for r in nis_entries if "mode=" in r.note}
    for name in config.mode_names:
        assert any(f"mode={name}" in note for note in per_mode_notes)


# ---------------------------------------------------------------------------
# Dispatch through solve_state, reproducibility, occlusion
# ---------------------------------------------------------------------------


def test_solve_state_dispatches_to_imm_for_an_imm_graph() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=5, n_static=0)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    estimate = solve_state(query, graph)
    assert estimate.mode_probabilities is not None  # only true of an IMM result


def test_imm_reproducibility_at_earlier_graph_rev() -> None:
    graph = StateGraph()
    config = default_imm_config()
    first_batch = _walk_then_stop(n_walk=6, n_static=0)
    run_imm_filter(
        graph, first_batch, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    rev_after_first = graph.graph_rev
    query = StateQuery(
        at_ts_ns=first_batch[-1].ts_ns, horizon_ns=0, graph_rev=rev_after_first
    )
    before = solve_state(query, graph)

    second_batch = [
        _obs(100.0 + i, 0.0, 0.0, first_batch[-1].ts_ns + (i + 1) * SECOND_NS)
        for i in range(3)
    ]
    run_imm_filter(
        graph, second_batch, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    assert graph.graph_rev > rev_after_first

    after = solve_state(query, graph)
    assert before.mean == after.mean
    assert before.cov == after.cov
    assert before.mode_probabilities == after.mode_probabilities


def test_imm_covariance_grows_through_occlusion_gap() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=8, n_static=0)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    last_ts = observations[-1].ts_ns
    horizon_ns = 20 * SECOND_NS

    variances = []
    for gap_s in (1, 2, 4, 8):
        query = StateQuery(
            at_ts_ns=last_ts + gap_s * SECOND_NS,
            horizon_ns=horizon_ns,
            graph_rev=graph.graph_rev,
        )
        estimate = solve_state(query, graph)
        variances.append(estimate.cov[0][0])
        assert estimate.observed is False
        assert estimate.mode_probabilities is not None

    assert variances == sorted(variances)
    assert len(set(variances)) == len(variances)


def test_imm_extrapolation_beyond_horizon_raises() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=4, n_static=0)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns + 100 * SECOND_NS,
        horizon_ns=5 * SECOND_NS,
        graph_rev=graph.graph_rev,
    )
    with pytest.raises(EpisodeError, match="horizon_ns"):
        solve_state(query, graph)


def test_imm_smoothed_horizon_raises_not_implemented() -> None:
    graph = StateGraph()
    config = default_imm_config()
    observations = _walk_then_stop(n_walk=3, n_static=0)
    run_imm_filter(
        graph, observations, config, measurement_model_for("cam-1"), manifest_sha="m"
    )
    query = StateQuery(
        at_ts_ns=observations[-1].ts_ns,
        horizon_ns=0,
        graph_rev=graph.graph_rev,
        horizon_kind="smoothed",
    )
    with pytest.raises(NotImplementedError):
        resolve_imm_state(query, graph)
