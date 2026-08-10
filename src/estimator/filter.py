"""The single-entity filter: predict -> observe -> correct, over a StateGraph (§15).

STRUCTURAL — the prior firewall (§17)
---------------------------------------
:func:`run_single_entity_filter` takes no parameter that could carry a
behavioral prior. The first state is bootstrapped directly from the first
observation — position = that observation, velocity = 0, covariance =
``block_diag(R_first, LARGE_VELOCITY_VARIANCE * I)`` — never from an
injected assumption about where the entity "probably" is or how it
"probably" behaves. When priors arrive later (behavioural models, scene
priors, whatever form they take), they enter through a separate, recorded
path — not a parameter smuggled into this function's signature. See
``tests/test_estimator_filter.py::test_run_single_entity_filter_has_no_prior_parameter``,
which inspects the live signature rather than trusting this docstring.

Replay, not re-derivation
--------------------------
Every predict/update step appends one :class:`~src.model.episode.Factor` to
the graph, carrying (via the new ``payload`` slot) the numeric
:class:`~src.estimator.state.StateEstimate` it produced plus the motion and
measurement model instances used to produce it. :func:`resolve_state` (what
:func:`src.model.episode.solve_state` delegates to) never recomputes
anything the graph already has a payload for — it walks
``graph.factors_as_of(query.graph_rev)``, finds the latest payload-bearing
factor at or before ``query.at_ts_ns``, and either returns it directly (an
exact hit) or predicts forward from it (an occlusion/extrapolation gap).
Reproducibility (re-solving at an earlier ``graph_rev`` after more factors
have been appended) falls out of this for free: ``factors_as_of`` already
filters by revision, and payloads are never overwritten.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from src.estimator import consistency
from src.estimator.measurement_model import MeasurementModel
from src.estimator.motion_model import STATE_DIM, MotionModel, apply_velocity_covariance_floor
from src.estimator.state import StateEstimate
from src.model.episode import Factor, StateGraph, StateQuery
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.ulid import generate_ulid

FloatArray = npt.NDArray[np.float64]

LARGE_VELOCITY_VARIANCE_MPS2 = 100.0
"""(m/s)^2 initial velocity variance at bootstrap. Deliberately huge: the
first observation carries no velocity information at all, and this says so
numerically rather than guessing zero-with-confidence."""

UPDATE_RULE_VERSION = "kalman-joseph-form-v1"
"""Joseph-form covariance update (numerically stable, stays PD under
floating-point roundoff, unlike the textbook ``(I - KH) @ P``)."""
UPDATE_RULE_SHA = hashlib.sha256(UPDATE_RULE_VERSION.encode("utf-8")).hexdigest()


class FilterError(RuntimeError):
    """Raised when the filter or a state query cannot be resolved.

    Caught and re-raised as :class:`~src.model.episode.EpisodeError` at
    ``solve_state``'s boundary — this type is the estimator package's own
    vocabulary, not something a caller of ``src.model`` should need to
    import directly.
    """


@dataclass(frozen=True)
class _AppendedState:
    """What ``StateGraph``'s payload slot holds for this filter: the numeric
    result plus what produced it, enough to replay or extrapolate without
    recomputing anything from scratch."""

    estimate: StateEstimate
    motion_model: MotionModel
    measurement_model: MeasurementModel


def _distance_m(position_m: FloatArray, sensor_origin_m: FloatArray) -> float:
    return float(np.linalg.norm(position_m - sensor_origin_m))


def run_single_entity_filter(
    graph: StateGraph,
    observations: Sequence[Observation],
    motion_model: MotionModel,
    measurement_model: MeasurementModel,
    manifest_sha: str,
    sensor_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> StateGraph:
    """Run predict-then-update over ``observations``, appending one Factor
    per step to ``graph``. Mutates and returns ``graph``.

    Args:
        graph: The append-only store to write into. Not required to be
            empty — a caller may continue an existing single-entity graph,
            though nothing in this Day-20 scope exercises that.
        observations: Every ``measurement`` must be a
            :class:`~src.model.measurement.WorldPositionMeasurement`
            (already-unprojected 3D position) — this filter works in
            metric world space, not pixel space. Must be strictly
            increasing in ``ts_ns``; out-of-order input is refused rather
            than silently re-sorted, because silently re-sorting would
            hide a real bug (an ingest pipeline that reordered observations)
            behind a filter that just happens to still produce an answer.
        motion_model, measurement_model: The swappable components (§15).
        manifest_sha: The run that produced these observations/models.
        sensor_origin_m: World-frame sensor position, for computing the
            range :meth:`MeasurementModel.R` needs. Defaults to the world
            origin — a documented simplification for a single,
            already-unprojected position stream with no sensor-position
            registry in this codebase yet; a caller with a real camera
            position (e.g. from clip extrinsics) should pass it explicitly.

    Raises:
        FilterError: on empty input, out-of-order timestamps, a
            non-position measurement, or a non-positive ``dt`` between
            consecutive observations.
    """
    if not observations:
        raise FilterError("run_single_entity_filter needs at least one observation")

    timestamps = [obs.ts_ns for obs in observations]
    if timestamps != sorted(timestamps):
        raise FilterError(
            "observations must be strictly increasing in ts_ns; out-of-order "
            "input is refused rather than silently re-sorted"
        )

    origin = np.array(sensor_origin_m, dtype=np.float64)
    previous: _AppendedState | None = None
    previous_factor_id: str | None = None

    for observation in observations:
        if not isinstance(observation.measurement, WorldPositionMeasurement):
            raise FilterError(
                f"observation {observation.observation_id} carries a "
                f"{type(observation.measurement).__name__}, not "
                "WorldPositionMeasurement — the single-entity filter only "
                "consumes unprojected 3D position readings"
            )
        z = np.array(
            [
                observation.measurement.x_m,
                observation.measurement.y_m,
                observation.measurement.z_m,
            ],
            dtype=np.float64,
        )
        distance_m = _distance_m(z, origin)
        R = measurement_model.R(distance_m)

        inputs: tuple[str, ...]
        if previous is None:
            mean = np.concatenate([z, np.zeros(3, dtype=np.float64)])
            cov = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
            cov[:3, :3] = R
            cov[3:, 3:] = LARGE_VELOCITY_VARIANCE_MPS2 * np.eye(3, dtype=np.float64)
            nis = consistency.no_observation_nis_stub()
            factor_kind = "bootstrap"
            inputs = (str(observation.observation_id),)
        else:
            dt_s = (observation.ts_ns - previous.estimate.ts_ns) / 1e9
            if dt_s <= 0:
                raise FilterError(
                    f"non-positive dt ({dt_s}s) between observations at "
                    f"{previous.estimate.ts_ns} and {observation.ts_ns}"
                )
            F = motion_model.F(dt_s)
            Q = motion_model.Q(dt_s)
            prior_mean = previous.estimate.mean_array()
            prior_cov = previous.estimate.cov_array()
            predicted_mean = F @ prior_mean
            predicted_cov = F @ prior_cov @ F.T + Q

            H = measurement_model.H()
            innovation = z - H @ predicted_mean
            innovation_cov = H @ predicted_cov @ H.T + R
            kalman_gain = predicted_cov @ H.T @ np.linalg.inv(innovation_cov)
            mean = predicted_mean + kalman_gain @ innovation
            i_kh = np.eye(STATE_DIM, dtype=np.float64) - kalman_gain @ H
            # Joseph form: numerically stable, stays symmetric PD under
            # floating-point roundoff even when the textbook (I-KH)@P does not.
            cov = i_kh @ predicted_cov @ i_kh.T + kalman_gain @ R @ kalman_gain.T
            # Day 22, Objective 2: a no-op unless motion_model declares a
            # velocity floor (person/asset_carried with it enabled) -- see
            # src.estimator.motion_model's module docstring.
            cov = apply_velocity_covariance_floor(
                cov, motion_model.velocity_covariance_floor_mps2(dt_s)
            )
            nis = consistency.compute_nis(innovation, innovation_cov)
            factor_kind = "measurement_update"
            assert previous_factor_id is not None
            inputs = (previous_factor_id, str(observation.observation_id))

        estimate = StateEstimate(
            ts_ns=observation.ts_ns,
            mean=tuple(mean.tolist()),
            cov=tuple(tuple(row) for row in cov.tolist()),
            observed=True,
            motion_model_sha=motion_model.sha,
            measurement_model_sha=measurement_model.sha,
            update_rule_sha=UPDATE_RULE_SHA,
            graph_rev=graph.graph_rev + 1,
            residuals=(nis, *consistency.stub_residuals()),
        )
        payload = _AppendedState(
            estimate=estimate,
            motion_model=motion_model,
            measurement_model=measurement_model,
        )
        factor = graph.append_factor(
            str(generate_ulid()), factor_kind, inputs, manifest_sha, payload=payload
        )
        previous = payload
        previous_factor_id = factor.factor_id

    return graph


def _payload_factors(
    graph: StateGraph, graph_rev: int
) -> list[tuple[Factor, _AppendedState]]:
    out: list[tuple[Factor, _AppendedState]] = []
    for f in graph.factors_as_of(graph_rev):
        payload = graph.payload_for(f.factor_id)
        if isinstance(payload, _AppendedState):
            out.append((f, payload))
    return out


def _is_imm_graph(graph: StateGraph, graph_rev: int) -> bool:
    """Whether the payloads at or before ``graph_rev`` are IMM's, not
    single-model's — decides which resolver ``resolve_state`` delegates
    to. A graph is built by exactly one of
    :func:`run_single_entity_filter` / ``run_imm_filter`` in every case
    this codebase constructs today, so checking the first payload found is
    sufficient; a graph deliberately mixing the two would be a caller
    error this function does not need to detect.
    """
    from src.estimator.imm import _ImmAppendedState

    for f in graph.factors_as_of(graph_rev):
        payload = graph.payload_for(f.factor_id)
        if payload is not None:
            return isinstance(payload, _ImmAppendedState)
    return False


def resolve_state(query: StateQuery, graph: StateGraph) -> StateEstimate:
    """The "filtered" half of ``src.model.episode.solve_state`` (§15).

    Dispatches to :func:`src.estimator.imm.resolve_imm_state` when
    ``graph`` was populated by ``run_imm_filter`` rather than
    :func:`run_single_entity_filter` — see :func:`_is_imm_graph`.

    Raises:
        NotImplementedError: if ``query.horizon_kind == "smoothed"``.
        FilterError: if the graph has no resolvable state before
            ``query.at_ts_ns``, or resolving it needs more extrapolation
            than ``query.horizon_ns`` allows.
    """
    if query.horizon_kind == "smoothed":
        raise NotImplementedError(
            "smoothed-horizon resolution (a backward pass using factors "
            "after at_ts_ns) is out of Day 20's single-entity scope; only "
            "'filtered' (causal, forward-only) is implemented — see "
            "StateQuery.horizon_kind's docstring"
        )

    if _is_imm_graph(graph, query.graph_rev):
        from src.estimator.imm import ImmError, resolve_imm_state

        try:
            return resolve_imm_state(query, graph)
        except ImmError as exc:
            raise FilterError(str(exc)) from exc

    candidates = _payload_factors(graph, query.graph_rev)
    if not candidates:
        raise FilterError(
            f"no resolvable state in this graph at graph_rev={query.graph_rev} "
            "— the graph is empty, or carries no filter-produced factors "
            "at or before this revision"
        )

    at_or_before = [(f, p) for f, p in candidates if p.estimate.ts_ns <= query.at_ts_ns]
    if not at_or_before:
        earliest = candidates[0][1].estimate.ts_ns
        raise FilterError(
            f"query.at_ts_ns={query.at_ts_ns} predates this graph's earliest "
            f"resolvable state (ts_ns={earliest}); there is nothing to "
            "interpolate or extrapolate from"
        )

    _, latest = at_or_before[-1]
    dt_ns = query.at_ts_ns - latest.estimate.ts_ns
    if dt_ns == 0:
        return latest.estimate
    if dt_ns > query.horizon_ns:
        raise FilterError(
            f"resolving at_ts_ns={query.at_ts_ns} needs {dt_ns}ns of "
            f"extrapolation beyond the last resolvable factor "
            f"(ts_ns={latest.estimate.ts_ns}), exceeding "
            f"horizon_ns={query.horizon_ns}"
        )

    dt_s = dt_ns / 1e9
    F = latest.motion_model.F(dt_s)
    Q = latest.motion_model.Q(dt_s)
    predicted_mean = F @ latest.estimate.mean_array()
    predicted_cov = F @ latest.estimate.cov_array() @ F.T + Q
    predicted_cov = apply_velocity_covariance_floor(
        predicted_cov, latest.motion_model.velocity_covariance_floor_mps2(dt_s)
    )

    return StateEstimate(
        ts_ns=query.at_ts_ns,
        mean=tuple(predicted_mean.tolist()),
        cov=tuple(tuple(row) for row in predicted_cov.tolist()),
        observed=False,
        motion_model_sha=latest.motion_model.sha,
        measurement_model_sha=latest.estimate.measurement_model_sha,
        update_rule_sha=latest.estimate.update_rule_sha,
        graph_rev=query.graph_rev,
        residuals=(
            consistency.no_observation_nis_stub(),
            *consistency.stub_residuals(),
        ),
    )
