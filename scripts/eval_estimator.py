"""Score the single-entity state estimator against exact synthetic GT.

Day 20/21's hard scope rule: **no timing, throughput, CPU-percentage, or
latency claim is made anywhere in this script.** Everything below is
accuracy and consistency — position/velocity error against v3-indoor's and
v4.1-gate's analytic ``agent_xyz`` ground truth, the NIS/NEES consistency
residuals src/estimator's filter already computes, and (Day 21) GT motion
regimes and innovation whiteness. If a benchmark needs to run at all, it
goes through Day 19's environment gate (``src.bench.environment``) and is
expected to refuse on this machine — see ``docs/reference_hardware.md``.

    python scripts/eval_estimator.py
    python scripts/eval_estimator.py --version v3-indoor --version v4.1-gate

Method
------
For each agent's track in a clip, this script does NOT feed the estimator
real detections (no detector exists yet) — it synthesizes observations by
adding measurement noise to the exact GT position, with sigma drawn from
the SAME :class:`~src.estimator.measurement_model.MeasurementModel` the
filter itself uses (an illustrative, explicitly-unmeasured envelope — see
that module's docstring; there is no real camera yet, Day 19). This tests
the estimator's own machinery — motion model, measurement model, Kalman
update, consistency residuals — in isolation from detection/tracking
error, which is a separate, not-yet-built pipeline stage.

All three methods compared (the filter, and both trivial baselines) are
evaluated from the same query index onward per track (frame 2 of a
0-indexed track) so none gets "free" information the others lack: by
frame 2 the filter has incorporated 3 observations (bootstrap + 2
updates), ``constant_velocity_no_update`` has a genuine one-step
prediction (its first two observations only ever *define* its velocity,
never serve as a "prediction" of themselves), and ``copy_previous_position``
needs only one prior observation. Comparing them from frame 1 would let
``constant_velocity_no_update`` score using observation 1 as its own
"prediction" of observation 1 — a walkover, not a baseline.

Day 21 adds three things. Per scored frame: a GT motion regime (classified
from exact GT alone, never from the filter's own estimate — see
:mod:`src.estimator.regime`) and, for the single-model filter, a
standardized innovation (for the whiteness diagnostic,
:mod:`src.estimator.diagnostics`), reconstructed by replaying the same
predict step the filter itself took, using only the filter's public
``F``/``Q``/``H``/``R`` interface. And per golden set: the same evaluation
re-run with :mod:`src.estimator.imm` in place of the single-model filter
(``--filter both``, the default), so the two can be compared per regime —
innovation whiteness is not recomputed for IMM (it has no single innovation
sequence — it has one per mode — and was not the requested comparison;
IMM's per-mode NIS is already attached to every estimate it produces).

Exit codes:
    0  every requested set was scored (or cleanly empty)
    1  the Day-10 validity gate refused, or a requested set could not be
       loaded/scored
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.config import IronConfig
from src.contracts.frames import AffineTransform, FrameGeometry
from src.contracts.ground_truth import GENERATOR_AXES, gt_position_track
from src.data import validity
from src.data.depth_eval import DISTANCE_BUCKETS
from src.data.golden import GoldenSetError, available_versions, load_golden_set
from src.eval.baselines import compute_baselines, margin, require_baseline
from src.estimator.consistency import (
    chi2_upper_bound,
    compute_collapsed_gaussian_nees_diagnostic,
    compute_mixture_nees,
    compute_nees,
    empirical_coverage_by_sampling,
    fraction_outside_bound,
)
from src.estimator.diagnostics import is_white
from src.estimator.filter import run_single_entity_filter
from src.estimator.informativeness import (
    STATE_DOF,
    UNINFORMATIVE_MARGIN_NATS,
    CalibrationAndInformativeness,
    ConstantCovarianceBaseline,
    fit_constant_covariance,
    constant_baseline_coverage,
    gaussian_log_density,
    mixture_log_density,
)
from src.estimator.imm import ImmConfig, default_imm_config, run_imm_filter
from src.estimator.measurement_model import MeasurementModel, measurement_model_for
from src.estimator.motion_model import (
    MotionModel,
    motion_model_for,
    pedestrian_velocity_covariance_floor_mps2,
)
from src.estimator.regime import MOTION_REGIMES, MotionRegime, classify_track
from src.estimator.state import ConsistencyResidual
from src.model.episode import StateGraph, StateQuery, solve_state
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.uncertainty import Uncertainty
from src.model.ulid import generate_ulid
from src.model.world import UNREGISTERED

FilterKind = Literal["single_model", "imm"]

CONFIG_SPECS: dict[str, tuple[FilterKind, bool]] = {
    "A": ("single_model", False),
    "B": ("single_model", True),
    "C": ("imm", False),
    "D": ("imm", True),
}
"""Day 22, Objective 3's four configs: (filter_kind, velocity_covariance_floor)."""

CONFIG_DESCRIPTIONS: dict[str, str] = {
    "A": "single model (Day 20, current)",
    "B": "single model + velocity covariance floor (Day 22)",
    "C": "IMM (Day 21)",
    "D": "IMM + velocity covariance floor (Day 22)",
}

FloatArray = npt.NDArray[np.float64]

EVAL_SEED = 20260731
"""Pinned seed for synthesized observation noise -- same convention as
scripts/cascade_bench.py's --seed default."""

EVAL_SEED_MIXTURE_SAMPLING = 20260810
"""Separate pinned seed for empirical_coverage_by_sampling's Monte-Carlo
draws (Day 22, Objective 1) -- deliberately NOT the same stream as
EVAL_SEED. Both single-model and IMM runs must see IDENTICAL noisy
observations (see EVAL_SEED's own consumer), so the observation-noise RNG
must never be perturbed by how many extra draws the mixture-sampling
coverage check happens to consume; a shared stream would make the two
filters' comparison depend on evaluation order, not on the filter."""

BASE_TS_NS = 1_785_000_000 * 1_000_000_000
FIRST_COMPARABLE_INDEX = 2
"""See module docstring: the first frame index at which all three compared
methods (filter, copy_previous, constant_velocity_no_update) have a
genuine, non-circular answer."""

MIN_REGIME_FRAMES_FOR_A_CONCLUSION = 10
"""Below this many pooled frames, a regime's numbers are reported (never
hidden) but flagged as too thin to support a conclusion -- matches
src.estimator.diagnostics.is_white's own floor for the same reason."""


def _frame_of_reference(twin_rev: int = UNREGISTERED) -> FrameOfReference:
    """Day 31, Objective 4: UNREGISTERED, not 1.

    These are synthetic clips that record no twin revision at all, and
    `src/data/scorecard.py` and `src/inspector/artifacts.py` already wrap
    the very same clips as UNREGISTERED. Defaulting to 1 here asserted a
    revision that does not exist, and asserted it more confidently than
    the two consumers that had it right -- the same false-declaration
    pattern Day 30 found in WorldPositionArray, in a different type.
    """
    return FrameOfReference(
        geometry=FrameGeometry(1, 1),
        to_canonical=AffineTransform.identity(),
        twin_rev=twin_rev,
    )


def _camera_position_world(extrinsics: FloatArray) -> FloatArray:
    """Invert a world-to-camera [4,4] extrinsics matrix for the camera's
    own world-frame position: 0 = R @ x_world + t => x_world = -R^T @ t."""
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    result: FloatArray = -rotation.T @ translation
    return result


def _camera_depth_m(position_m: FloatArray, extrinsics: FloatArray) -> float:
    homogeneous = np.append(position_m, 1.0)
    return float(abs((extrinsics @ homogeneous)[2]))


def _xyz(observation: Observation) -> tuple[float, float, float]:
    measurement = observation.measurement
    assert isinstance(measurement, WorldPositionMeasurement)
    return (measurement.x_m, measurement.y_m, measurement.z_m)


@dataclass
class FrameRecord:
    """Everything one scored frame contributes to the evaluation."""

    regime: MotionRegime
    distance_m: float
    position_sq_error: float
    axis_sq_error: FloatArray
    velocity_sq_error: float
    nees: ConsistencyResidual
    standardized_innovation_x: float | None
    copy_previous_sq_error: float
    cv_no_update_sq_error: float
    cv_no_update_velocity_sq_error: float
    velocity_variance_diag_mps2: tuple[float, float, float]
    """Day 25, Objective 1: the filter's own POSTERIOR velocity-diagonal
    covariance entries (x/y/z, (m/s)^2) at this frame -- read directly off
    ``estimate.cov_array()``, i.e. after any velocity-covariance floor has
    already been applied (:func:`~src.estimator.filter.run_single_entity_filter`
    clamps before returning). This is what settles whether the floor
    actually binds in a real run, as opposed to comparing its closed-form
    value against a synthetic walk's convergence (see
    ``scripts/velocity_floor_frame_rate_sweep.py``)."""
    log_predictive_density: float = float("nan")
    """Day 31, Objective 1: log density of this frame's PREDICTIVE
    distribution evaluated at the true state, nats.

    For a single-Gaussian posterior this is derived from NEES and
    ``log det P`` (see :func:`~src.estimator.informativeness.
    gaussian_log_density`) rather than recomputed from the error vector,
    so it cannot drift from the consistency stage's own number. For IMM it
    is the exact MIXTURE density -- Day 22's collapsed-Gaussian caveat
    does not apply to a log score, because a density has no
    single-Gaussian assumption to violate.

    Defaulted to NaN rather than required, deliberately: this is the one
    field a FrameRecord constructed by an older test fixture can be
    missing, and a NaN propagates into an unreportable margin instead of
    into a plausible wrong one."""
    nees_mixture: ConsistencyResidual | None = None
    """Day 22, Objective 1(b): probability-weighted per-mode NEES -- valid
    for a gaussian_mixture posterior. None for single-model records (no
    mixture exists to weight)."""
    covered_by_sampling: bool | None = None
    """Day 22, Objective 1(a): whether GT fell inside the mixture's
    empirical (nonparametric) 95% credible region. None for single-model
    records."""


@dataclass
class TrackResult:
    """One agent-track's contribution to a golden set's evaluation."""

    frames: list[FrameRecord] = field(default_factory=list)


def _make_observations(
    track_xyz: FloatArray,
    extrinsics: FloatArray,
    measurement_model: MeasurementModel,
    fps: float,
    sensor_id: str,
    rng: np.random.Generator,
) -> list[Observation]:
    dt_ns = int(round(1e9 / fps))
    observations = []
    for t in range(track_xyz.shape[0]):
        gt = track_xyz[t]
        distance = _camera_depth_m(gt, extrinsics)
        sigma = measurement_model.sigma_m(distance)
        noisy = gt + rng.normal(0.0, sigma, size=3)
        ts_ns = BASE_TS_NS + t * dt_ns
        observations.append(
            Observation(
                observation_id=generate_ulid(now_ns=ts_ns),
                sensor_id=sensor_id,
                ts_ns=ts_ns,
                frame_ref=f"{sensor_id}/frame-{t}",
                measurement=WorldPositionMeasurement(
                    x_m=float(noisy[0]), y_m=float(noisy[1]), z_m=float(noisy[2])
                ),
                uncertainty=Uncertainty(
                    kind="gaussian_3d", params=(("sigma_m", float(sigma)),)
                ),
                frame_of_reference=_frame_of_reference(),
                producer_shas=("scripts/eval_estimator.py",),
                envelope_status=(
                    "within_envelope"
                    if measurement_model.is_within_envelope(distance)
                    else "outside_envelope"
                ),
            )
        )
    return observations


def _evaluate_track(
    track_xyz: FloatArray,
    extrinsics: FloatArray,
    sensor_id: str,
    fps: float,
    motion_model: MotionModel,
    measurement_model: MeasurementModel,
    rng: np.random.Generator,
) -> TrackResult | None:
    frames = track_xyz.shape[0]
    if frames <= FIRST_COMPARABLE_INDEX:
        return None

    observations = _make_observations(
        track_xyz, extrinsics, measurement_model, fps, sensor_id, rng
    )
    camera_position = _camera_position_world(extrinsics)

    graph = StateGraph()
    run_single_entity_filter(
        graph,
        observations,
        motion_model,
        measurement_model,
        manifest_sha="scripts/eval_estimator.py",
        sensor_origin_m=tuple(camera_position.tolist()),  # type: ignore[arg-type]
    )

    dt_s = 1.0 / fps
    gt_velocity = np.zeros_like(track_xyz)
    gt_velocity[1:] = (track_xyz[1:] - track_xyz[:-1]) / dt_s
    gt_velocity[0] = gt_velocity[1]

    regimes = classify_track(track_xyz, dt_s)

    obs_xyz = np.array([_xyz(o) for o in observations])
    cv_velocity = (obs_xyz[1] - obs_xyz[0]) / dt_s
    cv_position = obs_xyz[1].copy()

    H = measurement_model.H()

    # Innovation reconstruction needs the PREVIOUS corrected estimate,
    # replaying the same predict step the filter itself took internally,
    # through the filter's own public F/Q/H/R interface only.
    query0 = StateQuery(
        at_ts_ns=observations[0].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
    )
    prev_estimate = solve_state(query0, graph)

    result = TrackResult()
    for t in range(1, frames):
        if t >= 2:
            cv_position = cv_position + cv_velocity * dt_s

        query = StateQuery(
            at_ts_ns=observations[t].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
        )
        estimate = solve_state(query, graph)

        gt_pos = track_xyz[t]
        distance = _camera_depth_m(gt_pos, extrinsics)

        # Reconstruct the pre-update predicted state from the PREVIOUS
        # corrected estimate, to get the innovation the filter itself saw.
        predicted_mean = motion_model.f(prev_estimate.mean_array(), dt_s)
        predicted_cov = motion_model.F(
            dt_s
        ) @ prev_estimate.cov_array() @ motion_model.F(dt_s).T + motion_model.Q(dt_s)
        R = measurement_model.R(distance)
        innovation = obs_xyz[t] - H @ predicted_mean
        innovation_cov = H @ predicted_cov @ H.T + R
        sigma_x = float(np.sqrt(innovation_cov[0, 0]))
        standardized_innovation_x = (
            float(innovation[0] / sigma_x) if sigma_x > 0 else None
        )
        prev_estimate = estimate

        if t < FIRST_COMPARABLE_INDEX:
            continue

        pos_err = estimate.position_m() - gt_pos
        vel_err = estimate.velocity_mps() - gt_velocity[t]
        full_gt = np.concatenate([gt_pos, gt_velocity[t]])
        error6 = estimate.mean_array() - full_gt

        prev_obs = obs_xyz[t - 1]
        copy_previous_err = prev_obs - gt_pos
        cv_err = cv_position - gt_pos
        cv_vel_err = cv_velocity - gt_velocity[t]

        result.frames.append(
            FrameRecord(
                regime=regimes[t],
                distance_m=distance,
                position_sq_error=float(np.dot(pos_err, pos_err)),
                axis_sq_error=pos_err**2,
                velocity_sq_error=float(np.dot(vel_err, vel_err)),
                nees=compute_nees(error6, estimate.cov_array()),
                log_predictive_density=_gaussian_frame_log_density(
                    error6, estimate.cov_array()
                ),
                standardized_innovation_x=standardized_innovation_x,
                copy_previous_sq_error=float(
                    np.dot(copy_previous_err, copy_previous_err)
                ),
                cv_no_update_sq_error=float(np.dot(cv_err, cv_err)),
                cv_no_update_velocity_sq_error=float(np.dot(cv_vel_err, cv_vel_err)),
                velocity_variance_diag_mps2=tuple(  # type: ignore[arg-type]
                    np.diag(estimate.cov_array())[3:6].tolist()
                ),
            )
        )

    return result


def _evaluate_track_imm(
    track_xyz: FloatArray,
    extrinsics: FloatArray,
    sensor_id: str,
    fps: float,
    imm_config: ImmConfig,
    measurement_model: MeasurementModel,
    rng: np.random.Generator,
    sampling_rng: np.random.Generator,
) -> TrackResult | None:
    """The IMM counterpart of :func:`_evaluate_track`.

    Same observations (same ``rng`` draw sequence -- a caller evaluating
    both filters on the same track should construct a fresh ``rng`` at the
    same seed for each, so both see IDENTICAL noisy observations; see
    :func:`_score_golden_set`), same regimes, same baselines. Innovation
    whiteness is not computed here — IMM has no single innovation sequence
    (see module docstring) — every ``FrameRecord`` carries
    ``standardized_innovation_x=None``.

    Day 22, Objective 1: three NEES-family numbers are computed per frame,
    not one, because Day 21's single pooled NEES collapsed the mixture to
    one (mean, cov) and tested Gaussianity on the result — an assumption
    IMM explicitly violates (see src/estimator/consistency.py's module
    docstring). ``FrameRecord.nees`` here is the SAME collapsed
    computation Day 21 used (kept, via
    :func:`~src.estimator.consistency.compute_collapsed_gaussian_nees_diagnostic`,
    now explicitly labeled as invalid for judging IMM's consistency rather
    than silently trusted); ``nees_mixture`` and ``covered_by_sampling``
    are the two mixture-valid alternatives Objective 1 asks for
    "alongside" it. ``sampling_rng`` is a SEPARATE stream from ``rng`` --
    see :data:`EVAL_SEED_MIXTURE_SAMPLING`.
    """
    frames = track_xyz.shape[0]
    if frames <= FIRST_COMPARABLE_INDEX:
        return None

    observations = _make_observations(
        track_xyz, extrinsics, measurement_model, fps, sensor_id, rng
    )
    camera_position = _camera_position_world(extrinsics)

    graph = StateGraph()
    run_imm_filter(
        graph,
        observations,
        imm_config,
        measurement_model,
        manifest_sha="scripts/eval_estimator.py",
        sensor_origin_m=tuple(camera_position.tolist()),  # type: ignore[arg-type]
    )

    dt_s = 1.0 / fps
    gt_velocity = np.zeros_like(track_xyz)
    gt_velocity[1:] = (track_xyz[1:] - track_xyz[:-1]) / dt_s
    gt_velocity[0] = gt_velocity[1]

    regimes = classify_track(track_xyz, dt_s)

    obs_xyz = np.array([_xyz(o) for o in observations])
    cv_velocity = (obs_xyz[1] - obs_xyz[0]) / dt_s
    cv_position = obs_xyz[1].copy()

    result = TrackResult()
    for t in range(1, frames):
        if t >= 2:
            cv_position = cv_position + cv_velocity * dt_s

        query = StateQuery(
            at_ts_ns=observations[t].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
        )
        estimate = solve_state(query, graph)

        gt_pos = track_xyz[t]
        distance = _camera_depth_m(gt_pos, extrinsics)

        if t < FIRST_COMPARABLE_INDEX:
            continue

        pos_err = estimate.position_m() - gt_pos
        vel_err = estimate.velocity_mps() - gt_velocity[t]
        full_gt = np.concatenate([gt_pos, gt_velocity[t]])
        error6 = estimate.mean_array() - full_gt

        prev_obs = obs_xyz[t - 1]
        copy_previous_err = prev_obs - gt_pos
        cv_err = cv_position - gt_pos
        cv_vel_err = cv_velocity - gt_velocity[t]

        components = estimate.mode_components()
        assert components is not None  # every IMM estimate carries mode data
        mode_weights = {name: w for name, (w, _, _) in components.items()}
        mode_means = {name: m for name, (_, m, _) in components.items()}
        mode_covs = {name: c for name, (_, _, c) in components.items()}

        result.frames.append(
            FrameRecord(
                regime=regimes[t],
                distance_m=distance,
                position_sq_error=float(np.dot(pos_err, pos_err)),
                axis_sq_error=pos_err**2,
                velocity_sq_error=float(np.dot(vel_err, vel_err)),
                nees=compute_collapsed_gaussian_nees_diagnostic(
                    error6, estimate.cov_array()
                ),
                # The mixture's OWN density, not the collapsed one -- see
                # FrameRecord.log_predictive_density.
                log_predictive_density=mixture_log_density(
                    full_gt,
                    [mode_weights[name] for name in components],
                    [mode_means[name] for name in components],
                    [mode_covs[name] for name in components],
                ),
                standardized_innovation_x=None,
                copy_previous_sq_error=float(
                    np.dot(copy_previous_err, copy_previous_err)
                ),
                cv_no_update_sq_error=float(np.dot(cv_err, cv_err)),
                cv_no_update_velocity_sq_error=float(np.dot(cv_vel_err, cv_vel_err)),
                velocity_variance_diag_mps2=tuple(  # type: ignore[arg-type]
                    np.diag(estimate.cov_array())[3:6].tolist()
                ),
                nees_mixture=compute_mixture_nees(
                    full_gt, mode_weights, mode_means, mode_covs
                ),
                covered_by_sampling=empirical_coverage_by_sampling(
                    full_gt, mode_weights, mode_means, mode_covs, rng=sampling_rng
                ),
            )
        )

    return result


def _rmse(sq_errors: list[float]) -> float:
    return float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")


def _gaussian_frame_log_density(error: FloatArray, cov: FloatArray) -> float:
    """One frame's Gaussian log predictive density at the truth.

    Derived from the same NEES the consistency stage computes, plus
    ``log det P`` -- not recomputed from the error vector, so the two
    numbers cannot disagree about the same frame. ``slogdet`` rather than
    ``log(det(...))`` because a 6x6 posterior covariance with metres and
    metres-per-second on its diagonal has a determinant small enough to
    underflow to zero in float64 while every eigenvalue is perfectly
    healthy.
    """
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError(
            "posterior covariance is not positive-definite, so its log "
            "predictive density is undefined; this is an estimator defect, "
            "not a scoring one"
        )
    nees = float(error @ np.linalg.solve(cov, error))
    return gaussian_log_density(nees, float(log_det))


def _coverage_stats(records: list[FrameRecord]) -> dict[str, Any]:
    nees_list = [r.nees for r in records]
    within = [r.within_bound for r in nees_list if r.within_bound is not None]
    fraction_outside = fraction_outside_bound(nees_list, "nees")
    coverage = (
        1.0 - fraction_outside if not np.isnan(fraction_outside) else float("nan")
    )
    return {
        "n": len(records),
        "nees_pass_rate_within_95": float(np.mean(within)) if within else float("nan"),
        "empirical_coverage_95": coverage,
    }


def _calibration_and_informativeness(
    records: list[FrameRecord],
    baseline: ConstantCovarianceBaseline,
    filter_kind: FilterKind,
) -> CalibrationAndInformativeness:
    """Day 31, Objective 1 -- the coverage figure and its informativeness
    margin, co-emitted.

    Returns a type whose every field is required, so there is no
    representation of "coverage, and no margin". Day 30's config B is the
    worked example of why: a cessation coverage of 0.9946 was quoted,
    adopted and defended across three ADR revisions while the velocity
    uncertainty behind it was a constant, and nothing in the report shape
    put the two facts next to each other.

    ``filter_kind`` selects which coverage figure to pair with the
    margin -- the mixture-VALID sampling coverage for IMM, the NEES-based
    one for a genuinely single-Gaussian posterior -- so this cannot pair a
    margin with a coverage number Day 22 already declared invalid for that
    posterior family. The margin itself needs no such branch: a log
    density is exact for both families.
    """
    if not records:
        return CalibrationAndInformativeness(
            coverage_95=float("nan"),
            baseline_coverage_95=float("nan"),
            elpd_nats=float("nan"),
            baseline_elpd_nats=float("nan"),
            n=0,
        )
    position_sq = [r.position_sq_error for r in records]
    velocity_sq = [r.velocity_sq_error for r in records]

    if filter_kind == "imm":
        coverage = _mixture_coverage_stats(records)["empirical_coverage_by_sampling_95"]
    else:
        coverage = _coverage_stats(records)["empirical_coverage_95"]

    densities = np.array([r.log_predictive_density for r in records], dtype=np.float64)
    baseline_densities = np.array(
        [
            baseline.log_density(position, velocity)
            for position, velocity in zip(position_sq, velocity_sq)
        ],
        dtype=np.float64,
    )
    return CalibrationAndInformativeness(
        coverage_95=float(coverage),
        baseline_coverage_95=constant_baseline_coverage(
            baseline, position_sq, velocity_sq, chi2_upper_bound(STATE_DOF, 0.95)
        ),
        elpd_nats=float(np.mean(densities)),
        baseline_elpd_nats=float(np.mean(baseline_densities)),
        n=len(records),
    )


def _mixture_coverage_stats(records: list[FrameRecord]) -> dict[str, Any]:
    """The two mixture-VALID consistency numbers (Day 22, Objective 1),
    computed alongside (never instead of) :func:`_coverage_stats`'s
    collapsed-Gaussian number -- see FrameRecord.nees_mixture/
    covered_by_sampling's docstrings for what each one is valid for."""
    mixture_residuals = [r.nees_mixture for r in records if r.nees_mixture is not None]
    within = [r.within_bound for r in mixture_residuals if r.within_bound is not None]
    fraction_outside = (
        fraction_outside_bound(mixture_residuals, "nees")
        if mixture_residuals
        else float("nan")
    )
    mixture_nees_coverage = (
        1.0 - fraction_outside if not np.isnan(fraction_outside) else float("nan")
    )
    sampled = [
        r.covered_by_sampling for r in records if r.covered_by_sampling is not None
    ]
    return {
        "n": len(sampled),
        "mixture_nees_pass_rate_within_95": (
            float(np.mean(within)) if within else float("nan")
        ),
        "mixture_nees_coverage_95": mixture_nees_coverage,
        "empirical_coverage_by_sampling_95": (
            float(np.mean(sampled)) if sampled else float("nan")
        ),
    }


_FLOOR_PROBE_DT_S = 1.0 / 12.0
"""Any positive timestep. Since Day 24's correction the velocity floor is
ABSOLUTE — ``pedestrian_velocity_covariance_floor_mps2`` ignores its
``dt_s`` argument entirely — so the value probed here does not affect the
result. This project's own 12fps is used so the call reads as the real
configuration rather than as an arbitrary number, and the function is
called rather than its formula re-typed, so there is no second copy of
the derivation to drift from the first."""


def _floor_sigma_v_mps() -> float:
    """The velocity floor as a sigma (m/s), not a variance."""
    return float(np.sqrt(pedestrian_velocity_covariance_floor_mps2(_FLOOR_PROBE_DT_S)))


def _sigma_v_distribution(records: list[FrameRecord]) -> dict[str, Any]:
    """Day 25, Objective 1: the filter's own converged sigma_v (m/s) over
    ``records``, as a distribution -- a single mean hides exactly the
    question this objective asks (does the floor bind on SOME frames of a
    regime, or none). Per frame, sigma_v is sqrt of the mean of the 3
    velocity-diagonal variance entries (isotropic collapse -- the floor
    itself is applied identically to all three axes, see
    ``apply_velocity_covariance_floor``).

    Day 30, Objective 3 adds ``frames_at_floor``/``fraction_at_floor``.
    Day 25 reported min/p50/max and read "min == p50 == max == 1.5000" as
    confirmation that the clamp fires — which it is, but it is also the
    signature of something else the same numbers cannot distinguish: a
    filter whose velocity uncertainty is CONSTANT. Three order statistics
    coinciding is suggestive; the fraction of frames actually pinned is
    the number that settles it, and it was one line away the whole time.
    """
    if not records:
        return {
            "n": 0,
            "min_mps": float("nan"),
            "p50_mps": float("nan"),
            "max_mps": float("nan"),
            "frames_at_floor": 0,
            "fraction_at_floor": float("nan"),
        }
    sigma_v = np.sqrt(
        np.array([np.mean(r.velocity_variance_diag_mps2) for r in records])
    )
    floor_mps = _floor_sigma_v_mps()
    # The clamp is a max(), so a floored frame sits at the floor exactly.
    # rtol rather than == because sigma_v goes through a mean-then-sqrt.
    at_floor = int(np.sum(np.isclose(sigma_v, floor_mps, rtol=1e-9, atol=0.0)))
    return {
        "n": len(records),
        "min_mps": float(np.min(sigma_v)),
        "p50_mps": float(np.median(sigma_v)),
        "max_mps": float(np.max(sigma_v)),
        "floor_mps": floor_mps,
        "frames_at_floor": at_floor,
        "fraction_at_floor": at_floor / len(records),
    }


def _baseline_margin_block(records: list[FrameRecord]) -> dict[str, Any]:
    """Position RMSE for filter/copy-previous/constant-velocity-no-update
    over exactly ``records`` -- the Objective-1 gap, now computable for any
    slice (a distance bucket, a regime, the whole set)."""
    filter_rmse = _rmse([r.position_sq_error for r in records])
    copy_previous_rmse = _rmse([r.copy_previous_sq_error for r in records])
    cv_rmse = _rmse([r.cv_no_update_sq_error for r in records])
    margin_vs_copy_previous = (
        copy_previous_rmse - filter_rmse
        if not (np.isnan(filter_rmse) or np.isnan(copy_previous_rmse))
        else float("nan")
    )
    margin_vs_cv = (
        cv_rmse - filter_rmse
        if not (np.isnan(filter_rmse) or np.isnan(cv_rmse))
        else float("nan")
    )
    return {
        "n": len(records),
        "filter_rmse_m": filter_rmse,
        "copy_previous_rmse_m": copy_previous_rmse,
        "constant_velocity_no_update_rmse_m": cv_rmse,
        "margin_vs_copy_previous_m": margin_vs_copy_previous,
        "margin_vs_constant_velocity_m": margin_vs_cv,
    }


def _innovation_block(records: list[list[float]]) -> dict[str, Any]:
    white, correlation, bound, n_pairs = is_white(records)
    return {
        "white": white,
        "lag1_autocorrelation": correlation,
        "white_noise_bound_95": bound,
        "n_pairs": n_pairs,
    }


def _score_golden_set(
    version: str,
    root: Path,
    config: IronConfig,
    filter_kind: FilterKind = "single_model",
    imm_config: ImmConfig | None = None,
    velocity_covariance_floor: bool = False,
) -> dict[str, Any] | None:
    try:
        golden = load_golden_set(root, version)
    except GoldenSetError as exc:
        print(f"ERROR loading {version}: {exc}", file=sys.stderr)
        return None

    datasets = {c.source_dataset for c in golden.clips if c.source_dataset}
    dataset = datasets.pop() if len(datasets) == 1 else f"synthetic-indoor-{version}"
    clip_root = config.paths.resolved_data_dir / "synthetic" / dataset

    fps_by_clip: dict[str, float] = {}
    manifest_path = clip_root / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        fps_by_clip = {c["clip_id"]: float(c["fps"]) for c in manifest.get("clips", [])}

    gate_frames = None
    for clip in golden.clips:
        candidate = clip_root / f"{clip.clip_id}.npz"
        if candidate.exists():
            with np.load(candidate) as sample:
                gate_frames = np.asarray(sample["rgb"][:4])
            break

    if gate_frames is not None:
        verdict = validity.evaluate("state_estimation", version, frames=gate_frames)
        if not verdict.passed:
            print(f"\nREFUSED: {version} cannot score state_estimation.")
            print(f"  {verdict.reason}")
            print("  No accuracy metric is emitted. The refusal IS the result.")
            return {"version": version, "refused": True, "reason": verdict.reason}

    if filter_kind == "imm" and imm_config is None:
        imm_config = default_imm_config(
            velocity_covariance_floor=velocity_covariance_floor
        )
    motion_model = motion_model_for(
        "person", velocity_covariance_floor=velocity_covariance_floor
    )
    measurement_model = measurement_model_for(
        sensor=version, capability="state_estimation"
    )
    # Same seed regardless of filter_kind: both filters must see IDENTICAL
    # noisy observations for the Objective-4 comparison to be about the
    # filter, not about which draw of noise each one happened to get.
    rng = np.random.default_rng(EVAL_SEED)
    # Separate stream (Day 22): consumed only by IMM's mixture-sampling
    # coverage check, so it never perturbs the observation-noise draws
    # single_model and imm runs must share. See EVAL_SEED_MIXTURE_SAMPLING.
    sampling_rng = np.random.default_rng(EVAL_SEED_MIXTURE_SAMPLING)

    tracks_scored = 0
    clips_with_no_agents = 0
    all_records: list[FrameRecord] = []
    innovations_by_track: list[list[float]] = []
    innovations_by_regime: dict[str, list[list[float]]] = {
        r: [] for r in MOTION_REGIMES
    }

    for clip in golden.clips:
        clip_path = clip_root / f"{clip.clip_id}.npz"
        if not clip_path.exists():
            continue
        with np.load(clip_path) as data:
            agent_xyz = np.asarray(data["agent_xyz"], dtype=np.float64)
            extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        fps = fps_by_clip.get(clip.clip_id, 12.0)

        n_agents = agent_xyz.shape[1]
        if n_agents == 0:
            clips_with_no_agents += 1
            continue

        for agent_index in range(n_agents):
            # Day 30, Objective 4: declared at the boundary. This
            # script's own numbers are magnitudes and per-axis RMSE; the
            # per-axis report names its convention explicitly below
            # (`axis_convention`), which it did not before.
            track = gt_position_track(
                agent_xyz[:, agent_index, :], GENERATOR_AXES
            ).values
            if filter_kind == "imm":
                assert imm_config is not None
                result = _evaluate_track_imm(
                    track,
                    extrinsics,
                    clip.clip_id,
                    fps,
                    imm_config,
                    measurement_model,
                    rng,
                    sampling_rng,
                )
            else:
                result = _evaluate_track(
                    track,
                    extrinsics,
                    clip.clip_id,
                    fps,
                    motion_model,
                    measurement_model,
                    rng,
                )
            if result is None:
                continue
            tracks_scored += 1
            all_records.extend(result.frames)

            track_innovations = [
                r.standardized_innovation_x
                for r in result.frames
                if r.standardized_innovation_x is not None
            ]
            if track_innovations:
                innovations_by_track.append(track_innovations)

            per_regime_track: dict[str, list[float]] = {r: [] for r in MOTION_REGIMES}
            for r in result.frames:
                if r.standardized_innovation_x is not None:
                    per_regime_track[r.regime].append(r.standardized_innovation_x)
            for regime_name, values in per_regime_track.items():
                if values:
                    innovations_by_regime[regime_name].append(values)

    if tracks_scored == 0:
        return {
            "version": version,
            "refused": False,
            "tracks_scored": 0,
            "clips_with_no_agents": clips_with_no_agents,
            "note": "no agent track in this set had enough frames to evaluate",
        }

    # -- the informativeness baseline (Day 31, Objective 1) -------------
    #
    # Fitted ONCE, over every scored frame of the set, and then used
    # unchanged for every per-regime margin below. Fitting it per regime
    # would hand the trivial predictor the ground-truth regime label --
    # information no deployable predictor has, and enough of it to make a
    # constant look like an estimator. See ConstantCovarianceBaseline.
    informativeness_baseline = fit_constant_covariance(
        [r.position_sq_error for r in all_records],
        [r.velocity_sq_error for r in all_records],
        fitted_on=f"{version}, all {len(all_records)} scored frames",
    )

    # -- pooled, whole-set summary (Day 20 shape, preserved) -------------
    position_rmse = _rmse([r.position_sq_error for r in all_records])
    velocity_rmse = _rmse([r.velocity_sq_error for r in all_records])
    copy_previous_rmse = _rmse([r.copy_previous_sq_error for r in all_records])
    cv_no_update_rmse = _rmse([r.cv_no_update_sq_error for r in all_records])
    cv_no_update_velocity_rmse = _rmse(
        [r.cv_no_update_velocity_sq_error for r in all_records]
    )

    require_baseline("estimator.position_rmse_m")
    require_baseline("estimator.velocity_rmse_mps")
    position_baselines = compute_baselines(
        "estimator.position_rmse_m",
        copy_previous_rmse_m=copy_previous_rmse,
        constant_velocity_rmse_m=cv_no_update_rmse,
    )
    velocity_baselines = compute_baselines(
        "estimator.velocity_rmse_mps",
        constant_velocity_rmse_mps=cv_no_update_velocity_rmse,
    )
    position_margin = margin(position_rmse, position_baselines, higher_is_better=False)
    velocity_margin = margin(velocity_rmse, velocity_baselines, higher_is_better=False)

    axis_arr = np.array([r.axis_sq_error for r in all_records])
    axis_rmse = (
        np.sqrt(axis_arr.mean(axis=0)).tolist() if axis_arr.size else [float("nan")] * 3
    )

    # -- by distance bucket, INCLUDING the baseline margin (Objective 1) --
    by_distance_bucket = {}
    for name, low, high in DISTANCE_BUCKETS:
        bucket_records = [r for r in all_records if low <= r.distance_m < high]
        by_distance_bucket[name] = _baseline_margin_block(bucket_records)

    # -- by motion regime: baseline margin AND consistency (Objective 2) -
    by_regime = {}
    for regime_name in MOTION_REGIMES:
        regime_records = [r for r in all_records if r.regime == regime_name]
        block = _baseline_margin_block(regime_records)
        block["sigma_v_mps"] = _sigma_v_distribution(regime_records)
        block["consistency"] = _coverage_stats(regime_records)
        block["calibration"] = _calibration_and_informativeness(
            regime_records, informativeness_baseline, filter_kind
        ).as_dict()
        if filter_kind == "imm":
            block["consistency_mixture"] = _mixture_coverage_stats(regime_records)
        block["innovation"] = _innovation_block(innovations_by_regime[regime_name])
        block["thin_evidence"] = (
            len(regime_records) < MIN_REGIME_FRAMES_FOR_A_CONCLUSION
        )
        by_regime[regime_name] = block

    consistency = _coverage_stats(all_records)
    consistency_mixture = (
        _mixture_coverage_stats(all_records) if filter_kind == "imm" else None
    )
    innovation = _innovation_block(innovations_by_track)

    return {
        "version": version,
        "filter_kind": filter_kind,
        "velocity_covariance_floor": velocity_covariance_floor,
        "refused": False,
        "tracks_scored": tracks_scored,
        "clips_with_no_agents": clips_with_no_agents,
        "points_scored": len(all_records),
        "motion_model_sha": motion_model.sha,
        "measurement_model_sha": measurement_model.sha,
        "imm_config_sha": imm_config.sha if imm_config is not None else None,
        "position": {
            "rmse_m": position_rmse,
            "axis_rmse_m_xyz": axis_rmse,
            # Which axis is which. Named because "xyz" alone does
            # not say WHOSE xyz: `agent_xyz` is [x, y_up, z_depth]
            # while src/model/world.py's frame is [x, y, z_up], and
            # a reader taking index 2 for height would be reading
            # depth error as vertical error -- the Day-29 bug,
            # reachable through a report key instead of an array.
            "axis_convention": GENERATOR_AXES.value,
            "baselines": [b.as_dict() for b in position_baselines],
            "margin_m": position_margin,
        },
        "velocity": {
            "rmse_mps": velocity_rmse,
            "baselines": [b.as_dict() for b in velocity_baselines],
            "margin_mps": velocity_margin,
        },
        "by_distance_bucket": by_distance_bucket,
        "by_regime": by_regime,
        "consistency": consistency,
        "consistency_mixture": consistency_mixture,
        "innovation": innovation,
    }


def _print_report(report: dict[str, Any]) -> None:
    version = report["version"]
    print("=" * 78)
    print(f"ESTIMATOR ACCURACY: {version}")
    print("=" * 78)
    if report.get("refused"):
        print(f"REFUSED: {report['reason']}")
        return
    if report.get("tracks_scored", 0) == 0:
        print(f"No trajectory could be scored: {report.get('note', 'unknown reason')}")
        print(f"  clips with no agents: {report.get('clips_with_no_agents', 0)}")
        return

    print(f"tracks scored     : {report['tracks_scored']}")
    print(f"points scored     : {report['points_scored']}")
    print(
        f"clips w/ no agents: {report['clips_with_no_agents']} (skipped, not scoreable)"
    )
    print()
    pos = report["position"]
    print(
        f"position RMSE     : {pos['rmse_m']:.4f} m  (x/y/z: "
        f"x {pos['axis_rmse_m_xyz'][0]:.4f} / y_up "
        f"{pos['axis_rmse_m_xyz'][1]:.4f} / z_depth "
        f"{pos['axis_rmse_m_xyz'][2]:.4f} m)"
    )
    for b in pos["baselines"]:
        print(f"    vs {b['name']:26} {b['value']:.4f} m")
    print(
        f"    margin            {pos['margin_m']:+.4f} m "
        "(positive = filter beats baseline)"
    )
    print()
    vel = report["velocity"]
    print(f"velocity RMSE     : {vel['rmse_mps']:.4f} m/s")
    for b in vel["baselines"]:
        print(f"    vs {b['name']:26} {b['value']:.4f} m/s")
    print(f"    margin            {vel['margin_mps']:+.4f} m/s")
    print()

    print("by GT distance bucket -- filter / copy-previous / constant-velocity RMSE:")
    for name, block in report["by_distance_bucket"].items():
        if block["n"] == 0:
            print(f"    {name:8} n=0 (empty)")
            continue
        print(
            f"    {name:8} n={block['n']:5}  "
            f"filter {block['filter_rmse_m']:.4f} m  "
            f"copy-prev {block['copy_previous_rmse_m']:.4f} m "
            f"(margin {block['margin_vs_copy_previous_m']:+.4f})  "
            f"const-vel {block['constant_velocity_no_update_rmse_m']:.4f} m "
            f"(margin {block['margin_vs_constant_velocity_m']:+.4f})"
        )
    print()

    print("by GT motion regime -- filter / baselines / NEES coverage / innovation:")
    for name, block in report["by_regime"].items():
        thin = " [THIN EVIDENCE]" if block["thin_evidence"] else ""
        if block["n"] == 0:
            print(f"    {name:12} n=0 (empty){thin}")
            continue
        cons = block["consistency"]
        innov = block["innovation"]
        sv = block["sigma_v_mps"]
        print(
            f"    {name:12} n={block['n']:5}  "
            f"filter {block['filter_rmse_m']:.4f} m  "
            f"copy-prev margin {block['margin_vs_copy_previous_m']:+.4f}  "
            f"const-vel margin {block['margin_vs_constant_velocity_m']:+.4f}{thin}"
        )
        print(
            f"                 converged sigma_v (m/s) min/p50/max: "
            f"{sv['min_mps']:.4f} / {sv['p50_mps']:.4f} / {sv['max_mps']:.4f}"
        )
        print(
            f"                 NEES coverage {cons['empirical_coverage_95']:.4f} "
            f"(target 0.95)   "
            f"innovation lag1-autocorr {innov['lag1_autocorrelation']:+.4f} "
            f"(white-noise bound ±{innov['white_noise_bound_95']:.4f}, "
            f"n_pairs={innov['n_pairs']})   "
            f"{'WHITE' if innov['white'] else 'NOT WHITE'}"
        )
        mix = block.get("consistency_mixture")
        if mix is not None:
            print(
                f"                 [mixture-valid, n={mix['n']:5}] "
                f"per-mode-weighted NEES coverage "
                f"{mix['mixture_nees_coverage_95']:.4f}   "
                f"sampling-HPD coverage "
                f"{mix['empirical_coverage_by_sampling_95']:.4f}   "
                f"(collapsed-Gaussian NEES above is a Day-22 diagnostic, "
                f"NOT a validated check for this posterior)"
            )
    print()

    cons = report["consistency"]
    print(f"NEES pass rate (within 95% bound): {cons['nees_pass_rate_within_95']:.4f}")
    print(
        f"empirical 95% coverage           : "
        f"{cons['empirical_coverage_95']:.4f}  (target: 0.95)"
    )
    if (
        not np.isnan(cons["empirical_coverage_95"])
        and cons["empirical_coverage_95"] < 0.90
    ):
        print(
            "    ** OVERCONFIDENT: fewer than 90% of estimates fall within their own "
            "95% bound. The filter's covariance understates its true error. **"
        )
    elif (
        not np.isnan(cons["empirical_coverage_95"])
        and cons["empirical_coverage_95"] > 0.995
    ):
        print(
            "    UNDERCONFIDENT: essentially everything falls within bound; the "
            "filter's covariance is looser than its true error needs."
        )
    innov = report["innovation"]
    print(
        f"innovation whiteness (pooled)     : "
        f"lag1-autocorr {innov['lag1_autocorrelation']:+.4f}, "
        f"bound ±{innov['white_noise_bound_95']:.4f}, "
        f"n_pairs={innov['n_pairs']} -> "
        f"{'WHITE' if innov['white'] else 'NOT WHITE'}"
    )
    mix = report.get("consistency_mixture")
    if mix is not None:
        print()
        print(
            "** Day-22 Objective 1: the pooled NEES above collapses IMM's "
            "mixture to one (mean, cov) and is a DIAGNOSTIC ONLY, not a "
            "validated consistency check for this posterior. **"
        )
        print(
            f"mixture-valid per-mode-weighted NEES coverage : "
            f"{mix['mixture_nees_coverage_95']:.4f}  (target: 0.95)"
        )
        print(
            f"mixture-valid empirical (sampling) coverage   : "
            f"{mix['empirical_coverage_by_sampling_95']:.4f}  (target: 0.95)"
        )


NO_TRADE_CESSATION_REGIME = "cessation"
"""Day 22 Objective 3 restates the criterion around cessation specifically
-- Day 21/22's diagnosis pinned the actual failure there (onset was
already well-calibrated under the single model; maneuver has zero frames
in either golden set by construction). Checking "any transient regime"
the way Day 21 did would let an unrelated onset wiggle satisfy a
criterion meant to certify a cessation fix."""
NO_TRADE_STEADY_REGIMES = ("static", "sustained")
NO_TRADE_MATERIAL_IMPROVEMENT = 0.05
"""Cessation's empirical coverage must close its gap to NOMINAL_COVERAGE
by at least this much (in coverage points) for a config to count as
having "moved materially toward nominal" there. Declared, not fitted."""
NO_TRADE_DEGRADATION_TOLERANCE = 0.02
"""Day 25, Objective 3: now doing double duty on the directional
criterion. A steady regime's coverage moving further from
NOMINAL_COVERAGE than this, in EITHER direction, is what makes a move
count as a "degradation" worth naming at all (absorbs sampling noise
between two runs on the same, same-seed observations). Which of the two
outcomes a degradation THEN produces -- FAIL_OVERCONFIDENT or a
PASS_WITH_COST entry -- is decided separately, by the sign of the
candidate's own coverage error against NOMINAL_COVERAGE, not by this
constant. Declared, not fitted, same as NO_TRADE_MATERIAL_IMPROVEMENT."""
NOMINAL_COVERAGE = 0.95
"""The target empirical coverage every consistency check in this script
is judged against. Below this = overconfident (the covariance understates
true error); above this = underconfident (conservative, not misleading).
Named so the sign convention driving the Day 25 directional criterion is
legible at every call site, not a repeated magic 0.95."""


def _informativeness_margin(block: dict[str, Any]) -> float:
    """One regime's ELPD margin over the fitted constant baseline, or NaN
    if the regime is absent. See
    :mod:`src.estimator.informativeness`."""
    calibration = block.get("calibration")
    if not isinstance(calibration, dict) or not calibration.get("n"):
        return float("nan")
    return float(calibration.get("elpd_margin_nats", float("nan")))


def _regime_coverage(block: dict[str, Any]) -> float:
    """The one coverage figure to judge a regime by: the mixture-VALID
    sampling-based coverage when present (an IMM config's block), else the
    NEES-based coverage (valid as-is for a genuinely single-Gaussian
    posterior -- configs A and B)."""
    mixture = block.get("consistency_mixture")
    if mixture is not None:
        value = mixture.get("empirical_coverage_by_sampling_95", float("nan"))
        return float(value)
    return float(
        block.get("consistency", {}).get("empirical_coverage_95", float("nan"))
    )


@dataclass(frozen=True)
class NoTradeUnscoreable:
    """Cessation itself does not have enough frames in common between
    baseline and candidate to support any conclusion -- distinct from
    every other verdict below. Silently defaulting to pass or fail on
    thin evidence would be exactly the bounded-null-read-as-limit mistake
    this project has caught before (see the `iron-eval-discipline` skill's
    "Bounded Nulls" section)."""

    baseline: str
    candidate: str
    n_cessation: int


@dataclass(frozen=True)
class NoTradeNoImprovement:
    """Cessation is scoreable but did not move materially toward nominal.
    There is no measured benefit here to weigh a cost against -- this is
    neither a pass nor a directional failure, it is simply not yet a
    result worth adopting a config over."""

    baseline: str
    candidate: str
    n_cessation: int
    cessation_delta_toward_nominal: float


@dataclass(frozen=True)
class NoTradeFailOverconfident:
    """Non-negotiable. A steady regime's coverage moved toward
    overconfidence (below NOMINAL_COVERAGE) by more than
    NO_TRADE_DEGRADATION_TOLERANCE -- independent of how much cessation
    improved. See ADR 0010's Day 25 revision for why overconfidence and
    underconfidence are not symmetric risks for this product."""

    baseline: str
    candidate: str
    regime: str
    coverage_before: float
    coverage_after: float
    magnitude: float


@dataclass(frozen=True)
class NoTradePass:
    """Cessation improved materially; no steady regime degraded toward
    overconfidence beyond tolerance. Any steady-regime movement present
    was toward underconfidence, within tolerance, or absent entirely."""

    baseline: str
    candidate: str
    cessation_delta_toward_nominal: float
    n_cessation: int


@dataclass(frozen=True)
class NoTradePassWithCost:
    """Cessation improved materially; at least one steady regime degraded,
    but strictly toward underconfidence -- an efficiency cost (a wider,
    more conservative band), not a danger. `regime`/`magnitude` name the
    single worst such degradation; `costs` carries the full per-regime
    breakdown so the trade is numeric and complete in the record, never
    implied by a single collapsed number."""

    baseline: str
    candidate: str
    regime: str
    magnitude: float
    costs: tuple[tuple[str, float], ...]
    cessation_delta_toward_nominal: float
    n_cessation: int


@dataclass(frozen=True)
class NoTradeUninformative:
    """Day 31, Objective 1. Cessation calibration improved, no steady
    regime became overconfident — and the candidate's per-frame
    uncertainty stopped carrying information a fitted constant does not
    already carry.

    A distinct type rather than a cost, because it is not a trade: there
    is nothing on the other side of it. A constant-variance predictor
    satisfies every clause of the directional criterion trivially —
    cessation coverage improves, and no steady regime moves toward
    overconfidence, because nothing moves at all. Config B (Day 30) is
    the worked example, and this verdict exists so that the next one is
    caught by the criterion rather than by someone re-reading a sigma_v
    table two weeks later.

    `regime` names the worst COLLAPSE: a regime where the baseline
    configuration's uncertainty was informative and the candidate's is
    not. A candidate that was already uninformative where the baseline
    was too is not a collapse and is reported through the margins in the
    per-regime table instead — this verdict is about what a change
    DESTROYED, and both being degenerate is a fact about the pair, not
    about the change.
    """

    baseline: str
    candidate: str
    regime: str
    baseline_margin_nats: float
    candidate_margin_nats: float
    cessation_delta_toward_nominal: float
    n_cessation: int


NoTradeVerdict = (
    NoTradeUnscoreable
    | NoTradeNoImprovement
    | NoTradeFailOverconfident
    | NoTradeUninformative
    | NoTradePass
    | NoTradePassWithCost
)
"""STRUCTURAL (Day 25, Objective 3): `_no_trade_verdict` returns one of
these five dataclasses, never a bare boolean or a status string a caller
could collapse to "it passed" -- `NoTradePassWithCost` and `NoTradePass`
are distinct types with different fields, so a caller that only handles
`NoTradePass` fails type-checking (or an `isinstance`/match miss) rather
than silently treating a cost-bearing candidate as clean."""


def _no_trade_verdict(
    baseline_label: str,
    baseline_report: dict[str, Any],
    candidate_label: str,
    candidate_report: dict[str, Any],
) -> NoTradeVerdict:
    """Day 22 Objective 3 built this around cessation specifically; Day 25
    Objective 3 makes it DIRECTIONAL. Overconfidence (coverage below
    NOMINAL_COVERAGE) and underconfidence (coverage above it) are not
    equally dangerous for an evidence system: an overconfident estimator's
    stated uncertainty band is a lie, and every downstream confidence
    inherits it; an underconfident one is merely more conservative than it
    needs to be. See ADR 0010's Day 25 revision for the full rationale.

    A steady regime (`NO_TRADE_STEADY_REGIMES`) moving toward
    overconfidence by more than `NO_TRADE_DEGRADATION_TOLERANCE` fails the
    candidate outright (`NoTradeFailOverconfident`) -- non-negotiable,
    independent of cessation's own improvement. A move toward
    underconfidence is instead a COST (`NoTradePassWithCost`), weighed
    numerically against cessation's improvement rather than silently
    forgiven. Direction is read off the CANDIDATE's own coverage-error
    sign against `NOMINAL_COVERAGE` (below = overconfident), not the sign
    of the change itself -- a regime that was already overconfident at
    baseline and stays there is still `overconfident`, not "improving in
    the safe direction" just because the gap narrowed.

    Cessation must still improve materially for a `NoTradePass`/
    `NoTradePassWithCost` -- there is nothing to trade a cost against
    otherwise; a candidate that neither fails on overconfidence nor
    improves cessation gets `NoTradeNoImprovement`, not a default pass.
    """
    baseline_regimes = baseline_report.get("by_regime", {})
    candidate_regimes = candidate_report.get("by_regime", {})

    cessation_baseline = baseline_regimes.get(NO_TRADE_CESSATION_REGIME, {})
    cessation_candidate = candidate_regimes.get(NO_TRADE_CESSATION_REGIME, {})
    n_cessation = min(cessation_baseline.get("n", 0), cessation_candidate.get("n", 0))
    if n_cessation < MIN_REGIME_FRAMES_FOR_A_CONCLUSION:
        return NoTradeUnscoreable(baseline_label, candidate_label, n_cessation)

    base_cess_cov = _regime_coverage(cessation_baseline)
    cand_cess_cov = _regime_coverage(cessation_candidate)
    cessation_delta = float("nan")
    if not (np.isnan(base_cess_cov) or np.isnan(cand_cess_cov)):
        cessation_delta = abs(base_cess_cov - NOMINAL_COVERAGE) - abs(
            cand_cess_cov - NOMINAL_COVERAGE
        )
    cessation_improved = (
        not np.isnan(cessation_delta)
        and cessation_delta >= NO_TRADE_MATERIAL_IMPROVEMENT
    )

    fail: NoTradeFailOverconfident | None = None
    costs: list[tuple[str, float]] = []
    for name in NO_TRADE_STEADY_REGIMES:
        base_cov = _regime_coverage(baseline_regimes.get(name, {}))
        cand_cov = _regime_coverage(candidate_regimes.get(name, {}))
        if np.isnan(base_cov) or np.isnan(cand_cov):
            continue
        delta_toward_nominal = abs(base_cov - NOMINAL_COVERAGE) - abs(
            cand_cov - NOMINAL_COVERAGE
        )
        if delta_toward_nominal >= -NO_TRADE_DEGRADATION_TOLERANCE:
            continue  # improved, flat, or within sampling-noise tolerance
        magnitude = -delta_toward_nominal
        overconfident = (cand_cov - NOMINAL_COVERAGE) < 0
        if overconfident:
            candidate_fail = NoTradeFailOverconfident(
                baseline=baseline_label,
                candidate=candidate_label,
                regime=name,
                coverage_before=base_cov,
                coverage_after=cand_cov,
                magnitude=magnitude,
            )
            if fail is None or magnitude > fail.magnitude:
                fail = candidate_fail
        else:
            costs.append((name, magnitude))

    if fail is not None:
        return fail

    if not cessation_improved:
        return NoTradeNoImprovement(
            baseline_label, candidate_label, n_cessation, cessation_delta
        )

    # Day 31, Objective 1: calibration improved -- but did the candidate
    # keep ESTIMATING? Checked before any pass is issued, and after the
    # overconfidence check, which stays non-negotiable and first.
    collapses: list[tuple[str, float, float]] = []
    for name in (*NO_TRADE_STEADY_REGIMES, NO_TRADE_CESSATION_REGIME):
        base_margin = _informativeness_margin(baseline_regimes.get(name, {}))
        cand_margin = _informativeness_margin(candidate_regimes.get(name, {}))
        if np.isnan(base_margin) or np.isnan(cand_margin):
            continue
        base_informative = base_margin > UNINFORMATIVE_MARGIN_NATS
        cand_informative = cand_margin > UNINFORMATIVE_MARGIN_NATS
        if base_informative and not cand_informative:
            collapses.append((name, base_margin, cand_margin))
    if collapses:
        worst = max(collapses, key=lambda item: item[1] - item[2])
        return NoTradeUninformative(
            baseline=baseline_label,
            candidate=candidate_label,
            regime=worst[0],
            baseline_margin_nats=worst[1],
            candidate_margin_nats=worst[2],
            cessation_delta_toward_nominal=cessation_delta,
            n_cessation=n_cessation,
        )

    if costs:
        worst_regime, worst_magnitude = max(costs, key=lambda item: item[1])
        return NoTradePassWithCost(
            baseline=baseline_label,
            candidate=candidate_label,
            regime=worst_regime,
            magnitude=worst_magnitude,
            costs=tuple(costs),
            cessation_delta_toward_nominal=cessation_delta,
            n_cessation=n_cessation,
        )

    return NoTradePass(
        baseline=baseline_label,
        candidate=candidate_label,
        cessation_delta_toward_nominal=cessation_delta,
        n_cessation=n_cessation,
    )


def _print_no_trade_verdict(version: str, verdict: NoTradeVerdict) -> None:
    print(f"  {verdict.baseline} -> {verdict.candidate} ({version}):")
    if isinstance(verdict, NoTradeUnscoreable):
        print(
            f"    UNSCOREABLE -- cessation has only {verdict.n_cessation} frame(s) "
            f"in common (< {MIN_REGIME_FRAMES_FOR_A_CONCLUSION} floor); no conclusion "
            "can be drawn about the regime this criterion is actually about"
        )
        return
    if isinstance(verdict, NoTradeNoImprovement):
        print(
            f"    cessation delta-toward-nominal: "
            f"{verdict.cessation_delta_toward_nominal:+.4f} (n={verdict.n_cessation}, "
            "not improved -- no benefit here to weigh a cost against)"
        )
        print("    NO-TRADE CRITERION: NO_IMPROVEMENT")
        return
    if isinstance(verdict, NoTradeFailOverconfident):
        print(
            f"    {verdict.regime} moved TOWARD OVERCONFIDENCE: coverage "
            f"{verdict.coverage_before:.4f} -> {verdict.coverage_after:.4f} "
            f"(magnitude {verdict.magnitude:.4f}, below nominal "
            f"{NOMINAL_COVERAGE:.2f}) -- non-negotiable"
        )
        print("    NO-TRADE CRITERION: FAIL_OVERCONFIDENT")
        return
    if isinstance(verdict, NoTradeUninformative):
        print(
            f"    cessation delta-toward-nominal: "
            f"{verdict.cessation_delta_toward_nominal:+.4f} "
            f"(n={verdict.n_cessation}, IMPROVED)"
        )
        print(
            f"    ...but {verdict.regime} informativeness COLLAPSED: "
            f"{verdict.baseline_margin_nats:+.4f} -> "
            f"{verdict.candidate_margin_nats:+.4f} nats over a fitted "
            f"constant (threshold {UNINFORMATIVE_MARGIN_NATS:+.4f})"
        )
        print("    NO-TRADE CRITERION: UNINFORMATIVE")
        return
    # NoTradePass / NoTradePassWithCost both improved cessation materially.
    print(
        f"    cessation delta-toward-nominal: "
        f"{verdict.cessation_delta_toward_nominal:+.4f} "
        f"(n={verdict.n_cessation}, IMPROVED)"
    )
    if isinstance(verdict, NoTradePassWithCost):
        for name, magnitude in verdict.costs:
            print(
                f"    {name} moved toward UNDERCONFIDENCE by {magnitude:.4f} "
                "(cost, not a failure)"
            )
        print(
            f"    NO-TRADE CRITERION: PASS_WITH_COST "
            f"(worst: {verdict.regime}, magnitude {verdict.magnitude:.4f})"
        )
    else:
        print("    NO-TRADE CRITERION: PASS")


def _print_four_way_summary(
    version: str, per_config: dict[str, dict[str, Any]]
) -> None:
    """Objective 3: all scored configs side by side, per regime -- position
    RMSE, coverage (mixture-valid where applicable), NEES/NIS pass rate,
    both trivial-baseline margins, and the frame count every number here
    is conditioned on."""
    print("=" * 78)
    print(f"FOUR-WAY SUMMARY, per regime: {version}")
    print("=" * 78)
    labels = [label for label in ("A", "B", "C", "D") if label in per_config]
    for regime_name in MOTION_REGIMES:
        print(f"  {regime_name}:")
        any_frames = False
        for label in labels:
            block = per_config[label].get("by_regime", {}).get(regime_name, {})
            n = block.get("n", 0)
            if n == 0:
                print(f"    {label}: n=0 (empty)")
                continue
            any_frames = True
            coverage = _regime_coverage(block)
            nees_pass = block.get("consistency", {}).get(
                "nees_pass_rate_within_95", float("nan")
            )
            thin = " [THIN EVIDENCE]" if block.get("thin_evidence") else ""
            # Day 31: coverage never prints without its informativeness
            # margin beside it. See CalibrationAndInformativeness.
            margin_nats = _informativeness_margin(block)
            informative = (
                ""
                if np.isnan(margin_nats)
                else (
                    "" if margin_nats > UNINFORMATIVE_MARGIN_NATS else " UNINFORMATIVE"
                )
            )
            print(
                f"    {label}: n={n:5}  RMSE {block['filter_rmse_m']:.4f} m  "
                f"coverage {coverage:.4f}  informativeness {margin_nats:+.4f} nats"
                f"{informative}  NEES/NIS pass {nees_pass:.4f}  "
                f"margin(copy-prev) {block['margin_vs_copy_previous_m']:+.4f}  "
                f"margin(CV-dead-reckon) {block['margin_vs_constant_velocity_m']:+.4f}"
                f"{thin}"
            )
        if not any_frames:
            print("    (empty in every config)")
    print()


def _nan_to_none(value: Any) -> Any:
    """NaN is not valid JSON (Day 18: ``src.data.scorecard``'s ``Undefined``
    exists for exactly this reason). An empty distance bucket or an
    unscoreable margin produces a real NaN internally -- the console report
    prints it as "empty"/"n/a" contextually, but the JSON artifact must
    never carry the literal token, so this walks the report tree right
    before serialization and swaps every NaN float for ``null``."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        action="append",
        default=None,
        help="golden-set version (repeatable)",
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--artifact-dir", default=None, type=Path)
    parser.add_argument(
        "--config",
        action="append",
        choices=sorted(CONFIG_SPECS),
        default=None,
        help=(
            "which config(s) to score, repeatable (A=single model, "
            "B=single model+floor, C=IMM, D=IMM+floor); default: all four"
        ),
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = (
        Path(args.root)
        if args.root
        else config.paths.resolve(config.eval.golden_sets_dir)
    )
    versions = args.version or ["v3-indoor", "v4.1-gate"]
    labels = args.config or sorted(CONFIG_SPECS)

    reports = []
    verdicts = []
    any_refused_or_failed = False
    for version in versions:
        per_config: dict[str, dict[str, Any]] = {}
        for label in labels:
            filter_kind, floor = CONFIG_SPECS[label]
            report = _score_golden_set(
                version,
                root,
                config,
                filter_kind=filter_kind,
                velocity_covariance_floor=floor,
            )
            if report is None:
                print(
                    f"Available: {available_versions(root) or 'none'}", file=sys.stderr
                )
                any_refused_or_failed = True
                continue
            report["config"] = label
            reports.append(report)
            per_config[label] = report
            print(f"[config {label}: {CONFIG_DESCRIPTIONS[label]}]")
            _print_report(report)
            print()
            if report.get("refused"):
                any_refused_or_failed = True

        if len(per_config) > 1:
            _print_four_way_summary(version, per_config)

        if "A" in per_config:
            print("=" * 78)
            print(f"NO-TRADE VERDICTS vs config A, {version}")
            print("=" * 78)
            for candidate_label in ("B", "C", "D"):
                if candidate_label not in per_config:
                    continue
                verdict = _no_trade_verdict(
                    "A", per_config["A"], candidate_label, per_config[candidate_label]
                )
                _print_no_trade_verdict(version, verdict)
                verdicts.append(
                    {
                        "version": version,
                        "kind": type(verdict).__name__,
                        **dataclasses.asdict(verdict),
                    }
                )
            print()

    if args.artifact_dir:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        out = args.artifact_dir / "estimator_accuracy.json"
        # allow_nan=False is the structural backstop: if _nan_to_none ever
        # misses a spot, this raises instead of silently writing invalid
        # JSON, the same way scorecard.py refuses a bare NaN at emission.
        payload = json.dumps(
            _nan_to_none({"reports": reports, "no_trade_verdicts": verdicts}),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        out.write_text(payload + "\n")
        print(f"Written: {out}")

    return 1 if any_refused_or_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
