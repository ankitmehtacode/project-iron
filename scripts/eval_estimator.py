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
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from src.config import IronConfig
from src.contracts.frames import AffineTransform, FrameGeometry
from src.data import validity
from src.data.depth_eval import DISTANCE_BUCKETS
from src.data.golden import GoldenSetError, available_versions, load_golden_set
from src.eval.baselines import compute_baselines, margin, require_baseline
from src.estimator.consistency import (
    compute_collapsed_gaussian_nees_diagnostic,
    compute_mixture_nees,
    compute_nees,
    empirical_coverage_by_sampling,
    fraction_outside_bound,
)
from src.estimator.diagnostics import is_white
from src.estimator.filter import run_single_entity_filter
from src.estimator.imm import ImmConfig, default_imm_config, run_imm_filter
from src.estimator.measurement_model import MeasurementModel, measurement_model_for
from src.estimator.motion_model import MotionModel, motion_model_for
from src.estimator.regime import MOTION_REGIMES, MotionRegime, classify_track
from src.estimator.state import ConsistencyResidual
from src.model.episode import StateGraph, StateQuery, solve_state
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.uncertainty import Uncertainty
from src.model.ulid import generate_ulid

FilterKind = Literal["single_model", "imm"]

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


def _frame_of_reference(twin_rev: int = 1) -> FrameOfReference:
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
                standardized_innovation_x=standardized_innovation_x,
                copy_previous_sq_error=float(
                    np.dot(copy_previous_err, copy_previous_err)
                ),
                cv_no_update_sq_error=float(np.dot(cv_err, cv_err)),
                cv_no_update_velocity_sq_error=float(np.dot(cv_vel_err, cv_vel_err)),
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
                standardized_innovation_x=None,
                copy_previous_sq_error=float(
                    np.dot(copy_previous_err, copy_previous_err)
                ),
                cv_no_update_sq_error=float(np.dot(cv_err, cv_err)),
                cv_no_update_velocity_sq_error=float(np.dot(cv_vel_err, cv_vel_err)),
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
    sampled = [r.covered_by_sampling for r in records if r.covered_by_sampling is not None]
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
        imm_config = default_imm_config()
    motion_model = motion_model_for("person")
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
            track = agent_xyz[:, agent_index, :]
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
        block["consistency"] = _coverage_stats(regime_records)
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
        f"{pos['axis_rmse_m_xyz'][0]:.4f}/{pos['axis_rmse_m_xyz'][1]:.4f}/"
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
        print(
            f"    {name:12} n={block['n']:5}  "
            f"filter {block['filter_rmse_m']:.4f} m  "
            f"copy-prev margin {block['margin_vs_copy_previous_m']:+.4f}  "
            f"const-vel margin {block['margin_vs_constant_velocity_m']:+.4f}{thin}"
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


NO_TRADE_TRANSIENT_REGIMES = ("onset", "cessation", "maneuver")
NO_TRADE_STEADY_REGIMES = ("static", "sustained")
NO_TRADE_MATERIAL_IMPROVEMENT = 0.05
"""A transient regime's empirical coverage must close its gap to the 0.95
nominal by at least this much (in coverage points) for IMM to count as
having "moved materially toward nominal" there. Declared, not fitted."""
NO_TRADE_DEGRADATION_TOLERANCE = 0.02
"""A steady regime's coverage may move at most this much FURTHER from
nominal under IMM before it counts as "degraded" -- allows for sampling
noise between two runs on the same (identical, same-seed) observations
without calling every microscopic wiggle a regression."""


def _print_comparison(
    single_report: dict[str, Any], imm_report: dict[str, Any]
) -> dict[str, Any]:
    """Objective 4: single-model vs IMM, per regime, plus the no-trade verdict.

    Returns the verdict as data (not just printed text) since Objective 6's
    report needs to cite it precisely.
    """
    version = single_report["version"]
    print("=" * 78)
    print(f"SINGLE-MODEL vs IMM, per regime: {version}")
    print("=" * 78)

    verdict: dict[str, Any] = {"version": version, "regimes": {}}
    any_transient_improved = False
    any_steady_degraded = False

    for name in MOTION_REGIMES:
        single_block = single_report.get("by_regime", {}).get(name, {})
        imm_block = imm_report.get("by_regime", {}).get(name, {})
        n_single = single_block.get("n", 0)
        n_imm = imm_block.get("n", 0)
        if n_single == 0 and n_imm == 0:
            print(f"  {name:12} n=0 in both (empty)")
            continue

        single_cov = single_block.get("consistency", {}).get(
            "empirical_coverage_95", float("nan")
        )
        # Day 22, Objective 1: IMM's coverage figure here is the
        # mixture-VALID sampling-based one (compare against a
        # single-Gaussian's chi-square-based coverage on the single-model
        # side, since that posterior genuinely is Gaussian) -- not the
        # collapsed-Gaussian diagnostic Day 21 used, which is retained
        # elsewhere in the report but not used to drive this verdict.
        imm_cov = imm_block.get("consistency_mixture", {}).get(
            "empirical_coverage_by_sampling_95", float("nan")
        )
        single_dev = (
            abs(single_cov - 0.95) if not np.isnan(single_cov) else float("nan")
        )
        imm_dev = abs(imm_cov - 0.95) if not np.isnan(imm_cov) else float("nan")
        delta = (
            single_dev - imm_dev
            if not (np.isnan(single_dev) or np.isnan(imm_dev))
            else float("nan")
        )  # positive = IMM closer to nominal than single-model was

        regime_kind = "transient" if name in NO_TRADE_TRANSIENT_REGIMES else "steady"
        print(
            f"  {name:12} n={n_single:5}->{n_imm:<5}  "
            f"coverage: single {single_cov:.4f} -> IMM {imm_cov:.4f}  "
            f"(delta-toward-nominal {delta:+.4f})  [{regime_kind}]"
        )
        single_pos = single_block.get("filter_rmse_m", float("nan"))
        imm_pos = imm_block.get("filter_rmse_m", float("nan"))
        print(
            f"               position RMSE: single {single_pos:.4f} m -> "
            f"IMM {imm_pos:.4f} m"
        )

        verdict["regimes"][name] = {
            "n_single": n_single,
            "n_imm": n_imm,
            "single_coverage": single_cov,
            "imm_coverage": imm_cov,
            "delta_toward_nominal": delta,
            "kind": regime_kind,
            "single_position_rmse_m": single_pos,
            "imm_position_rmse_m": imm_pos,
        }

        if (
            regime_kind == "transient"
            and not np.isnan(delta)
            and delta >= NO_TRADE_MATERIAL_IMPROVEMENT
        ):
            any_transient_improved = True
        if (
            regime_kind == "steady"
            and not np.isnan(delta)
            and delta < -NO_TRADE_DEGRADATION_TOLERANCE
        ):
            any_steady_degraded = True
            print(
                f"    ** REGRESSION: {name} moved {abs(delta):.4f} FURTHER "
                "from nominal coverage under IMM **"
            )

    no_trade_satisfied = any_transient_improved and not any_steady_degraded
    verdict["any_transient_improved"] = any_transient_improved
    verdict["any_steady_degraded"] = any_steady_degraded
    verdict["no_trade_satisfied"] = no_trade_satisfied

    print()
    if no_trade_satisfied:
        print(
            "NO-TRADE CRITERION: SATISFIED -- a transient regime improved "
            "materially toward nominal coverage, and no steady regime degraded."
        )
    else:
        print("NO-TRADE CRITERION: NOT SATISFIED.")
        if not any_transient_improved:
            print(
                "  - no transient regime (onset/cessation/maneuver) improved "
                "materially toward nominal coverage"
            )
        if any_steady_degraded:
            print("  - at least one steady regime (static/sustained) degraded")
    print()
    return verdict


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
        "--filter",
        choices=("single_model", "imm", "both"),
        default="both",
        help="which filter(s) to score; 'both' also prints the per-regime comparison",
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = (
        Path(args.root)
        if args.root
        else config.paths.resolve(config.eval.golden_sets_dir)
    )
    versions = args.version or ["v3-indoor", "v4.1-gate"]
    if args.filter == "both":
        kinds: list[FilterKind] = ["single_model", "imm"]
    else:
        kinds = [args.filter]

    reports = []
    comparisons = []
    any_refused_or_failed = False
    for version in versions:
        per_kind: dict[str, dict[str, Any]] = {}
        for kind in kinds:
            report = _score_golden_set(version, root, config, filter_kind=kind)
            if report is None:
                print(
                    f"Available: {available_versions(root) or 'none'}", file=sys.stderr
                )
                any_refused_or_failed = True
                continue
            reports.append(report)
            per_kind[kind] = report
            _print_report(report)
            print()
            if report.get("refused"):
                any_refused_or_failed = True

        if "single_model" in per_kind and "imm" in per_kind:
            verdict = _print_comparison(per_kind["single_model"], per_kind["imm"])
            comparisons.append(verdict)

    if args.artifact_dir:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        out = args.artifact_dir / "estimator_accuracy.json"
        # allow_nan=False is the structural backstop: if _nan_to_none ever
        # misses a spot, this raises instead of silently writing invalid
        # JSON, the same way scorecard.py refuses a bare NaN at emission.
        payload = json.dumps(
            _nan_to_none({"reports": reports, "comparisons": comparisons}),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        out.write_text(payload + "\n")
        print(f"Written: {out}")

    return 1 if any_refused_or_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
