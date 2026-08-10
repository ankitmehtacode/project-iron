"""Score the single-entity state estimator against exact synthetic GT.

Day 20's hard scope rule: **no timing, throughput, CPU-percentage, or
latency claim is made anywhere in this script.** Everything below is
accuracy and consistency — position/velocity error against v3-indoor's and
v4.1-gate's analytic ``agent_xyz`` ground truth, and the NIS/NEES
consistency residuals src/estimator's filter already computes. If a
benchmark needs to run at all, it goes through Day 19's environment gate
(``src.bench.environment``) and is expected to refuse on this machine —
see ``docs/reference_hardware.md``.

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
from typing import Any

import numpy as np
import numpy.typing as npt

from src.config import IronConfig
from src.contracts.frames import AffineTransform, FrameGeometry
from src.data import validity
from src.data.depth_eval import DISTANCE_BUCKETS
from src.data.golden import GoldenSetError, available_versions, load_golden_set
from src.eval.baselines import compute_baselines, margin, require_baseline
from src.estimator.consistency import compute_nees, fraction_outside_bound
from src.estimator.filter import run_single_entity_filter
from src.estimator.measurement_model import MeasurementModel, measurement_model_for
from src.estimator.motion_model import MotionModel, motion_model_for
from src.estimator.state import ConsistencyResidual
from src.model.episode import StateGraph, StateQuery, solve_state
from src.model.frame_of_reference import FrameOfReference
from src.model.measurement import WorldPositionMeasurement
from src.model.observation import Observation
from src.model.uncertainty import Uncertainty
from src.model.ulid import generate_ulid

FloatArray = npt.NDArray[np.float64]

EVAL_SEED = 20260731
"""Pinned seed for synthesized observation noise -- same convention as
scripts/cascade_bench.py's --seed default."""

BASE_TS_NS = 1_785_000_000 * 1_000_000_000
FIRST_COMPARABLE_INDEX = 2
"""See module docstring: the first frame index at which all three compared
methods (filter, copy_previous, constant_velocity_no_update) have a
genuine, non-circular answer."""


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
class TrackResult:
    """One agent-track's contribution to a golden set's evaluation."""

    position_sq_errors: list[float] = field(default_factory=list)
    per_axis_sq_errors: list[FloatArray] = field(default_factory=list)
    velocity_sq_errors: list[float] = field(default_factory=list)
    distances_m: list[float] = field(default_factory=list)
    nees_residuals: list[ConsistencyResidual] = field(default_factory=list)
    copy_previous_sq_errors: list[float] = field(default_factory=list)
    cv_no_update_sq_errors: list[float] = field(default_factory=list)
    cv_no_update_velocity_sq_errors: list[float] = field(default_factory=list)


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

    obs_xyz = np.array([_xyz(o) for o in observations])
    cv_velocity = (obs_xyz[1] - obs_xyz[0]) / dt_s
    cv_position = obs_xyz[1].copy()

    result = TrackResult()
    for t in range(1, frames):
        # Advance the coasting baseline every step from t=1 onward so its
        # state at FIRST_COMPARABLE_INDEX reflects genuine free-running
        # dead reckoning, not a lucky first step.
        if t >= 2:
            cv_position = cv_position + cv_velocity * dt_s

        if t < FIRST_COMPARABLE_INDEX:
            continue

        query = StateQuery(
            at_ts_ns=observations[t].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
        )
        estimate = solve_state(query, graph)

        gt_pos = track_xyz[t]
        pos_err = estimate.position_m() - gt_pos
        result.position_sq_errors.append(float(np.dot(pos_err, pos_err)))
        result.per_axis_sq_errors.append(pos_err**2)

        vel_err = estimate.velocity_mps() - gt_velocity[t]
        result.velocity_sq_errors.append(float(np.dot(vel_err, vel_err)))

        result.distances_m.append(_camera_depth_m(gt_pos, extrinsics))

        full_gt = np.concatenate([gt_pos, gt_velocity[t]])
        error6 = estimate.mean_array() - full_gt
        result.nees_residuals.append(compute_nees(error6, estimate.cov_array()))

        prev_obs = obs_xyz[t - 1]
        copy_previous_err = prev_obs - gt_pos
        result.copy_previous_sq_errors.append(
            float(np.dot(copy_previous_err, copy_previous_err))
        )

        cv_err = cv_position - gt_pos
        result.cv_no_update_sq_errors.append(float(np.dot(cv_err, cv_err)))
        cv_vel_err = cv_velocity - gt_velocity[t]
        result.cv_no_update_velocity_sq_errors.append(
            float(np.dot(cv_vel_err, cv_vel_err))
        )

    return result


def _rmse(sq_errors: list[float]) -> float:
    return float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")


def _score_golden_set(
    version: str, root: Path, config: IronConfig
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

    motion_model = motion_model_for("person")
    measurement_model = measurement_model_for(
        sensor=version, capability="state_estimation"
    )
    rng = np.random.default_rng(EVAL_SEED)

    tracks_scored = 0
    clips_with_no_agents = 0
    bucket_sq_errors: dict[str, list[float]] = {
        name: [] for name, _, _ in DISTANCE_BUCKETS
    }
    bucket_axis_sq_errors: dict[str, list[FloatArray]] = {
        name: [] for name, _, _ in DISTANCE_BUCKETS
    }
    all_position_sq: list[float] = []
    all_axis_sq: list[FloatArray] = []
    all_velocity_sq: list[float] = []
    all_nees: list[ConsistencyResidual] = []
    all_copy_previous_sq: list[float] = []
    all_cv_no_update_sq: list[float] = []
    all_cv_no_update_velocity_sq: list[float] = []

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
            all_position_sq.extend(result.position_sq_errors)
            all_axis_sq.extend(result.per_axis_sq_errors)
            all_velocity_sq.extend(result.velocity_sq_errors)
            all_nees.extend(result.nees_residuals)
            all_copy_previous_sq.extend(result.copy_previous_sq_errors)
            all_cv_no_update_sq.extend(result.cv_no_update_sq_errors)
            all_cv_no_update_velocity_sq.extend(result.cv_no_update_velocity_sq_errors)

            for sq, distance in zip(result.position_sq_errors, result.distances_m):
                for name, low, high in DISTANCE_BUCKETS:
                    if low <= distance < high:
                        bucket_sq_errors[name].append(sq)
            for axis_sq, distance in zip(result.per_axis_sq_errors, result.distances_m):
                for name, low, high in DISTANCE_BUCKETS:
                    if low <= distance < high:
                        bucket_axis_sq_errors[name].append(axis_sq)

    if tracks_scored == 0:
        return {
            "version": version,
            "refused": False,
            "tracks_scored": 0,
            "clips_with_no_agents": clips_with_no_agents,
            "note": "no agent track in this set had enough frames to evaluate",
        }

    position_rmse = _rmse(all_position_sq)
    velocity_rmse = _rmse(all_velocity_sq)
    copy_previous_rmse = _rmse(all_copy_previous_sq)
    cv_no_update_rmse = _rmse(all_cv_no_update_sq)
    cv_no_update_velocity_rmse = _rmse(all_cv_no_update_velocity_sq)

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

    per_axis = np.array(all_axis_sq) if all_axis_sq else np.zeros((0, 3))
    axis_rmse = (
        np.sqrt(per_axis.mean(axis=0)).tolist() if per_axis.size else [float("nan")] * 3
    )

    by_bucket = {}
    for name, _, _ in DISTANCE_BUCKETS:
        axis_arr = (
            np.array(bucket_axis_sq_errors[name])
            if bucket_axis_sq_errors[name]
            else None
        )
        by_bucket[name] = {
            "n": len(bucket_sq_errors[name]),
            "position_rmse_m": _rmse(bucket_sq_errors[name]),
            "axis_rmse_m": (
                np.sqrt(axis_arr.mean(axis=0)).tolist()
                if axis_arr is not None and axis_arr.size
                else [float("nan")] * 3
            ),
        }

    nees_within = [r.within_bound for r in all_nees if r.within_bound is not None]
    fraction_outside_95 = fraction_outside_bound(all_nees, "nees")
    empirical_coverage_95 = (
        1.0 - fraction_outside_95 if not np.isnan(fraction_outside_95) else float("nan")
    )
    nees_pass_rate = float(np.mean(nees_within)) if nees_within else float("nan")

    return {
        "version": version,
        "refused": False,
        "tracks_scored": tracks_scored,
        "clips_with_no_agents": clips_with_no_agents,
        "points_scored": len(all_position_sq),
        "motion_model_sha": motion_model.sha,
        "measurement_model_sha": measurement_model.sha,
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
        "by_distance_bucket": by_bucket,
        "consistency": {
            "nees_pass_rate_within_95": nees_pass_rate,
            "empirical_coverage_95": empirical_coverage_95,
            "n_nees_scored": len(nees_within),
        },
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
    print("by GT distance bucket:")
    for name, bucket in report["by_distance_bucket"].items():
        if bucket["n"] == 0:
            print(f"    {name:8} n=0 (empty)")
            continue
        axis = bucket["axis_rmse_m"]
        print(
            f"    {name:8} n={bucket['n']:5}  "
            f"position RMSE {bucket['position_rmse_m']:.4f} m  "
            f"(x/y/z: {axis[0]:.4f}/{axis[1]:.4f}/{axis[2]:.4f})"
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
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = (
        Path(args.root)
        if args.root
        else config.paths.resolve(config.eval.golden_sets_dir)
    )
    versions = args.version or ["v3-indoor", "v4.1-gate"]

    reports = []
    any_refused_or_failed = False
    for version in versions:
        report = _score_golden_set(version, root, config)
        if report is None:
            print(f"Available: {available_versions(root) or 'none'}", file=sys.stderr)
            any_refused_or_failed = True
            continue
        reports.append(report)
        _print_report(report)
        print()
        if report.get("refused"):
            any_refused_or_failed = True

    if args.artifact_dir:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        out = args.artifact_dir / "estimator_accuracy.json"
        # allow_nan=False is the structural backstop: if _nan_to_none ever
        # misses a spot, this raises instead of silently writing invalid
        # JSON, the same way scorecard.py refuses a bare NaN at emission.
        payload = json.dumps(
            _nan_to_none(reports), indent=2, sort_keys=True, allow_nan=False
        )
        out.write_text(payload + "\n")
        print(f"Written: {out}")

    return 1 if any_refused_or_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
