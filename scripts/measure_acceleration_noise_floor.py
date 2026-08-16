"""Characterize the causal acceleration estimator's noise floor (Day 29,
Objective 1) -- the measurement Day 28 named but did not run.

HARD SCOPE RULE, unchanged: no timing, throughput, CPU, or latency claim is
made anywhere in this script. Everything below is accuracy/noise-floor
measurement against exact synthetic GT.

Why this script exists
-----------------------
Day 28 found the two-term slip model failed its acceptance criterion and
traced the cause to `|a_hat|` -- the carrier acceleration estimate that
feeds the acceleration-scaled slip term -- reading 2-7x
`PERSON_SIGMA_A_MPS2` even during GT-labeled STEADY motion, on a five-point
spot check of one v5-cessation track. This script replaces that spot check
with a full-population, instrumented measurement (Day 25's rule: instrument
the running estimator, do not compute this from a closed-form
error-propagation argument) across both golden sets: bias, standard
deviation, and noise floor per GT regime and per frame rate, the true
carrier/asset relative-acceleration signal from GT, the resulting SNR, and
an ablation-based attribution of where the noise floor's magnitude comes
from (process-noise inflation vs. measurement noise propagated through the
filter).

`a_hat` replicated exactly
----------------------------
`src.estimator.joint.run_joint_filter`'s causal formula, reproduced here
bit-for-bit against a plain `run_single_entity_filter` carrier graph (the
same carrier config `scripts/eval_joint_estimator.py` scores as
"carrier_independent" -- `motion_model_for("person",
velocity_covariance_floor=True)`, Day 25's adopted config B):

    a_hat[t] = || v_est[t-1] - v_est[t-2] || / dt_s      (t >= 2)

`v_est[k]` is the filter's POSTERIOR (mean) velocity at step k, read back
from a real `StateGraph` via `resolve_state` -- never recomputed from a
closed-form propagation formula. `a_hat[t]` is what informs the process
noise for the predict step t-1 -> t (one-step causal lag, matching
`OffsetSlipModel`'s own docstring). `a_hat[0]` and `a_hat[1]` are the
filter's own degenerate bootstrap values (always exactly 0.0, both velocity
readings equal the zero-velocity bootstrap state) and are excluded from
every statistic below, the same way `FIRST_COMPARABLE_INDEX=2` already
excludes them from `scripts/eval_estimator.py`'s own accuracy numbers.

GT acceleration, causally aligned
------------------------------------
`gt_accel[k] = (gt_velocity[k] - gt_velocity[k-1]) / dt_s` for k >= 1
(`gt_velocity` itself finite-differenced from GT position, same convention
as `src.estimator.regime.classify_track` and `scripts/eval_estimator.py`).
`a_hat[t]` is compared against `gt_accel[t-1]` -- the TRUE acceleration
during the same (t-2, t-1) interval `a_hat[t]`'s own finite difference
spans -- not `gt_accel[t]`, which would compare a causal, lagged estimate
against a acceleration interval it could not have seen yet.

The SNR this reports
-----------------------
Since every golden set's carried-asset GT is a RIGID offset (Day 26
Objective 4's own synthesis: no synthetic slip in GT), the "true relative
acceleration between carrier and asset" the slip model would need to
resolve is, by construction, exactly the carrier's own GT acceleration --
any asset slip a real system would need to detect is caused BY a change in
carrier motion, and there is no other source of relative acceleration in
this project's synthetic GT. The declared reference scale the slip model
already uses for this (`PERSON_SIGMA_A_MPS2`) is reused here as the "true
signal" scale, exactly per Day 28's own framing: this constant is "a bound
on plausible TRUE human acceleration," never validated as a bound on this
estimator's own noise floor -- this script is that validation.
`snr = PERSON_SIGMA_A_MPS2 / noise_floor_std`, where `noise_floor_std` is
the sample standard deviation of `a_hat` restricted to frames GT labels
`static` or `sustained` (GT acceleration is at or near zero there, so any
`a_hat` reading is, definitionally, noise, not signal).

Attribution ablation
------------------------
Three carrier configs, same observations (same RNG draw, so the input
noise stream is identical across configs -- only the FILTER differs):

  - ``baseline``: the real, currently-used config (Q = process noise at
    `PERSON_SIGMA_A_MPS2`, R = the real measurement model, floor enabled)
    -- the config `a_hat`'s production noise floor is measured against
    above.
  - ``q_near_zero``: same R, `sigma_a_mps2` collapsed to a small positive
    floor (must stay `> 0` -- see `ConstantVelocityMotionModel.
    __post_init__`) -- isolates how much of the noise floor is
    PROCESS-NOISE inflation (a high-Q filter trusts each new noisy
    observation more, so the posterior velocity MEAN chases measurement
    noise harder step to step).
  - ``r_near_zero``: same Q, measurement sigma collapsed to a small
    positive floor via a synthetic near-noiseless envelope -- isolates how
    much of the noise floor is MEASUREMENT noise propagated through the
    filter (even at fixed process noise, position observations still carry
    the same sigma-scaled draw, and every draw perturbs the estimated
    velocity mean by some amount before any Kalman-gain damping).

Both ablations report `a_hat`'s noise floor (steady-regime std) under the
modified config; a large drop relative to baseline attributes a large share
of the noise floor to that source. Neither ablation is a candidate fix --
Day 29 Objective 2 evaluates named candidates separately; this script is
diagnostic only.

    python scripts/measure_acceleration_noise_floor.py
    python scripts/measure_acceleration_noise_floor.py --version v5-cessation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_estimator as ee  # noqa: E402

from src.config import IronConfig  # noqa: E402
from src.data import validity  # noqa: E402
from src.contracts.ground_truth import (  # noqa: E402
    GENERATOR_AXES,
    gt_position_track,
)
from src.data.golden import GoldenSetError, load_golden_set  # noqa: E402
from src.estimator.filter import run_single_entity_filter  # noqa: E402
from src.estimator.measurement_model import MeasurementModel  # noqa: E402
from src.estimator.motion_model import (  # noqa: E402
    PERSON_SIGMA_A_MPS2,
    ConstantVelocityMotionModel,
    MotionModel,
    motion_model_for,
)
from src.estimator.regime import MOTION_REGIMES, classify_track  # noqa: E402
from src.model.envelope import Envelope, EnvelopeCurve  # noqa: E402
from src.model.episode import StateGraph, StateQuery, solve_state  # noqa: E402

FloatArray = npt.NDArray[np.float64]

EVAL_SEED = ee.EVAL_SEED
FIRST_ACCEL_INDEX = 2
"""a_hat[0] and a_hat[1] are the filter's own degenerate bootstrap values
(both readings equal the zero-velocity bootstrap state) -- see module
docstring. Same exclusion convention as ee.FIRST_COMPARABLE_INDEX."""

STEADY_REGIMES = ("static", "sustained")
"""GT acceleration is at or near zero here by definition -- any a_hat
reading in these frames is noise, not signal (see module docstring's SNR
section)."""

NEAR_ZERO_SIGMA_A_MPS2 = 1e-4
"""Ablation floor for process noise -- must stay > 0
(ConstantVelocityMotionModel.__post_init__ rejects <= 0), chosen four
orders of magnitude below PERSON_SIGMA_A_MPS2 so the ablation is close to
a true zero-process-noise filter without being non-positive-definite."""

NEAR_ZERO_MEASUREMENT_SIGMA_M = 1e-4
"""Ablation floor for measurement noise -- same reasoning, four orders of
magnitude below the illustrative envelope's tightest calibrated sigma
(0.03m at 1m range)."""


def _tiny_measurement_model(sensor: str) -> MeasurementModel:
    """A near-noiseless measurement model: same shape as
    ``measurement_model_for``'s illustrative envelope, sigma collapsed to
    :data:`NEAR_ZERO_MEASUREMENT_SIGMA_M` at every range. Ablation-only --
    never used for a real evaluation number, only to isolate R's
    contribution to a_hat's noise floor."""
    curve = EnvelopeCurve(
        independent_variable="range_m",
        points=((1.0, NEAR_ZERO_MEASUREMENT_SIGMA_M), (25.0, NEAR_ZERO_MEASUREMENT_SIGMA_M)),
    )
    envelope = Envelope(
        capability="state_estimation",
        camera_id=sensor,
        twin_rev=0,
        curve=curve,
        sample_count=2,
        manifest_sha="day29-ablation-near-zero-measurement-noise",
    )
    return MeasurementModel(envelope=envelope)


def _tiny_process_noise_motion_model() -> MotionModel:
    """Same kind ("person"), same velocity-covariance-floor behaviour as
    the real carrier config, sigma_a_mps2 collapsed to
    :data:`NEAR_ZERO_SIGMA_A_MPS2`. Ablation-only -- isolates Q's
    contribution to a_hat's noise floor."""
    return ConstantVelocityMotionModel(
        kind="person",
        sigma_a_mps2=NEAR_ZERO_SIGMA_A_MPS2,
        velocity_variance_floor_enabled=True,
    )


def _a_hat_series(
    track_xyz: FloatArray,
    extrinsics: FloatArray,
    sensor_id: str,
    fps: float,
    motion_model: MotionModel,
    measurement_model: MeasurementModel,
    rng: np.random.Generator,
) -> FloatArray | None:
    """Run the real filter over one track, return a_hat[t] for every t in
    range(len(track)) -- NaN at t=0,1 (degenerate/undefined), the
    instrumented causal estimate for t>=2. Reads the filter's POSTERIOR
    MEAN back via resolve_state; never a closed-form propagation."""
    frames = track_xyz.shape[0]
    if frames <= FIRST_ACCEL_INDEX:
        return None

    observations = ee._make_observations(
        track_xyz, extrinsics, measurement_model, fps, sensor_id, rng
    )
    camera_position = ee._camera_position_world(extrinsics)
    graph = StateGraph()
    run_single_entity_filter(
        graph,
        observations,
        motion_model,
        measurement_model,
        manifest_sha="scripts/measure_acceleration_noise_floor.py",
        sensor_origin_m=tuple(camera_position.tolist()),  # type: ignore[arg-type]
    )

    dt_s = 1.0 / fps
    velocities = np.full((frames, 3), np.nan, dtype=np.float64)
    for t in range(frames):
        query = StateQuery(
            at_ts_ns=observations[t].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
        )
        estimate = solve_state(query, graph)
        velocities[t] = estimate.velocity_mps()

    a_hat = np.full(frames, np.nan, dtype=np.float64)
    for t in range(FIRST_ACCEL_INDEX, frames):
        a_hat[t] = float(np.linalg.norm(velocities[t - 1] - velocities[t - 2]) / dt_s)
    return a_hat


def _gt_accel_series(track_xyz: FloatArray, dt_s: float) -> FloatArray:
    """gt_accel[k] = (gt_velocity[k] - gt_velocity[k-1]) / dt_s, k>=1;
    gt_accel[0] mirrors gt_accel[1] (no frame -1 to difference against),
    same convention regime.py/eval_estimator.py already use for GT
    velocity's own frame-0 edge case."""
    frames = track_xyz.shape[0]
    gt_velocity = np.zeros_like(track_xyz)
    gt_velocity[1:] = (track_xyz[1:] - track_xyz[:-1]) / dt_s
    gt_velocity[0] = gt_velocity[1]
    gt_accel = np.zeros(frames, dtype=np.float64)
    if frames >= 2:
        diff = (gt_velocity[1:] - gt_velocity[:-1]) / dt_s
        gt_accel[1:] = np.linalg.norm(diff, axis=1)
        gt_accel[0] = gt_accel[1]
    return gt_accel


def _closed_form_a_hat_std(
    track_xyz: FloatArray,
    extrinsics: FloatArray,
    sensor_id: str,
    fps: float,
    motion_model: MotionModel,
    measurement_model: MeasurementModel,
    rng: np.random.Generator,
) -> float:
    """Day 25's cross-check side: the filter's OWN reported steady-state
    velocity covariance, propagated through the (independence-assuming)
    closed-form variance-of-a-difference formula
    Var(a_hat) ~= 2 * mean(diag(Cov(v))) / dt^2 -- NOT a substitute for the
    instrumented number above, reported alongside it per the
    closed-form-vs-instrumented rule so any divergence is itself visible."""
    frames = track_xyz.shape[0]
    if frames <= FIRST_ACCEL_INDEX:
        return float("nan")
    observations = ee._make_observations(
        track_xyz, extrinsics, measurement_model, fps, sensor_id, rng
    )
    camera_position = ee._camera_position_world(extrinsics)
    graph = StateGraph()
    run_single_entity_filter(
        graph,
        observations,
        motion_model,
        measurement_model,
        manifest_sha="scripts/measure_acceleration_noise_floor.py",
        sensor_origin_m=tuple(camera_position.tolist()),  # type: ignore[arg-type]
    )
    dt_s = 1.0 / fps
    variances = []
    for t in range(frames):
        query = StateQuery(
            at_ts_ns=observations[t].ts_ns, horizon_ns=0, graph_rev=graph.graph_rev
        )
        estimate = solve_state(query, graph)
        variances.append(float(np.mean(np.diag(estimate.cov_array())[3:6])))
    mean_var = float(np.mean(variances)) if variances else float("nan")
    predicted_variance_a_hat = 2.0 * mean_var / (dt_s**2)
    return float(np.sqrt(predicted_variance_a_hat))


def _bias_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    arr = np.array(values, dtype=np.float64)
    return {"n": len(values), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _score_version(version: str, root: Path, config: IronConfig) -> dict[str, Any] | None:
    try:
        golden = load_golden_set(root, version)
    except GoldenSetError as exc:
        print(f"ERROR loading {version}: {exc}")
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
            return {"version": version, "refused": True}
        print(f"[Day-10 validity gate] {version}: PASS state_estimation")

    measurement_model = ee.measurement_model_for(sensor=version, capability="state_estimation")
    tiny_measurement_model = _tiny_measurement_model(version)
    baseline_motion_model = motion_model_for("person", velocity_covariance_floor=True)
    q_near_zero_motion_model = _tiny_process_noise_motion_model()

    records: list[dict[str, Any]] = []
    fps_seen: set[float] = set()
    closed_form_stds: list[float] = []
    ablation_records: dict[str, list[float]] = {
        "baseline": [],
        "q_near_zero": [],
        "r_near_zero": [],
    }
    tracks_scored = 0

    for clip in golden.clips:
        clip_path = clip_root / f"{clip.clip_id}.npz"
        if not clip_path.exists():
            continue
        with np.load(clip_path) as data:
            if "agent_xyz" not in data:
                continue
            agent_xyz = np.asarray(data["agent_xyz"], dtype=np.float64)
            extrinsics = np.asarray(data["extrinsics"], dtype=np.float64)
        fps = fps_by_clip.get(clip.clip_id, 12.0)
        fps_seen.add(fps)
        dt_s = 1.0 / fps
        n_agents = agent_xyz.shape[1]

        for agent_index in range(n_agents):
            # Day 30, Objective 4: the slice goes through the GT
            # contract so the axis convention is DECLARED at the boundary
            # rather than assumed by every consumer downstream. This
            # script only takes magnitudes, which are permutation
            # invariant, so no number here changes -- the point is that a
            # future edit that reaches for a component cannot pick the
            # wrong index silently.
            track = gt_position_track(
                agent_xyz[:, agent_index, :], GENERATOR_AXES
            ).values
            if track.shape[0] <= FIRST_ACCEL_INDEX:
                continue

            regimes = classify_track(track, dt_s)
            gt_accel = _gt_accel_series(track, dt_s)

            rng = np.random.default_rng(EVAL_SEED)
            a_hat = _a_hat_series(
                track, extrinsics, f"{clip.clip_id}/{agent_index}", fps,
                baseline_motion_model, measurement_model, rng,
            )
            if a_hat is None:
                continue
            tracks_scored += 1

            for t in range(FIRST_ACCEL_INDEX, track.shape[0]):
                records.append(
                    {
                        "regime": regimes[t - 1],
                        "fps": fps,
                        "a_hat": float(a_hat[t]),
                        "gt_accel": float(gt_accel[t - 1]),
                    }
                )

            rng_cf = np.random.default_rng(EVAL_SEED)
            closed_form_stds.append(
                _closed_form_a_hat_std(
                    track, extrinsics, f"{clip.clip_id}/{agent_index}", fps,
                    baseline_motion_model, measurement_model, rng_cf,
                )
            )

            # Ablations, restricted to steady-regime frames for a direct
            # noise-floor comparison (see module docstring).
            rng_q = np.random.default_rng(EVAL_SEED)
            a_hat_q = _a_hat_series(
                track, extrinsics, f"{clip.clip_id}/{agent_index}", fps,
                q_near_zero_motion_model, measurement_model, rng_q,
            )
            rng_r = np.random.default_rng(EVAL_SEED)
            a_hat_r = _a_hat_series(
                track, extrinsics, f"{clip.clip_id}/{agent_index}", fps,
                baseline_motion_model, tiny_measurement_model, rng_r,
            )
            for t in range(FIRST_ACCEL_INDEX, track.shape[0]):
                if regimes[t - 1] not in STEADY_REGIMES:
                    continue
                ablation_records["baseline"].append(float(a_hat[t]))
                if a_hat_q is not None:
                    ablation_records["q_near_zero"].append(float(a_hat_q[t]))
                if a_hat_r is not None:
                    ablation_records["r_near_zero"].append(float(a_hat_r[t]))

    if tracks_scored == 0:
        return {"version": version, "refused": False, "tracks_scored": 0}

    by_regime: dict[str, Any] = {}
    for regime in MOTION_REGIMES:
        regime_records = [r for r in records if r["regime"] == regime]
        a_hat_stats = _bias_std([r["a_hat"] for r in regime_records])
        gt_stats = _bias_std([r["gt_accel"] for r in regime_records])
        bias = (
            a_hat_stats["mean"] - gt_stats["mean"]
            if regime_records
            else float("nan")
        )
        by_regime[regime] = {
            "n": len(regime_records),
            "a_hat_mean": a_hat_stats["mean"],
            "a_hat_std": a_hat_stats["std"],
            "gt_accel_mean": gt_stats["mean"],
            "bias_a_hat_minus_gt": bias,
        }

    steady_records = [r["a_hat"] for r in records if r["regime"] in STEADY_REGIMES]
    noise_floor_std = float(np.std(steady_records)) if steady_records else float("nan")
    snr = (
        PERSON_SIGMA_A_MPS2 / noise_floor_std
        if noise_floor_std and not np.isnan(noise_floor_std) and noise_floor_std > 0
        else float("nan")
    )

    # Per-regime SNR using each regime's own mean GT |accel| (by_regime,
    # above) as the signal -- NOT a single pooled median across
    # onset+cessation+maneuver. A pooled median across those three washes
    # out to ~0.0: "cessation" (Day 23's redefinition) mixes genuine
    # anticipatory-deceleration frames with post-stop RECOVERY frames whose
    # own GT acceleration is already back near zero, so the majority of
    # transient-labeled frames carry little signal even though the regime
    # exists because of the minority that do -- a median over the pooled
    # set is dominated by the near-zero majority and reports a signal scale
    # smaller than the regime's own mean shows. This was caught by
    # inspecting the first run's output (median=0.0000 despite by_regime
    # means of 0.5-13.3 m/s^2) rather than trusted -- see the Day-29 report.
    transient_regimes = ("onset", "cessation", "maneuver")
    transient_regime_snr = {
        regime: (
            by_regime[regime]["gt_accel_mean"] / noise_floor_std
            if by_regime[regime]["n"] > 0
            and noise_floor_std
            and not np.isnan(noise_floor_std)
            and noise_floor_std > 0
            else float("nan")
        )
        for regime in transient_regimes
    }

    ablation = {
        name: {
            "n": len(vals),
            "noise_floor_std": float(np.std(vals)) if vals else float("nan"),
        }
        for name, vals in ablation_records.items()
    }

    return {
        "version": version,
        "refused": False,
        "tracks_scored": tracks_scored,
        "fps_observed": sorted(fps_seen),
        "by_regime": by_regime,
        "noise_floor_std_steady_regimes": noise_floor_std,
        "snr_vs_person_sigma_a": snr,
        "transient_regime_snr": transient_regime_snr,
        "closed_form_a_hat_std_mean": (
            float(np.nanmean(closed_form_stds)) if closed_form_stds else float("nan")
        ),
        "ablation": ablation,
    }


def _print_report(report: dict[str, Any]) -> None:
    version = report["version"]
    print("=" * 78)
    print(f"ACCELERATION NOISE FLOOR: {version}")
    print("=" * 78)
    if report.get("refused") or report.get("tracks_scored", 0) == 0:
        print("No trajectory scored (refused or empty).")
        return

    print(f"tracks scored: {report['tracks_scored']}")
    print(f"fps observed in this set: {report['fps_observed']}")
    print()
    print("by GT regime -- a_hat mean/std vs GT |accel| mean, bias:")
    print(
        f"{'regime':<12} {'n':>6} {'a_hat mean':>11} {'a_hat std':>10} "
        f"{'gt accel mean':>14} {'bias':>9}"
    )
    for regime, block in report["by_regime"].items():
        if block["n"] == 0:
            print(f"{regime:<12} {0:>6}  (empty)")
            continue
        print(
            f"{regime:<12} {block['n']:>6} {block['a_hat_mean']:>11.4f} "
            f"{block['a_hat_std']:>10.4f} {block['gt_accel_mean']:>14.4f} "
            f"{block['bias_a_hat_minus_gt']:>+9.4f}"
        )
    print()
    print(
        f"noise floor (std of a_hat, static+sustained regimes): "
        f"{report['noise_floor_std_steady_regimes']:.4f} m/s^2"
    )
    print(f"  PERSON_SIGMA_A_MPS2 = {PERSON_SIGMA_A_MPS2:.4f} m/s^2")
    print(f"  SNR (PERSON_SIGMA_A_MPS2 / noise floor) = {report['snr_vs_person_sigma_a']:.4f}")
    print("  per-regime SNR (regime's own mean GT |accel| / noise floor):")
    for regime, value in report["transient_regime_snr"].items():
        print(f"    {regime:<12} SNR = {value:.4f}")
    print(
        f"  closed-form predicted a_hat std (2*mean(Var(v))/dt^2, sqrt'd): "
        f"{report['closed_form_a_hat_std_mean']:.4f} m/s^2"
    )
    divergence = (
        report["closed_form_a_hat_std_mean"] - report["noise_floor_std_steady_regimes"]
    )
    print(f"  divergence (closed-form - instrumented): {divergence:+.4f} m/s^2")
    print()
    print("noise attribution ablation (steady-regime a_hat std):")
    for name, block in report["ablation"].items():
        print(f"  {name:<14} n={block['n']:6}  std={block['noise_floor_std']:.4f} m/s^2")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="append", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--artifact-dir", default=None, type=Path)
    args = parser.parse_args(argv)

    config = IronConfig.load()
    root = (
        Path(args.root)
        if args.root
        else config.paths.resolve(config.eval.golden_sets_dir)
    )
    versions = args.version or ["v5-cessation", "v3-indoor"]

    any_failed = False
    reports = []
    for version in versions:
        report = _score_version(version, root, config)
        if report is None:
            any_failed = True
            continue
        reports.append(report)
        _print_report(report)

    if args.artifact_dir:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        out = args.artifact_dir / "acceleration_noise_floor.json"

        def _nan_to_none(value: Any) -> Any:
            if isinstance(value, float) and np.isnan(value):
                return None
            if isinstance(value, dict):
                return {k: _nan_to_none(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_nan_to_none(v) for v in value]
            return value

        out.write_text(
            json.dumps(_nan_to_none({"reports": reports}), indent=2, sort_keys=True)
            + "\n"
        )
        print(f"Written: {out}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
