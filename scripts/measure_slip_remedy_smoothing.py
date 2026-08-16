"""Day 29, Objective 2 -- test Candidate B (multi-frame smoothing of the
causal acceleration estimate) against Objective 1's own noise-floor
measurement, with the SNR prediction stated BEFORE running the evaluation.

HARD SCOPE RULE, unchanged: no timing, throughput, CPU, or latency claim is
made anywhere in this script.

Why Candidate B, and not Candidate A, is the one tested today
-------------------------------------------------------------
`scripts/measure_acceleration_noise_floor.py`'s ablation found the noise
floor is almost entirely MEASUREMENT-noise-driven, not process-noise-driven
(q_near_zero: 6.84/7.31 m/s^2, essentially unchanged from baseline
6.86/7.33; r_near_zero: 0.74/0.02 m/s^2, a >9x/>300x collapse). Both
Candidate A (a Kalman-filtered, model-based acceleration state) and
Candidate B (an explicit windowed average of the existing causal a_hat
sequence) are, physically, the SAME remedy at bottom: more temporal
averaging to suppress R-driven noise. A Kalman smoother's advantage over a
boxcar average is a better small-N constant and adaptive weighting, not a
different asymptotic noise-vs-window-length scaling -- both are bounded by
the same underlying physics once R dominates. Candidate B is tested first
because it requires no change to the state model (no new motion-model kind,
no new state dimension, no touching STATE_DIM=6 anywhere in
src/estimator/joint.py or src/estimator/filter.py) and its lag cost is
directly, transparently measurable as a window length -- exactly what this
objective asks to be reported. If B's own lag cost disqualifies it against
this project's already-declared PEDESTRIAN_STOP_DURATION_S bound, A would
need to beat the SAME noise-vs-averaging-window physics to do
meaningfully better, which the ablation gives no reason to expect (R
dominates regardless of *how* the averaging is done) -- so B's result
closes the question for both, per this day's own discipline ("test at most
two remedies... a fifth attempt after a fourth failure is not diligence").

Prediction, stated before running (naive sqrt(K) noise-averaging scaling)
---------------------------------------------------------------------------
Objective 1's steady-regime noise floor is ~6.86-7.33 m/s^2 pooled;
PERSON_SIGMA_A_MPS2 = 1.5 m/s^2. Under naive independent-sample averaging,
noise std falls as 1/sqrt(K) for a K-frame window, so reaching SNR>=1 needs
roughly K >= (noise_floor / PERSON_SIGMA_A_MPS2)^2 ~= (7.1/1.5)^2 ~= 22
frames -- at 12fps, ~1.83s. PEDESTRIAN_STOP_DURATION_S (this project's own
already-declared bound on how long a voluntary pedestrian stop takes) is
1.0s. PREDICTION, stated before running: a smoothing window wide enough to
reach SNR>=1 will need roughly 1.8x PEDESTRIAN_STOP_DURATION_S -- i.e. the
remedy is self-defeating in exactly the cessation regime this whole
investigation exists to fix, because the smoothed estimate would still be
averaging in PRE-STOP motion for most of the stop's own settling window.
This script measures the ACTUAL (not naively-predicted) noise floor per
window size and checks whether the prediction holds.

Method
------
For K in a sweep, `a_hat_smoothed[t] = mean(a_hat[t-K+1 : t+1])` -- a
CAUSAL boxcar average of Objective 1's own already-computed, already-causal
per-step a_hat values (only past values are averaged; no future leakage).
Requires K valid past a_hat values (t >= FIRST_ACCEL_INDEX + K - 1); frames
before that are excluded from this K's statistics, the same
never-fabricate-what-was-not-measured discipline as Objective 1's own
FIRST_ACCEL_INDEX exclusion.

    python scripts/measure_slip_remedy_smoothing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_estimator as ee  # noqa: E402
import measure_acceleration_noise_floor as noise_floor  # noqa: E402

from src.config import IronConfig  # noqa: E402
from src.data import validity  # noqa: E402
from src.contracts.ground_truth import (  # noqa: E402
    GENERATOR_AXES,
    gt_position_track,
)
from src.data.golden import GoldenSetError, load_golden_set  # noqa: E402
from src.estimator.motion_model import PERSON_SIGMA_A_MPS2, motion_model_for  # noqa: E402
from src.estimator.motion_model import PEDESTRIAN_STOP_DURATION_S  # noqa: E402
from src.estimator.regime import classify_track  # noqa: E402

FloatArray = npt.NDArray[np.float64]

WINDOW_SIZES_FRAMES = (1, 2, 3, 5, 8, 13, 21, 34)
"""Fibonacci-spaced sweep: dense at the small end (where lag is cheap),
sparse at the large end (where the search is really just confirming the
prediction fails harder, not locating a precise crossing)."""

FPS = 12.0
"""Both golden sets carry exactly this frame rate (measured, see
measure_acceleration_noise_floor.py's own module docstring on frame-rate
availability) -- used only to convert a window's frame count to seconds
for the lag comparison against PEDESTRIAN_STOP_DURATION_S."""

PREDICTED_MIN_K_FRAMES = 22
"""Naive sqrt(K) prediction from the module docstring: (noise_floor /
PERSON_SIGMA_A_MPS2)^2 ~= (7.1/1.5)^2 ~= 22, using Objective 1's pooled
noise floor. Kept as a literal, pre-registered number -- not rederived
after seeing this script's own results below."""


def _smoothed_series(a_hat: FloatArray, window: int) -> FloatArray:
    """Causal boxcar average -- out[t] = mean(a_hat[t-window+1:t+1]) where
    every element of that slice is not NaN, else NaN (insufficient causal
    history at this window size, same convention as a_hat's own [0,1]
    bootstrap-degenerate NaNs)."""
    n = len(a_hat)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(n):
        start = t - window + 1
        if start < 0:
            continue
        window_vals = a_hat[start : t + 1]
        if np.any(np.isnan(window_vals)):
            continue
        out[t] = float(np.mean(window_vals))
    return out


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
            return {"version": version, "refused": True}

    measurement_model = ee.measurement_model_for(sensor=version, capability="state_estimation")
    carrier_motion_model = motion_model_for("person", velocity_covariance_floor=True)

    # window -> list of (regime, smoothed value) across every track
    by_window: dict[int, list[dict[str, Any]]] = {w: [] for w in WINDOW_SIZES_FRAMES}
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
            if track.shape[0] <= noise_floor.FIRST_ACCEL_INDEX:
                continue
            regimes = classify_track(track, dt_s)

            rng = np.random.default_rng(noise_floor.EVAL_SEED)
            a_hat = noise_floor._a_hat_series(
                track, extrinsics, f"{clip.clip_id}/{agent_index}", fps,
                carrier_motion_model, measurement_model, rng,
            )
            if a_hat is None:
                continue
            tracks_scored += 1

            for window in WINDOW_SIZES_FRAMES:
                smoothed = _smoothed_series(a_hat, window)
                for t in range(track.shape[0]):
                    if np.isnan(smoothed[t]):
                        continue
                    # gt_accel index alignment matches noise_floor.py:
                    # a_hat[t] estimates the (t-2, t-1) interval.
                    regime = regimes[t - 1] if t >= 1 else regimes[0]
                    by_window[window].append(
                        {"regime": regime, "value": float(smoothed[t])}
                    )

    if tracks_scored == 0:
        return {"version": version, "refused": False, "tracks_scored": 0}

    steady = ("static", "sustained")
    per_window: dict[int, dict[str, Any]] = {}
    for window, records in by_window.items():
        steady_vals = [r["value"] for r in records if r["regime"] in steady]
        std = float(np.std(steady_vals)) if steady_vals else float("nan")
        snr = (
            PERSON_SIGMA_A_MPS2 / std
            if std and not np.isnan(std) and std > 0
            else float("nan")
        )
        per_window[window] = {
            "n_steady": len(steady_vals),
            "noise_floor_std": std,
            "snr": snr,
            "lag_frames": window,
            "lag_s": window / FPS,
            "lag_vs_pedestrian_stop_duration": window / FPS / PEDESTRIAN_STOP_DURATION_S,
        }

    return {
        "version": version,
        "refused": False,
        "tracks_scored": tracks_scored,
        "per_window": per_window,
    }


def _print_report(report: dict[str, Any]) -> None:
    version = report["version"]
    print("=" * 78)
    print(f"CANDIDATE B (windowed smoothing): {version}")
    print("=" * 78)
    if report.get("refused") or report.get("tracks_scored", 0) == 0:
        print("No trajectory scored (refused or empty).")
        return
    print(f"tracks scored: {report['tracks_scored']}")
    print()
    print(
        f"{'K (frames)':>10} {'n':>6} {'noise std':>10} {'SNR':>7} "
        f"{'lag (s)':>8} {'lag / stop_duration':>20}"
    )
    for window, block in report["per_window"].items():
        print(
            f"{window:>10} {block['n_steady']:>6} {block['noise_floor_std']:>10.4f} "
            f"{block['snr']:>7.4f} {block['lag_s']:>8.3f} "
            f"{block['lag_vs_pedestrian_stop_duration']:>20.3f}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    config = IronConfig.load()
    root = config.paths.resolve(config.eval.golden_sets_dir)
    versions = ["v5-cessation", "v3-indoor"]

    print(
        "PREDICTION (stated before running): naive 1/sqrt(K) noise scaling "
        "from Objective 1's measured noise floor (~6.86-7.33 m/s^2 pooled) "
        f"predicts SNR>=1 needs roughly K~={PREDICTED_MIN_K_FRAMES} frames "
        f"(~{PREDICTED_MIN_K_FRAMES / FPS:.2f}s at 12fps), "
        f"~{PREDICTED_MIN_K_FRAMES / FPS / PEDESTRIAN_STOP_DURATION_S:.2f}x "
        f"PEDESTRIAN_STOP_DURATION_S ({PEDESTRIAN_STOP_DURATION_S:.1f}s) -- "
        "predicting the remedy is self-defeating for cessation before "
        "running the sweep below."
    )
    print()

    reports = []
    any_failed = False
    for version in versions:
        report = _score_version(version, root, config)
        if report is None:
            any_failed = True
            continue
        reports.append(report)
        _print_report(report)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
