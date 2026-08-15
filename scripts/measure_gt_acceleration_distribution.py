"""Where do the golden sets' unphysical GT accelerations fall? (Day 30,
Objective 1).

HARD SCOPE RULE, unchanged: no timing, throughput, CPU, or latency claim
is made anywhere in this script. Distributions and counts only.

The question this exists to answer
------------------------------------
Day 29 measured a peak GT horizontal acceleration of 36.58 m/s^2 on
v5-cessation -- 3.7g, where a world-class sprinter peaks near 1g -- and
recorded it as a single scalar with no idea WHERE in the data it sits.
That distinction decides whether the set is usable. If the >1g transients
are spread thinly across every regime, they are a nuisance. If they
concentrate in `cessation`, then Day 21's NEES ~815 cessation diagnosis
measured a filter's response to a teleport-to-zero rather than to a
person stopping, and ADR 0010's config A->B adoption -- which rests on
cessation behaviour -- was decided on an artifact of the generator.

So this reports the acceleration distribution PER REGIME, using the same
regime partition (`src.estimator.regime.classify_track`) that every
per-regime estimator number since Day 21 has been computed against. Same
partition, same GT, same finite-difference convention: the numbers here
are directly comparable to those, which is the whole point.

Definition, and the two frames that are not reported
------------------------------------------------------
GT velocity uses this project's standing convention (forward difference,
frame 0 mirrors frame 1 -- `src.estimator.regime.classify_track`,
`scripts/eval_estimator.py`, `scripts/measure_gt_constraint_violations.py`).

Acceleration is `(v[t] - v[t-1]) / dt`. Under the mirror convention
`v[0] == v[1]`, so `a[1]` is IDENTICALLY ZERO for every track regardless
of what the track does -- an artifact of the convention, not a
measurement. Day 29's script counted it (`if t >= 1`), which put one
guaranteed zero into every track's distribution. Acceleration genuinely
needs three positions, so it is reported here for `t >= 2` only, and the
number of frames dropped for this reason is reported per set rather than
left implicit. This does not move the maximum (the extremum is a
mid-track transient, not frame 1) but it does move the percentiles, so
the two scripts' distributions are not interchangeable and this one says
which it is.

Axis convention: `agent_xyz` is `[x, y_up, z_depth]` -- index 1 is
vertical. Verified against the generator, not assumed; see
`src.contracts.ground_truth.GroundTruthAxes` for the typed statement of
this, and `scripts/measure_gt_constraint_violations.py`'s `VERTICAL_AXIS`
for the Day-29 bug that made it worth typing. Acceleration MAGNITUDE is
axis-independent, so nothing here depends on getting that right -- the
per-axis breakdown does, and is reported separately.

    python scripts/measure_gt_acceleration_distribution.py
    python scripts/measure_gt_acceleration_distribution.py --version v6-motion
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

from src.config import IronConfig  # noqa: E402
from src.contracts.ground_truth import (  # noqa: E402
    GENERATOR_AXES,
    GtAccelerationTrack,
    gt_position_track,
)
from src.data.golden import GoldenSetError, load_golden_set  # noqa: E402
from src.estimator.constraints import STANDARD_GRAVITY_MPS2  # noqa: E402
from src.estimator.regime import (  # noqa: E402
    MOTION_REGIMES,
    STATIC_SPEED_THRESHOLD_MPS,
    classify_track,
)

FloatArray = npt.NDArray[np.float64]

HISTOGRAM_EDGES_MPS2: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    STANDARD_GRAVITY_MPS2,
    20.0,
    40.0,
    float("inf"),
)
"""Bin edges for the reported histogram, in m/s^2.

Not equal-width: the interesting structure is at both ends. Below ~2 m/s^2
is ordinary pedestrian gait variation; `STANDARD_GRAVITY_MPS2` is the
physiological ceiling this whole objective is about (a world-class
sprinter's peak horizontal acceleration is ~1g and a stopping pedestrian
is far below that); above it is motion no body on foot can produce.
"""


def _histogram(values: FloatArray) -> list[dict[str, Any]]:
    """Counts per :data:`HISTOGRAM_EDGES_MPS2` bin, with the labels the
    report prints. Returned rather than printed so the JSON carries the
    same numbers the console does."""
    edges = np.array(HISTOGRAM_EDGES_MPS2)
    counts, _ = np.histogram(values, bins=edges)
    total = max(1, int(values.size))
    out: list[dict[str, Any]] = []
    for i, count in enumerate(counts):
        lo, hi = HISTOGRAM_EDGES_MPS2[i], HISTOGRAM_EDGES_MPS2[i + 1]
        label = f"[{lo:.2f}, {hi:.2f})" if np.isfinite(hi) else f"[{lo:.2f}, inf)"
        out.append(
            {
                "bin": label,
                "lo_mps2": lo,
                "hi_mps2": hi if np.isfinite(hi) else None,
                "count": int(count),
                "fraction": round(int(count) / total, 6),
            }
        )
    return out


def _summarize(values: FloatArray) -> dict[str, Any]:
    """p50/p95/max/mean and the >1g fraction for one bucket of
    acceleration magnitudes. An empty bucket reports its emptiness rather
    than a zero -- a regime with no frames and a regime whose frames are
    all at rest are different facts, and `0.0` would read as the second."""
    if values.size == 0:
        return {
            "frames": 0,
            "p50_mps2": None,
            "p95_mps2": None,
            "max_mps2": None,
            "mean_mps2": None,
            "frames_above_1g": 0,
            "fraction_above_1g": None,
        }
    above = int(np.sum(values > STANDARD_GRAVITY_MPS2))
    return {
        "frames": int(values.size),
        "p50_mps2": round(float(np.percentile(values, 50)), 4),
        "p95_mps2": round(float(np.percentile(values, 95)), 4),
        "max_mps2": round(float(np.max(values)), 4),
        "mean_mps2": round(float(np.mean(values)), 4),
        "frames_above_1g": above,
        "fraction_above_1g": round(above / values.size, 6),
    }


def _stop_events(
    speed: FloatArray, magnitude: FloatArray, static_threshold_mps: float
) -> list[float]:
    """Peak |a| across each moving->static transition in one track.

    A frame count answers "how much of the data is unphysical"; this
    answers "how many of the EVENTS are." They come apart badly on a set
    like v5-cessation, where the `cessation` regime is mostly a recovery
    tail sitting at exactly zero acceleration and the whole transient is
    one or two frames — a small frame fraction can still mean every stop
    event in the set is impossible, which is the version of the finding
    that decides whether the set can be used.

    The window is the transition frame and the one before it, since the
    deceleration is carried by the velocity step between them.
    """
    peaks: list[float] = []
    moving = speed >= static_threshold_mps
    for t in range(1, speed.size):
        if moving[t - 1] and not moving[t]:
            window = magnitude[max(0, t - 1) : t + 1]
            if window.size:
                peaks.append(float(np.max(window)))
    return peaks


def _measure_version(
    version: str, root: Path, config: IronConfig
) -> dict[str, Any] | None:
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

    by_regime: dict[str, list[float]] = {regime: [] for regime in MOTION_REGIMES}
    all_accel: list[float] = []
    vertical_accel: list[float] = []
    tracks = 0
    frames_seen = 0
    frames_dropped_undefined = 0
    worst = {"value": 0.0, "where": "", "regime": ""}
    stop_event_peaks: list[float] = []

    for clip in golden.clips:
        clip_path = clip_root / f"{clip.clip_id}.npz"
        if not clip_path.exists():
            continue
        with np.load(clip_path) as data:
            if "agent_xyz" not in data:
                continue
            raw = np.asarray(data["agent_xyz"], dtype=np.float64)
        dt_s = 1.0 / fps_by_clip.get(clip.clip_id, 12.0)

        for agent_index in range(raw.shape[1]):
            positions = gt_position_track(raw[:, agent_index, :], GENERATOR_AXES)
            first = GtAccelerationTrack.FIRST_MEANINGFUL_INDEX
            if positions.frames <= first:
                # Acceleration needs three positions. A 2-frame track has
                # a velocity and no acceleration; skipping it silently
                # would make `frames_seen` disagree with the regime counts.
                frames_dropped_undefined += positions.frames
                continue
            tracks += 1
            frames_seen += positions.frames
            frames_dropped_undefined += first  # frames 0-1, see module docstring

            regimes = classify_track(positions.values_in(GENERATOR_AXES), dt_s)
            velocity = positions.differentiate(dt_s)
            accel = velocity.differentiate(dt_s)
            magnitude = accel.magnitude()
            vertical = np.abs(accel.vertical_component())
            stop_event_peaks.extend(
                _stop_events(velocity.speed(), magnitude, STATIC_SPEED_THRESHOLD_MPS)
            )

            for t in range(first, positions.frames):
                value = float(magnitude[t])
                by_regime[regimes[t]].append(value)
                all_accel.append(value)
                vertical_accel.append(float(vertical[t]))
                if value > worst["value"]:
                    worst = {
                        "value": value,
                        "where": f"{clip.clip_id}/agent{agent_index}/frame{t}",
                        "regime": regimes[t],
                    }

    all_array = np.array(all_accel, dtype=np.float64)
    regime_summaries = {
        regime: _summarize(np.array(values, dtype=np.float64))
        for regime, values in by_regime.items()
    }

    total_above_1g = int(np.sum(all_array > STANDARD_GRAVITY_MPS2))
    above_1g_by_regime = {
        regime: summary["frames_above_1g"]
        for regime, summary in regime_summaries.items()
    }
    cessation_share = (
        above_1g_by_regime.get("cessation", 0) / total_above_1g
        if total_above_1g
        else None
    )

    peaks = np.array(stop_event_peaks, dtype=np.float64)
    stop_events = {
        "count": int(peaks.size),
        "above_1g": int(np.sum(peaks > STANDARD_GRAVITY_MPS2)),
        "fraction_above_1g": (
            round(float(np.mean(peaks > STANDARD_GRAVITY_MPS2)), 6)
            if peaks.size
            else None
        ),
        "median_peak_mps2": round(float(np.median(peaks)), 4) if peaks.size else None,
        "max_peak_mps2": round(float(np.max(peaks)), 4) if peaks.size else None,
    }

    return {
        "version": version,
        "dataset": dataset,
        "stop_events": stop_events,
        "tracks": tracks,
        "frames_in_tracks": frames_seen,
        "frames_with_defined_acceleration": int(all_array.size),
        "frames_dropped_acceleration_undefined": frames_dropped_undefined,
        "overall": _summarize(all_array),
        "vertical_axis_only": _summarize(np.array(vertical_accel, dtype=np.float64)),
        "histogram": _histogram(all_array),
        "by_regime": regime_summaries,
        "above_1g_by_regime": above_1g_by_regime,
        "total_frames_above_1g": total_above_1g,
        "cessation_share_of_above_1g": (
            round(cessation_share, 6) if cessation_share is not None else None
        ),
        "worst_frame": worst,
        "one_g_mps2": STANDARD_GRAVITY_MPS2,
    }


def _print_version(result: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print(f"{result['version']}  ({result['dataset']})")
    print("=" * 78)
    print(
        f"  {result['tracks']} tracks, "
        f"{result['frames_with_defined_acceleration']} frames with defined "
        f"acceleration ({result['frames_dropped_acceleration_undefined']} "
        "dropped: frames 0-1 of each track, see docstring)"
    )
    overall = result["overall"]
    print(
        f"  overall |a|: p50 {overall['p50_mps2']}  p95 {overall['p95_mps2']}  "
        f"max {overall['max_mps2']}  mean {overall['mean_mps2']}"
    )
    print(
        f"  above 1g ({result['one_g_mps2']:.5f} m/s^2): "
        f"{overall['frames_above_1g']} frames "
        f"({(overall['fraction_above_1g'] or 0.0) * 100:.2f}%)"
    )

    print("\n  histogram of |a| (m/s^2):")
    for row in result["histogram"]:
        bar = "#" * int(round(row["fraction"] * 50))
        print(
            f"    {row['bin']:>18}  {row['count']:6d}  "
            f"{row['fraction']*100:6.2f}%  {bar}"
        )

    print("\n  per regime:")
    print(
        f"    {'regime':<11} {'frames':>7} {'p50':>9} {'p95':>9} "
        f"{'max':>9} {'>1g':>7} {'>1g %':>8}"
    )
    for regime in MOTION_REGIMES:
        s = result["by_regime"][regime]
        if s["frames"] == 0:
            print(f"    {regime:<11} {0:>7}      (regime absent from this set)")
            continue
        print(
            f"    {regime:<11} {s['frames']:>7} {s['p50_mps2']:>9.3f} "
            f"{s['p95_mps2']:>9.3f} {s['max_mps2']:>9.3f} "
            f"{s['frames_above_1g']:>7} {(s['fraction_above_1g'] or 0.0)*100:>7.2f}%"
        )

    vertical = result["vertical_axis_only"]
    print(
        f"\n  vertical axis (index 1) only: max |a_y| "
        f"{vertical['max_mps2']} m/s^2 "
        f"({vertical['frames_above_1g']} frames above 1g)"
    )

    worst = result["worst_frame"]
    if worst["where"]:
        print(
            f"  worst frame: {worst['value']:.4f} m/s^2 at {worst['where']} "
            f"(regime: {worst['regime']})"
        )
    events = result["stop_events"]
    if events["count"]:
        print(
            f"  stop events (moving->static transitions): {events['count']}, "
            f"peak |a| median {events['median_peak_mps2']} max "
            f"{events['max_peak_mps2']} m/s^2 -- "
            f"{events['above_1g']}/{events['count']} "
            f"({(events['fraction_above_1g'] or 0.0)*100:.1f}%) exceed 1g"
        )
    else:
        print("  stop events (moving->static transitions): none in this set.")

    share = result["cessation_share_of_above_1g"]
    if result["total_frames_above_1g"] == 0:
        print("  NO frame in this set exceeds 1g.")
    elif share is not None:
        print(
            f"  cessation's share of all >1g frames: {share*100:.1f}% "
            f"({result['above_1g_by_regime'].get('cessation', 0)}/"
            f"{result['total_frames_above_1g']})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="append", help="golden set version")
    parser.add_argument("--json", type=Path, help="write results as JSON")
    args = parser.parse_args()

    config = IronConfig.load()
    root = config.paths.resolve(config.eval.golden_sets_dir)
    versions = args.version or ["v5-cessation", "v3-indoor"]

    results = []
    for version in versions:
        result = _measure_version(version, root, config)
        if result is None:
            return 1
        results.append(result)
        _print_version(result)

    print()
    print("=" * 78)
    print("GATING QUESTION: do the >1g transients concentrate in cessation?")
    for result in results:
        share = result["cessation_share_of_above_1g"]
        if result["total_frames_above_1g"] == 0:
            print(f"  {result['version']}: no >1g frames at all.")
        else:
            cessation_fraction = (
                result["by_regime"]["cessation"]["fraction_above_1g"] or 0.0
            )
            print(
                f"  {result['version']}: {share*100:.1f}% of >1g frames are "
                f"cessation-regime "
                f"({result['above_1g_by_regime'].get('cessation', 0)}/"
                f"{result['total_frames_above_1g']}); cessation frames above "
                f"1g: {cessation_fraction*100:.2f}% of that regime."
            )
        events = result["stop_events"]
        if events["count"]:
            print(
                f"    ...and {events['above_1g']}/{events['count']} stop "
                "EVENTS peak above 1g (median peak "
                f"{events['median_peak_mps2']} m/s^2)."
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
