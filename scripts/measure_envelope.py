"""Measure where the motion gate's real wake threshold falls.

``MotionGateConfig.envelope_threshold_px`` derives, from the gate's own
arithmetic, the smallest *foreground* region that can wake it. The scorecard
uses that boundary to decide whether a miss is a gate defect or the camera's
physical limit — so if the derivation is wrong, every one of those verdicts is
wrong, and it is wrong in the direction that either excuses real defects or
manufactures fake ones.

The derivation is about foreground area. The scorecard measures *silhouette*
area. Those differ: a displaced object marks the pixels it arrived at and the
ones it left, so its foreground footprint can approach twice its silhouette,
and the gate would then wake on movers roughly half the derived size.

This script measures the ratio instead of assuming it. It sweeps silhouette
area across the derived threshold, runs the real gate on synthesised frames,
and reports the wake rate per size and the interpolated 50% point.

    python scripts/measure_envelope.py
    python scripts/measure_envelope.py --json outputs/envelope/sweep.json

Exit codes:
    0  the sweep ran and the measured wake point is within tolerance
    1  measurement and derivation disagree by more than the tolerance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cascade import MotionGate, StageContext  # noqa: E402
from src.config import IronConfig  # noqa: E402

# Native raster the synthetic targets are drawn on, matching the golden set.
NATIVE_HEIGHT, NATIVE_WIDTH = 720, 1280

# Background grey. Flat, because a textured background would add its own
# foreground response and the point here is to isolate the target's.
BACKGROUND_LEVEL = 110
TARGET_LEVEL = 210


def render_sweep_clip(
    silhouette_gate_px: float,
    *,
    frames: int,
    step_px: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A flat scene containing one moving square of a known silhouette area.

    The square is sized in *gate* pixels and drawn at native resolution, so it
    exercises the same downscale path the gate uses in production.
    """
    scale = (NATIVE_HEIGHT * NATIVE_WIDTH) / (
        IronConfig.load().cascade.gate_width * IronConfig.load().cascade.gate_height
    )
    side = max(1, int(round(np.sqrt(silhouette_gate_px * scale))))

    clip = np.full(
        (frames, NATIVE_HEIGHT, NATIVE_WIDTH, 3), BACKGROUND_LEVEL, dtype=np.uint8
    )
    # Start near the top-left so a fast target has room to run. Travel is
    # clamped below rather than reserved here: reserving it would make the
    # start position depend on speed, and the target would then be measured at
    # a different part of the frame for every displacement.
    top = int(rng.integers(side, max(side + 1, NATIVE_HEIGHT // 4)))
    left = int(rng.integers(side, max(side + 1, NATIVE_WIDTH // 4)))

    for index in range(frames):
        # Stationary through warmup so the background model can converge on a
        # scene that contains the target; otherwise the first motion measured
        # is the model still learning.
        offset = 0 if index < WARMUP_HOLD else (index - WARMUP_HOLD) * step_px
        y = min(top + offset, NATIVE_HEIGHT - side)
        x = min(left + offset, NATIVE_WIDTH - side)
        clip[index, y : y + side, x : x + side] = TARGET_LEVEL

    return clip


WARMUP_HOLD = 14


def wake_rate_for_size(
    silhouette_gate_px: float, gate_config: Any, *, trials: int, step_px: int
) -> float:
    """Share of trials in which a mover of this size woke the gate."""
    frames = WARMUP_HOLD + 8
    wakes = 0
    for trial in range(trials):
        rng = np.random.default_rng(1000 + trial)
        clip = render_sweep_clip(
            silhouette_gate_px, frames=frames, step_px=step_px, rng=rng
        )
        gate = MotionGate(gate_config)
        woke = False
        for index in range(frames):
            output = gate.process(
                clip[index][np.newaxis, ...], StageContext(index, index)
            )
            # Only frames after the target starts moving count. A wake during
            # the hold would be the background model settling, not detection.
            if index > WARMUP_HOLD and output.wake_next:
                woke = True
        wakes += int(woke)
    return wakes / trials


def interpolate_crossing(curve: list[tuple[float, float]]) -> float | None:
    """Silhouette area at which the wake rate first crosses 50%.

    Linear interpolation between the bracketing samples. Returns ``None`` when
    the curve never crosses, which is itself a result worth reporting.
    """
    for (area_a, rate_a), (area_b, rate_b) in zip(curve, curve[1:]):
        if rate_a < 0.5 <= rate_b:
            if rate_b == rate_a:
                return area_b
            span = (0.5 - rate_a) / (rate_b - rate_a)
            return area_a + span * (area_b - area_a)
    return None


def _measure_model(args: Any) -> int:
    """Sweep several speeds and emit the speed-aware envelope model.

    A single wake threshold does not exist. The gate's background model absorbs
    a slow mover into the background regardless of how large it is, so the
    silhouette area needed to wake it is a function of speed, not a constant.
    The scorecard needs that whole relation, because classifying a miss as
    "below the physical envelope" against the wrong speed is how a real gate
    defect gets excused.
    """
    config = IronConfig.load()
    gate_config = config.cascade.motion_gate_config()
    derived = gate_config.envelope_threshold_px()

    print(f"gate raster        : {gate_config.gate_width}x{gate_config.gate_height}")
    print(f"DERIVED threshold  : {derived:.1f} gate px (FOREGROUND area)")
    print()
    print(f"{'displacement px/f':>18}  {'50% wake point':>16}  {'vs derived':>10}")
    print("-" * 50)

    samples: list[dict[str, Any]] = []
    for displacement in args.displacements:
        curve: list[tuple[float, float]] = []
        area = args.start
        while area <= args.stop + 1e-9:
            curve.append(
                (
                    area,
                    wake_rate_for_size(
                        area, gate_config, trials=args.trials, step_px=displacement
                    ),
                )
            )
            area += args.step
        crossing = interpolate_crossing(curve)
        ratio = f"{crossing / derived:.2f}x" if crossing else "never"
        shown = f"{crossing:.1f}" if crossing else "never wakes"
        print(f"{displacement:>18}  {shown:>16}  {ratio:>10}")
        samples.append(
            {
                "displacement_gate_px_per_frame": displacement
                * gate_config.gate_width
                / NATIVE_WIDTH,
                "displacement_native_px_per_frame": displacement,
                "wake_threshold_silhouette_gate_px": crossing,
                "curve": [{"silhouette_gate_px": a, "wake_rate": r} for a, r in curve],
            }
        )

    from src.provenance import current_stack_string

    model = {
        "gate_width": gate_config.gate_width,
        "gate_height": gate_config.gate_height,
        "min_foreground_fraction": gate_config.min_foreground_fraction,
        "derived_foreground_threshold_px": derived,
        # Day 17: was config.cascade.measured_stack, a hardcoded config
        # constant that would keep asserting the pinned stack even if this
        # specific calibration run executed under a drifted interpreter.
        # Live now, measured at the moment this file is written.
        "measured_stack": current_stack_string(),
        "trials_per_size": args.trials,
        "native_shape": [NATIVE_HEIGHT, NATIVE_WIDTH],
        # The area range actually swept. Without it, a speed whose curve never
        # crossed 50% is indistinguishable from one that cannot wake at all —
        # and the first envelope made exactly that mistake, sweeping only to
        # 320 gate px and recording "never wakes" for speeds that in fact wake
        # at 567. A consumer must be able to tell "unreachable" from "outside
        # what was measured".
        "swept_area_gate_px": {
            "start": args.start,
            "stop": args.stop,
            "step": args.step,
        },
        "samples": samples,
        "note": (
            "Silhouette area needed to wake the gate, per mover speed. The "
            "derived foreground threshold is exact arithmetic about FOREGROUND "
            "area; it is not a silhouette predictor, because a slow mover is "
            "absorbed into the background model at any size. Measured, not "
            "assumed."
        ),
    }

    if args.model_out:
        path = Path(args.model_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
        print(f"\nenvelope model written to {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=20.0)
    parser.add_argument("--stop", type=float, default=200.0)
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument(
        "--displacement",
        type=int,
        default=6,
        help="native pixels the target moves per frame",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.35,
        help=(
            "allowed relative disagreement between the derived threshold and "
            "the measured 50%% wake point, as a fraction of the derived value"
        ),
    )
    parser.add_argument("--json", default=None, help="write the curve here")
    parser.add_argument(
        "--displacements",
        type=int,
        nargs="+",
        default=None,
        help=(
            "sweep these displacements and emit a speed-aware envelope model "
            "instead of a single curve"
        ),
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="write the measured envelope model here (implies --displacements)",
    )
    args = parser.parse_args(argv)

    if args.displacements:
        return _measure_model(args)

    config = IronConfig.load()
    gate_config = config.cascade.motion_gate_config()
    derived = gate_config.envelope_threshold_px()

    print(f"gate raster        : {gate_config.gate_width}x{gate_config.gate_height}")
    print(f"min_foreground_frac: {gate_config.min_foreground_fraction}")
    print(f"DERIVED threshold  : {derived:.1f} gate px (foreground area)")
    print(f"target displacement: {args.displacement} native px/frame")
    print()
    print(f"{'silhouette gate px':>20}  {'wake rate':>10}")
    print("-" * 34)

    curve: list[tuple[float, float]] = []
    area = args.start
    while area <= args.stop + 1e-9:
        rate = wake_rate_for_size(
            area, gate_config, trials=args.trials, step_px=args.displacement
        )
        curve.append((area, rate))
        bar = "#" * int(round(rate * 20))
        print(f"{area:>20.1f}  {rate:>10.2f}  {bar}")
        area += args.step

    measured = interpolate_crossing(curve)
    print()
    if measured is None:
        print("MEASURED 50% wake point: never crossed in this sweep range")
        print("The derivation cannot be validated against this curve.")
        return 1

    ratio = measured / derived
    disagreement = abs(measured - derived) / derived
    print(f"MEASURED 50% wake point: {measured:.1f} gate px (silhouette area)")
    print(f"measured / derived     : {ratio:.2f}")
    print(
        f"disagreement           : {disagreement:.0%} (tolerance {args.tolerance:.0%})"
    )

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "gate_width": gate_config.gate_width,
                    "gate_height": gate_config.gate_height,
                    "min_foreground_fraction": gate_config.min_foreground_fraction,
                    "derived_threshold_px": derived,
                    "measured_wake_point_px": measured,
                    "ratio_measured_over_derived": ratio,
                    "displacement_px_per_frame": args.displacement,
                    "trials_per_size": args.trials,
                    "curve": [
                        {"silhouette_gate_px": a, "wake_rate": r} for a, r in curve
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\ncurve written to {path}")

    if disagreement > args.tolerance:
        print()
        print("DISAGREEMENT EXCEEDS TOLERANCE — the derivation does not predict")
        print("this gate's behaviour. The measured value is the real envelope;")
        print("the scorecard must use it, and the discrepancy must be recorded.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
