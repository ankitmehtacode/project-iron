"""Sub-stream vs. downscaled-main: does skipping software downscale help?

The hypothesis (Day 12): most IP cameras expose a main stream and a lower
resolution sub-stream. If the sub-stream is already near the 320x180 gate
raster, the motion gate can subscribe to it directly and skip the software
downscale entirely — the camera's own encoder ASIC does that work for free,
and the CPU-expensive main stream is only decoded once the gate wakes.
`cascade_bench.py` established a real, unresolved budget miss (4.57% idle
core cost vs. a 3.00% target); if this closes it architecturally rather
than by tuning thresholds, it is worth doing.

This is a HARNESS, not a result. It requires two real captures from the
SAME scene at the SAME moments — one at the camera's sub-stream resolution,
one at main-stream resolution downscaled in software — and no camera has
been connected yet (Day 12 is readiness work; see
docs/day12/infinigen_throughput.md's sibling finding that this project is
currently blocked on hardware for real-camera work generally). Running
this against two unrelated video files would produce a comparison between
different footage, not between two paths through the same footage — and
that number would look like a real one while measuring nothing. This
script refuses to run without matched captures rather than fabricate one
from mismatched inputs.

The risk to measure once footage exists, stated rather than assumed: a
heavily-compressed sub-stream may degrade MOG2's background model —
lower bitrate means coarser block-based compression artifacts that add
false variance a slow-moving true edge would not. This script reports
that as `mog2_variance_delta`, not as a pass/fail — the reader forms the
verdict once real numbers exist.

    python scripts/measure_substream_hypothesis.py \\
        --main-capture data/captures/site1_main.npz \\
        --sub-capture data/captures/site1_sub.npz

Exit codes:
    0  comparison ran, results printed and written
    1  no matched captures supplied — this is expected and reported plainly
       until a camera exists; not a bug
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.cascade import MotionGate, MotionGateConfig, StageContext  # noqa: E402


@dataclass
class StreamResult:
    label: str
    wake_decisions: list[bool]
    per_frame_seconds: list[float]

    @property
    def total_seconds(self) -> float:
        return sum(self.per_frame_seconds)

    @property
    def wake_rate(self) -> float:
        return float(np.mean(self.wake_decisions)) if self.wake_decisions else float("nan")


def run_gate(frames: np.ndarray, config: MotionGateConfig, label: str) -> StreamResult:
    """Feed a decoded clip through the motion gate, timing each frame."""
    gate = MotionGate(config)
    wakes = []
    times = []
    for index in range(frames.shape[0]):
        t0 = time.perf_counter()
        output = gate.process(frames[index][np.newaxis, ...], StageContext(index, index))
        times.append(time.perf_counter() - t0)
        wakes.append(bool(output.wake_next))
    return StreamResult(label=label, wake_decisions=wakes, per_frame_seconds=times)


def mog2_variance_delta(main_frames: np.ndarray, sub_frames: np.ndarray) -> float | None:
    """Frame-to-frame pixel variance, sub minus main — a compression-artifact proxy.

    Not a substitute for actually running MOG2 and comparing its foreground
    masks (the real test), which needs the two captures to be temporally
    matched to the frame. This is a cheap, always-computable proxy that
    flags whether the sub-stream is meaningfully noisier before the real
    comparison is run. Returns ``None`` when the two clips have different
    frame counts, since a per-frame variance delta requires them aligned.
    """
    if main_frames.shape[0] != sub_frames.shape[0]:
        return None
    main_diff = np.diff(main_frames.astype(np.float64), axis=0)
    sub_diff = np.diff(sub_frames.astype(np.float64), axis=0)
    return float(sub_diff.var() - main_diff.var())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-capture",
        type=Path,
        default=None,
        help="decoded .npz (key 'rgb') of the main-stream capture",
    )
    parser.add_argument(
        "--sub-capture",
        type=Path,
        default=None,
        help="decoded .npz (key 'rgb') of the sub-stream capture, SAME "
        "scene and moments as --main-capture",
    )
    parser.add_argument("--gate-width", type=int, default=320)
    parser.add_argument("--gate-height", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("outputs/day12/substream_hypothesis.json"))
    args = parser.parse_args(argv)

    if args.main_capture is None or args.sub_capture is None:
        print(
            "No matched main/sub-stream captures supplied. This harness is "
            "ready but cannot run — it needs two captures of the SAME "
            "scene at the SAME moments, one at each stream resolution, "
            "and no camera has been connected as of Day 12. Running this "
            "against two unrelated clips would compare different footage "
            "and report a number that looks real but measures nothing. "
            "Re-run with --main-capture and --sub-capture once "
            "scripts/discover_cameras.py has found a camera and both "
            "streams have been captured for the same window.",
            file=sys.stderr,
        )
        return 1

    if not args.main_capture.exists() or not args.sub_capture.exists():
        print(
            f"capture file missing: "
            f"{args.main_capture if not args.main_capture.exists() else args.sub_capture}",
            file=sys.stderr,
        )
        return 1

    main_frames = np.load(args.main_capture)["rgb"]
    sub_frames = np.load(args.sub_capture)["rgb"]

    if main_frames.shape[0] != sub_frames.shape[0]:
        print(
            f"frame count mismatch: main has {main_frames.shape[0]}, sub has "
            f"{sub_frames.shape[0]}. These captures are not the same moments "
            "and a wake-parity comparison between them would not mean "
            "anything.",
            file=sys.stderr,
        )
        return 1

    downscale_config = MotionGateConfig(gate_width=args.gate_width, gate_height=args.gate_height)
    # The sub-stream path: if the sub-stream is already at or below the gate
    # raster, gate_width/gate_height beyond the source shape is a no-op —
    # MotionGateConfig's own downscale clamps to source_shape (see
    # src/cascade/motion.py), so no separate "skip downscale" flag is
    # needed here; feeding sub-resolution frames through the SAME config
    # IS the "no software downscale needed" path when the source is
    # already small enough.
    subscale_config = MotionGateConfig(gate_width=args.gate_width, gate_height=args.gate_height)

    main_result = run_gate(main_frames, downscale_config, "main (software downscale)")
    sub_result = run_gate(sub_frames, subscale_config, "sub-stream (camera-native)")

    agreement = np.mean(
        [a == b for a, b in zip(main_result.wake_decisions, sub_result.wake_decisions)]
    )
    cost_ratio = (
        sub_result.total_seconds / main_result.total_seconds
        if main_result.total_seconds > 0
        else float("nan")
    )
    variance_delta = mog2_variance_delta(main_frames, sub_frames)

    print(f"main : {main_result.wake_rate:.3f} wake rate, {main_result.total_seconds*1000:.1f} ms total")
    print(f"sub  : {sub_result.wake_rate:.3f} wake rate, {sub_result.total_seconds*1000:.1f} ms total")
    print(f"wake-decision parity: {agreement:.3%}")
    print(f"cost ratio (sub/main): {cost_ratio:.3f} ({'cheaper' if cost_ratio < 1 else 'not cheaper'})")
    if variance_delta is not None:
        print(f"frame-diff variance delta (sub - main): {variance_delta:+.4f}")

    payload: dict[str, Any] = {
        "main_wake_rate": main_result.wake_rate,
        "sub_wake_rate": sub_result.wake_rate,
        "wake_decision_parity": float(agreement),
        "cost_ratio_sub_over_main": float(cost_ratio),
        "mog2_variance_delta_sub_minus_main": variance_delta,
        "n_frames": int(main_frames.shape[0]),
        "interpretation_note": (
            "Parity < 1.0 means the sub-stream and downscaled-main paths "
            "disagree on when to wake — investigate before adopting the "
            "sub-stream path regardless of any cost saving. A positive "
            "variance delta is evidence (not proof) that sub-stream "
            "compression may be adding false MOG2 variance; only a direct "
            "MOG2 foreground-mask comparison on real footage settles it."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
