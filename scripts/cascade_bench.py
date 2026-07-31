"""Benchmark the motion gate on a synthetic sequence with a known answer.

The sequence is 60 seconds at 12 fps, 720p: static for the first 40 seconds,
then a moving blob for the last 20. So roughly one third of frames contain
motion, and the gate should wake the next stage for about that share — plus a
little, because hysteresis holds it awake briefly after the blob stops.

The point of a synthetic sequence with a known answer is that it separates two
failure modes that look identical on real footage. A gate that wakes on 100% of
frames and a gate that wakes on the correct 33% both produce correct downstream
output; only one of them meets the budget. Real footage cannot tell you which
you have, because you do not know the true answer.

Run:
    python scripts/cascade_bench.py
    python scripts/cascade_bench.py --seconds 20 --fps 12
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator

import numpy as np

from src.cascade import CascadeRunner, MotionGate, MotionGateConfig, StageContext

NANOSECONDS_PER_SECOND = 1_000_000_000

# Fraction of frames that should contain motion, and how far the observed wake
# rate may drift from it. The tolerance is one-sided in spirit: waking a little
# extra is hysteresis doing its job, waking far less means missed events.
EXPECTED_MOTION_SHARE = 1.0 / 3.0
TOLERANCE = 0.10


def synth_sequence(
    seconds: int,
    fps: int,
    width: int,
    height: int,
    static_seconds: int,
    seed: int,
) -> Iterator[tuple[np.ndarray, StageContext]]:
    """Yield frames: a static textured scene, then a moving blob.

    The background is textured rather than flat, and carries mild per-frame
    sensor noise. A flat noiseless background would make the benchmark trivial
    and would not exercise the ``min_foreground_fraction`` floor at all — which
    is the parameter that decides whether the gate fires on grain.
    """
    rng = np.random.default_rng(seed)
    background = rng.integers(60, 190, size=(height, width, 3), dtype=np.uint8)
    total_frames = seconds * fps
    motion_starts = static_seconds * fps
    blob = max(24, height // 12)

    for index in range(total_frames):
        frame = background.copy()
        noise = rng.integers(-4, 5, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if index >= motion_starts:
            travelled = index - motion_starts
            moving_frames = max(1, total_frames - motion_starts)
            x = int((width - blob) * (travelled / moving_frames))
            y = height // 2 - blob // 2
            frame[y : y + blob, x : x + blob] = 255

        ctx = StageContext(
            frame_index=index,
            ts_ns=int(index * NANOSECONDS_PER_SECOND / fps),
        )
        yield frame[np.newaxis, ...], ctx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--static-seconds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args(argv)

    gate = MotionGate(MotionGateConfig())
    runner = CascadeRunner([gate], stats_interval_s=1e9)  # emit only at flush

    total_frames = args.seconds * args.fps
    print(
        f"Sequence : {args.seconds}s at {args.fps} fps, "
        f"{args.width}x{args.height} ({total_frames} frames)"
    )
    print(
        f"           static for {args.static_seconds}s, "
        f"moving blob for {args.seconds - args.static_seconds}s"
    )
    print(f"Backend  : {gate.backend}")
    print()

    started = time.perf_counter()
    stats = runner.run(
        synth_sequence(
            args.seconds,
            args.fps,
            args.width,
            args.height,
            args.static_seconds,
            args.seed,
        )
    )
    wall_s = time.perf_counter() - started

    gate_stats = stats.stages[0]
    wake_share = gate_stats.woke_next / gate_stats.frames_seen
    per_frame_ms = gate_stats.p50_ms

    # A frame budget at this frame rate, used to express stage-0 cost as a
    # share of one core — the form the Tier-1 budget is written in.
    frame_budget_ms = 1000.0 / args.fps
    core_share = per_frame_ms / frame_budget_ms

    print(stats.summary())
    print()
    print(
        f"Woke next stage on : {gate_stats.woke_next}/{gate_stats.frames_seen} "
        f"frames ({wake_share * 100:.1f}%)"
    )
    print(
        f"Expected           : ~{EXPECTED_MOTION_SHARE * 100:.1f}% "
        f"(+/- {TOLERANCE * 100:.0f} points, plus hysteresis)"
    )
    print()
    print("Stage-0 cost per frame:")
    print(
        f"  p50 {gate_stats.p50_ms:.3f} ms | p95 {gate_stats.p95_ms:.3f} ms | "
        f"p99 {gate_stats.p99_ms:.3f} ms | max {gate_stats.max_ms:.3f} ms"
    )
    print(
        f"  {core_share * 100:.2f}% of one core per camera at {args.fps} fps "
        f"(budget: under 3% idle)"
    )
    print(
        f"  wall clock {wall_s:.2f}s for {total_frames} frames "
        f"({total_frames / wall_s:.0f} frames/s single-threaded)"
    )
    print()

    low = EXPECTED_MOTION_SHARE - TOLERANCE
    high = EXPECTED_MOTION_SHARE + TOLERANCE
    if not low <= wake_share <= high:
        print(
            f"FAIL: gate woke the next stage on {wake_share * 100:.1f}% of frames, "
            f"outside the expected {low * 100:.1f}%-{high * 100:.1f}%.",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: wake rate {wake_share * 100:.1f}% is within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
