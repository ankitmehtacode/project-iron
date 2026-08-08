"""Benchmark the motion gate: wake correctness, wake parity, and cost.

Three questions, and all three have to be answered together.

**Does it wake on the right frames?** Each synthetic scenario has a known
number of moving frames, so the expected wake rate is arithmetic rather than
opinion. Real footage cannot answer this, because nobody knows the true answer
for it.

**Does reduced-resolution gating decide the same things?** Every scenario is
run twice — once gating at full resolution, once at the configured gate
resolution — and the two wake decisions are compared frame by frame. Anything
other than exact agreement is reported and fails the run. Thresholds are
deliberately not retuned to manufacture parity: a threshold tuned until the
numbers agree proves only that it was tuned.

**Does it fit the budget?** Stage-0 cost is reported as a share of one core per
camera and asserted against ``cascade.idle_core_budget_fraction``. The Tier-1
claim is exactly as true as this number.

The small-target scenario is the one that matters. A blob ~15 px tall *at gate
resolution* is a person at roughly 60 px in the source frame — someone across a
car park. If downscaling loses them, the gate sleeps through exactly the events
it exists to catch, and no cost saving is worth that.

    python scripts/cascade_bench.py
    python scripts/cascade_bench.py --seconds 20 --static-seconds 13
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.cascade import MotionGate, MotionGateConfig, StageContext
from src.config import IronConfig

NANOSECONDS_PER_SECOND = 1_000_000_000
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_VIDEO = REPO_ROOT / "src" / "interface" / "ui" / "data" / "raw" / "test_video.mp4"

# How far the observed wake rate may sit from the share of frames that actually
# contain motion. Generous upward: hysteresis holding the gate awake after
# motion stops is the feature working, not drift.
TOLERANCE = 0.12

# Reduced-resolution gating must stay at least this much cheaper than
# full-resolution gating. A ratio is machine-independent, so it survives moving
# between a laptop, a CI runner and reference hardware — which is why it, not
# the absolute cost, is the load-bearing regression guard.
#
# Set from the guard's PURPOSE, not fitted to observations. A gate with the
# downscale removed scores exactly 1.00x; the job is to separate that from a
# working gate, with margin. Measured 2026-08-01 on the pinned stack:
#
#     static              3.37x
#     near target         3.35x
#     SMALL target        3.31x
#     real upscaled 720p  2.62x   <- real texture compresses the ratio
#
# The previous floor of 3.0 was calibrated when the scenario set was
# synthetic-only, and the real-footage scenario would have failed it — a guard
# fitted to an incomplete sample, not a real regression. 2.0 sits clear of the
# lowest genuine measurement and far above the 1.00x that removal produces.
MIN_SPEEDUP = 2.0


@dataclass(frozen=True)
class Scenario:
    """One benchmark sequence with a known correct answer."""

    name: str
    frames: list[np.ndarray]
    expected_wake_share: float | None
    note: str
    exercises_downscale: bool = True
    """Whether this scenario's source is above gate resolution.

    ``False`` for footage already at or below it, where the downscale is a
    no-op by construction and the speedup is ~1.00x legitimately. Those
    scenarios are excluded from the speedup guard — including them would make
    the guard fire on correct behaviour, and a guard that cries wolf gets
    lowered until it is useless.
    """

    def __len__(self) -> int:
        return len(self.frames)


def _textured_background(
    height: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.integers(60, 190, size=(height, width, 3), dtype=np.uint8)


def moving_blob_scenario(
    name: str,
    seconds: int,
    fps: int,
    width: int,
    height: int,
    static_seconds: int,
    blob_px: int,
    seed: int,
    note: str,
) -> Scenario:
    """A static textured scene, then a blob of ``blob_px`` height crossing it.

    The background carries mild per-frame sensor noise. A flat noiseless
    background would make the benchmark trivial and would never exercise
    ``min_foreground_fraction``, which is the parameter deciding whether the
    gate fires on grain.
    """
    rng = np.random.default_rng(seed)
    background = _textured_background(height, width, rng)
    total = seconds * fps
    motion_starts = static_seconds * fps

    frames: list[np.ndarray] = []
    for index in range(total):
        frame = np.clip(
            background.astype(np.int16)
            + rng.integers(-4, 5, size=background.shape, dtype=np.int16),
            0,
            255,
        ).astype(np.uint8)
        if index >= motion_starts:
            travelled = index - motion_starts
            moving_frames = max(1, total - motion_starts)
            x = int((width - blob_px) * (travelled / moving_frames))
            y = height // 2 - blob_px // 2
            frame[y : y + blob_px, x : x + blob_px] = 255
        frames.append(frame[np.newaxis, ...])

    return Scenario(
        name=name,
        frames=frames,
        expected_wake_share=(total - motion_starts) / total,
        note=note,
    )


def static_scenario(seconds: int, fps: int, width: int, height: int) -> Scenario:
    """Nothing moves. The idle case the whole cascade exists for."""
    rng = np.random.default_rng(4242)
    background = _textured_background(height, width, rng)
    frames = [
        np.clip(
            background.astype(np.int16)
            + rng.integers(-4, 5, size=background.shape, dtype=np.int16),
            0,
            255,
        ).astype(np.uint8)[np.newaxis, ...]
        for _ in range(seconds * fps)
    ]
    return Scenario(
        name="static",
        frames=frames,
        expected_wake_share=0.0,
        note="no motion; sensor noise only",
    )


def real_video_scenario(
    max_frames: int, upscale_to: tuple[int, int] | None = None
) -> Scenario | None:
    """Frames from the repository's test video.

    No expected wake share: nobody has labelled this footage, so claiming one
    would be inventing ground truth. It is here for the parity check, which
    needs no labels — only that both gate resolutions agree.

    ``upscale_to`` exists because of a measurement defect found on day 6. The
    source is 320x176, which is already BELOW the 320x180 gate resolution, so
    the downscale is a no-op and both "resolutions" process a bit-identical
    raster. That scenario reported EXACT parity while testing nothing about
    downscaling at all — a vacuous pass that reads exactly like a real one.

    Upscaling to 720p first puts real image statistics — sensor noise,
    compression artefacts, natural texture — through the actual downscale path.
    It is not native 720p footage and is labelled accordingly; the honest fix
    is real footage from a 720p camera, which Site Zero will provide.
    """
    try:
        import cv2
    except ImportError:
        return None
    if not REAL_VIDEO.exists():
        return None

    capture = cv2.VideoCapture(str(REAL_VIDEO))
    if not capture.isOpened():
        return None
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if upscale_to is not None:
                frame = cv2.resize(frame, upscale_to, interpolation=cv2.INTER_CUBIC)
            frames.append(frame[np.newaxis, ...])
    finally:
        capture.release()

    if len(frames) < 10:
        return None

    height, width = frames[0].shape[1], frames[0].shape[2]
    if upscale_to is None:
        name = "real_native"
        note = (
            f"{len(frames)} frames of real footage at {width}x{height}. NOTE: "
            "already at or below gate resolution, so the downscale is a no-op "
            "and parity here is vacuous — retained only to show the raw path "
            "runs."
        )
    else:
        name = "REAL_UPSCALED_720p"
        note = (
            f"{len(frames)} frames of real footage upscaled to {width}x{height}; "
            "real image statistics through the actual downscale path. Not "
            "native 720p — Site Zero supplies that."
        )
    return Scenario(
        name=name,
        frames=frames,
        expected_wake_share=None,
        note=note,
        exercises_downscale=upscale_to is not None,
    )


def run_gate(scenario: Scenario, config: MotionGateConfig) -> tuple[list[bool], float]:
    """Run one gate over a scenario exactly once per frame.

    Deliberately does not go through :class:`CascadeRunner`. An earlier version
    called both, which ran the gate twice per frame — inflating the measured
    cost and, worse, feeding every frame to the MOG2 background model twice, so
    the model was learning from a sequence the scenario never contained.

    Returns:
        ``(wake decisions per frame, p50 ms per frame)``.
    """
    gate = MotionGate(config)
    decisions: list[bool] = []
    durations_ms: list[float] = []

    for index, frame in enumerate(scenario.frames):
        ctx = StageContext(frame_index=index, ts_ns=index * NANOSECONDS_PER_SECOND)
        start = time.perf_counter()
        output = gate.process(frame, ctx)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
        decisions.append(output.wake_next)

    return decisions, float(np.percentile(durations_ms, 50))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--static-seconds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help=(
            "multiplier on the core budget, for machines that are not the "
            "reference hardware. The Tier-1 budget is defined on a dedicated "
            "NUC-class box; a shared CI runner or a laptop with a browser open "
            "measures slower for reasons that have nothing to do with this "
            "code. Raising this does NOT change the product budget, and the "
            "unscaled number is always printed."
        ),
    )
    args = parser.parse_args(argv)

    config = IronConfig.load()
    gate_config = config.cascade.motion_gate_config()
    full_res_config = MotionGateConfig(
        min_foreground_fraction=gate_config.min_foreground_fraction,
        stay_awake_frames=gate_config.stay_awake_frames,
        diff_threshold=gate_config.diff_threshold,
        warmup_frames=gate_config.warmup_frames,
        gate_width=0,  # downscaling disabled
        gate_height=0,
    )

    # A person ~60 px tall at 720p lands at ~15 px once gated at 320x180.
    small_px = max(8, round(args.height * 15 / config.cascade.gate_height))
    large_px = max(small_px * 3, args.height // 4)

    scenarios: list[Scenario] = [
        static_scenario(args.seconds // 3, args.fps, args.width, args.height),
        moving_blob_scenario(
            "near_target",
            args.seconds,
            args.fps,
            args.width,
            args.height,
            args.static_seconds,
            large_px,
            args.seed,
            f"{large_px}px blob at source = "
            f"~{round(large_px * config.cascade.gate_height / args.height)}px at gate",
        ),
        moving_blob_scenario(
            "SMALL_TARGET",
            args.seconds,
            args.fps,
            args.width,
            args.height,
            args.static_seconds,
            small_px,
            args.seed + 1,
            f"{small_px}px blob at source = ~15px at gate "
            "(distant person; the case downscaling could lose)",
        ),
    ]
    # Both: the native clip (vacuous parity, shows the path runs) and the
    # upscaled one (genuine parity through the downscale).
    for real in (
        real_video_scenario(args.seconds * args.fps),
        real_video_scenario(
            args.seconds * args.fps, upscale_to=(args.width, args.height)
        ),
    ):
        if real is not None:
            scenarios.append(real)

    print(f"Gate resolution : {config.cascade.gate_width}x{config.cascade.gate_height}")
    print(f"Source          : {args.width}x{args.height} at {args.fps} fps")
    print(
        f"Budget          : {config.cascade.idle_core_budget_fraction:.1%} of one core"
    )
    print(f"Backend         : {MotionGate(gate_config).backend}")
    print()

    frame_budget_ms = 1000.0 / args.fps
    failures: list[str] = []
    small_target_diverged = False
    worst_cost_share = 0.0
    speedups: list[float] = []
    scenario_speedups: dict[str, float] = {}

    header = (
        f"{'scenario':<18} {'frames':>7} {'wake@full':>10} {'wake@gate':>10} "
        f"{'parity':>8} {'p50 ms':>8} {'% core':>8}"
    )
    print(header)
    print("-" * len(header))

    for scenario in scenarios:
        full_decisions, full_ms = run_gate(scenario, full_res_config)
        gate_decisions, gate_ms = run_gate(scenario, gate_config)

        divergences = [
            index
            for index, (a, b) in enumerate(zip(full_decisions, gate_decisions))
            if a != b
        ]
        parity = "EXACT" if not divergences else f"{len(divergences)} differ"

        full_share = sum(full_decisions) / len(full_decisions)
        gate_share = sum(gate_decisions) / len(gate_decisions)
        core_share = gate_ms / frame_budget_ms
        worst_cost_share = max(worst_cost_share, core_share)
        if gate_ms > 0:
            scenario_speedups[scenario.name] = full_ms / gate_ms
            if scenario.exercises_downscale:
                speedups.append(full_ms / gate_ms)

        print(
            f"{scenario.name:<18} {len(scenario):>7} {full_share * 100:>9.1f}% "
            f"{gate_share * 100:>9.1f}% {parity:>8} {gate_ms:>8.3f} "
            f"{core_share * 100:>7.2f}%"
        )
        print(f"                   {scenario.note}")

        if divergences:
            detail = (
                f"{scenario.name}: {len(divergences)} of {len(scenario)} wake "
                f"decisions differ between full-resolution and "
                f"{config.cascade.gate_width}x{config.cascade.gate_height} "
                f"gating (first at frame {divergences[0]})"
            )
            failures.append(detail)
            if scenario.name == "SMALL_TARGET":
                small_target_diverged = True

        if scenario.expected_wake_share is not None:
            low = scenario.expected_wake_share - TOLERANCE
            high = scenario.expected_wake_share + TOLERANCE
            if not low <= gate_share <= high:
                failures.append(
                    f"{scenario.name}: woke on {gate_share:.1%} of frames, "
                    f"outside the expected {low:.1%}-{high:.1%}"
                )

    budget = config.cascade.idle_core_budget_fraction
    ceiling = config.cascade.regression_ceiling_fraction
    effective_ceiling = ceiling * args.budget_scale
    median_speedup = float(np.median(speedups)) if speedups else 0.0
    # Gate on the MINIMUM, not the median. A median stays comfortable while one
    # scenario's downscale collapses, and the collapsed scenario is exactly the
    # one that would have caught the regression.
    min_speedup = float(np.min(speedups)) if speedups else 0.0

    print()
    from src.provenance import current_stack_string

    print(f"Worst stage-0 cost : {worst_cost_share:.2%} of one core per camera")
    print(f"Measured on        : {current_stack_string()}")
    print(f"PRODUCT BUDGET     : {budget:.2%}  <- what Tier-1 needs")
    print(f"Regression ceiling : {effective_ceiling:.2%}  <- what CI gates on")
    if args.budget_scale != 1.0:
        print(f"                     (x{args.budget_scale:g} for shared runners)")
    print(
        f"Speedup vs full-res: {min_speedup:.2f}x min / {median_speedup:.2f}x "
        f"median (floor {MIN_SPEEDUP:.1f}x on the MIN)"
    )
    gated = {s.name for s in scenarios if s.exercises_downscale}
    for name, ratio in sorted(scenario_speedups.items(), key=lambda kv: kv[1]):
        mark = "" if name in gated else "   (no-op: source below gate res)"
        print(f"    {name:22} {ratio:5.2f}x{mark}")

    # Two separate verdicts. The product budget is a claim about what the
    # product needs and is reported whether or not it is met; the ceiling is a
    # regression gate. Collapsing them would let CI go green by moving the
    # target, which is how a missed budget quietly becomes a met one.
    print()
    if worst_cost_share > budget:
        print(
            f"PRODUCT BUDGET MISSED: {worst_cost_share:.2%} against a "
            f"{budget:.2%} target ({worst_cost_share / budget:.2f}x over).\n"
            "This is reported, not gated. Thresholds are NOT retuned to close "
            "it —\nthe remedy is a separate measured change."
        )
    else:
        print(f"Product budget MET: {worst_cost_share:.2%} within {budget:.2%}.")

    if worst_cost_share > effective_ceiling:
        failures.append(
            f"stage-0 cost {worst_cost_share:.2%} exceeds the "
            f"{effective_ceiling:.2%} REGRESSION CEILING — stage 0 got worse "
            "than its last measurement, which is a regression regardless of "
            "the product budget"
        )
    if min_speedup < MIN_SPEEDUP:
        worst = min(
            ((n, r) for n, r in scenario_speedups.items() if n in gated),
            key=lambda kv: kv[1],
        )
        failures.append(
            f"scenario {worst[0]!r} gates only {worst[1]:.2f}x cheaper than "
            f"full-resolution, below the {MIN_SPEEDUP:.1f}x floor — the "
            "downscale may have been removed or defeated (removal scores 1.00x)"
        )

    print()
    if small_target_diverged:
        print(
            "STOP: the SMALL_TARGET scenario diverged.\n"
            "\n"
            "Reduced-resolution gating changed the wake decision for a distant\n"
            "target — the case where a missed wake loses an event outright.\n"
            "Do NOT retune min_foreground_fraction to force agreement: a\n"
            "threshold tuned until the numbers match proves only that it was\n"
            "tuned. Investigate the downscale filter and the target's pixel\n"
            "footprint at gate resolution first.",
            file=sys.stderr,
        )

    if failures:
        print("FAIL:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"\nPASS (no regression): wake decisions identical at both resolutions "
        f"across all {len(scenarios)} scenarios; stage-0 cost "
        f"{worst_cost_share:.2%} within the {effective_ceiling:.2%} ceiling; "
        f"{min_speedup:.2f}x cheaper than full-resolution gating at worst.\n"
        "Note: 'no regression' is not 'budget met' — see the product-budget "
        "verdict above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
