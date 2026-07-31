"""Tests for the cascade runtime.

The properties that decide whether the Tier-1 budget is reachable: a static
scene must wake nothing, a moving one must wake, hysteresis must hold the gate
open briefly after motion stops, and the wake accounting must count frames a
stage was offered rather than only the ones it ran on.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from src.cascade import (
    CascadeRunner,
    DetectorStage,
    FrameDifferenceBackend,
    JsonlStatsSink,
    MotionGate,
    MotionGateConfig,
    SemanticsStage,
    Stage,
    StageContext,
    StageOutput,
    TrackerStage,
)

WIDTH, HEIGHT = 160, 120


def gray_frame(value: int = 100) -> np.ndarray:
    return np.full((1, HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def frame_with_blob(offset: int, size: int = 40) -> np.ndarray:
    frame = gray_frame()
    frame[0, 20 : 20 + size, offset : offset + size] = 255
    return frame


def ctx(index: int) -> StageContext:
    return StageContext(frame_index=index, ts_ns=index * 83_000_000)


def deterministic_gate(**overrides: object) -> MotionGate:
    """A gate on the frame-difference backend.

    MOG2 carries adaptive internal state whose exact behaviour varies with the
    OpenCV build, which would make these assertions a test of OpenCV rather
    than of the gate logic.
    """
    defaults: dict[str, object] = {"warmup_frames": 1, "stay_awake_frames": 3}
    defaults.update(overrides)
    config = MotionGateConfig(**defaults)  # type: ignore[arg-type]
    return MotionGate(config, backend=FrameDifferenceBackend(config.diff_threshold))


# ---------------------------------------------------------------------------
# Motion gate
# ---------------------------------------------------------------------------


def test_static_input_wakes_nothing() -> None:
    """The case the whole cascade exists for: an idle camera costs stage 0 only."""
    gate = deterministic_gate()
    wakes = 0
    for index in range(40):
        output = gate.process(gray_frame(), ctx(index))
        wakes += int(output.wake_next)
    assert wakes == 0


def test_moving_input_wakes() -> None:
    gate = deterministic_gate()
    gate.process(frame_with_blob(0), ctx(0))  # warm-up frame
    output = gate.process(frame_with_blob(50), ctx(1))
    assert output.wake_next
    assert output.detail["moving"] is True


def test_warmup_suppresses_the_first_frames() -> None:
    """An unbuilt background model sees foreground everywhere.

    Without this, every stream would wake the whole cascade at startup and on
    every reconnect.
    """
    gate = deterministic_gate(warmup_frames=5)
    for index in range(5):
        assert gate.process(frame_with_blob(index * 10), ctx(index)).wake_next is False


# Motion stopping is simulated by REPEATING the last frame, not by removing the
# blob. Frame differencing compares against the previous frame, so a blob
# vanishing is itself a large change and correctly reads as motion. An object
# that stopped but is still in shot produces no difference at all, which is the
# situation hysteresis exists for.


def test_hysteresis_holds_the_gate_open_after_motion_stops() -> None:
    """Flapping costs more than staying awake, and fragments tracks."""
    gate = deterministic_gate(stay_awake_frames=4)
    gate.process(gray_frame(), ctx(0))
    assert gate.process(frame_with_blob(40), ctx(1)).wake_next

    stopped = frame_with_blob(40)
    held = [gate.process(stopped, ctx(index)).wake_next for index in range(2, 8)]

    assert held[:4] == [True, True, True, True], "hysteresis did not hold"
    assert held[4:] == [False, False], "hysteresis did not expire"


def test_hysteresis_countdown_is_reported() -> None:
    gate = deterministic_gate(stay_awake_frames=3)
    gate.process(gray_frame(), ctx(0))
    gate.process(frame_with_blob(40), ctx(1))
    assert gate.awake_remaining == 3

    output = gate.process(frame_with_blob(40), ctx(2))
    assert output.detail["held_awake_by_hysteresis"] is True
    assert output.detail["moving"] is False
    assert gate.awake_remaining == 2


def test_renewed_motion_resets_the_countdown() -> None:
    gate = deterministic_gate(stay_awake_frames=3)
    gate.process(gray_frame(), ctx(0))
    gate.process(frame_with_blob(20), ctx(1))
    gate.process(frame_with_blob(20), ctx(2))  # stopped: countdown ticks
    assert gate.awake_remaining == 2
    gate.process(frame_with_blob(80), ctx(3))  # moved again: countdown reset
    assert gate.awake_remaining == 3


def test_zero_hysteresis_sleeps_immediately() -> None:
    gate = deterministic_gate(stay_awake_frames=0)
    gate.process(gray_frame(), ctx(0))
    assert gate.process(frame_with_blob(40), ctx(1)).wake_next
    assert gate.process(frame_with_blob(40), ctx(2)).wake_next is False


def test_a_disappearing_object_is_motion() -> None:
    """Pins the behaviour that made the hysteresis tests above subtle.

    An object leaving the frame is a real event, not an absence of one, and the
    gate must wake for it.
    """
    gate = deterministic_gate()
    gate.process(frame_with_blob(40), ctx(0))
    gate.process(frame_with_blob(40), ctx(1))
    assert gate.process(gray_frame(), ctx(2)).detail["moving"] is True


def test_gate_writes_the_motion_fraction_into_context() -> None:
    gate = deterministic_gate()
    context = ctx(0)
    gate.process(gray_frame(), context)
    assert "motion_fraction" in context.payload


def test_gate_reports_its_backend() -> None:
    """A benchmark must never silently compare two different estimators."""
    assert deterministic_gate().backend == "frame-difference"


def test_frame_difference_reports_no_motion_without_a_baseline() -> None:
    """Claiming motion on frame one would wake every stream at startup."""
    backend = FrameDifferenceBackend(threshold=25)
    assert backend.foreground_fraction(np.zeros((10, 10), dtype=np.uint8)) == 0.0


def test_gate_rejects_malformed_frames() -> None:
    gate = deterministic_gate()
    with pytest.raises(ValueError, match=r"\[N, H, W\]"):
        gate.process(np.zeros((10, 10), dtype=np.uint8), ctx(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_foreground_fraction": -0.1},
        {"min_foreground_fraction": 1.5},
        {"stay_awake_frames": -1},
        {"diff_threshold": 300},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MotionGateConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Runner short-circuit and accounting
# ---------------------------------------------------------------------------


class _RecordingStage:
    """A stage that counts its own invocations and wakes on demand."""

    def __init__(self, name: str, wake_next: bool = True) -> None:
        self._name = name
        self._wake_next = wake_next
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def process(self, frame_batch: np.ndarray, context: StageContext) -> StageOutput:
        self.calls += 1
        return StageOutput(woke=True, wake_next=self._wake_next)

    def should_wake_next(self, output: StageOutput) -> bool:
        return output.wake_next


def test_short_circuit_skips_every_later_stage() -> None:
    """The saving that makes the idle budget arithmetic work."""
    first = _RecordingStage("first", wake_next=False)
    second = _RecordingStage("second")
    third = _RecordingStage("third")
    runner = CascadeRunner([first, second, third], stats_interval_s=1e9)

    for index in range(25):
        assert runner.process_frame(gray_frame(), ctx(index)) == 1

    assert first.calls == 25
    assert second.calls == 0
    assert third.calls == 0


def test_all_stages_run_when_every_gate_opens() -> None:
    stages = [_RecordingStage(f"s{i}") for i in range(3)]
    runner = CascadeRunner(stages, stats_interval_s=1e9)
    assert runner.process_frame(gray_frame(), ctx(0)) == 3
    assert all(stage.calls == 1 for stage in stages)


def test_skipped_stages_still_count_the_frame() -> None:
    """Wake fraction must be measured against frames offered, not frames run.

    Otherwise a stage that runs twice in a thousand frames reports a 100% wake
    rate, and the budget looks fine right up until it is not.
    """
    runner = CascadeRunner(
        [_RecordingStage("gate", wake_next=False), _RecordingStage("heavy")],
        stats_interval_s=1e9,
    )
    for index in range(10):
        runner.process_frame(gray_frame(), ctx(index))

    gate_stats, heavy_stats = runner.stats()
    assert gate_stats.frames_seen == 10
    assert gate_stats.wakes == 10
    assert gate_stats.wake_fraction == 1.0

    assert heavy_stats.frames_seen == 10, "skipped frames were not counted"
    assert heavy_stats.wakes == 0
    assert heavy_stats.wake_fraction == 0.0


def test_latency_percentiles_exclude_skipped_frames() -> None:
    """Zero-cost samples for skipped frames would make a heavy stage look cheap."""
    runner = CascadeRunner(
        [_RecordingStage("gate", wake_next=False), _RecordingStage("heavy")],
        stats_interval_s=1e9,
    )
    for index in range(5):
        runner.process_frame(gray_frame(), ctx(index))
    assert runner.stats()[1].p99_ms == 0.0


def test_woke_next_is_counted() -> None:
    runner = CascadeRunner([_RecordingStage("gate")], stats_interval_s=1e9)
    for index in range(7):
        runner.process_frame(gray_frame(), ctx(index))
    assert runner.stats()[0].woke_next == 7


def test_runner_requires_at_least_one_stage() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        CascadeRunner([])


def test_stats_are_emitted_as_jsonl() -> None:
    buffer = io.StringIO()
    runner = CascadeRunner(
        [_RecordingStage("gate", wake_next=False), _RecordingStage("heavy")],
        stats_interval_s=1e9,
        sink=JsonlStatsSink(buffer),
    )
    for index in range(12):
        runner.process_frame(gray_frame(), ctx(index))
    runner.flush()

    lines = [line for line in buffer.getvalue().splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["frames_processed"] == 12
    assert [s["stage"] for s in record["stages"]] == ["gate", "heavy"]
    assert record["stages"][0]["wake_fraction"] == 1.0
    assert record["stages"][1]["wake_fraction"] == 0.0


def test_stats_fps_math() -> None:
    runner = CascadeRunner([_RecordingStage("gate")], stats_interval_s=1e9)
    for index in range(30):
        runner.process_frame(gray_frame(), ctx(index))
    stats = runner.flush()
    assert stats.frames_processed == 30
    assert stats.frames_per_second > 0


# ---------------------------------------------------------------------------
# End-to-end with the real gate
# ---------------------------------------------------------------------------


def test_gate_in_a_runner_prevents_the_stub_from_raising() -> None:
    """A static scene must never reach the unimplemented detector.

    Doubles as proof that the short-circuit is real: if it were not, the stub
    would raise NotImplementedError and this test would fail loudly.
    """
    runner = CascadeRunner(
        [deterministic_gate(), DetectorStage()], stats_interval_s=1e9
    )
    for index in range(30):
        runner.process_frame(gray_frame(), ctx(index))
    assert runner.stats()[1].wakes == 0


def test_stubs_declare_themselves_unimplemented() -> None:
    for stage in (DetectorStage(), TrackerStage(), SemanticsStage()):
        with pytest.raises(NotImplementedError, match="stub"):
            stage.process(gray_frame(), ctx(0))


def test_stages_satisfy_the_protocol() -> None:
    for stage in (MotionGate(), DetectorStage(), TrackerStage(), SemanticsStage()):
        assert isinstance(stage, Stage)


def test_wake_rate_tracks_the_share_of_moving_frames() -> None:
    """The bench assertion in miniature: roughly a third moving, roughly a
    third waking, plus hysteresis."""
    gate = deterministic_gate(stay_awake_frames=2)
    runner = CascadeRunner([gate], stats_interval_s=1e9)

    total, moving_from = 90, 60
    for index in range(total):
        frame = (
            frame_with_blob((index - moving_from) * 2)
            if index >= moving_from
            else gray_frame()
        )
        runner.process_frame(frame, ctx(index))

    share = runner.stats()[0].woke_next / total
    assert 0.28 <= share <= 0.45, f"woke on {share:.1%} of frames"


# ---------------------------------------------------------------------------
# Reduced-resolution gating (day 2)
# ---------------------------------------------------------------------------


def test_gate_runs_at_the_configured_resolution() -> None:
    """A 720p frame must be gated at 320x180, not at 720p."""
    gate = MotionGate(MotionGateConfig(warmup_frames=1))
    gate.process(np.zeros((1, 720, 1280, 3), dtype=np.uint8), ctx(0))
    assert gate.gate_shape == (180, 320)


def test_small_source_is_not_upscaled() -> None:
    """Upscaling would invent detail and cost time for nothing."""
    gate = MotionGate(MotionGateConfig(warmup_frames=1))
    gate.process(np.zeros((1, 176, 320, 3), dtype=np.uint8), ctx(0))
    assert gate.gate_shape == (176, 320)


def test_downscaling_can_be_disabled() -> None:
    config = MotionGateConfig(warmup_frames=1, gate_width=0, gate_height=0)
    assert config.downscaling_enabled is False
    gate = MotionGate(config)
    gate.process(np.zeros((1, 480, 640, 3), dtype=np.uint8), ctx(0))
    assert gate.gate_shape == (480, 640)


def test_negative_gate_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        MotionGateConfig(gate_width=-1)


def test_downscale_preserves_a_small_moving_target() -> None:
    """The failure mode that would make this optimization unacceptable.

    Nearest-neighbour subsampling drops whole rows and columns, so a distant
    person can lose most of their pixels to the discard pattern and fall under
    the foreground threshold — the gate would sleep through exactly the events
    it exists to catch. Area averaging keeps the energy.
    """
    from src.cascade.motion import _downscale

    source = np.zeros((720, 1280), dtype=np.uint8)
    source[300:360, 600:660] = 255  # 60px target, as at 720p

    small = _downscale(source, 180, 320)
    assert small.shape == (180, 320)
    assert small.max() > 200, "the target did not survive the downscale"
    # 60px -> ~15px, so ~225 px of a 57,600 px frame: still over the 0.2% floor.
    assert np.count_nonzero(small > 128) >= 150


def test_greyscale_matches_the_numpy_channel_mean() -> None:
    """cv2.transform must compute the mean, not ITU-R luma.

    Luma weighting is channel-order dependent, and BGR/RGB confusion here is
    silent. Rounding may differ by one grey level; anything larger means the
    weights changed.
    """
    from src.cascade.motion import _to_gray

    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    reference = frame.mean(axis=2, dtype=np.float32).astype(np.uint8)
    produced = _to_gray(frame)
    assert np.abs(produced.astype(np.int16) - reference.astype(np.int16)).max() <= 1


def test_gate_resolution_is_config_driven() -> None:
    from src.config import IronConfig

    cascade = IronConfig.load().cascade
    produced = cascade.motion_gate_config()
    assert produced.gate_width == cascade.gate_width == 320
    assert produced.gate_height == cascade.gate_height == 180
