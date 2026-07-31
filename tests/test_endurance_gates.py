"""Tests for the endurance gates and harness control flow.

These need no model weights, which is the point of extracting the gate logic
out of the inference loop: previously the only way to exercise "does this
detect a leak?" was to run the thing the gate was meant to check.

The synthetic series are chosen to separate the two ways a leak gate fails in
practice — missing a real leak, and firing on ordinary allocator noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import IronConfig
from src.endurance.gates import (
    Verdict,
    evaluate_leak,
    summarize_latency,
)
from src.endurance.memory import MemorySampler, platform_fidelity_note
from src.endurance.runner import (
    DETERMINISM_COSINE_TOLERANCE,
    ExitCode,
    WeightsUnavailable,
    check_determinism,
    execute,
)

# One second per clip makes MB/clip and MB/hour convert by exactly 3600,
# so the expected numbers in these tests are checkable by hand.
ONE_SECOND = 1.0
THRESHOLD = 50.0


# ---------------------------------------------------------------------------
# The three required series: flat passes, a ramp fails, noisy-flat passes.
# ---------------------------------------------------------------------------


def test_flat_series_passes() -> None:
    """Perfectly stable memory must pass."""
    verdict = evaluate_leak([500.0] * 60, ONE_SECOND, THRESHOLD)
    assert verdict.verdict is Verdict.PASS, verdict.summary()
    assert verdict.slope_mb_per_clip == pytest.approx(0.0, abs=1e-9)


def test_one_mb_per_clip_ramp_fails() -> None:
    """A 1 MB/clip ramp is 3600 MB/hour and must fail unambiguously."""
    series = [500.0 + index for index in range(60)]
    verdict = evaluate_leak(series, ONE_SECOND, THRESHOLD)
    assert verdict.verdict is Verdict.FAIL, verdict.summary()
    assert verdict.slope_mb_per_clip == pytest.approx(1.0, rel=1e-6)
    assert verdict.mb_per_hour == pytest.approx(3600.0, rel=1e-6)


def test_noisy_flat_series_passes() -> None:
    """Zero-mean noise on a flat baseline must not be reported as a leak.

    This is the false-positive case that makes a gate get switched off. The
    noise amplitude here (+/-15 MB) is far larger than the growth the gate
    detects, so passing requires the regression to actually separate trend from
    scatter rather than reacting to the spread.
    """
    rng = np.random.default_rng(20260731)
    series = [500.0 + float(rng.normal(0.0, 15.0)) for _ in range(200)]
    verdict = evaluate_leak(series, ONE_SECOND, THRESHOLD)
    assert verdict.verdict is Verdict.PASS, verdict.summary()


# ---------------------------------------------------------------------------
# Properties of the gate itself.
# ---------------------------------------------------------------------------


def test_uses_confidence_bound_not_point_estimate() -> None:
    """Noise must make the gate stricter, never more permissive.

    A gate on the point estimate lets a noisy run hide a real leak inside its
    own error bars. Gating the CI upper bound means uncertainty counts against
    the run.
    """
    verdict = evaluate_leak([500.0 + i for i in range(40)], ONE_SECOND, THRESHOLD)
    assert verdict.mb_per_hour_ci_upper >= verdict.mb_per_hour


def test_a_small_leak_hidden_in_noise_is_still_caught() -> None:
    """0.05 MB/clip is 180 MB/hour and must fail even under heavy noise."""
    rng = np.random.default_rng(7)
    series = [
        500.0 + 0.05 * index + float(rng.normal(0.0, 3.0)) for index in range(400)
    ]
    verdict = evaluate_leak(series, ONE_SECOND, THRESHOLD)
    assert verdict.verdict is Verdict.FAIL, verdict.summary()


def test_shrinking_memory_is_not_a_failure() -> None:
    series = [900.0 - index for index in range(50)]
    assert evaluate_leak(series, ONE_SECOND, THRESHOLD).verdict is Verdict.PASS


def test_throughput_scales_the_verdict() -> None:
    """The same MB/clip slope is a different MB/hour at a different clip rate.

    This is why the gate takes measured seconds-per-clip rather than assuming
    one. A slow pipeline leaking 1 MB/clip leaks far less per hour than a fast
    one doing the same.
    """
    series = [500.0 + 0.02 * index for index in range(100)]
    fast = evaluate_leak(series, seconds_per_clip=0.5, threshold_mb_per_hour=THRESHOLD)
    slow = evaluate_leak(series, seconds_per_clip=60.0, threshold_mb_per_hour=THRESHOLD)
    assert fast.mb_per_hour > slow.mb_per_hour
    assert fast.verdict is Verdict.FAIL
    assert slow.verdict is Verdict.PASS


def test_old_two_magic_number_logic_would_have_missed_this() -> None:
    """Regression guard for the gate this replaced.

    The previous rule failed only when total growth exceeded 150 MB AND the
    last-10-vs-first-10 trend exceeded 30 MB. A steady 0.5 MB/clip leak over 40
    clips grows 20 MB total, so it passed both — while being 1800 MB/hour.
    """
    series = [500.0 + 0.5 * index for index in range(40)]
    total_growth = series[-1] - series[0]
    assert total_growth < 150.0, "premise of this test no longer holds"

    verdict = evaluate_leak(series, ONE_SECOND, THRESHOLD)
    assert verdict.verdict is Verdict.FAIL
    assert verdict.mb_per_hour == pytest.approx(1800.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Inconclusive is neither a pass nor a failure.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 2])
def test_too_few_samples_is_inconclusive(count: int) -> None:
    verdict = evaluate_leak([500.0] * count, ONE_SECOND, THRESHOLD)
    assert verdict.verdict is Verdict.INCONCLUSIVE
    assert "at least" in verdict.reason


def test_unknown_throughput_is_inconclusive() -> None:
    verdict = evaluate_leak([500.0, 501.0, 502.0], 0.0, THRESHOLD)
    assert verdict.verdict is Verdict.INCONCLUSIVE


def test_non_finite_readings_are_inconclusive() -> None:
    verdict = evaluate_leak([500.0, float("nan"), 502.0], ONE_SECOND, THRESHOLD)
    assert verdict.verdict is Verdict.INCONCLUSIVE


def test_inconclusive_is_not_reported_as_failed() -> None:
    """A short run must not look like a regression on a dashboard."""
    assert evaluate_leak([500.0], ONE_SECOND, THRESHOLD).failed is False


# ---------------------------------------------------------------------------
# Latency summary.
# ---------------------------------------------------------------------------


def test_latency_percentiles() -> None:
    summary = summarize_latency([index / 1000.0 for index in range(1, 101)])
    assert summary.count == 100
    assert summary.p50_ms == pytest.approx(50.5, rel=1e-6)
    assert summary.max_ms == pytest.approx(100.0, rel=1e-6)
    assert summary.p95_ms < summary.p99_ms < summary.max_ms


def test_latency_tail_is_visible_where_a_mean_would_hide_it() -> None:
    """The case percentiles exist for: a fine mean over an unusable tail."""
    durations = [0.040] * 99 + [0.900]
    summary = summarize_latency(durations)
    assert summary.mean_ms < 50.0
    assert summary.max_ms == pytest.approx(900.0, rel=1e-6)


def test_empty_latency_is_reported_as_empty() -> None:
    assert summarize_latency([]).count == 0


def test_negative_duration_raises() -> None:
    """A backwards clock invalidates every derived number; say so."""
    with pytest.raises(ValueError, match="backwards"):
        summarize_latency([0.1, -0.2])


# ---------------------------------------------------------------------------
# Determinism check.
# ---------------------------------------------------------------------------


class _FakeExtractor:
    """Returns a scripted sequence of embeddings, one per extract() call."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self._outputs = outputs
        self.calls = 0

    def extract(self, video: np.ndarray) -> dict[str, np.ndarray]:
        output = self._outputs[min(self.calls, len(self._outputs) - 1)]
        self.calls += 1
        return {"semantic_tracks": output}


def test_determinism_reports_byte_identical() -> None:
    embeddings = np.linspace(0.0, 1.0, 2 * 3 * 8).reshape(1, 2, 3, 8)
    result = check_determinism(_FakeExtractor([embeddings, embeddings]), np.zeros(1))
    assert result.passed
    assert result.byte_identical
    assert "byte-identical" in result.detail


def test_determinism_reports_cosine_when_not_byte_identical() -> None:
    """Tiny reduction-order jitter must pass, and be reported as cosine.

    INT8 kernels reorder reductions with thread count, so byte-identity is not
    guaranteed. The distinction matters to anyone trying to reproduce a number
    on different hardware.
    """
    base = np.full((1, 2, 3, 8), 0.5)
    jittered = base + 1e-12
    result = check_determinism(_FakeExtractor([base, jittered]), np.zeros(1))
    assert result.passed
    assert not result.byte_identical
    assert result.max_cosine_distance <= DETERMINISM_COSINE_TOLERANCE
    assert "cosine" in result.detail


def test_determinism_fails_on_real_divergence() -> None:
    first = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, 2, 3, 1))
    second = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (1, 2, 3, 1))
    result = check_determinism(_FakeExtractor([first, second]), np.zeros(1))
    assert not result.passed
    assert result.max_cosine_distance > DETERMINISM_COSINE_TOLERANCE


def test_determinism_fails_on_shape_change() -> None:
    result = check_determinism(
        _FakeExtractor([np.zeros((1, 2, 3, 8)), np.zeros((1, 2, 3, 4))]), np.zeros(1)
    )
    assert not result.passed
    assert "shape changed" in result.detail


# ---------------------------------------------------------------------------
# End-to-end control flow, with an injected extractor (no weights needed).
# ---------------------------------------------------------------------------


class _StableExtractor:
    def extract(self, video: np.ndarray) -> dict[str, np.ndarray]:
        return {"semantic_tracks": np.zeros((1, 2, 4, 8), dtype=np.float64) + 0.25}


class _ExplodingExtractor:
    def __init__(self, fail_on_call: int) -> None:
        self._fail_on = fail_on_call
        self.calls = 0

    def extract(self, video: np.ndarray) -> dict[str, np.ndarray]:
        self.calls += 1
        if self.calls == self._fail_on:
            raise RuntimeError("synthetic inference failure")
        return {"semantic_tracks": np.zeros((1, 2, 4, 8), dtype=np.float64) + 0.25}


@pytest.fixture()
def scoped_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IronConfig:
    """A config whose logs land in a temporary directory."""
    monkeypatch.setenv("IRON_PATHS__LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("IRON_PIPELINE__CLIP_H", "16")
    monkeypatch.setenv("IRON_PIPELINE__CLIP_W", "16")
    return IronConfig.load()


def test_execute_completes_and_writes_jsonl(scoped_config: IronConfig) -> None:
    code = execute(scoped_config, mode="steady", iterations=6, factory=_StableExtractor)
    assert code == ExitCode.OK

    metrics_path = (
        scoped_config.paths.resolved_log_dir / scoped_config.endurance.metrics_filename
    )
    lines = [line for line in metrics_path.read_text().splitlines() if line]
    assert len(lines) == 6

    import json

    records = [json.loads(line) for line in lines]
    assert [record["clip"] for record in records] == [1, 2, 3, 4, 5, 6]
    assert all(record["config_sha"] == scoped_config.config_sha() for record in records)
    assert records[0]["phase"] == "warmup"
    assert records[1]["phase"] == "measured"


def test_any_exception_aborts_with_nonzero_exit(scoped_config: IronConfig) -> None:
    """No tolerance counter: the first failure ends the run.

    The previous harness allowed five errors before aborting, so it could
    report memory stability for a pipeline that was not producing output.
    """
    extractor = _ExplodingExtractor(fail_on_call=3)
    code = execute(
        scoped_config, mode="steady", iterations=20, factory=lambda: extractor
    )
    assert code == ExitCode.PIPELINE_RAISED
    assert extractor.calls == 3, "run continued past the failing clip"


def test_traceback_reaches_the_log(scoped_config: IronConfig) -> None:
    execute(
        scoped_config,
        mode="steady",
        iterations=10,
        factory=lambda: _ExplodingExtractor(fail_on_call=2),
    )
    log_text = (
        scoped_config.paths.resolved_log_dir / scoped_config.endurance.log_filename
    ).read_text()
    assert "FATAL" in log_text
    assert "synthetic inference failure" in log_text
    assert "Traceback" in log_text


def test_missing_weights_exits_two_without_a_traceback(
    scoped_config: IronConfig,
) -> None:
    """A missing checkpoint is an environment problem, not a regression."""

    def refusing_factory() -> _StableExtractor:
        raise WeightsUnavailable("weights not found at /models/vjepa2_int8.xml")

    code = execute(scoped_config, mode="steady", iterations=5, factory=refusing_factory)
    assert code == ExitCode.PREREQUISITES_MISSING

    log_text = (
        scoped_config.paths.resolved_log_dir / scoped_config.endurance.log_filename
    ).read_text()
    assert "weights not found at" in log_text
    assert "Traceback" not in log_text


def test_reinit_mode_rebuilds_the_extractor_each_cycle(
    scoped_config: IronConfig,
) -> None:
    """The load/unload path is where framework leaks actually live."""
    built = 0

    def counting_factory() -> _StableExtractor:
        nonlocal built
        built += 1
        return _StableExtractor()

    code = execute(scoped_config, mode="reinit", iterations=5, factory=counting_factory)
    assert code == ExitCode.OK
    assert built == 5


def test_unknown_mode_is_rejected(scoped_config: IronConfig) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        execute(scoped_config, mode="sideways", iterations=3, factory=_StableExtractor)


# ---------------------------------------------------------------------------
# Memory sampler.
# ---------------------------------------------------------------------------


def test_sampler_returns_a_usable_rss() -> None:
    sample = MemorySampler(track_allocations=False).sample()
    assert sample.rss_mb > 0


def test_unavailable_metrics_are_none_not_zero() -> None:
    """None means "not measured"; 0.0 would be a measurement that never happened."""
    sample = MemorySampler(track_allocations=False).sample()
    for value in (sample.uss_mb, sample.pss_mb):
        assert value is None or value > 0


def test_platform_note_states_what_is_available() -> None:
    note = platform_fidelity_note()
    assert "RSS" in note
    assert "USS/PSS" in note or "unavailable" in note


def test_allocation_diff_needs_a_baseline() -> None:
    sampler = MemorySampler(track_allocations=True)
    try:
        assert sampler.allocation_diff() == []
        _ = [object() for _ in range(1000)]
        assert isinstance(sampler.allocation_diff(), list)
    finally:
        sampler.stop()
