"""Tests for the benchmark environment gate (Day 19).

Day 18's cascade-bench figure moved 4.50% -> 5.10% -> 9.28% purely from
machine state (concurrent load, then battery throttling), and nothing in the
harness caught it before the numbers were printed. These tests pin the two
properties that close that gap: a bad snapshot refuses to start, and a
snapshot that goes bad mid-run invalidates the artifact rather than letting
it be written.

Environments are constructed directly via ``dataclasses.replace`` on a known
-clean baseline rather than by mocking ``pmset``/``psutil`` calls -- the
condition-detection helpers (``_power_state``, ``_throttle_state``) are
thin, mechanical parsers, and what actually needs a regression guard is the
gate's *decision* given a snapshot, which is the same regardless of which
platform produced it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.bench.environment import (
    BenchmarkEnvironment,
    BenchmarkGuard,
    BenchmarkInvalidated,
    BenchmarkRefused,
    EnvironmentMismatch,
    EnvironmentPair,
    GuardThresholds,
    write_benchmark_artifact,
)

CLEAN = BenchmarkEnvironment(
    captured_at_utc="2026-08-11T00:00:00+00:00",
    hostname="bench-host",
    platform="Darwin",
    on_ac_power=True,
    ac_power_detail="Now drawing from 'AC Power'",
    battery_percent=87.0,
    throttle_pct=100.0,
    throttle_detail="CPU_Speed_Limit=100 (pmset -g therm)",
    load_avg_1=0.5,
    load_avg_5=0.6,
    load_avg_15=0.7,
    core_count_logical=12,
    core_count_physical=8,
    high_cpu_process_count=0,
    high_cpu_process_names=(),
)


def test_clean_environment_has_no_violations() -> None:
    assert CLEAN.violations(GuardThresholds()) == []


def test_battery_power_is_a_violation() -> None:
    env = dataclasses.replace(
        CLEAN, on_ac_power=False, ac_power_detail="Now drawing from 'Battery Power'"
    )
    reasons = env.violations(GuardThresholds())
    assert any("AC power" in r for r in reasons)


def test_throttled_cpu_is_a_violation() -> None:
    env = dataclasses.replace(
        CLEAN, throttle_pct=46.0, throttle_detail="CPU_Speed_Limit=46 (pmset -g therm)"
    )
    reasons = env.violations(GuardThresholds())
    assert any("throttled" in r and "46" in r for r in reasons)


def test_high_load_average_is_a_violation() -> None:
    env = dataclasses.replace(CLEAN, load_avg_1=20.0)
    reasons = env.violations(GuardThresholds())
    assert any("load average" in r for r in reasons)


def test_competing_process_is_a_violation() -> None:
    env = dataclasses.replace(
        CLEAN, high_cpu_process_count=2, high_cpu_process_names=("pytest", "pytest")
    )
    reasons = env.violations(GuardThresholds())
    assert any("competing process" in r and "pytest" in r for r in reasons)


@pytest.mark.parametrize(
    "field,value",
    [
        ("on_ac_power", None),
        ("throttle_pct", None),
        ("load_avg_1", None),
    ],
)
def test_undeterminable_conditions_fail_closed(field: str, value: object) -> None:
    """An unmeasurable condition refuses, it does not pass by default."""
    env = dataclasses.replace(CLEAN, **{field: value})
    reasons = env.violations(GuardThresholds())
    assert reasons, f"{field}=None produced no violation -- an unmeasured risk is not an absent one"


# ---------------------------------------------------------------------------
# BenchmarkGuard.begin: refuses before any scenario runs
# ---------------------------------------------------------------------------


def test_guard_begin_refuses_on_bad_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = dataclasses.replace(CLEAN, on_ac_power=False)
    monkeypatch.setattr(BenchmarkEnvironment, "capture", classmethod(lambda cls, *a, **k: bad))

    guard = BenchmarkGuard()
    with pytest.raises(BenchmarkRefused, match="AC power"):
        guard.begin()


def test_guard_begin_passes_on_clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BenchmarkEnvironment, "capture", classmethod(lambda cls, *a, **k: CLEAN))

    guard = BenchmarkGuard()
    start = guard.begin()
    assert start == CLEAN


def test_guard_begin_message_names_every_failing_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = dataclasses.replace(CLEAN, on_ac_power=False, throttle_pct=46.0)
    monkeypatch.setattr(BenchmarkEnvironment, "capture", classmethod(lambda cls, *a, **k: bad))

    guard = BenchmarkGuard()
    with pytest.raises(BenchmarkRefused) as excinfo:
        guard.begin()
    message = str(excinfo.value)
    assert "AC power" in message
    assert "throttled" in message


# ---------------------------------------------------------------------------
# BenchmarkGuard.end: mid-run drift invalidates the run
# ---------------------------------------------------------------------------


def test_end_is_valid_when_nothing_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BenchmarkEnvironment, "capture", classmethod(lambda cls, *a, **k: CLEAN))
    guard = BenchmarkGuard()
    pair = guard.end(CLEAN)
    assert pair.valid
    assert pair.invalid_reason == ""


def test_end_detects_mid_run_throttle_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """This is the Day 18 case: the start looked clean, and the machine
    throttled partway through -- a start-only check would have missed it."""
    throttled_end = dataclasses.replace(
        CLEAN, throttle_pct=46.0, throttle_detail="CPU_Speed_Limit=46 (pmset -g therm)"
    )
    monkeypatch.setattr(
        BenchmarkEnvironment, "capture", classmethod(lambda cls, *a, **k: throttled_end)
    )

    guard = BenchmarkGuard()
    pair = guard.end(CLEAN)  # CLEAN stands in for the captured start snapshot

    assert not pair.valid
    assert "throttled during the run" in pair.invalid_reason


def test_end_detects_power_source_change(monkeypatch: pytest.MonkeyPatch) -> None:
    switched = dataclasses.replace(CLEAN, on_ac_power=False)
    monkeypatch.setattr(
        BenchmarkEnvironment, "capture", classmethod(lambda cls, *a, **k: switched)
    )

    guard = BenchmarkGuard()
    pair = guard.end(CLEAN)

    assert not pair.valid
    assert "power source changed" in pair.invalid_reason


def test_end_detects_battery_drift_beyond_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    drained = dataclasses.replace(CLEAN, battery_percent=CLEAN.battery_percent - 5.0)
    monkeypatch.setattr(
        BenchmarkEnvironment, "capture", classmethod(lambda cls, *a, **k: drained)
    )

    guard = BenchmarkGuard(GuardThresholds(max_battery_drift_pct=2.0))
    pair = guard.end(CLEAN)

    assert not pair.valid
    assert "battery percentage moved" in pair.invalid_reason


# ---------------------------------------------------------------------------
# write_benchmark_artifact: a partial/invalid artifact is impossible to write
# ---------------------------------------------------------------------------


def _pair(valid: bool, reason: str = "") -> EnvironmentPair:
    return EnvironmentPair(
        start=CLEAN, end=CLEAN, valid=valid, invalid_reason=reason, sha="deadbeef"
    )


def test_write_benchmark_artifact_refuses_when_invalid(tmp_path: Path) -> None:
    pair = _pair(valid=False, reason="machine throttled during the run")
    target = tmp_path / "cascade_bench.json"

    with pytest.raises(BenchmarkInvalidated, match="throttled"):
        write_benchmark_artifact(target, {"worst_cost_share": 0.045}, pair)

    assert not target.exists()


def test_write_benchmark_artifact_succeeds_when_valid(tmp_path: Path) -> None:
    pair = _pair(valid=True)
    target = tmp_path / "cascade_bench.json"

    write_benchmark_artifact(target, {"worst_cost_share": 0.045}, pair)

    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["worst_cost_share"] == 0.045
    assert payload["environment"]["valid"] is True
    assert payload["environment"]["start"]["hostname"] == "bench-host"


# ---------------------------------------------------------------------------
# require_comparable
# ---------------------------------------------------------------------------


def test_require_comparable_raises_on_invalid_self() -> None:
    invalid = _pair(valid=False, reason="throttled")
    other = _pair(valid=True)
    with pytest.raises(EnvironmentMismatch, match="invalid"):
        invalid.require_comparable(other)


def test_require_comparable_raises_on_invalid_other() -> None:
    valid = _pair(valid=True)
    other = _pair(valid=False, reason="throttled")
    with pytest.raises(EnvironmentMismatch, match="invalid"):
        valid.require_comparable(other)


def test_require_comparable_raises_on_different_machine() -> None:
    other_host = dataclasses.replace(CLEAN, hostname="other-host", core_count_logical=4)
    pair_a = EnvironmentPair(
        start=CLEAN, end=CLEAN, valid=True, invalid_reason="", sha="host-a-sha"
    )
    pair_b = EnvironmentPair(
        start=other_host, end=other_host, valid=True, invalid_reason="", sha="host-b-sha"
    )
    with pytest.raises(EnvironmentMismatch, match="different machine"):
        pair_a.require_comparable(pair_b)


def test_require_comparable_passes_for_same_machine_valid_pairs() -> None:
    pair_a = EnvironmentPair(
        start=CLEAN, end=CLEAN, valid=True, invalid_reason="", sha="same-sha"
    )
    pair_b = EnvironmentPair(
        start=CLEAN, end=CLEAN, valid=True, invalid_reason="", sha="same-sha"
    )
    pair_a.require_comparable(pair_b)  # must not raise


# ---------------------------------------------------------------------------
# Live capture smoke test: the real thing runs on this machine without error
# ---------------------------------------------------------------------------


def test_capture_runs_on_this_machine() -> None:
    """No mocking -- confirms the real pmset/psutil/getloadavg path doesn't
    crash, without asserting on values that vary by machine and moment."""
    env = BenchmarkEnvironment.capture(sample_seconds=0.05)
    assert env.hostname
    assert env.core_count_logical > 0
    assert isinstance(env.violations(GuardThresholds()), list)
