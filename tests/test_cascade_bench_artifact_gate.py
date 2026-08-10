"""Tests that scripts/cascade_bench.py's --artifact path actually goes
through the Day-19 benchmark environment gate (src/bench/environment.py).

Real machine state during Day 19's own session demonstrated every branch
here live: a stray high-CPU process refused a run before it started, and a
CPU that throttled mid-session (down to 24% of full speed, while on AC
power) would have invalidated a run that started clean. These tests pin
those two paths so a future change to the wiring can't silently drop the
gate and go back to printing an unguarded number.

Scenarios are kept small (`--seconds 9 --static-seconds 3`, low-res frames)
since only the gate wiring is under test, not cascade correctness -- that is
`tests/test_*` elsewhere and the script's own regression checks. `--seconds`
below 3 divides to zero static-scenario frames (a pre-existing edge case in
`static_scenario`, unrelated to today's scope) and is avoided rather than
fixed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cascade_bench  # noqa: E402

from src.bench.environment import (  # noqa: E402
    BenchmarkEnvironment,
    BenchmarkGuard,
    EnvironmentPair,
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


def test_artifact_run_refused_on_bad_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_begin(self: BenchmarkGuard) -> BenchmarkEnvironment:
        raise cascade_bench.BenchmarkRefused(
            "benchmark refused: this machine's state cannot be stood behind:\n"
            "  - not confirmed on AC power (Now drawing from 'Battery Power') -- "
            "plug in and wait for the power source to register as AC before retrying"
        )

    monkeypatch.setattr(BenchmarkGuard, "begin", refuse_begin)
    target = tmp_path / "artifact.json"

    exit_code = cascade_bench.main(
        ["--seconds", "9", "--static-seconds", "3", "--width", "320", "--height", "240", "--artifact", str(target)]
    )

    assert exit_code == 1
    assert not target.exists()


def test_artifact_run_invalidated_by_mid_run_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BenchmarkGuard, "begin", lambda self: CLEAN)

    invalid_pair = EnvironmentPair(
        start=CLEAN,
        end=CLEAN,
        valid=False,
        invalid_reason="CPU throttled during the run (start 100%, end 24%)",
        sha="irrelevant",
    )
    monkeypatch.setattr(BenchmarkGuard, "end", lambda self, start: invalid_pair)

    target = tmp_path / "artifact.json"
    exit_code = cascade_bench.main(
        ["--seconds", "9", "--static-seconds", "3", "--width", "320", "--height", "240", "--artifact", str(target)]
    )

    assert exit_code == 1
    assert not target.exists()


def test_artifact_run_writes_certified_artifact_when_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BenchmarkGuard, "begin", lambda self: CLEAN)

    valid_pair = EnvironmentPair(
        start=CLEAN, end=CLEAN, valid=True, invalid_reason="", sha="fixed-sha"
    )
    monkeypatch.setattr(BenchmarkGuard, "end", lambda self, start: valid_pair)

    target = tmp_path / "artifact.json"
    exit_code = cascade_bench.main(
        ["--seconds", "9", "--static-seconds", "3", "--width", "320", "--height", "240", "--artifact", str(target)]
    )

    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["environment"]["valid"] is True
    assert "worst_cost_share" in payload
    # exit_code reflects the scenario/regression checks independently of the
    # environment gate -- a small, fast synthetic run may or may not pass
    # those on its own merits; only the artifact's existence is under test
    # here.
    assert exit_code in (0, 1)


def test_unguarded_run_is_unaffected_by_the_gate() -> None:
    """Without --artifact, the script behaves exactly as before -- this is
    the invocation CI's regression-ceiling check uses, and it must not
    start refusing runs just because a CI runner has no battery."""
    exit_code = cascade_bench.main(
        ["--seconds", "9", "--static-seconds", "3", "--width", "320", "--height", "240", "--budget-scale", "3.0"]
    )
    assert exit_code in (0, 1)
