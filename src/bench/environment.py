"""Benchmark environment gate: refuse to produce a measurement the machine
cannot stand behind.

Day 18's cascade-bench figure went 4.50% (Day 17, trustworthy) -> 5.10%
(concurrent CPU load from an unrelated test run) -> 9.28% (a battery at 5%,
throttled to ``CPU_Speed_Limit = 46`` by macOS) in the space of one session,
with nothing about the *code* changing between readings. Both bad numbers
printed to the console exactly like the good one. Nothing in the harness
distinguished them, so both got quoted in a draft before the machine state
was diagnosed after the fact.

The fix here is not "measure the machine state and report it alongside the
number" -- Day 18 effectively did that manually, by running ``pmset``
afterwards. It is "measure the machine state *before* the benchmark runs a
single frame, and refuse to produce a result at all if that state cannot be
stood behind." A benchmark that prints a number under bad conditions is worse
than one that refuses, because the number gets quoted and the refusal does
not.

Two checks, not one, because a machine can pass at the start and still ruin
the run:

- :func:`BenchmarkGuard.begin` -- captured and evaluated before any scenario
  runs. Fails closed: a condition this module cannot determine (no
  ``pmset``/``cpufreq`` on this platform, ``os.getloadavg`` unavailable) is
  treated as a refusal, not a pass, because an unmeasured risk is not an
  absent one.
- :func:`BenchmarkGuard.end` -- captured after the run and compared against
  the start. Mid-run throttling -- exactly what produced the 9.28% reading,
  which was measured *after* the concurrent load had already cleared -- is
  caught here even though the start-of-run state looked clean.

A benchmark artifact is only written through :func:`write_benchmark_artifact`,
which raises rather than writes when the run it describes was invalidated.
There is no code path that produces a partial or best-effort artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_OWN_PID = os.getpid()
_DEFAULT_SAMPLE_SECONDS = 0.3


class BenchmarkRefused(RuntimeError):
    """Raised by :meth:`BenchmarkGuard.begin` when the machine is not safe to
    benchmark on. Naming the exact condition and the remedy is the point --
    see :meth:`BenchmarkEnvironment.violations`."""


class BenchmarkInvalidated(RuntimeError):
    """Raised when a run's environment drifted past tolerance, or when code
    tries to write an artifact for an invalidated run."""


class EnvironmentMismatch(RuntimeError):
    """Raised by :meth:`EnvironmentPair.require_comparable` when two
    benchmark artifacts were not captured under comparable conditions."""


@dataclass(frozen=True)
class GuardThresholds:
    """Tunable limits for :class:`BenchmarkGuard`.

    Defaults are set from what a "quiet, AC-powered, unthrottled machine"
    means, not fitted to make this machine pass. ``max_high_cpu_processes``
    at 0 means *any* competing process above ``high_cpu_process_pct`` refuses
    the run -- Day 18's contaminating case was two ``pytest`` invocations,
    not dozens, so a nonzero tolerance would have let that exact failure
    through.
    """

    min_throttle_pct: float = 100.0
    max_load_avg_1min_per_core: float = 0.5
    max_battery_drift_pct: float = 2.0
    high_cpu_process_pct: float = 20.0
    max_high_cpu_processes: int = 0


def _power_state(system: str) -> tuple[bool | None, str, float | None]:
    """Return ``(on_ac_power, detail, battery_percent)``.

    ``on_ac_power=None`` means undeterminable, which callers must treat as a
    refusal -- see the module docstring on failing closed.
    """
    if system == "Darwin":
        try:
            completed = subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"pmset -g batt failed: {exc}", None
        output = completed.stdout
        if "AC Power" in output:
            on_ac: bool | None = True
        elif "Battery Power" in output:
            on_ac = False
        else:
            return (
                None,
                f"pmset -g batt did not report a recognizable power source: "
                f"{output.strip()[:200]!r}",
                None,
            )
        match = re.search(r"(\d+)%", output)
        battery_pct = float(match.group(1)) if match else None
        first_line = next(
            (line.strip() for line in output.splitlines() if line.strip()), ""
        )
        return on_ac, first_line, battery_pct

    if system == "Linux":
        supply_root = Path("/sys/class/power_supply")
        if not supply_root.exists():
            return True, "no /sys/class/power_supply -- assuming AC-only hardware", None
        mains_online: bool | None = None
        battery_pct = None
        try:
            entries = list(supply_root.iterdir())
        except OSError as exc:
            return None, f"cannot read {supply_root}: {exc}", None
        for entry in entries:
            type_file = entry / "type"
            if not type_file.exists():
                continue
            try:
                kind = type_file.read_text().strip()
            except OSError:
                continue
            if kind == "Mains":
                online_file = entry / "online"
                if online_file.exists():
                    try:
                        mains_online = online_file.read_text().strip() == "1"
                    except OSError:
                        pass
            elif kind == "Battery":
                cap_file = entry / "capacity"
                if cap_file.exists():
                    try:
                        battery_pct = float(cap_file.read_text().strip())
                    except (OSError, ValueError):
                        pass
        if mains_online is None:
            if battery_pct is None:
                return True, "no Mains or Battery supply reported -- assuming a desktop", None
            return (
                None,
                "a battery is present but no Mains supply entry was found under "
                f"{supply_root}",
                battery_pct,
            )
        return mains_online, f"{supply_root} Mains online={mains_online}", battery_pct

    return None, f"power source not determinable on platform {system!r}", None


def _throttle_state(system: str) -> tuple[float | None, str]:
    """Return ``(throttle_pct, detail)``. ``100.0`` means full speed.

    ``None`` means undeterminable, which callers must treat as a refusal.
    """
    if system == "Darwin":
        try:
            completed = subprocess.run(
                ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"pmset -g therm failed: {exc}"
        match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", completed.stdout)
        if not match:
            return (
                None,
                "pmset -g therm did not report CPU_Speed_Limit: "
                f"{completed.stdout.strip()[:200]!r}",
            )
        return float(match.group(1)), f"CPU_Speed_Limit={match.group(1)} (pmset -g therm)"

    if system == "Linux":
        cur = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        max_f = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if cur.exists() and max_f.exists():
            try:
                cur_hz = float(cur.read_text().strip())
                max_hz = float(max_f.read_text().strip())
            except (OSError, ValueError):
                return None, f"could not read {cur} / {max_f}"
            if max_hz > 0:
                pct = min(100.0, 100.0 * cur_hz / max_hz)
                return pct, f"scaling_cur_freq/cpuinfo_max_freq = {cur_hz:.0f}/{max_hz:.0f}"
        return (
            None,
            f"no cpufreq scaling info under {cur.parent} -- cannot confirm unthrottled",
        )

    return None, f"throttle state not determinable on platform {system!r}"


def _high_cpu_processes(
    threshold_pct: float, sample_seconds: float
) -> tuple[int, tuple[str, ...]]:
    """Sample every other process's CPU% over ``sample_seconds`` and return
    the count and names of those at or above ``threshold_pct``.

    Excludes this process itself. ``psutil.Process.cpu_percent`` requires a
    priming call before the interval it measures, hence the two passes.
    """
    import psutil

    primed = []
    for proc in psutil.process_iter(["pid"]):
        if proc.info["pid"] == _OWN_PID:
            continue
        try:
            proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        primed.append(proc)

    time.sleep(sample_seconds)

    hits: list[str] = []
    for proc in primed:
        try:
            pct = proc.cpu_percent(None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if pct >= threshold_pct:
            try:
                hits.append(proc.name())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                hits.append(f"pid {proc.pid}")
    return len(hits), tuple(sorted(hits))


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """One point-in-time snapshot of everything that can invalidate a
    wall-clock or CPU-percentage benchmark."""

    captured_at_utc: str
    hostname: str
    platform: str
    on_ac_power: bool | None
    ac_power_detail: str
    battery_percent: float | None
    throttle_pct: float | None
    throttle_detail: str
    load_avg_1: float | None
    load_avg_5: float | None
    load_avg_15: float | None
    core_count_logical: int
    core_count_physical: int | None
    high_cpu_process_count: int
    high_cpu_process_names: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        thresholds: GuardThresholds | None = None,
        sample_seconds: float = _DEFAULT_SAMPLE_SECONDS,
    ) -> "BenchmarkEnvironment":
        thresholds = thresholds or GuardThresholds()
        system = platform.system()
        on_ac, ac_detail, battery_pct = _power_state(system)
        throttle_pct, throttle_detail = _throttle_state(system)
        load1: float | None
        load5: float | None
        load15: float | None
        try:
            load1, load5, load15 = os.getloadavg()
        except OSError:
            load1 = load5 = load15 = None

        import psutil

        core_logical = psutil.cpu_count(logical=True) or 0
        core_physical = psutil.cpu_count(logical=False)

        high_count, high_names = _high_cpu_processes(
            thresholds.high_cpu_process_pct, sample_seconds
        )

        return cls(
            captured_at_utc=datetime.now(timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            platform=system,
            on_ac_power=on_ac,
            ac_power_detail=ac_detail,
            battery_percent=battery_pct,
            throttle_pct=throttle_pct,
            throttle_detail=throttle_detail,
            load_avg_1=load1,
            load_avg_5=load5,
            load_avg_15=load15,
            core_count_logical=core_logical,
            core_count_physical=core_physical,
            high_cpu_process_count=high_count,
            high_cpu_process_names=high_names,
        )

    def violations(self, thresholds: GuardThresholds) -> list[str]:
        """Every reason this snapshot fails the gate. Empty means safe.

        Each reason names the condition and what to do about it, matching
        ``scripts/env_gate.py``'s remedy convention -- a refusal that does
        not say what to do about it just gets worked around.
        """
        reasons: list[str] = []

        if self.on_ac_power is not True:
            reasons.append(
                f"not confirmed on AC power ({self.ac_power_detail}) -- plug in "
                "and wait for the power source to register as AC before retrying"
            )

        if self.throttle_pct is None:
            reasons.append(
                f"CPU throttle state could not be determined ({self.throttle_detail}) "
                "-- refusing rather than assuming unthrottled"
            )
        elif self.throttle_pct < thresholds.min_throttle_pct:
            reasons.append(
                f"CPU throttled to {self.throttle_pct:.0f}% of full speed "
                f"(below the {thresholds.min_throttle_pct:.0f}% floor) -- "
                f"{self.throttle_detail}; let the machine cool, ensure it is "
                "charging (not just plugged in at low battery), and retry"
            )

        if self.load_avg_1 is None:
            reasons.append(
                "1-minute load average unavailable on this platform -- refusing "
                "rather than assuming an idle machine"
            )
        else:
            per_core = self.load_avg_1 / max(self.core_count_logical, 1)
            if per_core > thresholds.max_load_avg_1min_per_core:
                reasons.append(
                    f"1-min load average {self.load_avg_1:.2f} across "
                    f"{self.core_count_logical} logical cores ({per_core:.2f}/core) "
                    f"exceeds the {thresholds.max_load_avg_1min_per_core:.2f}/core "
                    "limit -- close other work and retry"
                )

        if self.high_cpu_process_count > thresholds.max_high_cpu_processes:
            names = ", ".join(self.high_cpu_process_names) or "unnamed"
            reasons.append(
                f"{self.high_cpu_process_count} competing process(es) at or above "
                f"{thresholds.high_cpu_process_pct:.0f}% CPU ({names}) -- this is "
                "exactly the condition that produced Day 18's 5.10% cascade-bench "
                "reading; close them and retry"
            )

        return reasons

    def as_dict(self) -> dict[str, Any]:
        return {
            "captured_at_utc": self.captured_at_utc,
            "hostname": self.hostname,
            "platform": self.platform,
            "on_ac_power": self.on_ac_power,
            "ac_power_detail": self.ac_power_detail,
            "battery_percent": self.battery_percent,
            "throttle_pct": self.throttle_pct,
            "throttle_detail": self.throttle_detail,
            "load_avg_1": self.load_avg_1,
            "load_avg_5": self.load_avg_5,
            "load_avg_15": self.load_avg_15,
            "core_count_logical": self.core_count_logical,
            "core_count_physical": self.core_count_physical,
            "high_cpu_process_count": self.high_cpu_process_count,
            "high_cpu_process_names": list(self.high_cpu_process_names),
        }


def _identity_sha(env: BenchmarkEnvironment) -> str:
    """Hash of the fields that identify *which machine*, not *what moment*.

    Same exclusion principle as ``RunManifest.compute_sha``: two environment
    snapshots on the same machine, both valid, should compare equal here even
    though their timestamps and load averages differ.
    """
    payload = {
        "hostname": env.hostname,
        "platform": env.platform,
        "core_count_logical": env.core_count_logical,
        "core_count_physical": env.core_count_physical,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EnvironmentPair:
    """Start- and end-of-run environment snapshots, and whether the run
    between them is trustworthy."""

    start: BenchmarkEnvironment
    end: BenchmarkEnvironment
    valid: bool
    invalid_reason: str
    sha: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.as_dict(),
            "end": self.end.as_dict(),
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "sha": self.sha,
        }

    def require_comparable(self, other: "EnvironmentPair") -> None:
        """Raise unless two benchmark artifacts were captured under
        comparable conditions -- same rule as
        :meth:`~src.cascade.envelope.MeasuredEnvelope.require_comparable` and
        :meth:`~src.data.scorecard.Scorecard.require_comparable`.

        Both pairs must be valid (a run that drifted mid-flight is not a
        baseline anything can be compared against) and captured on the same
        machine identity. Throttle state and load are not compared directly
        here because a valid pair is, by construction, one where they were
        already within the gate's tolerance.
        """
        if not self.valid:
            raise EnvironmentMismatch(
                f"this environment pair is invalid ({self.invalid_reason}) and "
                "cannot be used as a comparison baseline"
            )
        if not other.valid:
            raise EnvironmentMismatch(
                f"the other environment pair is invalid ({other.invalid_reason}) "
                "and cannot be compared against"
            )
        if self.sha != other.sha:
            raise EnvironmentMismatch(
                "these benchmark artifacts were captured under different "
                f"machine identities ({self.start.hostname}/"
                f"{self.start.core_count_logical} cores vs "
                f"{other.start.hostname}/{other.start.core_count_logical} cores) "
                "-- not comparable. Re-measure both on the same machine, or wait "
                "for reference hardware (docs/reference_hardware.md)."
            )


def _drift_reasons(
    start: BenchmarkEnvironment, end: BenchmarkEnvironment, thresholds: GuardThresholds
) -> list[str]:
    """Reasons the machine's state changed enough during the run to
    invalidate it -- distinct from :meth:`BenchmarkEnvironment.violations`,
    which checks a single snapshot against an absolute floor.

    This is what Day 18 needed and did not have: the run that produced 9.28%
    started on a machine already past its throttle floor, so a single
    end-of-run check would have caught it anyway -- but a run that starts
    clean and throttles midway (the more dangerous case, because the
    *start*-of-run gate would have said yes) is only caught by comparing the
    two snapshots.
    """
    reasons: list[str] = []

    if start.throttle_pct is not None and end.throttle_pct is not None:
        if end.throttle_pct < start.throttle_pct:
            reasons.append(
                f"CPU throttled during the run (start {start.throttle_pct:.0f}%, "
                f"end {end.throttle_pct:.0f}%) -- this is exactly the mid-run "
                "drift that produced Day 18's 9.28% reading"
            )
    elif start.throttle_pct is not None or end.throttle_pct is not None:
        reasons.append(
            "throttle state became undeterminable partway through the run "
            f"(start detail: {start.throttle_detail!r}, end detail: "
            f"{end.throttle_detail!r})"
        )

    if start.on_ac_power != end.on_ac_power:
        reasons.append(
            f"power source changed during the run (start on_ac_power="
            f"{start.on_ac_power}, end on_ac_power={end.on_ac_power})"
        )

    if start.battery_percent is not None and end.battery_percent is not None:
        drift = abs(start.battery_percent - end.battery_percent)
        if drift > thresholds.max_battery_drift_pct:
            reasons.append(
                f"battery percentage moved {drift:.1f} points during the run "
                f"(start {start.battery_percent:.0f}%, end "
                f"{end.battery_percent:.0f}%), exceeding the "
                f"{thresholds.max_battery_drift_pct:.1f}-point tolerance"
            )

    return reasons


class BenchmarkGuard:
    """Wraps one benchmark run: refuse to start on a bad machine, and refuse
    to certify a run the machine's state changed underneath.

    Usage::

        guard = BenchmarkGuard()
        start = guard.begin()          # raises BenchmarkRefused, exits before
                                        # any scenario runs
        ... run the benchmark ...
        pair = guard.end(start)        # raises nothing; pair.valid says
                                        # whether the run may be trusted
        write_benchmark_artifact(path, payload, pair)   # raises if not valid
    """

    def __init__(self, thresholds: GuardThresholds | None = None) -> None:
        self.thresholds = thresholds or GuardThresholds()

    def begin(self) -> BenchmarkEnvironment:
        env = BenchmarkEnvironment.capture(self.thresholds)
        reasons = env.violations(self.thresholds)
        if reasons:
            raise BenchmarkRefused(
                "benchmark refused: this machine's state cannot be stood "
                "behind:\n" + "\n".join(f"  - {r}" for r in reasons)
            )
        return env

    def end(self, start: BenchmarkEnvironment) -> EnvironmentPair:
        end_env = BenchmarkEnvironment.capture(self.thresholds)
        end_violations = end_env.violations(self.thresholds)
        drift = _drift_reasons(start, end_env, self.thresholds)
        all_reasons = end_violations + drift
        return EnvironmentPair(
            start=start,
            end=end_env,
            valid=not all_reasons,
            invalid_reason="; ".join(all_reasons),
            sha=_identity_sha(start),
        )


def write_benchmark_artifact(
    path: Path, payload: dict[str, Any], pair: EnvironmentPair
) -> Path:
    """Write ``payload`` merged with ``pair``'s environment block to
    ``path``, refusing when the run it describes is invalid.

    This is the only path in this module that writes to disk, and it is
    structurally impossible to reach with an invalid pair: every caller must
    hold an :class:`EnvironmentPair` (only produced by
    :meth:`BenchmarkGuard.end`), and this function checks ``pair.valid``
    before touching the filesystem.

    Raises:
        BenchmarkInvalidated: if ``pair.valid`` is False.
    """
    if not pair.valid:
        raise BenchmarkInvalidated(
            f"refusing to write {path}: this run's environment was invalid or "
            f"drifted ({pair.invalid_reason}). Discard this run -- it is not a "
            "measurement of anything except today's machine state -- and "
            "re-measure on a quiet, AC-powered, unthrottled machine."
        )
    full = dict(payload)
    full["environment"] = pair.as_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n")
    return path
