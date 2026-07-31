"""Endurance and memory-stability harness.

Split out of the former monolithic ``endurance_run.py`` so the parts that make
decisions — the leak regression and the latency summary in
:mod:`src.endurance.gates` — are pure functions of sampled numbers, and can be
tested against synthetic series without model weights. Previously the gate
logic was inline in a 200-clip loop, so the only way to exercise it was to run
the thing it was meant to check.
"""

from src.endurance.gates import (
    LatencySummary,
    LeakVerdict,
    Verdict,
    evaluate_leak,
    summarize_latency,
)
from src.endurance.memory import MemorySample, MemorySampler
from src.endurance.runner import (
    DeterminismResult,
    EnduranceRunner,
    ExitCode,
    RunOutcome,
    WeightsUnavailable,
    check_determinism,
    execute,
)

__all__ = [
    "DeterminismResult",
    "EnduranceRunner",
    "ExitCode",
    "LatencySummary",
    "LeakVerdict",
    "MemorySample",
    "MemorySampler",
    "RunOutcome",
    "Verdict",
    "WeightsUnavailable",
    "check_determinism",
    "evaluate_leak",
    "execute",
    "summarize_latency",
]
