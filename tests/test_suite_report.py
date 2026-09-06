"""Day 32, Objective 2 -- the harness-truthfulness fix, unit-tested.

`scripts/suite_report.py` is what replaces "the background run reported
completed, exit code 0" (Day 31) with a parsed count backed by a file on
disk. The reconciliation check -- gated executed + deselected == full
executed -- is the part that would have caught a test silently dropping out
of both runs, so it is proven here to actually fail on a constructed
mismatch before being trusted (this project's standing convention: see the
Day-24 registry-gate tests and the Verdicts/citation lints' own
falsifiability checks).

An end-to-end run of `main()` against the real `tests/` tree is deliberately
NOT exercised here -- it would recursively run this repo's own suite inside
itself, and the full (no-exclusions) invocation alone runs over two hours
(Day 31, FOUNDATION_REPORT.md). `docs/dismissal_audit.md`'s own re-test of
`test_synthetic_indoor.py` and `tests/test_cascade_bench_artifact_gate.py`'s
existing coverage exercise the pytest-subprocess path already; what is
untested elsewhere, and what matters for Objective 2's structural claim, is
the arithmetic in `reconcile()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import suite_report as sr  # noqa: E402


def _run(
    label: str,
    tests: int,
    failures: int = 0,
    errors: int = 0,
    skipped: int = 0,
    deselected: int = 0,
) -> sr.SuiteRun:
    return sr.SuiteRun(
        label=label,
        marker_expr=None,
        junit_path=REPO_ROOT / "artifacts" / "pytest" / f"{label}.xml",
        log_path=REPO_ROOT / "artifacts" / "pytest" / f"{label}.log",
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        deselected=deselected,
        returncode=0,
    )


def test_reconcile_accepts_matching_counts() -> None:
    gated = _run("gated", tests=1001, skipped=1, deselected=21)
    full = _run("full", tests=1022)
    ok, message = sr.reconcile(gated, full)
    assert ok
    assert "reconciled" in message


def test_reconcile_rejects_a_dropped_test() -> None:
    """The falsifiability check: construct the exact shape of the Day-31
    defect -- a test present in the full run but missing from both the
    gated run's executed count and its deselected count -- and confirm
    reconcile() actually catches it rather than passing by construction."""
    gated = _run("gated", tests=1001, skipped=1, deselected=21)
    full = _run("full", tests=1023)  # one more than gated+deselected accounts for
    ok, message = sr.reconcile(gated, full)
    assert not ok
    assert "DISCREPANCY" in message
    assert "1 test(s) unaccounted for" in message


def test_reconcile_rejects_an_over_counted_gated_run() -> None:
    """The other direction: gated + deselected exceeding full is just as
    much a discrepancy as falling short of it -- e.g. a test counted twice
    by a marker expression matching it in both the executed and deselected
    tally would otherwise pass silently."""
    gated = _run("gated", tests=1002, skipped=1, deselected=21)
    full = _run("full", tests=1022)
    ok, message = sr.reconcile(gated, full)
    assert not ok
    assert "DISCREPANCY" in message
    assert "extra 1 test(s)" in message


def test_render_line_never_reports_a_bare_exit_code() -> None:
    """Objective 2's hard requirement: the rendered line is always built
    from parsed counts, never from `returncode`."""
    run = _run("gated", tests=10, failures=1, skipped=2, deselected=3)
    line = sr.render_line(run)
    assert "7 passed" in line
    assert "2 skipped" in line
    assert "3 deselected" in line
    assert "1 FAILED" in line
    assert "returncode" not in line.lower()
    assert "exit code" not in line.lower()
