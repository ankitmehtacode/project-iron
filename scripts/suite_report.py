"""Suite counts a report can stand behind: parsed artifacts, not notifications.

Day 31 recorded a failing background run as "completed, exit code 0" because
the harness's own completion notification was trusted instead of the output
it was summarizing -- a summary of a process disagreeing with the process
(the same species as the Day-25 closed-form-vs-running-filter divergence).
This script is the fix for the suite-count half of that: it never reports an
exit code as a substitute for a count, and it writes the count to a file
before printing it, so a number that ends up in FOUNDATION_REPORT.md can be
traced back to something on disk (`tests/test_report_suite_provenance.py`
enforces that traceability structurally).

It runs pytest twice -- the CI-gated suite (`-m "not requires_weights"`, the
Makefile's own `test` target) and the full suite (no marker exclusions) --
each with `--junitxml` for pass/fail/skip/error counts (a stdlib-adjacent
artifact; no new dependency) and stdout captured to a `.log` for the
`N deselected` figure junitxml does not carry.

Then it reconciles: every test the gated run deselected plus every test it
ran must equal the full run's total. A mismatch there is exactly what would
hide a test that silently stopped being collected by either run -- the
discrepancy Objective 2 exists to surface automatically rather than by
someone reading two numbers side by side and noticing they don't add up.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "pytest"

_DESELECTED_RE = re.compile(r"(\d+) deselected")


@dataclass
class SuiteRun:
    label: str
    marker_expr: str | None
    junit_path: Path
    log_path: Path
    tests: int
    failures: int
    errors: int
    skipped: int
    deselected: int
    returncode: int

    @property
    def executed(self) -> int:
        """Every testcase junitxml recorded -- passed, failed, errored or
        skipped. Deliberately excludes deselected: those were never run."""
        return self.tests

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "marker_expr": self.marker_expr,
            "junit_path": str(self.junit_path.relative_to(REPO_ROOT)),
            "log_path": str(self.log_path.relative_to(REPO_ROOT)),
            "tests": self.tests,
            "failures": self.failures,
            "errors": self.errors,
            "skipped": self.skipped,
            "deselected": self.deselected,
            "returncode": self.returncode,
        }


def _run_pytest(label: str, marker_expr: str | None) -> SuiteRun:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    junit_path = ARTIFACT_DIR / f"{label}.xml"
    log_path = ARTIFACT_DIR / f"{label}.log"

    cmd = [PYTHON, "-m", "pytest", f"--junitxml={junit_path}", "-q"]
    if marker_expr is not None:
        cmd += ["-m", marker_expr]

    completed = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")

    root = ET.parse(junit_path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        raise RuntimeError(f"{junit_path} has no <testsuite> element to parse")

    deselected_match = _DESELECTED_RE.search(completed.stdout)
    deselected = int(deselected_match.group(1)) if deselected_match else 0

    return SuiteRun(
        label=label,
        marker_expr=marker_expr,
        junit_path=junit_path,
        log_path=log_path,
        tests=int(suite.get("tests", 0)),
        failures=int(suite.get("failures", 0)),
        errors=int(suite.get("errors", 0)),
        skipped=int(suite.get("skipped", 0)),
        deselected=deselected,
        returncode=completed.returncode,
    )


def reconcile(gated: SuiteRun, full: SuiteRun) -> tuple[bool, str]:
    """gated.executed + gated.deselected must equal full.executed.

    This is the check that would have caught the Day-24/Day-31 shape of
    problem automatically: a test present in the full collection but absent
    from both the gated run's executed set AND its deselected count has
    silently stopped being collected by anything.
    """
    accounted = gated.executed + gated.deselected
    if accounted == full.executed:
        return True, (
            f"reconciled: {gated.label} executed={gated.executed} + "
            f"deselected={gated.deselected} = {accounted} == "
            f"{full.label} executed={full.executed}"
        )
    direction = "missing" if accounted < full.executed else "extra"
    return False, (
        f"DISCREPANCY: {gated.label} executed={gated.executed} + "
        f"deselected={gated.deselected} = {accounted}, but {full.label} "
        f"executed={full.executed} ({direction} "
        f"{abs(full.executed - accounted)} test(s) unaccounted for)"
    )


def render_line(run: SuiteRun) -> str:
    parts = [f"{run.tests - run.skipped - run.failures - run.errors} passed"]
    if run.skipped:
        parts.append(f"{run.skipped} skipped")
    if run.deselected:
        parts.append(f"{run.deselected} deselected")
    if run.failures:
        parts.append(f"{run.failures} FAILED")
    if run.errors:
        parts.append(f"{run.errors} errors")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-prefix",
        default="suite",
        help="artifact filename prefix, e.g. 'day32' -> "
        "artifacts/pytest/day32_gated.xml",
    )
    args = parser.parse_args(argv)

    gated = _run_pytest(f"{args.label_prefix}_gated", "not requires_weights")
    full = _run_pytest(f"{args.label_prefix}_full", None)

    ok, message = reconcile(gated, full)

    summary_path = ARTIFACT_DIR / f"{args.label_prefix}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "gated": gated.as_dict(),
                "full": full.as_dict(),
                "reconciled": ok,
                "reconciliation_message": message,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"gated ({gated.marker_expr}): {render_line(gated)}")
    print(f"  artifact: {gated.junit_path.relative_to(REPO_ROOT)}")
    print(f"full (no exclusions):        {render_line(full)}")
    print(f"  artifact: {full.junit_path.relative_to(REPO_ROOT)}")
    print()
    print(message)
    print(f"summary written to {summary_path.relative_to(REPO_ROOT)}")

    if not ok:
        return 1
    if gated.failures or gated.errors or full.failures or full.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
