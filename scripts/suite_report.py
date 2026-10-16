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

It also lists the checks that did not run (Day 42): every ``pytest.skip``
in the full run, grouped by reason, with each path the reason names searched
for elsewhere on this machine. A skip whose named prerequisite exists under a
different path is a DEFECT, not a skip: the check could have run, and is
pointed at the wrong place. (Six golden-vector and known-bug tests skipped on
every machine for 42 days because they looked for an INT8 IR that ADR 0008
declares lost, while a manifested FP32 export sat in models/export/.) Every
day's Verdicts block restates the table (``tests/test_report_did_not_run.py``).

Then it reconciles: every test the gated run deselected plus every test it
ran must equal the full run's total. A mismatch there is exactly what would
hide a test that silently stopped being collected by either run -- the
discrepancy Objective 2 exists to surface automatically rather than by
someone reading two numbers side by side and noticing they don't add up.
"""

from __future__ import annotations

import argparse
import json
import os
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
_PATH_RE = re.compile(r"(?:^|(?<=[\s'\"(]))(/[^\s;:,'\"()\[\]]+)")
"""An ABSOLUTE path: a slash at the start of the reason or after whitespace,
a quote or a parenthesis — not the middle of ``scripts/fetch_weights.py``."""
_TEMP_RE = re.compile(
    r"(/private)?/var/folders/[^\s;:,'\"()]+|/tmp/[^\s;:,'\"()]+"
    r"|pytest-of-[^/\s]+/pytest-\d+"
)
_SKIP_SEARCH_EXCLUDE = {
    ".git",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".hypothesis",
}
DEFAULT_SEARCH_ROOTS = (REPO_ROOT, Path.home() / ".cache")
"""Where a skipped prerequisite is looked for "elsewhere on this machine":
the repository (models/, data/, exports) and the user cache (HuggingFace and
torch hub land there). ``--search-root`` adds more."""


@dataclass(frozen=True)
class SkipGroup:
    """Every skipped test that gave one (normalised) reason."""

    reason: str
    tests: tuple[str, ...]
    named_paths: tuple[str, ...]
    found_elsewhere: tuple[str, ...]
    similar_elsewhere: tuple[str, ...] = ()
    """Same extension and same stem up to its last ``_`` part (e.g.
    ``vjepa2_vitl_int8.xml`` -> ``vjepa2_vitl_fp32.xml``): possibly the same
    prerequisite in another form. Reported for a human to judge; not a
    defect by itself."""

    @property
    def defect(self) -> bool:
        """True when a prerequisite the reason says is missing exists under
        another path on this machine with the same name: the check could have
        run."""
        return bool(self.found_elsewhere)

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "count": len(self.tests),
            "tests": list(self.tests),
            "named_paths": list(self.named_paths),
            "found_elsewhere": list(self.found_elsewhere),
            "similar_elsewhere": list(self.similar_elsewhere),
            "defect": self.defect,
        }


def normalise_reason(message: str, temp: re.Pattern[str] | None = _TEMP_RE) -> str:
    """Per-run temporary paths would split one reason into many groups."""
    return (temp.sub("<tmp>", message) if temp is not None else message).strip()


def named_paths(reason: str) -> tuple[str, ...]:
    """Absolute paths a skip reason names; ``<tmp>`` (normalised temporary
    directories) excluded."""
    found = []
    for match in _PATH_RE.findall(reason):
        path = match.rstrip(".")
        if "<tmp>" in path:
            continue
        found.append(path)
    return tuple(dict.fromkeys(found))


def _family(name: str) -> tuple[str, str] | None:
    """``("vjepa2_vitl", ".xml")`` for ``vjepa2_vitl_int8.xml``; None when the
    stem has no ``_`` part to drop."""
    stem, suffix = Path(name).stem, Path(name).suffix
    if "_" not in stem or not suffix:
        return None
    return (stem.rsplit("_", 1)[0], suffix)


def find_similar(path: str, roots: tuple[Path, ...]) -> tuple[str, ...]:
    """Files of the same family (see :class:`SkipGroup`) under ``roots``."""
    family = _family(Path(path).name)
    if family is None:
        return ()
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [
                d
                for d in subdirs
                if d not in _SKIP_SEARCH_EXCLUDE and not d.startswith(".venv")
            ]
            for name in files:
                if name != Path(path).name and _family(name) == family:
                    hits.append(str(Path(directory) / name))
    return tuple(sorted(set(hits)))


def find_elsewhere(path: str, roots: tuple[Path, ...]) -> tuple[str, ...]:
    """Files named like ``path`` that exist under ``roots`` at another path."""
    target = Path(path)
    if not target.name:
        return ()
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [
                d
                for d in subdirs
                if d not in _SKIP_SEARCH_EXCLUDE and not d.startswith(".venv")
            ]
            if target.name in files:
                hit = Path(directory) / target.name
                if hit != target:
                    hits.append(str(hit))
    return tuple(sorted(set(hits)))


def skips_by_reason(
    junit_path: Path,
    roots: tuple[Path, ...] = DEFAULT_SEARCH_ROOTS,
    temp: re.Pattern[str] | None = _TEMP_RE,
) -> list[SkipGroup]:
    """Group every ``pytest.skip`` (not xfail: an xfail ran) by reason."""
    root = ET.parse(junit_path).getroot()
    grouped: dict[str, list[str]] = {}
    for case in root.iter("testcase"):
        skipped = case.find("skipped")
        if skipped is None or skipped.get("type") == "pytest.xfail":
            continue
        reason = normalise_reason(
            skipped.get("message", "") or "(no reason given)", temp
        )
        test = f"{case.get('classname')}::{case.get('name')}"
        grouped.setdefault(reason, []).append(test)
    groups = []
    for reason, tests in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        paths = named_paths(reason)
        elsewhere = tuple(
            hit
            for path in paths
            if not Path(path).exists()
            for hit in find_elsewhere(path, roots)
        )
        similar = tuple(
            hit
            for path in paths
            if not Path(path).exists()
            for hit in find_similar(path, roots)
        )
        groups.append(
            SkipGroup(
                reason=reason,
                tests=tuple(sorted(tests)),
                named_paths=paths,
                found_elsewhere=elsewhere,
                similar_elsewhere=similar,
            )
        )
    return groups


def render_did_not_run(groups: list[SkipGroup]) -> str:
    """The Markdown table every day's Verdicts block restates."""
    lines = [
        "| skipped | reason | prerequisite found elsewhere on this machine |",
        "|---:|---|---|",
    ]
    if not groups:
        lines.append("| 0 | (every collected check ran) | — |")
    for g in groups:
        reason = g.reason.replace("|", "\\|")
        if len(reason) > 200:
            reason = reason[:197] + "..."
        if g.defect:
            where = "**DEFECT**: " + "; ".join(f"`{p}`" for p in g.found_elsewhere)
        elif g.similar_elsewhere:
            where = "POSSIBLE DEFECT, similar: " + "; ".join(
                f"`{p}`" for p in g.similar_elsewhere
            )
        else:
            where = "no"
        lines.append(f"| {len(g.tests)} | {reason} | {where} |")
    return "\n".join(lines)


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
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="extra directory to search for skipped prerequisites",
    )
    parser.add_argument(
        "--did-not-run-only",
        type=Path,
        metavar="JUNIT_XML",
        help="only print the checks-that-did-not-run table for an existing run",
    )
    args = parser.parse_args(argv)
    roots = DEFAULT_SEARCH_ROOTS + tuple(args.search_root)

    if args.did_not_run_only is not None:
        groups = skips_by_reason(args.did_not_run_only, roots)
        print(render_did_not_run(groups))
        return 1 if any(g.defect for g in groups) else 0

    gated = _run_pytest(f"{args.label_prefix}_gated", "not requires_weights")
    full = _run_pytest(f"{args.label_prefix}_full", None)

    ok, message = reconcile(gated, full)
    did_not_run = skips_by_reason(full.junit_path, roots)
    table_path = ARTIFACT_DIR / f"{args.label_prefix}_did_not_run.md"
    table_path.write_text(render_did_not_run(did_not_run) + "\n", encoding="utf-8")

    summary_path = ARTIFACT_DIR / f"{args.label_prefix}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "gated": gated.as_dict(),
                "full": full.as_dict(),
                "reconciled": ok,
                "reconciliation_message": message,
                "did_not_run": [g.as_dict() for g in did_not_run],
                "skip_defects": sum(g.defect for g in did_not_run),
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
    print()
    print("Checks that did not run (full run):")
    print(render_did_not_run(did_not_run))
    print(f"summary written to {summary_path.relative_to(REPO_ROOT)}")
    print(f"table written to {table_path.relative_to(REPO_ROOT)}")

    if not ok:
        return 1
    if any(g.defect for g in did_not_run):
        return 1
    if gated.failures or gated.errors or full.failures or full.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
