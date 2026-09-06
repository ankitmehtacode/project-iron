"""Day 32, Objective 2 -- suite-count provenance, structural.

The same shape as the Day-25 Verdicts lint (`tests/test_report_verdicts.py`)
and the Day-27 citation lint (`tests/test_data_model_citations.py`): a defect
found once gets a mechanism, not a promise to remember. Day 31's defect was
a suite result -- "completed, exit code 0" -- that disagreed with the actual
output, caught only because someone happened to read the real log instead of
trusting the notification.

From Day 32 onward, a line in `FOUNDATION_REPORT.md` shaped like a pytest
summary ("N passed[, M skipped][, K deselected]...") must cite a
`scripts/suite_report.py` artifact under `artifacts/pytest/` that actually
exists on disk, within the same paragraph. A claimed count with no such
citation fails here -- "traceable to a file on disk," per the objective's
own wording -- rather than being discovered by someone re-running the suite
and getting a different number.

Days 1-31 predate this convention and are NOT retrofitted, the same call
the Verdicts lint (Days 1-19) and the citation lint made for their own
introduction days: reconstructing an artifact for a run that already
finished and was never captured would not be a measurement, it would be a
new run standing in for one that no longer exists to verify. `docs/
dismissal_audit.md` records that none of those historical numbers are
artifact-backed, and marks them unverified rather than leaving that
unstated -- this lint enforces the boundary going forward, not backward.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "FOUNDATION_REPORT.md"

_DAY_HEADING = re.compile(r"(?m)^# (.+)$")
_DAY_NUMBER = re.compile(r"Day (\d+)")
_SUITE_RESULT = re.compile(r"\b\d+\s+passed\b")
_ARTIFACT_CITATION = re.compile(r"artifacts/pytest/[\w./-]+\.(?:xml|log|json)")

MINIMUM_ENFORCED_DAY = 32
"""This convention's own introduction day. Days before it are not
retrofitted -- see the module docstring."""


def _day_sections() -> list[tuple[int, str, str]]:
    text = REPORT_PATH.read_text(encoding="utf-8")
    headings = list(_DAY_HEADING.finditer(text))
    sections = []
    for i, match in enumerate(headings):
        heading_text = match.group(1)
        day_match = _DAY_NUMBER.search(heading_text)
        if day_match is None:
            continue
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections.append((int(day_match.group(1)), heading_text, text[start:end]))
    return sections


def _paragraphs(body: str) -> list[str]:
    return re.split(r"\n\s*\n", body)


def _uncited_suite_claims(body: str) -> list[str]:
    """Suite-result-shaped lines in ``body`` with no artifact citation to a
    file that actually exists, in the same paragraph."""
    uncited = []
    for paragraph in _paragraphs(body):
        if not _SUITE_RESULT.search(paragraph):
            continue
        citations = _ARTIFACT_CITATION.findall(paragraph)
        backed = any((REPO_ROOT / c).exists() for c in citations)
        if not backed:
            claim = _SUITE_RESULT.search(paragraph)
            snippet = paragraph[max(0, claim.start() - 20) : claim.end() + 40]
            uncited.append(snippet.strip().replace("\n", " "))
    return uncited


def test_foundation_report_exists() -> None:
    assert REPORT_PATH.exists(), f"{REPORT_PATH} not found"


def test_every_suite_claim_from_day_32_onward_cites_an_existing_artifact() -> None:
    sections = _day_sections()
    enforced = [s for s in sections if s[0] >= MINIMUM_ENFORCED_DAY]
    assert enforced, (
        f"no day sections with day number >= {MINIMUM_ENFORCED_DAY} were "
        "found -- the day-heading regex or numbering may have changed; this "
        "test cannot verify anything until that is fixed"
    )

    failures: dict[str, list[str]] = {}
    for day_number, heading_text, body in enforced:
        uncited = _uncited_suite_claims(body)
        if uncited:
            failures[heading_text] = uncited

    assert not failures, (
        "suite-result claim(s) with no existing artifacts/pytest/*.xml (or "
        f".log/.json) citation in the same paragraph: {failures} -- run "
        "scripts/suite_report.py and cite its output, per Day 32 Objective 2"
    )


def test_the_lint_actually_catches_an_uncited_claim() -> None:
    """Falsifiability check, matching this project's convention (Day-24
    registry-gate tests, the Verdicts and citation lints' own fixtures):
    construct a claim shaped exactly like a real one but with no backing
    artifact, and confirm the detection logic flags it."""
    fake_body = (
        "\n\nRan the gated suite: 1259 passed, 7 skipped, 0 failures. No "
        "artifact cited here at all.\n\n"
    )
    uncited = _uncited_suite_claims(fake_body)
    assert uncited, "a suite claim with no artifact citation should be flagged"


def test_the_lint_accepts_a_claim_backed_by_a_real_artifact() -> None:
    """The complementary check: a claim citing a file that genuinely
    exists must NOT be flagged, or every real Day-32+ entry would fail
    alongside the ones this lint exists to catch."""
    real_artifact = "artifacts/pytest/day32_gated_summary.json"
    assert (REPO_ROOT / real_artifact).exists(), (
        f"{real_artifact} must exist for this test to mean anything -- "
        "run scripts/suite_report.py --label-prefix day32"
    )
    fake_body = f"\n\nRan the gated suite: 1259 passed, 7 skipped, 0 failures ({real_artifact}).\n\n"
    uncited = _uncited_suite_claims(fake_body)
    assert not uncited, f"a claim citing a real artifact was still flagged: {uncited}"
