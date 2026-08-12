"""Day 25, Objective 4 — the report-to-summary promotion gap, structural.

Three numbers a day's own prompt required an explicit answer to (Day 20's
per-distance-bucket baseline margin, Day 23/24's cessation frame-support
verdict, Day 24's parallel-maintenance audit) were all present somewhere in
FOUNDATION_REPORT.md's prose, and still never made it into a session
summary. The report was honest; the promotion step out of it was lossy.

Every day section from Day 20 onward now opens with a ``## Verdicts``
block: one line per question that day's prompt asked for an explicit
answer to, each pointing at the objective containing the evidence — see
any Day 20-25 section for the shape. This test is the structural half of
the fix: a day section with no ``## Verdicts`` block, or an empty one,
fails here rather than being discovered by a reader going looking for a
number that was "in the report somewhere."

Days 1-19 predate this convention and are not retrofitted (Day 25's own
scope was Days 20-24; earlier days are numbered inconsistently in the
report's own headings -- e.g. "Day 3 (resumed, post-amendment)" -- and are
out of scope here).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "FOUNDATION_REPORT.md"

_DAY_HEADING = re.compile(r"(?m)^# (.+)$")
_DAY_NUMBER = re.compile(r"Day (\d+)")
_VERDICTS_HEADING = re.compile(r"(?m)^## Verdicts\s*$")
_NEXT_HEADING = re.compile(r"(?m)^#{1,2} ")

MINIMUM_RETROFITTED_DAY = 20
"""Day 25, Objective 4's own retrofit scope (Days 20-24) plus every day
from Day 25 onward, which opens with this block by construction."""


def _day_sections() -> list[tuple[int, str, str]]:
    """Every top-level ``# ...`` section whose heading names a day number,
    as (day_number, heading_text, section_body)."""
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


def test_foundation_report_exists() -> None:
    assert REPORT_PATH.exists(), f"{REPORT_PATH} not found"


def test_every_day_from_20_onward_has_a_nonempty_verdicts_block() -> None:
    sections = _day_sections()
    retrofitted = [s for s in sections if s[0] >= MINIMUM_RETROFITTED_DAY]
    assert retrofitted, (
        f"no day sections with day number >= {MINIMUM_RETROFITTED_DAY} were "
        "found at all -- the day-heading regex or day numbering may have "
        "changed; this test cannot verify anything until that is fixed"
    )

    missing: list[str] = []
    empty: list[str] = []
    for day_number, heading_text, body in retrofitted:
        verdicts_match = _VERDICTS_HEADING.search(body)
        if verdicts_match is None:
            missing.append(heading_text)
            continue
        content_start = verdicts_match.end()
        next_heading = _NEXT_HEADING.search(body, content_start)
        content_end = next_heading.start() if next_heading else len(body)
        block = body[content_start:content_end].strip()
        if not block:
            empty.append(heading_text)

    assert not missing, (
        "day section(s) with no '## Verdicts' block: "
        f"{missing} -- every day from Day {MINIMUM_RETROFITTED_DAY} onward "
        "must open with one (Day 25, Objective 4)"
    )
    assert not empty, (
        f"day section(s) with an empty '## Verdicts' block: {empty} -- an "
        "empty block is the same defect as a missing one"
    )


def test_a_dummy_day_section_with_no_verdicts_block_is_caught() -> None:
    """The lint's own falsifiability check: construct a day section shaped
    exactly like a real one but missing the block, and confirm the
    detection logic (not just this test file) would flag it -- mirrors
    this project's convention of proving a gate can fail before trusting
    that it passing means something (see e.g. Day 23/24's registry-gate
    tests)."""
    fake_report = (
        "# Day 999\n\n"
        "Some headline prose with no Verdicts block at all.\n\n"
        "## Objective 0 — push, start and end of day\n\n"
        "Nothing here either.\n"
    )
    headings = list(_DAY_HEADING.finditer(fake_report))
    assert len(headings) == 1
    body = fake_report[headings[0].end() :]
    assert _VERDICTS_HEADING.search(body) is None
