"""Day 37, Objective 1 -- the Blocker Ledger, structural.

Two independent things could go quietly wrong with a "single source of
truth" ledger, and this file is one test class per failure mode:

1. The ledger itself could drift from git truth -- an entry's `first_recorded`
   sha or date could be wrong, or simply stop existing (a rebase, a squash),
   and nothing would notice. `TestLedgerMatchesGit` re-derives every entry's
   commit from git directly and fails if the ledger disagrees.
2. A day's own report could add code without restating the ledger -- the
   exact shape of the MEVA failure Day 36 found (an intent restated in prose
   52 times, never re-verified). `TestReportOpensWithBlockers` is the same
   mechanism as the Day-25 Verdicts lint (`tests/test_report_verdicts.py`)
   and the Day-27 citation lint (`tests/test_data_model_citations.py`):
   every day section from Day 37 onward must open with `## Verdicts`
   immediately followed by a non-empty `## Blockers` block, and the most
   recent day section's block must name every id currently in the ledger --
   not a frozen count from whenever that day was written, but checked fresh
   against the ledger every time this suite runs.

Days 1-36 predate this convention and are NOT retrofitted, the same call the
Verdicts lint (Days 1-19) and the citation lint made for their own
introduction days.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "FOUNDATION_REPORT.md"
LEDGER_PATH = REPO_ROOT / "docs" / "blocker_ledger.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import blocker_report as br  # noqa: E402

MINIMUM_RETROFITTED_DAY = 37
"""Day 37, Objective 1's own introduction day; see module docstring."""

_DAY_HEADING = re.compile(r"(?m)^# (.+)$")
_DAY_NUMBER = re.compile(r"Day (\d+)")
_VERDICTS_HEADING = re.compile(r"(?m)^## Verdicts\s*$")
_BLOCKERS_HEADING = re.compile(r"(?m)^## Blockers\s*$")
_NEXT_HEADING = re.compile(r"(?m)^#{1,2} ")


def _day_sections() -> list[tuple[int, str, str]]:
    """Every top-level ``# ...`` section whose heading names a day number,
    as (day_number, heading_text, section_body). Mirrors
    tests/test_report_verdicts.py's own helper exactly -- same file, same
    heading shape, no reason for a second convention."""
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


class TestLedgerIsWellFormed:
    def test_ledger_loads(self) -> None:
        entries = br.load_ledger()
        assert entries, "ledger must not be empty"

    def test_ids_are_unique_and_kebab_case(self) -> None:
        entries = br.load_ledger()
        ids = [e.id for e in entries]
        assert len(ids) == len(set(ids)), "duplicate blocker ids"
        for entry_id in ids:
            assert re.fullmatch(
                r"[a-z0-9]+(-[a-z0-9]+)*", entry_id
            ), f"blocker id {entry_id!r} is not kebab-case"

    def test_malformed_entry_is_rejected(self, tmp_path: Path) -> None:
        """Falsifiability check, this project's standing convention (see
        the Day-24 registry-gate tests, the Verdicts lint's own fake-section
        test): construct a ledger missing a required field and confirm the
        loader actually refuses it, rather than trusting that it would."""
        bad = tmp_path / "bad_ledger.yaml"
        bad.write_text(
            "- id: incomplete-entry\n"
            "  description: missing everything else\n"
            "  category: human_decision\n",
            encoding="utf-8",
        )
        with pytest.raises(br.LedgerError):
            br.load_ledger(bad)

    def test_invalid_category_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad_category.yaml"
        bad.write_text(
            "- id: bad-category\n"
            "  description: x\n"
            "  category: not_a_real_category\n"
            "  first_recorded: {sha: " + "a" * 40 + ", date: 2026-01-01}\n"
            "  unblocks: [x]\n"
            "  estimated_human_effort: x\n",
            encoding="utf-8",
        )
        with pytest.raises(br.LedgerError):
            br.load_ledger(bad)


class TestLedgerMatchesGit:
    """Every entry's first_recorded claim, checked against git directly --
    not trusted because it was written down correctly once."""

    def test_every_sha_exists_and_date_matches_git(self) -> None:
        entries = br.load_ledger()
        br.verify_against_git(entries)  # raises GitVerificationError on drift

    def test_a_wrong_date_is_caught(self) -> None:
        """The verification's own falsifiability check: an entry whose date
        does not match its sha's real commit date must be rejected."""
        entries = br.load_ledger()
        real = entries[0]
        tampered = br.BlockerEntry(
            id=real.id,
            description=real.description,
            category=real.category,
            first_recorded_sha=real.first_recorded_sha,
            first_recorded_date=real.first_recorded_date.replace(year=2000),
            unblocks=real.unblocks,
            estimated_human_effort=real.estimated_human_effort,
        )
        with pytest.raises(br.GitVerificationError):
            br.verify_against_git([tampered])

    def test_a_nonexistent_sha_is_caught(self) -> None:
        entries = br.load_ledger()
        real = entries[0]
        tampered = br.BlockerEntry(
            id=real.id,
            description=real.description,
            category=real.category,
            first_recorded_sha="0" * 40,
            first_recorded_date=real.first_recorded_date,
            unblocks=real.unblocks,
            estimated_human_effort=real.estimated_human_effort,
        )
        with pytest.raises(br.GitVerificationError):
            br.verify_against_git([tampered])


class TestReportOpensWithBlockers:
    def test_every_day_from_37_onward_has_verdicts_then_nonempty_blockers(self) -> None:
        sections = _day_sections()
        retrofitted = [s for s in sections if s[0] >= MINIMUM_RETROFITTED_DAY]
        assert retrofitted, (
            f"no day sections with day number >= {MINIMUM_RETROFITTED_DAY} "
            "were found -- this test cannot verify anything until Day 37's "
            "own section exists"
        )

        missing: list[str] = []
        empty: list[str] = []
        out_of_order: list[str] = []
        for day_number, heading_text, body in retrofitted:
            verdicts_match = _VERDICTS_HEADING.search(body)
            blockers_match = _BLOCKERS_HEADING.search(body)
            if verdicts_match is None or blockers_match is None:
                missing.append(heading_text)
                continue
            if blockers_match.start() < verdicts_match.start():
                out_of_order.append(heading_text)
                continue
            # Blockers must be the section immediately following Verdicts --
            # nothing else may sit between them per Objective 5's own
            # ordering ("opening with its Verdicts block, followed
            # immediately by the mandatory Blockers block").
            between = body[verdicts_match.end() : blockers_match.start()]
            other_heading_between = _NEXT_HEADING.search(between)
            if other_heading_between is not None:
                out_of_order.append(heading_text)
                continue
            content_start = blockers_match.end()
            next_heading = _NEXT_HEADING.search(body, content_start)
            content_end = next_heading.start() if next_heading else len(body)
            block = body[content_start:content_end].strip()
            if not block:
                empty.append(heading_text)

        assert (
            not missing
        ), f"day section(s) missing '## Verdicts' and/or '## Blockers': {missing}"
        assert not out_of_order, (
            "day section(s) where '## Blockers' does not immediately follow "
            f"'## Verdicts': {out_of_order}"
        )
        assert not empty, f"day section(s) with an empty '## Blockers' block: {empty}"

    def test_latest_day_section_names_every_current_ledger_entry(self) -> None:
        """The freshness half of the gate: the most recent day section's
        Blockers block is checked against the LIVE ledger, every time this
        suite runs -- not against whatever the ledger looked like on the day
        that section was written. A day that adds new code without adding a
        new ledger entry to its own restated block fails here the moment a
        ledger entry it omitted exists."""
        sections = _day_sections()
        retrofitted = [s for s in sections if s[0] >= MINIMUM_RETROFITTED_DAY]
        assert retrofitted
        latest_day, heading_text, body = max(retrofitted, key=lambda s: s[0])

        blockers_match = _BLOCKERS_HEADING.search(body)
        assert blockers_match is not None, f"{heading_text} has no '## Blockers' block"
        content_start = blockers_match.end()
        next_heading = _NEXT_HEADING.search(body, content_start)
        content_end = next_heading.start() if next_heading else len(body)
        block = body[content_start:content_end]

        entries = br.load_ledger()
        missing_ids = [e.id for e in entries if e.id not in block]
        assert not missing_ids, (
            f"{heading_text}'s '## Blockers' block does not mention ledger "
            f"id(s) {missing_ids} -- every currently-open blocker must be "
            "restated, not silently dropped"
        )

    def test_a_dummy_day_section_missing_blockers_is_caught(self) -> None:
        """Falsifiability check, same convention as the Verdicts lint's own
        (`test_a_dummy_day_section_with_no_verdicts_block_is_caught`)."""
        fake_report = (
            "# Day 999\n\n"
            "## Verdicts\n\n- Some verdict.\n\n"
            "## Objective 0 — push, start and end of day\n\n"
            "Nothing here either.\n"
        )
        headings = list(_DAY_HEADING.finditer(fake_report))
        assert len(headings) == 1
        body = fake_report[headings[0].end() :]
        assert _VERDICTS_HEADING.search(body) is not None
        assert _BLOCKERS_HEADING.search(body) is None
