"""Day 37, Objective 1 -- the Blocker Ledger, rendered honestly.

Day 36 quantified one blocker precisely: MEVA's license verification was
restated as "still pending" 52 separate times across 34 FOUNDATION_REPORT.md
day-sections, and never once acted on. That is not a MEVA-specific failure --
it is what happens to any blocker whose only record is a sentence in a
narrative report that nothing forces anyone to re-check. `docs/
blocker_ledger.yaml` is the fix applied to the project's full human-action
backlog, not just the one instance that happened to get counted: a single,
structured record of every blocker in this project that is gated on a human
action, not a code defect.

This script is the honest-aging half of that fix. It NEVER estimates an
entry's age from a day number -- day numbers in this project's narrative do
not track calendar days (Days 20-36 were all committed on 2026-09-07,
verified directly against `git log`). Age is always `as_of - first_recorded
date`, both real calendar dates.

    python scripts/blocker_report.py                  # table, sorted oldest-first
    python scripts/blocker_report.py --as-of 2026-09-08
    python scripts/blocker_report.py --verify-git      # also checks every sha
                                                        # against real git history

Exit codes:
    0  ledger loaded and rendered (with --verify-git: every sha verified)
    1  the ledger is malformed, or (with --verify-git) a sha/date entry does
       not match what git actually records -- this script refuses to render
       a table it cannot vouch for rather than print stale numbers silently
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "docs" / "blocker_ledger.yaml"

VALID_CATEGORIES = frozenset(
    {"human_verification", "human_capture", "human_procurement", "human_decision"}
)


class LedgerError(ValueError):
    """The ledger file itself is malformed -- missing/invalid fields."""


@dataclass(frozen=True)
class BlockerEntry:
    id: str
    description: str
    category: str
    first_recorded_sha: str
    first_recorded_date: date
    unblocks: tuple[str, ...]
    estimated_human_effort: str

    def age_days(self, as_of: date) -> int:
        return (as_of - self.first_recorded_date).days


def load_ledger(path: Path = LEDGER_PATH) -> list[BlockerEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise LedgerError(f"{path} must contain a non-empty list of entries")

    entries: list[BlockerEntry] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw):
        entries.append(_parse_entry(item, index=i, path=path))
    for entry in entries:
        if entry.id in seen_ids:
            raise LedgerError(f"duplicate blocker id: {entry.id!r}")
        seen_ids.add(entry.id)
    return entries


def _parse_entry(item: Any, *, index: int, path: Path) -> BlockerEntry:
    if not isinstance(item, dict):
        raise LedgerError(f"{path}: entry #{index} is not a mapping")

    def require(field: str) -> Any:
        if field not in item or item[field] in (None, ""):
            raise LedgerError(
                f"{path}: entry #{index} missing required field {field!r}"
            )
        return item[field]

    entry_id = require("id")
    category = require("category")
    if category not in VALID_CATEGORIES:
        raise LedgerError(
            f"{path}: entry {entry_id!r} has invalid category {category!r}, "
            f"must be one of {sorted(VALID_CATEGORIES)}"
        )

    first_recorded = require("first_recorded")
    if (
        not isinstance(first_recorded, dict)
        or "sha" not in first_recorded
        or "date" not in first_recorded
    ):
        raise LedgerError(
            f"{path}: entry {entry_id!r}'s first_recorded must have "
            "both 'sha' and 'date'"
        )
    recorded_date = first_recorded["date"]
    if not isinstance(recorded_date, date):
        raise LedgerError(
            f"{path}: entry {entry_id!r}'s first_recorded.date must parse as a "
            f"YAML date (got {recorded_date!r})"
        )

    unblocks = require("unblocks")
    if not isinstance(unblocks, list) or not unblocks:
        raise LedgerError(
            f"{path}: entry {entry_id!r}'s unblocks must be a non-empty list"
        )

    return BlockerEntry(
        id=entry_id,
        description=str(require("description")).strip(),
        category=category,
        first_recorded_sha=str(first_recorded["sha"]),
        first_recorded_date=recorded_date,
        unblocks=tuple(str(u).strip() for u in unblocks),
        estimated_human_effort=str(require("estimated_human_effort")).strip(),
    )


class GitVerificationError(ValueError):
    """A ledger entry's first_recorded sha/date does not match real git history."""


def verify_against_git(
    entries: list[BlockerEntry], *, repo_root: Path = REPO_ROOT
) -> None:
    """Confirm every entry's sha actually exists in this repository's history,
    and that the date recorded in the ledger matches what git recorded for
    that commit. This is the check that keeps the ledger from drifting into
    the exact failure mode it exists to catch -- an assertion nobody
    re-verifies."""
    for entry in entries:
        result = subprocess.run(
            ["git", "cat-file", "-e", entry.first_recorded_sha],
            cwd=repo_root,
            capture_output=True,
        )
        if result.returncode != 0:
            raise GitVerificationError(
                f"{entry.id}: sha {entry.first_recorded_sha!r} does not exist "
                "in this repository's git history"
            )
        show = subprocess.run(
            [
                "git",
                "show",
                "-s",
                "--format=%ad",
                "--date=short",
                entry.first_recorded_sha,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        committed_date = date.fromisoformat(show.stdout.strip())
        if committed_date != entry.first_recorded_date:
            raise GitVerificationError(
                f"{entry.id}: ledger records first_recorded.date="
                f"{entry.first_recorded_date}, but git shows "
                f"{entry.first_recorded_sha} was committed on {committed_date}"
            )


def sorted_oldest_first(entries: list[BlockerEntry], as_of: date) -> list[BlockerEntry]:
    return sorted(entries, key=lambda e: (-e.age_days(as_of), e.id))


def render_table(entries: list[BlockerEntry], as_of: date) -> str:
    ordered = sorted_oldest_first(entries, as_of)
    lines = [
        f"Blocker Ledger -- as of {as_of.isoformat()} ({len(ordered)} open blockers)",
        "",
        "| age (days) | id | category | first recorded | estimated human effort |",
        "|---:|---|---|---|---|",
    ]
    for e in ordered:
        lines.append(
            f"| {e.age_days(as_of)} | `{e.id}` | {e.category} | "
            f"{e.first_recorded_date.isoformat()} (`{e.first_recorded_sha[:7]}`) | "
            f"{e.estimated_human_effort} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=None,
        help="date to compute ages against (default: today)",
    )
    parser.add_argument(
        "--verify-git",
        action="store_true",
        help="also verify every entry's sha/date against real git history",
    )
    args = parser.parse_args(argv)
    as_of = args.as_of or date.today()

    try:
        entries = load_ledger()
        if args.verify_git:
            verify_against_git(entries)
    except (LedgerError, GitVerificationError) as exc:
        print(f"blocker_report: {exc}", file=sys.stderr)
        return 1

    print(render_table(entries, as_of))
    return 0


if __name__ == "__main__":
    sys.exit(main())
