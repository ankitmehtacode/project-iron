"""Checks that did not run: listed, grouped by reason, and audited (Day 42).

Six tests skipped on every machine for weeks because they looked for an
INT8 IR that ADR 0008 declares lost, while a manifested FP32 export sat in
``models/export/``. A skip is a statement that a check could not run; when
its named prerequisite exists elsewhere on the machine, that statement is
false and the skip is a defect. Two halves:

- ``scripts/suite_report.py`` groups every ``pytest.skip`` by reason and
  searches for each named prerequisite (tested here on hand-built junit);
- every day section of ``FOUNDATION_REPORT.md`` from Day 42 on restates the
  table in its Verdicts block (linted here).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import suite_report as sr  # noqa: E402
from test_blocker_ledger import (  # noqa: E402
    _BLOCKERS_HEADING,
    _VERDICTS_HEADING,
    _day_sections,
)

FIRST_DAY = 42
_TABLE_HEADING = re.compile(r"(?m)^### Checks that did not run\s*$")
_TABLE_HEADER = "| skipped | reason | prerequisite found elsewhere on this machine |"


def _junit(tmp_path: Path, cases: list[tuple[str, str | None, str | None]]) -> Path:
    """``(name, skip_type, message)``; skip_type None means the test ran."""
    rows = []
    for name, kind, message in cases:
        inner = (
            f'<skipped type="{kind}" message="{message}"/>' if kind is not None else ""
        )
        rows.append(
            f'<testcase classname="tests.test_x" name="{name}">{inner}</testcase>'
        )
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites><testsuite tests="%d">%s</testsuite></testsuites>'
        % (len(cases), "".join(rows))
    )
    return path


def test_skips_are_grouped_by_reason_and_xfails_are_not_skips(tmp_path: Path) -> None:
    path = _junit(
        tmp_path,
        [
            ("a", "pytest.skip", "no GPU"),
            ("b", "pytest.skip", "no GPU"),
            ("c", "pytest.xfail", "known bug"),
            ("d", None, None),
            ("e", "pytest.skip", "dataset absent"),
        ],
    )
    groups = sr.skips_by_reason(path, roots=())
    assert [(g.reason, len(g.tests)) for g in groups] == [
        ("no GPU", 2),
        ("dataset absent", 1),
    ]


def test_temporary_directories_do_not_split_one_reason(tmp_path: Path) -> None:
    path = _junit(
        tmp_path,
        [
            (
                "a",
                "pytest.skip",
                "nothing under /private/var/folders/x/T/pytest-of-u/pytest-1/o",
            ),
            (
                "b",
                "pytest.skip",
                "nothing under /private/var/folders/x/T/pytest-of-u/pytest-9/o",
            ),
        ],
    )
    (group,) = sr.skips_by_reason(path, roots=())
    assert len(group.tests) == 2 and group.named_paths == ()


def test_a_prerequisite_present_elsewhere_is_a_defect(tmp_path: Path) -> None:
    real = tmp_path / "models" / "weights" / "ckpt.pth"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"x")
    missing = tmp_path / "models" / "old" / "ckpt.pth"
    path = _junit(
        tmp_path, [("a", "pytest.skip", f"checkpoint not found at {missing}")]
    )
    # temp=None: these fixtures live in pytest's own temporary directory,
    # which the real report normalises away.
    (group,) = sr.skips_by_reason(path, roots=(tmp_path,), temp=None)
    assert group.defect and group.found_elsewhere == (str(real),)
    assert "**DEFECT**" in sr.render_did_not_run([group])


def test_the_int8_export_shape_is_reported_as_a_possible_defect(tmp_path: Path) -> None:
    """The case that motivated this: a different precision of the same
    model under another directory."""
    fp32 = tmp_path / "models" / "export" / "vjepa2_vitl_fp32.xml"
    fp32.parent.mkdir(parents=True)
    fp32.write_text("<xml/>")
    missing = tmp_path / "models" / "int8" / "vjepa2_vitl_int8.xml"
    path = _junit(tmp_path, [("a", "pytest.skip", f"IR not found at {missing}")])
    (group,) = sr.skips_by_reason(path, roots=(tmp_path,), temp=None)
    assert not group.defect
    assert group.similar_elsewhere == (str(fp32),)
    assert "POSSIBLE DEFECT" in sr.render_did_not_run([group])


def test_a_relative_path_is_not_read_as_an_absolute_one(tmp_path: Path) -> None:
    path = _junit(
        tmp_path, [("a", "pytest.skip", "fetch weights with scripts/fetch_weights.py")]
    )
    (group,) = sr.skips_by_reason(path, roots=(REPO_ROOT,))
    assert group.named_paths == () and not group.defect


def test_an_empty_table_says_every_check_ran() -> None:
    assert "every collected check ran" in sr.render_did_not_run([])


def test_every_day_from_42_restates_the_table_in_its_verdicts_block() -> None:
    sections = [s for s in _day_sections() if s[0] >= FIRST_DAY]
    assert sections, "no Day-42-or-later section to check"
    missing = []
    for _, heading, body in sections:
        verdicts = _VERDICTS_HEADING.search(body)
        blockers = _BLOCKERS_HEADING.search(body)
        if verdicts is None or blockers is None:
            missing.append(f"{heading}: no Verdicts/Blockers")
            continue
        block = body[verdicts.end() : blockers.start()]
        table = _TABLE_HEADING.search(block)
        if table is None or _TABLE_HEADER not in block[table.end() :]:
            missing.append(heading)
    assert not missing, (
        "Verdicts block(s) without a '### Checks that did not run' table "
        f"(scripts/suite_report.py writes it): {missing}"
    )
