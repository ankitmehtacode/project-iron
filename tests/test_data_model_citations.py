"""Day 27, Objective 3 -- the data-model-of-record audit, structural.

Every `§N` citation this codebase has made since Day 13 pointed at a
document ("IRON_DATA_MODEL v0.3") that was discussed but never checked
in -- `docs/data_model/v0.3.md` closes that gap, reconstructed from what
the code actually implements. This test is the structural half of the
fix, the same shape as the Day-25 Verdicts lint
(`tests/test_report_verdicts.py`): a NEW `§N` citation added to `src/`,
`tests/`, or `docs/adr/` that is not a section `docs/data_model/v0.3.md`
actually defines is exactly the failure this document exists to prevent
recurring, and fails here rather than being discovered by someone
grepping for it three days -- or three years -- later.

Two citation styles are deliberately NOT checked here, because they are
not part of this data model's numbering at all: RFC citations (e.g. "RFC
3550 §6.4.1" in `src/ingest/`, citing an external standard, not this
project's own spec) and `docs/site_zero_consent_TEMPLATE.md`'s own
internal section numbers (self-contained within that one document,
verified separately by `tests/test_identity_adr.py`). Excluding them is
not a loophole -- their sections ARE defined, just not by this file, and
folding them into this document's numbering would create ambiguity this
project has already been burned by once (Day 25/26's own "architecture
in conversation treated as architecture of record" pattern, applied in
reverse here: pretending a well-defined RFC section is undefined would
be its own kind of false alarm).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_MODEL_PATH = REPO_ROOT / "docs" / "data_model" / "v0.3.md"
THIS_FILE = Path(__file__).resolve()
"""Excluded from the scan below: this file's own §999 fixture citation
(used to prove the detection logic fires, see
test_the_lint_actually_catches_a_dangling_citation) is test data, not a
real citation in the codebase."""

_SECTION_HEADING = re.compile(r"(?m)^## §(\d+)")
_CITATION = re.compile(r"§(\d+)")
_SCAN_DIRS = ("src", "tests", "docs/adr")
_EXTERNAL_REFERENCE_MARKERS = (
    "RFC",
    "consent template",
    "site_zero_consent",
    "own report",
    "paraphrased as",
)
"""A §N citation within a 3-line window (previous+current+next) of any of
these is a reference to a document with its OWN, separately-defined
numbering -- not part of IRON_DATA_MODEL v0.3 -- and is excluded rather
than flagged as dangling: "RFC" (an external standard, e.g. "RFC 3550
§6.4.1" in `src/ingest/`), "consent template" / "site_zero_consent"
(`docs/site_zero_consent_TEMPLATE.md`'s own sections, self-contained and
covered separately by `tests/test_identity_adr.py`), "own report" (a
day's own `FOUNDATION_REPORT.md` entry citing its OWN internal item
numbering, as ADR 0009 does quoting Day 1's report -- narrative
self-reference, not a data-model claim), "paraphrased as" (prose
describing an ALREADY-RECORDED finding about a missing citation --
ADR 0011's own account of Day 26's "Data model v0.3 §3" discovery -- not
a fresh claim on that section). A 3-line window, not just the citation's
own line, because markdown soft-wraps a sentence across lines and the
trigger phrase can land on either side of the break."""

_CONSENT_TEMPLATE_SECTIONS = {1, 2, 5, 7, 8}
"""`docs/site_zero_consent_TEMPLATE.md`'s own confirmed section range
(grep-verified). A file that mentions the consent template ANYWHERE
may refer to "the §7 limitation" many sentences after establishing that
§7 means the consent template's §7 (ADR 0001 does this) -- too far for
any reasonable line window. Rather than parse markdown structure to find
the antecedent, a file that mentions the consent template anywhere has
every citation IN THIS SPECIFIC RANGE treated as consent-template
sections, since this project's own data model has never cited any of
them (its real citations are §0/§10/§15/§17, verified disjoint from this
set) -- narrow enough not to mask a genuine future data-model citation
that happens to share a file with an unrelated consent-template mention
outside this range."""


def _defined_sections() -> set[int]:
    text = DATA_MODEL_PATH.read_text(encoding="utf-8")
    return {int(m.group(1)) for m in _SECTION_HEADING.finditer(text)}


def _citations_in(path: Path) -> list[tuple[int, int, str]]:
    """(line_number, section_number, line_text) for every §N citation in
    ``path`` not excluded by :data:`_EXTERNAL_REFERENCE_MARKERS` (checked
    in a 3-line window) or :data:`_CONSENT_TEMPLATE_SECTIONS` (checked
    file-wide, only when the file mentions the consent template at all)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    file_mentions_consent_template = any(
        "consent template" in line or "site_zero_consent" in line for line in lines
    )
    out = []
    for match in _CITATION.finditer(text):
        line_index = text.count("\n", 0, match.start())  # 0-based
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.start())
        line_text = text[line_start : line_end if line_end != -1 else None].strip()
        section = int(match.group(1))

        window = "\n".join(lines[max(0, line_index - 1) : line_index + 2])
        if any(marker in window for marker in _EXTERNAL_REFERENCE_MARKERS):
            continue
        if file_mentions_consent_template and section in _CONSENT_TEMPLATE_SECTIONS:
            continue
        out.append((line_index + 1, section, line_text))
    return out


def test_data_model_document_exists() -> None:
    assert DATA_MODEL_PATH.exists(), f"{DATA_MODEL_PATH} not found"


def test_data_model_defines_at_least_the_known_sections() -> None:
    """Falsifiability floor: if this drops to zero, the heading regex or
    the document itself broke silently."""
    defined = _defined_sections()
    assert defined >= {0, 10, 15, 17}


def test_no_dangling_data_model_citation_in_source_or_adrs() -> None:
    defined = _defined_sections()
    dangling: list[str] = []
    for scan_dir in _SCAN_DIRS:
        root = REPO_ROOT / scan_dir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in (".py", ".md"):
                continue
            if path == DATA_MODEL_PATH or path.resolve() == THIS_FILE:
                continue
            for line_number, section, line_text in _citations_in(path):
                if section not in defined:
                    rel = path.relative_to(REPO_ROOT)
                    dangling.append(
                        f"{rel}:{line_number} cites §{section} -- {line_text!r}"
                    )
    assert not dangling, (
        "dangling data-model citation(s) found (§N with no matching "
        f"'## §N' heading in {DATA_MODEL_PATH.relative_to(REPO_ROOT)}):\n"
        + "\n".join(dangling)
    )


def test_the_lint_actually_catches_a_dangling_citation(tmp_path: Path) -> None:
    """Falsifiability, per this project's own convention (Day 24/25's
    registry-gate tests, the Verdicts lint's own fake-section test):
    prove the detection logic would fire before trusting that it passing
    on the real tree means anything."""
    fake_source = tmp_path / "fake_module.py"
    fake_source.write_text('"""Cites §999, which nothing defines."""\n')
    citations = _citations_in(fake_source)
    assert citations == [(1, 999, '"""Cites §999, which nothing defines."""')]
    defined = _defined_sections()
    assert 999 not in defined


def test_external_reference_citations_are_excluded_not_silently_passed(
    tmp_path: Path,
) -> None:
    """The exclusion markers are deliberate and narrow -- confirm real
    data-model citations are NOT swept up as collateral damage, and that
    each external-reference style is actually recognized."""
    fake = tmp_path / "fake_module.py"
    # Blank-line separated so each example's window (prev+current+next
    # line) cannot bleed into its neighbor's -- these are five
    # INDEPENDENT cases, not one paragraph.
    fake.write_text(
        "\n\n".join(
            [
                "# (RFC 3550 §6.4.1)",
                "# The consent template §7 limitation applies here.",
                "# site_zero_consent §2 covers biometric derivatives.",
                "# Day 1's own report says so directly (§7).",
                "# stage 4 of §15 always runs",
            ]
        )
        + "\n"
    )
    citations = _citations_in(fake)
    sections_found = [section for _, section, _ in citations]
    assert sections_found == [15], (
        f"expected only the real data-model citation (§15) to survive "
        f"exclusion, got {sections_found}"
    )
