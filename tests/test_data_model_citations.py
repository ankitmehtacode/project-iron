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

Day 28, Objective 4 extension -- provision status and pointer resolution
--------------------------------------------------------------------------
Neither constraint typing nor the hypothesis store's `PRUNED_BY_BUDGET`
(both traced and implemented Day 28 -- see `docs/data_model/v0.3.md`'s
"Two provisions, traced" section) was ever a dangling `§N` citation: both
were normative claims inside sections, or a punch-list topic, that
already existed. The citation lint above cannot catch a provision that
carries no `§N` citation at all. This extension adds the check that
CAN be made cheaply: every normative provision this document records
(each `§N` section's own `**Status: ...**` line, and each row of the
"Catalog" table) states a status, and a status of `IMPLEMENTED` names a
pointer -- a path, optionally `path::symbol` -- that must resolve to
something real (file/directory exists; a named symbol is actually
defined in it, checked by text search, not a full import). A provision
marked `IMPLEMENTED` whose pointer does not resolve fails here, the same
way a dangling `§N` citation fails above.

This does not close the gap `PRUNED_BY_BUDGET` itself fell through --
that citation never touched version control before the audit that found
it missing, and no lint scanning `src/`, `tests/`, or `docs/adr/` can see
a claim that was never written into any of them. See the doc's own
account of that provision's trace for why the closing action there is
about citation discipline, not tooling.
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


_STATUS_LINE = re.compile(
    r"\*\*Status:\s*(?P<status>[^*]+?)(?:\s+—\s+(?P<pointers>(?:`[^`]+`,?\s*)+))?\.\*\*"
)
_TABLE_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")
_BACKTICK_SPAN = re.compile(r"`([^`]+)`")
_ADR_REFERENCE = re.compile(r"ADR (\d{4})")


class Provision:
    """One normative claim recorded in the data model doc: a `§N`
    section's own Status line, or one row of the Catalog table.

    ``pointers`` are repo-relative paths (optionally ``path::symbol``);
    ``adr_pointers`` are ``ADR NNNN`` references, checked differently
    (see :func:`_adr_pointer_resolves`) since the doc never spells out
    an ADR file's full slug, only its number.
    """

    def __init__(
        self,
        label: str,
        status: str,
        pointers: tuple[str, ...],
        adr_pointers: tuple[str, ...] = (),
    ) -> None:
        self.label = label
        self.status = status
        self.pointers = pointers
        self.adr_pointers = adr_pointers

    @property
    def is_implemented(self) -> bool:
        # Same prefix-match convention this document already used before
        # today for the Catalog table's Status cells (e.g. "IMPLEMENTED,
        # matches its ADR").
        return self.status.strip().startswith("IMPLEMENTED")


def _pointer_resolves(pointer: str) -> bool:
    """A pointer is a repo-relative path, optionally ``path::symbol``.

    Resolves if the path exists and, when a symbol is named, the file's
    text defines it. Text search, not an import -- this project's
    provisions span modules with heavy runtime dependencies that a doc
    lint should not need to load.
    """
    path_part, _, symbol = pointer.partition("::")
    path = REPO_ROOT / path_part
    if not path.exists():
        return False
    if not symbol or path.is_dir():
        return True
    text = path.read_text(encoding="utf-8", errors="ignore")
    escaped = re.escape(symbol)
    return bool(
        re.search(rf"^\s*(class|def)\s+{escaped}\b", text, re.MULTILINE)
        or re.search(rf"^{escaped}\s*[:=]", text, re.MULTILINE)
    )


def _path_like_pointers(raw_pointers: str | None) -> tuple[str, ...]:
    """Extract backtick spans that look like file paths (contain '/') from
    a captured pointer group. Bare symbol mentions used as informal
    shorthand for "the same file as the previous item" (e.g. the Catalog
    table's ``EvidenceCommitment`` beside
    ``src/model/evidence.py::compute_merkle_root``) are not paths and are
    not pointers this lint can check on their own."""
    if not raw_pointers:
        return ()
    return tuple(span for span in _BACKTICK_SPAN.findall(raw_pointers) if "/" in span)


def _section_status_provisions(text: str) -> list[Provision]:
    provisions = []
    for match in _STATUS_LINE.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        pointers = _path_like_pointers(match.group("pointers"))
        provisions.append(
            Provision(
                label=f"line {line_no} Status block",
                status=match.group("status"),
                pointers=pointers,
            )
        )
    return provisions


def _catalog_table_provisions(text: str) -> list[Provision]:
    provisions = []
    in_table = False
    for line_no, line in enumerate(text.split("\n"), start=1):
        if line.strip().startswith("| Provision | Status |"):
            in_table = True
            continue
        if not in_table:
            continue
        row_match = _TABLE_ROW.match(line)
        if row_match is None:
            in_table = False
            continue
        cells = [c.strip() for c in row_match.group("cells").split("|")]
        if len(cells) != 4 or set(cells[0]) <= {"-"}:
            continue  # header separator row ("| --- | --- | --- | --- |")
        provision_name, status, governing_doc, code = cells
        pointers = _path_like_pointers(code)
        adr_match = _ADR_REFERENCE.search(governing_doc)
        adr_pointers = (adr_match.group(1),) if adr_match else ()
        provisions.append(
            Provision(
                label=f"Catalog row {provision_name!r}",
                status=status,
                pointers=pointers,
                adr_pointers=adr_pointers,
            )
        )
    return provisions


def _adr_pointer_resolves(adr_number: str) -> bool:
    """An ADR pointer names a number, not an exact filename (the doc's
    own Catalog table only ever writes "ADR 0011", never the file's full
    slug) -- resolves if any file in ``docs/adr/`` starts with that
    zero-padded number."""
    adr_dir = REPO_ROOT / "docs" / "adr"
    return any(adr_dir.glob(f"{adr_number}-*.md"))


def _all_provisions(text: str) -> list[Provision]:
    return _section_status_provisions(text) + _catalog_table_provisions(text)


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


def test_every_section_status_line_declares_a_recognized_status() -> None:
    """Falsifiability floor for the parser itself: if this drops to zero,
    the Status-line regex broke silently, same convention as
    test_data_model_defines_at_least_the_known_sections above."""
    text = DATA_MODEL_PATH.read_text(encoding="utf-8")
    provisions = _section_status_provisions(text)
    assert len(provisions) >= 4, (
        "expected at least the four §-section Status lines (§0/§10/§15/§17); "
        f"found {len(provisions)} -- did the Status-line format change?"
    )


def test_no_broken_implemented_pointer() -> None:
    """Day 28, Objective 4: every provision recorded in the data model
    doc as IMPLEMENTED (a §-section Status line or a Catalog row) must
    name a pointer that resolves -- a real path, and a real symbol in it
    when one is named. This is the check a bare `§N`-citation lint could
    never have caught for either provision traced today: neither was a
    dangling citation, both were unrecorded claims inside sections that
    already existed."""
    text = DATA_MODEL_PATH.read_text(encoding="utf-8")
    broken: list[str] = []
    for provision in _all_provisions(text):
        if not provision.is_implemented:
            continue
        for pointer in provision.pointers:
            if not _pointer_resolves(pointer):
                broken.append(
                    f"{provision.label}: pointer {pointer!r} does not resolve"
                )
        for adr_number in provision.adr_pointers:
            if not _adr_pointer_resolves(adr_number):
                broken.append(
                    f"{provision.label}: no docs/adr/{adr_number}-*.md "
                    f"for ADR {adr_number}"
                )
    assert not broken, (
        "provision(s) marked IMPLEMENTED with a pointer that does not "
        "resolve:\n" + "\n".join(broken)
    )


def test_no_provision_marked_implemented_with_zero_pointers() -> None:
    """An IMPLEMENTED provision with no pointer at all is unfalsifiable
    by this lint -- it would pass test_no_broken_implemented_pointer
    vacuously. Every current §-section and Catalog row names at least
    one path; a future one that doesn't should be caught here rather
    than silently slipping past the pointer-resolution check above."""
    text = DATA_MODEL_PATH.read_text(encoding="utf-8")
    unpointed = [
        provision.label
        for provision in _all_provisions(text)
        if provision.is_implemented
        and not provision.pointers
        and not provision.adr_pointers
    ]
    assert not unpointed, (
        "provision(s) marked IMPLEMENTED with no pointer to check at all "
        f"(add a path or an ADR reference): {unpointed}"
    )


def test_the_lint_actually_catches_a_broken_implemented_pointer() -> None:
    """Falsifiability, per this project's own convention: prove the
    pointer-resolution check would fire on a real broken pointer before
    trusting that it passing on the actual doc means anything."""
    fake_text = (
        "## §0 -- Fake section\n\n"
        "**Status: IMPLEMENTED — `src/model/this_file_does_not_exist.py`.** "
        "Filler prose.\n"
    )
    provisions = _section_status_provisions(fake_text)
    assert len(provisions) == 1
    assert provisions[0].is_implemented
    assert provisions[0].pointers == ("src/model/this_file_does_not_exist.py",)
    assert not _pointer_resolves(provisions[0].pointers[0])


def test_the_lint_does_not_flag_a_real_pointer() -> None:
    fake_text = (
        "## §0 -- Fake section\n\n"
        "**Status: IMPLEMENTED — `src/model/hypothesis.py::HypothesisStore`.** "
        "Filler prose.\n"
    )
    provisions = _section_status_provisions(fake_text)
    assert _pointer_resolves(provisions[0].pointers[0])


def test_the_lint_does_not_flag_a_symbol_only_mention_as_a_broken_pointer() -> None:
    """A bare backtick span with no '/' (e.g. the Catalog table's
    ``EvidenceCommitment`` beside a real path) is informal shorthand, not
    a pointer this lint claims to check -- see _path_like_pointers."""
    fake_text = (
        "| Provision | Status | Governing document | Code |\n"
        "| --- | --- | --- | --- |\n"
        "| Fake | IMPLEMENTED, matches its ADR | ADR 0007 | "
        "`src/model/evidence.py::compute_merkle_root`, `SomeBareSymbol` |\n"
    )
    provisions = _catalog_table_provisions(fake_text)
    assert len(provisions) == 1
    assert provisions[0].pointers == ("src/model/evidence.py::compute_merkle_root",)


def test_no_assumed_nowhere_recorded_provision_remains_unnamed() -> None:
    """Day 28, Objective 4's own acceptance: after today's two
    implementations, any provision still ASSUMED-NOWHERE-RECORDED must
    be listed by name in the doc, not silently dropped. Both provisions
    traced today (constraint typing, the hypothesis store) must now
    read IMPLEMENTED."""
    text = DATA_MODEL_PATH.read_text(encoding="utf-8")
    assert "constraint typing" in text.lower()
    assert "PRUNED_BY_BUDGET" in text
    for provision in _all_provisions(text):
        label_lower = provision.label.lower()
        if "constraint typing" in label_lower or "hypothesis store" in label_lower:
            assert provision.is_implemented, (
                f"{provision.label} should read IMPLEMENTED after Day 28's "
                "Objectives 2/3 -- got status {provision.status!r}"
            )
