"""Structural guards on scripts/fetch_dataset.py's registry writer.

Day 40. ``_update_registry_yaml`` used to round-trip the whole file through
``yaml.safe_load``/``yaml.safe_dump``, which drops every ``#`` comment, while its
docstring said comments survived. Last night's (Day 39) ``verify_license`` run
removed all 36 comment lines from configs/datasets.yaml that way (a2c1837),
including the header recording *why* every snapshot starts null and that the
in-code blocklist outranks the file. These tests pin the contract the docstring
only claimed: a write changes the lines of the field it updates, and nothing
else.

Every fixture here is a copy of the real registry or synthetic YAML; nothing is
fetched.
"""

from __future__ import annotations

import difflib
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_dataset  # noqa: E402

from src.data.registry import LicenseSnapshot, RegistryError  # noqa: E402

REAL_REGISTRY = Path(__file__).resolve().parent.parent / "configs" / "datasets.yaml"


@pytest.fixture
def registry_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "datasets.yaml"
    shutil.copyfile(REAL_REGISTRY, path)
    monkeypatch.setattr(fetch_dataset, "REGISTRY_PATH", path)
    return path


def _comment_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.lstrip().startswith("#")]


def _entry_line_span(text: str, name: str) -> range:
    """Line indices (0-based) from ``- name: <name>`` up to the next entry."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln == f"- name: {name}")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("- name: ")),
        len(lines),
    )
    return range(start, end)


def _changed_old_lines(before: str, after: str) -> set[int]:
    matcher = difflib.SequenceMatcher(
        a=before.splitlines(), b=after.splitlines(), autojunk=False
    )
    changed: set[int] = set()
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag != "equal":
            # an insertion touches no old line; anchor it to the line before it
            changed.update(range(i1, i2) if i2 > i1 else [max(i1 - 1, 0)])
    return changed


def _attestation_snapshot() -> dict[str, object]:
    """The exact shape verify_license writes for a hosting: huggingface entry."""
    sha = "f6571836e7a2bfbdf76a2f6ccaa5f11a660e832f"
    return LicenseSnapshot(
        url=f"https://huggingface.co/datasets/bdager/CHIRLA/raw/{sha}/README.md",
        verified_date=date(2026, 10, 1),
        text_sha256="a" * 64,
        verified_by="Test Human",
        verified_class="CC-BY-4.0",
        resolved_commit_sha=sha,
    ).model_dump(mode="json")


def test_noop_write_is_byte_identical(registry_copy: Path) -> None:
    before = registry_copy.read_bytes()
    fetch_dataset._update_registry_yaml("CHIRLA", {})
    assert registry_copy.read_bytes() == before


def test_rewriting_a_value_it_already_has_is_byte_identical(
    registry_copy: Path,
) -> None:
    before = registry_copy.read_bytes()
    current = yaml.safe_load(before)
    chirla = next(d for d in current["datasets"] if d["name"] == "CHIRLA")
    fetch_dataset._update_registry_yaml("CHIRLA", {"lane": chirla["lane"]})
    assert registry_copy.read_bytes() == before


def test_attestation_write_touches_only_that_entry(registry_copy: Path) -> None:
    """The real next write: CHIRLA's license_snapshot, null -> a mapping."""
    before = registry_copy.read_text()
    snapshot = _attestation_snapshot()

    fetch_dataset._update_registry_yaml("CHIRLA", {"license_snapshot": snapshot})
    after = registry_copy.read_text()

    span = _entry_line_span(before, "CHIRLA")
    stray = sorted(i for i in _changed_old_lines(before, after) if i not in span)
    assert not stray, [before.splitlines()[i] for i in stray]
    assert _comment_lines(after) == _comment_lines(before)

    old_payload = yaml.safe_load(before)
    new_payload = yaml.safe_load(after)
    for old, new in zip(old_payload["datasets"], new_payload["datasets"], strict=True):
        if old["name"] == "CHIRLA":
            assert new == {**old, "license_snapshot": snapshot}
        else:
            assert new == old


SYNTHETIC = """\
# Header comment -- the why lives here.
#
# A second header paragraph.
datasets:
  # -- section one --------------------------------
- name: Alpha
  lane: R
  # an inline note between two fields
  notes: first entry
  license_snapshot: null
  # -- section two --------------------------------
  # Context that belongs to the section below.
- name: Beta
  lane: S
  notes: 'a quoted scalar

    with a paragraph break'
# trailing comment at end of file
"""


def test_every_comment_survives_add_replace_and_nested_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails against the pre-Day-40 writer, which kept zero of these comments."""
    path = tmp_path / "datasets.yaml"
    path.write_text(SYNTHETIC)
    monkeypatch.setattr(fetch_dataset, "REGISTRY_PATH", path)

    fetch_dataset._update_registry_yaml(
        "Alpha", {"license_snapshot": {"url": "https://x.example/LICENSE"}}
    )
    fetch_dataset._update_registry_yaml("Alpha", {"hosting": "direct_url"})
    fetch_dataset._update_registry_yaml("Beta", {"notes": "replaced"})

    after = path.read_text()
    assert _comment_lines(after) == _comment_lines(SYNTHETIC)
    assert yaml.safe_load(after)["datasets"] == [
        {
            "name": "Alpha",
            "lane": "R",
            "notes": "first entry",
            "license_snapshot": {"url": "https://x.example/LICENSE"},
            "hosting": "direct_url",
        },
        {"name": "Beta", "lane": "S", "notes": "replaced"},
    ]


def test_unknown_entry_raises_and_leaves_the_file_alone(registry_copy: Path) -> None:
    before = registry_copy.read_bytes()
    with pytest.raises(RegistryError):
        fetch_dataset._update_registry_yaml("no-such-dataset", {"lane": "R"})
    assert registry_copy.read_bytes() == before
