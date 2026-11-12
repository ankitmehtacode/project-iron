"""scripts/sign_consent.py: the sole-subject consent signature (Day 45)."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.fetch_dataset as fetch
import scripts.sign_consent as sign
import src.data.registry as reg
from src.data.registry import DatasetRegistry, LaneViolation

MINIMAL_REGISTRY = """\
datasets:
- name: site-zero-solo
  lane: C
  subsystem: everything
  hypothesis_class: ours, consent signed by the sole subject
  notes: 'test fixture'
  # a comment that must survive the signature edit
- name: PhysicalAI-SmartSpaces
  lane: S
  subsystem: everything
"""


@pytest.fixture()
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # A fixed two-entry registry, not a copy of the real one: the real
    # site-zero-solo entry is signed once the owner signs, and these tests
    # must not depend on whether that has happened.
    (tmp_path / "configs").mkdir()
    registry = tmp_path / "configs" / "datasets.yaml"
    registry.write_text(MINIMAL_REGISTRY)
    monkeypatch.setattr(fetch, "REGISTRY_PATH", registry)
    monkeypatch.setattr(sign, "REGISTRY_PATH", registry)
    monkeypatch.setattr(sign, "ROOT", tmp_path)
    monkeypatch.setattr(reg, "_REPO_ROOT", tmp_path)
    return tmp_path


def _args(sandbox: Path, *extra: str) -> list[str]:
    return [
        "site-zero-solo",
        "--signed-by",
        "Test Signer",
        "--captured-on",
        "2026-11-12",
        "--consent-dir",
        str(sandbox / "docs" / "consent"),
        *extra,
    ]


def test_refuses_without_the_signature_flag(sandbox: Path) -> None:
    assert sign.main(_args(sandbox)) == 2
    assert not (sandbox / "docs" / "consent").exists()


def test_signing_opens_the_dataset_and_is_not_repeatable(sandbox: Path) -> None:
    registry_path = sandbox / "configs" / "datasets.yaml"
    with pytest.raises(LaneViolation, match="no consent record"):
        DatasetRegistry.load(registry_path).open_for_eval("site-zero-solo")
    assert sign.main(_args(sandbox, "--i-am-the-sole-subject")) == 0
    entry = DatasetRegistry.load(registry_path).open_for_eval("site-zero-solo")
    assert entry.consent_record is not None and entry.consent_record.subjects == 1
    assert sign.main(_args(sandbox, "--i-am-the-sole-subject")) == 2


def test_only_the_consent_field_changes(sandbox: Path) -> None:
    registry_path = sandbox / "configs" / "datasets.yaml"
    before = registry_path.read_text().splitlines()
    assert sign.main(_args(sandbox, "--i-am-the-sole-subject")) == 0
    after = registry_path.read_text().splitlines()
    removed = [line for line in before if line not in after]
    assert removed == []
    added = [line for line in after if line not in before]
    assert added[0].strip() == "consent_record:"


def test_refuses_a_lane_that_is_not_c(sandbox: Path) -> None:
    args = _args(sandbox, "--i-am-the-sole-subject")
    args[0] = "PhysicalAI-SmartSpaces"
    assert sign.main(args) == 2


def test_iron_run_refuses_unconsented_footage(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data.footage as footage
    from src.data.footage import FOOTAGE_ROOTS
    from src.main import main

    monkeypatch.setattr(footage, "REGISTRY_PATH", sandbox / "configs" / "datasets.yaml")
    tmp_path = sandbox

    video = FOOTAGE_ROOTS["site-zero-solo"] / "B.mp4"
    rc = main(
        [
            "run",
            "--video",
            str(video),
            "--calibration",
            str(tmp_path / "none.yaml"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out").exists()
