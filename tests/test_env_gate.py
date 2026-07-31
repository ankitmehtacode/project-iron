"""Tests for the environment gate.

The gate decides whether a day's measurements are trustworthy, so a bug that
made it pass wrongly would be worse than having no gate at all: every downstream
verdict would inherit unearned confidence. These tests pin the two ways it could
fail open — accepting a wrong version, and reporting only the first problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from env_gate import (  # noqa: E402
    GateResult,
    Row,
    check_models,
    module_for,
    pinned_versions,
    render_checklist,
    run_gate,
)

from src.config import IronConfig  # noqa: E402


def test_pins_are_parsed_from_requirements() -> None:
    pins = pinned_versions()
    assert pins, "no pins parsed from locking-requirements.txt"
    # torch carries a trailing --index-url on its line; the pin must still parse.
    assert pins.get("torch") == "2.2.0"
    assert pins.get("openvino") == "2024.6.0"


def test_distribution_names_map_to_import_names() -> None:
    """opencv-python-headless imports as cv2; a naive mapping reports it absent."""
    assert module_for("opencv-python-headless") == "cv2"
    assert module_for("PyYAML") == "yaml"
    assert module_for("torch") == "torch"
    assert module_for("pytest-cov") == "pytest_cov"


def test_gate_reports_every_failure_not_just_the_first() -> None:
    """Bringing an environment up must be one read, not a rerun-and-discover loop."""
    result, _ = run_gate(IronConfig.load())
    assert len(result.rows) >= 8, "gate is checking fewer rows than expected"
    # This environment has several distinct problems; all must be listed.
    assert len(result.failures) > 1
    checklist = render_checklist(result)
    for row in result.failures:
        assert row.name in checklist


def test_missing_models_are_recorded_in_the_fingerprint() -> None:
    """Absence is recorded as present=False, never omitted.

    Omitting a missing model would make the fingerprint of a broken environment
    look like one that simply used fewer models.
    """
    _, fingerprint = check_models(IronConfig.load())
    assert set(fingerprint) == {"vjepa_xml", "vjepa_bin", "cotracker_checkpoint"}
    for entry in fingerprint.values():
        assert "present" in entry
        if not entry["present"]:
            assert "sha256" not in entry


def test_present_model_gets_a_sha_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present artifact must be fingerprinted, not merely ticked off."""
    import hashlib

    models = tmp_path / "models"
    (models / "int8").mkdir(parents=True)
    xml = models / "int8" / "vjepa2_vitl_int8.xml"
    xml.write_bytes(b"<net/>")

    monkeypatch.setenv("IRON_PATHS__MODELS_DIR", str(models))
    rows, fingerprint = check_models(IronConfig.load())

    entry = fingerprint["vjepa_xml"]
    assert entry["present"] is True
    assert entry["sha256"] == hashlib.sha256(b"<net/>").hexdigest()
    assert entry["size_bytes"] == 6
    assert any(row.name == "model vjepa_xml" and row.passed for row in rows)


def test_result_passes_only_when_every_row_passes() -> None:
    assert GateResult([Row("a", True, ""), Row("b", True, "")]).passed is True
    assert GateResult([Row("a", True, ""), Row("b", False, "")]).passed is False


def test_checklist_names_a_remedy_for_each_failure() -> None:
    """A checklist without remedies is a complaint, not a checklist."""
    result, _ = run_gate(IronConfig.load())
    if not result.failures:
        pytest.skip("environment is complete; nothing to render")
    for row in result.failures:
        assert row.remedy, f"{row.name} failed without a remedy"
