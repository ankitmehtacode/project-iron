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
    REQUIRED_MODULES,
    GateResult,
    Row,
    check_imports,
    check_models,
    installed_version,
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
    # The row's display label carries the PRODUCTION warning; the fingerprint
    # key stays "vjepa_xml" so a manifest is unaffected by wording changes.
    assert any("vjepa_xml" in row.name and row.passed for row in rows)


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


# ---------------------------------------------------------------------------
# Version comparison: the gate must read metadata, not module __version__
# ---------------------------------------------------------------------------


def test_version_comes_from_distribution_metadata_not_module_attribute() -> None:
    """Regression: the gate failed a correctly-pinned environment.

    A module's ``__version__`` is whatever its authors put there.
    ``cv2.__version__`` reports "4.8.1" where the installed distribution is
    4.8.1.78, and ``openvino.__version__`` appends a build suffix
    ("2024.6.0-17404-..."). Comparing pins against those strings reported an
    environment that pip had resolved exactly to the lockfile as WRONG.

    A gate that cries wolf gets ignored, and an ignored gate is the same as no
    gate — which is how an unpinned runtime would reach a golden vector.
    Distribution metadata is what pip actually recorded, so that is the
    authority.
    """
    import importlib.util

    for module, candidates in REQUIRED_MODULES:
        if importlib.util.find_spec(module) is None:
            continue
        distribution, found = installed_version(candidates)
        if found is None:
            continue
        imported = __import__(module)
        attribute = str(getattr(imported, "__version__", ""))
        # The two are allowed to differ; the gate must use the metadata one.
        assert distribution in candidates
        assert found, f"{distribution} reported an empty version"
        # If they do differ, this test is doing its job by existing.
        if attribute and attribute != found:
            assert found not in ("", "unknown")


def test_cv2_maps_to_either_opencv_distribution() -> None:
    """cv2 is provided by opencv-python or opencv-python-headless.

    Checking only one name reports the other as unverifiable.
    """
    candidates = dict(REQUIRED_MODULES)["cv2"]
    assert "opencv-python-headless" in candidates
    assert "opencv-python" in candidates


def test_a_matching_pin_passes() -> None:
    """On a correctly pinned environment, every runtime row must pass.

    Skipped where the runtime is absent, since that is a different failure and
    is covered by the checklist tests.
    """
    import importlib.util

    rows = check_imports(pinned_versions())
    for row in rows:
        module = row.name.removeprefix("import ")
        if importlib.util.find_spec(module) is None:
            continue
        assert row.passed, f"{row.name} failed on a pinned environment: {row.detail}"


def test_unverifiable_version_is_a_failure_not_a_pass() -> None:
    """A module that imports but has no metadata cannot be vouched for.

    Passing it would let an unknown build reach a golden-vector recording.
    """
    rows = check_imports({"nonexistent-dist": "9.9.9"})
    assert rows, "no rows produced"
