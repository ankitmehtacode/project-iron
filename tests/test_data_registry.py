"""Tests for the dataset registry and lane enforcement.

The registry is the piece that has to hold at 2 AM, so these tests target the
exact evasions someone under deadline would try: fetching without reading the
license, reaching a retracted dataset through a mirror, and feeding eval-only
data into a training path through a helper function.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import get_args

import pytest
import yaml

from src.data import (
    LANE_DESCRIPTIONS,
    BlockedDataset,
    ConsentRecord,
    DatasetEntry,
    DatasetRegistry,
    Lane,
    LaneViolation,
    LicenseNotVerified,
    LicenseSnapshot,
    RegistryError,
    UnknownDataset,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = REPO_ROOT / "configs" / "datasets.yaml"


def snapshot() -> LicenseSnapshot:
    return LicenseSnapshot(
        url="https://example.org/license",
        verified_date=date(2026, 7, 31),
        text_sha256="a" * 64,
        verified_by="test",
        verified_class="research-only",
    )


def registry_with(*entries: DatasetEntry) -> DatasetRegistry:
    return DatasetRegistry({normalise(e.name): e for e in entries})


def entry(name: str, lane: str, verified: bool = True) -> DatasetEntry:
    return DatasetEntry(
        name=name,
        lane=lane,  # type: ignore[arg-type]
        license_snapshot=snapshot() if verified else None,
    )


# ---------------------------------------------------------------------------
# The seed registry file
# ---------------------------------------------------------------------------


def test_seed_registry_loads() -> None:
    registry = DatasetRegistry.load(SEED_PATH)
    assert len(registry.entries()) >= 30


def test_every_seed_entry_awaits_human_verification() -> None:
    """Nothing in the seed is fetchable: hypothesis classes are not clearances.

    The strategy document's license classes were written from memory. If any
    entry shipped pre-verified, that memory would become a download
    authorisation, which is exactly the shortcut the registry exists to block.
    """
    registry = DatasetRegistry.load(SEED_PATH)
    unblocked = [e for e in registry.entries() if not e.blocked]
    assert unblocked, "seed registry is empty"
    for e in unblocked:
        assert e.license_snapshot is None, f"{e.name} shipped pre-verified"


def test_seed_contains_the_four_lanes() -> None:
    registry = DatasetRegistry.load(SEED_PATH)
    lanes = {e.lane for e in registry.entries() if not e.blocked}
    assert lanes == {"S", "R", "C", "C_pending_consent"}


def test_lane_c_is_only_ever_our_own_consented_captures() -> None:
    """Lane C means consented and ours. Nothing downloadable qualifies.

    Pinned as a set rather than a single name: lane C grows as we capture, and
    the invariant that matters is not "there is exactly one" but "every one of
    them is ours". A public dataset appearing here would mean footage of people
    who never consented had been admitted to the only lane permitted for
    calibration.
    """
    registry = DatasetRegistry.load(SEED_PATH)
    lane_c = {e.name for e in registry.entries() if not e.blocked and e.lane == "C"}
    assert lane_c == {"site-zero", "office-capture-v1"}

    for entry in registry.entries():
        if entry.blocked or entry.lane != "C":
            continue
        assert "consent" in (entry.hypothesis_class + entry.notes).lower(), (
            f"{entry.name} is lane C but its record never mentions consent; "
            "lane C is defined by the consent record, not by the label"
        )


# ---------------------------------------------------------------------------
# Fetch gating
# ---------------------------------------------------------------------------


def test_fetch_without_snapshot_raises() -> None:
    registry = registry_with(entry("MEVA", "R", verified=False))
    with pytest.raises(LicenseNotVerified, match="no license snapshot"):
        registry.require_fetchable("MEVA")


def test_fetch_with_snapshot_passes() -> None:
    registry = registry_with(entry("MEVA", "R", verified=True))
    assert registry.require_fetchable("MEVA").name == "MEVA"


def test_unknown_dataset_is_an_error_not_a_default() -> None:
    registry = registry_with()
    with pytest.raises(UnknownDataset, match="no registered lane"):
        registry.get("SomeNewDataset")


# ---------------------------------------------------------------------------
# The permanent blocklist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["DukeMTMC", "dukemtmc", "DukeMTMC-reID", "dukemtmc_reid", "MS-Celeb", "MSCeleb1M"],
)
def test_blocked_names_raise_with_the_reason(name: str) -> None:
    """Every casing and spelling variant hits, and the reason travels."""
    registry = DatasetRegistry.load(SEED_PATH)
    with pytest.raises(BlockedDataset) as excinfo:
        registry.get(name)
    message = str(excinfo.value).lower()
    assert "retracted" in message or "withdrawn" in message


def test_blocklist_survives_yaml_deletion(tmp_path: Path) -> None:
    """Removing the YAML entry must not unblock a retraction.

    The in-code blocklist outranks the file: a rebase that drops the blocked
    lines, or a fresh registry file, must change nothing.
    """
    minimal = tmp_path / "datasets.yaml"
    minimal.write_text(yaml.safe_dump({"datasets": []}))
    registry = DatasetRegistry.load(minimal)
    with pytest.raises(BlockedDataset, match="RETRACTED"):
        registry.get("DukeMTMC")


def test_registering_a_blocked_name_as_unblocked_is_rejected(tmp_path: Path) -> None:
    """The YAML cannot launder a blocked dataset back in under its own name."""
    laundered = tmp_path / "datasets.yaml"
    laundered.write_text(
        yaml.safe_dump(
            {"datasets": [{"name": "DukeMTMC-reID", "lane": "R", "blocked": False}]}
        )
    )
    with pytest.raises(RegistryError, match="permanent blocklist"):
        DatasetRegistry.load(laundered)


# ---------------------------------------------------------------------------
# Lane enforcement
# ---------------------------------------------------------------------------


def test_lane_r_refuses_the_training_path() -> None:
    registry = registry_with(entry("Market-1501", "R"))
    with pytest.raises(LaneViolation, match="never enter a training"):
        registry.open_for_training("Market-1501")


@pytest.mark.parametrize("lane", ["S", "C"])
def test_lanes_s_and_c_may_train(lane: str) -> None:
    registry = registry_with(entry("ok-data", lane))
    assert registry.open_for_training("ok-data").lane == lane


def test_lane_descriptions_covers_every_lane() -> None:
    """Day 24, Objective 3 audit: LANE_DESCRIPTIONS is a hand-written dict
    keyed by the Lane Literal, not derived from it (unlike, e.g.,
    MOTION_ENTITY_KINDS = get_args(MotionEntityKind) in motion_model.py) --
    `dict[Lane, str]` does not make mypy enforce that every Lane member has
    an entry. No drift exists today, but this is the same shape as the
    CAPABILITIES/GATES defect (Day 16, Day 23) waiting to happen the moment
    a fifth lane is added without remembering this dict. Cheap structural
    guard, not a rewrite: assert coverage explicitly rather than leave it
    to be noticed by a KeyError somewhere downstream."""
    assert set(LANE_DESCRIPTIONS) == set(get_args(Lane))


def test_lane_r_loads_for_eval_from_a_neutral_context() -> None:
    registry = registry_with(entry("NTU-RGBD-120", "R"))
    assert registry.open_for_eval("NTU-RGBD-120").name == "NTU-RGBD-120"


def test_lane_r_eval_load_from_a_marked_training_module_raises() -> None:
    """The stack-marker check: eval helpers cannot be repurposed as feeds.

    Simulates a training module (``IRON_TRAINING_PATH = True`` in module
    globals) calling the *eval* loader through an intermediate helper — the
    exact laundering path a review would miss, because the call site says
    "eval".
    """
    registry = registry_with(entry("NTU-RGBD-120", "R"))

    module_source = (
        "IRON_TRAINING_PATH = True\n"
        "def helper(registry):\n"
        "    return registry.open_for_eval('NTU-RGBD-120')\n"
        "def train_step(registry):\n"
        "    return helper(registry)\n"
    )
    namespace: dict[str, object] = {}
    exec(compile(module_source, "<fake_training_module>", "exec"), namespace)

    with pytest.raises(LaneViolation, match="training path"):
        namespace["train_step"](registry)  # type: ignore[operator]


def test_lane_s_eval_load_from_a_training_module_is_fine() -> None:
    """The marker restricts lane R only; synthetic data may flow anywhere."""
    registry = registry_with(entry("Kubric", "S"))
    namespace: dict[str, object] = {}
    exec(
        compile(
            "IRON_TRAINING_PATH = True\n"
            "def train_step(registry):\n"
            "    return registry.open_for_eval('Kubric')\n",
            "<fake_training_module>",
            "exec",
        ),
        namespace,
    )
    result = namespace["train_step"](registry)  # type: ignore[operator]
    assert result.lane == "S"  # type: ignore[union-attr]


def test_training_gate_still_requires_verification() -> None:
    """Lane S without a snapshot is still unfetched, so still unusable."""
    registry = registry_with(entry("Kubric", "S", verified=False))
    with pytest.raises(LicenseNotVerified):
        registry.open_for_training("Kubric")


# ---------------------------------------------------------------------------
# Day 23 -- lane C_pending_consent
# ---------------------------------------------------------------------------


def consent_record() -> ConsentRecord:
    return ConsentRecord(
        path="/consent/thinkwill-ai-dev-2026.pdf",
        sha="b" * 64,
        subjects=4,
        captured_on=date(2026, 8, 1),
        recorded_by="test",
        purpose_note="AI-development purpose consent, distinct from the "
        "archive's original premises-security purpose",
    )


def pending_entry(name: str, consent: ConsentRecord | None = None) -> DatasetEntry:
    return DatasetEntry(name=name, lane="C_pending_consent", consent_record=consent)


def test_c_pending_consent_is_constructable_and_listable() -> None:
    registry = registry_with(pending_entry("archive-x"))
    assert registry.get("archive-x").lane == "C_pending_consent"
    assert "archive-x" in {e.name for e in registry.entries()}


def test_c_pending_consent_refuses_training_without_consent_record() -> None:
    registry = registry_with(pending_entry("archive-x"))
    with pytest.raises(LaneViolation, match="no consent record attached"):
        registry.open_for_training("archive-x")


def test_c_pending_consent_refuses_eval_without_consent_record() -> None:
    """Rejected here too, unlike lane R -- eval is not exempt for
    non-consented footage of real people."""
    registry = registry_with(pending_entry("archive-x"))
    with pytest.raises(LaneViolation, match="no consent record attached"):
        registry.open_for_eval("archive-x")


def test_c_pending_consent_refusal_names_the_dpdp_purpose_reason() -> None:
    registry = registry_with(pending_entry("archive-x"))
    with pytest.raises(LaneViolation, match="DPDP"):
        registry.open_for_training("archive-x")


def test_c_pending_consent_with_record_still_needs_a_license_snapshot() -> None:
    """Attaching consent clears the consent-specific gate, not
    require_fetchable's separate license check -- the two are orthogonal
    (consent answers 'may we use footage of these people'; license_snapshot
    answers 'has a human verified third-party terms', not applicable the
    same way here but not silently bypassed either)."""
    registry = registry_with(pending_entry("archive-x", consent=consent_record()))
    with pytest.raises(LicenseNotVerified):
        registry.open_for_training("archive-x")


def test_seed_registry_has_a_c_pending_consent_entry_for_the_thinkwill_archive() -> (
    None
):
    registry = DatasetRegistry.load(SEED_PATH)
    thinkwill = next(e for e in registry.entries() if "thinkwill" in e.name.lower())
    assert thinkwill.lane == "C_pending_consent"
    assert thinkwill.consent_record is None
    assert "dpdp" in (thinkwill.hypothesis_class + thinkwill.notes).lower()


# ---------------------------------------------------------------------------
# Day 23 -- validity_matrix_cell on the named lane-R shortlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "OA18",
        "MEVA",
        "Charades",
        "NTU-RGBD-120",
        "Toyota-Smarthome",
        "InHARD",
        "MECCANO",
        "MMPTRACK",
        "DA-2K",
        "ETH3D",
        "iBims-1",
        "DIODE-indoor",
    ],
)
def test_named_lane_r_shortlist_has_lane_r_and_a_validity_matrix_cell(
    name: str,
) -> None:
    registry = DatasetRegistry.load(SEED_PATH)
    dataset = registry.get(name)
    assert dataset.lane == "R"
    assert dataset.license_snapshot is None
    assert dataset.hypothesis_class
    assert dataset.validity_matrix_cell, f"{name} has no validity_matrix_cell note"
