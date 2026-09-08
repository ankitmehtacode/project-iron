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
    ConsentPosture,
    ConsentRecord,
    DatasetEntry,
    DatasetRegistry,
    DeploymentScopeRestriction,
    DeploymentScopeUnresolved,
    Hosting,
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


# ---------------------------------------------------------------------------
# Day 35 -- consent_posture: staged performer vs. real, unconsenting subject
# ---------------------------------------------------------------------------


def test_dataset_entry_accepts_every_consent_posture_literal() -> None:
    for posture in get_args(ConsentPosture):
        entry = DatasetEntry(name="x", lane="R", consent_posture=posture)
        assert entry.consent_posture == posture


def test_dataset_entry_consent_posture_defaults_to_none() -> None:
    assert DatasetEntry(name="x", lane="R").consent_posture is None


def test_every_lane_r_seed_entry_has_a_consent_posture_backfilled() -> None:
    """The Day 35 rule made structural: this distinction has been implicit
    for thirty days and every pre-existing lane-R entry gets it, not only
    entries added today."""
    registry = DatasetRegistry.load(SEED_PATH)
    missing = [
        e.name
        for e in registry.entries()
        if e.lane == "R" and e.consent_posture is None
    ]
    assert not missing, f"lane-R entries with no consent_posture: {missing}"


@pytest.mark.parametrize(
    "name",
    ["i-LIDS", "PETS2009", "MEVA", "CAVIAR"],
)
def test_staged_actor_cctv_datasets_are_recorded_as_such(name: str) -> None:
    registry = DatasetRegistry.load(SEED_PATH)
    assert registry.get(name).consent_posture == "staged_actors"


@pytest.mark.parametrize(
    "name",
    ["ChokePoint", "PRW", "CUHK-SYSU"],
)
def test_real_surveillance_cctv_datasets_are_recorded_as_such(name: str) -> None:
    registry = DatasetRegistry.load(SEED_PATH)
    assert registry.get(name).consent_posture == "public_cctv_no_consent"


@pytest.mark.parametrize(
    "name",
    [
        "i-LIDS",
        "PETS2009",
        "ChokePoint",
        "PRW",
        "CUHK-SYSU",
        "CAVIAR",
        "UCSD-Anomaly-Detection",
    ],
)
def test_day_35_cctv_datasets_are_registered_lane_r_with_a_validity_cell(
    name: str,
) -> None:
    registry = DatasetRegistry.load(SEED_PATH)
    dataset = registry.get(name)
    assert dataset.lane == "R"
    assert dataset.license_snapshot is None
    assert dataset.hypothesis_class
    assert dataset.validity_matrix_cell, f"{name} has no validity_matrix_cell note"


def test_meva_verification_priority_is_not_buried_by_todays_additions() -> None:
    """Objective 4's explicit instruction: today's registrations must not
    bury MEVA's standing first-in-queue position."""
    registry = DatasetRegistry.load(SEED_PATH)
    meva = registry.get("MEVA")
    assert "verify first" in meva.notes.lower()


def test_pets2009_and_chokepoint_name_the_identity_ambiguity_eval_target() -> None:
    """Recorded as a forward-looking eval target only, per Objective 4 —
    not implemented today, just not lost either."""
    registry = DatasetRegistry.load(SEED_PATH)
    for name in ("PETS2009", "ChokePoint"):
        notes = registry.get(name).notes.lower()
        assert "forward-looking eval target" in notes
        assert "associationverdict" in notes


# -- CHIRLA (Day 36) ---------------------------------------------------------


def test_chirla_is_registered_lane_r_unverified_with_a_validity_cell() -> None:
    registry = DatasetRegistry.load(SEED_PATH)
    chirla = registry.get("CHIRLA")
    assert chirla.lane == "R"
    assert chirla.license_snapshot is None
    assert chirla.consent_posture == "staged_actors"
    assert chirla.hypothesis_class
    assert "multi_camera" in chirla.validity_matrix_cell
    assert "reappearance" in chirla.validity_matrix_cell


def test_chirla_cites_a_commit_it_was_actually_read_at() -> None:
    """Objective 1's dossier requirement: every fact must cite the exact
    file/commit read, not "the repo" in general — otherwise a future
    re-read of a changed README silently invalidates this entry with
    nothing to detect the mismatch."""
    registry = DatasetRegistry.load(SEED_PATH)
    notes = registry.get("CHIRLA").notes
    assert "README.md" in notes
    assert "commit" in notes.lower()
    # a real, full-length git sha, not a placeholder
    import re

    assert re.search(r"\b[0-9a-f]{40}\b", notes)


def test_chirla_is_exactly_as_inert_as_every_other_lane_r_entry() -> None:
    """STRUCTURAL (Objective 1): registering CHIRLA must not create a
    dataset that can be fetched, trained on, or calibrated with —
    license_snapshot: null refuses require_fetchable, and being lane R
    (not lane C) refuses open_for_training, exactly like every other
    unverified lane-R entry. Nothing about adding a richer notes/
    validity_matrix_cell payload changes that."""
    registry = DatasetRegistry.load(SEED_PATH)

    with pytest.raises(LicenseNotVerified, match="no license snapshot"):
        registry.require_fetchable("CHIRLA")

    # open_for_training checks require_fetchable before the lane check, so
    # today's real (unverified) CHIRLA entry is refused for lack of a
    # license snapshot before its lane is ever consulted — belt AND
    # suspenders, and this confirms the first belt already catches it.
    with pytest.raises(LicenseNotVerified, match="no license snapshot"):
        registry.open_for_training("CHIRLA")

    # Isolate the SECOND belt: even a hypothetically-verified CHIRLA (a
    # snapshot attached, license check cleared) must still be refused by
    # the lane check alone, because it is lane R. Confirms this is not
    # inert only because nobody has verified it yet.
    hypothetically_verified = entry("CHIRLA", "R", verified=True)
    lane_only_registry = registry_with(hypothetically_verified)
    with pytest.raises(LaneViolation, match="never enter a training"):
        lane_only_registry.open_for_training("CHIRLA")


def test_chirla_consent_posture_cannot_satisfy_a_lane_c_only_loader() -> None:
    """STRUCTURAL (Objective 1): consent_posture is not a lane and confers
    no permission by itself — a lane-C-only loader (calibration; see
    scripts/build_calibration_set.py's require_lane_c) must refuse CHIRLA
    on lane alone, the same refusal every lane-R entry gets, regardless of
    what its consent_posture says (Day 38: "staged_actors", upgraded from
    "unknown" — see Day-38's test for that upgrade). This pins that a
    permissive consent_posture is not a backdoor around the lane check."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import build_calibration_set as builder

    registry = DatasetRegistry.load(SEED_PATH)
    with pytest.raises(LaneViolation, match="lane R"):
        builder.require_lane_c(registry, "CHIRLA")


# ---------------------------------------------------------------------------
# Day 38: deployment_scope_restriction — a third, independent legal axis
# ---------------------------------------------------------------------------


def test_chirla_deployment_scope_restriction_is_recorded_and_unresolved() -> None:
    """Objective 1: CHIRLA's HuggingFace "Out-of-Scope Use" clause is
    recorded verbatim, with a source, and unresolved — the whole point of
    today's schema addition is that strong-looking license/consent
    evidence must not quietly resolve this third, independent field."""
    registry = DatasetRegistry.load(SEED_PATH)
    chirla = registry.get("CHIRLA")
    restriction = chirla.deployment_scope_restriction
    assert restriction is not None
    assert restriction.text is not None
    assert "surveillance" in restriction.text
    assert "identification" in restriction.text
    assert "huggingface.co" in restriction.source.lower()
    assert restriction.resolved is False
    assert restriction.resolution is None


def test_chirla_deployment_scope_is_independent_of_license_and_consent() -> None:
    """STRUCTURAL: CHIRLA scores cleanly on license (three sources agree)
    and consent (IRB-approved, unusually strong) yet still carries an
    unresolved deployment_scope_restriction — the three fields must not be
    conflatable into one "is this dataset clean" bit."""
    registry = DatasetRegistry.load(SEED_PATH)
    chirla = registry.get("CHIRLA")
    assert chirla.consent_posture == "staged_actors"
    assert "CC-BY-4.0" in chirla.hypothesis_class
    assert chirla.deployment_scope_restriction is not None
    assert chirla.deployment_scope_restriction.resolved is False


def test_require_product_claim_clearance_refuses_chirla_while_unresolved() -> None:
    """Objective 1's enforcement half: a report/document-generation path
    that would cite CHIRLA's results for a product- or capability-level
    claim must be refused, loudly, while resolved is False — never a
    silent pass-through."""
    registry = DatasetRegistry.load(SEED_PATH)
    with pytest.raises(DeploymentScopeUnresolved, match="surveillance"):
        registry.require_product_claim_clearance("CHIRLA")


def test_require_product_claim_clearance_is_orthogonal_to_eval_use() -> None:
    """Pure internal algorithm benchmarking goes through open_for_eval
    alone and is unaffected by an unresolved deployment_scope_restriction
    — only a call that explicitly asks for product-claim clearance is
    gated. (CHIRLA itself is not reachable here because it also has no
    license_snapshot; a hypothetically-verified entry isolates the
    deployment-scope check from the license check the same way
    test_chirla_is_exactly_as_inert_as_every_other_lane_r_entry does.)"""
    restriction = DeploymentScopeRestriction(
        text="Any deployment aimed at surveillance...",
        source="https://huggingface.co/datasets/bdager/CHIRLA",
        resolved=False,
    )
    hypothetically_verified = DatasetEntry(
        name="CHIRLA",
        lane="R",
        license_snapshot=snapshot(),
        deployment_scope_restriction=restriction,
    )
    registry = registry_with(hypothetically_verified)

    # Pure eval use: unaffected.
    registry.open_for_eval("CHIRLA")

    # A product-claim use: refused.
    with pytest.raises(DeploymentScopeUnresolved):
        registry.require_product_claim_clearance("CHIRLA")


def test_require_product_claim_clearance_passes_once_resolved() -> None:
    """The gate is not permanent — once a human records a resolution, the
    same call site clears. Never auto-flippable from within this module."""
    restriction = DeploymentScopeRestriction(
        text="Any deployment aimed at surveillance...",
        source="https://huggingface.co/datasets/bdager/CHIRLA",
        resolved=True,
        resolution="counsel reviewed 2026-09-08: narrow reading applies",
    )
    resolved_entry = DatasetEntry(
        name="CHIRLA",
        lane="R",
        license_snapshot=snapshot(),
        deployment_scope_restriction=restriction,
    )
    registry = registry_with(resolved_entry)
    cleared = registry.require_product_claim_clearance("CHIRLA")
    assert cleared.name == "CHIRLA"


def test_product_claim_clearance_is_noop_without_a_restriction() -> None:
    """Most of the registry has no deployment_scope_restriction recorded at
    all (see docs/registry_deployment_scope_audit.md) — that must not be
    confused with "resolved"; the gate simply does not apply."""
    registry = DatasetRegistry.load(SEED_PATH)
    meva = registry.require_product_claim_clearance("MEVA")
    assert meva.deployment_scope_restriction is None


# ---------------------------------------------------------------------------
# Day 39: hosting — a fourth axis, independent of license/consent/scope
# ---------------------------------------------------------------------------


def test_dataset_entry_accepts_every_hosting_literal() -> None:
    for hosting in get_args(Hosting):
        entry = DatasetEntry(name="x", lane="R", hosting=hosting)
        assert entry.hosting == hosting


def test_dataset_entry_hosting_defaults_to_unknown() -> None:
    """Unlike consent_posture (default None), hosting has no "not
    applicable" case — every entry is hosted somewhere — so the honest
    default is the explicit "unknown" value, not None."""
    assert DatasetEntry(name="x", lane="R").hosting == "unknown"


def test_chirla_hosting_is_huggingface() -> None:
    """Objective 1: CHIRLA is the entry that exposed why this field needs
    to exist (Day 39) — backfilled explicitly rather than left unknown."""
    registry = DatasetRegistry.load(SEED_PATH)
    assert registry.get("CHIRLA").hosting == "huggingface"


def test_only_chirla_is_backfilled_today() -> None:
    """STRUCTURAL (Objective 1): today backfills exactly one entry.
    Classifying the other 62 is deliberately out of scope — the same
    reaudit deployment_scope_restriction got on Day 38, not attempted
    today. If this count ever grows, it should grow because a real
    reaudit pass updated it, not because a stray edit crept in."""
    registry = DatasetRegistry.load(SEED_PATH)
    classified = [e.name for e in registry.entries() if e.hosting != "unknown"]
    assert classified == ["CHIRLA"]
